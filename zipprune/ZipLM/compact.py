"""Physical width pruning for LLaMA/Qwen2/Mistral with exact GQA ownership.

Variable Q counts per KV group are represented by an explicit gather. This
does not assume that Q heads surviving in different groups are balanced.
The deployment backend is PyTorch SDPA; FlashAttention kernels are not used.
"""
import json
import inspect
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def _linear_slice(layer, rows=None, cols=None):
    weight = layer.weight
    if rows is not None:
        weight = weight[rows]
    if cols is not None:
        weight = weight[:, cols]
    out = nn.Linear(weight.shape[1], weight.shape[0], bias=layer.bias is not None,
                    device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        out.weight.copy_(weight)
        if out.bias is not None:
            out.bias.copy_(layer.bias if rows is None else layer.bias[rows])
    return out


def _head_rows(heads, head_dim):
    return [h * head_dim + d for h in heads for d in range(head_dim)]


class CompactMLP(nn.Module):
    def __init__(self, source, dimensions):
        super().__init__()
        self.gate_proj = _linear_slice(source.gate_proj, rows=dimensions)
        self.up_proj = _linear_slice(source.up_proj, rows=dimensions)
        self.down_proj = _linear_slice(source.down_proj, cols=dimensions)
        self.act_fn = source.act_fn
        self.intermediate_size = len(dimensions)

    def forward(self, hidden_states):
        if not self.intermediate_size:
            result = torch.zeros_like(hidden_states)
            return result if self.down_proj.bias is None else result + self.down_proj.bias
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class CompactAttention(nn.Module):
    def __init__(self, source, heads, legacy_cache_output=True):
        super().__init__()
        self.head_dim = int(source.head_dim)
        original_q = source.q_proj.out_features // self.head_dim
        original_kv = source.k_proj.out_features // self.head_dim
        if original_q % original_kv:
            raise ValueError("Original Q/KV grouping is not divisible")
        group_size = original_q // original_kv
        kv_heads = sorted({h // group_size for h in heads})
        kv_index = {h: i for i, h in enumerate(kv_heads)}
        self.register_buffer("q_to_kv", torch.tensor([kv_index[h // group_size] for h in heads], dtype=torch.long),
                             persistent=False)
        self.num_heads = len(heads)
        self.num_key_value_heads = len(kv_heads)
        self.q_proj = _linear_slice(source.q_proj, rows=_head_rows(heads, self.head_dim))
        self.k_proj = _linear_slice(source.k_proj, rows=_head_rows(kv_heads, self.head_dim))
        self.v_proj = _linear_slice(source.v_proj, rows=_head_rows(kv_heads, self.head_dim))
        self.o_proj = _linear_slice(source.o_proj, cols=_head_rows(heads, self.head_dim))
        self.layer_idx = source.layer_idx
        self.rotary_emb = getattr(source, "rotary_emb", None)
        self.rotary_uses_length = self.rotary_emb is not None and "seq_len" in inspect.signature(self.rotary_emb.forward).parameters
        self.scaling = getattr(source, "scaling", self.head_dim ** -0.5)
        self.sliding_window = (getattr(source.config, "sliding_window", None)
                               if source.config.model_type == "mistral" else getattr(source, "sliding_window", None))
        self.attention_dropout = float(getattr(source, "attention_dropout", 0.0))
        self.legacy_cache_output = legacy_cache_output

    def _pack(self, output, weights, cache):
        return (output, weights, cache) if self.legacy_cache_output else (output, weights)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        batch, length, _ = hidden_states.shape
        if past_key_value is not None and type(past_key_value).__name__ != "DynamicCache":
            raise ValueError("Compact attention currently supports DynamicCache only")
        if self.num_heads == 0:
            # Even a removed layer advances the dynamic cache's sequence length,
            # including layer 0, from which HF infers positions at decoding time.
            if past_key_value is not None:
                empty = hidden_states.new_empty((batch, 0, length, self.head_dim))
                past_key_value.update(empty, empty, self.layer_idx, {"cache_position": cache_position})
            output = torch.zeros_like(hidden_states)
            if self.o_proj.bias is not None:
                output = output + self.o_proj.bias
            return self._pack(output, None, past_key_value)
        q = self.q_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is None:
            if self.rotary_emb is None or position_ids is None:
                raise ValueError("RoPE position embeddings or position_ids are required")
            if self.rotary_uses_length:
                cos, sin = self.rotary_emb(v, seq_len=int(position_ids.max().item()) + 1)
                cos, sin = cos[position_ids], sin[position_ids]
            else:
                cos, sin = self.rotary_emb(v, position_ids)
        else:
            cos, sin = position_embeddings
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        def rotate(x):
            half = x.shape[-1] // 2
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        q, k = q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx,
                                         {"sin": sin.squeeze(1), "cos": cos.squeeze(1), "cache_position": cache_position})
        # Preserve each original shared-KV relationship, including uneven groups.
        mapping = self.q_to_kv.to(k.device)
        k, v = k.index_select(1, mapping), v.index_select(1, mapping)
        key_length = k.shape[-2]
        mask = attention_mask[..., :key_length] if attention_mask is not None else None
        if mask is not None and (mask.ndim != 4 or mask.shape[1] != 1):
            raise ValueError("Expected a broadcastable [batch, 1, query, key] attention mask")
        # Use a lower-right causal mask for multi-token cached decoding.
        if self.sliding_window is not None or (mask is None and key_length != length and length > 1):
            positions = torch.arange(key_length - length, key_length, device=q.device)
            keys = torch.arange(key_length, device=q.device)
            allowed = keys[None, :] <= positions[:, None]
            if self.sliding_window is not None:
                allowed &= keys[None, :] > positions[:, None] - self.sliding_window
            causal = torch.zeros((length, key_length), device=q.device, dtype=q.dtype)
            causal.masked_fill_(~allowed, torch.finfo(q.dtype).min)
            mask = causal[None, None] if mask is None else mask.masked_fill(~allowed[None, None], torch.finfo(q.dtype).min)
        causal = mask is None and length > 1
        if output_attentions:
            scores = q @ k.transpose(-1, -2) * self.scaling
            if causal:
                allowed = torch.ones(length, key_length, device=q.device, dtype=torch.bool).tril()
                scores.masked_fill_(~allowed, torch.finfo(q.dtype).min)
            if mask is not None:
                scores = scores + mask
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            output = weights @ v
        else:
            output = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                attn_mask=mask, is_causal=causal, scale=self.scaling,
                dropout_p=self.attention_dropout if self.training else 0.0)
            weights = None
        output = output.transpose(1, 2).reshape(batch, length, self.num_heads * self.head_dim)
        return self._pack(self.o_proj(output), weights, past_key_value)


def compact_model(model, manifest=None):
    """Replace masked modules in place; optionally reconstruct a saved layout."""
    import transformers
    from packaging.version import Version

    if model.config.model_type not in {"llama", "qwen2", "mistral"}:
        raise ValueError(f"Unsupported compact model: {model.config.model_type}")
    if getattr(model.config, "pretraining_tp", 1) != 1:
        raise ValueError("Set pretraining_tp=1 for compact deployment")
    legacy = Version(transformers.__version__) < Version("4.48")
    if manifest is not None and manifest["format_version"] != 1:
        raise ValueError("Unsupported compact checkpoint format")
    if manifest is not None and len(manifest["layers"]) != len(model.model.layers):
        raise ValueError("Compact manifest does not cover every model layer")
    layouts = []
    for index, layer in enumerate(model.model.layers):
        attention, mlp = layer.self_attn, layer.mlp
        if isinstance(attention, CompactAttention):
            raise ValueError("Model is already compact")
        if manifest is None:
            head_dim = attention.head_dim
            num_heads = attention.q_proj.out_features // head_dim
            heads = attention.o_proj.weight.detach().reshape(-1, num_heads, head_dim).ne(0).any(dim=(0, 2))
            dims = mlp.down_proj.weight.detach().ne(0).any(dim=0)
            layout = {"query_heads": heads.nonzero().flatten().cpu().tolist(),
                      "mlp_dimensions": dims.nonzero().flatten().cpu().tolist()}
        else:
            layout = manifest["layers"][index]
        layer.self_attn = CompactAttention(attention, layout["query_heads"], legacy)
        layer.mlp = CompactMLP(mlp, layout["mlp_dimensions"])
        layouts.append(layout)
    model.config._attn_implementation = "sdpa"
    model.eval()
    model._ziplm_compact_manifest = {
        "format_version": 1, "backend": "torch_sdpa_explicit_gqa",
        "transformers_version": transformers.__version__, "layers": layouts,
    }
    return model._ziplm_compact_manifest


def save_compact_model(model, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "_ziplm_compact_manifest"):
        raise ValueError("Model has not been physically compacted")
    model.config.save_pretrained(path)
    (path / "compact_manifest.json").write_text(json.dumps(model._ziplm_compact_manifest, indent=2) + "\n")
    torch.save(model.state_dict(), path / "compact_weights.pth")


def load_compact_model(path, device="cpu", dtype=None):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    path = Path(path)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    config._attn_implementation = "sdpa"
    manifest = json.loads((path / "compact_manifest.json").read_text())
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
        compact_model(model, manifest)
    state = torch.load(path / "compact_weights.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    del state
    if dtype is not None:
        # HF initializes RoPE frequencies in fp32 even for bf16 weights. Calling
        # model.to(dtype=bf16) here would round these nonpersistent buffers and
        # change position embeddings after reload (especially LLaMA-3.1 RoPE).
        for parameter in model.parameters():
            parameter.data = parameter.data.to(dtype=dtype)
    return model.to(device=device).eval()
