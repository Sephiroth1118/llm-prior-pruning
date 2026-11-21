"""GQA-aware attention for compact (physically pruned) Llama models, for the speed test only.

zipprune/ZipLM/compact.py (unchanged; the accuracy benchmarks use it) gathers a copy of the cached keys and values for every
kept query head at every step, which dominates batched decoding. Here each layer's kept query heads are reordered by the
key/value head they share and every group is padded to the layer's largest group with all-zero query heads, whose
o_proj columns are zero, so the layer computes exactly the same function. Then
  decoding (one new token): the query heads of a group are folded into a short query sequence of their key/value head,
                            so attention reads every cached key/value head once, with no copy (plain SDPA, no mask);
  prefill (several tokens): SDPA with enable_gqa=True.
Padding costs extra query-side work only (at most the original 4 heads per group of Llama-3.1-8B), never extra key/value
memory. to_gqa(model) converts every CompactAttention of a compacted model in place.
"""
import torch
from torch import nn
from torch.nn import functional as F

from zipprune.ZipLM.compact import CompactAttention


class GQACompactAttention(nn.Module):
    def __init__(self, source: CompactAttention):
        super().__init__()
        self.head_dim, self.layer_idx = source.head_dim, source.layer_idx
        self.scaling, self.legacy_cache_output = source.scaling, source.legacy_cache_output
        self.attention_dropout = source.attention_dropout
        self.rotary_emb, self.rotary_uses_length = source.rotary_emb, source.rotary_uses_length
        if getattr(source, "sliding_window", None) is not None:
            raise ValueError("sliding-window attention is not supported here")
        self.num_key_value_heads = source.num_key_value_heads
        self.k_proj, self.v_proj = source.k_proj, source.v_proj
        mapping = source.q_to_kv.tolist()
        groups = [[q for q, kv in enumerate(mapping) if kv == j] for j in range(self.num_key_value_heads)]
        self.group = max((len(g) for g in groups), default=0)
        self.num_heads = self.group * self.num_key_value_heads           # padded
        d, hidden = self.head_dim, source.q_proj.in_features
        weight_q, weight_o = source.q_proj.weight, source.o_proj.weight
        q = torch.zeros(self.num_heads * d, hidden, dtype=weight_q.dtype, device=weight_q.device)
        o = torch.zeros(weight_o.shape[0], self.num_heads * d, dtype=weight_o.dtype, device=weight_o.device)
        for j, members in enumerate(groups):
            for slot, h in enumerate(members):
                new = j * self.group + slot
                q[new * d:(new + 1) * d] = weight_q[h * d:(h + 1) * d]
                o[:, new * d:(new + 1) * d] = weight_o[:, h * d:(h + 1) * d]
        if source.q_proj.bias is not None or source.o_proj.bias is not None:
            raise ValueError("biased projections are not supported here")
        self.q_proj = nn.Linear(hidden, self.num_heads * d, bias=False, device=q.device, dtype=q.dtype)
        self.o_proj = nn.Linear(self.num_heads * d, weight_o.shape[0], bias=False, device=o.device, dtype=o.dtype)
        with torch.no_grad():
            self.q_proj.weight.copy_(q)
            self.o_proj.weight.copy_(o)

    def _pack(self, output, cache):
        return (output, None, cache) if self.legacy_cache_output else (output, None)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        batch, length, _ = hidden_states.shape
        if self.num_heads == 0:                           # removed layer: as CompactAttention
            if past_key_value is not None:
                empty = hidden_states.new_empty((batch, 0, length, self.head_dim))
                past_key_value.update(empty, empty, self.layer_idx, {"cache_position": cache_position})
            return self._pack(torch.zeros_like(hidden_states), past_key_value)
        d, g, kv = self.head_dim, self.group, self.num_key_value_heads
        q = self.q_proj(hidden_states).view(batch, length, self.num_heads, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, length, kv, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, length, kv, d).transpose(1, 2)
        if position_embeddings is None:
            if self.rotary_emb is None or position_ids is None:
                raise ValueError("RoPE position embeddings or position_ids are required")
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
        key_length = k.shape[-2]
        mask = attention_mask[..., :key_length] if attention_mask is not None else None
        if length == 1:
            # one new token: the g query heads of a group become g query positions of their key/value head
            qf = q.reshape(batch, kv, g, d)
            if mask is not None:
                mask = mask.expand(batch, 1, 1, key_length).expand(batch, 1, g, key_length)
            out = F.scaled_dot_product_attention(qf, k, v, attn_mask=mask, scale=self.scaling)
            out = out.reshape(batch, self.num_heads, 1, d)
        else:
            if mask is None and key_length != length:
                positions = torch.arange(key_length - length, key_length, device=q.device)
                allowed = torch.arange(key_length, device=q.device)[None, :] <= positions[:, None]
                mask = torch.zeros((length, key_length), device=q.device, dtype=q.dtype)
                mask = mask.masked_fill(~allowed, torch.finfo(q.dtype).min)[None, None]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None, scale=self.scaling,
                                                 enable_gqa=g > 1)
        out = out.transpose(1, 2).reshape(batch, length, self.num_heads * d)
        return self._pack(self.o_proj(out), past_key_value)


def to_gqa(model):
    for layer in model.model.layers:
        if isinstance(layer.self_attn, CompactAttention):
            layer.self_attn = GQACompactAttention(layer.self_attn)
    return model
