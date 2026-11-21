#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn


@dataclass
class CalibrationResult:
    wanda_scores: Dict[str, Any]
    kl_divergence_from_dense: Optional[float]
    num_kl_tokens: int
    num_samples: int

    @property
    def forward_metrics(self) -> Dict[str, Any]:
        return {
            "kl_divergence_from_dense": self.kl_divergence_from_dense,
            "num_kl_tokens": self.num_kl_tokens,
            "num_samples": self.num_samples,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wanda_scores": self.wanda_scores,
            "forward_metrics": self.forward_metrics,
        }


def _flatten_hidden_states(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x
    return x.reshape(-1, x.shape[-1])


def _module_name(layer_idx: int, suffix: str) -> str:
    return f"model.layers.{layer_idx}.{suffix}"


class InputNormCollector:
    def __init__(self):
        self.sumsq: Dict[str, torch.Tensor] = {}
        self.num_tokens: Dict[str, int] = defaultdict(int)
        self.handles = []

    def add_module(self, module_name: str, module: nn.Module):
        def _hook(_mod, inputs, _output):
            if not inputs:
                return
            x = inputs[0]
            if x is None:
                return

            x = _flatten_hidden_states(x.detach()).float()
            cur = (x * x).sum(dim=0).cpu()

            if module_name not in self.sumsq:
                self.sumsq[module_name] = cur
            else:
                self.sumsq[module_name] += cur

            self.num_tokens[module_name] += x.shape[0]

        self.handles.append(module.register_forward_hook(_hook))

    def get_input_l2(self, module_name: str) -> torch.Tensor:
        if module_name not in self.sumsq:
            raise KeyError(f"No activation stats collected for {module_name}")
        return torch.sqrt(self.sumsq[module_name] + 1e-12)

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class CalibrationHook(InputNormCollector):
    def __init__(
        self,
        dense_logits_cache: Optional[List[torch.Tensor]] = None,
        collect_wanda: bool = True,
    ):
        super().__init__()
        self.dense_logits_cache = dense_logits_cache
        self.collect_wanda = collect_wanda
        self.total_kl = 0.0
        self.total_kl_tokens = 0

    @torch.no_grad()
    def run(
        self,
        model,
        input_ids_list: List[torch.Tensor],
        device: str = "cuda",
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        samples = input_ids_list if max_samples is None else input_ids_list[:max_samples]
        model.eval()
        input_device = _resolve_model_input_device(model, device)
        if self.collect_wanda:
            register_wanda_target_modules(model, self)

        if self.dense_logits_cache is not None and len(self.dense_logits_cache) != len(samples):
            raise ValueError(
                f"dense_logits_cache length ({len(self.dense_logits_cache)}) "
                f"does not match number of samples ({len(samples)})"
            )

        try:
            for idx, input_ids in enumerate(samples):
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                input_ids = input_ids.to(input_device)

                outputs = model(input_ids=input_ids)

                if self.dense_logits_cache is None:
                    continue

                teacher_logits = self.dense_logits_cache[idx].to(outputs.logits.device)
                cur_kl = token_level_kl_from_logits(outputs.logits, teacher_logits)
                self.total_kl += cur_kl.item()
                self.total_kl_tokens += input_ids.shape[1] - 1
        finally:
            self.close()

        result = {
            "kl_divergence_from_dense": None,
            "num_kl_tokens": self.total_kl_tokens,
            "num_samples": len(samples),
        }
        if self.dense_logits_cache is not None:
            result["kl_divergence_from_dense"] = self.total_kl / max(self.total_kl_tokens, 1)
        return result


def weighted_row_scores(weight: torch.Tensor, input_l2: torch.Tensor) -> torch.Tensor:
    w = weight.detach().float().cpu().abs()
    x = input_l2.detach().float().cpu()
    return (w * x.unsqueeze(0)).sum(dim=1)


def weighted_col_scores(weight: torch.Tensor, input_l2: torch.Tensor) -> torch.Tensor:
    w = weight.detach().float().cpu().abs()
    x = input_l2.detach().float().cpu()
    return x * w.sum(dim=0)


def summarize_tensor(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float().cpu()
    if x.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q10": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "q90": 0.0,
            "sum": 0.0,
            "count": 0,
        }

    return {
        "mean": x.mean().item(),
        "std": x.std(unbiased=False).item(),
        "min": x.min().item(),
        "max": x.max().item(),
        "q10": torch.quantile(x, 0.10).item(),
        "q25": torch.quantile(x, 0.25).item(),
        "q50": torch.quantile(x, 0.50).item(),
        "q75": torch.quantile(x, 0.75).item(),
        "q90": torch.quantile(x, 0.90).item(),
        "sum": x.sum().item(),
        "count": x.numel(),
    }


def get_llama_layers(model):
    return model.model.layers


def register_wanda_target_modules(model, collector: InputNormCollector):
    layers = get_llama_layers(model)
    for layer_idx, layer in enumerate(layers):
        collector.add_module(_module_name(layer_idx, "self_attn.q_proj"), layer.self_attn.q_proj)
        collector.add_module(_module_name(layer_idx, "self_attn.k_proj"), layer.self_attn.k_proj)
        collector.add_module(_module_name(layer_idx, "self_attn.v_proj"), layer.self_attn.v_proj)
        collector.add_module(_module_name(layer_idx, "self_attn.o_proj"), layer.self_attn.o_proj)
        collector.add_module(_module_name(layer_idx, "mlp.gate_proj"), layer.mlp.gate_proj)
        collector.add_module(_module_name(layer_idx, "mlp.up_proj"), layer.mlp.up_proj)
        collector.add_module(_module_name(layer_idx, "mlp.down_proj"), layer.mlp.down_proj)


def token_level_kl_from_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute KL(teacher || student) summed over next-token positions.

    student_logits: [B, T, V]
    teacher_logits: [B, T, V]
    """
    student_logits = student_logits[:, :-1, :].float()
    teacher_logits = teacher_logits[:, :-1, :].float()

    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    teacher_probs = torch.exp(teacher_log_probs)

    kl = teacher_probs * (teacher_log_probs - student_log_probs)
    kl = kl.sum(dim=-1)  # [B, T-1]
    return kl.sum()


def _resolve_model_input_device(model, requested_device: str = "cuda") -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None and hasattr(input_embeddings, "weight"):
        return input_embeddings.weight.device
    return next(model.parameters()).device


@torch.no_grad()
def build_dense_logits_cache(
    model,
    input_ids_list: List[torch.Tensor],
    device: str = "cuda",
    max_samples: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Run the dense model once and cache logits on CPU.
    """
    samples = input_ids_list if max_samples is None else input_ids_list[:max_samples]
    model.eval()
    input_device = _resolve_model_input_device(model, device)

    dense_cache = []
    for input_ids in samples:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(input_device)
        outputs = model(input_ids=input_ids)
        dense_cache.append(outputs.logits.detach().cpu())

    return dense_cache


def collect_activation_stats(
    model,
    input_ids_list: List[torch.Tensor],
    device: str = "cuda",
    max_samples: Optional[int] = None,
    dense_logits_cache: Optional[List[torch.Tensor]] = None,
    compute_wanda: bool = True,
):
    """
    Return:
        collector, forward_metrics
    """
    collector = CalibrationHook(
        dense_logits_cache=dense_logits_cache,
        collect_wanda=compute_wanda,
    )
    forward_metrics = collector.run(
        model=model,
        input_ids_list=input_ids_list,
        device=device,
        max_samples=max_samples,
    )
    return collector, forward_metrics


def compute_attention_scores_for_layer(layer, layer_idx: int, collector: InputNormCollector, config) -> Dict:
    attn = layer.self_attn

    q_in = collector.get_input_l2(_module_name(layer_idx, "self_attn.q_proj"))
    k_in = collector.get_input_l2(_module_name(layer_idx, "self_attn.k_proj"))
    v_in = collector.get_input_l2(_module_name(layer_idx, "self_attn.v_proj"))
    o_in = collector.get_input_l2(_module_name(layer_idx, "self_attn.o_proj"))

    q_row = weighted_row_scores(attn.q_proj.weight, q_in)
    k_row = weighted_row_scores(attn.k_proj.weight, k_in)
    v_row = weighted_row_scores(attn.v_proj.weight, v_in)
    o_col = weighted_col_scores(attn.o_proj.weight, o_in)

    configured_num_heads = int(config.num_attention_heads)
    num_kv_heads = int(getattr(config, "num_key_value_heads", configured_num_heads))
    head_dim = int(getattr(config, "head_dim", 0) or 0)
    if head_dim <= 0:
        hidden_size = int(config.hidden_size)
        if hidden_size % configured_num_heads != 0:
            raise ValueError(
                f"Cannot infer head_dim at layer {layer_idx}: "
                f"hidden_size={hidden_size}, num_attention_heads={configured_num_heads}"
            )
        head_dim = hidden_size // configured_num_heads

    if q_row.numel() % head_dim != 0:
        raise ValueError(
            f"Unexpected q_proj out_features at layer {layer_idx}: "
            f"{q_row.numel()} is not divisible by head_dim={head_dim}"
        )
    if o_col.numel() % head_dim != 0:
        raise ValueError(
            f"Unexpected o_proj in_features at layer {layer_idx}: "
            f"{o_col.numel()} is not divisible by head_dim={head_dim}"
        )
    if k_row.numel() != num_kv_heads * head_dim:
        raise ValueError(
            f"Unexpected k_proj out_features at layer {layer_idx}: "
            f"{k_row.numel()} vs expected {num_kv_heads * head_dim}"
        )
    if v_row.numel() != num_kv_heads * head_dim:
        raise ValueError(
            f"Unexpected v_proj out_features at layer {layer_idx}: "
            f"{v_row.numel()} vs expected {num_kv_heads * head_dim}"
        )

    num_heads = q_row.numel() // head_dim
    o_num_heads = o_col.numel() // head_dim
    if o_num_heads != num_heads:
        raise ValueError(
            f"Mismatch between q/o projected head counts at layer {layer_idx}: "
            f"q_proj={num_heads}, o_proj={o_num_heads}"
        )
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_attention_heads ({num_heads}) is not divisible by "
            f"num_key_value_heads ({num_kv_heads})"
        )

    group_size = num_heads // num_kv_heads

    q_head = q_row.view(num_heads, head_dim).sum(dim=1)
    o_head = o_col.view(num_heads, head_dim).sum(dim=1)
    k_group = k_row.view(num_kv_heads, head_dim).sum(dim=1)
    v_group = v_row.view(num_kv_heads, head_dim).sum(dim=1)

    active_head_mask = (q_head > 1e-8) | (o_head > 1e-8)

    kv_contrib = torch.zeros(num_heads, dtype=torch.float32)
    for h in range(num_heads):
        g = h // group_size
        if active_head_mask[h]:
            kv_contrib[h] = (k_group[g] + v_group[g]) / float(group_size)

    per_head = q_head + o_head + kv_contrib
    per_head = per_head * active_head_mask.float()

    active_scores = per_head[active_head_mask]

    return {
        "layer_idx": layer_idx,
        "num_attention_heads": int(num_heads),
        "num_key_value_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "active_head_count": int(active_head_mask.sum().item()),
        "pruned_head_count": int((~active_head_mask).sum().item()),
        "per_head_scores": per_head.tolist(),
        "summary": summarize_tensor(active_scores),
    }


def compute_mlp_scores_for_layer(layer, layer_idx: int, collector: InputNormCollector) -> Dict:
    mlp = layer.mlp

    gate_in = collector.get_input_l2(_module_name(layer_idx, "mlp.gate_proj"))
    up_in = collector.get_input_l2(_module_name(layer_idx, "mlp.up_proj"))
    down_in = collector.get_input_l2(_module_name(layer_idx, "mlp.down_proj"))

    gate_row = weighted_row_scores(mlp.gate_proj.weight, gate_in)
    up_row = weighted_row_scores(mlp.up_proj.weight, up_in)
    down_col = weighted_col_scores(mlp.down_proj.weight, down_in)

    if not (gate_row.numel() == up_row.numel() == down_col.numel()):
        raise ValueError(
            f"Intermediate size mismatch at layer {layer_idx}: "
            f"gate={gate_row.numel()}, up={up_row.numel()}, down={down_col.numel()}"
        )

    per_dim = gate_row + up_row + down_col
    active_dim_mask = per_dim > 1e-8
    active_scores = per_dim[active_dim_mask]

    return {
        "layer_idx": layer_idx,
        "intermediate_size": int(per_dim.numel()),
        "active_dim_count": int(active_dim_mask.sum().item()),
        "pruned_dim_count": int((~active_dim_mask).sum().item()),
        "summary": summarize_tensor(active_scores),
    }


def _add_cross_layer_standardization(block_list: List[Dict], summary_key: str = "summary", mean_key: str = "mean"):
    means = torch.tensor([float(item[summary_key][mean_key]) for item in block_list], dtype=torch.float32)

    mu = means.mean()
    sigma = means.std(unbiased=False)
    if sigma.item() < 1e-12:
        z = torch.zeros_like(means)
    else:
        z = (means - mu) / sigma

    min_v = means.min()
    max_v = means.max()
    if (max_v - min_v).item() < 1e-12:
        mm = torch.zeros_like(means)
    else:
        mm = (means - min_v) / (max_v - min_v)

    for i, item in enumerate(block_list):
        item["z_score"] = float(z[i].item())
        item["minmax_score"] = float(mm[i].item())


def compute_model_wanda_scores(model, collector: InputNormCollector) -> Dict:
    layers = get_llama_layers(model)

    out = {
        "model_type": model.config.model_type,
        "num_layers": len(layers),
        "attention": [],
        "mlp": [],
    }

    for layer_idx, layer in enumerate(layers):
        out["attention"].append(
            compute_attention_scores_for_layer(
                layer=layer,
                layer_idx=layer_idx,
                collector=collector,
                config=model.config,
            )
        )
        out["mlp"].append(
            compute_mlp_scores_for_layer(
                layer=layer,
                layer_idx=layer_idx,
                collector=collector,
            )
        )

    _add_cross_layer_standardization(out["attention"])
    _add_cross_layer_standardization(out["mlp"])

    return out


def compute_model_metrics(
    model,
    input_ids_list: List[torch.Tensor],
    device: str = "cuda",
    max_samples: Optional[int] = None,
    dense_logits_cache: Optional[List[torch.Tensor]] = None,
    compute_wanda: bool = True,
) -> CalibrationResult:
    collector, forward_metrics = collect_activation_stats(
        model=model,
        input_ids_list=input_ids_list,
        device=device,
        max_samples=max_samples,
        dense_logits_cache=dense_logits_cache,
        compute_wanda=compute_wanda,
    )
    return CalibrationResult(
        wanda_scores=compute_model_wanda_scores(model, collector) if compute_wanda else {},
        kl_divergence_from_dense=forward_metrics["kl_divergence_from_dense"],
        num_kl_tokens=forward_metrics["num_kl_tokens"],
        num_samples=forward_metrics["num_samples"],
    )
