#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


_LAYER_INDEX_RE = re.compile(r"(?:^|\.)(\d+)(?:\.|$)")


@dataclass
class RouterStats:
    usage_count: List[List[int]]
    router_prob_sum: List[List[float]]
    avg_router_prob: List[List[float]]
    usage_rate: List[List[float]]
    importance: List[List[float]]
    total_routed_tokens_per_layer: List[int]


def _extract_layer_index(module_name: str) -> Optional[int]:
    matches = list(_LAYER_INDEX_RE.finditer(module_name))
    if not matches:
        return None
    for match in matches:
        idx = int(match.group(1))
        if idx < 10000:
            return idx
    return int(matches[0].group(1))


def get_moe_layers(model: nn.Module) -> List[Tuple[int, str, nn.Module]]:
    discovered: List[Tuple[int, str, nn.Module]] = []
    for module_name, module in model.named_modules():
        gate = getattr(module, "gate", None)
        experts = getattr(module, "experts", None)
        if gate is None or experts is None:
            continue
        if not isinstance(gate, nn.Linear):
            continue
        if not isinstance(experts, (nn.ModuleList, list)):
            continue
        layer_idx = _extract_layer_index(module_name)
        if layer_idx is None:
            continue
        discovered.append((layer_idx, module_name, module))
    discovered.sort(key=lambda item: (item[0], item[1]))
    return discovered


def get_num_experts(model: nn.Module) -> int:
    cfg_num_experts = int(getattr(model.config, "num_experts", 0))
    if cfg_num_experts > 0:
        return cfg_num_experts
    layers = get_moe_layers(model)
    if not layers:
        raise ValueError("could not detect MoE layers in the model")
    _, _, first = layers[0]
    return int(len(first.experts))


def _empty_router_stats(num_layers: int, num_experts: int) -> RouterStats:
    zero_int = [[0 for _ in range(num_experts)] for _ in range(num_layers)]
    zero_float = [[0.0 for _ in range(num_experts)] for _ in range(num_layers)]
    return RouterStats(
        usage_count=[row[:] for row in zero_int],
        router_prob_sum=[row[:] for row in zero_float],
        avg_router_prob=[row[:] for row in zero_float],
        usage_rate=[row[:] for row in zero_float],
        importance=[row[:] for row in zero_float],
        total_routed_tokens_per_layer=[0 for _ in range(num_layers)],
    )


@torch.no_grad()
def collect_router_statistics(
    model,
    input_ids_list: List[torch.Tensor],
    device: str = "cuda",
    max_samples: Optional[int] = None,
) -> RouterStats:
    moe_layers = get_moe_layers(model)
    if not moe_layers:
        raise ValueError("no MoE layers were detected; cannot build expert groups")
    num_layers = max(item[0] for item in moe_layers) + 1
    num_experts = get_num_experts(model)
    top_k = int(getattr(model.config, "num_experts_per_tok", 1))

    stats = _empty_router_stats(num_layers=num_layers, num_experts=num_experts)
    data = input_ids_list[:max_samples] if max_samples is not None else input_ids_list

    for sample in data:
        outputs = model(sample.to(device), output_router_logits=True, use_cache=False)
        router_logits = getattr(outputs, "router_logits", None)
        if router_logits is None:
            raise RuntimeError(
                "model did not return router_logits even though output_router_logits=True; "
                "please check the target MoE model implementation"
            )
        if len(router_logits) != num_layers:
            raise ValueError(
                f"expected {num_layers} router logit tensors, got {len(router_logits)}"
            )
        for layer_idx, logits in enumerate(router_logits):
            probs = torch.softmax(logits.float(), dim=-1)
            routing_probs, selected_experts = torch.topk(probs, top_k, dim=-1)
            selected_flat = selected_experts.reshape(-1).cpu()
            probs_flat = routing_probs.reshape(-1).cpu()
            usage_counts = torch.bincount(selected_flat, minlength=num_experts).tolist()
            prob_sums = torch.zeros(num_experts, dtype=torch.float64)
            prob_sums.scatter_add_(0, selected_flat, probs_flat.to(torch.float64))
            prob_sums_list = prob_sums.tolist()
            for expert_idx in range(num_experts):
                stats.usage_count[layer_idx][expert_idx] += int(usage_counts[expert_idx])
                stats.router_prob_sum[layer_idx][expert_idx] += float(prob_sums_list[expert_idx])
            stats.total_routed_tokens_per_layer[layer_idx] += int(selected_flat.numel())

    for layer_idx in range(num_layers):
        total_routed = max(int(stats.total_routed_tokens_per_layer[layer_idx]), 1)
        for expert_idx in range(num_experts):
            usage = int(stats.usage_count[layer_idx][expert_idx])
            prob_sum = float(stats.router_prob_sum[layer_idx][expert_idx])
            avg_prob = prob_sum / max(usage, 1)
            usage_rate = float(usage) / float(total_routed)
            importance = usage_rate * avg_prob
            stats.avg_router_prob[layer_idx][expert_idx] = float(avg_prob)
            stats.usage_rate[layer_idx][expert_idx] = float(usage_rate)
            stats.importance[layer_idx][expert_idx] = float(importance)
    return stats


def build_fixed_groups_from_router_stats(
    stats: RouterStats,
    num_groups: int = 4,
) -> Dict[str, Any]:
    num_layers = len(stats.importance)
    if num_layers == 0:
        raise ValueError("router statistics are empty")
    num_experts = len(stats.importance[0])
    if num_experts % num_groups != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by num_groups={num_groups}"
        )

    groups: Dict[str, Dict[str, List[int]]] = {}
    importance_obj: Dict[str, Dict[str, float]] = {}
    group_size = num_experts // num_groups
    for layer_idx in range(num_layers):
        ranked = sorted(
            range(num_experts),
            key=lambda expert_idx: (-float(stats.importance[layer_idx][expert_idx]), int(expert_idx)),
        )
        layer_groups: Dict[str, List[int]] = {}
        for group_idx in range(num_groups):
            start = group_idx * group_size
            end = start + group_size
            layer_groups[str(group_idx)] = ranked[start:end]
        groups[str(layer_idx)] = layer_groups
        importance_obj[str(layer_idx)] = {
            str(expert_idx): float(stats.importance[layer_idx][expert_idx])
            for expert_idx in range(num_experts)
        }

    payload = {
        "num_layers": int(num_layers),
        "num_experts": int(num_experts),
        "num_groups": int(num_groups),
        "groups": groups,
        "importance": importance_obj,
        "metadata": {
            "grouping_method": "router_usage_x_avg_router_prob",
            "description": "Group 0 has highest-importance experts; group 3 has lowest-importance experts.",
        },
    }
    validate_expert_groups(payload)
    return payload


def validate_expert_groups(groups_payload: Dict[str, Any]) -> None:
    num_layers = int(groups_payload["num_layers"])
    num_experts = int(groups_payload["num_experts"])
    num_groups = int(groups_payload["num_groups"])
    groups = groups_payload["groups"]
    importance = groups_payload.get("importance", {})
    expected_group_size = num_experts // num_groups

    for layer_idx in range(num_layers):
        layer_key = str(layer_idx)
        if layer_key not in groups:
            raise ValueError(f"missing groups for layer {layer_idx}")
        layer_groups = groups[layer_key]
        if len(layer_groups) != num_groups:
            raise ValueError(f"layer {layer_idx} must contain {num_groups} groups")
        seen: List[int] = []
        for group_idx in range(num_groups):
            expert_ids = list(layer_groups[str(group_idx)])
            if len(expert_ids) != expected_group_size:
                raise ValueError(
                    f"layer {layer_idx} group {group_idx} must contain {expected_group_size} experts"
                )
            seen.extend(int(x) for x in expert_ids)
        if sorted(seen) != list(range(num_experts)):
            raise ValueError(
                f"layer {layer_idx} groups must form a partition of expert ids 0..{num_experts-1}"
            )
        if layer_key in importance:
            layer_importance = importance[layer_key]
            means: List[float] = []
            for group_idx in range(num_groups):
                ids = [int(x) for x in layer_groups[str(group_idx)]]
                vals = [float(layer_importance[str(expert_idx)]) for expert_idx in ids]
                means.append(sum(vals) / max(len(vals), 1))
            for prev, curr in zip(means, means[1:]):
                if prev + 1e-12 < curr:
                    raise ValueError(
                        f"layer {layer_idx} group importance must be monotonic descending: {means}"
                    )


def load_expert_groups(path: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expert_groups JSON must be a dict")
    validate_expert_groups(payload)
    return payload


def save_expert_groups(path: str, payload: Dict[str, Any]) -> None:
    validate_expert_groups(payload)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_expert_groups(payload: Dict[str, Any]) -> Dict[str, Any]:
    num_layers = int(payload["num_layers"])
    num_groups = int(payload["num_groups"])
    importance = payload.get("importance", {})
    groups = payload["groups"]
    summary: Dict[str, Any] = {"layers": {}}
    for layer_idx in range(num_layers):
        layer_key = str(layer_idx)
        layer_groups = groups[layer_key]
        layer_importance = importance.get(layer_key, {})
        group_mean_importance = {}
        top_experts = sorted(
            [(int(k), float(v)) for k, v in layer_importance.items()],
            key=lambda item: (-item[1], item[0]),
        )[: min(8, len(layer_importance))]
        for group_idx in range(num_groups):
            ids = [int(x) for x in layer_groups[str(group_idx)]]
            vals = [float(layer_importance[str(expert_idx)]) for expert_idx in ids] if layer_importance else []
            group_mean_importance[str(group_idx)] = 0.0 if not vals else float(sum(vals) / len(vals))
        summary["layers"][layer_key] = {
            "top_experts": top_experts,
            "group_mean_importance": group_mean_importance,
            "group_sizes": {str(group_idx): len(layer_groups[str(group_idx)]) for group_idx in range(num_groups)},
        }
    return summary
