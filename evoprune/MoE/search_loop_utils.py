#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from evoprune.metrics import compute_perplexity

from .policy import (
    build_policy_json_from_moe_group_levels,
    validate_moe_group_levels_shape,
)
from .reward_utils import MoERewardConfig, score_candidate
from .simplified_candidate_utils import (
    build_stitched_candidate,
    compression_ratio_from_policy,
)


def _policy_from_moe_group_levels(
    moe_group_levels: List[List[int]],
    policy_bounds: Dict[str, Any],
    expert_groups: Dict[str, Any],
) -> Dict[str, Any]:
    return build_policy_json_from_moe_group_levels(policy_bounds, expert_groups, moe_group_levels)


def _avg_levels_from_group_levels(moe_group_levels: List[List[int]]) -> List[float]:
    return [float(sum(row)) / max(len(row), 1) for row in moe_group_levels]


def clamp_moe_group_levels(
    moe_group_levels: List[List[int]],
    policy_bounds: Dict[str, Any],
    num_groups: int,
) -> Tuple[List[List[int]], List[str]]:
    num_layers = int(policy_bounds["num_layers"])
    validated, warnings = validate_moe_group_levels_shape(moe_group_levels, num_layers, num_groups)
    clamped: List[List[int]] = []
    for layer_idx, row in enumerate(validated):
        layer_obj = policy_bounds["layers"][str(layer_idx)]
        mins = {int(bounds["min_level"]) for bounds in layer_obj["experts"].values()}
        maxs = {int(bounds["max_level"]) for bounds in layer_obj["experts"].values()}
        if len(mins) != 1 or len(maxs) != 1:
            raise ValueError(f"layer {layer_idx} has heterogeneous expert bounds")
        min_level = next(iter(mins))
        max_level = next(iter(maxs))
        clamped_row = []
        for group_idx, level in enumerate(row):
            clamped_level = max(min_level, min(max_level, int(level)))
            if clamped_level != int(level):
                warnings.append(
                    f"clamped moe_group_levels[{layer_idx}][{group_idx}] from {level} to {clamped_level}"
                )
            clamped_row.append(clamped_level)
        clamped.append(clamped_row)
    return clamped, warnings


def history_entry(
    moe_group_levels: List[List[int]],
    reward,
    ppl: Optional[float],
    compression_ratio: Optional[float],
    clamped_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "moe_group_levels": [[int(v) for v in row] for row in moe_group_levels],
        "effective_avg_level_per_layer": _avg_levels_from_group_levels(moe_group_levels),
        "kl": float(reward.kl_divergence_from_dense),
        "ppl": None if ppl is None else float(ppl),
        "reward": float(reward.score),
        "flops_reduction": None if compression_ratio is None else float(compression_ratio),
        "clamped": bool(clamped_warnings),
        "clamp_warnings": list(clamped_warnings or []),
    }


def sort_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(history, key=lambda item: float(item["reward"]), reverse=True)


def history_slice_best(history: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
    return sorted(sort_history(history)[: max(int(k), 0)], key=lambda item: float(item["reward"]))


def history_slice_worst(history: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
    ranked = sort_history(history)
    return sorted(ranked[-max(int(k), 0) :], key=lambda item: float(item["reward"]))


def recent_history_entries(history: Optional[List[Dict[str, Any]]], k: int) -> List[Dict[str, Any]]:
    return list((history or [])[-max(int(k), 0) :])


def build_agent_observation(
    target_flops_reduction: float,
    history: List[Dict[str, Any]],
    num_strong_examples: int,
    num_weak_examples: int,
    chronological_history: Optional[List[Dict[str, Any]]] = None,
    agent_history_size: int = 20,
) -> Dict[str, Any]:
    ranked = sort_history(history)
    return {
        "target_flops_reduction": float(target_flops_reduction),
        "group_semantics": {
            "0": "most important expert group; usually prune less",
            "3": "least important expert group; can usually be pruned more",
        },
        "weak_examples": history_slice_worst(ranked, k=num_weak_examples),
        "strong_examples": history_slice_best(ranked, k=num_strong_examples),
        "recent_history": recent_history_entries(chronological_history, k=agent_history_size),
    }


def evaluate_policy(
    model_name_or_path: str,
    db_path: str,
    policy_bounds: Dict[str, Any],
    policy_json: Dict[str, Any],
    eval_data,
    dense_logits_cache,
    target_flops_reduction: float,
    device: str,
    dtype: str,
    low_cpu_mem_usage: bool,
    attn_implementation: Optional[str],
    reward_cfg: Optional[MoERewardConfig],
    ppl_eval_data,
    expert_groups_path: Optional[str],
):
    candidate = build_stitched_candidate(
        model_name_or_path=model_name_or_path,
        db_path=db_path,
        policy_bounds=policy_bounds,
        policy=policy_json,
        dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        attn_implementation=attn_implementation,
        device=device,
        expert_groups_path=expert_groups_path,
    )
    try:
        metrics = candidate.evaluate(
            input_ids_list=eval_data,
            device=device,
            dense_logits_cache=dense_logits_cache,
        )
        ppl = None
        if ppl_eval_data is not None:
            ppl = float(compute_perplexity(candidate.model, ppl_eval_data, batch_size=1))
        reward = candidate.score(
            target_flops_reduction=target_flops_reduction,
            ppl=ppl,
            metrics=metrics,
            reward_cfg=reward_cfg,
            valid=True,
        )
        return metrics, reward, ppl
    finally:
        del candidate
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_array_policy(
    args,
    moe_group_levels: List[List[int]],
    policy_bounds: Dict[str, Any],
    expert_groups: Dict[str, Any],
    eval_data,
    dense_logits_cache,
    reward_cfg: Optional[MoERewardConfig],
    ppl_eval_data,
):
    clamped, clamp_warnings = clamp_moe_group_levels(
        moe_group_levels,
        policy_bounds,
        num_groups=int(expert_groups["num_groups"]),
    )
    policy_json = _policy_from_moe_group_levels(clamped, policy_bounds, expert_groups)
    try:
        metrics, reward, ppl = evaluate_policy(
            model_name_or_path=args.model_name_or_path,
            db_path=args.db_path,
            policy_bounds=policy_bounds,
            policy_json=policy_json,
            eval_data=eval_data,
            dense_logits_cache=dense_logits_cache,
            target_flops_reduction=args.target_flops_reduction,
            device=args.device,
            dtype=args.dtype,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            attn_implementation=args.attn_implementation,
            reward_cfg=reward_cfg,
            ppl_eval_data=ppl_eval_data,
            expert_groups_path=args.expert_groups_path,
        )
        return clamped, policy_json, metrics, reward, ppl, None, clamp_warnings
    except Exception as exc:
        compression_ratio = compression_ratio_from_policy(
            policy={
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj": int(level)
                for layer_idx, row in enumerate(clamped)
                for group_idx, level in enumerate(row)
                for expert_idx in expert_groups["groups"][str(layer_idx)][str(group_idx)]
            },
            policy_bounds=policy_bounds,
        )
        reward = score_candidate(
            target_flops_reduction=float(args.target_flops_reduction),
            compression_ratio=compression_ratio,
            kl_divergence_from_dense=None,
            ppl=None,
            reward_cfg=reward_cfg,
            valid=False,
            error_message=str(exc),
        )
        return clamped, policy_json, None, reward, None, str(exc), clamp_warnings
