#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from evoprune.wanda_utils import compute_model_metrics

from .reward_utils import MoEReward, MoERewardConfig, score_candidate
from .stitch_utils import StitchedMoEModelResult, load_base_model, stitch_model


@dataclass
class MoECandidateMetrics:
    kl_divergence_from_dense: Optional[float]
    num_kl_tokens: int
    num_samples: int
    compression_ratio: float


def compression_ratio_from_policy(policy: Dict[str, int], policy_bounds: Dict[str, Any]) -> float:
    cols_removed_mlp = [int(x) for x in policy_bounds.get("cols_removed_mlp", [])]
    numerator = 0.0
    denominator = 0.0
    for layer_key, layer_obj in policy_bounds["layers"].items():
        for expert_key, expert_bounds in layer_obj["experts"].items():
            module_name = f"model.layers.{int(layer_key)}.mlp.experts.{int(expert_key)}.down_proj"
            min_level = int(expert_bounds["min_level"])
            max_level = int(expert_bounds["max_level"])
            level = int(policy[module_name])
            if cols_removed_mlp:
                min_removed = float(cols_removed_mlp[min_level])
                max_removed = float(cols_removed_mlp[max_level])
                removed = float(cols_removed_mlp[level])
                numerator += max(0.0, removed - min_removed)
                denominator += max(max_removed - min_removed, 1.0)
            else:
                numerator += float(level - min_level)
                denominator += float(max(max_level - min_level, 1))
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


class StitchedMoEModelCandidate:
    def __init__(self, stitched: StitchedMoEModelResult, policy_bounds: Dict[str, Any]):
        self.model = stitched.model
        self.policy = stitched.policy
        self.model_name_or_path = stitched.model_name_or_path
        self.db_path = stitched.db_path
        self.dtype = stitched.dtype
        self.attn_implementation = stitched.attn_implementation
        self.device = stitched.device
        self.policy_bounds = policy_bounds

    def evaluate(
        self,
        input_ids_list: List[torch.Tensor],
        device: str = "cuda",
        dense_logits_cache: Optional[List[torch.Tensor]] = None,
    ) -> MoECandidateMetrics:
        metrics = compute_model_metrics(
            model=self.model,
            input_ids_list=input_ids_list,
            device=device,
            max_samples=None,
            dense_logits_cache=dense_logits_cache,
            compute_wanda=False,
        )
        return MoECandidateMetrics(
            kl_divergence_from_dense=metrics.kl_divergence_from_dense,
            num_kl_tokens=metrics.num_kl_tokens,
            num_samples=metrics.num_samples,
            compression_ratio=compression_ratio_from_policy(self.policy, self.policy_bounds),
        )

    def score(
        self,
        target_flops_reduction: float,
        ppl: Optional[float],
        metrics: MoECandidateMetrics,
        reward_cfg: Optional[MoERewardConfig] = None,
        valid: bool = True,
        error_message: Optional[str] = None,
    ) -> MoEReward:
        return score_candidate(
            target_flops_reduction=float(target_flops_reduction),
            compression_ratio=metrics.compression_ratio,
            kl_divergence_from_dense=metrics.kl_divergence_from_dense,
            ppl=ppl,
            reward_cfg=reward_cfg,
            valid=valid,
            error_message=error_message,
        )


def build_stitched_candidate(
    model_name_or_path: str,
    db_path: str,
    policy_bounds: Dict[str, Any],
    policy: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    default_level: int = 0,
    dtype: str = "auto",
    low_cpu_mem_usage: bool = False,
    attn_implementation: Optional[str] = None,
    device: str = "cuda",
    expert_groups_path: Optional[str] = None,
) -> StitchedMoEModelCandidate:
    stitched = stitch_model(
        model_name_or_path=model_name_or_path,
        db_path=db_path,
        config_path=config_path,
        policy=policy,
        default_level=default_level,
        dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        attn_implementation=attn_implementation,
        device=device,
        sync_zero_masks=True,
        expert_groups_path=expert_groups_path,
    )
    return StitchedMoEModelCandidate(stitched, policy_bounds=policy_bounds)


def build_teacher_logits_cache(
    model_name_or_path: str,
    input_ids_list: List[torch.Tensor],
    dtype: str = "auto",
    low_cpu_mem_usage: bool = False,
    attn_implementation: Optional[str] = None,
    device: str = "cuda",
) -> List[torch.Tensor]:
    model = load_base_model(
        model_name_or_path=model_name_or_path,
        dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        attn_implementation=attn_implementation,
        device=device,
    )
    try:
        logits: List[torch.Tensor] = []
        with torch.no_grad():
            for sample in input_ids_list:
                outputs = model(sample.to(device))
                logits.append(outputs.logits.detach().float().cpu())
        return logits
    finally:
        del model
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
