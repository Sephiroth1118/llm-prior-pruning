#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from evoprune.reward_utils import (
    ModelStructConfig,
    RewardConfig,
    estimate_current_flops_reduction,
    score_pruning_trial,
)
from evoprune.stitch_utils import StitchedModelResult, stitch_model
from evoprune.wanda_utils import (
    CalibrationResult,
    build_dense_logits_cache,
    collect_activation_stats,
    compute_model_metrics,
    compute_model_wanda_scores,
)


@dataclass
class CandidateMetrics:
    kl_divergence_from_dense: Optional[float]
    num_kl_tokens: int
    num_samples: int
    wanda_scores: Dict[str, Any]
    flops_reduction: float


class StitchedModelCandidate:
    """User-facing facade for evaluating and scoring one stitched model candidate."""

    def __init__(self, stitched: StitchedModelResult):
        self.model = stitched.model
        self.policy = stitched.policy
        self.sync_info = stitched.sync_info
        self.struct_cfg = ModelStructConfig.from_model_config(self.model.config)
        self.model_name_or_path = stitched.model_name_or_path
        self.db_path = stitched.db_path
        self.dtype = stitched.dtype
        self.attn_implementation = stitched.attn_implementation
        self.device = stitched.device
        self.sync_zero_masks = stitched.sync_zero_masks

    @staticmethod
    def _model_has_meta_parameters(model) -> bool:
        return any(getattr(param, "is_meta", False) for param in model.parameters())

    def evaluate(
        self,
        input_ids_list: List[torch.Tensor],
        device: str = "cuda",
        max_samples: Optional[int] = None,
        dense_logits_cache: Optional[List[torch.Tensor]] = None,
        compute_wanda: bool = False,
    ) -> CandidateMetrics:
        """Run one calibration pass and return KL, optional Wanda, and FLOPs reduction."""
        wanda_scores: Dict[str, Any] = {}
        if compute_wanda and self._model_has_meta_parameters(self.model):
            collector, forward_metrics = collect_activation_stats(
                model=self.model,
                input_ids_list=input_ids_list,
                device=device,
                max_samples=max_samples,
                dense_logits_cache=dense_logits_cache,
                compute_wanda=True,
            )
            wanda_model = None
            try:
                wanda_model = stitch_model(
                    model_name_or_path=self.model_name_or_path,
                    db_path=self.db_path,
                    policy=self.policy,
                    dtype=self.dtype,
                    low_cpu_mem_usage=False,
                    attn_implementation=None,
                    device="cpu",
                    sync_zero_masks=self.sync_zero_masks,
                ).model
                wanda_scores = compute_model_wanda_scores(wanda_model, collector)
            finally:
                if wanda_model is not None:
                    del wanda_model
            metrics = CalibrationResult(
                wanda_scores=wanda_scores,
                kl_divergence_from_dense=forward_metrics["kl_divergence_from_dense"],
                num_kl_tokens=forward_metrics["num_kl_tokens"],
                num_samples=forward_metrics["num_samples"],
            )
        else:
            metrics = compute_model_metrics(
                model=self.model,
                input_ids_list=input_ids_list,
                device=device,
                max_samples=max_samples,
                dense_logits_cache=dense_logits_cache,
                compute_wanda=compute_wanda,
            )
        flops_reduction = estimate_current_flops_reduction(
            struct_cfg=self.struct_cfg,
            sync_info=self.sync_info,
        )
        return CandidateMetrics(
            kl_divergence_from_dense=metrics.kl_divergence_from_dense,
            num_kl_tokens=metrics.num_kl_tokens,
            num_samples=metrics.num_samples,
            wanda_scores=metrics.wanda_scores,
            flops_reduction=flops_reduction,
        )

    def score(
        self,
        target_flops_reduction: float,
        input_ids_list: Optional[List[torch.Tensor]] = None,
        device: str = "cuda",
        max_samples: Optional[int] = None,
        dense_logits_cache: Optional[List[torch.Tensor]] = None,
        valid: bool = True,
        reward_cfg: Optional[RewardConfig] = None,
        metrics: Optional[CandidateMetrics] = None,
    ):
        """Score this candidate, optionally reusing metrics from a prior evaluate() call."""
        if metrics is None:
            if input_ids_list is None:
                raise ValueError("input_ids_list is required when metrics is not provided")
            metrics = self.evaluate(
                input_ids_list=input_ids_list,
                device=device,
                max_samples=max_samples,
                dense_logits_cache=dense_logits_cache,
            )
        return score_pruning_trial(
            target_flops_reduction=target_flops_reduction,
            kl_divergence_from_dense=metrics.kl_divergence_from_dense,
            struct_cfg=self.struct_cfg,
            sync_info=self.sync_info,
            valid=valid,
            reward_cfg=reward_cfg,
        )


def build_stitched_candidate(
    model_name_or_path: str,
    db_path: str,
    policy: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    default_level: int = 0,
    attn_prune_name: str = "o_proj",
    mlp_prune_name: str = "down_proj",
    dtype: str = "auto",
    low_cpu_mem_usage: bool = False,
    attn_implementation: Optional[str] = None,
    device: str = "cuda",
    sync_zero_masks: bool = True,
) -> StitchedModelCandidate:
    """Build a stitched candidate from a config file or in-memory policy."""
    stitched = stitch_model(
        model_name_or_path=model_name_or_path,
        db_path=db_path,
        config_path=config_path,
        policy=policy,
        default_level=default_level,
        attn_prune_name=attn_prune_name,
        mlp_prune_name=mlp_prune_name,
        dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        attn_implementation=attn_implementation,
        device=device,
        sync_zero_masks=sync_zero_masks,
    )
    return StitchedModelCandidate(stitched)


def build_teacher_logits_cache(
    model_name_or_path: str,
    input_ids_list: List[torch.Tensor],
    dtype: str = "auto",
    low_cpu_mem_usage: bool = False,
    attn_implementation: Optional[str] = None,
    device: str = "cuda",
    max_samples: Optional[int] = None,
) -> List[torch.Tensor]:
    """Build dense-model logits cache for later KL computation against stitched candidates."""
    from evoprune.stitch_utils import load_base_model

    model = load_base_model(
        model_name_or_path=model_name_or_path,
        dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        attn_implementation=attn_implementation,
        device=device,
    )
    try:
        return build_dense_logits_cache(
            model=model,
            input_ids_list=input_ids_list,
            device=device,
            max_samples=max_samples,
        )
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
