#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class MoERewardConfig:
    w_kl: float = 1.0
    w_invalid: float = 5.0
    w_band: float = 8.0
    band_miss_penalty: float = 1.0
    near_target_penalty_free_width: float = 0.02
    target_band_width: float = 0.0
    kl_high_penalty_tau: float = 2.0
    kl_high_penalty_alpha: float = 2.0
    kl_clip: float = 10.0


@dataclass
class MoEReward:
    score: float
    kl_divergence_from_dense: float
    ppl: Optional[float]
    compression_ratio: float
    target_flops_reduction: float
    compression_gap: float
    budget_penalty: float
    invalid_penalty: float
    valid: bool
    error_message: Optional[str] = None


def _penalized_kl(kl_divergence_from_dense: float, cfg: MoERewardConfig) -> float:
    if kl_divergence_from_dense <= cfg.kl_high_penalty_tau:
        return kl_divergence_from_dense
    return cfg.kl_high_penalty_tau + cfg.kl_high_penalty_alpha * (
        kl_divergence_from_dense - cfg.kl_high_penalty_tau
    )


def _compute_budget_distance(
    compression_ratio: float,
    target_flops_reduction: float,
    target_band_width: float,
) -> float:
    target = max(float(target_flops_reduction), 1e-8)
    lower = max(0.0, float(target_flops_reduction) - max(float(target_band_width), 0.0))
    if float(compression_ratio) < lower:
        return (lower - float(compression_ratio)) / target
    return 0.0


def _compute_budget_penalty(
    compression_ratio: float,
    target_flops_reduction: float,
    cfg: MoERewardConfig,
) -> float:
    budget_distance = _compute_budget_distance(
        compression_ratio=compression_ratio,
        target_flops_reduction=target_flops_reduction,
        target_band_width=cfg.target_band_width,
    )
    if budget_distance <= 0.0:
        return 0.0
    fixed_penalty = float(cfg.band_miss_penalty)
    near_miss_lower = max(0.0, float(target_flops_reduction) - max(float(cfg.near_target_penalty_free_width), 0.0))
    if float(compression_ratio) >= near_miss_lower:
        fixed_penalty = 0.0
    return fixed_penalty + float(cfg.w_band) * budget_distance


def score_candidate(
    target_flops_reduction: float,
    compression_ratio: float,
    kl_divergence_from_dense: Optional[float],
    ppl: Optional[float],
    reward_cfg: Optional[MoERewardConfig] = None,
    valid: bool = True,
    error_message: Optional[str] = None,
) -> MoEReward:
    cfg = reward_cfg or MoERewardConfig()
    kl_value = float(kl_divergence_from_dense) if kl_divergence_from_dense is not None else float(cfg.kl_clip)
    ppl_value = float(ppl) if ppl is not None else None
    compression_gap = abs(float(compression_ratio) - float(target_flops_reduction))
    invalid = (not valid) or (ppl_value is not None and (not math.isfinite(ppl_value)))
    kl_used = min(_penalized_kl(kl_value, cfg), float(cfg.kl_clip))
    budget_penalty = _compute_budget_penalty(
        compression_ratio=float(compression_ratio),
        target_flops_reduction=float(target_flops_reduction),
        cfg=cfg,
    )
    invalid_penalty = float(cfg.w_invalid) if invalid else 0.0
    score = -(
        float(cfg.w_kl) * kl_used
        + budget_penalty
        + invalid_penalty
    )
    return MoEReward(
        score=float(score),
        kl_divergence_from_dense=kl_value,
        ppl=ppl_value,
        compression_ratio=float(compression_ratio),
        target_flops_reduction=float(target_flops_reduction),
        compression_gap=float(compression_gap),
        budget_penalty=float(budget_penalty),
        invalid_penalty=float(invalid_penalty),
        valid=not invalid,
        error_message=error_message,
    )
