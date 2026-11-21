#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, Optional, Any, List
import math


@dataclass
class ModelStructConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_layers: int

    @classmethod
    def from_model_config(cls, config):
        num_attention_heads = int(config.num_attention_heads)
        hidden_size = int(config.hidden_size)
        head_dim = hidden_size // num_attention_heads
        num_key_value_heads = int(getattr(config, "num_key_value_heads", num_attention_heads))
        intermediate_size = int(config.intermediate_size)
        num_layers = int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0)))
        return cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            num_layers=num_layers,
        )


@dataclass
class RewardConfig:
    w_kl: float = 1.0
    w_invalid: float = 5.0
    w_band: float = 8.0
    band_miss_penalty: float = 1.0
    near_target_penalty_free_width: float = 0.02

    target_band_width: float = 0.0
    kl_high_penalty_tau: float = 2.0
    kl_high_penalty_alpha: float = 2.0
    kl_clip: float = 10.0
    kl_unstable_threshold: float = 2.0
    label_good_threshold: float = -0.25
    label_ok_threshold: float = -0.75
    label_bad_threshold: float = -1.5

    kl_alert_threshold: float = 0.1

    reward_mode: str = "legacy"

    def __post_init__(self):
        if self.reward_mode not in ("legacy", "raw_kl"):
            raise ValueError(f"Unknown reward mode: {self.reward_mode}")


@dataclass
class RewardResult:
    score: float
    label: str
    valid: bool

    kl_divergence_from_dense: float
    kl_used: float

    current_flops_reduction: float
    target_flops_reduction: float
    target_band_width: float
    budget_distance: float
    budget_penalty: float
    within_target_band: bool

    invalid_penalty: float

    diagnosis: str
    advice: str

    @property
    def gap_to_target(self) -> float:
        return self.budget_distance

    @property
    def gap_penalty(self) -> float:
        return self.budget_penalty


@dataclass
class _LayerMetrics:
    layer_idx: int
    active_query_heads: int
    active_kv_groups: int
    active_mlp_dims: int


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _compute_budget_distance(
    current_flops_reduction: float,
    target_flops_reduction: float,
    target_band_width: float,
) -> float:
    target = max(target_flops_reduction, 1e-8)
    lower = max(0.0, target_flops_reduction - max(target_band_width, 0.0))
    if current_flops_reduction < lower:
        return (lower - current_flops_reduction) / target
    return 0.0


def _compute_budget_penalty(
    budget_distance: float,
    cfg: RewardConfig,
    current_flops_reduction: Optional[float] = None,
    target_flops_reduction: Optional[float] = None,
) -> float:
    if budget_distance <= 0.0:
        return 0.0
    fixed_penalty = cfg.band_miss_penalty
    if current_flops_reduction is not None and target_flops_reduction is not None:
        near_miss_lower = max(0.0, target_flops_reduction - max(cfg.near_target_penalty_free_width, 0.0))
        if current_flops_reduction >= near_miss_lower:
            fixed_penalty = 0.0
    return fixed_penalty + cfg.w_band * budget_distance


def _penalized_kl(kl_divergence_from_dense: float, cfg: RewardConfig) -> float:
    if kl_divergence_from_dense <= cfg.kl_high_penalty_tau:
        return kl_divergence_from_dense
    return cfg.kl_high_penalty_tau + cfg.kl_high_penalty_alpha * (
        kl_divergence_from_dense - cfg.kl_high_penalty_tau
    )


def make_invalid_reward(
    target_flops_reduction: float,
    current_flops_reduction: float,
    reward_cfg: Optional[RewardConfig],
    error_message: str,
) -> RewardResult:
    cfg = reward_cfg or RewardConfig()
    target = max(target_flops_reduction, 1e-8)
    lower = max(0.0, target_flops_reduction - max(cfg.target_band_width, 0.0))
    if current_flops_reduction < lower:
        budget_distance = (lower - current_flops_reduction) / target
    else:
        budget_distance = 0.0
    budget_penalty = _compute_budget_penalty(
        budget_distance,
        cfg,
        current_flops_reduction=current_flops_reduction,
        target_flops_reduction=target_flops_reduction,
    )
    kl_used = min(_penalized_kl(cfg.kl_clip, cfg), cfg.kl_clip)
    invalid_penalty = cfg.w_invalid
    score = -invalid_penalty - budget_penalty - cfg.w_kl * kl_used
    return RewardResult(
        score=score,
        label="invalid",
        valid=False,
        kl_divergence_from_dense=cfg.kl_clip,
        kl_used=kl_used,
        current_flops_reduction=current_flops_reduction,
        target_flops_reduction=target_flops_reduction,
        target_band_width=cfg.target_band_width,
        budget_distance=budget_distance,
        budget_penalty=budget_penalty,
        within_target_band=budget_distance <= 0.0,
        invalid_penalty=invalid_penalty,
        diagnosis=f"search step failed: {error_message}",
        advice="fix the invalid action or evaluation failure before continuing search",
    )


def _infer_num_layers(
    struct_cfg: ModelStructConfig,
    sync_info: Optional[Dict[str, Any]],
) -> int:
    if struct_cfg.num_layers > 0:
        return struct_cfg.num_layers
    if sync_info is not None:
        return max(len(sync_info.get("attention", [])), len(sync_info.get("mlp", [])))
    return 0


def _build_layer_metrics(
    struct_cfg: ModelStructConfig,
    sync_info: Optional[Dict[str, Any]] = None,
) -> List[_LayerMetrics]:
    num_layers = _infer_num_layers(struct_cfg, sync_info)
    if num_layers <= 0:
        return []

    attn_sync = {
        int(item["layer_idx"]): item for item in (sync_info or {}).get("attention", [])
    }
    mlp_sync = {
        int(item["layer_idx"]): item for item in (sync_info or {}).get("mlp", [])
    }
    num_heads = max(struct_cfg.num_attention_heads, 1)
    metrics = []
    for layer_idx in range(num_layers):
        attn_item = attn_sync.get(layer_idx)
        mlp_item = mlp_sync.get(layer_idx)

        active_query_heads = (
            int(attn_item["active_query_head_count"])
            if attn_item is not None
            else num_heads
        )
        active_kv_groups = (
            int(attn_item["active_kv_group_count"])
            if attn_item is not None
            else active_query_heads
        )
        active_mlp_dims = (
            int(mlp_item["active_mlp_dim_count"])
            if mlp_item is not None
            else max(struct_cfg.intermediate_size, 1)
        )
        metrics.append(
            _LayerMetrics(
                layer_idx=layer_idx,
                active_query_heads=active_query_heads,
                active_kv_groups=active_kv_groups,
                active_mlp_dims=active_mlp_dims,
            )
        )
    return metrics


def estimate_current_flops_reduction(
    struct_cfg: ModelStructConfig,
    sync_info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Estimate current FLOPs reduction using stitched structure statistics.

    Proxy per layer:
      Attention:
        q_proj : hidden_size -> hidden_size, scaled by active query ratio
        o_proj : hidden_size -> hidden_size, scaled by active query ratio
        k_proj : hidden_size -> (num_kv_heads * head_dim), scaled by active kv ratio
        v_proj : hidden_size -> (num_kv_heads * head_dim), scaled by active kv ratio

      MLP:
        gate_proj : hidden_size -> intermediate_size, scaled by active mlp ratio
        up_proj   : hidden_size -> intermediate_size, scaled by active mlp ratio
        down_proj : intermediate_size -> hidden_size, scaled by active mlp ratio
    """
    layer_metrics = _build_layer_metrics(
        struct_cfg=struct_cfg,
        sync_info=sync_info,
    )
    if not layer_metrics:
        return 0.0

    h = float(struct_cfg.hidden_size)
    m = float(struct_cfg.intermediate_size)
    kv_hidden = float(struct_cfg.num_key_value_heads * struct_cfg.head_dim)

    # dense per-layer proxy units
    dense_attn = (h * h) + (h * h) + (h * kv_hidden) + (h * kv_hidden)   # q, o, k, v
    dense_mlp = (h * m) + (h * m) + (m * h)                              # gate, up, down
    dense_per_layer = dense_attn + dense_mlp

    dense_total = dense_per_layer * len(layer_metrics)
    sparse_total = 0.0

    num_heads = max(struct_cfg.num_attention_heads, 1)
    num_kv_heads = max(struct_cfg.num_key_value_heads, 1)
    for item in layer_metrics:
        q_active_ratio = _clip01(item.active_query_heads / num_heads)
        kv_active_ratio = _clip01(item.active_kv_groups / num_kv_heads)
        mlp_active_ratio = _clip01(item.active_mlp_dims / max(struct_cfg.intermediate_size, 1))

        sparse_attn = (
            (h * h) * q_active_ratio +      # q_proj
            (h * h) * q_active_ratio +      # o_proj
            (h * kv_hidden) * kv_active_ratio +  # k_proj
            (h * kv_hidden) * kv_active_ratio    # v_proj
        )

        sparse_mlp = (
            (h * m) * mlp_active_ratio +    # gate_proj
            (h * m) * mlp_active_ratio +    # up_proj
            (m * h) * mlp_active_ratio      # down_proj
        )

        sparse_total += (sparse_attn + sparse_mlp)

    if dense_total <= 0:
        return 0.0

    reduction = 1.0 - (sparse_total / dense_total)
    return _clip01(reduction)


def score_pruning_trial(
    target_flops_reduction: float,
    kl_divergence_from_dense: float,
    struct_cfg: ModelStructConfig,
    sync_info: Optional[Dict[str, Any]] = None,
    valid: bool = True,
    reward_cfg: Optional[RewardConfig] = None,
) -> RewardResult:
    cfg = reward_cfg or RewardConfig()

    target_flops_reduction = _safe_float(target_flops_reduction)
    kl_divergence_from_dense = _safe_float(kl_divergence_from_dense, default=cfg.kl_clip)

    current_flops_reduction = estimate_current_flops_reduction(
        struct_cfg=struct_cfg,
        sync_info=sync_info,
    )
    current_flops_reduction = _safe_float(current_flops_reduction)

    invalid_penalty = cfg.w_invalid if not valid else 0.0

    kl_used = (
        kl_divergence_from_dense
        if cfg.reward_mode == "raw_kl"
        else min(_penalized_kl(kl_divergence_from_dense, cfg), cfg.kl_clip)
    )
    budget_distance = _compute_budget_distance(
        current_flops_reduction=current_flops_reduction,
        target_flops_reduction=target_flops_reduction,
        target_band_width=cfg.target_band_width,
    )
    budget_penalty = _compute_budget_penalty(
        budget_distance,
        cfg,
        current_flops_reduction=current_flops_reduction,
        target_flops_reduction=target_flops_reduction,
    )
    within_target_band = budget_distance <= 0.0

    score = (
        - cfg.w_kl * kl_used
        - budget_penalty
        - invalid_penalty
    )

    # label
    if not valid:
        label = "invalid"
    elif not within_target_band:
        label = "out_of_budget"
    elif kl_divergence_from_dense > cfg.kl_unstable_threshold:
        label = "in_budget_unstable"
    elif score >= cfg.label_good_threshold:
        label = "in_budget_good"
    elif score >= cfg.label_ok_threshold:
        label = "in_budget_ok"
    else:
        label = "in_budget_suboptimal"

    # diagnosis + advice
    reasons = []
    advices = []

    if not within_target_band:
        reasons.append("policy is below the target compression rate")
        advices.append("increase pruning slightly until the target compression rate is met")

    if kl_used > cfg.kl_alert_threshold:
        reasons.append("KL divergence is high")
        advices.append("reallocate pruning away from sensitive layers while staying in budget")

    if not reasons:
        reasons.append("policy is within budget and quality is acceptable")
        advices.append("use small local edits to further reduce KL")

    diagnosis = "; ".join(reasons)
    advice = "; ".join(dict.fromkeys(advices))

    return RewardResult(
        score=score,
        label=label,
        valid=valid,
        kl_divergence_from_dense=kl_divergence_from_dense,
        kl_used=kl_used,
        current_flops_reduction=current_flops_reduction,
        target_flops_reduction=target_flops_reduction,
        target_band_width=cfg.target_band_width,
        budget_distance=budget_distance,
        budget_penalty=budget_penalty,
        within_target_band=within_target_band,
        invalid_penalty=invalid_penalty,
        diagnosis=diagnosis,
        advice=advice,
    )
