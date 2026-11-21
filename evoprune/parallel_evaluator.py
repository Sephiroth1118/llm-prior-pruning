#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from evoprune.metrics import compute_perplexity
from evoprune.reward_utils import (
    ModelStructConfig,
    RewardConfig,
    estimate_current_flops_reduction,
    score_pruning_trial,
)
from evoprune.stitch_utils import (
    apply_policy_to_model,
    capture_sync_restore_state,
    load_base_model,
    normalize_policy,
    restore_sync_modified_weights,
    synchronize_zero_masks_without_shrink,
)
from evoprune.wanda_utils import compute_model_metrics

_WORKER_DEVICE: Optional[str] = None
_WORKER_DENSE_LOGITS_CACHE = None
_WORKER_SHARED_CFG: Optional[Dict[str, Any]] = None
_WORKER_MODEL = None
_WORKER_STRUCT_CFG: Optional[ModelStructConfig] = None
_WORKER_RESTORE_STATE: Optional[Dict[str, torch.Tensor]] = None


def _worker_init(
    device: str,
    teacher_cache_path: str,
    shared_cfg: Dict[str, Any],
):
    global _WORKER_DEVICE, _WORKER_DENSE_LOGITS_CACHE, _WORKER_SHARED_CFG
    global _WORKER_MODEL, _WORKER_STRUCT_CFG, _WORKER_RESTORE_STATE
    _WORKER_DEVICE = device
    _WORKER_SHARED_CFG = shared_cfg
    _WORKER_DENSE_LOGITS_CACHE = torch.load(teacher_cache_path, map_location="cpu")
    if _WORKER_DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(_WORKER_DEVICE)
    _WORKER_MODEL = load_base_model(
        model_name_or_path=_WORKER_SHARED_CFG["model_name_or_path"],
        dtype=_WORKER_SHARED_CFG["dtype"],
        low_cpu_mem_usage=_WORKER_SHARED_CFG["low_cpu_mem_usage"],
        attn_implementation=_WORKER_SHARED_CFG["attn_implementation"],
        device=_WORKER_DEVICE,
    )
    _WORKER_STRUCT_CFG = ModelStructConfig.from_model_config(_WORKER_MODEL.config)
    _WORKER_RESTORE_STATE = capture_sync_restore_state(_WORKER_MODEL)


def _apply_worker_policy(policy_json):
    """Every evaluation starts from the same restored model state."""
    assert _WORKER_MODEL is not None and _WORKER_RESTORE_STATE is not None
    restore_sync_modified_weights(_WORKER_MODEL, _WORKER_RESTORE_STATE)
    apply_policy_to_model(_WORKER_MODEL, db_path=_WORKER_SHARED_CFG["db_path"],
                          policy=normalize_policy(policy_json))
    sync_info = None
    if _WORKER_SHARED_CFG["sync_zero_masks"]:
        _, sync_info = synchronize_zero_masks_without_shrink(_WORKER_MODEL)
    return sync_info


def _evaluate_policy_worker(
    policy_json: Dict[str, Any],
    target_flops_reduction: float,
    compute_ppl: bool,
    max_samples: Optional[int],
):
    assert _WORKER_DEVICE is not None
    assert _WORKER_SHARED_CFG is not None
    assert _WORKER_MODEL is not None
    assert _WORKER_STRUCT_CFG is not None
    assert _WORKER_RESTORE_STATE is not None
    try:
        sync_info = _apply_worker_policy(policy_json)
        eval_data = _WORKER_SHARED_CFG["eval_data"]
        dense_logits_cache = _WORKER_DENSE_LOGITS_CACHE
        ppl_eval_data = _WORKER_SHARED_CFG["ppl_eval_data"]
        if max_samples is not None:
            eval_data = eval_data[:max_samples]
            dense_logits_cache = dense_logits_cache[:max_samples]
            if ppl_eval_data is not None:
                ppl_eval_data = ppl_eval_data[:max_samples]
        metrics = compute_model_metrics(
            model=_WORKER_MODEL,
            input_ids_list=eval_data,
            device=_WORKER_DEVICE,
            max_samples=None,
            dense_logits_cache=dense_logits_cache,
            compute_wanda=False,
        )
        flops_reduction = estimate_current_flops_reduction(
            struct_cfg=_WORKER_STRUCT_CFG,
            sync_info=sync_info,
        )
        reward = score_pruning_trial(
            target_flops_reduction=target_flops_reduction,
            kl_divergence_from_dense=metrics.kl_divergence_from_dense,
            struct_cfg=_WORKER_STRUCT_CFG,
            sync_info=sync_info,
            valid=True,
            reward_cfg=RewardConfig(
                target_band_width=float(_WORKER_SHARED_CFG["target_band_width"]),
                reward_mode=_WORKER_SHARED_CFG.get("reward_mode", "legacy"),
            ),
        )
        ppl = None
        if compute_ppl and ppl_eval_data is not None:
            ppl = float(compute_perplexity(_WORKER_MODEL, ppl_eval_data, batch_size=1))
        return {
            "metrics": metrics,
            "reward": reward,
            "ppl": ppl,
            "device": _WORKER_DEVICE,
            "flops_reduction": flops_reduction,
            "error_type": None,
            "error_message": None,
        }
    except Exception as exc:
        return {
            "metrics": None,
            "reward": None,
            "ppl": None,
            "device": _WORKER_DEVICE,
            "error_type": "evaluation_error",
            "error_message": str(exc),
        }


class ParallelEvaluator:
    def __init__(
        self,
        devices: List[str],
        teacher_cache_path: str,
        model_name_or_path: str,
        db_path: str,
        eval_data,
        ppl_eval_data,
        dtype: str,
        low_cpu_mem_usage: bool,
        attn_implementation: Optional[str],
        sync_zero_masks: bool,
        target_band_width: float,
        reward_mode: str = "legacy",
    ):
        RewardConfig(reward_mode=reward_mode)  # Validate before creating workers.
        self.devices = devices
        self.teacher_cache_path = str(Path(teacher_cache_path))
        self._executors: List[ProcessPoolExecutor] = []
        shared_cfg = {
            "model_name_or_path": model_name_or_path,
            "db_path": db_path,
            "eval_data": eval_data,
            "ppl_eval_data": ppl_eval_data,
            "dtype": dtype,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "attn_implementation": attn_implementation,
            "sync_zero_masks": sync_zero_masks,
            "target_band_width": target_band_width,
            "reward_mode": reward_mode,
        }
        ctx = get_context("spawn")
        for device in devices:
            self._executors.append(
                ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=ctx,
                    initializer=_worker_init,
                    initargs=(device, self.teacher_cache_path, shared_cfg),
                )
            )

    def evaluate_candidates(
        self,
        policies: List[Dict[str, Any]],
        target_flops_reduction: float,
        compute_ppl: bool,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        futures = []
        for idx, policy_json in enumerate(policies):
            executor = self._executors[idx % len(self._executors)]
            futures.append(
                executor.submit(
                    _evaluate_policy_worker,
                    policy_json,
                    target_flops_reduction,
                    compute_ppl,
                    max_samples,
                )
            )
        return [future.result() for future in futures]

    def close(self):
        for executor in self._executors:
            executor.shutdown(wait=True, cancel_futures=False)
        self._executors.clear()
