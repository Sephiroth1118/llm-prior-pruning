#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from evoprune.metrics import compute_perplexity
from evoprune.wanda_utils import compute_model_metrics

from .reward_utils import MoERewardConfig, score_candidate
from .simplified_candidate_utils import compression_ratio_from_policy
from .stitch_utils import (
    apply_policy_to_model,
    capture_sync_restore_state,
    load_base_model,
    normalize_moe_policy,
    restore_sync_modified_weights,
)

_WORKER_DEVICE: Optional[str] = None
_WORKER_DENSE_LOGITS_CACHE = None
_WORKER_SHARED_CFG: Optional[Dict[str, Any]] = None
_WORKER_MODEL = None
_WORKER_RESTORE_STATE: Optional[Dict[str, torch.Tensor]] = None


def _worker_init(device: str, teacher_cache_path: str, shared_cfg: Dict[str, Any]):
    global _WORKER_DEVICE, _WORKER_DENSE_LOGITS_CACHE, _WORKER_SHARED_CFG
    global _WORKER_MODEL, _WORKER_RESTORE_STATE
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
    _WORKER_RESTORE_STATE = capture_sync_restore_state(_WORKER_MODEL)


def _evaluate_policy_worker(policy_json: Dict[str, Any], compute_ppl: bool):
    assert _WORKER_DEVICE is not None
    assert _WORKER_SHARED_CFG is not None
    assert _WORKER_MODEL is not None
    assert _WORKER_RESTORE_STATE is not None
    try:
        restore_sync_modified_weights(_WORKER_MODEL, _WORKER_RESTORE_STATE)
        normalized_policy = normalize_moe_policy(
            policy_json,
            policy_bounds=_WORKER_SHARED_CFG["policy_bounds"],
            expert_groups=_WORKER_SHARED_CFG.get("expert_groups"),
        )
        apply_policy_to_model(_WORKER_MODEL, db_path=_WORKER_SHARED_CFG["db_path"], policy=normalized_policy)
        metrics = compute_model_metrics(
            model=_WORKER_MODEL,
            input_ids_list=_WORKER_SHARED_CFG["eval_data"],
            device=_WORKER_DEVICE,
            max_samples=None,
            dense_logits_cache=_WORKER_DENSE_LOGITS_CACHE,
            compute_wanda=False,
        )
        compression_ratio = compression_ratio_from_policy(
            normalized_policy,
            _WORKER_SHARED_CFG["policy_bounds"],
        )
        ppl = None
        if compute_ppl and _WORKER_SHARED_CFG["ppl_eval_data"] is not None:
            ppl = float(compute_perplexity(_WORKER_MODEL, _WORKER_SHARED_CFG["ppl_eval_data"], batch_size=1))
        reward = score_candidate(
            target_flops_reduction=float(_WORKER_SHARED_CFG["target_flops_reduction"]),
            compression_ratio=compression_ratio,
            kl_divergence_from_dense=metrics.kl_divergence_from_dense,
            ppl=ppl,
            reward_cfg=MoERewardConfig(),
            valid=True,
        )
        return {
            "metrics": metrics,
            "reward": reward,
            "ppl": ppl,
            "device": _WORKER_DEVICE,
            "compression_ratio": compression_ratio,
            "error_type": None,
            "error_message": None,
        }
    except Exception as exc:
        return {
            "metrics": None,
            "reward": score_candidate(
                target_flops_reduction=float(_WORKER_SHARED_CFG["target_flops_reduction"]),
                compression_ratio=0.0,
                kl_divergence_from_dense=None,
                ppl=None,
                reward_cfg=MoERewardConfig(),
                valid=False,
                error_message=str(exc),
            ),
            "ppl": None,
            "device": _WORKER_DEVICE,
            "compression_ratio": 0.0,
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
        policy_bounds: Dict[str, Any],
        expert_groups: Optional[Dict[str, Any]],
        eval_data,
        ppl_eval_data,
        target_flops_reduction: float,
        dtype: str,
        low_cpu_mem_usage: bool,
        attn_implementation: Optional[str],
    ):
        self.devices = devices
        self.teacher_cache_path = str(Path(teacher_cache_path))
        self._executors: List[ProcessPoolExecutor] = []
        shared_cfg = {
            "model_name_or_path": model_name_or_path,
            "db_path": db_path,
            "policy_bounds": policy_bounds,
            "expert_groups": expert_groups,
            "eval_data": eval_data,
            "ppl_eval_data": ppl_eval_data,
            "target_flops_reduction": float(target_flops_reduction),
            "dtype": dtype,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "attn_implementation": attn_implementation,
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
        compute_ppl: bool,
    ) -> List[Dict[str, Any]]:
        futures = []
        for idx, policy_json in enumerate(policies):
            executor = self._executors[idx % len(self._executors)]
            futures.append(executor.submit(_evaluate_policy_worker, policy_json, compute_ppl))
        return [future.result() for future in futures]

    def close(self):
        for executor in self._executors:
            executor.shutdown(wait=True, cancel_futures=False)
        self._executors.clear()
