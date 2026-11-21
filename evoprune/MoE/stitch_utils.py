#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM

from evoprune.transformers_compat import patch_transformers_pruning_utils

from .db import MoEDatabase
from .expert_groups import load_expert_groups
from .model_utils import DEFAULT_EXPERT_DOWN_PROJ_REGEX, parse_expert_module_name
from .policy import (
    build_policy_json_from_mlp_levels,
    convert_legacy_layer_policy_to_moe_group_levels,
    normalize_moe_policy,
)


@dataclass
class StitchedMoEModelResult:
    model: Any
    policy: Dict[str, int]
    model_name_or_path: str
    db_path: str
    dtype: str
    attn_implementation: Optional[str]
    device: str


def resolve_torch_dtype(dtype_str: str):
    if dtype_str == "auto":
        return "auto"
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def load_policy_json(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Policy JSON must be a dict")
    return obj


def load_base_model(
    model_name_or_path: str,
    dtype: str = "auto",
    low_cpu_mem_usage: bool = False,
    attn_implementation: Optional[str] = None,
    device: str = "cuda",
):
    patch_transformers_pruning_utils()
    torch_dtype = resolve_torch_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        attn_implementation=attn_implementation,
    )
    model.eval()
    if device != "auto":
        model = model.to(device)
    return model


def load_policy(
    db_path: str,
    config_path: Optional[str] = None,
    policy: Optional[Dict[str, int]] = None,
    default_level: int = 0,
    expert_groups_path: Optional[str] = None,
) -> Dict[str, int]:
    db = MoEDatabase(db_path)
    policy_bounds = db.get_policy_level_bounds()
    expert_groups = load_expert_groups(expert_groups_path) if expert_groups_path else None
    if policy is not None:
        if isinstance(policy, dict) and "mlp_levels" in policy and "layers" not in policy:
            policy = build_policy_json_from_mlp_levels(db.get_policy_level_bounds(), policy["mlp_levels"])
        if isinstance(policy, dict) and "moe_levels" in policy and "layers" not in policy and expert_groups is not None:
            policy = convert_legacy_layer_policy_to_moe_group_levels(policy, num_groups=int(expert_groups["num_groups"]))
        return normalize_moe_policy(policy, policy_bounds=policy_bounds, expert_groups=expert_groups)
    if config_path is not None:
        loaded = load_policy_json(config_path)
        if isinstance(loaded, dict) and "mlp_levels" in loaded and "layers" not in loaded:
            loaded = build_policy_json_from_mlp_levels(db.get_policy_level_bounds(), loaded["mlp_levels"])
        if isinstance(loaded, dict) and "moe_levels" in loaded and "layers" not in loaded and expert_groups is not None:
            loaded = convert_legacy_layer_policy_to_moe_group_levels(loaded, num_groups=int(expert_groups["num_groups"]))
        return normalize_moe_policy(loaded, policy_bounds=policy_bounds, expert_groups=expert_groups)
    return {entry.name: int(default_level) for entry in db.entries}


def _sync_single_expert(expert_module) -> None:
    down_w = expert_module.down_proj.weight.data
    active_mask = down_w.detach().float().cpu().norm(dim=0) > 1e-8
    pruned = ~active_mask
    if pruned.any():
        gate_w = expert_module.gate_proj.weight.data
        up_w = expert_module.up_proj.weight.data
        gate_w[pruned.to(gate_w.device), :] = 0
        up_w[pruned.to(up_w.device), :] = 0
        down_w[:, pruned.to(down_w.device)] = 0


def capture_sync_restore_state(
    model,
    expert_down_proj_regex: str = DEFAULT_EXPERT_DOWN_PROJ_REGEX,
) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        parsed = parse_expert_module_name(
            module_name,
            expert_down_proj_regex=expert_down_proj_regex,
        )
        if parsed is None:
            continue
        expert_module_name = module_name.rsplit(".", 1)[0]
        expert_module = model.get_submodule(expert_module_name)
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(expert_module, proj_name, None)
            if proj is not None and hasattr(proj, "weight"):
                state[f"{expert_module_name}.{proj_name}.weight"] = proj.weight.detach().cpu().clone()
    return state


def restore_sync_modified_weights(model, restore_state: Dict[str, torch.Tensor]) -> None:
    for tensor_name, tensor in restore_state.items():
        module_name, param_name = tensor_name.rsplit(".", 1)
        module = model.get_submodule(module_name)
        param = getattr(module, param_name)
        param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))


def synchronize_expert_zero_masks(model, policy: Dict[str, int]) -> None:
    touched = set()
    for module_name in policy:
        expert_module_name = module_name.rsplit(".", 1)[0]
        if expert_module_name in touched:
            continue
        touched.add(expert_module_name)
        expert_module = model.get_submodule(expert_module_name)
        if all(hasattr(expert_module, attr) for attr in ("gate_proj", "up_proj", "down_proj")):
            _sync_single_expert(expert_module)


def apply_policy_to_model(model, db_path: str, policy: Dict[str, int], sync_zero_masks: bool = True):
    for module_name, level in policy.items():
        weight_path = os.path.join(db_path, module_name, f"{int(level)}.pth")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing DB slice: {weight_path}")
        layer = model.get_submodule(module_name)
        sparse_weight = torch.load(weight_path, map_location="cpu")
        if tuple(sparse_weight.shape) != tuple(layer.weight.data.shape):
            raise ValueError(
                f"Shape mismatch at {module_name}: DB {tuple(sparse_weight.shape)} vs model {tuple(layer.weight.data.shape)}"
            )
        layer.weight.data.copy_(sparse_weight.to(device=layer.weight.device, dtype=layer.weight.dtype))
    if sync_zero_masks:
        synchronize_expert_zero_masks(model, policy)
    return model


def stitch_model(
    model_name_or_path: str,
    db_path: str,
    config_path: Optional[str] = None,
    policy: Optional[Dict[str, int]] = None,
    default_level: int = 0,
    dtype: str = "auto",
    low_cpu_mem_usage: bool = False,
    attn_implementation: Optional[str] = None,
    device: str = "cuda",
    sync_zero_masks: bool = True,
    expert_groups_path: Optional[str] = None,
) -> StitchedMoEModelResult:
    model = load_base_model(
        model_name_or_path=model_name_or_path,
        dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        attn_implementation=attn_implementation,
        device=device,
    )
    normalized_policy = load_policy(
        db_path=db_path,
        config_path=config_path,
        policy=policy,
        default_level=default_level,
        expert_groups_path=expert_groups_path,
    )
    model = apply_policy_to_model(model, db_path=db_path, policy=normalized_policy, sync_zero_masks=sync_zero_masks)
    return StitchedMoEModelResult(
        model=model,
        policy=normalized_policy,
        model_name_or_path=model_name_or_path,
        db_path=db_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        device=device,
    )
