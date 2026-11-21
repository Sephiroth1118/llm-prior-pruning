#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import torch
from accelerate import dispatch_model, infer_auto_device_map
from transformers import AutoModelForCausalLM

from evoprune.transformers_compat import patch_transformers_pruning_utils


DEFAULT_ATTN_PRUNE_NAME = "o_proj"
DEFAULT_MLP_PRUNE_NAME = "down_proj"
DEFAULT_DEFAULT_LEVEL = 0


@dataclass
class StitchedModelResult:
    model: Any
    policy: Dict[str, int]
    sync_info: Optional[Dict[str, Any]]
    model_name_or_path: str
    db_path: str
    dtype: str
    attn_implementation: Optional[str]
    device: str
    sync_zero_masks: bool


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


def list_db_layer_dirs(db_path: str) -> List[str]:
    if not os.path.isdir(db_path):
        raise FileNotFoundError(f"DB path does not exist or is not a directory: {db_path}")

    names = []
    for name in sorted(os.listdir(db_path)):
        full = os.path.join(db_path, name)
        if os.path.isdir(full):
            names.append(name)
    return names


def _list_level_files(layer_db_dir: str) -> List[int]:
    levels = []
    for name in os.listdir(layer_db_dir):
        if not name.endswith(".pth"):
            continue
        stem = os.path.splitext(name)[0]
        if stem.isdigit():
            levels.append(int(stem))
    return sorted(levels)


def build_layer_name(
    layer_id: int,
    module_type: str,
    attn_prune_name: str = DEFAULT_ATTN_PRUNE_NAME,
    mlp_prune_name: str = DEFAULT_MLP_PRUNE_NAME,
) -> str:
    if module_type == "attn":
        return f"model.layers.{layer_id}.self_attn.{attn_prune_name}"
    if module_type == "mlp":
        return f"model.layers.{layer_id}.mlp.{mlp_prune_name}"
    raise ValueError(f"Unsupported module_type: {module_type}")


def get_policy_level_bounds(
    db_path: str,
    attn_prune_name: str = DEFAULT_ATTN_PRUNE_NAME,
    mlp_prune_name: str = DEFAULT_MLP_PRUNE_NAME,
) -> Dict[str, Any]:
    layers = {}
    max_layer_idx = -1

    for layer_name in list_db_layer_dirs(db_path):
        level_dir = os.path.join(db_path, layer_name)
        levels = _list_level_files(level_dir)
        if not levels:
            continue

        parts = layer_name.split(".")
        if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
            continue
        layer_idx = int(parts[2])
        max_layer_idx = max(max_layer_idx, layer_idx)
        layer_key = str(layer_idx)
        if layer_key not in layers:
            layers[layer_key] = {
                "attn_min": 0,
                "attn_max": 0,
                "mlp_min": 0,
                "mlp_max": 0,
            }

        if layer_name == build_layer_name(layer_idx, "attn", attn_prune_name, mlp_prune_name):
            layers[layer_key]["attn_min"] = min(levels)
            layers[layer_key]["attn_max"] = max(levels)
        elif layer_name == build_layer_name(layer_idx, "mlp", attn_prune_name, mlp_prune_name):
            layers[layer_key]["mlp_min"] = min(levels)
            layers[layer_key]["mlp_max"] = max(levels)

    return {
        "policy_type": "db_level_bounds",
        "num_layers": max_layer_idx + 1,
        "layers": layers,
    }


def load_policy_json(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Policy JSON must be a dict")
    return obj


def normalize_policy(
    policy: Dict[str, Any],
    attn_prune_name: str = DEFAULT_ATTN_PRUNE_NAME,
    mlp_prune_name: str = DEFAULT_MLP_PRUNE_NAME,
) -> Dict[str, int]:
    if not isinstance(policy, dict):
        raise ValueError("policy must be a dict")

    if "db_level_policy" in policy:
        nested_policy = policy["db_level_policy"]
        if not isinstance(nested_policy, dict):
            raise ValueError("'db_level_policy' must be a dict")
        policy = nested_policy

    if "policy" in policy and isinstance(policy["policy"], dict):
        nested_policy = policy["policy"]
        if "attn_levels" in nested_policy and "mlp_levels" in nested_policy:
            attn_levels = nested_policy["attn_levels"]
            mlp_levels = nested_policy["mlp_levels"]
            if len(attn_levels) != len(mlp_levels):
                raise ValueError("policy.attn_levels and policy.mlp_levels must have the same length")
            return {
                **{
                    build_layer_name(i, "attn", attn_prune_name, mlp_prune_name): int(attn_levels[i])
                    for i in range(len(attn_levels))
                },
                **{
                    build_layer_name(i, "mlp", attn_prune_name, mlp_prune_name): int(mlp_levels[i])
                    for i in range(len(mlp_levels))
                },
            }

    if "attn_levels" in policy and "mlp_levels" in policy:
        attn_levels = policy["attn_levels"]
        mlp_levels = policy["mlp_levels"]
        if len(attn_levels) != len(mlp_levels):
            raise ValueError("attn_levels and mlp_levels must have the same length")
        return {
            **{
                build_layer_name(i, "attn", attn_prune_name, mlp_prune_name): int(attn_levels[i])
                for i in range(len(attn_levels))
            },
            **{
                build_layer_name(i, "mlp", attn_prune_name, mlp_prune_name): int(mlp_levels[i])
                for i in range(len(mlp_levels))
            },
        }

    if "layers" not in policy or "num_layers" not in policy:
        return {str(name): int(level) for name, level in policy.items()}

    if "policy_type" in policy and policy.get("policy_type") != "db_level_policy":
        raise ValueError("policy_type must be 'db_level_policy'")

    num_layers = int(policy["num_layers"])
    layers = policy["layers"]
    if not isinstance(layers, dict):
        raise ValueError("'layers' must be a dict")

    normalized = {}
    for i in range(num_layers):
        layer_key = str(i)
        if layer_key not in layers:
            raise ValueError(f"Missing layer '{layer_key}' in policy")
        item = layers[layer_key]
        if "attn_level" not in item or "mlp_level" not in item:
            raise ValueError(f"Layer '{layer_key}' must contain attn_level and mlp_level")
        normalized[build_layer_name(i, "attn", attn_prune_name, mlp_prune_name)] = int(item["attn_level"])
        normalized[build_layer_name(i, "mlp", attn_prune_name, mlp_prune_name)] = int(item["mlp_level"])
    return normalized


def load_policy(
    config_path: Optional[str],
    db_path: str,
    default_level: int = DEFAULT_DEFAULT_LEVEL,
    attn_prune_name: str = DEFAULT_ATTN_PRUNE_NAME,
    mlp_prune_name: str = DEFAULT_MLP_PRUNE_NAME,
) -> Dict[str, int]:
    db_layers = list_db_layer_dirs(db_path)

    if config_path is None:
        return {layer_name: int(default_level) for layer_name in db_layers}

    raw_policy = normalize_policy(
        load_policy_json(config_path),
        attn_prune_name=attn_prune_name,
        mlp_prune_name=mlp_prune_name,
    )

    full_policy = {}
    for layer_name in db_layers:
        full_policy[layer_name] = int(raw_policy.get(layer_name, default_level))
    return full_policy


def load_base_model(
    model_name_or_path: str,
    dtype: str = "auto",
    low_cpu_mem_usage: bool = False,
    attn_implementation: Optional[str] = None,
    device: str = "cuda",
    local_files_only: bool = False,
):
    patch_transformers_pruning_utils()
    torch_dtype = resolve_torch_dtype(dtype)

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "attn_implementation": attn_implementation,
        "local_files_only": local_files_only,
    }
    if device == "auto":
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **load_kwargs,
    )
    model.eval()
    if device != "auto":
        model = model.to(device)
    return model


def _redispatch_model_auto(model, dtype: str = "auto"):
    torch_dtype = resolve_torch_dtype(dtype)
    no_split_module_classes = getattr(model, "_no_split_modules", None)
    model.tie_weights()
    device_map = infer_auto_device_map(
        model,
        no_split_module_classes=no_split_module_classes,
        dtype=None if torch_dtype == "auto" else torch_dtype,
    )
    return dispatch_model(model, device_map=device_map)


def apply_policy_to_model(model, db_path: str, policy: Dict[str, int]):
    """
    Replace DB-controlled weights (typically o_proj / down_proj)
    with full-shape sparse slices from the database.
    """
    for layer_name, level in policy.items():
        weight_path = os.path.join(db_path, layer_name, f"{level}.pth")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing DB slice: {weight_path}")

        try:
            layer = model.get_submodule(layer_name)
        except Exception as e:
            raise ValueError(f"Layer not found in model: {layer_name}") from e

        if not hasattr(layer, "weight"):
            raise ValueError(f"Target module has no weight attribute: {layer_name}")

        orig_dtype = layer.weight.dtype
        target_device = layer.weight.device

        sparse_weight = torch.load(weight_path, map_location="cpu")
        if not isinstance(sparse_weight, torch.Tensor):
            raise TypeError(f"Expected a tensor in {weight_path}, got {type(sparse_weight)}")

        sparse_weight = sparse_weight.to(dtype=orig_dtype)

        if tuple(sparse_weight.shape) != tuple(layer.weight.data.shape):
            raise ValueError(
                f"Shape mismatch at {layer_name}: "
                f"DB {tuple(sparse_weight.shape)} vs model {tuple(layer.weight.data.shape)}"
            )

        layer.weight.data.copy_(sparse_weight.to(target_device))

    return model


def capture_sync_restore_state(model) -> Dict[str, torch.Tensor]:
    restore_state = {}
    for layer_idx, layer in enumerate(model.model.layers):
        restore_state[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = (
            layer.self_attn.q_proj.weight.detach().cpu().clone()
        )
        restore_state[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = (
            layer.self_attn.k_proj.weight.detach().cpu().clone()
        )
        restore_state[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = (
            layer.self_attn.v_proj.weight.detach().cpu().clone()
        )
        restore_state[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = (
            layer.mlp.gate_proj.weight.detach().cpu().clone()
        )
        restore_state[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = (
            layer.mlp.up_proj.weight.detach().cpu().clone()
        )
    return restore_state


def restore_sync_modified_weights(model, restore_state: Dict[str, torch.Tensor]):
    for name, dense_weight in restore_state.items():
        layer = model.get_submodule(name.rsplit(".", 1)[0])
        layer.weight.data.copy_(dense_weight.to(device=layer.weight.device, dtype=layer.weight.dtype))
    return model


def _zero_query_head_rows(linear_layer, head_mask: torch.Tensor, head_dim: int):
    w = linear_layer.weight.data
    num_heads = head_mask.numel()
    expected_rows = num_heads * head_dim
    if w.shape[0] != expected_rows:
        raise ValueError(f"Row mismatch: got {w.shape[0]}, expected {expected_rows}")

    pruned = ~head_mask
    if pruned.any():
        row_mask = pruned.repeat_interleave(head_dim).to(w.device)
        w[row_mask, :] = 0


def _zero_kv_group_rows(linear_layer, kv_group_mask: torch.Tensor, head_dim: int):
    w = linear_layer.weight.data
    num_kv_heads = kv_group_mask.numel()
    expected_rows = num_kv_heads * head_dim
    if w.shape[0] != expected_rows:
        raise ValueError(f"KV row mismatch: got {w.shape[0]}, expected {expected_rows}")

    pruned = ~kv_group_mask
    if pruned.any():
        row_mask = pruned.repeat_interleave(head_dim).to(w.device)
        w[row_mask, :] = 0


def _infer_active_query_head_mask_from_o(layer) -> torch.Tensor:
    """
    Infer active query/output heads from zero/non-zero column blocks of o_proj.
    Returns bool tensor of shape [num_attention_heads], True=active.
    """
    attn = layer.self_attn
    w = attn.o_proj.weight.data.detach().float().cpu()

    if not hasattr(attn, "head_dim"):
        raise ValueError("Cannot find self_attn.head_dim")

    head_dim = attn.head_dim

    q_out_features = attn.q_proj.weight.shape[0]
    if q_out_features % head_dim != 0:
        raise ValueError(
            f"q_proj out_features ({q_out_features}) not divisible by head_dim ({head_dim})"
        )
    num_heads = q_out_features // head_dim

    expected_in = num_heads * head_dim
    if w.shape[1] != expected_in:
        raise ValueError(f"o_proj in_features mismatch: got {w.shape[1]}, expected {expected_in}")

    block_norms = w.view(w.shape[0], num_heads, head_dim).norm(dim=(0, 2))
    return block_norms > 1e-8


def _sync_attention_zero_masks(layer):
    """
    Keep shapes unchanged. Infer pruned query heads from o_proj and propagate zero masks:
      - q_proj: zero rows for pruned query heads
      - o_proj: enforce zero input columns
      - MHA/GQA: propagate to k_proj/v_proj
    """
    attn = layer.self_attn
    active_q_mask = _infer_active_query_head_mask_from_o(layer)

    if not hasattr(attn, "head_dim"):
        raise ValueError("Cannot find self_attn.head_dim")
    head_dim = attn.head_dim

    num_heads = active_q_mask.numel()

    k_out_features = attn.k_proj.weight.shape[0]
    if k_out_features % head_dim != 0:
        raise ValueError(
            f"k_proj out_features ({k_out_features}) not divisible by head_dim ({head_dim})"
        )
    num_kv_heads = k_out_features // head_dim

    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads={num_heads} not divisible by num_kv_heads={num_kv_heads}")
    group_size = num_heads // num_kv_heads

    _zero_query_head_rows(attn.q_proj, active_q_mask, head_dim=head_dim)

    o_w = attn.o_proj.weight.data
    pruned_q = (~active_q_mask).repeat_interleave(head_dim).to(o_w.device)
    if pruned_q.any():
        o_w[:, pruned_q] = 0

    active_kv_mask = torch.zeros(num_kv_heads, dtype=torch.bool)
    for g in range(num_kv_heads):
        start = g * group_size
        end = (g + 1) * group_size
        active_kv_mask[g] = bool(active_q_mask[start:end].any())

    _zero_kv_group_rows(attn.k_proj, active_kv_mask, head_dim=head_dim)
    _zero_kv_group_rows(attn.v_proj, active_kv_mask, head_dim=head_dim)

    return {
        "active_query_head_mask": active_q_mask.int().tolist(),
        "active_query_head_count": int(active_q_mask.sum().item()),
        "pruned_query_head_count": int((~active_q_mask).sum().item()),
        "active_kv_group_count": int(active_kv_mask.sum().item()),
        "pruned_kv_group_count": int((~active_kv_mask).sum().item()),
    }


def _infer_active_mlp_dim_mask_from_down(layer) -> torch.Tensor:
    """
    Infer active intermediate dims from zero/non-zero input columns of down_proj.
    True = active.
    """
    w = layer.mlp.down_proj.weight.data.detach().float().cpu()
    col_norms = w.norm(dim=0)
    return col_norms > 1e-8


def _sync_mlp_zero_masks(layer):
    """
    Keep shapes unchanged. Infer pruned intermediate dims from down_proj and propagate zero masks:
      - gate_proj: zero corresponding rows
      - up_proj: zero corresponding rows
      - down_proj: enforce corresponding input columns zero
    """
    mlp = layer.mlp
    active_dim_mask = _infer_active_mlp_dim_mask_from_down(layer)

    pruned = ~active_dim_mask

    gate_w = mlp.gate_proj.weight.data
    up_w = mlp.up_proj.weight.data
    down_w = mlp.down_proj.weight.data

    row_mask = pruned.to(gate_w.device)
    if row_mask.any():
        gate_w[row_mask, :] = 0
        up_w[row_mask, :] = 0

    col_mask = pruned.to(down_w.device)
    if col_mask.any():
        down_w[:, col_mask] = 0

    return {
        "active_mlp_dim_count": int(active_dim_mask.sum().item()),
        "pruned_mlp_dim_count": int((~active_dim_mask).sum().item()),
    }


def synchronize_zero_masks_without_shrink(model):
    """
    Synchronize qkv / up / gate zero masks from o_proj / down_proj,
    while keeping all tensor shapes unchanged.
    """
    layers = model.model.layers
    sync_info = {"attention": [], "mlp": []}

    for layer_idx, layer in enumerate(layers):
        attn_info = _sync_attention_zero_masks(layer)
        attn_info["layer_idx"] = layer_idx
        sync_info["attention"].append(attn_info)

        mlp_info = _sync_mlp_zero_masks(layer)
        mlp_info["layer_idx"] = layer_idx
        sync_info["mlp"].append(mlp_info)

    return model, sync_info


def _model_has_meta_parameters(model) -> bool:
    for param in model.parameters():
        if getattr(param, "is_meta", False):
            return True
    return False


def stitch_model(
    model_name_or_path: str,
    db_path: str,
    config_path: Optional[str] = None,
    policy: Optional[Dict[str, int]] = None,
    default_level: int = DEFAULT_DEFAULT_LEVEL,
    attn_prune_name: str = DEFAULT_ATTN_PRUNE_NAME,
    mlp_prune_name: str = DEFAULT_MLP_PRUNE_NAME,
    dtype: str = "auto",
    low_cpu_mem_usage: bool = False,
    attn_implementation: Optional[str] = None,
    device: str = "cuda",
    sync_zero_masks: bool = True,
    local_files_only: bool = False,
) -> StitchedModelResult:
    load_device = device
    load_low_cpu_mem_usage = low_cpu_mem_usage
    if device == "auto" and sync_zero_masks:
        load_device = "cpu"
        load_low_cpu_mem_usage = False

    model = load_base_model(
        model_name_or_path=model_name_or_path,
        dtype=dtype,
        low_cpu_mem_usage=load_low_cpu_mem_usage,
        attn_implementation=attn_implementation,
        device=load_device,
        local_files_only=local_files_only,
    )

    if policy is None:
        policy = load_policy(
            config_path=config_path,
            db_path=db_path,
            default_level=default_level,
            attn_prune_name=attn_prune_name,
            mlp_prune_name=mlp_prune_name,
        )
    else:
        policy = normalize_policy(
            policy,
            attn_prune_name=attn_prune_name,
            mlp_prune_name=mlp_prune_name,
        )

    model = apply_policy_to_model(model, db_path=db_path, policy=policy)

    sync_info = None
    if sync_zero_masks:
        if _model_has_meta_parameters(model):
            warnings.warn(
                "Skipping zero-mask synchronization because some parameters remain on the meta "
                "device under auto device mapping / CPU offload loading.",
                stacklevel=2,
            )
        else:
            model, sync_info = synchronize_zero_masks_without_shrink(model)

    if device == "auto" and load_device == "cpu":
        model = _redispatch_model_auto(model, dtype=dtype)

    return StitchedModelResult(
        model=model,
        policy=policy,
        sync_info=sync_info,
        model_name_or_path=model_name_or_path,
        db_path=db_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        device=device,
        sync_zero_masks=sync_zero_masks,
    )
