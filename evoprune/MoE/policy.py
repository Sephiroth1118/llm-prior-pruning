#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .db import MoEDatabase


def build_moe_policy_json(db: MoEDatabase, level_by_module: Dict[str, int]) -> Dict[str, object]:
    layers = {}
    for layer_idx, entries in db.by_layer().items():
        expert_levels = {}
        for entry in entries:
            expert_levels[str(entry.expert_idx)] = int(level_by_module[entry.name])
        layers[str(layer_idx)] = {"expert_levels": expert_levels}
    return {
        "policy_type": "moe_expert_level_policy",
        "num_layers": int(db.num_layers),
        "layers": layers,
    }


def build_uniform_policy_json(db: MoEDatabase, level: int) -> Dict[str, object]:
    return build_moe_policy_json(
        db=db,
        level_by_module={entry.name: int(level) for entry in db.entries},
    )


def build_policy_json_from_expert_levels_by_layer(expert_levels_by_layer: List[List[int]]) -> Dict[str, object]:
    layers = {}
    for layer_idx, expert_levels in enumerate(expert_levels_by_layer):
        layers[str(layer_idx)] = {
            "expert_levels": {str(expert_idx): int(level) for expert_idx, level in enumerate(expert_levels)}
        }
    return {
        "policy_type": "moe_expert_level_policy",
        "num_layers": len(expert_levels_by_layer),
        "layers": layers,
    }


def build_policy_json_from_mlp_levels(
    policy_bounds: Dict[str, object],
    mlp_levels: List[int],
) -> Dict[str, object]:
    num_layers = int(policy_bounds["num_layers"])
    if len(mlp_levels) != num_layers:
        raise ValueError(f"mlp_levels must contain exactly {num_layers} integers")
    layers = {}
    bounds_layers = policy_bounds["layers"]
    for layer_idx in range(num_layers):
        layer_obj = bounds_layers[str(layer_idx)]
        level = int(mlp_levels[layer_idx])
        layers[str(layer_idx)] = {
            "expert_levels": {
                str(expert_idx): level
                for expert_idx in range(int(layer_obj["num_experts"]))
            }
        }
    return {
        "policy_type": "moe_expert_level_policy",
        "num_layers": num_layers,
        "layers": layers,
    }


def validate_moe_group_levels_shape(
    moe_group_levels: Any,
    num_layers: int,
    num_groups: int,
) -> Tuple[List[List[int]], List[str]]:
    warnings: List[str] = []
    if not isinstance(moe_group_levels, list):
        raise ValueError("moe_group_levels must be a list")
    if len(moe_group_levels) != num_layers:
        raise ValueError(f"moe_group_levels must contain exactly {num_layers} rows")
    validated: List[List[int]] = []
    for layer_idx, row in enumerate(moe_group_levels):
        if not isinstance(row, list):
            raise ValueError(f"moe_group_levels[{layer_idx}] must be a list")
        if len(row) != num_groups:
            raise ValueError(
                f"moe_group_levels[{layer_idx}] must contain exactly {num_groups} integers"
            )
        validated_row = []
        for group_idx, value in enumerate(row):
            clamped = max(0, min(10, int(value)))
            if clamped != int(value):
                warnings.append(
                    f"clamped moe_group_levels[{layer_idx}][{group_idx}] from {value} to {clamped}"
                )
            validated_row.append(clamped)
        validated.append(validated_row)
    return validated, warnings


def expand_moe_group_levels_to_level_by_module(
    policy_bounds: Dict[str, Any],
    expert_groups: Dict[str, Any],
    moe_group_levels: List[List[int]],
) -> Dict[str, int]:
    num_layers = int(policy_bounds["num_layers"])
    num_groups = int(expert_groups["num_groups"])
    validated, _ = validate_moe_group_levels_shape(moe_group_levels, num_layers, num_groups)
    level_by_module: Dict[str, int] = {}
    for layer_idx in range(num_layers):
        layer_groups = expert_groups["groups"][str(layer_idx)]
        for group_idx in range(num_groups):
            level = int(validated[layer_idx][group_idx])
            for expert_idx in layer_groups[str(group_idx)]:
                level_by_module[
                    f"model.layers.{layer_idx}.mlp.experts.{int(expert_idx)}.down_proj"
                ] = level
    return level_by_module


def build_policy_json_from_moe_group_levels(
    policy_bounds: Dict[str, Any],
    expert_groups: Dict[str, Any],
    moe_group_levels: List[List[int]],
) -> Dict[str, object]:
    level_by_module = expand_moe_group_levels_to_level_by_module(
        policy_bounds=policy_bounds,
        expert_groups=expert_groups,
        moe_group_levels=moe_group_levels,
    )
    layers: Dict[str, Dict[str, Dict[str, int]]] = {}
    for module_name, level in sorted(level_by_module.items()):
        parts = module_name.split(".")
        layer_idx = int(parts[2])
        expert_idx = int(parts[5])
        layers.setdefault(str(layer_idx), {"expert_levels": {}})
        layers[str(layer_idx)]["expert_levels"][str(expert_idx)] = int(level)
    return {
        "policy_type": "moe_expert_level_policy",
        "num_layers": int(policy_bounds["num_layers"]),
        "layers": layers,
    }


def convert_legacy_layer_policy_to_moe_group_levels(
    policy: Dict[str, object],
    num_groups: int,
) -> Dict[str, object]:
    if "moe_levels" in policy:
        source = policy["moe_levels"]
    elif "mlp_levels" in policy:
        source = policy["mlp_levels"]
    else:
        raise ValueError("expected legacy policy to contain 'moe_levels' or 'mlp_levels'")
    if not isinstance(source, list):
        raise ValueError("legacy layer-wise policy values must be a list")
    return {
        "policy_type": "moe_group_level_policy",
        "num_layers": len(source),
        "num_groups": int(num_groups),
        "moe_group_levels": [[int(level)] * int(num_groups) for level in source],
    }


def normalize_moe_policy(
    policy: Dict[str, object],
    policy_bounds: Dict[str, Any] | None = None,
    expert_groups: Dict[str, Any] | None = None,
) -> Dict[str, int]:
    if "layers" not in policy:
        if "moe_group_levels" in policy:
            if policy_bounds is None or expert_groups is None:
                raise ValueError(
                    "expert_groups_path is required for group-wise policy format 'moe_group_levels'"
                )
            return expand_moe_group_levels_to_level_by_module(
                policy_bounds=policy_bounds,
                expert_groups=expert_groups,
                moe_group_levels=policy["moe_group_levels"],
            )
        if "moe_levels" in policy or "mlp_levels" in policy:
            if expert_groups is not None and policy_bounds is not None:
                converted = convert_legacy_layer_policy_to_moe_group_levels(
                    policy=policy,
                    num_groups=int(expert_groups["num_groups"]),
                )
                return expand_moe_group_levels_to_level_by_module(
                    policy_bounds=policy_bounds,
                    expert_groups=expert_groups,
                    moe_group_levels=converted["moe_group_levels"],
                )
            if "moe_levels" in policy:
                values = policy["moe_levels"]
            else:
                values = policy["mlp_levels"]
            return {
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj": int(level)
                for layer_idx, level in enumerate(values)
                for expert_idx in range(int(policy_bounds["layers"][str(layer_idx)]["num_experts"]))  # type: ignore[index]
            }
        return {str(k): int(v) for k, v in policy.items()}
    normalized: Dict[str, int] = {}
    layers = policy["layers"]
    if not isinstance(layers, dict):
        raise ValueError("'layers' must be a dict")
    for layer_key, layer_obj in layers.items():
        if not isinstance(layer_obj, dict):
            raise ValueError(f"layer '{layer_key}' must be a dict")
        expert_levels = layer_obj.get("expert_levels")
        if not isinstance(expert_levels, dict):
            raise ValueError(f"layer '{layer_key}' must contain expert_levels dict")
        for expert_key, level in expert_levels.items():
            normalized[f"model.layers.{int(layer_key)}.mlp.experts.{int(expert_key)}.down_proj"] = int(level)
    return normalized


def save_policy_json(path: str, policy: Dict[str, object]) -> None:
    Path(path).write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
