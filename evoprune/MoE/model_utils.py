#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch.nn as nn


DEFAULT_EXPERT_DOWN_PROJ_REGEX = r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj$"


@dataclass(frozen=True)
class ExpertModuleRef:
    name: str
    layer_idx: int
    expert_idx: int


def parse_expert_module_name(
    module_name: str,
    expert_down_proj_regex: str = DEFAULT_EXPERT_DOWN_PROJ_REGEX,
) -> Optional[ExpertModuleRef]:
    match = re.match(expert_down_proj_regex, module_name)
    if match is None:
        return None
    return ExpertModuleRef(
        name=module_name,
        layer_idx=int(match.group(1)),
        expert_idx=int(match.group(2)),
    )


def discover_expert_down_proj_modules(
    model: nn.Module,
    expert_down_proj_regex: str = DEFAULT_EXPERT_DOWN_PROJ_REGEX,
) -> Dict[str, nn.Module]:
    modules: Dict[str, nn.Module] = {}
    for module_name, module in model.named_modules():
        parsed = parse_expert_module_name(module_name, expert_down_proj_regex=expert_down_proj_regex)
        if parsed is None:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear for expert down_proj, got {type(module)} at {module_name}")
        modules[module_name] = module
    return dict(sorted(modules.items(), key=lambda item: (parse_expert_module_name(item[0]).layer_idx, parse_expert_module_name(item[0]).expert_idx)))


def infer_num_experts_per_layer(
    model: nn.Module,
    expert_down_proj_regex: str = DEFAULT_EXPERT_DOWN_PROJ_REGEX,
) -> Dict[int, int]:
    modules = discover_expert_down_proj_modules(model, expert_down_proj_regex=expert_down_proj_regex)
    counts: Dict[int, int] = {}
    for module_name in modules:
        parsed = parse_expert_module_name(module_name, expert_down_proj_regex=expert_down_proj_regex)
        assert parsed is not None
        counts[parsed.layer_idx] = max(counts.get(parsed.layer_idx, 0), parsed.expert_idx + 1)
    return counts


def infer_default_cols_removed_mlp(intermediate_size: int, num_levels: int) -> List[int]:
    if num_levels < 2:
        raise ValueError("num_levels must be at least 2")
    if intermediate_size <= 0:
        raise ValueError("intermediate_size must be positive")
    vals = []
    for i in range(num_levels):
        removed = int(round(i * intermediate_size / (num_levels - 1)))
        vals.append(min(max(removed, 0), intermediate_size))
    vals = sorted(set(vals))
    if vals[0] != 0:
        vals = [0] + vals
    if vals[-1] != intermediate_size:
        vals.append(intermediate_size)
    return vals
