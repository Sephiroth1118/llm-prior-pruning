#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from .model_utils import DEFAULT_EXPERT_DOWN_PROJ_REGEX, parse_expert_module_name


@dataclass(frozen=True)
class MoEModuleEntry:
    name: str
    layer_idx: int
    expert_idx: int
    levels: List[int]


class MoEDatabase:
    def __init__(
        self,
        db_path: str,
        expert_down_proj_regex: str = DEFAULT_EXPERT_DOWN_PROJ_REGEX,
    ) -> None:
        self.db_path = db_path
        self.expert_down_proj_regex = expert_down_proj_regex
        self.metadata_path = os.path.join(db_path, "metadata.pth")
        self.metadata = self._load_metadata()
        self.entries = self._discover_entries()

    def _load_metadata(self) -> Dict[str, Any]:
        if not os.path.exists(self.metadata_path):
            return {}
        obj = torch.load(self.metadata_path, map_location="cpu")
        if not isinstance(obj, dict):
            raise TypeError(f"Expected dict in {self.metadata_path}, got {type(obj)}")
        return obj

    def _discover_entries(self) -> List[MoEModuleEntry]:
        if not os.path.isdir(self.db_path):
            raise FileNotFoundError(f"DB path does not exist: {self.db_path}")
        entries: List[MoEModuleEntry] = []
        for name in sorted(os.listdir(self.db_path)):
            full = os.path.join(self.db_path, name)
            if not os.path.isdir(full):
                continue
            parsed = parse_expert_module_name(name, expert_down_proj_regex=self.expert_down_proj_regex)
            if parsed is None:
                continue
            levels = []
            for file_name in os.listdir(full):
                if file_name.endswith(".pth"):
                    stem = os.path.splitext(file_name)[0]
                    if stem.isdigit():
                        levels.append(int(stem))
            if not levels:
                continue
            entries.append(
                MoEModuleEntry(
                    name=name,
                    layer_idx=parsed.layer_idx,
                    expert_idx=parsed.expert_idx,
                    levels=sorted(levels),
                )
            )
        return sorted(entries, key=lambda item: (item.layer_idx, item.expert_idx))

    @property
    def cols_removed_mlp(self) -> List[int]:
        return [int(x) for x in self.metadata.get("cols_removed_mlp", [])]

    @property
    def num_layers(self) -> int:
        if not self.entries:
            return 0
        return max(item.layer_idx for item in self.entries) + 1

    def num_experts_for_layer(self, layer_idx: int) -> int:
        return sum(1 for item in self.entries if item.layer_idx == layer_idx)

    def by_layer(self) -> Dict[int, List[MoEModuleEntry]]:
        out: Dict[int, List[MoEModuleEntry]] = {}
        for entry in self.entries:
            out.setdefault(entry.layer_idx, []).append(entry)
        return out

    def level_path(self, module_name: str, level: int) -> str:
        return os.path.join(self.db_path, module_name, f"{int(level)}.pth")

    def load_weight(self, module_name: str, level: int) -> torch.Tensor:
        path = self.level_path(module_name, level)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing DB slice: {path}")
        weight = torch.load(path, map_location="cpu")
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"Expected tensor in {path}, got {type(weight)}")
        return weight

    def get_policy_level_bounds(self) -> Dict[str, Any]:
        layers: Dict[str, Any] = {}
        for layer_idx, entries in self.by_layer().items():
            expert_bounds = {}
            for entry in entries:
                expert_bounds[str(entry.expert_idx)] = {
                    "min_level": int(min(entry.levels)),
                    "max_level": int(max(entry.levels)),
                }
            layers[str(layer_idx)] = {
                "num_experts": len(entries),
                "experts": expert_bounds,
            }
        return {
            "policy_type": "moe_expert_level_bounds",
            "num_layers": int(self.num_layers),
            "cols_removed_mlp": [int(x) for x in self.cols_removed_mlp],
            "layers": layers,
        }
