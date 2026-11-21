#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class Prune2to4Config:
    model_name_or_path: str
    output_dir: str
    dtype: str = "bfloat16"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    low_cpu_mem_usage: bool = False
    keep_pattern_n: int = 2
    keep_pattern_m: int = 4
    target_suffixes: Tuple[str, ...] = DEFAULT_TARGET_SUFFIXES


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_model_and_tokenizer(
    model_name_or_path: str,
    dtype: str = "bfloat16",
    device: str = "cuda",
    attn_implementation: str = "sdpa",
    low_cpu_mem_usage: bool = False,
):
    torch_dtype = _resolve_dtype(dtype)
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
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    return model, tokenizer


def collect_target_linear_modules(
    model: nn.Module,
    target_suffixes: Iterable[str] = DEFAULT_TARGET_SUFFIXES,
) -> List[Tuple[str, nn.Linear]]:
    suffixes = tuple(target_suffixes)
    selected: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith(suffixes):
            selected.append((name, module))
    return selected


def _build_2to4_mask(weight: torch.Tensor, n: int = 2, m: int = 4) -> torch.Tensor:
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight for 2:{m} pruning, got shape {tuple(weight.shape)}")
    if m <= 0 or n <= 0 or n > m:
        raise ValueError(f"Invalid n:m pattern: {n}:{m}")

    device = weight.device
    abs_weight = weight.abs()
    rows, cols = abs_weight.shape
    pad_cols = (m - (cols % m)) % m
    if pad_cols > 0:
        abs_weight = torch.nn.functional.pad(abs_weight, (0, pad_cols))
    grouped = abs_weight.view(rows, -1, m)
    keep_idx = torch.topk(grouped, k=n, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(grouped, dtype=torch.bool, device=device)
    mask.scatter_(-1, keep_idx, True)
    mask = mask.view(rows, -1)
    if pad_cols > 0:
        mask = mask[:, :cols]
    return mask


def apply_2to4_pruning_(
    model: nn.Module,
    target_suffixes: Iterable[str] = DEFAULT_TARGET_SUFFIXES,
    n: int = 2,
    m: int = 4,
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    targets = collect_target_linear_modules(model, target_suffixes=target_suffixes)
    print(f"[2to4] pruning {len(targets)} linear modules with pattern {n}:{m}")
    total_params = 0
    total_nonzero_before = 0
    total_nonzero_after = 0

    for name, module in targets:
        with torch.no_grad():
            weight = module.weight.data
            mask = _build_2to4_mask(weight, n=n, m=m)
            nonzero_before = int(torch.count_nonzero(weight).item())
            weight.mul_(mask.to(weight.dtype))
            nonzero_after = int(torch.count_nonzero(weight).item())
            params = int(weight.numel())
        total_params += params
        total_nonzero_before += nonzero_before
        total_nonzero_after += nonzero_after
        density_after = float(nonzero_after) / float(params) if params > 0 else 0.0
        summary[name] = {
            "params": float(params),
            "nonzero_before": float(nonzero_before),
            "nonzero_after": float(nonzero_after),
            "density_after": density_after,
        }
        print(
            f"[2to4] {name}: nonzero {nonzero_before} -> {nonzero_after}, "
            f"density_after={density_after:.4f}"
        )

    overall_density_after = (
        float(total_nonzero_after) / float(total_params) if total_params > 0 else 0.0
    )
    print(
        f"[2to4] done. target params={total_params}, "
        f"nonzero_before={total_nonzero_before}, nonzero_after={total_nonzero_after}, "
        f"density_after={overall_density_after:.4f}"
    )
    summary["_overall"] = {
        "params": float(total_params),
        "nonzero_before": float(total_nonzero_before),
        "nonzero_after": float(total_nonzero_after),
        "density_after": overall_density_after,
    }
    return summary


def save_model_and_tokenizer(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    summary: Dict[str, Dict[str, float]],
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    import json

    (out / "prune_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

