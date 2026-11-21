#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info
from transformers import AutoTokenizer


DEFAULT_FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_FINEWEB_CONFIG = "sample-10BT"


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


def load_tokenizer_with_fallback(name_or_path: str, local_files_only: bool = False):
    attempts = [
        {"use_fast": False, "trust_remote_code": False, "local_files_only": local_files_only},
        {"use_fast": True, "trust_remote_code": False, "local_files_only": local_files_only},
        {"use_fast": False, "trust_remote_code": True, "local_files_only": local_files_only},
        {"use_fast": True, "trust_remote_code": True, "local_files_only": local_files_only},
    ]
    errors: List[str] = []
    for kwargs in attempts:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                elif tokenizer.unk_token_id is not None:
                    tokenizer.pad_token = tokenizer.unk_token
            return tokenizer
        except Exception as exc:
            errors.append(f"{kwargs}: {exc}")
    raise RuntimeError(
        "Failed to load tokenizer with all fallback strategies for "
        f"{name_or_path}. Errors:\n" + "\n".join(errors)
    )


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def write_json(path: str | os.PathLike[str], payload: Dict[str, Any]):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def format_token_count(num_tokens: int) -> str:
    if num_tokens >= 1_000_000_000:
        return f"{num_tokens / 1_000_000_000:.3f}B"
    if num_tokens >= 1_000_000:
        return f"{num_tokens / 1_000_000:.3f}M"
    if num_tokens >= 1_000:
        return f"{num_tokens / 1_000:.3f}K"
    return str(num_tokens)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    return torch.distributed.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return torch.distributed.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if is_distributed():
        torch.distributed.barrier()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def maybe_all_reduce_mean(value: float) -> float:
    if not is_distributed():
        return float(value)
    reduce_device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=reduce_device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor /= get_world_size()
    return float(tensor.item())


def init_distributed_if_needed() -> Tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not torch.distributed.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.distributed.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return get_rank(), get_world_size(), device
    if torch.cuda.is_available():
        return 0, 1, torch.device("cuda")
    return 0, 1, torch.device("cpu")


def infer_default_dataset_config(dataset_name: str, dataset_config: Optional[str]) -> Optional[str]:
    if dataset_config is not None:
        return dataset_config
    if dataset_name in {"fineweb_edu", DEFAULT_FINEWEB_DATASET}:
        return DEFAULT_FINEWEB_CONFIG
    return None


def iter_dataset_records(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    streaming: bool,
    seed: int,
    row_start: Optional[int] = None,
    row_end: Optional[int] = None,
    exclude_rows: Optional[Sequence[int]] = None,
) -> Iterable[Dict[str, Any]]:
    resolved_config = infer_default_dataset_config(dataset_name, dataset_config)
    load_name = DEFAULT_FINEWEB_DATASET if dataset_name == "fineweb_edu" else dataset_name
    dataset = load_dataset(
        load_name,
        resolved_config,
        split=split,
        streaming=streaming,
    )
    if row_start is not None or row_end is not None or exclude_rows:
        # Row numbers are those of the full split; a stream has none, so this needs the arrow dataset.
        if streaming:
            raise ValueError("row_start, row_end and exclude_rows require streaming=False")
        keep = np.arange(0 if row_start is None else row_start, dataset.num_rows if row_end is None else row_end)
        if exclude_rows:
            keep = np.setdiff1d(keep, np.asarray(exclude_rows, dtype=np.int64))
        return dataset.select(keep, keep_in_memory=True).shuffle(seed=seed, keep_in_memory=True)
    if streaming:
        if hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
        return dataset
    dataset = dataset.shuffle(seed=seed)
    return dataset


def _tokenize_text(
    tokenizer,
    text: str,
    add_eos_token: bool,
) -> List[int]:
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    if add_eos_token and tokenizer.eos_token_id is not None:
        token_ids = token_ids + [int(tokenizer.eos_token_id)]
    return token_ids


@dataclass
class PackedSequenceConfig:
    dataset_name: str
    dataset_config: Optional[str]
    split: str
    text_field: str
    seq_len: int
    seed: int
    streaming: bool = True
    add_eos_token_between_samples: bool = True
    row_start: Optional[int] = None
    row_end: Optional[int] = None
    exclude_rows_path: Optional[str] = None


def load_excluded_rows(path: Optional[str]) -> Optional[List[int]]:
    """Row numbers to skip, from a JSON object with a list under "rows" (autoresearch_20261010/code/ar_recovery_rows.py)."""
    if not path:
        return None
    return [int(row) for row in json.loads(Path(path).read_text(encoding="utf-8"))["rows"]]


class PackedTextIterableDataset(IterableDataset):
    def __init__(self, tokenizer, cfg: PackedSequenceConfig, rank: int = 0, world_size: int = 1):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        shard_mod = self.world_size * num_workers
        shard_idx = self.rank * num_workers + worker_id

        records = iter_dataset_records(
            dataset_name=self.cfg.dataset_name,
            dataset_config=self.cfg.dataset_config,
            split=self.cfg.split,
            streaming=self.cfg.streaming,
            seed=self.cfg.seed,
            row_start=self.cfg.row_start,
            row_end=self.cfg.row_end,
            exclude_rows=load_excluded_rows(self.cfg.exclude_rows_path),
        )

        buffer: List[int] = []
        for sample_idx, sample in enumerate(records):
            if shard_mod > 1 and (sample_idx % shard_mod) != shard_idx:
                continue
            text = sample.get(self.cfg.text_field)
            if not isinstance(text, str) or not text:
                continue
            token_ids = _tokenize_text(
                tokenizer=self.tokenizer,
                text=text,
                add_eos_token=self.cfg.add_eos_token_between_samples,
            )
            if not token_ids:
                continue
            buffer.extend(token_ids)
            while len(buffer) >= self.cfg.seq_len:
                current = buffer[: self.cfg.seq_len]
                buffer = buffer[self.cfg.seq_len :]
                yield {
                    "input_ids": torch.tensor(current, dtype=torch.long),
                    "labels": torch.tensor(current, dtype=torch.long),
                }


def materialize_packed_eval_data(
    tokenizer,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    text_field: str,
    seq_len: int,
    max_eval_tokens: int,
    seed: int,
    streaming: bool = True,
) -> List[torch.Tensor]:
    records = iter_dataset_records(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        streaming=streaming,
        seed=seed,
    )
    sequences: List[torch.Tensor] = []
    buffer: List[int] = []
    consumed_tokens = 0

    for sample in records:
        text = sample.get(text_field)
        if not isinstance(text, str) or not text:
            continue
        buffer.extend(_tokenize_text(tokenizer, text, add_eos_token=True))
        while len(buffer) >= seq_len and consumed_tokens < max_eval_tokens:
            current = buffer[:seq_len]
            buffer = buffer[seq_len:]
            sequences.append(torch.tensor(current, dtype=torch.long).unsqueeze(0))
            consumed_tokens += seq_len
            if consumed_tokens >= max_eval_tokens:
                return sequences
    return sequences


def causal_lm_collator(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
    }


def infer_model_is_zero_masked(model) -> bool:
    model = unwrap_model(model)
    target_names = (
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    )
    for name, param in model.named_parameters():
        if not name.endswith(target_names):
            continue
        if bool((param == 0).any().item()):
            return True
    return False


def load_pruning_masks(mask_path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(mask_path, map_location="cpu")
    if isinstance(payload, dict):
        if "masks" in payload and isinstance(payload["masks"], dict):
            payload = payload["masks"]
        elif "state_dict" in payload and isinstance(payload["state_dict"], dict):
            payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError("Pruning masks file must contain a dict of parameter_name -> mask tensor")
    masks: Dict[str, torch.Tensor] = {}
    for name, value in payload.items():
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        masks[str(name)] = value.to(dtype=torch.bool, device="cpu")
    return masks


def build_zero_masks_from_model(model) -> Dict[str, torch.Tensor]:
    model = unwrap_model(model)
    masks: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        zero_mask = param.detach().cpu().eq(0)
        if bool(zero_mask.any().item()):
            masks[name] = zero_mask
    return masks


def apply_pruning_masks_inplace(model, masks: Dict[str, torch.Tensor]):
    model = unwrap_model(model)
    param_map = dict(model.named_parameters())
    with torch.no_grad():
        for name, mask in masks.items():
            if name not in param_map:
                continue
            param = param_map[name]
            if tuple(mask.shape) != tuple(param.shape):
                raise ValueError(f"Mask shape mismatch for {name}: {tuple(mask.shape)} vs {tuple(param.shape)}")
            param.data.masked_fill_(mask.to(device=param.device), 0)


def mask_pruned_gradients(model, masks: Dict[str, torch.Tensor]):
    model = unwrap_model(model)
    param_map = dict(model.named_parameters())
    for name, mask in masks.items():
        if name not in param_map:
            continue
        param = param_map[name]
        if param.grad is None:
            continue
        if tuple(mask.shape) != tuple(param.shape):
            raise ValueError(f"Mask shape mismatch for {name}: {tuple(mask.shape)} vs {tuple(param.shape)}")
        param.grad.masked_fill_(mask.to(device=param.grad.device), 0)


def verify_masked_weights_are_zero(model, masks: Dict[str, torch.Tensor]) -> List[str]:
    model = unwrap_model(model)
    param_map = dict(model.named_parameters())
    problems: List[str] = []
    with torch.no_grad():
        for name, mask in masks.items():
            if name not in param_map:
                problems.append(f"Missing parameter for mask: {name}")
                continue
            param = param_map[name]
            if tuple(mask.shape) != tuple(param.shape):
                problems.append(f"Shape mismatch for mask {name}: {tuple(mask.shape)} vs {tuple(param.shape)}")
                continue
            masked_values = param.detach()[mask.to(device=param.device)]
            if masked_values.numel() > 0 and bool(torch.count_nonzero(masked_values).item()):
                problems.append(f"Masked weights are nonzero for {name}")
    return problems


def count_dense_parameters(model) -> int:
    model = unwrap_model(model)
    return sum(int(param.numel()) for param in model.parameters())


def maybe_read_effective_param_count_billion(model_dir: str | os.PathLike[str]) -> Optional[float]:
    model_path = Path(model_dir)
    candidate_files = [
        model_path / "compression_summary.json",
        model_path / "stitch_metrics.json",
        model_path.parent / "compression_summary.json",
        model_path.parent / "stitch_metrics.json",
    ]
    for candidate in candidate_files:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("effective_params_billion", "effective_param_count_billion"):
            value = payload.get(key)
            if value is not None:
                return float(value)
    return None


def extract_task_scores(payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    task_scores = payload.get("task_scores")
    if isinstance(task_scores, dict):
        return {str(k): (None if v is None else float(v)) for k, v in task_scores.items()}
    lm_eval = payload.get("lm_eval")
    if isinstance(lm_eval, dict) and isinstance(lm_eval.get("task_scores"), dict):
        return {
            str(k): (None if v is None else float(v))
            for k, v in lm_eval["task_scores"].items()
        }
    return {}


def average_task_scores(task_scores: Dict[str, Optional[float]]) -> Optional[float]:
    values = [float(v) for v in task_scores.values() if v is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def maybe_enable_tf32():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def get_wall_clock_seconds(start_time: float) -> float:
    return float(time.time() - start_time)
