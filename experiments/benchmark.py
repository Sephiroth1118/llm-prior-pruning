#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# load_compressed_weights is modified from EvoPress (https://github.com/IST-DASLab/EvoPress), lmeval.py, licensed under
# the Apache License 2.0. Changes: the level file is opened as UTF-8, and layer names and levels are parsed before loading.

import argparse
import gc
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root: evoprune, zipprune

from lm_eval import evaluator, utils
from lm_eval.tasks import TaskManager
from zipprune.stitch_utils import (
    load_base_model,
    normalize_policy,
    stitch_model,
    synchronize_zero_masks_without_shrink,
)
from evoprune.transformers_compat import patch_transformers_pruning_utils
from zipprune.parameter_accounting import PARAMETER_COUNT_VERSION, count_parameters
from zipprune.benchmark_protocol import artifact_signature, file_sha256, checkpoint_protocol_matches


REPO_ROOT = Path(__file__).resolve().parents[1]


DARWINLM_TASK_SPECS: List[Dict[str, object]] = [
    {"task": "sciq", "fewshot": 0, "batch_size": "4"},
    {"task": "piqa", "fewshot": 0, "batch_size": "4"},
    {"task": "winogrande", "fewshot": 5, "batch_size": "2"},
    {"task": "arc_easy", "fewshot": 0, "batch_size": "2"},
    {"task": "arc_challenge", "fewshot": 25, "batch_size": "1"},
    {"task": "hellaswag", "fewshot": 10, "batch_size": "1"},
    {"task": "boolq", "fewshot": 0, "batch_size": "2"},
    {"task": "mmlu", "fewshot": 5, "batch_size": "1"},
]


def parse_task_specs(tasks_arg: Optional[str]) -> List[Dict[str, object]]:
    if not tasks_arg:
        return list(DARWINLM_TASK_SPECS)

    requested = [task.strip() for task in tasks_arg.split(",") if task.strip()]
    if not requested:
        raise ValueError("--tasks was provided but no valid task names were found")

    spec_map = {str(spec["task"]): dict(spec) for spec in DARWINLM_TASK_SPECS}
    missing = [task for task in requested if task not in spec_map]
    if missing:
        raise ValueError(
            "Unsupported task(s) for DarwinLM compare script: "
            + ", ".join(missing)
        )

    return [spec_map[task] for task in requested]


def format_billions(value: int) -> str:
    return f"{value / 1_000_000_000:.4f}B"


def _handle_non_serializable(obj):
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def _transformers_major_version() -> int:
    try:
        return int(str(transformers.__version__).split(".", 1)[0])
    except Exception:
        return 0


def _lm_eval_dtype_arg_name() -> str:
    # model_args are parsed by lm-eval's HFLM, whose public parameter is
    # `dtype`. HFLM itself maps it to HF's `torch_dtype` (Transformers 4.x).
    # Passing torch_dtype here duplicates that keyword inside HFLM.
    return "dtype"


def parse_model_selection(models_arg: str) -> List[str]:
    aliases = {
        "ours": "compressed",
        "our": "compressed",
        "compressed": "compressed",
        "dense": "dense",
        "darwinlm": "darwinlm",
    }
    selected = []
    for item in models_arg.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(
                f"Unsupported benchmark model '{item}'. Use any of: dense,darwinlm,compressed."
            )
        canonical = aliases[key]
        if canonical not in selected:
            selected.append(canonical)
    if not selected:
        raise ValueError("--models must include at least one model")
    return selected


def add_darwinlm_to_path(darwinlm_root: str):
    if darwinlm_root and darwinlm_root not in sys.path:
        sys.path.insert(0, darwinlm_root)


def get_eval_data(
    darwinlm_root: str,
    dataset_name: str,
    tokens: int,
    sequence_length: int,
    tokenizer,
    max_samples: Optional[int] = None,
):
    add_darwinlm_to_path(darwinlm_root)
    from evoprune.data_utils import get_data

    eval_data = get_data(
        dataset_name,
        tokens,
        sequence_length,
        tokenizer,
        train=False,
    )
    if max_samples is not None:
        eval_data = eval_data[:max_samples]
    return eval_data


def load_tokenizer_with_fallback(name_or_path: str):
    errors = []
    attempts = [
        {"use_fast": False, "trust_remote_code": False},
        {"use_fast": True, "trust_remote_code": False},
        {"use_fast": False, "trust_remote_code": True},
        {"use_fast": True, "trust_remote_code": True},
    ]
    for kwargs in attempts:
        try:
            return AutoTokenizer.from_pretrained(name_or_path, **kwargs)
        except Exception as exc:
            errors.append(f"{kwargs}: {exc}")
    raise RuntimeError(
        "Failed to load tokenizer with all fallback strategies for "
        f"{name_or_path}. Errors:\n" + "\n".join(errors)
    )


def _resolve_model_input_device(model, requested_device: str = "cuda") -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None and hasattr(input_embeddings, "weight"):
        return input_embeddings.weight.device
    return next(model.parameters()).device


@torch.no_grad()
def compute_perplexity(model, input_ids_list: List[torch.Tensor], device: str):
    total_nll = 0.0
    total_tokens = 0
    input_device = _resolve_model_input_device(model, device)

    for input_ids in input_ids_list:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(input_device)

        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :].float()
        labels = input_ids[:, 1:]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="sum",
        )
        total_nll += loss.item()
        total_tokens += labels.numel()

    if total_tokens == 0:
        raise ValueError("PPL evaluation requires at least one next-token target")
    avg_nll = total_nll / total_tokens
    if not math.isfinite(avg_nll):
        raise ValueError("PPL evaluation produced non-finite loss")
    ppl = math.exp(avg_nll)
    return {
        "ppl": ppl,
        "num_samples": len(input_ids_list),
        "num_tokens": total_tokens,
    }


def get_parameter_number(model, metric="total"):
    return count_parameters(model, metric=metric)


def write_sparse_config_text(policy_json_path: str) -> str:
    policy_obj = json.loads(Path(policy_json_path).read_text(encoding="utf-8"))
    normalized = normalize_policy(policy_obj)
    fd, output_path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    lines = [f"{layer_name}: {level}" for layer_name, level in sorted(normalized.items())]
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _read_checkpoint(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_checkpoint(path: Path, payload: Dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_handle_non_serializable),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _checkpoint_protocol_matches(checkpoint: Dict[str, object], protocol: Dict[str, object]) -> bool:
    return checkpoint_protocol_matches(checkpoint, protocol)


def _checkpoint_model_state(checkpoint: Dict[str, object], model_key: str) -> Dict[str, object]:
    models = checkpoint.setdefault("models", {})
    if not isinstance(models, dict):
        checkpoint["models"] = {}
        models = checkpoint["models"]
    state = models.setdefault(model_key, {})
    if not isinstance(state, dict):
        models[model_key] = {}
        state = models[model_key]
    return state


def _checkpoint_has_all_tasks(
    state: Dict[str, object],
    task_order: List[str],
) -> bool:
    task_scores = state.get("task_scores")
    task_details = state.get("task_details")
    if not isinstance(task_scores, dict) or not isinstance(task_details, dict):
        return False
    for task in task_order:
        if task not in task_scores or task not in task_details:
            return False
        score = task_scores.get(task)
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            return False
        detail = task_details.get(task)
        if not isinstance(detail, dict):
            return False
        if detail.get("status") not in {"ok", "fallback"}:
            return False
    return True


def release_model(model, device: str):
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def load_compressed_weights(
    model: AutoModelForCausalLM,
    compressed_weights_path: Union[str, os.PathLike],
    compressed_config_path: Optional[str] = None,
    default_level: int = 0,
):
    if compressed_config_path:
        with open(compressed_config_path, "r", encoding="utf-8") as handle:
            for line in handle:
                layer_name, level = line.split(":")
                layer_name = layer_name.strip()
                level = int(level.strip())
                layer = model.get_submodule(layer_name)
                orig_dtype = layer.weight.dtype
                layer.weight.data = torch.load(
                    os.path.join(compressed_weights_path, layer_name, f"{level}.pth"),
                    map_location=layer.weight.device,
                ).to(orig_dtype)
    else:
        for layer_name in sorted(os.listdir(compressed_weights_path)):
            layer_dir = os.path.join(compressed_weights_path, layer_name)
            if not os.path.isdir(layer_dir):
                continue
            layer = model.get_submodule(layer_name.strip())
            orig_dtype = layer.weight.dtype
            layer.weight.data = torch.load(
                os.path.join(layer_dir, f"{default_level}.pth"),
                map_location=layer.weight.device,
            ).to(orig_dtype)
    return model


def run_lm_eval(
    model_name_or_path: str,
    tasks: List[str],
    batch_size: str,
    num_fewshot: int,
    device: str,
    dtype: str,
    tokenizer_name: Optional[str] = None,
    sparse_weights_path: Optional[str] = None,
    sparse_config_path: Optional[str] = None,
    sync_zero_masks: bool = True,
    parallelize: Optional[bool] = None,
    max_memory_per_gpu: Optional[str] = None,
    low_memory_mode: bool = False,
) -> Dict[str, float]:
    patch_transformers_pruning_utils()
    if parallelize is None:
        parallelize = False
    eval_device = "cuda" if device == "auto" else device

    dtype_for_eval = "bfloat16" if dtype == "bfloat16" else ("float16" if dtype == "auto" else dtype)
    model_args_parts = [
        f"pretrained={model_name_or_path}",
        "low_cpu_mem_usage=True",
        f"{_lm_eval_dtype_arg_name()}={dtype_for_eval}",
        "trust_remote_code=True",
    ]
    if tokenizer_name:
        model_args_parts.append(f"tokenizer={tokenizer_name}")
    if parallelize:
        model_args_parts.append("parallelize=True")
        if max_memory_per_gpu:
            model_args_parts.append(f"max_memory_per_gpu={max_memory_per_gpu}")
    model_args = ",".join(model_args_parts)

    sparse_config_for_eval = None
    if sparse_config_path:
        sparse_config_for_eval = write_sparse_config_text(sparse_config_path)
    from_pretrained_orig = AutoModelForCausalLM.from_pretrained

    def from_pretrained_compat(*model_init_args, **model_init_kwargs):
        pretrained = model_init_args[0] if model_init_args else model_init_kwargs.get("pretrained_model_name_or_path", "")
        if (Path(str(pretrained)) / "compact_manifest.json").is_file():
            if parallelize:
                raise ValueError("ZipLM compact evaluation requires a single device; omit --lm_eval_parallelize")
            from zipprune.ZipLM.compact import load_compact_model
            requested_dtype = model_init_kwargs.get("torch_dtype", model_init_kwargs.get("dtype"))
            if isinstance(requested_dtype, str):
                requested_dtype = None if requested_dtype == "auto" else getattr(torch, requested_dtype)
            return load_compact_model(str(pretrained), device="cpu", dtype=requested_dtype)
        if _transformers_major_version() < 5 and "dtype" in model_init_kwargs:
            dtype_value = model_init_kwargs.pop("dtype")
            if "torch_dtype" not in model_init_kwargs and dtype_value is not None:
                model_init_kwargs["torch_dtype"] = dtype_value
        return from_pretrained_orig(*model_init_args, **model_init_kwargs)

    if sparse_weights_path:
        def from_pretrained_overridden(*model_init_args, **model_init_kwargs):
            model = from_pretrained_compat(*model_init_args, **model_init_kwargs)
            model = load_compressed_weights(
                model,
                sparse_weights_path,
                sparse_config_for_eval,
                default_level=0,
            )
            if sync_zero_masks:
                model, _ = synchronize_zero_masks_without_shrink(model)
            return model

        AutoModelForCausalLM.from_pretrained = staticmethod(from_pretrained_overridden)
    elif _transformers_major_version() < 5 or (Path(model_name_or_path) / "compact_manifest.json").is_file():
        AutoModelForCausalLM.from_pretrained = staticmethod(from_pretrained_compat)

    try:
        if hasattr(utils, "setup_logging"):
            utils.setup_logging("INFO")
        eval_logger = logging.getLogger("lm_eval")
        eval_logger.setLevel(logging.INFO)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        task_manager = TaskManager("INFO")
        task_names = task_manager.match_tasks(tasks)
        missing = [task for task in tasks if task not in task_names and "*" not in task]
        if missing:
            raise ValueError(f"Tasks not found: {', '.join(missing)}")

        results = evaluator.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=task_names,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            max_batch_size=1 if low_memory_mode else None,
            device=eval_device,
            use_cache=None,
            limit=None,
            task_manager=task_manager,
            check_integrity=False,
            write_out=False,
            log_samples=False,
            gen_kwargs=None,
        )
        if results is None:
            raise RuntimeError("lm-eval returned no results")
    finally:
        AutoModelForCausalLM.from_pretrained = from_pretrained_orig
        if sparse_config_for_eval:
            Path(sparse_config_for_eval).unlink(missing_ok=True)
        if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    task_scores = {}
    for task in tasks:
        metrics = results["results"][task]
        selected_key = None
        for prefix in ("acc_norm", "acc", "exact_match", "f1"):
            for metric_key in metrics.keys():
                if metric_key.startswith(prefix):
                    selected_key = metric_key
                    break
            if selected_key is not None:
                break
        if selected_key is None:
            raise ValueError(f"Could not find a supported metric for task '{task}': {list(metrics.keys())}")
        task_scores[task] = float(metrics[selected_key])
    return task_scores


def _run_lm_eval_worker(result_queue, kwargs):
    try:
        scores = run_lm_eval(**kwargs)
        result_queue.put({"ok": True, "scores": scores})
    except Exception as exc:  # pragma: no cover
        result_queue.put({"ok": False, "error": repr(exc), "message": str(exc)})


def run_lm_eval_isolated(**kwargs) -> Dict[str, float]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_run_lm_eval_worker, args=(result_queue, kwargs))
    process.start()
    process.join()

    if process.exitcode != 0:
        raise RuntimeError(f"lm-eval worker exited with code {process.exitcode}")

    try:
        payload = result_queue.get_nowait()
    except queue.Empty as exc:
        raise RuntimeError("lm-eval worker exited without returning results") from exc

    if not payload.get("ok"):
        raise RuntimeError(payload.get("message") or payload.get("error") or "lm-eval worker failed")

    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise RuntimeError("lm-eval worker returned an invalid result payload")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    return scores


def run_ppl(
    model,
    model_name_or_path: str,
    dataset_name: str,
    eval_tokens: int,
    eval_sequence_length: int,
    tokenizer_name: Optional[str],
    max_samples: Optional[int],
    device: str,
) -> float:
    tokenizer = load_tokenizer_with_fallback(tokenizer_name or model_name_or_path)
    eval_data = get_eval_data(
        darwinlm_root=str(REPO_ROOT),
        dataset_name=dataset_name,
        tokens=eval_tokens,
        sequence_length=eval_sequence_length,
        tokenizer=tokenizer,
        max_samples=max_samples,
    )
    result = compute_perplexity(model=model, input_ids_list=eval_data, device=device)
    return float(result["ppl"])


def run_task_suite(
    model_name_or_path: str,
    task_specs: List[Dict[str, object]],
    device: str,
    dtype: str,
    tokenizer_name: Optional[str] = None,
    sparse_weights_path: Optional[str] = None,
    sparse_config_path: Optional[str] = None,
    sync_zero_masks: bool = True,
    parallelize: Optional[bool] = None,
    max_memory_per_gpu: Optional[str] = None,
    strict: bool = False,
    allow_logiqa2_fallback: bool = False,
    low_memory_mode: bool = False,
    resume_scores: Optional[Dict[str, Optional[float]]] = None,
    resume_details: Optional[Dict[str, Dict[str, object]]] = None,
    on_task_complete=None,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Dict[str, object]]]:
    scores: Dict[str, Optional[float]] = dict(resume_scores or {})
    details: Dict[str, Dict[str, object]] = dict(resume_details or {})
    for spec in task_specs:
        task = str(spec["task"])
        fewshot = int(spec["fewshot"])
        batch_size = "1" if low_memory_mode else str(spec["batch_size"])
        existing_detail = details.get(task)
        if (
            task in scores
            and isinstance(existing_detail, dict)
            and existing_detail.get("status") in {"ok", "fallback"}
        ):
            print(f"[resume] Skipping task '{task}' from checkpoint.")
            continue
        try:
            actual_task = task
            run_eval = run_lm_eval_isolated if low_memory_mode else run_lm_eval
            task_scores = run_eval(
                model_name_or_path=model_name_or_path,
                tasks=[actual_task],
                batch_size=batch_size,
                num_fewshot=fewshot,
                device=device,
                dtype=dtype,
                tokenizer_name=tokenizer_name,
                sparse_weights_path=sparse_weights_path,
                sparse_config_path=sparse_config_path,
                sync_zero_masks=sync_zero_masks,
                parallelize=parallelize,
                max_memory_per_gpu=max_memory_per_gpu,
                low_memory_mode=low_memory_mode,
            )
            metric_name = "selected_metric"
            metric_value = task_scores[actual_task]
            if not math.isfinite(float(metric_value)):
                raise ValueError(f"Non-finite score for task '{actual_task}'")
            scores[task] = float(metric_value)
            details[task] = {
                "fewshot": fewshot,
                "batch_size": batch_size,
                "actual_task": actual_task,
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "status": "ok",
            }
            if on_task_complete is not None:
                on_task_complete(scores, details)
        except Exception as exc:
            if (
                allow_logiqa2_fallback
                and task == "logiqa"
                and "Dataset scripts are no longer supported" in str(exc)
            ):
                fallback_task = "logiqa2"
                print(f"[WARNING] Task 'logiqa' is incompatible with current datasets version; falling back to '{fallback_task}'.")
                run_eval = run_lm_eval_isolated if low_memory_mode else run_lm_eval
                task_scores = run_eval(
                    model_name_or_path=model_name_or_path,
                    tasks=[fallback_task],
                    batch_size=batch_size,
                    num_fewshot=fewshot,
                    device=device,
                    dtype=dtype,
                    tokenizer_name=tokenizer_name,
                    sparse_weights_path=sparse_weights_path,
                    sparse_config_path=sparse_config_path,
                    sync_zero_masks=sync_zero_masks,
                    parallelize=parallelize,
                    max_memory_per_gpu=max_memory_per_gpu,
                    low_memory_mode=low_memory_mode,
                )
                metric_name = "selected_metric"
                metric_value = task_scores[fallback_task]
                if not math.isfinite(float(metric_value)):
                    raise ValueError(f"Non-finite score for task '{fallback_task}'")
                scores[task] = float(metric_value)
                details[task] = {
                    "fewshot": fewshot,
                    "batch_size": batch_size,
                    "actual_task": fallback_task,
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                    "status": "fallback",
                    "note": "Used logiqa2 because lm-eval logiqa is incompatible with current datasets version.",
                }
                if on_task_complete is not None:
                    on_task_complete(scores, details)
                continue
            if strict:
                raise
            scores[task] = None
            details[task] = {
                "fewshot": fewshot,
                "batch_size": batch_size,
                "actual_task": task,
                "metric_name": None,
                "metric_value": None,
                "status": "error",
                "error": str(exc),
            }
            if on_task_complete is not None:
                on_task_complete(scores, details)
            print(f"[WARNING] Task '{task}' failed: {exc}")
    return scores, details


def build_row(
    model_name: str,
    effective_params: int,
    dense_effective_params: Optional[int],
    ppl: float,
    task_scores: Dict[str, Optional[float]],
    tasks: List[str],
    parameter_metric: str = "unspecified",
) -> Dict[str, object]:
    valid_scores = [float(task_scores[task]) for task in tasks
                    if task_scores.get(task) is not None and math.isfinite(float(task_scores[task]))]
    avg = sum(valid_scores) / len(tasks) if tasks and len(valid_scores) == len(tasks) else None
    param_ratio = "" if dense_effective_params is None else f"{effective_params / dense_effective_params:.4f}"
    row = {
        "Model": model_name,
        "Params": format_billions(effective_params),
        "Param Metric": parameter_metric,
        "Param Ratio": param_ratio,
        "PPL": f"{ppl:.4f}",
    }
    for task in tasks:
        value = task_scores.get(task)
        row[task] = "" if value is None else f"{value:.4f}"
    row["Avg"] = "" if avg is None else f"{avg:.4f}"
    row["Tasks Complete"] = f"{len(valid_scores)}/{len(tasks)}"
    return row


def render_markdown_table(rows: List[Dict[str, object]], task_order: List[str]) -> str:
    headers = ["Model", "Params", "Param Metric", "Param Ratio", "PPL", *task_order, "Avg", "Tasks Complete"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _run_or_resume_benchmark_model(
    model_key: str,
    display_name: str,
    model_name_or_path: str,
    args,
    checkpoint: Dict[str, object],
    checkpoint_path: Path,
    task_specs: List[Dict[str, object]],
    task_order: List[str],
    sparse_weights_path: Optional[str] = None,
    sparse_config_path: Optional[str] = None,
    sync_zero_masks: bool = True,
):
    state = _checkpoint_model_state(checkpoint, model_key)
    state["display_name"] = display_name
    state["model_name_or_path"] = model_name_or_path

    if "counts" in state and "ppl" in state:
        print(f"[resume] Skipping {display_name} params/PPL from checkpoint.")
    else:
        if sparse_weights_path or sparse_config_path:
            compressed = stitch_model(
                model_name_or_path=model_name_or_path,
                db_path=sparse_weights_path,
                config_path=sparse_config_path,
                dtype=args.dtype,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                attn_implementation=args.attn_implementation,
                device=args.device,
                sync_zero_masks=sync_zero_masks,
            )
            model = compressed.model
        else:
            compressed = None
            model = load_base_model(
                model_name_or_path=model_name_or_path,
                dtype=args.dtype,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                attn_implementation=args.attn_implementation,
                device=args.device,
            )
        try:
            metric = args.parameter_metric
            if metric == "auto":
                metric = "structured" if compressed is not None else "total"
            state["counts"] = get_parameter_number(model, metric=metric)
            _write_checkpoint(checkpoint_path, checkpoint)
            state["ppl"] = run_ppl(
                model=model,
                model_name_or_path=model_name_or_path,
                dataset_name=args.ppl_dataset,
                eval_tokens=args.ppl_tokens,
                eval_sequence_length=args.ppl_sequence_length,
                tokenizer_name=args.tokenizer_name,
                max_samples=args.ppl_max_samples,
                device=args.device,
            )
            _write_checkpoint(checkpoint_path, checkpoint)
        finally:
            release_model(model, args.device)
            model = None
            if compressed is not None:
                del compressed
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()

    if _checkpoint_has_all_tasks(state, task_order):
        print(f"[resume] Skipping {display_name} task suite from checkpoint.")
    else:
        def on_task_complete(scores, details):
            state["task_scores"] = scores
            state["task_details"] = details
            _write_checkpoint(checkpoint_path, checkpoint)

        scores, details = run_task_suite(
            model_name_or_path=model_name_or_path,
            task_specs=task_specs,
            device=args.device,
            dtype=args.dtype,
            tokenizer_name=args.tokenizer_name,
            sparse_weights_path=sparse_weights_path,
            sparse_config_path=sparse_config_path,
            sync_zero_masks=sync_zero_masks,
            parallelize=args.lm_eval_parallelize,
            max_memory_per_gpu=args.lm_eval_max_memory_per_gpu,
            strict=args.strict_tasks,
            allow_logiqa2_fallback=args.allow_logiqa2_fallback,
            low_memory_mode=args.low_memory_mode,
            resume_scores=state.get("task_scores") if isinstance(state.get("task_scores"), dict) else None,
            resume_details=state.get("task_details") if isinstance(state.get("task_details"), dict) else None,
            on_task_complete=on_task_complete,
        )
        state["task_scores"] = scores
        state["task_details"] = details
        _write_checkpoint(checkpoint_path, checkpoint)

    return state


def main():
    patch_transformers_pruning_utils()
    parser = argparse.ArgumentParser(
        description="Run a DarwinLM-style Dense vs Compressed benchmark comparison."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--darwinlm_model_name_or_path", type=str, default=None)
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--compact_model_path", type=str, default=None,
                        help="Evaluate this physically compact ZipLM checkpoint for the compressed row.")
    parser.add_argument("--output_json", type=str, default="outputs/benchmark/results.json")
    parser.add_argument("--output_table", type=str, default="outputs/benchmark/table.md")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated subset of DarwinLM tasks to run.")
    parser.add_argument(
        "--models",
        type=str,
        default="dense,darwinlm,compressed",
        help="Comma-separated model rows to run: dense,darwinlm,compressed. Aliases: ours=compressed.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sync_zero_masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--parameter_metric", choices=["auto", "total", "nonzero", "structured"], default="auto",
        help="auto: structural counts for DB policies, tensor counts for loaded checkpoints. "
             "Use nonzero for 2:4, or structured for exported masked checkpoints (including MoE).",
    )
    parser.add_argument("--ppl_dataset", type=str, default="fineweb_edu")
    parser.add_argument("--ppl_tokens", type=int, default=163840)
    parser.add_argument("--ppl_sequence_length", type=int, default=4096)
    parser.add_argument("--ppl_max_samples", type=int, default=40)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lm_eval_parallelize", action="store_true")
    parser.add_argument("--lm_eval_max_memory_per_gpu", type=str, default=None)
    parser.add_argument("--low_memory_mode", action="store_true")
    parser.add_argument("--allow_logiqa2_fallback", action="store_true")
    parser.add_argument("--hf_hub_download_timeout", type=int, default=300)
    parser.add_argument("--hf_hub_etag_timeout", type=int, default=300)
    parser.add_argument("--strict_tasks", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint_json", type=str, default=None)
    args = parser.parse_args()

    if args.compact_model_path:
        if not (Path(args.compact_model_path) / "compact_manifest.json").is_file():
            raise ValueError("--compact_model_path must contain a ZipLM compact_manifest.json")
        if args.device == "auto" or args.lm_eval_parallelize:
            raise ValueError("ZipLM compact checkpoints currently require an explicit single device")
        args.tokenizer_name = args.tokenizer_name or args.model_name_or_path

    output_json = Path(args.output_json)
    output_table = Path(args.output_table)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_table.parent.mkdir(parents=True, exist_ok=True)

    task_specs = parse_task_specs(args.tasks)
    task_order = [str(item["task"]) for item in task_specs]
    selected_models = parse_model_selection(args.models)
    if "darwinlm" in selected_models and not args.darwinlm_model_name_or_path:
        raise ValueError("--darwinlm_model_name_or_path is required when --models includes darwinlm")
    checkpoint_path = Path(args.checkpoint_json) if args.checkpoint_json else output_json.with_suffix(".checkpoint.json")
    checkpoint: Dict[str, object] = _read_checkpoint(checkpoint_path) if args.resume else {}
    db_files = None
    if "compressed" in selected_models and not args.compact_model_path:
        policy = normalize_policy(json.loads(Path(args.config_path).read_text(encoding="utf-8")))
        db_files = [f"{module}/{level}.pth" for module, level in policy.items()]
    protocol = {
        "name": "darwinlm_style_compare",
        "parameter_count_version": PARAMETER_COUNT_VERSION,
        "parameter_metric": args.parameter_metric,
        "sync_zero_masks": args.sync_zero_masks,
        "policy_sha256": file_sha256(args.config_path),
        "db_manifest_sha256": artifact_signature(args.db_path, db_files) if db_files is not None else None,
        "model_manifest_sha256": artifact_signature(args.model_name_or_path),
        "compact_model_path": args.compact_model_path,
        "compact_manifest_sha256": artifact_signature(args.compact_model_path),
        "darwinlm_manifest_sha256": artifact_signature(args.darwinlm_model_name_or_path),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "tokenizer_name": args.tokenizer_name,
        "device": args.device,
        "lm_eval_max_memory_per_gpu": args.lm_eval_max_memory_per_gpu,
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
        "strict_tasks": args.strict_tasks,
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "model_name_or_path": args.model_name_or_path,
        "darwinlm_model_name_or_path": args.darwinlm_model_name_or_path,
        "models": selected_models,
        "db_path": args.db_path,
        "config_path": args.config_path,
        "ppl_dataset": args.ppl_dataset,
        "ppl_tokens": args.ppl_tokens,
        "ppl_sequence_length": args.ppl_sequence_length,
        "ppl_max_samples": args.ppl_max_samples,
        "task_specs": task_specs,
        "allow_logiqa2_fallback": args.allow_logiqa2_fallback,
        "lm_eval_parallelize": args.lm_eval_parallelize,
        "low_memory_mode": args.low_memory_mode,
    }
    if args.resume and not _checkpoint_protocol_matches(checkpoint, protocol):
        print(f"[resume] Ignoring checkpoint with mismatched protocol: {checkpoint_path}")
        checkpoint = {}
    checkpoint["protocol"] = protocol
    _write_checkpoint(checkpoint_path, checkpoint)

    dense_state = None
    if "dense" in selected_models:
        dense_state = _run_or_resume_benchmark_model(
            model_key="dense",
            display_name="Dense",
            model_name_or_path=args.model_name_or_path,
            args=args,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            task_specs=task_specs,
            task_order=task_order,
        )

    darwinlm_state = None
    if "darwinlm" in selected_models:
        darwinlm_state = _run_or_resume_benchmark_model(
            model_key="darwinlm",
            display_name="DarwinLM",
            model_name_or_path=args.darwinlm_model_name_or_path,
            args=args,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            task_specs=task_specs,
            task_order=task_order,
        )

    compressed_state = None
    if "compressed" in selected_models:
        compressed_state = _run_or_resume_benchmark_model(
            model_key="compressed",
            display_name="Compressed",
            model_name_or_path=args.compact_model_path or args.model_name_or_path,
            args=args,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            task_specs=task_specs,
            task_order=task_order,
            sparse_weights_path=None if args.compact_model_path else args.db_path,
            sparse_config_path=None if args.compact_model_path else args.config_path,
            sync_zero_masks=args.sync_zero_masks,
        )

    dense_effective_params = (
        int(dense_state["counts"]["Reported"])
        if dense_state is not None and "counts" in dense_state
        else None
    )

    rows = []
    if dense_state is not None:
        rows.append(
            build_row(
                model_name="Dense",
                effective_params=int(dense_state["counts"]["Reported"]),
                dense_effective_params=dense_effective_params,
                ppl=float(dense_state["ppl"]),
                parameter_metric=dense_state["counts"]["ReportedMetric"],
                task_scores=dense_state["task_scores"],
                tasks=task_order,
            )
        )
    if darwinlm_state is not None:
        rows.append(
            build_row(
                model_name="DarwinLM",
                effective_params=int(darwinlm_state["counts"]["Reported"]),
                dense_effective_params=dense_effective_params,
                ppl=float(darwinlm_state["ppl"]),
                parameter_metric=darwinlm_state["counts"]["ReportedMetric"],
                task_scores=darwinlm_state["task_scores"],
                tasks=task_order,
            )
        )
    if compressed_state is not None:
        rows.append(
            build_row(
                model_name="Compressed",
                effective_params=int(compressed_state["counts"]["Reported"]),
                dense_effective_params=dense_effective_params,
                ppl=float(compressed_state["ppl"]),
                parameter_metric=compressed_state["counts"]["ReportedMetric"],
                task_scores=compressed_state["task_scores"],
                tasks=task_order,
            )
        )

    output_payload = {
        "protocol": protocol,
        "dense": None if dense_state is None else {
            "counts": dense_state["counts"],
            "ppl": dense_state["ppl"],
            "task_scores": dense_state["task_scores"],
            "task_details": dense_state["task_details"],
        },
        "checkpoint_json": str(checkpoint_path),
        "darwinlm": None if darwinlm_state is None else {
            "model_name_or_path": args.darwinlm_model_name_or_path,
            "counts": darwinlm_state["counts"],
            "ppl": darwinlm_state["ppl"],
            "task_scores": darwinlm_state["task_scores"],
            "task_details": darwinlm_state["task_details"],
        },
        "compressed": None if compressed_state is None else {
            "counts": compressed_state["counts"],
            "ppl": compressed_state["ppl"],
            "task_scores": compressed_state["task_scores"],
            "task_details": compressed_state["task_details"],
        },
        "rows": rows,
        "markdown_table": render_markdown_table(rows, task_order),
    }
    output_json.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False, default=_handle_non_serializable),
        encoding="utf-8",
    )
    output_table.write_text(output_payload["markdown_table"] + "\n", encoding="utf-8")

    print(output_payload["markdown_table"])
    print(f"\nWrote JSON results to {output_json}")
    print(f"Wrote Markdown table to {output_table}")


if __name__ == "__main__":
    main()
