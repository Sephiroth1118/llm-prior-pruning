#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, get_scheduler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zipprune.recovery_utils import (
    PackedSequenceConfig,
    PackedTextIterableDataset,
    apply_pruning_masks_inplace,
    barrier,
    build_zero_masks_from_model,
    causal_lm_collator,
    count_dense_parameters,
    format_token_count,
    get_wall_clock_seconds,
    infer_default_dataset_config,
    infer_model_is_zero_masked,
    init_distributed_if_needed,
    is_main_process,
    load_pruning_masks,
    load_tokenizer_with_fallback,
    mask_pruned_gradients,
    maybe_all_reduce_mean,
    maybe_enable_tf32,
    maybe_read_effective_param_count_billion,
    resolve_torch_dtype,
    set_random_seed,
    unwrap_model,
    verify_masked_weights_are_zero,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-parameter post-compression recovery training.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_train_tokens", type=int, required=True)
    parser.add_argument("--debug_max_train_tokens", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flash_attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_every_tokens", type=int, default=50_000_000)
    parser.add_argument("--eval_every_tokens", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--micro_logging_steps", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--pruning_masks_path", type=str, default=None)
    parser.add_argument("--keep_pruned_weights_frozen", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_row_start", type=int, default=None, help="first training row of the split (needs --no-streaming)")
    parser.add_argument("--train_row_end", type=int, default=None, help="end of the training rows of the split (needs --no-streaming)")
    parser.add_argument("--exclude_rows_path", type=str, default=None, help="JSON with rows the training must not read (needs --no-streaming)")
    parser.add_argument("--compact_model_path", type=str, default=None,
                        help="train a ZipLM compact checkpoint; the pruned weights do not exist, so nothing is masked")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--deepspeed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--deepspeed_zero_stage", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--deepspeed_offload_optimizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deepspeed_offload_param", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deepspeed_save_16bit_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deepspeed_torch_adam", action="store_true")
    parser.add_argument("--empty_cache_every_steps", type=int, default=0)
    parser.add_argument("--gpu_memory_fraction", type=float, default=None,
                        help="cap on this process's share of its GPU, to leave room for other users of the card")
    return parser.parse_args()


def maybe_init_wandb(args: argparse.Namespace):
    if not args.wandb_project or not is_main_process():
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"[warning] wandb requested but unavailable: {exc}", flush=True)
        return None

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    return run


def choose_attn_implementation(flash_attention: bool) -> str:
    if flash_attention:
        return "flash_attention_2"
    return "sdpa"


def load_model(
    model_name_or_path: str,
    dtype: str,
    gradient_checkpointing: bool,
    flash_attention: bool,
    local_files_only: bool,
    low_cpu_mem_usage: bool,
):
    torch_dtype = resolve_torch_dtype(dtype)
    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "local_files_only": local_files_only,
    }
    if torch_dtype != "auto":
        load_kwargs["torch_dtype"] = torch_dtype

    attn_implementation = choose_attn_implementation(flash_attention)
    try:
        load_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    except Exception as exc:
        if flash_attention:
            print(f"[warning] flash_attention_2 unavailable, falling back to sdpa: {exc}", flush=True)
            load_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        else:
            raise

    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def load_compact_for_training(path: str, dtype: str, gradient_checkpointing: bool):
    from zipprune.ZipLM.compact import load_compact_model

    # SDPA only: the deployment backend of compact models (zipprune/ZipLM/compact.py).
    model = load_compact_model(path, device="cpu", dtype=resolve_torch_dtype(dtype))
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def restore_rope_frequencies(model, frequencies: Dict[str, torch.Tensor]):
    """DeepSpeed's bf16 conversion rounds these buffers; the deployed compact model keeps them in float32."""
    for name, buffer in unwrap_model(model).named_buffers():
        if name in frequencies:
            buffer.data = frequencies[name].to(buffer.device)


def save_compact_final(final_dir: Path, source: str, model, deepspeed_engine: bool):
    """final/compact_model: the source's config and layout with the trained weights, loadable by load_compact_model."""
    target = final_dir / "compact_model"
    if is_main_process():
        target.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "compact_manifest.json"):
            shutil.copy2(Path(source) / name, target / name)
    barrier()
    if deepspeed_engine:
        model.save_16bit_model(str(target), "compact_weights.pth")
    elif is_main_process():
        torch.save(unwrap_model(model).state_dict(), target / "compact_weights.pth")


def save_checkpoint(
    output_dir: str,
    model,
    tokenizer,
    optimizer,
    scheduler,
    training_state: Dict[str, Any],
):
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    unwrapped = unwrap_model(model)
    unwrapped.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "training_state": training_state,
        },
        save_path / "training_state.pt",
    )


def is_deepspeed_engine(model) -> bool:
    return hasattr(model, "save_checkpoint") and hasattr(model, "backward") and hasattr(model, "step")


def get_current_lr(optimizer, scheduler=None) -> float:
    if scheduler is not None:
        try:
            return float(scheduler.get_last_lr()[0])
        except Exception:
            pass
    if optimizer is not None and getattr(optimizer, "param_groups", None):
        return float(optimizer.param_groups[0].get("lr", 0.0))
    return 0.0


class ManualLRScheduler:
    def __init__(self, optimizer, scheduler_type: str, num_warmup_steps: int, num_training_steps: int):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.num_warmup_steps = max(int(num_warmup_steps), 0)
        self.num_training_steps = max(int(num_training_steps), 1)
        self.last_epoch = 0
        self.base_lrs = [float(group.get("lr", 0.0)) for group in self.optimizer.param_groups]
        self._last_lr = list(self.base_lrs)
        self._apply_lr()

    def _lr_scale(self) -> float:
        step = self.last_epoch
        if self.num_warmup_steps > 0 and step < self.num_warmup_steps:
            return float(step) / float(max(1, self.num_warmup_steps))

        if self.scheduler_type == "constant":
            return 1.0
        if self.scheduler_type == "constant_with_warmup":
            return 1.0

        decay_steps = max(1, self.num_training_steps - self.num_warmup_steps)
        progress = min(max((step - self.num_warmup_steps) / decay_steps, 0.0), 1.0)
        if self.scheduler_type == "linear":
            return max(0.0, 1.0 - progress)
        if self.scheduler_type == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        raise ValueError(f"Unsupported DeepSpeed manual lr_scheduler_type: {self.scheduler_type}")

    def _apply_lr(self):
        scale = self._lr_scale()
        self._last_lr = [base_lr * scale for base_lr in self.base_lrs]
        for group, lr in zip(self.optimizer.param_groups, self._last_lr):
            group["lr"] = lr

    def step(self):
        self.last_epoch += 1
        self._apply_lr()

    def get_last_lr(self):
        return list(self._last_lr)

    def state_dict(self):
        return {
            "scheduler_type": self.scheduler_type,
            "num_warmup_steps": self.num_warmup_steps,
            "num_training_steps": self.num_training_steps,
            "last_epoch": self.last_epoch,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = int(state_dict.get("last_epoch", self.last_epoch))
        self.base_lrs = [float(lr) for lr in state_dict.get("base_lrs", self.base_lrs)]
        self._apply_lr()


def build_deepspeed_config(args: argparse.Namespace, world_size: int, total_optimizer_steps: int) -> Dict[str, Any]:
    if args.deepspeed_config:
        with open(args.deepspeed_config, "r", encoding="utf-8") as f:
            return json.load(f)

    zero_config: Dict[str, Any] = {
        "stage": int(args.deepspeed_zero_stage),
        "overlap_comm": True,
        "contiguous_gradients": True,
    }
    if args.deepspeed_zero_stage == 3:
        zero_config.update(
            {
                "stage3_gather_16bit_weights_on_model_save": bool(args.deepspeed_save_16bit_model),
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
            }
        )
    if args.deepspeed_offload_optimizer:
        zero_config["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    if args.deepspeed_offload_param:
        zero_config["offload_param"] = {"device": "cpu", "pin_memory": True}

    config: Dict[str, Any] = {
        "train_micro_batch_size_per_gpu": int(args.per_device_train_batch_size),
        # This script keeps its existing manual gradient accumulation logic.
        "gradient_accumulation_steps": 1,
        "train_batch_size": int(args.per_device_train_batch_size * max(world_size, 1)),
        "gradient_clipping": float(args.max_grad_norm or 0.0),
        "zero_optimization": zero_config,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": float(args.learning_rate),
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": float(args.weight_decay),
            },
        },
        "steps_per_print": max(int(args.logging_steps), 1),
        "wall_clock_breakdown": False,
    }
    if args.deepspeed_torch_adam:
        config["optimizer"]["params"]["torch_adam"] = True  # torch.optim.AdamW; FusedAdam needs ninja to compile its C++ op
        if args.deepspeed_offload_optimizer:
            config["zero_force_ds_cpu_optimizer"] = False  # DeepSpeedCPUAdam needs ninja as well; AdamW then runs on the CPU
    if args.dtype == "bfloat16":
        config["bf16"] = {"enabled": True}
    elif args.dtype == "float16":
        config["fp16"] = {"enabled": True}
    return config


def initialize_deepspeed(model, args: argparse.Namespace, world_size: int, total_optimizer_steps: int):
    import deepspeed

    ds_config = build_deepspeed_config(args, world_size, total_optimizer_steps)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    return engine, optimizer, ds_config


def apply_pruning_masks_after_deepspeed_step(engine, masks: Dict[str, torch.Tensor]):
    try:
        import deepspeed
    except Exception:
        apply_pruning_masks_inplace(engine, masks)
        return

    param_map = dict(engine.module.named_parameters())
    with torch.no_grad():
        for name, mask in masks.items():
            param = param_map.get(name)
            if param is None:
                continue
            with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                if is_main_process():
                    if tuple(mask.shape) != tuple(param.shape):
                        raise ValueError(f"Mask shape mismatch for {name}: {tuple(mask.shape)} vs {tuple(param.shape)}")
                    param.data.masked_fill_(mask.to(device=param.device), 0)


def verify_deepspeed_masked_weights_are_zero(engine, masks: Dict[str, torch.Tensor]) -> List[str]:
    try:
        import deepspeed
    except Exception:
        return verify_masked_weights_are_zero(engine, masks)

    problems: List[str] = []
    param_map = dict(engine.module.named_parameters())
    with torch.no_grad():
        for name, mask in masks.items():
            param = param_map.get(name)
            if param is None:
                if is_main_process():
                    problems.append(f"Missing parameter for mask: {name}")
                continue
            with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                if not is_main_process():
                    continue
                if tuple(mask.shape) != tuple(param.shape):
                    problems.append(f"Shape mismatch for mask {name}: {tuple(mask.shape)} vs {tuple(param.shape)}")
                    continue
                masked_values = param.detach()[mask.to(device=param.device)]
                if masked_values.numel() > 0 and bool(torch.count_nonzero(masked_values).item()):
                    problems.append(f"Masked weights are nonzero for {name}")
    return problems


def save_deepspeed_checkpoint(
    output_dir: str,
    engine,
    tokenizer,
    scheduler,
    training_state: Dict[str, Any],
    save_16bit_model: bool,
):
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    client_state = {
        "training_state": training_state,
        "scheduler": None if scheduler is None else scheduler.state_dict(),
    }
    engine.save_checkpoint(str(save_path), client_state=client_state)
    if is_main_process():
        tokenizer.save_pretrained(save_path)
        torch.save(client_state, save_path / "training_state.pt")
    if save_16bit_model:
        engine.save_16bit_model(str(save_path / "hf_model"))


def maybe_run_train_eval_stub(output_dir: str, consumed_tokens: int):
    payload = {
        "status": "not_implemented_in_training_loop",
        "consumed_tokens": int(consumed_tokens),
    }
    write_json(Path(output_dir) / f"eval_at_{consumed_tokens}.json", payload)


def main():
    args = parse_args()
    if args.deepspeed:
        try:
            import deepspeed  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "DeepSpeed was requested but is not installed in this environment. "
                "Install it with: python -m pip install deepspeed"
            ) from exc
    args.dataset_config = infer_default_dataset_config(args.dataset_name, args.dataset_config)
    if args.debug_max_train_tokens is not None:
        args.max_train_tokens = min(args.max_train_tokens, args.debug_max_train_tokens)
    rank, world_size, device = init_distributed_if_needed()
    if args.gpu_memory_fraction and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    maybe_enable_tf32()
    set_random_seed(args.seed + rank)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer_with_fallback(
        args.tokenizer_name_or_path or args.model_name_or_path,
        local_files_only=args.local_files_only,
    )

    resume_path = Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    resume_is_deepspeed_checkpoint = bool(args.deepspeed and resume_path and (resume_path / "latest").exists())
    model_load_path = args.model_name_or_path if resume_is_deepspeed_checkpoint else (args.resume_from_checkpoint or args.model_name_or_path)
    if args.compact_model_path:
        model_load_path = args.compact_model_path
        model = load_compact_for_training(args.compact_model_path, args.dtype, args.gradient_checkpointing)
    else:
        model = load_model(
            model_name_or_path=model_load_path,
            dtype=args.dtype,
            gradient_checkpointing=args.gradient_checkpointing,
            flash_attention=args.flash_attention,
            local_files_only=args.local_files_only,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    model.to(device)

    dense_param_count = count_dense_parameters(model)
    effective_params_billion = maybe_read_effective_param_count_billion(model_load_path)
    if is_main_process():
        print(f"[sanity] dense tensor params: {dense_param_count:,}", flush=True)
        if effective_params_billion is not None:
            print(f"[sanity] effective params: {effective_params_billion:.6f}B", flush=True)

    masks = None
    if args.compact_model_path:
        args.keep_pruned_weights_frozen = False
    elif args.pruning_masks_path:
        masks = load_pruning_masks(args.pruning_masks_path)
        if args.keep_pruned_weights_frozen is None:
            args.keep_pruned_weights_frozen = True
        apply_pruning_masks_inplace(model, masks)
        problems = verify_masked_weights_are_zero(model, masks)
        if problems:
            raise RuntimeError("Mask verification failed before training:\n" + "\n".join(problems))
        if is_main_process():
            print("[warning] using fixed pruning masks during recovery; pruned weights will stay zero", flush=True)
    elif infer_model_is_zero_masked(model):
        if args.keep_pruned_weights_frozen is None:
            args.keep_pruned_weights_frozen = True
        if args.keep_pruned_weights_frozen:
            masks = build_zero_masks_from_model(model)
            if is_main_process():
                print("[info] inferred zero masks from checkpoint; pruned weights will stay zero", flush=True)
        elif is_main_process():
            print(
                "[warning] model appears zero-masked, but --no-keep_pruned_weights_frozen was set. "
                "Pruned weights may regrow during recovery.",
                flush=True,
            )
    elif args.keep_pruned_weights_frozen is None:
        args.keep_pruned_weights_frozen = False

    if world_size > 1:
        if args.deepspeed:
            pass
        else:
            model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, output_device=device.index if device.type == "cuda" else None)

    packed_cfg = PackedSequenceConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        text_field=args.text_field,
        seq_len=args.seq_len,
        seed=args.seed,
        streaming=args.streaming,
        add_eos_token_between_samples=True,
        row_start=args.train_row_start,
        row_end=args.train_row_end,
        exclude_rows_path=args.exclude_rows_path,
    )
    dataset = PackedTextIterableDataset(
        tokenizer=tokenizer,
        cfg=packed_cfg,
        rank=rank,
        world_size=world_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.num_workers,
        collate_fn=causal_lm_collator,
        pin_memory=(device.type == "cuda"),
    )
    data_iter = iter(dataloader)

    global_microbatch_tokens = args.per_device_train_batch_size * args.seq_len * world_size
    total_micro_steps = math.ceil(args.max_train_tokens / global_microbatch_tokens)
    total_optimizer_steps = math.ceil(total_micro_steps / args.gradient_accumulation_steps)

    ds_config = None
    if args.deepspeed:
        rope = {}
        if args.compact_model_path:
            rope = {name: buffer.clone() for name, buffer in unwrap_model(model).named_buffers() if name.endswith("inv_freq")}
        model, optimizer, ds_config = initialize_deepspeed(
            model=model,
            args=args,
            world_size=world_size,
            total_optimizer_steps=total_optimizer_steps,
        )
        restore_rope_frequencies(model, rope)
    else:
        optimizer = torch.optim.AdamW(
            unwrap_model(model).parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    if args.deepspeed:
        scheduler = ManualLRScheduler(
            optimizer=optimizer,
            scheduler_type=args.lr_scheduler_type,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=max(total_optimizer_steps, 1),
        )
    else:
        scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=max(total_optimizer_steps, 1),
        )

    consumed_tokens = 0
    micro_step = 0
    optimizer_step = 0
    next_save_tokens = args.save_every_tokens if args.save_every_tokens > 0 else None
    next_eval_tokens = args.eval_every_tokens if args.eval_every_tokens > 0 else None
    last_loss_value: Optional[float] = None
    loss_sum = 0.0
    loss_count = 0
    rolling_loss_sum = 0.0
    rolling_loss_count = 0

    if args.resume_from_checkpoint:
        if args.deepspeed and is_deepspeed_engine(model) and (Path(args.resume_from_checkpoint) / "latest").exists():
            _, client_state = model.load_checkpoint(args.resume_from_checkpoint)
            client_state = client_state or {}
            if client_state.get("scheduler") is not None:
                scheduler.load_state_dict(client_state["scheduler"])
            training_state = client_state.get("training_state", {})
            consumed_tokens = int(training_state.get("consumed_tokens", 0))
            micro_step = int(training_state.get("micro_step", 0))
            optimizer_step = int(training_state.get("optimizer_step", 0))
            last_loss_value = training_state.get("last_loss")
            loss_sum = float(training_state.get("loss_sum", 0.0))
            loss_count = int(training_state.get("loss_count", 0))
            if next_save_tokens is not None:
                while next_save_tokens <= consumed_tokens:
                    next_save_tokens += args.save_every_tokens
            if next_eval_tokens is not None:
                while next_eval_tokens <= consumed_tokens:
                    next_eval_tokens += args.eval_every_tokens
        else:
            state_path = Path(args.resume_from_checkpoint) / "training_state.pt"
            if state_path.exists():
                state = torch.load(state_path, map_location="cpu")
                if "optimizer" in state:
                    optimizer.load_state_dict(state["optimizer"])
                if state.get("scheduler") is not None:
                    scheduler.load_state_dict(state["scheduler"])
                training_state = state.get("training_state", {})
                consumed_tokens = int(training_state.get("consumed_tokens", 0))
                micro_step = int(training_state.get("micro_step", 0))
                optimizer_step = int(training_state.get("optimizer_step", 0))
                last_loss_value = training_state.get("last_loss")
                loss_sum = float(training_state.get("loss_sum", 0.0))
                loss_count = int(training_state.get("loss_count", 0))
                if next_save_tokens is not None:
                    while next_save_tokens <= consumed_tokens:
                        next_save_tokens += args.save_every_tokens
                if next_eval_tokens is not None:
                    while next_eval_tokens <= consumed_tokens:
                        next_eval_tokens += args.eval_every_tokens

    # The packed stream is deterministic for a given seed and world size, so a resumed run skips the batches it has
    # already trained on instead of reading the first documents again.
    for _ in range(micro_step):
        next(data_iter)
    if micro_step and is_main_process():
        print(f"[resume] skipped {micro_step} micro-batches, {format_token_count(consumed_tokens)} tokens", flush=True)

    if is_main_process():
        write_json(
            output_dir / "train_config.json",
            {
                "argv": sys.argv,
                "args": vars(args),
                "rank": rank,
                "world_size": world_size,
                "global_microbatch_tokens": global_microbatch_tokens,
                "total_micro_steps": total_micro_steps,
                "total_optimizer_steps": total_optimizer_steps,
                "dense_tensor_param_count": dense_param_count,
                "effective_params_billion": effective_params_billion,
                "deepspeed_config": ds_config,
            },
        )

    wandb_run = maybe_init_wandb(args)
    start_time = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    verified_after_first_step = False

    while consumed_tokens < args.max_train_tokens:
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / args.gradient_accumulation_steps
        if args.deepspeed:
            model.backward(loss)
        else:
            loss.backward()

        raw_loss_value = float(outputs.loss.detach().float().item())
        last_loss_value = maybe_all_reduce_mean(raw_loss_value)
        loss_sum += last_loss_value
        loss_count += 1
        rolling_loss_sum += last_loss_value
        rolling_loss_count += 1
        micro_step += 1
        consumed_tokens += global_microbatch_tokens

        if (
            is_main_process()
            and args.micro_logging_steps
            and args.micro_logging_steps > 0
            and micro_step % args.micro_logging_steps == 0
        ):
            print(
                f"[iter] micro_step={micro_step}/{total_micro_steps} "
                f"optim_step={optimizer_step}/{total_optimizer_steps} "
                f"tokens={format_token_count(consumed_tokens)} "
                f"loss={last_loss_value:.6f}",
                flush=True,
            )

        should_step = (micro_step % args.gradient_accumulation_steps == 0) or (consumed_tokens >= args.max_train_tokens)
        if should_step:
            if masks is not None and args.keep_pruned_weights_frozen and not args.deepspeed:
                mask_pruned_gradients(model, masks)
            if args.max_grad_norm and args.max_grad_norm > 0 and not args.deepspeed:
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.max_grad_norm)
            if args.deepspeed:
                model.step()
            else:
                optimizer.step()
            if masks is not None and args.keep_pruned_weights_frozen:
                if args.deepspeed:
                    apply_pruning_masks_after_deepspeed_step(model, masks)
                else:
                    apply_pruning_masks_inplace(model, masks)
            scheduler.step()
            if not args.deepspeed:
                optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            if (
                args.empty_cache_every_steps
                and args.empty_cache_every_steps > 0
                and optimizer_step % args.empty_cache_every_steps == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

            if masks is not None and not verified_after_first_step:
                problems = (
                    verify_deepspeed_masked_weights_are_zero(model, masks)
                    if args.deepspeed
                    else verify_masked_weights_are_zero(model, masks)
                )
                if problems:
                    raise RuntimeError("Mask verification failed after first optimizer step:\n" + "\n".join(problems))
                verified_after_first_step = True

            if is_main_process() and optimizer_step % args.logging_steps == 0:
                avg_loss = loss_sum / max(loss_count, 1)
                rolling_loss = rolling_loss_sum / max(rolling_loss_count, 1)
                log_line = (
                    f"[train] step={optimizer_step} "
                    f"micro_step={micro_step}/{total_micro_steps} "
                    f"optim_step={optimizer_step}/{total_optimizer_steps} "
                    f"tokens={format_token_count(consumed_tokens)} "
                    f"loss={last_loss_value:.6f} "
                    f"avg_loss={avg_loss:.6f} "
                    f"rolling_loss={rolling_loss:.6f} "
                    f"lr={get_current_lr(optimizer, scheduler):.8f}"
                )
                print(log_line, flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": last_loss_value,
                            "train/avg_loss": avg_loss,
                            "train/rolling_loss": rolling_loss,
                            "train/lr": get_current_lr(optimizer, scheduler),
                            "train/consumed_tokens": consumed_tokens,
                            "train/micro_step": micro_step,
                            "train/optimizer_step": optimizer_step,
                        },
                        step=optimizer_step,
                    )
                rolling_loss_sum = 0.0
                rolling_loss_count = 0

            if next_save_tokens is not None and consumed_tokens >= next_save_tokens:
                checkpoint_state = {
                    "consumed_tokens": consumed_tokens,
                    "micro_step": micro_step,
                    "optimizer_step": optimizer_step,
                    "last_loss": last_loss_value,
                    "loss_sum": loss_sum,
                    "loss_count": loss_count,
                }
                if args.deepspeed:
                    save_deepspeed_checkpoint(
                        output_dir=str(output_dir / f"checkpoint_tokens_{next_save_tokens}"),
                        engine=model,
                        tokenizer=tokenizer,
                        scheduler=scheduler,
                        training_state=checkpoint_state,
                        save_16bit_model=False,
                    )
                elif is_main_process():
                    save_checkpoint(
                        output_dir=str(output_dir / f"checkpoint_tokens_{next_save_tokens}"),
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        training_state=checkpoint_state,
                    )
                barrier()
                while next_save_tokens is not None and next_save_tokens <= consumed_tokens:
                    next_save_tokens += args.save_every_tokens

            if next_eval_tokens is not None and consumed_tokens >= next_eval_tokens:
                if is_main_process():
                    maybe_run_train_eval_stub(str(output_dir), consumed_tokens)
                barrier()
                while next_eval_tokens is not None and next_eval_tokens <= consumed_tokens:
                    next_eval_tokens += args.eval_every_tokens

    barrier()
    final_dir = output_dir / "final"
    final_state = {
        "consumed_tokens": consumed_tokens,
        "micro_step": micro_step,
        "optimizer_step": optimizer_step,
        "last_loss": last_loss_value,
        "loss_sum": loss_sum,
        "loss_count": loss_count,
    }
    if args.compact_model_path:
        save_compact_final(final_dir, args.compact_model_path, model, args.deepspeed)
    elif args.deepspeed:
        save_deepspeed_checkpoint(
            output_dir=str(final_dir),
            engine=model,
            tokenizer=tokenizer,
            scheduler=scheduler,
            training_state=final_state,
            save_16bit_model=bool(args.deepspeed_save_16bit_model),
        )
    elif is_main_process():
        save_checkpoint(
            output_dir=str(final_dir),
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=final_state,
        )
    barrier()
    if is_main_process():
        summary = {
            "model_name_or_path": args.model_name_or_path,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "text_field": args.text_field,
            "output_dir": str(output_dir),
            "max_train_tokens": int(args.max_train_tokens),
            "actually_consumed_tokens": int(consumed_tokens),
            "seq_len": int(args.seq_len),
            "per_device_train_batch_size": int(args.per_device_train_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "world_size": int(world_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "warmup_steps": int(args.warmup_steps),
            "lr_scheduler_type": args.lr_scheduler_type,
            "dtype": args.dtype,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "flash_attention": bool(args.flash_attention),
            "dense_tensor_param_count": int(dense_param_count),
            "effective_params_billion": effective_params_billion,
            "wall_clock_time_sec": get_wall_clock_seconds(start_time),
            "final_train_loss": last_loss_value,
            "avg_train_loss": None if loss_count == 0 else float(loss_sum / loss_count),
            "loss_count": int(loss_count),
            "keep_pruned_weights_frozen": bool(args.keep_pruned_weights_frozen),
            "pruning_masks_path": args.pruning_masks_path,
            "deepspeed": bool(args.deepspeed),
            "deepspeed_zero_stage": int(args.deepspeed_zero_stage),
            "deepspeed_config": ds_config,
            "empty_cache_every_steps": int(args.empty_cache_every_steps),
            "micro_logging_steps": int(args.micro_logging_steps),
            "argv": sys.argv,
            "args": vars(args),
        }
        write_json(output_dir / "train_summary.json", summary)
        final_loss_str = "nan" if last_loss_value is None else f"{last_loss_value:.6f}"
        print(
            f"[done] consumed_tokens={format_token_count(consumed_tokens)} "
            f"final_loss={final_loss_str}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.summary.update(summary)
            wandb_run.finish()

    barrier()


if __name__ == "__main__":
    main()
