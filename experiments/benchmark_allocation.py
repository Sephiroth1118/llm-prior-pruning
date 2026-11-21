#!/usr/bin/env python3
"""Benchmark one search champion: export a physically compact model, then run benchmark.py with the controlled-comparison
settings (FineWeb-Edu PPL 163,840 tokens / 4096 / 40 samples, 8 lm-eval tasks, bf16, SDPA, strict tasks).

    python benchmark_allocation.py --job <job.json>

The job file (see configs/benchmark/) names the model key, the allocation ("policy": attention levels followed by MLP
levels) and "out_dir" (relative to the work directory). Export validates that the masked model's logits match the compact
model (relative L2 <= 0.03 in bf16) and that a reload is bit-identical. A dense job (`"dense": true`) evaluates the
unpruned model instead. Baselines that are ready checkpoints: `"checkpoint"` is a magnitude 2:4 checkpoint, evaluated as a
plain model and counted by its non-zero parameters; `"darwinlm"` is the published DarwinLM checkpoint (benchmark.py's
darwinlm row)."""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
from common import resolve   # noqa: E402
import model_registry        # noqa: E402

MODELS = {key: {"model": model_registry.REGISTRY[key]["model"], "db": model_registry.REGISTRY[key]["db"],
                "layers": model_registry.REGISTRY[key]["layers"]} for key in ("qwen15b", "llama31", "qwen3b", "qwen7b")}


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    tmp.replace(path)


def db_policy(policy, layers):
    return {"policy_type": "db_level_policy", "num_layers": layers,
            "layers": {str(i): {"attn_level": int(policy[i]), "mlp_level": int(policy[layers + i])} for i in range(layers)}}


def export(job, out):
    import torch
    from transformers import AutoModelForCausalLM
    from zipprune.stitch_utils import apply_policy_to_model, synchronize_zero_masks_without_shrink, normalize_policy
    from zipprune.ZipLM.compact import compact_model, save_compact_model, load_compact_model
    from zipprune.retained import retained_parameters
    spec = MODELS[job["model"]]
    model = AutoModelForCausalLM.from_pretrained(spec["model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 low_cpu_mem_usage=True, local_files_only=True,
                                                 device_map={"": "cuda:0"}).eval()
    with torch.inference_mode():
        apply_policy_to_model(model, spec["db"], normalize_policy(json.loads((out / "policy.json").read_text())))
        _, sync = synchronize_zero_masks_without_shrink(model)
        expected = retained_parameters(model, sync)
        tokens = torch.arange(1000, 1064, device="cuda:0")[None]
        masked = model(tokens, use_cache=False).logits.float()
        compact_model(model)
        actual = sum(p.numel() for p in model.parameters())
        deployed = model(tokens, use_cache=False).logits.float()
        relative = float(((masked - deployed).norm() / masked.norm().clamp_min(1e-12)).item())
        limit = float(job.get("max_relative_l2", 0.03))   # per-job override, recorded in deployment.json
        if actual != expected or not torch.isfinite(deployed).all() or relative > limit:
            raise ValueError(f"Compact export validation failed: parameters {actual} vs {expected}, relative L2 {relative}")
        save_compact_model(model, out / "compact_model")
        reloaded = load_compact_model(out / "compact_model", device="cuda:0", dtype=torch.bfloat16)
        diff = float((reloaded(tokens, use_cache=False).logits.float() - deployed).abs().max().item())
        if diff != 0 or sum(p.numel() for p in reloaded.parameters()) != actual:
            raise ValueError("Reload validation failed")
    save(out / "deployment.json", {"physical_parameters": actual, "deployment_relative_l2": relative, "reload_max_abs": diff,
                                   "relative_l2_limit": limit})
    del model, reloaded
    gc.collect()          # the compact modules hold reference cycles; without this the process keeps ~10 GB during the benchmark
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    out = resolve(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    spec = MODELS[job["model"]]
    started = time.time()
    checkpoint, darwinlm = job.get("checkpoint"), job.get("darwinlm")
    checkpoint = str(resolve(checkpoint)) if checkpoint else None
    darwinlm = str(resolve(darwinlm)) if darwinlm else None
    if not checkpoint and not (out / "policy.json").exists():
        save(out / "policy.json", db_policy(job["policy"], spec["layers"]))
    timing = {}
    if not (job.get("dense") or checkpoint or darwinlm) and not (out / "deployment.json").exists():
        t0 = time.time()
        export(job, out)
        timing["export_seconds"] = time.time() - t0
    command = [sys.executable, "-B", str(HERE / "benchmark.py"), "--model_name_or_path", checkpoint or spec["model"],
               "--db_path", checkpoint or spec["db"],
               "--config_path", str(Path(checkpoint) / "prune_summary.json") if checkpoint else str(out / "policy.json"),
               "--tokenizer_name", spec["model"],
               "--parameter_metric", "nonzero" if checkpoint else "total", "--dtype", "bfloat16", "--device", "cuda:0",
               "--attn_implementation", "sdpa",
               "--low_cpu_mem_usage", "--strict_tasks", "--ppl_dataset", "fineweb_edu", "--ppl_tokens", "163840",
               "--ppl_sequence_length", "4096", "--ppl_max_samples", "40",
               "--output_json", str(out / "benchmark_results.json"), "--output_table", str(out / "benchmark_table.md")]
    if darwinlm:
        command += ["--models", "darwinlm", "--darwinlm_model_name_or_path", darwinlm]
    elif job.get("dense") or checkpoint:
        command += ["--models", "dense"]
    else:
        command += ["--models", "compressed", "--compact_model_path", str(out / "compact_model")]
    save(out / "benchmark_command.json", command)
    t0 = time.time()
    with open(out / "benchmark.log", "a") as log:
        subprocess.run(command, cwd=str(HERE), stdout=log, stderr=subprocess.STDOUT, check=True)
    timing["benchmark_seconds"] = time.time() - t0
    timing["total_seconds"] = time.time() - started
    previous = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}
    save(out / "timing.json", {**previous, **timing})
    print(json.dumps({"job": job["name"], **timing}))


if __name__ == "__main__":
    main()
