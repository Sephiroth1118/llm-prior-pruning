#!/usr/bin/env python3
"""Benchmark of one OLMoE-1B-7B-0924 allocation: moe/bench_jobs/<name>.json -> <out_dir>/benchmark_results.json.

Same benchmark as benchmark_allocation.py (FineWeb-Edu PPL 163,840 tokens / 4096 / 40 samples, 8 lm-eval tasks, bf16, SDPA, strict tasks),
with two differences. zipprune/ZipLM/compact.py does not cover MoE models, so a pruned model is a masked checkpoint saved by
stitch_model and counted by its non-zero parameters, as the magnitude 2:4 rows are. transformers 4.44.2 has no OLMoE, so these
jobs run with transformers 4.45.2 and torch 2.6; all rows of the MoE table share that environment, with the same datasets
and lm_eval as the other tables.
"""
import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
from common import resolve   # noqa: E402
import olmoe_common as C       # noqa: E402

MODEL, DB = C.MODEL, C.DB
LAYERS, GROUPS = 16, 4


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    tmp.replace(path)


def stitch(job, out):
    import torch
    from evoprune.MoE.stitch_utils import stitch_model
    result = stitch_model(str(MODEL), str(DB), policy=json.loads((out / "policy.json").read_text()), dtype="bfloat16",
                          attn_implementation="sdpa", device="cuda:0", sync_zero_masks=True, expert_groups_path=str(resolve(job["expert_groups"])))
    nonzero = sum(int(torch.count_nonzero(p).item()) for p in result.model.parameters())
    result.model.save_pretrained(out / "checkpoint")
    save(out / "deployment.json", {"nonzero_parameters": nonzero, "total_parameters": sum(p.numel() for p in result.model.parameters())})
    del result
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    out = resolve(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    started, timing = time.time(), {}
    if not (out / "policy.json").exists():
        policy = job.get("policy") or [0] * (LAYERS * GROUPS)
        save(out / "policy.json", {"policy_type": "moe_group_level_policy", "num_layers": LAYERS, "num_groups": GROUPS,
                                   "moe_group_levels": [[int(policy[GROUPS * l + g]) for g in range(GROUPS)] for l in range(LAYERS)]})
    if not job.get("dense") and not (out / "deployment.json").exists():
        t0 = time.time()
        stitch(job, out)
        timing["stitch_seconds"] = time.time() - t0
    model = MODEL if job.get("dense") else out / "checkpoint"
    command = [sys.executable, "-B", str(HERE / "benchmark.py"), "--model_name_or_path", str(model), "--db_path", str(model),
               "--config_path", str(out / "policy.json"), "--tokenizer_name", str(MODEL), "--parameter_metric", "nonzero",
               "--dtype", "bfloat16", "--device", "cuda:0", "--attn_implementation", "sdpa", "--low_cpu_mem_usage", "--strict_tasks",
               "--ppl_dataset", "fineweb_edu", "--ppl_tokens", "163840", "--ppl_sequence_length", "4096", "--ppl_max_samples", "40",
               "--models", "dense", "--output_json", str(out / "benchmark_results.json"), "--output_table", str(out / "benchmark_table.md")]
    save(out / "benchmark_command.json", command)
    t0 = time.time()
    with open(out / "benchmark.log", "a") as log:
        subprocess.run(command, cwd=str(HERE), stdout=log, stderr=subprocess.STDOUT, check=True)
    timing["benchmark_seconds"] = time.time() - t0
    timing["total_seconds"] = time.time() - started
    save(out / "timing.json", timing)
    print(json.dumps({"job": job["name"], **timing}))


if __name__ == "__main__":
    main()
