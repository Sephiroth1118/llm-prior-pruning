#!/usr/bin/env python3
"""One row of the recovery-training table, Llama-3.1-8B: full fine-tuning of a compact checkpoint, then its benchmark.

Settings: FineWeb-Edu sample-10BT, 100M tokens, sequences of 2048, four GPUs x 1 x 16 accumulation, AdamW lr 2e-5, cosine,
100 warmup steps, no weight decay, grad norm 1.0, bf16, DeepSpeed ZeRO-3. Training reads only the first half of the split and
skips the search and held-out documents (<work>/recovery/excluded_rows_fineweb.json, written by recovery_rows.py), so the
documents used for perplexity evaluation are never trained on. All rows train their compact checkpoint (LLM-Pruner exists only
in that form and the new allocations were exported in that form); pruned weights do not exist there, so they stay out by
construction. The attention backend is SDPA, as in the benchmark.

  --stage train   trains, reloads the result strictly on the CPU, moves it to
                  benchmark/llama31/recovery_<row>/compact_model and writes benchmark/jobs/llama31__recovery__<row>.json
  --stage bench   benchmark_allocation.py on that job (same benchmark as the one-shot rows; export is skipped as deployment.json exists)
  --pilot-tokens N   timing pilot: N tokens into recovery/pilot_<row> with a checkpoint halfway, then the second half again
                  resumed from that checkpoint into recovery/pilot_<row>_resume; prints tokens per second, peak GPU memory and
                  the losses of both halves, which match if the resume is exact; no benchmark

Usage: recovery_job.py --job ../configs/recovery/llama31__recovery_ours.json --stage train --gpus 0,1,2,3
"""
import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
from common import HF_DIR, PYTHON, RECOMMENDED_ENV, REPO, WORK, read_json, resolve, save_json   # noqa: E402

EXCLUDED = WORK / "recovery" / "excluded_rows_fineweb.json"
LLAMA = str(HF_DIR / "Llama-3.1-8B")                    # tokenizer of the source model


def log(message):
    print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} [recovery] {message}", flush=True)


def peak_memory():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return {int(a): int(b) for a, b in (line.split(",") for line in out.stdout.strip().splitlines())}


def complete_checkpoints(out):
    """Checkpoints whose training_state.pt exists; run_fullft_recovery.py writes it after the ZeRO state."""
    found = [path for path in out.glob("checkpoint_tokens_*") if (path / "training_state.pt").exists()]
    return sorted(found, key=lambda path: int(path.name.rsplit("_", 1)[1]))


def train(job, out, tokens, gpus, port, offload=False, fraction=None, resume=None, logging_steps=10):
    row_info = read_json(EXCLUDED)
    settings = job["training"]
    command = [PYTHON, "-B", "-m", "torch.distributed.run", "--nproc_per_node", str(len(gpus)), "--master_port", str(port),
               str(HERE / "run_fullft_recovery.py"), "--model_name_or_path", job["source"], "--compact_model_path", job["source"],
               "--tokenizer_name_or_path", LLAMA, "--dataset_name", row_info["dataset"], "--dataset_config", row_info["config"],
               "--dataset_split", row_info["split"], "--no-streaming", "--train_row_start", str(row_info["train_row_start"]),
               "--train_row_end", str(row_info["train_row_end"]), "--exclude_rows_path", str(EXCLUDED), "--output_dir", str(out),
               "--max_train_tokens", str(tokens), "--seq_len", str(settings["seq_len"]), "--per_device_train_batch_size", "1",
               # the steps of four GPUs x 16 accumulation, 131,072 tokens, on any number of GPUs
               "--gradient_accumulation_steps", str(settings["gradient_accumulation_steps"] * 4 // len(gpus)),
               "--learning_rate", str(settings["learning_rate"]), "--weight_decay", str(settings["weight_decay"]),
               "--warmup_steps", str(settings["warmup_steps"]), "--lr_scheduler_type", settings["lr_scheduler_type"],
               "--max_grad_norm", str(settings["max_grad_norm"]), "--dtype", "bfloat16", "--gradient_checkpointing", "--no-flash_attention",
               "--save_every_tokens", str(settings["save_every_tokens"]), "--logging_steps", str(logging_steps), "--empty_cache_every_steps", "10",
               "--seed", str(settings["seed"]), "--local_files_only", "--deepspeed", "--deepspeed_zero_stage", "3",
               "--deepspeed_torch_adam"]   # same AdamW update as FusedAdam, without the ninja build requirement
    if offload:      # one GPU: fp32 weights and AdamW states in CPU memory, about 55 GB
        command.append("--deepspeed_offload_optimizer")
    if fraction:     # shared cards: stay below this share of the GPU memory
        command += ["--gpu_memory_fraction", str(fraction)]
    if resume:
        command += ["--resume_from_checkpoint", str(resume)]
    env = {**os.environ, **RECOMMENDED_ENV, "PYTHONPATH": str(REPO),
           "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    out.mkdir(parents=True, exist_ok=True)
    peak = {g: 0 for g in gpus}
    with open(out / "train.log", "a") as handle:
        start = handle.tell()
        proc = subprocess.Popen(command, cwd=str(HERE), env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        failed_at = None
        while proc.poll() is None:
            time.sleep(30)
            used = peak_memory()
            peak = {g: max(peak[g], used.get(g, 0)) for g in gpus}
            # After a Python error the ranks can spin in the NCCL teardown and never exit.
            if failed_at is None and "Traceback" in (out / "train.log").read_text(errors="replace")[start:]:
                failed_at = time.time()
            if failed_at is not None and time.time() - failed_at > 120:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                break
            # Each checkpoint holds the fp32 weights and AdamW states, about 60 GB; only the newest complete one is kept.
            for path in complete_checkpoints(out)[:-1]:
                shutil.rmtree(path)
    save_json(out / "gpu_peak_mib.json", peak)
    if proc.returncode:
        raise RuntimeError(f"training failed with exit code {proc.returncode}; see {out / 'train.log'}")
    return peak


def reload_check(job, compact):
    """Strict reload on the CPU: same layout and parameter count as the source, finite weights."""
    code = ("import json, sys, torch\nfrom zipprune.ZipLM.compact import load_compact_model\n"
            "m = load_compact_model(sys.argv[1], device='cpu', dtype=torch.bfloat16)\n"
            "print(json.dumps({'parameters': sum(p.numel() for p in m.parameters()),"
            " 'finite': all(bool(torch.isfinite(p).all()) for p in m.parameters())}))\n")
    env = {**os.environ, **RECOMMENDED_ENV, "PYTHONPATH": str(REPO), "CUDA_VISIBLE_DEVICES": ""}
    done = subprocess.run([PYTHON, "-B", "-c", code, str(compact)], cwd=str(HERE), env=env, capture_output=True, text=True, check=True)
    result = json.loads(done.stdout.strip().splitlines()[-1])
    if result["parameters"] != job["parameters"] or not result["finite"]:
        raise RuntimeError(f"trained checkpoint failed the reload check: {result} (expected {job['parameters']} parameters)")
    if (compact / "compact_manifest.json").read_text() != (Path(job["source"]) / "compact_manifest.json").read_text():
        raise RuntimeError("trained checkpoint has another layout than its source")
    return result


def bench_paths(job):
    out = WORK / "benchmark" / "llama31" / f"recovery_{job['row']}"
    return out, WORK / "benchmark" / "jobs" / f"llama31__recovery__{job['row']}.json"


def stage_train(job, port, gpus, offload, fraction):
    bench_dir, bench_job = bench_paths(job)
    if bench_job.exists():
        return
    settings = job["training"]
    out = WORK / "recovery" / "runs" / job["row"]
    resume = (complete_checkpoints(out) or [None])[-1]
    log(f"training {job['row']} from {resume or job['source']} ({settings['tokens']} tokens, GPUs {gpus})")
    t0 = time.time()
    train(job, out, settings["tokens"], gpus, port, offload, fraction, resume)
    compact = out / "final" / "compact_model"
    reload_check(job, compact)
    bench_dir.mkdir(parents=True, exist_ok=True)
    os.replace(compact, bench_dir / "compact_model")
    # benchmark_allocation.py skips the export when deployment.json exists; benchmark.py only hashes policy.json for a compact model.
    save_json(bench_dir / "deployment.json", {"physical_parameters": job["parameters"], "source_compact_model": job["source"],
                                              "recovery_training": str(out / "train_summary.json"), "train_seconds": time.time() - t0})
    if job.get("policy_file"):
        shutil.copy2(job["policy_file"], bench_dir / "policy.json")
    else:
        save_json(bench_dir / "policy.json", {"policy_type": "none", "note": job["method"]})
    save_json(bench_job, {"name": bench_job.stem, "model": "llama31", "group": "recovery", "seed": None, "row": job["row"],
                          "source": job["source"], "out_dir": str(bench_dir.relative_to(WORK))})
    for path in out.glob("checkpoint_tokens_*"):      # the ZeRO state written for a restart, about 55 GB; the run is complete
        shutil.rmtree(path)
    log(f"training {job['row']} done in {(time.time() - t0) / 3600:.1f} h; job {bench_job.name}")


def stage_bench(job, gpu):
    _, bench_job = bench_paths(job)
    env = {**os.environ, **RECOMMENDED_ENV, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(REPO)}
    log(f"benchmark {job['row']} on gpu{gpu}")
    (WORK / "logs").mkdir(parents=True, exist_ok=True)
    with open(WORK / "logs" / f"bench_{bench_job.stem}.log", "a") as handle:
        subprocess.run([PYTHON, "-B", str(HERE / "benchmark_allocation.py"), "--job", str(bench_job)], cwd=str(HERE), env=env,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    log(f"benchmark {job['row']} done")


def step_losses(log_file):
    return {int(line.split("step=", 1)[1].split()[0]): float(line.split(" loss=", 1)[1].split()[0])
            for line in log_file.read_text(errors="replace").splitlines() if line.startswith("[train] step=")}


def pilot(job, tokens, port, gpus, offload, fraction):
    out = WORK / "recovery" / f"pilot_{job['row']}"
    step = job["training"]["seq_len"] * job["training"]["gradient_accumulation_steps"] * 4
    # one checkpoint just past half of the run; a second one at the end would replace it before the resumed half starts
    pilot_job = {**job, "training": {**job["training"], "save_every_tokens": (tokens // step // 2 + 1) * step}}
    peak = train(pilot_job, out, tokens, gpus, port, offload, fraction, logging_steps=1)
    check = reload_check(job, out / "final" / "compact_model")
    summary = read_json(out / "train_summary.json")
    rate = summary["actually_consumed_tokens"] / summary["wall_clock_time_sec"]
    resumed = WORK / "recovery" / f"pilot_{job['row']}_resume"
    train(pilot_job, resumed, tokens, gpus, port, offload, fraction, complete_checkpoints(out)[-1], logging_steps=1)
    report = {"row": job["row"], "tokens": summary["actually_consumed_tokens"], "seconds": summary["wall_clock_time_sec"],
              "tokens_per_second": rate, "hours_for_100M": 1e8 / rate / 3600, "gpu_peak_mib": peak, "reload": check,
              "losses": step_losses(out / "train.log"), "losses_resumed": step_losses(resumed / "train.log")}
    save_json(out / "pilot.json", report)
    print(json.dumps(report, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "bench"))
    parser.add_argument("--gpu", type=int, default=0, help="GPU of --stage bench")
    parser.add_argument("--pilot-tokens", type=int)
    parser.add_argument("--port", type=int, default=29511)
    parser.add_argument("--gpus", default="0,1,2,3", help="comma-separated GPUs for training")
    parser.add_argument("--offload-optimizer", action="store_true", help="AdamW states in CPU memory (needed on one GPU)")
    parser.add_argument("--memory-fraction", type=float, help="cap on the share of each training GPU")
    args = parser.parse_args()
    job = read_json(args.job)
    job["source"] = str(resolve(job["source"]))
    if job.get("policy_file"):
        job["policy_file"] = str(resolve(job["policy_file"]))
    gpus = [int(g) for g in args.gpus.split(",")]
    if args.pilot_tokens:
        pilot(job, args.pilot_tokens, args.port, gpus, args.offload_optimizer, args.memory_fraction)
    elif args.stage == "train":
        stage_train(job, args.port, gpus, args.offload_optimizer, args.memory_fraction)
    elif args.stage == "bench":
        stage_bench(job, args.gpu)
    else:
        parser.error("--stage or --pilot-tokens is required")


if __name__ == "__main__":
    main()
