#!/usr/bin/env python3
"""Run one search job file with the runner that belongs to it.

    python run_job.py --job ../configs/section6_plain_ea/llama31/llama31_llm_nogp_s0.json [--device cuda:0]

The runner is job["runner"] when present. Jobs of the conventional optimizers and of the LLM as the optimizer name the
optimizer in job["optimizer"] and go to run_classic.py. Every other job follows job["model"] (default qwen15b): qwen15b -> run_qwen15b.py, llama31 -> run_llama31.py, olmoe -> run_olmoe.py, every other model
of model_registry.REGISTRY -> run_model.py. Runs are written to <work>/runs/<name>/ (OLMoE: <work>/moe/runs/<name>/).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNERS = {"qwen15b": "run_qwen15b.py", "llama31": "run_llama31.py", "olmoe": "run_olmoe.py"}
RENAMED = {"ar_run_classic.py": "run_classic.py"}     # value of "runner" in older job files


def runner_of(job):
    name = job.get("runner")
    if name:
        return RENAMED.get(name, name)
    if job.get("optimizer"):
        return "run_classic.py"
    return RUNNERS.get(job.get("model", "qwen15b"), "run_model.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    job_path = args.job.resolve()                     # the runner is started in experiments/
    job = json.loads(job_path.read_text())
    command = [sys.executable, "-B", str(HERE / runner_of(job)), "--job", str(job_path), "--device", args.device]
    print(" ".join(command), flush=True)
    return subprocess.call(command, cwd=str(HERE))


if __name__ == "__main__":
    sys.exit(main())
