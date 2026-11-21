#!/usr/bin/env python3
"""Write the benchmark job files of the compression tables from held-out results.

For each model, the champion with the lowest held-out KL (results of heldout_models.py) becomes one job for
benchmark_allocation.py; dense jobs are written for models without an existing dense row.

    python make_benchmark_jobs.py --model llama31 --heldout <work>/reports/heldout_compression_llama31/results.json [--dense]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
from common import WORK, read_json, save_json   # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=("qwen15b", "llama31", "qwen3b", "qwen7b"))
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--dense", action="store_true", help="also write the dense job of the model")
    args = parser.parse_args()
    bench = WORK / "benchmark"
    (bench / "jobs").mkdir(parents=True, exist_ok=True)
    best = min(read_json(args.heldout)["champions"], key=lambda c: c["heldout_kl"])
    run = best["candidate_id"]
    job = {"name": f"{args.model}__compression__{run}", "model": args.model, "group": "compression", "seed": None,
           "source": best["source"], "source_sha256": best["source_sha256"], "entry_id": best["entry_id"],
           "policy": best["policy"], "search_kl": best["search_kl"], "heldout_kl": best["heldout_kl"],
           "flops_reduction": best["flops_reduction"], "out_dir": f"benchmark/{args.model}/compression_{run}"}
    save_json(bench / "jobs" / f"{job['name']}.json", job)
    names = [job["name"]]
    if args.dense:      # the dense job ignores the policy but writes policy.json from it
        dense = {"name": f"{args.model}__dense", "model": args.model, "group": "dense", "seed": None, "dense": True,
                 "policy": best["policy"], "out_dir": f"benchmark/{args.model}/dense"}
        save_json(bench / "jobs" / f"{dense['name']}.json", dense)
        names.append(dense["name"])
    print("\n".join(str(bench / "jobs" / f"{n}.json") for n in names))


if __name__ == "__main__":
    main()
