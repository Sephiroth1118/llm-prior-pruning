#!/usr/bin/env python3
"""Run one conventional-optimizer baseline (or the LLM as the optimizer) under the fineweb_exact protocol.

Same evaluator, cache, FLOPs band, initial design and result format as run_llama31.py; the search loop is
classic_optimizers.run_search with the optimizer named in the job ("random", "tpe", "gpbo", "de", "pso", "ga").
--fake replaces the GPU evaluator with a deterministic synthetic KL for CPU tests of the loop.

    python run_classic.py --job ../configs/<folder>/<name>.json [--device cuda:0]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
from common import WORK, read_json, save_json   # noqa: E402
import model_registry                                                 # noqa: E402
from evoprune.flops import ExactFlops, MODEL_SPECS                    # noqa: E402
import numpy as np                                                    # noqa: E402
import random                                                         # noqa: E402
import eval_cache                                                       # noqa: E402
import classic_optimizers                                                     # noqa: E402
import spi_search                                                      # noqa: E402
import run_llama31 as RL                                             # noqa: E402  (LAYERS, MODEL_KEY, paths, initial_design)


def model_module(job):
    """LLaMA-3.1-8B (default) or Qwen2.5-1.5B: constants and initial design of that model's runner."""
    if job.get("model", "llama31") != "qwen15b":
        return RL
    from types import SimpleNamespace
    import run_qwen15b as RQ
    from common import FLOPS_TABLE as QF, CALIBRATION as QC, TEACHER_CACHE as QT
    return SimpleNamespace(LAYERS=28, MODEL_KEY="qwen15b", FLOPS_TABLE=QF, CALIBRATION=QC, TEACHER_CACHE=QT,
                           initial_design=RQ.initial_design)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fake", action="store_true", help="synthetic evaluator (CPU test)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    job = read_json(args.job)
    RL = model_module(job)
    out = args.out or WORK / "runs" / job["name"]
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save_json(out / "job.json", job)

    from evoprune.EA import Policy
    from evoprune.stitch_utils import get_policy_level_bounds
    LAYERS, MODEL_KEY = RL.LAYERS, RL.MODEL_KEY
    paths = {k: model_registry.REGISTRY[MODEL_KEY][k] for k in ("model", "db")}
    level_bounds = get_policy_level_bounds(paths["db"])
    flops = ExactFlops(read_json(RL.FLOPS_TABLE), MODEL_SPECS[MODEL_KEY])
    decoder = spi_search.StructureDecoder(np.concatenate([flops.attn, flops.mlp]), flops.dense_total)
    seed, budget = int(job["seed"]), int(job.get("budget", 320))
    initial, initial_source = RL.initial_design(seed, level_bounds, flops, int(job.get("init_count", 16)), decoder)

    cache = {} if args.fake else eval_cache.load_global_cache(model=MODEL_KEY)
    stats = {"cache_hits": 0, "new_gpu_evaluations": 0, "started_unix": time.time()}
    evaluator = None
    own_cache = out / "eval_cache.jsonl"

    def fake_kl(p):
        frac = decoder.fractions[np.arange(len(p)), list(p)]
        w = np.concatenate([np.linspace(3.0, 1.5, LAYERS), 0.6 + 0.8 * np.abs(np.linspace(-1, 1, LAYERS))])
        h = (hash(tuple(p)) % 1000) / 1000 * 0.02
        return float(0.3 + (w * frac ** 2).sum() / 4 + h)

    def evaluate(policies):
        nonlocal evaluator
        if args.fake:
            return [{"kl": fake_kl(p), "reward": -fake_kl(p), "flops_reduction": decoder.budget(p), "error": None} for p in policies]
        pending = [p for p in policies if p not in cache]
        stats["cache_hits"] += len(policies) - len(pending)
        if pending:
            if evaluator is None:
                import torch
                from evoprune.parallel_evaluator import ParallelEvaluator
                tokens = torch.load(RL.CALIBRATION / "search_tokens.pt", map_location="cpu", weights_only=False)
                assert len(tokens) == 8
                evaluator = ParallelEvaluator(devices=[args.device], teacher_cache_path=str(RL.TEACHER_CACHE),
                    model_name_or_path=paths["model"], db_path=paths["db"], eval_data=tokens, ppl_eval_data=None,
                    dtype="bfloat16", low_cpu_mem_usage=False, attn_implementation="sdpa", sync_zero_masks=True,
                    target_band_width=.02, reward_mode="raw_kl")
            results = evaluator.evaluate_candidates(
                policies=[Policy.from_arrays(list(p[:LAYERS]), list(p[LAYERS:])).to_db_policy() for p in pending],
                target_flops_reduction=.5, compute_ppl=False, max_samples=8)
            with own_cache.open("a") as handle:
                for p, r in zip(pending, results):
                    if r["error_message"]:
                        raise RuntimeError(f"Model evaluation failed: {r['error_message']}")
                    metrics = r["metrics"]
                    assert metrics.num_samples == 8 and metrics.num_kl_tokens == 8 * 2047
                    kl = float(metrics.kl_divergence_from_dense)
                    row = {"kl": kl, "reward": -kl, "flops_reduction": float(r["flops_reduction"]), "error": None}
                    handle.write(json.dumps({"protocol": "fineweb_exact", "model": MODEL_KEY, "policy": list(p),
                                             "result": row}) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    cache[p] = row
                    stats["new_gpu_evaluations"] += 1
        return [dict(cache[p]) for p in policies]

    rng = random.Random(seed * 7919 + 13)
    space = classic_optimizers.Space(decoder, rng)
    name = job["optimizer"]
    extra = {}
    if name == "random":       # the same generator as the initial design, continued past the first 16 draws
        draws, _ = RL.initial_design(seed, level_bounds, flops, budget, decoder)
        extra["draws"] = draws[len(initial):]
    if name == "llm":                          # the LLM as the optimizer (llm_optimizer.py)
        import llm_optimizer
        optimizer = llm_optimizer.LLMOptimizer(space, seed, decoder=decoder, level_bounds=level_bounds, layers=LAYERS,
                                            run_name=job["name"], **job.get("llm", {}))
    else:
        optimizer = classic_optimizers.OPTIMIZERS[name](space, seed, **job.get("params", {}), **extra)

    def checkpoint(state):
        payload = {"name": job["name"], "group": job.get("group"), "seed": seed, "phase": state.phase,
                   "final": state.phase == "complete", "job": job, "optimizer": name,
                   "config": {"protocol": {"name": "fineweb_exact"}, "model": paths, "batch": classic_optimizers.BATCH[name]},
                   "initial_source": initial_source, "entries": state.entries, "repair_log": space.repair_log,
                   "summary": {"best_kl": min(e["kl"] for e in state.entries), "evaluations": len(state.entries)},
                   "stats": {**stats, "wall_seconds": time.time() - stats["started_unix"]}, "updated_unix": time.time(),
                   **(optimizer.report() if hasattr(optimizer, "report") else {})}
        save_json(out / ("results.json" if state.phase == "complete" else "progress.json"), payload)

    try:
        state = classic_optimizers.run_search(optimizer, initial, budget, evaluate, space, checkpoint)
    finally:
        if evaluator is not None:
            evaluator.close()
    entries = state.entries
    assert len(entries) == budget and len({tuple(e["policy"]) for e in entries}) == budget
    assert all(.500 <= e["flops_reduction"] <= .502 and e["reward"] == -e["kl"] and not e.get("error") for e in entries)
    print(f"{job['name']}: {name} best KL {min(e['kl'] for e in entries):.4f} over {budget} evaluations; repair {space.repair_log}")


if __name__ == "__main__":
    main()
