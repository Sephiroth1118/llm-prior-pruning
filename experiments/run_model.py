#!/usr/bin/env python3
"""One resumable search run on any model of model_registry.REGISTRY (job["model"] selects the model).
Same search as run_llama31.py; only the model constants (layers, paths, FLOPs table, windows, teacher cache) differ.

    python run_model.py --job ../configs/<folder>/<name>.json [--device cuda:0]
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
from common import WORK, guard_environment, read_json, save_json, sha256   # noqa: E402

guard_environment()

import numpy as np                        # noqa: E402
import eval_cache                           # noqa: E402
import llm_templates                             # noqa: E402
import spi_search                          # noqa: E402

import model_registry                          # noqa: E402

# Set by configure(job["model"]) at the start of main().
LAYERS = MODEL_KEY = MODEL_NAME = FLOPS_TABLE = CALIBRATION = TEACHER_CACHE = REG = None


def configure(key):
    global LAYERS, MODEL_KEY, MODEL_NAME, FLOPS_TABLE, CALIBRATION, TEACHER_CACHE, REG
    REG = model_registry.REGISTRY[key]
    p = model_registry.paths(key)
    LAYERS, MODEL_KEY, MODEL_NAME = REG["layers"], key, REG["name"]
    FLOPS_TABLE, CALIBRATION, TEACHER_CACHE = p["flops"], p["fineweb"], p["teacher"]


def initial_design(seed, level_bounds, flops, count=16, decoder=None):
    """count random feasible policies (build_random_initial_policy generator, exact FLOPs target 0.501 +- 0.001, seeds seed*1000+i);
    decoder-infeasible or repeated draws are skipped. The first 16 reproduce the earlier LLaMA runs."""
    from evoprune import initial_policy as loop
    cost = lambda policy: float(flops([list(policy.attn_levels) + list(policy.mlp_levels)])[0])
    out, skipped, i = [], [], 0
    while len(out) < count:
        db = loop.build_random_initial_policy(level_bounds, seed=seed * 1000 + i, cost_fn=cost, target=0.501, tolerance=0.001)
        arrays = loop.policy_arrays_from_db_policy(db, LAYERS)
        p = tuple(arrays["attn_levels"] + arrays["mlp_levels"])
        if (decoder is not None and not decoder.feasible(p)) or p in out:
            skipped.append(seed * 1000 + i)
        else:
            out.append(p)
        i += 1
    return out, {"generator": "build_random_initial_policy, exact FLOPs 0.501+-0.001", "skipped_seeds": skipped}


def make_proposer(job, out, decoder, budget):
    rounds = job.get("rounds", [])

    def propose(round_id, state, spec):
        source, n = spec["source"], spec["n"]
        rng = random.Random(job["seed"] * 10000 + 500 + round_id)
        if source == "random":
            return [decoder.random_template(rng) for _ in range(n)], {"round": round_id + 1, "source": "random"}
        if source == "fixed":                     # hand-written / transferred template: no feedback, same
            items = []                            # template in every slot; stochastic rounding gives each slot its own anchor
            for item in spec["templates"][:n]:
                item = {"name": item.get("name", "fixed"), "weights": list(item["weights"]),
                        "rationale": item.get("rationale", "fixed template, no LLM")}
                decoder.validate(item)
                items.append(item)
            metadata = {"round": round_id + 1, "source": "fixed"}
            save_json(out / f"round{round_id + 1}_accepted.json", {"templates": items, **metadata})
            return items, metadata
        if source == "llm":                       # template prompt and evidence payload (numbers substituted only for HPO variants)
            system = llm_templates.system_prompt_for(per_round=n, rounds=len(rounds), alphabet=job.get("alphabet", (0, .25, .5, 1, 2, 4)),
                                              budget=budget, refine=job.get("refine_budget", 16))
            user, tried = llm_templates.evidence(state, decoder, round_id, model_name=MODEL_NAME, layers=LAYERS), None
        elif source == "llm_v2":                  # actual schedule + novelty rule
            tried = [b["template"]["weights"] for b in state.branches]
            system = llm_templates.SYSTEM_V2.format(rounds=len(rounds), per_round=n, budget=budget)
            user = llm_templates.evidence(state, decoder, round_id, model_name=MODEL_NAME, layers=LAYERS, extra={
                "round_of": len(rounds), "total_budget": budget, "tried_templates": tried})
        else:
            raise ValueError(f"unknown source {source}")
        save_json(out / f"round{round_id + 1}_request.json", {"system": system, "user": user})
        if source == "llm" and job.get("template_protocol") == "reask":   # re-ask until valid and decodable
            import template_reask
            return template_reask.propose(job, out, decoder, state, round_id, n, system, user, len(rounds))
        attempts = []
        for attempt in range(1 + (job.get("reask", 1) if source == "llm_v2" else 0)):
            try:
                value, info = llm_templates.get_response(system, user, job["name"], f"round{round_id + 1}.{attempt}",
                                                  allow_api=job.get("allow_api", False))
            except llm_templates.BudgetExceeded as exc:
                value, info = {"error_type": "BudgetExceeded", "detail": str(exc)}, {"source": "refused", "key": None}
            trial_rng = random.Random(job["seed"] * 10000 + 500 + round_id)
            valid, errors = llm_templates.parse_structures(value, decoder, n, trial_rng, tried)
            attempts.append({"response_source": info["source"], "key": info.get("key"), "errors": errors,
                             "usage": value.get("usage"), "model": value.get("model")})
            if not errors or info["source"] == "refused":
                break
            # One corrective re-ask (v2 only): show the rejection reasons and ask again.
            user = {**user, "previous_answer_rejected": errors,
                    "instruction": f"Return exactly {n} valid, NEW structures (not in tried_templates, not identical)."}
        valid, errors = llm_templates.parse_structures(value, decoder, n, rng, tried)
        metadata = {"round": round_id + 1, "source": source, "fallbacks": errors, "attempts": attempts,
                    "usage": value.get("usage"), "model": value.get("model")}
        save_json(out / f"round{round_id + 1}_accepted.json", {"templates": valid, **metadata})
        return valid, metadata

    return propose


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-gpu", action="store_true", help="fail on any cache miss (replay validation)")
    parser.add_argument("--compare", type=Path, help="reference results.json whose entries must match exactly")
    parser.add_argument("--out", type=Path, help="override run directory (validation only)")
    args = parser.parse_args()
    job = read_json(args.job)
    configure(job["model"])
    out = args.out or WORK / "runs" / job["name"]
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save_json(out / "job.json", job)

    from evoprune.EA import Policy, PolicyBounds, weighted_budget_transfer_mutation
    from evoprune.stitch_utils import get_policy_level_bounds
    paths = {"model": REG["model"], "db": REG["db"]}
    level_bounds = get_policy_level_bounds(paths["db"])
    bounds = PolicyBounds.from_mapping(level_bounds)
    flops = model_registry.ExactFlops(read_json(FLOPS_TABLE), REG["spec"])   # equals evoprune.flops.ExactFlops when head_dim == hidden/heads
    if job.get("alphabet"):
        decoder = spi_search.AlphabetDecoder(np.concatenate([flops.attn, flops.mlp]), flops.dense_total, weights=job["alphabet"])
    else:
        decoder = spi_search.StructureDecoder(np.concatenate([flops.attn, flops.mlp]), flops.dense_total)
    seed, budget = int(job["seed"]), int(job.get("budget", 320))
    initial, initial_source = initial_design(seed, level_bounds, flops, int(job.get("init_count", 16)), decoder)

    cache = eval_cache.load_global_cache(model=MODEL_KEY)
    stats = {"cache_hits": 0, "new_gpu_evaluations": 0, "started_unix": time.time()}
    evaluator = None
    own_cache = out / "eval_cache.jsonl"

    def evaluate(policies):
        nonlocal evaluator
        pending = [p for p in policies if p not in cache]
        stats["cache_hits"] += len(policies) - len(pending)
        if pending:
            if args.no_gpu:
                raise RuntimeError(f"cache miss in --no-gpu mode at evaluation {len(search.entries) + 1}")
            if evaluator is None:
                import torch
                from evoprune.parallel_evaluator import ParallelEvaluator
                tokens = torch.load(CALIBRATION / "search_tokens.pt", map_location="cpu", weights_only=False)
                assert len(tokens) == 8
                evaluator = ParallelEvaluator(devices=[args.device], teacher_cache_path=str(TEACHER_CACHE),
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
                    if REG.get("flops_override"):     # evaluator assumes head_dim = hidden/heads (wrong for Nemo); use exact FLOPs
                        row = {"kl": kl, "reward": -kl, "flops_reduction": decoder.budget(p),
                               "flops_reduction_evaluator": float(r["flops_reduction"]), "error": None}
                    else:
                        row = {"kl": kl, "reward": -kl, "flops_reduction": float(r["flops_reduction"]), "error": None}
                    handle.write(json.dumps({"protocol": "fineweb_exact", "model": MODEL_KEY, "policy": list(p),
                                             "result": row}) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    cache[p] = row
                    stats["new_gpu_evaluations"] += 1
        return [dict(cache[p]) for p in policies]

    schedule = spi_search.make_schedule(job.get("ea"))
    mutator = spi_search.StrengthMutator(weighted_budget_transfer_mutation, Policy, bounds, decoder, LAYERS,
                                        strength=job.get("refine_strength", .0075), tolerance=.001)
    common = dict(rounds=job.get("rounds", []), schedule=schedule, mu=job.get("mu", 8), batch=job.get("batch", 8),
                  refine_budget=job.get("refine_budget", 16), refine_strength=job.get("refine_strength", .0075))
    if job.get("gp"):
        import gp_guide
        gp = job["gp"]
        guide = gp_guide.GPGuide(bounds, mutator.cost, Policy, LAYERS, warmup=gp.get("warmup", 80),
                              interval=gp.get("interval", 80), window=gp.get("window", 80),
                              minimum_pairs=gp.get("minimum_pairs", 8),
                              min_weight=gp.get("min_weight", 0.25), max_weight=gp.get("max_weight", 4.0))
        search = spi_search.GPSearch(evaluate, mutator, decoder, make_proposer(job, out, decoder, budget), seed,
                                    guide=guide, **common)
    else:
        search = spi_search.ResearchSearch(evaluate, mutator, decoder, make_proposer(job, out, decoder, budget), seed, **common)

    def checkpoint(state):
        payload = {"name": job["name"], "group": job.get("group"), "seed": seed, "phase": state.phase,
                   "final": state.phase == "complete", "job": job, "schedule": schedule.describe(),
                   "config": {"protocol": {"name": "fineweb_exact"}, "model": paths},
                   "initial_source": initial_source, "summary": state.summary(), "entries": state.entries,
                   "branches": state.branches, "calls": state.calls, "ea_log": state.ea_log,
                   "fallback_log": state.fallback_log, "gp_log": getattr(state, "gp_log", []),
                   "stats": {**stats, "wall_seconds": time.time() - stats["started_unix"]},
                   "updated_unix": time.time()}
        save_json(out / ("results.json" if state.phase == "complete" else "progress.json"), payload)

    try:
        search.run(initial, budget, checkpoint)
    finally:
        if evaluator is not None:
            evaluator.close()
    entries = search.entries
    assert len(entries) == budget and len({tuple(e["policy"]) for e in entries}) == budget
    assert all(.500 <= e["flops_reduction"] <= .502 and e["reward"] == -e["kl"] and not e.get("error") for e in entries)
    if args.compare:
        reference = read_json(args.compare)
        keys = ("id", "policy", "kl", "parent", "phase", "branch", "origin")
        mismatch = next((i for i, (a, b) in enumerate(zip(entries, reference["entries"]))
                         if any(a[k] != b[k] for k in keys)), None)
        same_branches = search.branches == reference.get("branches", search.branches)
        report = {"compared": min(len(entries), len(reference["entries"])), "first_mismatch": mismatch,
                  "branches_identical": same_branches, "reference": str(args.compare),
                  "reference_sha256": sha256(args.compare)}
        save_json(out / "comparison.json", report)
        print(json.dumps(report))
        if mismatch is not None or not same_branches:
            sys.exit(3)
    print(json.dumps({"name": job["name"], **search.summary(), **stats}))


if __name__ == "__main__":
    main()
