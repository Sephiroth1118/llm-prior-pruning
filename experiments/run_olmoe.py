#!/usr/bin/env python3
"""One resumable search on OLMoE-1B-7B-0924: job file -> <work>/moe/runs/<name>/{progress,results}.json.

run_model.py with the model constants of olmoe_common.py: same search (spi_search.ResearchSearch), mutation, proposer and
re-ask protocol, FineWeb search windows (8 x 2048), raw KL against a teacher cache, 16 shared random initial policies per seed.
The evaluator is evoprune/MoE/parallel_evaluator.py, whose KL is the same compute_model_metrics as the main evaluator's.
--fake replaces the GPU evaluator with a synthetic KL and the LLM with canned answers, one of them invalid, for a CPU test of the
whole loop; it needs neither the model nor the API.
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
from common import guard_environment, read_json, resolve, save_json   # noqa: E402

guard_environment()

import numpy as np                        # noqa: E402
import llm_templates                             # noqa: E402
import spi_search                          # noqa: E402
import olmoe_common as C                 # noqa: E402


def initial_design(seed, cost, decoder, count=16):
    """count random feasible policies (the generator of the main runs: seeds seed*1000+i, target 0.501 +- 0.001)."""
    from evoprune import initial_policy as loop
    out, skipped, i = [], [], 0
    while len(out) < count:
        db = loop.build_random_initial_policy(C.level_bounds(), seed=seed * 1000 + i, cost_fn=cost, target=0.501, tolerance=0.001)
        arrays = loop.policy_arrays_from_db_policy(db, C.PSEUDO_LAYERS)
        p = tuple(arrays["attn_levels"] + arrays["mlp_levels"])
        if not decoder.feasible(p) or p in out:
            skipped.append(seed * 1000 + i)
        else:
            out.append(p)
        i += 1
    return out, {"generator": "build_random_initial_policy, expert parameters 0.501+-0.001", "skipped_seeds": skipped}


def make_proposer(job, out, decoder):
    rounds = job.get("rounds", [])

    def propose(round_id, state, spec):
        source, n = spec["source"], spec["n"]
        rng = random.Random(job["seed"] * 10000 + 500 + round_id)
        if source == "random":
            return [decoder.random_template(rng) for _ in range(n)], {"round": round_id + 1, "source": "random"}
        if source == "fixed":
            items = []
            for item in spec["templates"][:n]:
                item = {"name": item.get("name", "fixed"), "weights": list(item["weights"]),
                        "rationale": item.get("rationale", "fixed template, no LLM")}
                decoder.validate(item)
                items.append(item)
            metadata = {"round": round_id + 1, "source": "fixed"}
            save_json(out / f"round{round_id + 1}_accepted.json", {"templates": items, **metadata})
            return items, metadata
        if source != "llm":
            raise ValueError(f"unknown source {source}")
        system = C.system_prompt()
        user = llm_templates.evidence(state, decoder, round_id, model_name=C.NAME, layers=C.LAYERS)
        save_json(out / f"round{round_id + 1}_request.json", {"system": system, "user": user})
        import template_reask
        return template_reask.propose(job, out, decoder, state, round_id, n, system, user, len(rounds))

    return propose


def fake_llm(system, user, run_name, tag, allow_api=True):
    """Canned answers: the first answer of seed 1, round 1 holds a template without enough capacity, to exercise the re-ask."""
    if "rejected_structures" in user:
        items = [{"name": "replacement", "weights": [0, 1, 2, 4, 0, 1, 2, 4], "rationale": "fake"}]
    elif tag.startswith("round1.ask0") and run_name.endswith("_s1"):
        items = [{"name": "too few cells", "weights": [0, 0, 0, 4, 0, 0, 0, 0], "rationale": "fake"},
                 {"name": "monotone", "weights": [.25, .5, 1, 2, .25, .5, 1, 2], "rationale": "fake"}]
    else:
        items = [{"name": "keep group 1", "weights": [0, 1, 2, 4, 0, 1, 2, 4], "rationale": "fake"},
                 {"name": "uniform", "weights": [1] * 8, "rationale": "fake"}]
    return {"raw_response": json.dumps({"structures": items}), "usage": None, "model": "fake"}, {"source": "fake", "key": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fake", action="store_true", help="synthetic evaluator and LLM (CPU test)")
    args = parser.parse_args()
    job = read_json(args.job)
    out = C.run_dir(job["name"])
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save_json(out / "job.json", job)

    from evoprune.EA import Policy, PolicyBounds, weighted_budget_transfer_mutation
    bounds = PolicyBounds.from_mapping(C.level_bounds())
    decoder = C.make_decoder()
    seed, budget = int(job["seed"]), int(job.get("budget", 320))
    cost = lambda policy: decoder.budget(tuple(policy.attn_levels) + tuple(policy.mlp_levels))
    initial, initial_source = initial_design(seed, cost, decoder, int(job.get("init_count", 16)))

    cache = {}
    for path in sorted((C.MOE / "runs").glob("*/eval_cache.jsonl")):     # the arms of a seed share their first 16 evaluations
        with open(path, "rb") as handle:
            for line in handle:
                if line.endswith(b"\n"):
                    record = json.loads(line)
                    cache[tuple(record["policy"])] = record["result"]
    stats = {"cache_hits": 0, "new_gpu_evaluations": 0, "started_unix": time.time()}
    evaluator = None
    own_cache = out / "eval_cache.jsonl"
    # Synthetic objective of the CPU test: pruning important groups and early layers costs more, plus a small policy-specific term.
    weight = np.outer(np.linspace(1.6, 1.0, C.LAYERS), [3.0, 1.6, .9, .5]).ravel()

    def fake_kl(p):
        frac = decoder.fractions[np.arange(len(p)), list(p)]
        return float(.3 + (weight * frac ** 2).sum() / 8 + (sum(i * v for i, v in enumerate(p)) % 997) / 997 * .02)

    def evaluate(policies):
        nonlocal evaluator
        if args.fake:
            return [{"kl": fake_kl(p), "reward": -fake_kl(p), "flops_reduction": decoder.budget(p), "error": None} for p in policies]
        pending = [p for p in policies if p not in cache]
        stats["cache_hits"] += len(policies) - len(pending)
        if pending:
            if evaluator is None:
                import torch
                from evoprune.MoE.db import MoEDatabase
                from evoprune.MoE.expert_groups import load_expert_groups
                from evoprune.MoE.parallel_evaluator import ParallelEvaluator
                tokens = torch.load(C.MOE / "fineweb" / "search_tokens.pt", map_location="cpu", weights_only=False)
                assert len(tokens) == 8
                evaluator = ParallelEvaluator(devices=[args.device], teacher_cache_path=str(C.MOE / "teacher_search.pt"),
                    model_name_or_path=str(C.MODEL), db_path=str(C.DB), policy_bounds=MoEDatabase(str(C.DB)).get_policy_level_bounds(),
                    expert_groups=load_expert_groups(str(resolve(job["expert_groups"]))), eval_data=tokens, ppl_eval_data=None,
                    target_flops_reduction=.5, dtype="bfloat16", low_cpu_mem_usage=False, attn_implementation="sdpa")
            results = evaluator.evaluate_candidates(policies=[C.policy_json(p) for p in pending], compute_ppl=False)
            with own_cache.open("a") as handle:
                for p, r in zip(pending, results):
                    if r["error_message"]:
                        raise RuntimeError(f"Model evaluation failed: {r['error_message']}")
                    metrics = r["metrics"]
                    assert metrics.num_samples == 8 and metrics.num_kl_tokens == 8 * 2047
                    kl = float(metrics.kl_divergence_from_dense)
                    assert abs(float(r["compression_ratio"]) - decoder.budget(p)) < 1e-9
                    row = {"kl": kl, "reward": -kl, "flops_reduction": decoder.budget(p), "error": None}
                    handle.write(json.dumps({"protocol": "fineweb_exact", "model": C.MODEL_KEY, "policy": list(p), "result": row}) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    cache[p] = row
                    stats["new_gpu_evaluations"] += 1
        return [dict(cache[p]) for p in policies]

    if args.fake:
        llm_templates.get_response = fake_llm
    schedule = spi_search.make_schedule(job.get("ea"))
    mutator = spi_search.StrengthMutator(weighted_budget_transfer_mutation, Policy, bounds, decoder, C.PSEUDO_LAYERS,
                                        strength=job["refine_strength"], tolerance=.001)
    search = spi_search.ResearchSearch(evaluate, mutator, decoder, make_proposer(job, out, decoder), seed, rounds=job.get("rounds", []),
                                      schedule=schedule, mu=job["mu"], batch=job["batch"], refine_budget=job["refine_budget"],
                                      refine_strength=job["refine_strength"])

    def checkpoint(state):
        payload = {"name": job["name"], "group": job.get("group"), "seed": seed, "phase": state.phase,
                   "final": state.phase == "complete", "job": job, "schedule": schedule.describe(),
                   "config": {"protocol": {"name": "fineweb_exact"}, "model": str(C.MODEL), "db": str(C.DB), "fake": args.fake},
                   "initial_source": initial_source, "summary": state.summary(), "entries": state.entries,
                   "branches": state.branches, "calls": state.calls, "ea_log": state.ea_log,
                   "fallback_log": state.fallback_log, "stats": {**stats, "wall_seconds": time.time() - stats["started_unix"]},
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
    print(json.dumps({"name": job["name"], **search.summary(), **stats}))


if __name__ == "__main__":
    main()
