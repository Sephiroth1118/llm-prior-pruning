#!/usr/bin/env python3
"""Preparation of the OLMoE-1B-7B-0924 case study.

  check             what is still missing before the runs: model weights and level database
  tokens            frozen FineWeb documents of the Qwen protocol, re-tokenized (CPU)
  teacher           dense logits on the search and held-out windows; database level 0 must equal the dense weights (GPU)
  groups            expert groups from router statistics on the search windows (GPU)
  jobs              moe/jobs/*: EA, LLM template, rule of thumb, random template; 4 seeds
  distill           own template from the champions of the EA and the LLM template, and its jobs
  heldout           held-out KL of the LLM template champions (GPU); the lowest gives the "Ours" row
  bench-jobs        moe/bench_jobs/*: Dense, Uniform level 5, Ours
  prompt            write the MoE template prompt to prompts/llm_template_system_moe.txt
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
from common import CALIBRATION, WORK, read_json, save_json   # noqa: E402
import olmoe_common as C                 # noqa: E402

SEEDS = (0, 1, 2, 3)
ARMS = ("ea", "llm", "random", "rule")
HELDOUT_WINDOWS = 32


def base_job(arm, seed, rounds, allow_api=False):
    name = f"olmoe_{arm}_nogp_s{seed}"
    return {"name": name, "group": f"olmoe_{arm}_nogp", "seed": seed, "phase": "MOE", "budget": 320, "allow_api": allow_api,
            "rounds": rounds, "refine_budget": 16, "refine_strength": .015, "ea": {"schedule": "fixed", "start": .015},
            "mu": 8, "batch": 4, "init_count": 16, "model": C.MODEL_KEY, "expert_groups": rel(C.GROUPS_FILE)}


def fixed_rounds(template):
    return [{"source": "fixed", "n": 2, "templates": [template, template]} for _ in range(2)]


def rel(path):
    """Paths in job files are relative to the work directory when they lie inside it."""
    return os.path.relpath(path, WORK) if Path(path).is_relative_to(WORK) else str(path)


def cmd_check(_):
    problems = []
    if not (C.MODEL / "config.json").exists():
        problems.append(f"model weights missing: {C.MODEL} (download allenai/OLMoE-1B-7B-0924)")
    if not C.DB.is_dir():
        problems.append(f"level database missing: {C.DB} (see README, Level databases)")
    else:
        modules = [d for d in C.DB.iterdir() if d.name.startswith("model.layers.")]
        incomplete = [d.name for d in modules if len(list(d.glob("*.pth"))) != 11]
        if len(modules) != C.LAYERS * C.EXPERTS or incomplete:
            problems.append(f"database: {len(modules)} modules, {len(incomplete)} incomplete")
    print("\n".join(problems) or "ready")
    return 1 if problems else 0


def cmd_tokens(_):
    import torch
    import model_registry
    from transformers import AutoTokenizer
    target = C.MOE / "fineweb"
    if (target / "manifest.json").exists():
        return
    docs = read_json(CALIBRATION / "documents.json")
    search_docs, heldout_docs = list(docs["search"]), list(docs["heldout"])
    tokenizer = AutoTokenizer.from_pretrained(C.MODEL, use_fast=True)
    moved = 0
    while True:
        search, used, tail = model_registry._pack(search_docs, tokenizer, model_registry.SEARCH_WINDOWS)
        if len(search) == model_registry.SEARCH_WINDOWS:
            break
        search_docs.append(heldout_docs.pop(0))      # same remedy as the other models
        moved += 1
    heldout, used_h, tail_h = model_registry._pack(heldout_docs, tokenizer, HELDOUT_WINDOWS)
    assert not {d["row"] for d in used} & {d["row"] for d in used_h}
    target.mkdir(parents=True, exist_ok=True)
    torch.save(search, target / "search_tokens.pt")
    torch.save(heldout, target / "heldout_tokens.pt")
    save_json(target / "documents.json", {"search": used, "heldout": used_h})
    save_json(target / "manifest.json", {"source_documents": str(CALIBRATION / "documents.json"), "tokenizer": str(C.MODEL),
                                         "search": {"windows": len(search), "documents": len(used), "discarded_tail": tail},
                                         "heldout": {"windows": len(heldout), "documents": len(used_h), "discarded_tail": tail_h},
                                         "moved_heldout_documents_to_search": moved})


def dense_model(device):
    import torch
    from evoprune.MoE.stitch_utils import load_base_model
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(C.GPU_MEM_FRACTION)
    return load_base_model(str(C.MODEL), dtype="bfloat16", attn_implementation="sdpa", device=device)


def cmd_teacher(args):
    import torch
    model = dense_model(args.device)
    wrong = []
    for name in ("model.layers.0.mlp.experts.0.down_proj", "model.layers.7.mlp.experts.31.down_proj", "model.layers.15.mlp.experts.63.down_proj"):
        level0 = torch.load(C.DB / name / "0.pth", map_location="cpu")
        if not torch.equal(level0.float(), model.get_submodule(name).weight.detach().float().cpu()):
            wrong.append(name)
    if wrong:
        raise RuntimeError(f"database level 0 differs from the dense weights of this model revision: {wrong}")
    with torch.no_grad():
        for part in ("search", "heldout"):
            tokens = torch.load(C.MOE / "fineweb" / f"{part}_tokens.pt", map_location="cpu", weights_only=False)
            logits = [model(t.to(args.device)).logits.detach().float().cpu() for t in tokens]
            torch.save(logits, str(C.MOE / f"teacher_{part}.pt"))
            print(part, len(logits), logits[0].shape)


def cmd_groups(args):
    import torch
    from evoprune.MoE.expert_groups import build_fixed_groups_from_router_stats, collect_router_statistics, save_expert_groups
    tokens = torch.load(C.MOE / "fineweb" / "search_tokens.pt", map_location="cpu", weights_only=False)
    stats = collect_router_statistics(dense_model(args.device), tokens, device=args.device)
    payload = build_fixed_groups_from_router_stats(stats, num_groups=C.GROUPS)
    payload["metadata"]["calibration"] = "FineWeb search windows of the fineweb_exact protocol (8 x 2048), no PPL or held-out document"
    save_expert_groups(str(C.GROUPS_FILE), payload)


def cmd_jobs(_):
    (C.MOE / "jobs").mkdir(parents=True, exist_ok=True)
    rule = {"name": "moe_usage_rule", "weights": C.RULE_WEIGHTS,
            "rationale": "Fixed before any run of this template: the less the router uses a group, the more it is pruned, in both depth segments."}
    names = []
    for seed in SEEDS:
        for arm in ARMS:
            if arm == "ea":
                job = base_job(arm, seed, [])
            elif arm == "llm":
                job = {**base_job(arm, seed, [{"source": "llm", "n": 2}] * 2, allow_api=True), "template_protocol": "reask"}
            elif arm == "random":
                job = base_job(arm, seed, [{"source": "random", "n": 2}] * 2)
            else:
                job = base_job(arm, seed, fixed_rounds(rule))
            save_json(C.job_path(job["name"]), job)
            names.append(job["name"])


def quant(v):
    alpha = np.asarray(C.ALPHABET[1:])
    v = np.asarray(v, float) / np.max(v) * 2            # the largest cell equals 2, as for the transferred templates
    out = [0.0 if x < .125 else float(alpha[np.argmin(np.abs(np.log2(alpha) - np.log2(x)))]) for x in v]
    return [int(x) if float(x).is_integer() else x for x in out]


def cmd_distill(_):
    decoder = C.make_decoder()
    rows = []
    for arm in ("ea", "llm"):
        for seed in SEEDS:
            best = min(C.load_entries(f"olmoe_{arm}_nogp_s{seed}"), key=lambda e: e["kl"])
            frac = decoder.fractions[np.arange(len(best["policy"])), best["policy"]]
            rows.append([frac[decoder.cells == c].mean() for c in range(8)])
    mean = np.mean(rows, axis=0)
    template = {"name": "olmoe_own_320", "weights": quant(mean),
                "rationale": f"Mean shape of {len(rows)} champions (EA and LLM template, 320 evaluations), quantized to the alphabet"}
    decoder.validate(template)
    save_json(C.MOE / "own_template.json", {**template, "mean_fractions": [round(float(x), 4) for x in mean], "champions": len(rows)})
    names = []
    for seed in SEEDS:
        job = base_job("own", seed, fixed_rounds(template))
        save_json(C.job_path(job["name"]), job)
        names.append(job["name"])
    print(template["weights"], [round(float(x), 3) for x in mean])


def cmd_heldout(args):
    import torch
    from evoprune.MoE.db import MoEDatabase
    from evoprune.MoE.expert_groups import load_expert_groups
    from evoprune.MoE.parallel_evaluator import ParallelEvaluator
    tokens = torch.load(C.MOE / "fineweb" / "heldout_tokens.pt", map_location="cpu", weights_only=False)
    champions = []
    for seed in SEEDS:
        name = f"olmoe_llm_nogp_s{seed}"
        best = min(C.load_entries(name), key=lambda e: e["kl"])
        champions.append({"run": name, "seed": seed, "entry_id": best["id"], "search_kl": best["kl"], "policy": best["policy"],
                          "flops_reduction": best["flops_reduction"]})
    evaluator = ParallelEvaluator(devices=[args.device], teacher_cache_path=str(C.MOE / "teacher_heldout.pt"), model_name_or_path=str(C.MODEL),
                                  db_path=str(C.DB), policy_bounds=MoEDatabase(str(C.DB)).get_policy_level_bounds(),
                                  expert_groups=load_expert_groups(str(C.GROUPS_FILE)), eval_data=tokens, ppl_eval_data=None,
                                  target_flops_reduction=.5, dtype="bfloat16", low_cpu_mem_usage=False, attn_implementation="sdpa")
    try:
        results = evaluator.evaluate_candidates(policies=[C.policy_json(c["policy"]) for c in champions], compute_ppl=False)
    finally:
        evaluator.close()
    for champion, result in zip(champions, results):
        if result["error_message"]:
            raise RuntimeError(result["error_message"])
        champion["heldout_kl"] = float(result["metrics"].kl_divergence_from_dense)
    save_json(C.MOE / "heldout.json", {"selection": "minimum search KL of a run; held-out KL chooses the reported checkpoint", "windows": len(tokens),
                                       "champions": champions})
    print([(c["run"], round(c["search_kl"], 3), round(c["heldout_kl"], 3)) for c in champions])


def cmd_bench_jobs(_):
    (C.MOE / "bench_jobs").mkdir(parents=True, exist_ok=True)
    pick = min(read_json(C.MOE / "heldout.json")["champions"], key=lambda c: c["heldout_kl"])
    decoder = C.make_decoder()
    uniform = [5] * (C.LAYERS * C.GROUPS)
    assert abs(decoder.budget(uniform) - .5) < 1e-9, decoder.budget(uniform)
    jobs = [{"name": "olmoe__dense", "dense": True}, {"name": "olmoe__uniform_l5", "policy": uniform},
            {"name": f"olmoe__ours_s{pick['seed']}", "policy": pick["policy"], "source": pick["run"], "entry_id": pick["entry_id"],
             "search_kl": pick["search_kl"], "heldout_kl": pick["heldout_kl"]}]
    for job in jobs:
        job["out_dir"] = rel(C.MOE / "benchmark" / job["name"])
        job["expert_groups"] = rel(C.GROUPS_FILE)
        save_json(C.MOE / "bench_jobs" / f"{job['name']}.json", job)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("check", "tokens", "teacher", "groups", "jobs", "distill", "heldout", "bench-jobs", "prompt"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "prompt":
        C.PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
        C.PROMPT_FILE.write_text(C.system_prompt())
        print(C.PROMPT_FILE, C.prompt_sha())
        return 0
    return {"check": cmd_check, "tokens": cmd_tokens, "teacher": cmd_teacher, "groups": cmd_groups, "jobs": cmd_jobs,
            "distill": cmd_distill, "heldout": cmd_heldout, "bench-jobs": cmd_bench_jobs}[args.command](args) or 0


if __name__ == "__main__":
    sys.exit(main())
