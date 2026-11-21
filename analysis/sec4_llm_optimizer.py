"""Section 4 numbers for the LLM as the optimizer: best-so-far KL of the LLM optimizers, the anonymized gpt-5-mini, the
conventional optimizers, the EA and our method; paired comparisons (exact Wilcoxon, Holm); feasibility steps and cost.
The EA and our method use unguided mutation (no-GP runs); the EA with guided mutation is the row "EA+GM".
Qwen2.5-1.5B: seeds 0,1,2,4-9; Llama-3.1-8B: seeds 0-3 for every row.
Feeds Tables tab:llm-opt and tab:llm-feasibility (Section 4.3 text).
Reads:  <records>/runs/<group>_s<seed>/results.json (or <group>_rk_s<seed>), groups (see GROUPS)
        qwen15b_{llm,ea}_nogp, tuned2gp_baseline, qwen_classic_{random,tpe,gpbo,de,pso,ga,cmaes},
        qwen_llmopt_{gpt5mini,gpt5mini_anon,luna,deepseek}, and the llama31_*/llama_* counterparts.
Writes: <out_json> (summary, pairs, feasibility, curves) and prints the tables.
Usage: python sec4_llm_optimizer.py <records> <out_json>"""
import json, math, sys, itertools

from records import records_dir, runs_path, resolve_run as resolve

CLASSIC = {"Random": "random", "TPE": "tpe", "GP-BO": "gpbo", "DE": "de", "PSO": "pso", "GA": "ga", "CMA-ES": "cmaes"}
GROUPS = {
    "qwen15b": {"Ours": "qwen15b_llm_nogp", "EA": "qwen15b_ea_nogp", "EA+GM": "tuned2gp_baseline", **{k: f"qwen_classic_{v}" for k, v in CLASSIC.items()},
                "gpt-5-mini": "qwen_llmopt_gpt5mini", "gpt-5-mini anonymized": "qwen_llmopt_gpt5mini_anon",
                "gpt-5.6-luna": "qwen_llmopt_luna", "DeepSeek-V4-Flash": "qwen_llmopt_deepseek"},
    "llama31": {"Ours": "llama31_llm_nogp", "EA": "llama31_ea_nogp", "EA+GM": "llama_tuned_baseline", **{k: f"llama_classic_{v}" for k, v in CLASSIC.items()},
                "gpt-5-mini": "llama_llmopt_gpt5mini", "gpt-5-mini anonymized": "llama_llmopt_gpt5mini_anon",
                "gpt-5.6-luna": "llama_llmopt_luna", "DeepSeek-V4-Flash": "llama_llmopt_deepseek"},
}
SEEDS = {"qwen15b": [0, 1, 2, 4, 5, 6, 7, 8, 9], "llama31": [0, 1, 2, 3]}
LLM_ARMS = ("gpt-5-mini", "gpt-5-mini anonymized", "gpt-5.6-luna", "DeepSeek-V4-Flash")
CONVENTIONAL = ("Ours", "EA", "EA+GM", *CLASSIC)
CHECK = [80, 160, 320]
PRICES = {"gpt-5-mini": (0.25, 0.025, 2.00), "gpt-5.6-luna": (0.20, 0.02, 1.20), "deepseek-v4-flash": (0.30, 0.006, 1.20)}  # USD per 1M tokens: input, cached input, output


def curve(path):
    r = json.load(open(path))
    assert r.get("final") or r.get("phase") == "complete", path
    best, values = math.inf, []
    for e in r["entries"]:
        assert not e.get("error") and 0.500 <= e["flops_reduction"] <= 0.502 + 1e-12, path
        best = min(best, float(e["kl"])); values.append(best)
    return values[:320], r


def wilcoxon_exact(d):
    d = [x for x in d if x != 0]; n = len(d)
    if n == 0: return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i])); ranks = [0.0] * n; i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]): j += 1
        for k in range(i, j + 1): ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    w = sum(r for r, x in zip(ranks, d) if x > 0)
    dist = [sum(r for r, s in zip(ranks, signs) if s) for signs in itertools.product((0, 1), repeat=n)]
    lo = sum(v <= w + 1e-9 for v in dist) / len(dist); hi = sum(v >= w - 1e-9 for v in dist) / len(dist)
    return min(1.0, 2 * min(lo, hi))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj


def cost(model, calls):
    pin, pc, pout = PRICES[model]; total = 0.0
    for c in calls:
        u = c.get("usage")
        if not u: continue
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get("prompt_cache_hit_tokens") or 0
        total += ((u["prompt_tokens"] - cached) * pin + cached * pc + u["completion_tokens"] * pout) / 1e6
    return total


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    runs, out = runs_path(records_dir(argv[0])), argv[1]
    res, records = {}, {}
    for model, groups in GROUPS.items():
        res[model], records[model] = {}, {}
        for name, group in groups.items():
            seeds = [s for s in SEEDS[model] if (resolve(runs, group, s) / "results.json").exists()]
            loaded = {s: curve(resolve(runs, group, s) / "results.json") for s in seeds}
            res[model][name] = {s: c for s, (c, _) in loaded.items()}
            records[model][name] = {s: r for s, (_, r) in loaded.items()}

    summary = {}
    for model, methods in res.items():
        for name, cur in methods.items():
            row = {"seeds": sorted(cur)}
            for b in CHECK:
                vals = [c[b - 1] for c in cur.values()]
                row[f"mean@{b}"] = sum(vals) / len(vals)
                row[f"sd@{b}"] = (sum((v - row[f"mean@{b}"]) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
            row["drop160to320"] = row["mean@160"] - row["mean@320"]
            for ref in ("EA", "gpt-5-mini"):
                if ref == name or ref not in methods: continue
                common = [s for s in cur if s in methods[ref]]
                for b in CHECK:
                    d = [methods[ref][s][b - 1] - cur[s][b - 1] for s in common]          # > 0: this row better than ref
                    row[f"vs {ref}@{b}"] = {"wins": f"{sum(x > 0 for x in d)}/{len(d)}", "p": wilcoxon_exact(d)}
            summary[f"{model}/{name}"] = row

    # gpt-5-mini and its anonymized version against every conventional row, Holm within each (arm, budget) family
    pairs = {}
    for model, methods in res.items():
        for arm in ("gpt-5-mini", "gpt-5-mini anonymized"):
            if arm not in methods: continue
            for b in CHECK:
                names, ps, wins = [], [], []
                for other in CONVENTIONAL:
                    common = [s for s in methods[arm] if s in methods[other]]
                    d = [methods[other][s][b - 1] - methods[arm][s][b - 1] for s in common]   # > 0: LLM arm better
                    names.append(other); ps.append(wilcoxon_exact(d)); wins.append(f"{sum(x > 0 for x in d)}/{len(d)}")
                pairs[f"{model}/{arm}@{b}"] = {n: {"llm_wins": w, "p": p, "p_holm": h} for n, w, p, h in zip(names, wins, ps, holm(ps))}

    feas = {}
    for model, recs in records.items():
        for name in LLM_ARMS:
            if name not in recs: continue
            rs = list(recs[name].values()); summ = [r["llm_summary"] for r in rs]
            mean = lambda k: sum(s[k] for s in summ if s.get(k) is not None) / max(1, sum(s.get(k) is not None for s in summ))
            costs = [cost(s["model"], r["llm_calls"]) for s, r in zip(summ, rs)]
            feas[f"{model}/{name}"] = {"runs": len(rs), **{k: mean(k) for k in ("raw_feasible_share", "feasible_after_revision_share",
                                       "repaired_share", "mean_repair_changed", "format_fallback_share", "duplicate_share", "requests")},
                                       "cost_per_search": sum(costs) / len(costs), "cost_total": sum(costs),
                                       "wall_min": sum(r["stats"]["wall_seconds"] for r in rs) / len(rs) / 60}

    json.dump({"summary": summary, "pairs": pairs, "feasibility": feas,
               "curves": {m: {k: {str(s): c for s, c in v.items()} for k, v in d.items()} for m, d in res.items()}}, open(out, "w"))
    for model in GROUPS:
        print(f"\n{model}: mean best KL @80/160/320 (sd@320) | vs EA wins @80/160/320 (p@320) | drop 160->320")
        for name in GROUPS[model]:
            r = summary.get(f"{model}/{name}")
            if not r: continue
            ea = " ".join(r[f"vs EA@{b}"]["wins"] for b in CHECK) + f" (p {r['vs EA@320']['p']:.3f})" if "vs EA@80" in r else "-"
            print(f"  {name:22s} n={len(r['seeds'])} " + " / ".join(f"{r[f'mean@{b}']:.3f}" for b in CHECK)
                  + f" ({r['sd@320']:.3f}) | {ea} | {r['drop160to320']:.3f}")
    print("\npaired, LLM arm vs row: wins, p, Holm p")
    for k, v in pairs.items():
        print(" ", k, "  ".join(f"{n} {x['llm_wins']} {x['p']:.3f}/{x['p_holm']:.3f}" for n, x in v.items()))
    print("\nfeasibility and cost")
    for k, v in feas.items():
        print(" ", k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
