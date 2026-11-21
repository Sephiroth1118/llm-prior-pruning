"""Section 4 numbers, first version of the optimizer comparison: best-so-far KL of the EA (guided mutation) and the
conventional optimizers (Llama-3.1-8B, seven rows), mean/sd at 80, 160, 320, 1000 evaluations, wins and exact Wilcoxon
against the EA. The final Table tab:llm-opt and Fig. fig:optimizers (with the LLM optimizers and the plain EA) come from
sec4_llm_optimizer.py and fig_optimizers_gain.py.
Reads:  <records>/runs/<group>_s<seed>/results.json for tuned2gp_baseline (Qwen2.5-1.5B) and llama_tuned_baseline,
        llama_classic_{random,tpe,gpbo,de,pso,ga} (Llama-3.1-8B); every seed found.
Writes: <out_json> (summary and curves) and prints the summary.
Usage: python sec4_numbers.py <records> <out_json>"""
import json, math, sys, itertools

from records import records_dir, runs_path

GROUPS = {
    "qwen15b": {"EA": "tuned2gp_baseline"},
    "llama31": {"EA": "llama_tuned_baseline", "Random": "llama_classic_random", "TPE": "llama_classic_tpe",
                "GP-BO": "llama_classic_gpbo", "DE": "llama_classic_de", "PSO": "llama_classic_pso",
                "GA": "llama_classic_ga"},
}
CHECK = [80, 160, 320, 1000]

def curve(path):
    r = json.load(open(path))
    assert r.get("final") or r.get("phase") == "complete", path
    best, out = math.inf, []
    for e in sorted(r["entries"], key=lambda e: e["id"]):
        assert not e.get("error") and 0.500 <= e["flops_reduction"] <= 0.502 + 1e-12, (path, e["id"])
        best = min(best, float(e["kl"])); out.append(best)
    return out

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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    runs, out = runs_path(records_dir(argv[0])), argv[1]
    res = {}
    for model, groups in GROUPS.items():
        res[model] = {}
        for name, group in groups.items():
            seeds = sorted(int(p.name.rsplit("_s", 1)[1]) for p in runs.glob(group + "_s*") if p.name.rsplit("_s", 1)[1].isdigit())
            curves = {s: curve(runs / f"{group}_s{s}" / "results.json") for s in seeds}
            res[model][name] = {"group": group, "seeds": seeds, "budget": min(len(c) for c in curves.values()),
                                "curves": {str(s): c for s, c in curves.items()}}

    summary = {}
    for model, methods in res.items():
        ea = methods["EA"]
        for name, m in methods.items():
            common = [s for s in m["seeds"] if s in ea["seeds"]]
            row = {"seeds": m["seeds"], "budget": m["budget"]}
            for b in CHECK:
                if b > m["budget"]: continue
                vals = [m["curves"][str(s)][b - 1] for s in m["seeds"]]
                row[f"mean@{b}"] = sum(vals) / len(vals)
                row[f"sd@{b}"] = (sum((v - row[f'mean@{b}']) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
                if name != "EA":
                    d = [ea["curves"][str(s)][b - 1] - m["curves"][str(s)][b - 1] for s in common]   # >0: method better
                    row[f"wins@{b}"] = f"{sum(x > 0 for x in d)}/{len(d)}"
                    row[f"gain@{b}"] = sum(d) / len(d)
                    row[f"p@{b}"] = wilcoxon_exact(d)
                else:
                    sub = [m["curves"][str(s)][b - 1] for s in m["seeds"] if s < 4]
                    row[f"mean0-3@{b}"] = sum(sub) / len(sub)
            summary[f"{model}/{name}"] = row
    json.dump({"summary": summary, "curves": {m: {k: v["curves"] for k, v in d.items()} for m, d in res.items()}},
              open(out, "w"))
    for k, v in summary.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
