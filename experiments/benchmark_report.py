#!/usr/bin/env python3
"""Aggregate downstream benchmark results of the search champions (partial results are fine).

Reads the job files <work>/benchmark/jobs/*.json and the benchmark_results.json of each job, and writes
<work>/reports/benchmark/{report.md, metrics.json}; metrics.json is read by analysis/fig_kl_ppl.py and
analysis/sec6_downstream_seeds.py.

    python benchmark_report.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
from common import WORK, read_json, resolve, save_json   # noqa: E402

TASKS = ["sciq", "piqa", "winogrande", "arc_easy", "arc_challenge", "hellaswag", "boolq", "mmlu"]
ARMS = {"qwen15b": [("llm_orig", "LLM template"), ("random_orig", "random template"), ("baseline_fixed", "plain EA")],
        "llama31": [("llama_llm", "LLM template"), ("llama_random", "random template"), ("llama_baseline", "plain EA"),
                    ("llama_tuned_llm", "LLM template + tuned EA + GP"), ("llama_tuned_baseline", "tuned EA + GP")]}
PAIRS = {"qwen15b": [("llm_orig", "random_orig"), ("llm_orig", "baseline_fixed"), ("random_orig", "baseline_fixed")],
         "llama31": [("llama_llm", "llama_random"), ("llama_llm", "llama_baseline"), ("llama_random", "llama_baseline"),
                     ("llama_tuned_llm", "llama_tuned_baseline"), ("llama_tuned_llm", "llama_llm"), ("llama_tuned_baseline", "llama_baseline")]}
TITLE = {"qwen15b": "Qwen2.5-1.5B (champions of 1000 evaluations)", "llama31": "Llama-3.1-8B (champions of 320 evaluations)"}


def load():
    rows = {}
    for path in sorted((WORK / "benchmark" / "jobs").glob("*.json")):
        job = read_json(path)
        res = resolve(job["out_dir"]) / "benchmark_results.json"
        if not res.exists():
            continue
        want = "Dense" if job.get("dense") else "Compressed"
        row = next((r for r in read_json(res).get("rows", []) if r.get("Model") == want), None)
        if row is None or row.get("Tasks Complete") != "8/8":
            continue
        rows[(job["model"], job["group"], job["seed"])] = {
            "ppl": float(row["PPL"]), "avg": float(row["Avg"]), "params": row["Params"],
            **{t: float(row[t]) for t in TASKS}, "search_kl": job.get("search_kl")}
    return rows


def paired(rows, model, a, b, key, lower_better):
    seeds = sorted(s for (m, g, s) in rows if m == model and g == a and (m, b, s) in rows)
    d = [(rows[(model, b, s)][key] - rows[(model, a, s)][key]) * (1 if lower_better else -1) for s in seeds]
    out = {"n": len(d), "mean": float(np.mean(d)) if d else None, "wins": int(sum(x > 0 for x in d)), "deltas": d}
    if len(d) >= 5 and any(d):
        from scipy.stats import wilcoxon
        out["p"] = float(wilcoxon(d).pvalue)
    return out


def main():
    rows = load()
    lines = ["# Downstream benchmark of the champions", "",
             "Each search run contributes only its allocation with the lowest search KL; nothing is selected by benchmark results. Protocol of the controlled comparison: FineWeb-Edu PPL (163,840 tokens, sequence length 4096, second half of the dataset, disjoint from the search documents) and 8 lm-eval tasks; physically compact models, bf16, SDPA.", ""]
    summary = {}
    for model, arms in ARMS.items():
        lines += [f"## {TITLE[model]}", ""]
        dense = rows.get((model, "dense", None))
        lines += ["| group | n | PPL mean (lower is better) | PPL median | " + " | ".join(TASKS) + " | Avg mean (higher is better) | Avg std |",
                  "|---|---:|---:|---:|" + "---:|" * len(TASKS) + "---:|---:|"]
        if dense:
            lines.append(f"| dense model | 1 | {dense['ppl']:.3f} | {dense['ppl']:.3f} | " +
                         " | ".join(f"{dense[t]:.4f}" for t in TASKS) + f" | {dense['avg']:.4f} | - |")
        for g, label in arms:
            got = [v for (m, gg, s), v in rows.items() if m == model and gg == g]
            if not got:
                lines.append(f"| {label} | 0 | pending | | " + " | " * len(TASKS) + " | |")
                continue
            ppl = [v["ppl"] for v in got]
            avg = [v["avg"] for v in got]
            lines.append(f"| {label} | {len(got)} | {np.mean(ppl):.3f} | {np.median(ppl):.3f} | " +
                         " | ".join(f"{np.mean([v[t] for v in got]):.4f}" for t in TASKS) +
                         f" | {np.mean(avg):.4f} | {np.std(avg, ddof=1) if len(avg) > 1 else 0:.4f} |")
            summary[f"{model}/{g}"] = {"n": len(got), "ppl_mean": float(np.mean(ppl)), "avg_mean": float(np.mean(avg))}
        lines += ["", "Paired comparison (same seed; a positive gain means the first is better):", "",
                  "| comparison | metric | n | mean gain | wins | per seed | Wilcoxon p |", "|---|---|---:|---:|---:|---|---:|"]
        labels = dict(arms)
        for a, b in PAIRS[model]:
            la, lb = labels[a], labels[b]
            for key, lower, label in (("avg", False, "Avg"), ("ppl", True, "PPL")):
                p = paired(rows, model, a, b, key, lower)
                if p["n"]:
                    lines.append(f"| {la} vs {lb} | {label} | {p['n']} | {p['mean']:+.4f} | {p['wins']}/{p['n']} | " +
                                 ", ".join(f"{x:+.3f}" for x in p["deltas"]) + f" | {p.get('p', float('nan')):.3f} |")
        lines += ["", "| group/seed | search KL | PPL | Avg | parameters |", "|---|---:|---:|---:|---|"]
        for (m, g, s), v in sorted(rows.items(), key=lambda kv: (kv[0][1], -1 if kv[0][2] is None else kv[0][2])):
            if m == model and g != "dense":
                lines.append(f"| {g}/s{s} | {v['search_kl']:.4f} | {v['ppl']:.3f} | {v['avg']:.4f} | {v['params']} |")
        lines.append("")
    out = WORK / "reports" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    save_json(out / "metrics.json", {"rows": {f"{m}/{g}/s{s}": v for (m, g, s), v in rows.items()}, "summary": summary})
    print(f"{len(rows)} benchmark rows aggregated -> {out / 'report.md'}")


if __name__ == "__main__":
    main()
