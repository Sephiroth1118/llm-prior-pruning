"""Section 6 tables from the run records, on the searches with guided mutation (the first snapshot of the tables). The
paper's tab:spi-main, tab:matrix and tab:matrix-summary use the plain-EA runs (groups *_nogp): tab:spi-main is produced by
this script in mode "plain" (with the LLM as the optimizer as an extra row and Holm over the six methods compared with the
EA at each budget), the matrix and summary by sec6_matrix_plain.py.
(1) the compact comparison on Qwen2.5-1.5B and Llama-3.1-8B, (2) the eight-model template matrix, (3) the summary over
models with pooled and model-level Wilcoxon tests (Holm-corrected).
Relative gain of a method over the EA on one seed = (EA KL - method KL) / EA KL at the same budget.
Reads:  <records>/runs/<group>_s<seed>/results.json (or <group>_rk_s<seed>), groups from groups(): tuned2gp_baseline,
        llm_tuned2gp, tuned2gp_random, qwen_tmpl_own, qwen_tmpl_transfer, qwen15b_tmpl_olmo7b, qwen_classic_gpbo,
        llama_tuned_baseline, llama_tuned_llm, llama_tmpl_own, llama_tmpl_transfer, llama31_tmpl_olmo7b,
        llama_classic_gpbo, <model>_tuned_baseline, <model>_tuned_llm, <model>_tmpl_{own,<source>,humandepth}
Writes: <out_json> and prints the tables.
Usage: python sec6_tables.py <records> <out_json> [plain]"""
import json, math, sys
import numpy as np
from scipy.stats import wilcoxon

from records import records_dir, runs_path, resolve_run

RUNS = None          # <records>/runs, set in main()
PLAIN = False        # mode "plain": tab:spi-main of the paper
MODELS = ["qwen15b", "qwen3b", "qwen7b", "llama32_1b", "llama32_3b", "llama31", "olmo1b", "olmo7b"]
FAMILY = {"qwen15b": "qwen", "qwen3b": "qwen", "qwen7b": "qwen", "llama32_1b": "llama", "llama32_3b": "llama",
          "llama31": "llama", "olmo1b": "olmo", "olmo7b": "olmo"}
SOURCE = {"qwen": "qwen15b", "llama": "llama31", "olmo": "olmo7b"}


def groups_plain(model):
    """Run groups of tab:spi-main (plain EA)."""
    cross = "llama31" if model == "qwen15b" else "qwen15b"
    short = "qwen" if model == "qwen15b" else "llama"
    return {"EA": f"{model}_ea_nogp", "LLM": f"{model}_llm_nogp", "random": f"{model}_random_nogp", "own": f"{model}_own_nogp",
            f"tmpl:{cross}": f"{model}_tmpl_{cross}_nogp", "GP-BO": f"{short}_classic_gpbo", "LLMopt": f"{short}_llmopt_gpt5mini"}


def groups(model):
    if PLAIN:
        return groups_plain(model)
    if model == "qwen15b":
        g = {"EA": "tuned2gp_baseline", "LLM": "llm_tuned2gp", "random": "tuned2gp_random", "own": "qwen_tmpl_own",
             "tmpl:llama31": "qwen_tmpl_transfer", "tmpl:olmo7b": "qwen15b_tmpl_olmo7b", "GP-BO": "qwen_classic_gpbo"}
    elif model == "llama31":
        g = {"EA": "llama_tuned_baseline", "LLM": "llama_tuned_llm", "own": "llama_tmpl_own",
             "tmpl:qwen15b": "llama_tmpl_transfer", "tmpl:olmo7b": "llama31_tmpl_olmo7b", "GP-BO": "llama_classic_gpbo"}
    else:
        g = {"EA": f"{model}_tuned_baseline", "LLM": f"{model}_tuned_llm", "own": f"{model}_tmpl_own"}
        for src in SOURCE.values():
            if src != model:
                g[f"tmpl:{src}"] = f"{model}_tmpl_{src}"
    g["depth"] = f"{model}_tmpl_humandepth"
    if SOURCE[FAMILY[model]] == model:            # a family's source model: its family template is its own template
        g[f"tmpl:{model}"] = g["own"]
    return g


def curve(group, seed):
    p = resolve_run(RUNS, group, seed) / "results.json"
    if not p.exists():
        return None
    r = json.load(open(p))
    if not r.get("final") or len(r["entries"]) < 320:
        return None
    return np.minimum.accumulate([e["kl"] for e in sorted(r["entries"], key=lambda e: e["id"])])


def runs(model, arm):
    g = groups(model).get(arm)
    if not g:
        return {}
    seeds = [0, 1, 2, 4, 5, 6, 7, 8, 9] if model == "qwen15b" else range(10)     # seed 3 is excluded on Qwen2.5-1.5B
    out = {s: curve(g, s) for s in seeds}
    return {s: c for s, c in out.items() if c is not None}


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); m = len(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def wil(d):
    d = [x for x in d if x != 0]
    if len(d) < 2:
        return float("nan")
    return float(wilcoxon(d, alternative="two-sided", method="exact" if len(d) <= 25 else "approx").pvalue)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    global RUNS, PLAIN
    RUNS, OUT = runs_path(records_dir(argv[0])), argv[1]
    PLAIN = len(argv) > 2 and argv[2] == "plain"
    res = {"compact": {}, "matrix": {}, "summary": {}}
    # (1) compact comparison
    for model in ("qwen15b", "llama31"):
        ea = runs(model, "EA")
        target = float(np.mean([c[319] for c in ea.values()]))
        rows = {}
        arms = ("GP-BO", "EA", "random", f"tmpl:{'llama31' if model == 'qwen15b' else 'qwen15b'}", "LLM", "own")
        if PLAIN:
            arms = ("EA", "GP-BO", "LLMopt", "random", f"tmpl:{'llama31' if model == 'qwen15b' else 'qwen15b'}", "LLM", "own")
        for arm in arms:
            rr = runs(model, arm)
            if not rr:
                continue
            row = {"seeds": sorted(rr)}
            for b in (80, 160, 320, 1000):
                if all(len(c) >= b for c in rr.values()):
                    row[f"m{b}"] = float(np.mean([c[b - 1] for c in rr.values()]))
                    common = [s for s in rr if s in ea and len(ea[s]) >= b]
                    if arm != "EA" and common:
                        d = [ea[s][b - 1] - rr[s][b - 1] for s in common]
                        row[f"w{b}"] = [int(sum(x > 0 for x in d)), len(d)]
                        row[f"p{b}"] = wil(d)
            hits = [next((t + 1 for t, v in enumerate(c[:320]) if v <= target), None) for c in rr.values()]
            reached = sorted(h for h in hits if h)
            row["reach"] = [len(reached), len(hits)]
            n = len(hits)
            ordered = reached + [math.inf] * (n - len(reached))
            med = (ordered[(n - 1) // 2] + ordered[n // 2]) / 2
            row["median_evals"] = None if math.isinf(med) else med
            rows[arm] = row
        res["compact"][model] = {"target": target, "rows": rows}
    # Holm over the methods compared with the EA in the table, separately for each model and budget (Section 4.1)
    for model, v in res["compact"].items():
        for b in (80, 160, 320, 1000):
            keys = [a for a, r in v["rows"].items() if f"p{b}" in r]
            for a, q in zip(keys, holm([v["rows"][a][f"p{b}"] for a in keys]) if keys else []):
                v["rows"][a][f"q{b}"] = q
    if PLAIN:                                 # tab:matrix and tab:matrix-summary: sec6_matrix_plain.py
        json.dump(res, open(OUT, "w"), indent=1)
        for model, v in res["compact"].items():
            print(f"== compact {model} target {v['target']:.3f}")
            for arm, r in v["rows"].items():
                cells = " ".join(f"{b}:{r[f'm{b}']:.3f}" + (f"({r[f'w{b}'][0]}/{r[f'w{b}'][1]},p={r[f'p{b}']:.3f},q={r[f'q{b}']:.3f})" if f"w{b}" in r else "")
                                 for b in (80, 160, 320) if f"m{b}" in r)
                print(f"  {arm:14s} n={len(r['seeds'])} {cells} reach={r['reach']} med={r['median_evals']}")
        return
    # (2) matrix
    ARMS = ["EA", "LLM", "tmpl:qwen15b", "tmpl:llama31", "tmpl:olmo7b", "depth", "own"]
    for model in MODELS:
        ea = runs(model, "EA")
        seeds = sorted(ea) if model not in ("qwen15b", "llama31") else [0, 1, 2, 3] if model == "llama31" else [0, 1, 2, 4]
        cells = {}
        for arm in ARMS:
            rr = runs(model, arm)
            use = [s for s in seeds if s in rr]
            if len(use) < len(seeds):
                use = [s for s in rr if s in ea]
            if not use:
                continue
            cells[arm] = {"seeds": use, "m80": float(np.mean([rr[s][79] for s in use])), "m320": float(np.mean([rr[s][319] for s in use]))}
            if arm != "EA":
                cells[arm]["w320"] = int(sum(rr[s][319] < ea[s][319] for s in use))
        res["matrix"][model] = cells
    # (3) summary over models: relative gains, pooled over model-seed pairs and model level
    CLASSES = {"LLM template": lambda m: ["LLM"],
               "Same-family template": lambda m: [f"tmpl:{SOURCE[FAMILY[m]]}"] if SOURCE[FAMILY[m]] != m else [],
               "Cross-family template": lambda m: [f"tmpl:{s}" for s in SOURCE.values() if FAMILY[s] != FAMILY[m]],
               "Hand-designed": lambda m: ["depth"]}
    tests = []
    for cls, arms_of in CLASSES.items():
        out = {}
        for b in (80, 320):
            pairs, units = [], []
            for model in MODELS:
                ea = runs(model, "EA")
                for arm in arms_of(model):
                    rr = runs(model, arm)
                    d = [(ea[s][b - 1] - rr[s][b - 1]) / ea[s][b - 1] for s in rr if s in ea]
                    if d:
                        pairs += d; units.append(float(np.mean(d)))
            out[b] = {"units": len(units), "units_pos": int(sum(u > 0 for u in units)), "mean_unit": float(np.mean(units)),
                      "pairs": len(pairs), "pairs_pos": int(sum(x > 0 for x in pairs)),
                      "p_pooled": wil(pairs), "p_units": wil(units) if len(units) >= 6 else float("nan")}
            tests.append((cls, b))
        res["summary"][cls] = out
    adj = holm([res["summary"][c][b]["p_pooled"] for c, b in tests])
    for (c, b), q in zip(tests, adj):
        res["summary"][c][b]["q_pooled"] = q
    json.dump(res, open(OUT, "w"), indent=1)

    for model, v in res["compact"].items():
        print(f"== compact {model} target {v['target']:.3f}")
        for arm, r in v["rows"].items():
            cells = " ".join(f"{b}:{r[f'm{b}']:.3f}" + (f"({r[f'w{b}'][0]}/{r[f'w{b}'][1]},p={r[f'p{b}']:.3f},q={r[f'q{b}']:.3f})" if f"w{b}" in r else "")
                             for b in (80, 160, 320, 1000) if f"m{b}" in r)
            print(f"  {arm:14s} n={len(r['seeds'])} {cells} reach={r['reach']} med={r['median_evals']}")
    print("== matrix")
    for model, cells in res["matrix"].items():
        print(f"  {model:11s} " + " | ".join(f"{a}:{c['m80']:.3f}/{c['m320']:.3f}" + (f"({c['w320']})" if 'w320' in c else "") + f"[{len(c['seeds'])}]" for a, c in cells.items()))
    print("== summary")
    for c, v in res["summary"].items():
        for b, s in v.items():
            print(f"  {c:22s} @{b}: units {s['units_pos']}/{s['units']} mean {100*s['mean_unit']:5.1f}% p_units={s['p_units']:.4f} | pairs {s['pairs_pos']}/{s['pairs']} p={s['p_pooled']:.2e} q={s['q_pooled']:.2e}")


if __name__ == "__main__":
    main()
