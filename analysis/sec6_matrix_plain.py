"""Tables tab:matrix and tab:matrix-summary from the searches with the plain EA (no guided mutation), with the re-ask
reruns. Table tab:matrix uses seeds 0-3 (Qwen2.5-1.5B: 0, 1, 2, 4); the summary uses all available seeds and then the four
seeds; relative gain per seed = (EA - template) / EA; Wilcoxon over target-seed pairs; Holm over the 8 tests.
Also used as a library by fig_matrix.py (same group definitions) and sec6_sources_text.py.
Reads:  <records>/runs/<model>_{ea,llm,humandepth,own}_nogp_s<seed>/results.json and <model>_tmpl_<source>_nogp
        (source in qwen15b, llama31, olmo7b; or the _rk_s<seed> reruns) for the eight models of MODELS
Writes: nothing; prints the LaTeX rows of tab:matrix and the summary statistics.
Usage: python sec6_matrix_plain.py <records>"""
import json, sys
import numpy as np
from scipy.stats import wilcoxon
from records import records_dir, runs_path, resolve_run

RUNS = None          # <records>/runs, set by setup()
MODELS = ["qwen15b", "qwen3b", "qwen7b", "llama32_1b", "llama32_3b", "llama31", "olmo1b", "olmo7b"]
NAME = {"qwen15b": "Qwen2.5-1.5B", "qwen3b": "Qwen2.5-3B", "qwen7b": "Qwen2.5-7B", "llama32_1b": "Llama-3.2-1B",
        "llama32_3b": "Llama-3.2-3B", "llama31": "Llama-3.1-8B", "olmo1b": "OLMo-1B", "olmo7b": "OLMo-7B"}
FAMILY = {"qwen15b": "qwen", "qwen3b": "qwen", "qwen7b": "qwen", "llama32_1b": "llama", "llama32_3b": "llama",
          "llama31": "llama", "olmo1b": "olmo", "olmo7b": "olmo"}
SOURCE = {"qwen": "qwen15b", "llama": "llama31", "olmo": "olmo7b"}
def setup(records):
    """Point the module at a records directory (used by main and by importing scripts)."""
    global RUNS
    RUNS = runs_path(records_dir(records))


def groups(m):
    g = {"EA": f"{m}_ea_nogp", "LLM": f"{m}_llm_nogp", "depth": f"{m}_humandepth_nogp", "own": f"{m}_own_nogp"}
    for src in SOURCE.values():
        if src != m:
            g[f"tmpl:{src}"] = f"{m}_tmpl_{src}_nogp"
    return g
def curve(group, seed):
    p = resolve_run(RUNS, group, seed) / "results.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    if not r.get("final") or len(r["entries"]) < 320:
        return None
    return np.minimum.accumulate([e["kl"] for e in sorted(r["entries"], key=lambda e: e["id"])])
def runs(m, arm, seeds=range(10)):
    g = groups(m).get(arm)
    if not g:
        return {}
    out = {s: curve(g, s) for s in seeds}
    return {s: c for s, c in out.items() if c is not None}
def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); m = len(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()
def wil(d):
    d = [x for x in d if x != 0]
    return float(wilcoxon(d, alternative="two-sided", method="exact" if len(d) <= 25 else "approx").pvalue)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    setup(argv[0])
    print("== Table 7 (4 seeds)")
    COLS = ["EA", "LLM", "tmpl:qwen15b", "tmpl:llama31", "tmpl:olmo7b", "depth", "own"]
    for m in MODELS:
        seeds = [0, 1, 2, 4] if m == "qwen15b" else [0, 1, 2, 3]
        cell = {}
        for a in COLS:
            rr = runs(m, a, seeds)
            if rr:
                assert len(rr) == 4, (m, a, sorted(rr))
                cell[a] = (float(np.mean([c[79] for c in rr.values()])), float(np.mean([c[319] for c in rr.values()])))
        pool = [a for a in cell if a not in ("EA", "own")]
        b80, b320 = min(cell[a][0] for a in pool), min(cell[a][1] for a in pool)
        out = []
        for a in COLS:
            if a not in cell:
                out.append("--"); continue
            x, y = cell[a]
            sx = f"\\textbf{{{x:.3f}}}" if a in pool and x == b80 else f"{x:.3f}"
            sy = f"\\textbf{{{y:.3f}}}" if a in pool and y == b320 else f"{y:.3f}"
            s = f"{sx} / {sy}"
            if a == f"tmpl:{SOURCE[FAMILY[m]]}":
                s = f"\\emph{{{s}}}"
            out.append(s)
        print(f"{NAME[m]} & " + " & ".join(out) + " \\\\")
    CLASSES = {"LLM": lambda m: ["LLM"], "Same family": lambda m: [f"tmpl:{SOURCE[FAMILY[m]]}"] if SOURCE[FAMILY[m]] != m else [],
               "Cross family": lambda m: [f"tmpl:{s}" for s in SOURCE.values() if FAMILY[s] != FAMILY[m]], "Hand-designed": lambda m: ["depth"]}
    for label, seeds_of in (("all available seeds", lambda m: range(10)), ("four seeds", lambda m: [0, 1, 2, 4] if m == "qwen15b" else [0, 1, 2, 3])):
        print("== summary,", label)
        tests, res = [], {}
        for c, arms in CLASSES.items():
            for b in (80, 320):
                pairs, units, counts = [], [], []
                for m in MODELS:
                    ea = runs(m, "EA", seeds_of(m))
                    for a in arms(m):
                        rr = runs(m, a, seeds_of(m))
                        d = [(ea[s][b - 1] - rr[s][b - 1]) / ea[s][b - 1] for s in rr if s in ea]
                        if d:
                            pairs += d; units.append(float(np.mean(d))); counts.append(len(d))
                res[(c, b)] = dict(units=units, pairs=pairs, counts=counts, p=wil(pairs))
                tests.append((c, b))
        q = holm([res[t]["p"] for t in tests])
        for t, v in zip(tests, q):
            r = res[t]
            print(f"  {t[0]:13s} @{t[1]:<4d} wins {sum(u > 0 for u in r['units'])}/{len(r['units'])} gain {100 * np.mean(r['units']):.1f}% pooled-mean {100 * np.mean(r['pairs']):.1f}% pairs {sum(x > 0 for x in r['pairs'])}/{len(r['pairs'])} p={r['p']:.5f} holm={v:.5f} seeds {r['counts']}")


if __name__ == "__main__":
    main()
