"""Section 7 analyses from the run records (CPU only, read-only).

(A) Prior-target match: for every target model and template source, the distance between the macro-structure of the
    injected anchors and the target's converged macro-structure (mean over the EA's champions at 320 evaluations,
    so that no template enters the reference), against the paired relative gain over the EA at 80 and 320 evaluations.
(B) Two scales on Qwen2.5-1.5B (EA, LLM template, random template; 9 seeds x 1000 evaluations): PCA of the evaluated
    allocations, how much of the divergence the macro-structure explains globally and among good allocations, the
    macro distance of the best-so-far allocation to the converged macro-structure, and population diversity.
(C) (plain modes only) Llama-3.1-8B: best-so-far macro distance and population diversity of the EA and the LLM template.

Modes (third argument):
  (none)     searches with guided mutation (earlier snapshot)
  plain      searches with the plain EA (no guided mutation), as in Tables tab:spi-main, tab:matrix and tab:matrix-summary;
             match points use the seeds of tab:matrix, the reference uses all available EA seeds
  plain-all  as plain, but the match points use all available seeds of every combination (the mode of the paper's
             Fig. fig_match and fig_two_scales)
Reads:  <records>/runs/<group>_s<seed>/results.json (or <group>_rk_s<seed>); groups from groups(): in plain modes
        <model>_{ea,llm,own,humandepth}_nogp, <model>_tmpl_<source>_nogp, qwen15b_random_nogp for the eight models of SPECS;
        <records>/flops/<model>.json for the eight models
Writes: <out.json> (keys A, B, C) and prints a summary. fig_match.py and fig_two_scales.py draw from the output;
        sec7_robustness.py imports the helper functions of this module.
Usage: python sec7_analysis.py <records> <out.json> [plain|plain-all]"""
import json, sys
import numpy as np

from records import records_dir, runs_path, flops_table_path, resolve_run

PLAIN = False       # plain-EA searches (set by setup)
ALL_SEEDS = False   # match points from all available seeds (set by setup)
RUNS = None         # <records>/runs (set by setup)
RECORDS = None      # records directory (set by setup)
FR = {}             # model -> (removed fraction per module and level, cell index per module), filled by setup
SEEDS7 = lambda m: [0, 1, 2, 4] if m == "qwen15b" else [0, 1, 2, 3]   # seeds of Table tab:matrix
SPECS = {  # hidden, intermediate, heads, kv_heads, head_dim
    "qwen15b": (1536, 8960, 12, 2, 128), "qwen3b": (2048, 11008, 16, 2, 128), "qwen7b": (3584, 18944, 28, 4, 128),
    "llama32_1b": (2048, 8192, 32, 8, 64), "llama32_3b": (3072, 8192, 24, 8, 128), "llama31": (4096, 14336, 32, 8, 128),
    "olmo1b": (2048, 8192, 16, 16, 128), "olmo7b": (4096, 11008, 32, 32, 128)}
FAMILY = {"qwen15b": "qwen", "qwen3b": "qwen", "qwen7b": "qwen", "llama32_1b": "llama", "llama32_3b": "llama",
          "llama31": "llama", "olmo1b": "olmo", "olmo7b": "olmo"}
MODELS = list(SPECS)


def fractions(model):
    h, m, H, G, hd = SPECS[model]
    t = json.load(open(flops_table_path(RECORDS, model)))
    attn = 2 * h * hd * (np.asarray(t["heads"], float) + np.asarray(t["kv_groups"], float))
    mlp = 3 * h * np.asarray(t["mlp_dims"], float)
    costs = np.concatenate([attn, mlp])                       # (2L, levels)
    L = attn.shape[0]
    cells = np.tile(np.minimum(np.arange(L) * 4 // L, 3), 2); cells[L:] += 4
    return (costs[:, :1] - costs) / costs[:, :1], cells


def setup(records, plain=False, all_seeds=False):
    """Select the records directory and the mode (see the docstring), and load the FLOPs tables."""
    global PLAIN, ALL_SEEDS, RUNS, RECORDS, FR
    RECORDS = records_dir(records)
    RUNS = runs_path(RECORDS)
    PLAIN, ALL_SEEDS = plain, all_seeds
    FR = {m: fractions(m) for m in MODELS}


def module_fracs(model, policy):
    frac, _ = FR[model]
    return frac[np.arange(frac.shape[0]), policy]


def macro(model, policy):
    _, cells = FR[model]
    f = module_fracs(model, policy)
    return np.array([f[cells == c].mean() for c in range(8)])


def groups(model):
    if PLAIN:
        g = {"EA": f"{model}_ea_nogp", "LLM": f"{model}_llm_nogp", "own": f"{model}_own_nogp", "depth": f"{model}_humandepth_nogp"}
        for src in ("qwen15b", "llama31", "olmo7b"):
            if src != model:
                g[f"tmpl:{src}"] = f"{model}_tmpl_{src}_nogp"
        if model == "qwen15b":
            g["random"] = "qwen15b_random_nogp"
        return g
    if model == "qwen15b":
        g = {"EA": "tuned2gp_baseline", "LLM": "llm_tuned2gp", "random": "tuned2gp_random", "own": "qwen_tmpl_own",
             "tmpl:llama31": "qwen_tmpl_transfer", "tmpl:olmo7b": "qwen15b_tmpl_olmo7b"}
    elif model == "llama31":
        g = {"EA": "llama_tuned_baseline", "LLM": "llama_tuned_llm", "own": "llama_tmpl_own",
             "tmpl:qwen15b": "llama_tmpl_transfer", "tmpl:olmo7b": "llama31_tmpl_olmo7b"}
    else:
        g = {"EA": f"{model}_tuned_baseline", "LLM": f"{model}_tuned_llm", "own": f"{model}_tmpl_own"}
        for src in ("qwen15b", "llama31", "olmo7b"):
            g[f"tmpl:{src}"] = f"{model}_tmpl_own" if src == model else f"{model}_tmpl_{src}"
    g["depth"] = f"{model}_tmpl_humandepth"
    return g


def load(group, seed):
    p = resolve_run(RUNS, group, seed) / "results.json"
    if not p.exists():
        return None
    r = json.load(open(p))
    if not r.get("final") or len(r["entries"]) < 320:
        return None
    return sorted(r["entries"], key=lambda e: e["id"])


def best_curve(entries):
    return np.minimum.accumulate([e["kl"] for e in entries])


def spearman(x, y):
    def rk(v):
        v = np.asarray(v); o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r
    return float(np.corrcoef(rk(x), rk(y))[0, 1])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    setup(argv[0], plain=len(argv) > 2 and argv[2].startswith("plain"), all_seeds=len(argv) > 2 and argv[2] == "plain-all")
    OUT = argv[1]
    res = {"A": {}, "B": {}}
    # ---------------------------------------------------------------- (A) prior-target match
    points = []
    for model in MODELS:
        g = groups(model)
        ea = {s: load(g["EA"], s) for s in range(10)}
        ea = {s: e for s, e in ea.items() if e}
        target = np.mean([macro(model, min(e[:320], key=lambda x: x["kl"])["policy"]) for e in ea.values()], axis=0)
        res["A"][model] = {"target_macro": target.round(4).tolist(), "ea_seeds": sorted(ea)}
        for src, group in g.items():
            if src == "EA" or (src.startswith("tmpl:") and src[5:] == model):
                continue                                            # the family source's own template is "own"
            rows = []
            for s, e in ea.items():
                if PLAIN and not ALL_SEEDS and s not in SEEDS7(model):
                    continue
                x = load(group, s)
                if not x:
                    continue
                anchors = [a for a in x if a.get("phase") == "anchor"]
                if not anchors:
                    continue
                d = float(np.mean([np.abs(macro(model, a["policy"]) - target).mean() for a in anchors]))
                ce, cx = best_curve(e), best_curve(x)
                rows.append({"seed": s, "dist": d, "gain80": (ce[79] - cx[79]) / ce[79], "gain320": (ce[319] - cx[319]) / ce[319]})
            if rows:
                pt = {"model": model, "family": FAMILY[model], "source": src, "n": len(rows),
                      "dist": float(np.mean([r["dist"] for r in rows])),
                      "gain80": float(np.mean([r["gain80"] for r in rows])), "gain320": float(np.mean([r["gain320"] for r in rows])),
                      "same_family": src.startswith("tmpl:") and FAMILY[src[5:]] == FAMILY[model], "seeds": rows}
                points.append(pt)
    res["A"]["points"] = points
    for key in ("gain80", "gain320"):
        pts = [p for p in points if p["source"] != "own"]
        res["A"][f"spearman_{key}"] = spearman([p["dist"] for p in pts], [p[key] for p in pts])
        res["A"][f"spearman_{key}_with_own"] = spearman([p["dist"] for p in points], [p[key] for p in points])
        res["A"][f"n_points"] = len(pts)

    # ---------------------------------------------------------------- (B) two scales on Qwen2.5-1.5B
    model, seeds = "qwen15b", [0, 1, 2, 4, 5, 6, 7, 8, 9]
    arms = ({"EA": "qwen15b_ea_nogp", "LLM": "qwen15b_llm_nogp", "random": "qwen15b_random_nogp"} if PLAIN else
            {"EA": "tuned2gp_baseline", "LLM": "llm_tuned2gp", "random": "tuned2gp_random"})
    data = {a: {s: load(gp, s) for s in seeds} for a, gp in arms.items()}
    X, M, K, T, A, G = [], [], [], [], [], []
    for a, runs in data.items():
        for s, e in runs.items():
            for x in e:
                X.append(module_fracs(model, x["policy"])); M.append(macro(model, x["policy"])); K.append(x["kl"])
                T.append(x["evaluations"]); A.append(a); G.append(f"s{s}")
    # Folds are grouped by seed, not by search: the three arms of a seed share their initial design and can evaluate
    # identical allocations (EA and random template, seed 9: 936 of 1000), which would leak across folds.
    X, M, K, T = np.array(X), np.array(M), np.array(K), np.array(T)
    logk = np.log(K)
    # PCA of the module fractions
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    Z = Xc @ Vt[:3].T
    Mc = np.c_[np.ones(len(M)), M]


    def r2_linear(F, y):
        beta, *_ = np.linalg.lstsq(F, y, rcond=None)
        return float(1 - ((y - F @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())


    res["B"]["pca_var"] = var[:5].round(4).tolist()
    res["B"]["pc_r2_from_macro"] = [r2_linear(Mc, Z[:, i]) for i in range(3)]
    res["B"]["pc_loadings_cell"] = [[float(np.mean(Vt[i][FR[model][1] == c])) for c in range(8)] for i in range(3)]
    # divergence explained by macro-structure vs full vector: k-fold CV with gradient boosting (sklearn) if available
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import cross_val_score, KFold

        def cv_r2(F, y):
            return float(np.mean(cross_val_score(HistGradientBoostingRegressor(max_iter=300, random_state=0), F, y,
                                                 cv=KFold(5, shuffle=True, random_state=0), scoring="r2")))
        elite = K <= np.quantile(K, 0.10)
        # folds split by seed (see above), so that no seed contributes to both the training and the test folds
        from sklearn.model_selection import GroupKFold
        G = np.array(G)

        def cv_r2_grouped(F, y, g):
            return float(np.mean(cross_val_score(HistGradientBoostingRegressor(max_iter=300, random_state=0), F, y, groups=g,
                                                 cv=GroupKFold(5), scoring="r2")))
        res["B"]["r2_logkl_grouped"] = {"macro_all": cv_r2_grouped(M, logk, G), "full_all": cv_r2_grouped(X, logk, G),
                                        "macro_elite": cv_r2_grouped(M[elite], logk[elite], G[elite]),
                                        "full_elite": cv_r2_grouped(X[elite], logk[elite], G[elite]),
                                        "n_searches": int(len(set(G))), "n_searches_elite": int(len(set(G[elite])))}
        res["B"]["r2_logkl"] = {"macro_all": cv_r2(M, logk), "full_all": cv_r2(X, logk),
                                "macro_elite": cv_r2(M[elite], logk[elite]), "full_elite": cv_r2(X[elite], logk[elite]),
                                "elite_threshold": float(np.quantile(K, 0.10)), "n_all": int(len(K)), "n_elite": int(elite.sum())}
    except Exception as exc:                                      # noqa: BLE001
        res["B"]["r2_logkl"] = {"error": repr(exc)}
    # converged macro at 1000 (EA champions) and trajectories
    final = np.mean([macro(model, min(e, key=lambda x: x["kl"])["policy"]) for e in data["EA"].values()], axis=0)
    res["B"]["final_macro_ea1000"] = final.round(4).tolist()
    traj = {}
    for a, runs in data.items():
        dist, div_macro, div_full, kl = [], [], [], []
        for s, e in runs.items():
            best, bd, pop_d, pop_f = None, [], [], []
            pop = []
            for i, x in enumerate(e):
                if best is None or x["kl"] < best["kl"]:
                    best = x
                bd.append(float(np.abs(macro(model, best["policy"]) - final).mean()))
                pop = sorted(pop + [x], key=lambda r: r["kl"])[:8]
                if (i + 1) % 10 == 0 and len(pop) == 8:
                    mm = np.array([macro(model, p["policy"]) for p in pop]); ff = np.array([module_fracs(model, p["policy"]) for p in pop])
                    iu = np.triu_indices(8, 1)
                    pop_d.append(float(np.abs(mm[:, None] - mm[None]).mean(-1)[iu].mean()))
                    pop_f.append(float(np.abs(ff[:, None] - ff[None]).mean(-1)[iu].mean()))
            dist.append(bd); div_macro.append(pop_d); div_full.append(pop_f); kl.append(best_curve(e).tolist())
        traj[a] = {"macro_dist_best": np.mean(dist, 0).round(5).tolist(), "div_macro_top8": np.mean(div_macro, 0).round(5).tolist(),
                   "div_full_top8": np.mean(div_full, 0).round(5).tolist(), "best_kl": np.mean(kl, 0).round(5).tolist()}
    res["B"]["trajectories"] = traj
    # PCA coordinates for a figure: a subsample of all points plus best-so-far paths of each arm for every seed
    rng = np.random.default_rng(0)
    idx = rng.choice(len(K), size=min(6000, len(K)), replace=False)
    res["B"]["pca_points"] = {"z": Z[idx, :2].round(4).tolist(), "kl": K[idx].round(4).tolist(), "arm": [A[i] for i in idx]}
    paths = {}
    off = 0
    for a, runs in data.items():
        for s, e in runs.items():
            n = len(e); zz = Z[off:off + n, :2]; kk = K[off:off + n]
            bi = np.array([int(np.argmin(kk[:i + 1])) for i in range(n)])
            paths[f"{a}/s{s}"] = {"z": zz[bi].round(4).tolist()}
            off += n
    res["B"]["best_paths"] = paths
    # ---------------------------------------------------------------- (C) Llama-3.1-8B: best-so-far macro distance and diversity
    if PLAIN:
        model = "llama31"
        data = {a: {s: load(gp, s) for s in range(8)} for a, gp in (("EA", "llama31_ea_nogp"), ("LLM", "llama31_llm_nogp"))}
        ref = np.mean([macro(model, min(e, key=lambda x: x["kl"])["policy"]) for e in data["EA"].values()], axis=0)
        res["C"] = {}
        for a, runs in data.items():
            out = {}
            for b in (80, 160, 320):
                dist, div = [], []
                for e in runs.values():
                    pop = sorted(e[:b], key=lambda r: r["kl"])[:8]
                    mm = np.array([macro(model, p["policy"]) for p in pop]); iu = np.triu_indices(8, 1)
                    div.append(float(np.abs(mm[:, None] - mm[None]).mean(-1)[iu].mean()))
                    dist.append(float(np.abs(mm[0] - ref).mean()))
                out[b] = {"macro_dist_best": float(np.mean(dist)), "div_macro_top8": float(np.mean(div))}
            res["C"][a] = out
        print("C (Llama-3.1-8B, reference = EA champions at 320):", json.dumps(res["C"]))
    json.dump(res, open(OUT, "w"))
    print("A: points", len(points), {k: round(v, 3) for k, v in res["A"].items() if k.startswith("spearman")})
    for p in points:
        print(f"  {p['model']:11s} {p['source']:13s} n={p['n']} dist={p['dist']:.3f} gain80={100*p['gain80']:6.1f}% gain320={100*p['gain320']:5.1f}%")
    print("B: pca var", res["B"]["pca_var"], "pc r2 from macro", [round(v, 3) for v in res["B"]["pc_r2_from_macro"]])
    print("B: r2", res["B"]["r2_logkl"])
    for a, t in traj.items():
        md = t["macro_dist_best"]; dv = t["div_macro_top8"]; df = t["div_full_top8"]
        print(f"  {a:7s} macro dist best @17 {md[16]:.4f} @80 {md[79]:.4f} @320 {md[319]:.4f} @1000 {md[-1]:.4f} | div macro @80 {dv[7]:.4f} @320 {dv[31]:.4f} @500 {dv[49]:.4f} @1000 {dv[-1]:.4f} | div full @80 {df[7]:.4f} @500 {df[49]:.4f} @1000 {df[-1]:.4f}")


if __name__ == "__main__":
    main()
