"""Section 7 robustness checks (CPU only, read-only on the run records).

(R1) Prior-target match (Section 7.1) without pooling dependent points: the 38 (target, source) points of
     sec7_analysis.py part A, with the distance and the gain demeaned within each target model (does a closer
     template help more on the same target?) and within each source (does the same source help more where it
     matches better?). Spearman correlations of the residuals, with permutation p-values that shuffle the
     values within each target or source. Also the per-target Spearman correlations.
(R2) Two scales of the landscape (Section 7.2) on all eight models: for every model, all allocations evaluated by
     its plain-EA searches (EA, LLM template, transferred, hand-designed, own and, where run, random templates; first
     320 evaluations of every search), five-fold cross-validated R^2 of a gradient-boosted regression of log KL on
     the eight cell means and on all module fractions, over all allocations and over the 10% with the lowest KL,
     with the folds split by search. Also the share of variance on the first principal component and how well
     the cell means predict it.
Helpers (run loading, macro-structure, groups) are imported from sec7_analysis.py in its plain-all mode.
Reads:  the JSON of sec7_analysis.py <records> <sec7.json> plain-all; <records>/runs/<group>_s<seed>/results.json
        (or _rk_s<seed>) for <model>_{ea,llm,own,humandepth}_nogp, <model>_tmpl_<source>_nogp (all eight models) and
        llama31_random_nogp, qwen15b_random_nogp; <records>/flops/<model>.json
Writes: <out.json> and prints the results.
Usage: python sec7_robustness.py <records> <sec7.json> <out.json>"""
import json, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GroupKFold

import sec7_analysis as H

rng = np.random.default_rng(0)


def spearman(x, y):
    return H.spearman(x, y)


def demean(v, keys):
    v = np.asarray(v, float); out = v.copy()
    for k in set(keys):
        m = np.array([kk == k for kk in keys]); out[m] = v[m] - v[m].mean()
    return out


def perm_p(x, y, keys, n=20000):
    """Two-sided p of the Spearman correlation of x and y when y is shuffled within each key."""
    obs = spearman(x, y); keys = np.asarray(keys); y = np.asarray(y, float); hits = 0
    idx = {k: np.where(keys == k)[0] for k in set(keys)}
    for _ in range(n):
        yp = y.copy()
        for ii in idx.values():
            yp[ii] = y[rng.permutation(ii)]
        hits += abs(spearman(x, yp)) >= abs(obs) - 1e-12
    return (hits + 1) / (n + 1)


def kind(p):
    s = p["source"]
    if s == "LLM": return "LLM"
    if s == "depth": return "hand"
    if s == "random": return "random"
    return "same" if p["same_family"] else "cross"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    RECORDS, SEC7, OUT = argv[0], argv[1], argv[2]
    H.setup(RECORDS, plain=True, all_seeds=True)
    res = {"R1": {}, "R2": {}}
    # ---------------------------------------------------------------- (R1)
    pts = [p for p in json.load(open(SEC7))["A"]["points"] if p["source"] != "own"]
    targets = [p["model"] for p in pts]; sources = [p["source"] for p in pts]; kinds = [kind(p) for p in pts]
    for key in ("gain80", "gain320"):
        d = [p["dist"] for p in pts]; g = [p[key] for p in pts]
        r = {"n_points": len(pts), "pooled": spearman(d, g)}
        dt, gt = demean(d, targets), demean(g, targets)
        r["within_target"] = {"rho": spearman(dt, gt), "p_perm": perm_p(dt, gt, targets)}
        per = {}
        for m in sorted(set(targets)):
            ii = [i for i, t in enumerate(targets) if t == m]
            per[m] = {"n": len(ii), "rho": spearman([d[i] for i in ii], [g[i] for i in ii])}
        r["per_target"] = per
        r["per_target_negative"] = sum(v["rho"] < 0 for v in per.values())
        r["per_target_mean"] = float(np.mean([v["rho"] for v in per.values()]))
        ds, gs = demean(d, sources), demean(g, sources)
        r["within_source"] = {"rho": spearman(ds, gs), "p_perm": perm_p(ds, gs, sources), "n_sources": len(set(sources))}
        dk, gk = demean(d, kinds), demean(g, kinds)
        r["within_kind"] = {"rho": spearman(dk, gk), "p_perm": perm_p(dk, gk, kinds), "kinds": sorted(set(kinds))}
        per_src = {}
        for s in sorted(set(sources)):
            ii = [i for i, t in enumerate(sources) if t == s]
            if len(ii) >= 4:
                per_src[s] = {"n": len(ii), "rho": spearman([d[i] for i in ii], [g[i] for i in ii])}
        r["per_source"] = per_src
        res["R1"][key] = r
        print(key, json.dumps({k: v for k, v in r.items() if k not in ("per_target", "per_source")}), flush=True)
        print("  per target", {m: round(v["rho"], 2) for m, v in per.items()}, flush=True)
        print("  per source", {m: round(v["rho"], 2) for m, v in per_src.items()}, flush=True)

    # ---------------------------------------------------------------- (R2)
    def cv_r2(F, y, grp):
        return float(np.mean(cross_val_score(HistGradientBoostingRegressor(max_iter=300, random_state=0), F, y, groups=grp,
                                             cv=GroupKFold(5), scoring="r2")))


    for model in H.MODELS:
        g = H.groups(model)
        if model == "llama31":
            g["random"] = "llama31_random_nogp"
        X, M, K, G = [], [], [], []
        searches = {}
        for arm, group in g.items():
            if arm.startswith("tmpl:") and arm[5:] == model:
                continue
            for s in range(10):
                e = H.load(group, s)
                if not e:
                    continue
                searches[arm] = searches.get(arm, 0) + 1
                for x in e[:320]:
                    X.append(H.module_fracs(model, x["policy"])); M.append(H.macro(model, x["policy"])); K.append(x["kl"])
                    G.append(f"{arm}/s{s}")
        X, M, K, G = np.array(X), np.array(M), np.array(K), np.array(G)
        logk = np.log(K)
        elite = K <= np.quantile(K, 0.10)
        Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); var = S ** 2 / (S ** 2).sum()
        z1 = Xc @ Vt[0]; Mc = np.c_[np.ones(len(M)), M]
        beta, *_ = np.linalg.lstsq(Mc, z1, rcond=None)
        r2_pc1 = float(1 - ((z1 - Mc @ beta) ** 2).sum() / ((z1 - z1.mean()) ** 2).sum())
        out = {"searches": searches, "n_searches": int(len(set(G))), "n_all": int(len(K)), "n_elite": int(elite.sum()),
               "n_searches_elite": int(len(set(G[elite]))), "pc1_var": float(var[0]), "pc1_r2_from_macro": r2_pc1,
               "macro_all": cv_r2(M, logk, G), "full_all": cv_r2(X, logk, G),
               "macro_elite": cv_r2(M[elite], logk[elite], G[elite]), "full_elite": cv_r2(X[elite], logk[elite], G[elite])}
        res["R2"][model] = out
        print(model, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
