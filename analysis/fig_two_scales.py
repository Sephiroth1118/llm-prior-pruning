"""Fig. (Section 7.2): two scales of the allocation landscape on Qwen2.5-1.5B.
(a) PCA of the evaluated allocations with best-so-far paths of one seed; (b) macro distance of the best-so-far
allocation to the converged macro-structure; (c) share of the variance of log divergence explained by the
macro-structure and by the full allocation, over all allocations and over the best 10% (cross-validated R^2 with folds
grouped by seed, key r2_logkl_grouped; negative values are shown).
Reads:  the JSON written by sec7_analysis.py <records> <sec7.json> plain-all (key "B").
Writes: the PDF given as argument (300 dpi).
Usage: python fig_two_scales.py <sec7.json> <out.pdf>"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    B = json.load(open(argv[0]))["B"]
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.15, 1.15, 0.8]})
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.23, top=0.88, wspace=0.42)
    COL = {"EA": "#2a78d6", "LLM": "#e34948", "random": "#8a8984"}
    LAB = {"EA": "EA", "LLM": "LLM template", "random": "Random template"}

    # (a) PCA plane
    ax = axes[0]
    z = np.array(B["pca_points"]["z"]); kl = np.array(B["pca_points"]["kl"])
    order = np.argsort(-kl)
    sc = ax.scatter(z[order, 0], z[order, 1], c=np.log(kl[order]), cmap="Greys_r", s=2, linewidths=0, vmin=np.log(0.5),
                    vmax=np.log(3.0), alpha=0.6, rasterized=True, zorder=1)
    for arm in ("EA", "LLM"):
        p = np.array(B["best_paths"][f"{arm}/s0"]["z"])
        keep = np.r_[True, np.any(np.diff(p, axis=0) != 0, axis=1)]
        q = p[keep]
        ax.plot(q[:, 0], q[:, 1], color=COL[arm], linewidth=0.9, marker="o", markersize=1.8, zorder=3, label=LAB[arm])
        ax.scatter(q[:1, 0], q[:1, 1], s=18, facecolor="white", edgecolor=COL[arm], linewidth=0.9, zorder=4)
        ax.scatter(q[-1:, 0], q[-1:, 1], s=22, marker="s", facecolor=COL[arm], edgecolor="white", linewidth=0.6, zorder=5)
    var = B["pca_var"]
    ax.set_xlabel(f"PC1 ({100 * var[0]:.1f}%)\nless attention pruned $\\rightarrow$", labelpad=1)
    ax.set_ylabel(f"PC2 ({100 * var[1]:.1f}%)", labelpad=1)
    ax.set_title("(a) Evaluated allocations, PCA", fontsize=8, pad=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cb.set_ticks(np.log([0.8, 1.2, 2.0, 3.0])); cb.set_ticklabels(["0.8", "1.2", "2", "3"]); cb.ax.tick_params(labelsize=6)
    cb.ax.set_title("KL", fontsize=7, pad=2)
    ax.legend(loc="lower left", fontsize=6.5, frameon=False, handlelength=1.6)
    ax.set_xticks([]); ax.set_yticks([])

    # (b) macro distance of the best-so-far allocation to the converged macro-structure
    ax = axes[1]
    for arm in ("EA", "LLM", "random"):
        d = np.array(B["trajectories"][arm]["macro_dist_best"])
        x = np.arange(1, len(d) + 1)
        ax.plot(x[15:], d[15:], color=COL[arm], linestyle="--" if arm == "random" else "-", linewidth=1.1, label=LAB[arm])
    ax.set_xscale("log"); ax.set_xlim(16, 1000)
    ax.set_xticks([16, 80, 320, 1000]); ax.set_xticklabels(["16", "80", "320", "1000"])
    ax.set_xlabel("Evaluations", labelpad=1)
    ax.set_ylabel("Distance to EA reference", labelpad=1)
    ax.set_title("(b) Best allocation so far", fontsize=8, pad=3)
    ax.grid(True, linewidth=0.3, color="#e8e8e8")
    ax.legend(loc="upper right", fontsize=6.5, frameon=False, handlelength=1.8)

    # (c) cross-validated R^2 of log KL, five folds split by seed (negative: worse than predicting the mean)
    ax = axes[2]
    r = B["r2_logkl_grouped"]
    vals = {"All\nallocations": (r["macro_all"], r["full_all"]), "Lowest-KL\n10%": (r["macro_elite"], r["full_elite"])}
    xs = np.arange(2); w = 0.36
    m = [vals[k][0] for k in vals]; f = [vals[k][1] for k in vals]
    ax.bar(xs - w / 2, m, w, color="#2a78d6", label="8 cell means", zorder=2)
    ax.bar(xs + w / 2, f, w, color="#9ec5f4", label="56 module fractions", zorder=2)
    for x0, v in list(zip(xs - w / 2, m)) + list(zip(xs + w / 2, f)):
        ax.text(x0, v + (0.02 if v >= 0 else -0.02), f"{v:.2f}".replace("-", "\u2212"), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=6.5)
    ax.axhline(0, color="#1f1f1f", linewidth=0.6, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(list(vals), fontsize=7)
    ax.set_ylim(-0.35, 1.6); ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Cross-validated $R^2$ of log KL", labelpad=1)
    ax.set_title("(c) Prediction of log KL", fontsize=8, pad=3)
    ax.legend(loc="upper right", fontsize=6.5, frameon=False, handlelength=1.2, borderaxespad=0.2)
    ax.grid(True, axis="y", linewidth=0.3, color="#e8e8e8", zorder=0)
    fig.savefig(argv[1], dpi=300)
    print("saved", argv[1])


if __name__ == "__main__":
    main()
