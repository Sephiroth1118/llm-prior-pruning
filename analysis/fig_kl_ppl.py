"""Fig. (Section 3.4): search KL vs downstream perplexity (top) and average zero-shot accuracy (bottom) of the search
champions (champions chosen by search KL only); n in the panels is the number of searches.
Reads:  <records>/reports/benchmark/metrics.json  (rows "<model>/<group>/s<seed>"; see records.py)
Writes: the PDF given as argument.
Usage: python fig_kl_ppl.py <records> <out.pdf>"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from records import records_dir, reports_path

def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r

def spearman(x, y):
    rx, ry = ranks(x), ranks(y); n = len(x); mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out = argv[1]
    rows = json.load(open(reports_path(records_dir(argv[0])) / "benchmark" / "metrics.json"))["rows"]
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.3), constrained_layout=True)
    models = [("qwen15b", "Qwen2.5-1.5B", "#2b6a9e"), ("llama31", "Llama-3.1-8B", "#b4562f")]
    for row, (metric, label, scale, corner) in enumerate([("ppl", "Perplexity", 1.0, "upper left"),
                                                           ("avg", "Mean task accuracy (%)", 100.0, "upper right")]):
        for col, (key, title, color) in enumerate(models):
            ax = axes[row, col]
            pts = [(v["search_kl"], v[metric] * scale) for k, v in rows.items()
                   if k.startswith(key + "/") and "/dense/" not in k and v.get("search_kl") is not None
                   and v.get(metric) is not None]
            x, y = [p[0] for p in pts], [p[1] for p in pts]
            ax.scatter(x, y, s=10, color=color, edgecolor="white", linewidth=0.3, zorder=3)
            if row == 0:
                ax.set_title(title, fontsize=8, pad=3)
            else:
                ax.set_xlabel("Search divergence (KL)", labelpad=1)
            ax.grid(True, linewidth=0.3, color="#d0d0d0", zorder=0)
            left = corner == "upper left"
            ax.text(0.04 if left else 0.96, 0.95, f"$r_s = {spearman(x, y):.2f}$\n$n = {len(x)}$", transform=ax.transAxes,
                    va="top", ha="left" if left else "right", fontsize=7)
        axes[row, 0].set_ylabel(label, labelpad=1)
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main()
