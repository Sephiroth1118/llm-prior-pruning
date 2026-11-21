"""Fig. (Section 7.1): prior-target match against the early and late gain over the EA.
Shared vertical axis with a zero line; the exceptions discussed in Section 7.1 are numbered in the 80-evaluation panel.
Reads:  the JSON written by sec7_analysis.py <records> <sec7.json> plain-all (key "A").
Writes: the PDF given as argument.
Usage: python fig_match.py <sec7.json> <out.pdf>"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def cls(p):
    if p["source"] == "own":
        return "Own (reference)"
    if p["source"] == "LLM":
        return "LLM"
    if p["source"] == "random":
        return "Random"
    if p["source"] == "depth":
        return "Hand-designed"
    return "Same family" if p["same_family"] else "Cross family"


# validated light-mode slots (all-pairs check passes with marker shapes as secondary encoding)
STYLE = {"LLM": ("#e34948", "o", True), "Same family": ("#008300", "s", True), "Cross family": ("#6250d6", "^", True),
         "Hand-designed": ("#eda100", "D", True), "Random": ("#8a8984", "X", True), "Own (reference)": ("#1f1f1f", "o", False)}


def rk(v):
    v = np.asarray(v); o = np.argsort(v); out = np.empty(len(v)); out[o] = np.arange(len(v)); return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    r = json.load(open(argv[0]))["A"]
    pts = r["points"]
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.31, top=0.92, wspace=0.12)
    # exceptions discussed in Section 7.1: (target, source) -> label, text offset in points
    # numbered in the figure, named in the caption: 1 Qwen template on OLMo-7B, 2 hand-designed on OLMo-7B,
    # 3 LLM template on Llama-3.2-3B, 4 hand-designed on Llama-3.2-3B
    NOTE = {("olmo7b", "tmpl:qwen15b"): ("1", (4, 3)), ("olmo7b", "depth"): ("2", (4, 2)),
            ("llama32_3b", "LLM"): ("3", (3, -5)), ("llama32_3b", "depth"): ("4", (4, 2))}
    handles = {}
    for ax, key, title in zip(axes, ("gain80", "gain320"), ("80 evaluations", "320 evaluations")):
        ax.axhline(0, color="#1f1f1f", linewidth=0.6, zorder=2)
        for name, (color, marker, filled) in STYLE.items():
            sel = [p for p in pts if cls(p) == name]
            if not sel:
                continue
            h = ax.scatter([p["dist"] for p in sel], [100 * p[key] for p in sel], s=14, marker=marker,
                           facecolor=color if filled else "none", edgecolor="white" if filled else color,
                           linewidth=0.4 if filled else 0.7, zorder=3)
            handles[name] = h
        use = [p for p in pts if p["source"] != "own"]
        rho = float(np.corrcoef(rk([p["dist"] for p in use]), rk([p[key] for p in use]))[0, 1])
        ax.text(0.96, 0.95, f"$r_s$ = {rho:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set_title(title, fontsize=8, pad=3)
        ax.grid(True, linewidth=0.3, color="#e8e8e8", zorder=0)
        if key == "gain80":
            for p in pts:
                if (p["model"], p["source"]) in NOTE:
                    label, off = NOTE[(p["model"], p["source"])]
                    ax.annotate(label, xy=(p["dist"], 100 * p[key]), xytext=off, textcoords="offset points", fontsize=6.5,
                                color="#1f1f1f", ha="left", va="center", fontweight="bold")
    axes[0].set_ylabel("KL reduction over the EA (%)", labelpad=1)
    fig.text(0.56, 0.195, "Distance to the EA reference (smaller = closer)", ha="center", fontsize=8)
    fig.legend([handles[n] for n in STYLE if n in handles], [n for n in STYLE if n in handles], loc="lower center",
               bbox_to_anchor=(0.55, 0.0), ncol=3, frameon=False, fontsize=6.5, handletextpad=0.3, columnspacing=0.9)
    fig.savefig(argv[1])
    print("saved", argv[1])


if __name__ == "__main__":
    main()
