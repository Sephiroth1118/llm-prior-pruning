"""Fig. (Section 6.2): relative KL reduction over the EA of every template source on the eight models, at 80 and 320
evaluations, as two heatmaps side by side. Plain EA searches with the re-ask reruns, seeds 0-3 (Qwen2.5-1.5B: 0, 1, 2, 4);
same gain as Table tab:matrix-summary: per seed 100 x (EA - template) / EA, mean over seeds.
Reads:  <records>/runs/<model>_{ea,llm,humandepth,own}_nogp_s<seed>/results.json and
        <model>_tmpl_<source>_nogp for the template sources qwen15b, llama31, olmo7b (or the _rk_s<seed> reruns);
        models qwen15b, qwen3b, qwen7b, llama32_1b, llama32_3b, llama31, olmo1b, olmo7b.
Writes: the PDF given as argument, a PNG next to it and, optionally, the plotted values as JSON.
Usage: python fig_matrix.py <records> <out.pdf> [<out.json>]"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle

from records import records_dir, runs_path, resolve_run

MODELS = ["qwen15b", "qwen3b", "qwen7b", "llama32_1b", "llama32_3b", "llama31", "olmo1b", "olmo7b"]
NAME = {"qwen15b": "Qwen2.5-1.5B", "qwen3b": "Qwen2.5-3B", "qwen7b": "Qwen2.5-7B", "llama32_1b": "Llama-3.2-1B",
        "llama32_3b": "Llama-3.2-3B", "llama31": "Llama-3.1-8B", "olmo1b": "OLMo-1B", "olmo7b": "OLMo-7B"}
FAMILY = {"qwen15b": "qwen", "qwen3b": "qwen", "qwen7b": "qwen", "llama32_1b": "llama", "llama32_3b": "llama",
          "llama31": "llama", "olmo1b": "olmo", "olmo7b": "olmo"}
SOURCE = {"qwen": "qwen15b", "llama": "llama31", "olmo": "olmo7b"}
COLS = ["LLM", "tmpl:qwen15b", "tmpl:llama31", "tmpl:olmo7b", "depth", "own"]
HEAD = ["LLM", "Qwen", "Llama", "OLMo", "Hand-\ndesigned", "Own"]
BLUE, RED, MID = "#2a78d6", "#e34948", "#f0efec"   # diverging pair and neutral midpoint (dataviz reference palette)
LIMIT = 45.0


def groups(m):                                     # as sec6_matrix_plain.py
    g = {"EA": f"{m}_ea_nogp", "LLM": f"{m}_llm_nogp", "depth": f"{m}_humandepth_nogp", "own": f"{m}_own_nogp"}
    for src in SOURCE.values():
        if src != m:
            g[f"tmpl:{src}"] = f"{m}_tmpl_{src}_nogp"
    return g


def curve(runs, group, seed):
    p = resolve_run(runs, group, seed) / "results.json"
    r = json.loads(p.read_text())
    assert r.get("final") and len(r["entries"]) >= 320, p
    return np.minimum.accumulate([e["kl"] for e in sorted(r["entries"], key=lambda e: e["id"])])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    RUNS, OUT = runs_path(records_dir(argv[0])), argv[1]
    data = {}
    for m in MODELS:
        seeds = [0, 1, 2, 4] if m == "qwen15b" else [0, 1, 2, 3]
        g = groups(m)
        ea = {s: curve(RUNS, g["EA"], s) for s in seeds}
        row = {"ea": {b: float(np.mean([ea[s][b - 1] for s in seeds])) for b in (80, 320)}}
        for col in COLS:
            if col not in g:
                row[col] = None                          # the target is this template's source
                continue
            c = {s: curve(RUNS, g[col], s) for s in seeds}
            row[col] = {b: float(np.mean([100 * (ea[s][b - 1] - c[s][b - 1]) / ea[s][b - 1] for s in seeds])) for b in (80, 320)}
        data[m] = row

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6, "pdf.fonttype": 42})
    cmap = LinearSegmentedColormap.from_list("div", [RED, MID, BLUE])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), gridspec_kw={"wspace": 0.06})
    x_of = {c: i + (0.25 if c in ("depth", "own") else 0) + (0.3 if c == "own" else 0)   # gaps after the transferred
            for i, c in enumerate(COLS)}                                                # templates and before the reference
    for panel, (ax, b) in enumerate(zip(axes, (80, 320))):
        for r, m in enumerate(MODELS):
            y = r + (0.3 if FAMILY[m] == "llama" else 0.6 if FAMILY[m] == "olmo" else 0)   # gaps between families
            ax.text(-0.75, y + 0.5, f"{data[m]['ea'][b]:.3f}", ha="center", va="center", color="#6b6b6b", fontsize=7)
            for col in COLS:
                x, v = x_of[col], data[m][col]
                if v is None:
                    ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#fafaf9", edgecolor="white", linewidth=1.2))
                    ax.text(x + 0.5, y + 0.5, "—", ha="center", va="center", color="#9a9a9a", fontsize=7)
                    continue
                value = v[b]
                face = cmap(0.5 + np.clip(value, -LIMIT, LIMIT) / (2 * LIMIT))
                lum = np.dot(to_rgb(face), [0.2126, 0.7152, 0.0722])
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=face, edgecolor="white", linewidth=1.2))
                label = "0" if abs(value) < 0.5 else f"{value:+.0f}".replace("-", "−")
                ax.text(x + 0.5, y + 0.5, label, ha="center", va="center", fontsize=7.5,
                        color="white" if lum < 0.42 else "#1f1f1f")
                if col == f"tmpl:{SOURCE[FAMILY[m]]}":    # same-family transfer
                    ax.add_patch(Rectangle((x + 0.06, y + 0.06), 0.88, 0.88, fill=False, edgecolor="#1f1f1f", linewidth=0.9))
        ax.set_xlim(-1.25, x_of["own"] + 1)
        ax.set_ylim(len(MODELS) + 0.6, -1.15)
        ax.axis("off")
        for col, head in zip(COLS, HEAD):
            ax.text(x_of[col] + 0.5, -0.1, head, ha="center", va="bottom", fontsize=7.5, linespacing=1.0)
        ax.text(-0.75, -0.1, "EA\nKL", ha="center", va="bottom", fontsize=7.5, color="#6b6b6b", linespacing=1.0)
        ax.plot([x_of["tmpl:qwen15b"] + 0.08, x_of["tmpl:olmo7b"] + 0.92], [-0.98, -0.98], color="#6b6b6b", linewidth=0.6)
        ax.text((x_of["tmpl:qwen15b"] + x_of["tmpl:olmo7b"] + 1) / 2, -1.0, "Transferred from", ha="center", va="bottom",
                fontsize=7.5, color="#3d3d3d")
        ax.set_title(f"({'ab'[panel]}) {b} evaluations", fontsize=8.5, loc="left", x=0.0, y=1.0, pad=14)
        if panel == 0:
            for r, m in enumerate(MODELS):
                y = r + (0.3 if FAMILY[m] == "llama" else 0.6 if FAMILY[m] == "olmo" else 0)
                ax.text(-1.3, y + 0.5, NAME[m], ha="right", va="center", fontsize=7.5)
    cax = fig.add_axes([0.38, 0.085, 0.26, 0.025])
    bar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-LIMIT, LIMIT)), cax=cax, orientation="horizontal",
                       ticks=[-40, -20, 0, 20, 40])
    bar.outline.set_linewidth(0.4)
    bar.ax.tick_params(labelsize=6.5, width=0.4, length=2)
    bar.ax.set_xticklabels([t.replace("-", "−") for t in ["-40", "-20", "0", "20", "40"]])
    fig.text(0.665, 0.097, "KL reduction over the EA (%)", fontsize=7, va="center")
    fig.subplots_adjust(left=0.115, right=0.995, top=0.86, bottom=0.17)
    fig.savefig(OUT, dpi=300)              # the colour bar is a bitmap
    fig.savefig(str(OUT).replace(".pdf", ".png"), dpi=200)
    if len(argv) > 2:
        Path(argv[2]).write_text(json.dumps(data, indent=1))
    print({NAME[m]: {c: (None if data[m][c] is None else [round(data[m][c][80]), round(data[m][c][320])]) for c in COLS} for m in MODELS})


if __name__ == "__main__":
    main()
