"""Fig. (Section 4.3): KL reduction of every optimizer relative to the EA at the same number of evaluations.
Same runs and seeds as Table tab:llm-opt (Qwen2.5-1.5B nine seeds; Llama-3.1-8B seeds 0-3); y = per seed
100 x (EA - method) / EA of the best-so-far KL, mean over seeds, so the EA is the zero line. The LLM optimizers and three
reference searches are drawn in colour, the other conventional optimizers as thin grey lines. The y-axis is clipped at
-70%; a line that leaves it is marked with its value at 320 evaluations.
Reads:  <records>/runs/<group>_s<seed>/results.json (or <group>_rk_s<seed>), groups
        qwen15b_ea_nogp, tuned2gp_baseline, qwen_classic_{gpbo,random,tpe,de,pso,ga,cmaes},
        qwen_llmopt_{gpt5mini,gpt5mini_anon,deepseek,luna}, and the llama31_ea_nogp, llama_tuned_baseline,
        llama_classic_*, llama_llmopt_* counterparts (all listed in PANELS).
Writes: the PDF given as argument and a PNG next to it; prints the plotted values at 32/80/160/320 evaluations as JSON.
Usage: python fig_optimizers_gain.py <records> <out.pdf>"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

from records import records_dir, runs_path, resolve_run

B, YMIN, YMAX = 320, -70, 45
# key -> (legend label, colour, line style, width); grey context lines have label None
STYLE = {"gpt5mini": ("gpt-5-mini", "#1f1f1f", "-", 1.5),
         "gpt5mini_anon": ("gpt-5-mini, semantics removed", "#1f1f1f", "--", 1.2),
         "deepseek": ("DeepSeek-V4-Flash", "#1baf7a", "-", 1.2),
         "luna": ("gpt-5.6-luna", "#eda100", "-", 1.2),
         "ea_guided": ("EA with guided mutation", "#2a78d6", "-", 1.1),
         "gpbo": ("GP-BO", "#eb6834", "-", 1.1),
         "random": ("Random search", "#8a8984", "--", 1.1),
         "tpe": (None, "#c9c8c4", "-", 0.7), "de": (None, "#c9c8c4", "-", 0.7), "pso": (None, "#c9c8c4", "-", 0.7),
         "ga": (None, "#c9c8c4", "-", 0.7), "cmaes": (None, "#c9c8c4", "-", 0.7)}
PANELS = [("Qwen2.5-1.5B, nine seeds", [0, 1, 2, 4, 5, 6, 7, 8, 9], "qwen15b_ea_nogp",
           {"gpt5mini": "qwen_llmopt_gpt5mini", "gpt5mini_anon": "qwen_llmopt_gpt5mini_anon", "deepseek": "qwen_llmopt_deepseek",
            "luna": "qwen_llmopt_luna", "ea_guided": "tuned2gp_baseline", "gpbo": "qwen_classic_gpbo", "random": "qwen_classic_random",
            "tpe": "qwen_classic_tpe", "de": "qwen_classic_de", "pso": "qwen_classic_pso", "ga": "qwen_classic_ga", "cmaes": "qwen_classic_cmaes"}),
          ("Llama-3.1-8B, seeds 0–3", [0, 1, 2, 3], "llama31_ea_nogp",
           {"gpt5mini": "llama_llmopt_gpt5mini", "gpt5mini_anon": "llama_llmopt_gpt5mini_anon", "deepseek": "llama_llmopt_deepseek",
            "luna": "llama_llmopt_luna", "ea_guided": "llama_tuned_baseline", "gpbo": "llama_classic_gpbo", "random": "llama_classic_random",
            "tpe": "llama_classic_tpe", "de": "llama_classic_de", "pso": "llama_classic_pso", "ga": "llama_classic_ga", "cmaes": "llama_classic_cmaes"})]


def curve(runs, group, seed):
    p = resolve_run(runs, group, seed) / "results.json"
    r = json.loads(p.read_text())
    assert r.get("final") or r.get("phase") == "complete", p
    return np.minimum.accumulate([e["kl"] for e in sorted(r["entries"], key=lambda e: e["id"])])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    RUNS, OUT = runs_path(records_dir(argv[0])), argv[1]
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharey=True, gridspec_kw={"wspace": 0.08})
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.15, top=0.74)
    handles, summary = {}, {}
    for ax, (title, seeds, ea_group, groups) in zip(axes, PANELS):
        ea = {s: curve(RUNS, ea_group, s) for s in seeds}
        x = np.arange(16, B + 1)
        for key in reversed(list(STYLE)):               # grey context first, highlighted lines on top
            label, colour, ls, lw = STYLE[key]
            g = np.array([100 * (ea[s][15:B] - curve(RUNS, groups[key], s)[15:B]) / ea[s][15:B] for s in seeds])
            mean = g.mean(axis=0)
            line, = ax.plot(x, np.maximum(mean, YMIN - 5), color=colour, linestyle=ls, linewidth=lw, zorder=2 if label is None else 3)
            handles[label or "Other conventional optimizers"] = line
            summary[f"{title} | {label or key}"] = {str(b): round(float(mean[b - 16]), 1) for b in (32, 80, 160, 320)}
            if mean[-1] < YMIN:                          # off scale: mark where it leaves and its final value
                leave = int(np.argmax(mean < YMIN)) + 16
                # name the method, in the line's colour, in the empty lower left corner
                final = f"{mean[-1]:.0f}".replace("-", "\u2212")
                ax.annotate(f"{label}:\n{final}% at 320 evaluations", xy=(leave, YMIN), xytext=(17.5, YMIN + 10), ha="left",
                            va="center", fontsize=6.5, color=colour,
                            arrowprops=dict(arrowstyle="-", color=colour, linewidth=0.6, shrinkA=1, shrinkB=0),
                            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"))
        ax.axhline(0, color="#1f1f1f", linewidth=0.8, zorder=3)
        for b in (80, 160):
            ax.axvline(b, color="#b0b0b0", linewidth=0.5, linestyle=":", zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(16, B)
        ax.set_ylim(YMIN, YMAX)
        ticks = [16, 32, 80, 160, 320]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlabel("Evaluations (log scale)", labelpad=1)
        ax.set_title(title, fontsize=8, pad=3)
        ax.grid(True, axis="y", linewidth=0.3, color="#e0e0e0", zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("KL reduction over the EA (%)", labelpad=1)
    handles["EA (zero line)"] = plt.Line2D([], [], color="#1f1f1f", linewidth=0.8)
    order = ["gpt-5-mini", "gpt-5-mini, semantics removed", "DeepSeek-V4-Flash", "gpt-5.6-luna", "EA with guided mutation", "GP-BO",
             "Random search", "Other conventional optimizers", "EA (zero line)"]
    fig.legend([handles[k] for k in order], order, loc="upper center", bbox_to_anchor=(0.52, 1.0), ncol=3, frameon=False,
               fontsize=7, handlelength=2.4, columnspacing=1.4)
    fig.savefig(OUT)
    fig.savefig(str(OUT).replace(".pdf", ".png"), dpi=200)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
