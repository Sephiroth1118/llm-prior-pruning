"""Fig. (Section 6.1): KL reduction of every search relative to the EA at the same number of evaluations: per seed
100 x (EA - method) / EA of the best-so-far divergence, mean over seeds, so the EA is the zero line. Log x-axis from the
end of the shared initial design (16 evaluations); band: +-1 standard error of the mean over seeds for our method.
Reads:  <records>/runs/<group>_s<seed>/results.json (or <group>_rk_s<seed>), groups (see PANELS)
        Qwen2.5-1.5B (nine seeds, 1000 evaluations): qwen15b_{ea,random,own,llm}_nogp, qwen15b_tmpl_llama31_nogp_1k,
        qwen15b_own_nogp_1k, qwen_classic_gpbo_1k, qwen_llmopt_gpt5mini_1k;
        Llama-3.1-8B (eight seeds, 320 evaluations): llama31_{ea,random,own,llm}_nogp, llama31_tmpl_qwen15b_nogp,
        llama_classic_gpbo, llama_llmopt_gpt5mini.
Writes: the PDF given as argument and a PNG next to it; prints the plotted values as JSON.
Usage: python fig_spi_gain.py <records> <out.pdf>"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

from records import records_dir, runs_path, resolve_run

# name -> (colour, line style, width, label); colours as in the other figures
STYLE = {"LLM template (ours)": ("#e34948", "-", 1.6, "LLM template (ours)"),
         "Cross-family template": ("#6250d6", "-", 1.0, "Cross-family template"),
         "Random template": ("#8a8984", "--", 1.0, "Random template"),
         "GP-BO": ("#eb6834", "-", 1.0, "GP-BO"),
         "LLM optimizer (gpt-5-mini)": ("#1f1f1f", "-", 1.0, "LLM as optimizer"),
         "Own template (reference)": ("#8a8984", ":", 1.2, "Own template (ref.)")}
PANELS = [("Qwen2.5-1.5B, nine seeds", 1000, [0, 1, 2, 4, 5, 6, 7, 8, 9],
           {"EA": "qwen15b_ea_nogp", "Random template": "qwen15b_random_nogp", "Cross-family template": "qwen15b_tmpl_llama31_nogp_1k",
            "Own template (reference)": "qwen15b_own_nogp_1k", "LLM template (ours)": "qwen15b_llm_nogp", "GP-BO": "qwen_classic_gpbo_1k",
            "LLM optimizer (gpt-5-mini)": "qwen_llmopt_gpt5mini_1k"}),
          ("Llama-3.1-8B, eight seeds", 320, [0, 1, 2, 3, 4, 5, 6, 7],
           {"EA": "llama31_ea_nogp", "GP-BO": "llama_classic_gpbo", "Random template": "llama31_random_nogp",
            "Cross-family template": "llama31_tmpl_qwen15b_nogp", "Own template (reference)": "llama31_own_nogp",
            "LLM template (ours)": "llama31_llm_nogp", "LLM optimizer (gpt-5-mini)": "llama_llmopt_gpt5mini"})]


def curve(runs, group, seed):
    p = resolve_run(runs, group, seed) / "results.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    if not r.get("final"):
        return None
    return np.minimum.accumulate([e["kl"] for e in sorted(r["entries"], key=lambda e: e["id"])])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    RUNS, OUT = runs_path(records_dir(argv[0])), argv[1]
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), gridspec_kw={"wspace": 0.16})
    fig.subplots_adjust(left=0.07, right=0.975, bottom=0.16, top=0.78)
    handles = {}
    summary = {}
    for ax, (title, B, seeds, groups) in zip(axes, PANELS):
        ea = {s: curve(RUNS, groups["EA"], s) for s in seeds}
        x = np.arange(16, B + 1)
        for name, (colour, ls, lw, label) in STYLE.items():
            cs = {s: curve(RUNS, groups[name], s) for s in seeds}
            common = [s for s in seeds if cs[s] is not None and ea[s] is not None and len(cs[s]) >= B and len(ea[s]) >= B]
            assert len(common) == len(seeds), (title, name, common)
            g = np.array([100 * (ea[s][15:B] - cs[s][15:B]) / ea[s][15:B] for s in common])
            mean = g.mean(axis=0)
            if "ours" in name:
                se = g.std(axis=0, ddof=1) / np.sqrt(len(common))
                ax.fill_between(x, mean - se, mean + se, color=colour, alpha=0.15, linewidth=0, zorder=2)
            handles[label], = ax.plot(x, mean, color=colour, linestyle=ls, linewidth=lw, zorder=4 if "ours" in name else 3)
            summary[f"{title} | {label}"] = {str(b): round(float(mean[b - 16]), 1) for b in (32, 80, 160, 320, 1000) if b <= B}
        ax.axhline(0, color="#1f1f1f", linewidth=0.8, zorder=3)
        ax.axvline(80, color="#b0b0b0", linewidth=0.5, linestyle=":", zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(16, B)
        ticks = [16, 32, 80, 160, 320] + ([1000] if B == 1000 else [])
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
    order = ["LLM template (ours)", "Cross-family template", "Own template (ref.)", "Random template", "GP-BO", "LLM as optimizer", "EA (zero line)"]
    fig.legend([handles[k] for k in order], order, loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=4, frameon=False,
               fontsize=7, handlelength=2.4, columnspacing=1.4)
    fig.savefig(OUT)
    fig.savefig(str(OUT).replace(".pdf", ".png"), dpi=200)
    print(json.dumps(summary, indent=0))


if __name__ == "__main__":
    main()
