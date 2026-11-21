"""Section 7.3 (Table tab:llm-attention): how much of the attention compute the LLM optimizers' proposals remove, by phase
of the search (Qwen2.5-1.5B). Fraction removed per attention module = 1 - remaining attention compute / dense attention
compute, averaged over the 28 attention modules and over all proposals in a window and all seeds.
Reads:  <records>/runs/<group>_s<seed>/results.json (or <group>_rk_s<seed>), seeds 0,1,2,4-9, groups qwen_llmopt_gpt5mini,
        qwen_llmopt_gpt5mini_anon, qwen_llmopt_luna, qwen_llmopt_deepseek, qwen15b_ea_nogp;
        <records>/flops/qwen15b.json
Writes: nothing; prints one line per arm.
Usage: python sec7_llm_attention.py <records>"""
import json, sys
import numpy as np

from records import records_dir, runs_path, flops_table_path, resolve_run

h, hd, L = 1536, 128, 28                                        # Qwen2.5-1.5B hidden size, head dimension, layers
ARMS = [("gpt-5-mini", "qwen_llmopt_gpt5mini", [0, 1, 2, 4, 5, 6, 7, 8, 9]),
        ("gpt-5-mini, no semantics", "qwen_llmopt_gpt5mini_anon", [0, 1, 2, 4, 5, 6, 7, 8, 9]),
        ("gpt-5.6-luna", "qwen_llmopt_luna", [0, 1, 2, 4, 5, 6, 7, 8, 9]),
        ("DeepSeek-V4-Flash", "qwen_llmopt_deepseek", [0, 1, 2, 4, 5, 6, 7, 8, 9]),   # nine seeds as in Table tab:llm-opt
        ("EA", "qwen15b_ea_nogp", [0, 1, 2, 4, 5, 6, 7, 8, 9])]   # plain EA
WINDOWS = [("initial", 0, 16), ("first batch", 16, 24), ("17-80", 16, 80), ("81-160", 80, 160), ("161-320", 160, 320)]


def removed(frac, policy):
    return float(frac[np.arange(L), policy[:L]].mean())


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    records = records_dir(argv[0])
    RUNS = runs_path(records)
    t = json.load(open(flops_table_path(records, "qwen15b")))
    attn = 2 * h * hd * (np.asarray(t["heads"], float) + np.asarray(t["kv_groups"], float))
    frac = (attn[:, :1] - attn) / attn[:, :1]                       # (layers, levels): attention compute removed
    for label, group, seeds in ARMS:
        runs = [json.load(open(resolve_run(RUNS, group, s) / "results.json"))["entries"][:320] for s in seeds]
        cells = [np.mean([removed(frac, e["policy"]) for r in runs for e in r[a:b]]) for _, a, b in WINDOWS]
        best = np.mean([removed(frac, min(r, key=lambda e: e["kl"])["policy"]) for r in runs])
        print(f"{label:26s} " + "  ".join(f"{w}: {c:.2f}" for (w, _, _), c in zip(WINDOWS, cells)) + f"  | best allocation: {best:.2f}")


if __name__ == "__main__":
    main()
