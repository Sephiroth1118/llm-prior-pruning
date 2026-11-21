"""Section 5.1: macro-structure of the best allocation found within 320 evaluations, mean over seeds (CPU only, read-only).
Eight cells: attention quarters 1-4, then MLP quarters 1-4. Seeds as in Table tab:spi-main, those that have finished.
Reads:  <records>/runs/<group>_s<seed>/results.json for qwen_llmopt_{gpt5mini,deepseek,luna}, qwen15b_ea_nogp,
        llama_llmopt_{gpt5mini,deepseek,luna}, llama31_ea_nogp;
        <records>/flops/qwen15b.json and <records>/flops/llama31.json
Writes: nothing; prints the mean removed fraction per cell.
Usage: python sec5_macro_profile.py <records>"""
import json, sys
import numpy as np

from records import records_dir, runs_path, flops_table_path

SPECS = {"qwen15b": (1536, 128), "llama31": (4096, 128)}  # hidden, head_dim (as in sec7_analysis.py)
SEEDS = {"qwen15b": (0, 1, 2, 4, 5, 6, 7, 8, 9), "llama31": range(8)}
GROUPS = {"qwen15b": {"gpt-5-mini": "qwen_llmopt_gpt5mini", "DeepSeek": "qwen_llmopt_deepseek", "Luna": "qwen_llmopt_luna", "EA": "qwen15b_ea_nogp"},
          "llama31": {"gpt-5-mini": "llama_llmopt_gpt5mini", "DeepSeek": "llama_llmopt_deepseek", "Luna": "llama_llmopt_luna", "EA": "llama31_ea_nogp"}}


def fractions(records, model):
    h, hd = SPECS[model]
    t = json.load(open(flops_table_path(records, model)))
    attn = 2 * h * hd * (np.asarray(t["heads"], float) + np.asarray(t["kv_groups"], float))
    mlp = 3 * h * np.asarray(t["mlp_dims"], float)
    costs = np.concatenate([attn, mlp])
    L = attn.shape[0]
    cells = np.tile(np.minimum(np.arange(L) * 4 // L, 3), 2); cells[L:] += 4
    return (costs[:, :1] - costs) / costs[:, :1], cells


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    records = records_dir(argv[0])
    RUNS = runs_path(records)
    for model, groups in GROUPS.items():
        frac, cells = fractions(records, model)
        for name, group in groups.items():
            rows = []
            for s in SEEDS[model]:
                p = RUNS / f"{group}_s{s}" / "results.json"
                r = json.loads(p.read_text()) if p.exists() else {}
                if not r.get("final") or len(r["entries"]) < 320:
                    continue
                best = min(r["entries"][:320], key=lambda e: e["kl"])["policy"]
                x = frac[np.arange(frac.shape[0]), best]
                rows.append([x[cells == c].mean() for c in range(8)])
            m = np.mean(rows, axis=0)
            print(f"{model:8s} {name:10s} seeds={len(rows)} attn " + " ".join(f"{v:.3f}" for v in m[:4]) +
                  " | mlp " + " ".join(f"{v:.3f}" for v in m[4:]) + f"  MLP range {m[4:].min():.2f}-{m[4:].max():.2f}")


if __name__ == "__main__":
    main()
