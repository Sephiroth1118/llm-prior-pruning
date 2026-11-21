"""Appendix table tab:downstream-seeds: downstream results of the best allocations of all searches with the LLM template
(plain EA). Avg = mean of the eight task scores in percent; the selected allocation is the one of Tables tab:llama-downstream
and tab:cross-downstream (lowest held-out KL).
Reads:  <records>/benchmark/<model>/compression_<model>_llm_nogp*/benchmark_results.json for llama31, qwen3b, qwen7b
Writes: nothing; prints the LaTeX table rows.
Usage: python sec6_downstream_seeds.py <records>"""
import glob, json, statistics as st, sys

from records import records_dir, benchmark_path

SELECTED = {"llama31": "llama31_llm_nogp_s7", "qwen3b": "qwen3b_llm_nogp_s3", "qwen7b": "qwen7b_llm_nogp_s2"}
NAME = {"llama31": "Llama-3.1-8B", "qwen3b": "Qwen2.5-3B", "qwen7b": "Qwen2.5-7B"}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    BENCH = str(benchmark_path(records_dir(argv[0])))
    for m in SELECTED:
        rows = {}
        for p in sorted(glob.glob(f"{BENCH}/{m}/compression_{m}_llm_nogp*/benchmark_results.json")):
            r = json.load(open(p))["compressed"]
            rows[p.split("/")[-2].replace("compression_", "").replace("_rk_", "_")] = (r["ppl"], 100 * st.mean(r["task_scores"].values()))
        sel = rows[SELECTED[m]]
        ppl, avg = [v[0] for v in rows.values()], [v[1] for v in rows.values()]
        print(f"{NAME[m]} & {len(rows)} & {sel[0]:.2f} & ${st.mean(ppl):.2f}\\pm{st.stdev(ppl):.2f}$ & {sel[1]:.2f} & ${st.mean(avg):.2f}\\pm{st.stdev(avg):.2f}$ \\\\")


if __name__ == "__main__":
    main()
