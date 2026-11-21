"""Section 6.3 cost numbers from the run records of Table tab:spi-main (plain EA): seconds per evaluation and wall-clock per
search, LLM requests, tokens and fees of structural prior injection, and calls and fees of the LLM as the optimizer.
Reads:  <records>/runs/<group>_s<seed>/results.json (or <group>_rk_s<seed>) for qwen15b_{ea,llm}_nogp,
        qwen_llmopt_gpt5mini, qwen_llmopt_gpt5mini_1k, llama31_{ea,llm}_nogp, llama_llmopt_gpt5mini
Writes: nothing; prints the cost lines.
Usage: python sec6_cost.py <records>"""
import json, statistics as st, sys
from records import records_dir, runs_path, resolve_run
P_IN, P_CACHE, P_OUT = 0.25, 0.025, 2.00     # gpt-5-mini, USD per 1M tokens: input, cached input, output
def fee(u):
    c = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return ((u["prompt_tokens"] - c) * P_IN + c * P_CACHE + u["completion_tokens"] * P_OUT) / 1e6
SEEDS = {"qwen": [0, 1, 2, 4, 5, 6, 7, 8, 9], "llama": list(range(8))}
GROUPS = {"qwen": {"EA": "qwen15b_ea_nogp", "SPI": "qwen15b_llm_nogp", "LLMopt320": "qwen_llmopt_gpt5mini", "LLMopt1k": "qwen_llmopt_gpt5mini_1k"},
          "llama": {"EA": "llama31_ea_nogp", "SPI": "llama31_llm_nogp", "LLMopt320": "llama_llmopt_gpt5mini"}}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    RUNS = runs_path(records_dir(argv[0]))
    for m, groups in GROUPS.items():
        print(f"== {m}")
        for arm, g in groups.items():
            recs = []
            for s in SEEDS[m]:
                p = resolve_run(RUNS, g, s) / "results.json"
                if p.exists():
                    recs.append((s, json.loads(p.read_text())))
            if not recs:
                print(" ", arm, "no runs"); continue
            n_entries = [len(r["entries"]) for _, r in recs]
            per_eval = [r["stats"]["wall_seconds"] / max(1, r["stats"].get("new_gpu_evaluations", len(r["entries"]))) for _, r in recs]
            wall = [r["stats"]["wall_seconds"] / 3600 for _, r in recs]
            line = f"  {arm:9s} n={len(recs)} entries={min(n_entries)}-{max(n_entries)} s/eval median {st.median(per_eval):.1f} wall h median {st.median(wall):.2f}"
            if arm == "SPI":
                reqs, fees, pin, pout = [], [], [], []
                for _, r in recs:
                    atts = [a for c in r["calls"] for a in c.get("attempts", []) if a.get("usage")]
                    reqs.append(len(atts)); fees.append(sum(fee(a["usage"]) for a in atts))
                    pin += [a["usage"]["prompt_tokens"] for a in atts]; pout += [a["usage"]["completion_tokens"] for a in atts]
                line += f" | requests/search mean {st.mean(reqs):.2f} (max {max(reqs)}) in tok mean {st.mean(pin):.0f} out {st.mean(pout):.0f} fee/search mean {st.mean(fees):.4f} max {max(fees):.4f}"
            if arm.startswith("LLMopt"):
                calls = [len([c for c in r["llm_calls"] if c.get("usage")]) for _, r in recs]
                fees = [sum(fee(c["usage"]) for c in r["llm_calls"] if c.get("usage")) for _, r in recs]
                steps = {}
                for _, r in recs:
                    for c in r["llm_calls"]:
                        steps[c.get("step")] = steps.get(c.get("step"), 0) + 1
                line += f" | calls/search mean {st.mean(calls):.1f} fee mean {st.mean(fees):.3f} steps {steps}"
            print(line)


if __name__ == "__main__":
    main()
