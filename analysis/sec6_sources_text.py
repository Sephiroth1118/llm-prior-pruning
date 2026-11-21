"""Numbers in the text of Section 6.2: per-target mean relative gain over the EA (four seeds, as Table tab:matrix) and
same-family against LLM template; same rules and run groups as sec6_matrix_plain.py, whose definitions it imports.
Reads:  <records>/runs/<model>_{ea,llm,humandepth,own}_nogp_s<seed>/results.json and <model>_tmpl_<source>_nogp
        (or the _rk_s<seed> reruns), as sec6_matrix_plain.py
Writes: nothing; prints the numbers.
Usage: python sec6_sources_text.py <records>"""
import sys

import numpy as np

import sec6_matrix_plain as mp
from sec6_matrix_plain import MODELS, NAME, FAMILY, SOURCE, runs


def SEEDS(m):
    return [0, 1, 2, 4] if m == "qwen15b" else [0, 1, 2, 3]


arms = ["LLM", "tmpl:qwen15b", "tmpl:llama31", "tmpl:olmo7b", "depth", "own"]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    mp.setup(argv[0])
    for b in (80, 320):
        print(f"== {b} evaluations: mean relative gain over EA (%), wins/4")
        for m in MODELS:
            ea = runs(m, "EA", SEEDS(m))
            row = []
            for a in arms:
                rr = runs(m, a, SEEDS(m))
                if not rr:
                    row.append(f"{a}=--"); continue
                d = [(ea[s][b-1] - rr[s][b-1]) / ea[s][b-1] for s in rr]
                tag = "S" if a.startswith("tmpl:") and FAMILY[a[5:]] == FAMILY[m] else ("X" if a.startswith("tmpl:") else "")
                row.append(f"{a}{tag}={100*np.mean(d):+.1f}({sum(x>0 for x in d)})")
            print(f"{NAME[m]:13s} " + "  ".join(row))
    # same family vs LLM at 80, per target, seed wins
    print("== same family vs LLM template, seeds where same-family lower")
    for b in (80, 320):
        for m in MODELS:
            s = SOURCE[FAMILY[m]]
            if s == m: continue
            a, l = runs(m, f"tmpl:{s}", SEEDS(m)), runs(m, "LLM", SEEDS(m))
            print(b, NAME[m], sum(a[k][b-1] < l[k][b-1] for k in a), np.mean([a[k][b-1] for k in a]), np.mean([l[k][b-1] for k in l]))


if __name__ == "__main__":
    main()
