#!/usr/bin/env python3
"""Select the champion of every search run (the allocation with the lowest search KL) and write the champions file that
heldout_qwen15b.py evaluates on the held-out windows. Held-out divergence is used once afterwards, to choose among champions;
it never feeds back into a search.

    python select_champions.py --groups qwen15b_llm_nogp,qwen15b_ea_nogp --seeds 0,1,2,3 --out <file>

Run records are <work>/runs/<group>_s<seed>/results.json; the champion is the entry with the lowest "kl" of a finished run.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
from common import RUNS_DIR, WORK, read_json, save_json, sha256   # noqa: E402


def champions(groups, seeds, budget=1000):
    rows = []
    for group in groups:
        for seed in seeds:
            path = RUNS_DIR / f"{group}_s{seed}" / "results.json"
            if not path.exists():
                continue
            result = read_json(path)
            entries = result["entries"]
            if not result.get("final", True) or len(entries) < budget:
                continue
            best = min(entries[:budget], key=lambda e: e["kl"])
            rows.append({"candidate_id": f"{group}/seed{seed}", "arm": group, "seed": seed, "source": str(path),
                         "source_sha256": sha256(path), "entry_id": best["id"], "policy": best["policy"],
                         "search_kl": best["kl"], "flops_reduction": best["flops_reduction"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--groups", required=True, help="comma-separated run groups")
    parser.add_argument("--seeds", required=True, help="comma-separated seeds")
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=WORK / "reports" / "heldout_final_input" / "champions.json")
    args = parser.parse_args()
    rows = champions(args.groups.split(","), [int(s) for s in args.seeds.split(",")], args.budget)
    save_json(args.out, {"selection": "minimum search KL within the budget; held-out never selects", "champions": rows})
    print(f"{len(rows)} champions -> {args.out}")


if __name__ == "__main__":
    main()
