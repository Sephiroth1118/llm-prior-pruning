"""Evaluation caches: the union of the eval_cache.jsonl files of all runs in <work>/runs (plus the directories listed in
the environment variable SPI_EVAL_CACHES, os.pathsep-separated), for one model.

Same protocol everywhere: FineWeb search windows (8 x 2048), raw KL, the same teacher cache and the same database.
A policy that appears in several caches must have identical numbers; any disagreement aborts the run rather than
silently picking one. Reusing a cached value never changes a result; it only avoids re-evaluating the same allocation.
"""
import json
import math
import os
from pathlib import Path

from common import RUNS_DIR

PROTOCOL, MODEL = "fineweb_exact", "qwen15b"
KEYS = ("kl", "reward", "flops_reduction", "error")


def _normalize(result):
    row = {k: result.get(k) for k in KEYS}
    row["kl"], row["reward"], row["flops_reduction"] = float(row["kl"]), float(row["reward"]), float(row["flops_reduction"])
    return row


def cache_files(include_ours=True):
    files = []
    for directory in filter(None, os.environ.get("SPI_EVAL_CACHES", "").split(os.pathsep)):
        files += sorted(Path(directory).rglob("eval_cache.jsonl"))
    if include_ours:
        files += sorted(RUNS_DIR.rglob("eval_cache.jsonl"))
    return files


def load_global_cache(include_ours=True, tolerance=1e-9, model=MODEL):
    cache, origin, conflicts = {}, {}, []
    for path in cache_files(include_ours):
        with open(path, "rb") as handle:
            for line in handle:
                if not line.endswith(b"\n"):
                    continue                     # a torn final append of a live run; ignore it
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("protocol") != PROTOCOL or record.get("model") != model:
                    continue
                result = record["result"]
                if result.get("error") or not math.isfinite(float(result["kl"])):
                    continue
                policy = tuple(int(v) for v in record["policy"])
                row = _normalize(result)
                previous = cache.get(policy)
                if previous is None:
                    cache[policy], origin[policy] = row, str(path)
                elif (abs(previous["kl"] - row["kl"]) > tolerance or
                      abs(previous["flops_reduction"] - row["flops_reduction"]) > tolerance):
                    conflicts.append({"policy": list(policy), "a": origin[policy], "b": str(path),
                                      "kl_a": previous["kl"], "kl_b": row["kl"]})
    if conflicts:
        raise ValueError(f"{len(conflicts)} cached policies disagree across caches, e.g. {conflicts[0]}")
    return cache


if __name__ == "__main__":
    import time
    t0 = time.time()
    c = load_global_cache()
    print(json.dumps({"policies": len(c), "files": len(cache_files()), "seconds": round(time.time() - t0, 1)}))
