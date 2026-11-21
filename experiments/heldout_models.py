"""Held-out divergence of search champions on Llama-3.1-8B, Qwen2.5-3B and Qwen2.5-7B.

Generalizes heldout_qwen15b.py (Qwen2.5-1.5B only) to the models of the compression tables. The champion of a run is its
minimum-search-KL allocation; the held-out windows (documents disjoint from the search windows) never select anything
during a search. They are used once, to choose the single checkpoint per model that the downstream benchmark reports
(lowest held-out KL among the champions of the method). Blocks of 8 windows, one teacher-logit cache per block, resumable.

Usage: heldout_models.py --model llama31 --runs run_a,run_b,... --out <dir> --device cuda:0
"""
import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
from common import WORK  # noqa: E402
import model_registry  # noqa: E402

# model weights, level database, layers, held-out windows, teacher-cache folder
MODELS = {key: {"model": spec["model"], "db": spec["db"], "layers": spec["layers"],
                "tokens": model_registry.paths(key)["fineweb"] / "heldout_tokens.pt", "cache": model_registry.paths(key)["dir"]}
          for key, spec in model_registry.REGISTRY.items()}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--runs", required=True, help="comma-separated run names under <work>/runs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    from evoprune.EA import Policy
    from evoprune.parallel_evaluator import ParallelEvaluator
    from evoprune.simplified_candidate_utils import build_teacher_logits_cache

    started = time.time()
    spec = MODELS[args.model]
    layers = spec["layers"]
    tokens = torch.load(spec["tokens"], map_location="cpu", weights_only=False)
    assert all(t.numel() == 2048 for t in tokens)
    champions = []
    for name in args.runs.split(","):
        path = WORK / "runs" / name / "results.json"
        result = json.loads(path.read_text())
        assert result.get("final"), name
        best = min(result["entries"], key=lambda e: e["kl"])
        champions.append({"candidate_id": name, "source": str(path), "source_sha256": sha256(path), "entry_id": best["id"],
                          "policy": best["policy"], "search_kl": best["kl"], "flops_reduction": best["flops_reduction"]})
    manifest = {"model": args.model, "runs": args.runs.split(","), "tokens_sha256": sha256(spec["tokens"]),
                "windows": len(tokens), "block_windows": 8, "dtype": "bfloat16",
                "selection": "champion = minimum search KL of a run; held-out KL chooses the reported checkpoint"}
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Cannot resume a different evaluation"
    else:
        save(manifest_path, manifest)
    rows_all = []
    for start in range(0, len(tokens), 8):
        block_id = start // 8
        part = tokens[start:start + 8]
        cache = spec["cache"] / f"teacher_heldout_{manifest['tokens_sha256'][:12]}_block{block_id}_bf16.pt"
        block_path = args.out / f"block{block_id}.json"
        rows = json.loads(block_path.read_text()) if block_path.exists() else []
        todo = [c for c in champions if c["candidate_id"] not in {r["candidate_id"] for r in rows}]
        if todo:
            if not cache.exists():
                print(f"Building teacher cache for block {block_id} ({len(part)} windows)", flush=True)
                logits = build_teacher_logits_cache(spec["model"], part, dtype="bfloat16", low_cpu_mem_usage=False,
                                                    attn_implementation="sdpa", device=args.device, max_samples=None)
                assert len(logits) == len(part)
                tmp = cache.with_suffix(".tmp")
                torch.save(logits, tmp)
                os.replace(tmp, cache)
                del logits
                gc.collect()
                torch.cuda.empty_cache()
            evaluator = ParallelEvaluator(
                devices=[args.device], teacher_cache_path=str(cache), model_name_or_path=spec["model"], db_path=spec["db"],
                eval_data=part, ppl_eval_data=None, dtype="bfloat16", low_cpu_mem_usage=False,
                attn_implementation="sdpa", sync_zero_masks=True, target_band_width=.02, reward_mode="raw_kl")
            try:
                for champion in todo:
                    policy = Policy.from_arrays(champion["policy"][:layers], champion["policy"][layers:]).to_db_policy()
                    result = evaluator.evaluate_candidates([policy], target_flops_reduction=.5, compute_ppl=False,
                                                           max_samples=None)[0]
                    assert result["error_message"] is None, result["error_message"]
                    metrics = result["metrics"]
                    assert metrics.num_samples == len(part) and metrics.num_kl_tokens == len(part) * 2047
                    kl = float(metrics.kl_divergence_from_dense)
                    assert math.isfinite(kl) and kl >= 0
                    rows.append({"candidate_id": champion["candidate_id"], "block": block_id, "kl": kl,
                                 "num_kl_tokens": metrics.num_kl_tokens, "flops_reduction": result["flops_reduction"]})
                    save(block_path, rows)
                    print(f"block {block_id}: {len(rows)}/{len(champions)} champions", flush=True)
            finally:
                evaluator.close()
        rows_all.extend(rows)
    output = []
    for c in champions:
        rows = [r for r in rows_all if r["candidate_id"] == c["candidate_id"]]
        count = sum(r["num_kl_tokens"] for r in rows)
        assert count == len(tokens) * 2047
        kl = sum(r["kl"] * r["num_kl_tokens"] for r in rows) / count
        output.append({**c, "heldout_kl": kl, "gap": kl - c["search_kl"], "kl_tokens": count})
    best = min(output, key=lambda r: r["heldout_kl"])
    summary = {"model": args.model, "champions": len(output), "heldout_best": best["candidate_id"],
               "search_best": min(output, key=lambda r: r["search_kl"])["candidate_id"],
               "mean_search_kl": float(np.mean([r["search_kl"] for r in output])),
               "mean_heldout_kl": float(np.mean([r["heldout_kl"] for r in output])), "wall_seconds": time.time() - started}
    save(args.out / "results.json", {"manifest": manifest, "summary": summary, "champions": output})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
