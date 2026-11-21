#!/usr/bin/env python3
"""Evaluate frozen per-run champions of Qwen2.5-1.5B on all FineWeb held-out windows, with resumable blocks.

The champions file is written by select_champions.py.

    python heldout_qwen15b.py --champions <work>/reports/heldout_final_input/champions.json --out <dir> --devices cuda:0
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
from common import CALIBRATION, MODELS_DIR   # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def candidate_key(row):
    """Seeds repeat across arms in a paired experiment; cache by arm AND seed."""
    return row.get("candidate_id", f"{row.get('arm', 'none')}/seed{row['seed']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    from evoprune.EA import Policy
    from evoprune.parallel_evaluator import ParallelEvaluator
    from evoprune.simplified_candidate_utils import build_teacher_logits_cache

    started = time.time()
    calibration = CALIBRATION
    documents = json.loads((calibration / "documents.json").read_text())
    for field in ("row", "text_sha256"):
        assert not ({d[field] for d in documents["search"]} & {d[field] for d in documents["heldout"]}), field
    tokens = torch.load(calibration / "heldout_tokens.pt", map_location="cpu", weights_only=False)
    assert len(tokens) == 32 and all(t.numel() == 2048 for t in tokens), "Expected all 32 heldout windows"
    selected = json.loads(args.champions.read_text())["champions"]
    count_selected = len(selected)
    assert count_selected and len({candidate_key(c) for c in selected}) == count_selected
    configs = []
    for c in selected:
        assert sha256(Path(c["source"])) == c["source_sha256"], "Source changed after selection"
        result = json.loads(Path(c["source"]).read_text())
        assert result["config"]["protocol"]["name"] == "fineweb_exact"
        best = min(result["entries"], key=lambda e: e["kl"])
        assert best["policy"] == c["policy"] and best["id"] == c["entry_id"]
        configs.append(result["config"]["model"])
    assert all(c == configs[0] for c in configs)
    paths = configs[0]
    devices = args.devices.split(",")
    manifest = {"champions_sha256": sha256(args.champions), "tokens_sha256": sha256(calibration / "heldout_tokens.pt"),
                "documents_sha256": sha256(calibration / "documents.json"), "windows": 32,
                "input_tokens": 32 * 2048, "expected_kl_tokens": 32 * 2047,
                "block_windows": 8, "dtype": "bfloat16", "models": paths,
                "document_row_overlap": 0, "document_text_hash_overlap": 0,
                "usage": "diagnostic only; no search or descriptor-selection feedback"}
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Cannot resume a different evaluation"
    else:
        save(manifest_path, manifest)
    cache_dir = MODELS_DIR / "qwen15b"
    cache_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
    for start in range(0, len(tokens), 8):
        block_id = start // 8
        part = tokens[start:start+8]
        cache = cache_dir / f"qwen15b_fineweb_heldout_{manifest['tokens_sha256'][:12]}_block{block_id}_bf16.pt"
        block_path = args.out / f"block{block_id}.json"
        rows = json.loads(block_path.read_text()) if block_path.exists() else []
        done = {candidate_key(row) for row in rows}
        todo = [c for c in selected if candidate_key(c) not in done]
        if todo:
            if not cache.exists():
                print(f"Building teacher cache for block {block_id} (8 windows)", flush=True)
                logits = build_teacher_logits_cache(paths["model"], part, dtype="bfloat16", low_cpu_mem_usage=False,
                                                     attn_implementation="sdpa", device=devices[0], max_samples=None)
                assert len(logits) == 8
                tmp = cache.with_suffix(".tmp")
                torch.save(logits, tmp)
                os.replace(tmp, cache)
                del logits
                gc.collect()
                torch.cuda.empty_cache()
            evaluator = ParallelEvaluator(
                devices=devices, teacher_cache_path=str(cache), model_name_or_path=paths["model"], db_path=paths["db"],
                eval_data=part, ppl_eval_data=None, dtype="bfloat16", low_cpu_mem_usage=False,
                attn_implementation="sdpa", sync_zero_masks=True, target_band_width=.02, reward_mode="raw_kl")
            try:
                for offset in range(0, len(todo), len(devices)):
                    batch = todo[offset:offset+len(devices)]
                    policies = [Policy.from_arrays(c["policy"][:28], c["policy"][28:]).to_db_policy() for c in batch]
                    results = evaluator.evaluate_candidates(policies, target_flops_reduction=.5,
                                                            compute_ppl=False, max_samples=None)
                    for champion, result in zip(batch, results):
                        assert result["error_message"] is None, result["error_message"]
                        metrics = result["metrics"]
                        assert metrics.num_samples == 8 and metrics.num_kl_tokens == 8 * 2047
                        kl = float(metrics.kl_divergence_from_dense)
                        assert math.isfinite(kl) and kl >= 0
                        assert abs(result["flops_reduction"] - champion["flops_reduction"]) < 1e-12
                        rows.append({"seed": champion["seed"], "candidate_id": candidate_key(champion),
                                     "block": block_id, "kl": kl,
                                     "num_samples": metrics.num_samples, "num_kl_tokens": metrics.num_kl_tokens,
                                     "flops_reduction": result["flops_reduction"]})
                    save(block_path, rows)
                    print(f"block {block_id}: {len(rows)}/{count_selected} champions", flush=True)
            finally:
                evaluator.close()
        assert len(rows) == count_selected and {candidate_key(r) for r in rows} == {candidate_key(c) for c in selected}
        blocks.extend(rows)
    output = []
    for c in selected:
        rows = [r for r in blocks if candidate_key(r) == candidate_key(c)]
        count = sum(r["num_kl_tokens"] for r in rows)
        assert len(rows) == 4 and count == 32 * 2047
        kl = sum(r["kl"] * r["num_kl_tokens"] for r in rows) / count
        output.append({**c, "heldout_kl": kl, "gap": kl - c["search_kl"], "kl_tokens": count})
    # Mid-ranks support ties without introducing arbitrary ordering.
    def rank(x):
        return np.asarray([np.sum(x < v) + (np.sum(x == v) - 1) / 2 for v in x])
    x, y = np.asarray([r["search_kl"] for r in output]), np.asarray([r["heldout_kl"] for r in output])
    summary = {"champions": count_selected, "mean_search_kl": float(x.mean()), "mean_heldout_kl": float(y.mean()),
               "mean_gap": float((y-x).mean()), "spearman": float(np.corrcoef(rank(x), rank(y))[0, 1]),
               "search_best_seed": selected[int(x.argmin())]["seed"],
               "heldout_best_seed": selected[int(y.argmin())]["seed"],
               "search_best_candidate": candidate_key(selected[int(x.argmin())]),
               "heldout_best_candidate": candidate_key(selected[int(y.argmin())]), "wall_seconds": time.time()-started}
    save(args.out / "results.json", {"manifest": manifest, "summary": summary, "champions": output})
    lines = ["# Held-out evaluation of the champions", "",
             "The champion of each seed is selected by search KL alone; all 32 x 2048 held-out windows are used, 65,504 positions enter the KL.",
             "Search and held-out documents share neither row numbers nor text hashes. The result is diagnostic only and feeds back into no search.", "",
             f"Spearman rank correlation of search and held-out KL: {summary['spearman']:.4f}; mean {x.mean():.6f} -> {y.mean():.6f}.",
             "Search and held-out sets consist of different documents, so the mean difference mixes data differences and selection effects.", "",
             "| run | search KL | held-out KL | difference |", "|---|---:|---:|---:|"]
    for row in sorted(output, key=lambda r: r["search_kl"]):
        lines.append(f"| {candidate_key(row)} | {row['search_kl']:.6f} | {row['heldout_kl']:.6f} | {row['gap']:+.6f} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
