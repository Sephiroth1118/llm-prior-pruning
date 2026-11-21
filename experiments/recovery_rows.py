#!/usr/bin/env python3
"""Documents recovery training must not read (CPU only). Writes <work>/recovery/excluded_rows_fineweb.json.

FineWeb-Edu sample-10BT, split train, is used three ways. Rows are row numbers of the full split.
  PPL of benchmark.py    evoprune/data_utils.get_data(train=False, seed=0): second half, shuffled, first 40 samples.
  search and held-out    the frozen documents of models/qwen15b/fineweb/documents.json, first half.
  recovery training      run_fullft_recovery.py without the new options reads the whole split, so it can reach both.
Training therefore uses --train_row_start 0 --train_row_end <half> --exclude_rows_path <this file>. --check also goes
through the documents training would read and compares their text with all excluded documents, which covers duplicates.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
from common import CALIBRATION, HF_DIR, WORK, save_json   # noqa: E402

OUT = WORK / "recovery" / "excluded_rows_fineweb.json"
LLAMA = str(HF_DIR / "Llama-3.1-8B")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def ppl_rows(dataset, tokenizer, tokens=163840, length=4096, samples=40):
    """Rows behind benchmark.py's PPL samples: _load_fineweb_edu_dataset(False, 0) + _collect_fineweb_tokens, first 40 samples."""
    half = dataset.num_rows // 2
    order = np.random.default_rng(0).permutation(dataset.num_rows - half)      # what Dataset.shuffle(seed=0) draws
    rows, count, left = [], 0, tokens
    for i in order:
        row = half + int(i)
        ids = tokenizer(dataset[row]["text"], add_special_tokens=False).input_ids[:left]
        rows.append(row)
        pieces = max(1, (len(ids) - 1) // length + 1)
        count += pieces
        left -= len(ids)
        if count >= samples or left <= 0:
            return rows, half
    raise RuntimeError("not enough documents")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", type=int, default=0, help="documents of the training order to compare by text")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA)
    frozen = json.loads((CALIBRATION / "documents.json").read_text())
    search_held = sorted({d["row"] for part in ("search", "heldout") for d in frozen[part]})
    wrong = [d["row"] for part in ("search", "heldout") for d in frozen[part] if digest(dataset[d["row"]]["text"]) != d["text_sha256"]]
    ppl, half = ppl_rows(dataset, tokenizer)
    assert not wrong, f"rows of documents.json do not match this copy of the dataset: {wrong}"
    assert max(search_held) < half <= min(ppl)
    report = {"dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "split": "train", "num_rows": dataset.num_rows,
              "train_row_start": 0, "train_row_end": half, "rows": search_held,
              "rows_note": "search and held-out documents of the KL protocol, all below train_row_end",
              "ppl_rows": ppl, "ppl_rows_note": "excluded by train_row_end; listed for the record (Llama-3.1-8B tokenizer)",
              "documents_json_rows_verified_by_text_sha256": len(search_held)}
    if args.check:
        excluded = {digest(dataset[r]["text"]): r for r in search_held + ppl}
        keep = np.setdiff1d(np.arange(0, half), np.asarray(search_held))
        assert not set(keep[np.isin(keep, ppl)].tolist()) and not np.isin(keep, search_held).any()
        order = np.random.default_rng(args.seed).permutation(len(keep))[:args.check]
        started, hits, tokens = time.time(), [], 0
        for i in order:
            text = dataset[int(keep[i])]["text"]
            tokens += len(text)
            h = digest(text)
            if h in excluded:
                hits.append({"row": int(keep[i]), "same_text_as_row": excluded[h]})
        sample = [len(tokenizer(dataset[int(keep[i])]["text"], add_special_tokens=False).input_ids) for i in order[:2000]]
        report["check"] = {"documents_compared": int(len(order)), "same_text_hits": hits, "seconds": time.time() - started,
                           "seed": args.seed, "mean_tokens_per_document_first_2000": float(np.mean(sample)),
                           "estimated_tokens_covered": float(np.mean(sample)) * len(order)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_json(OUT, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "ppl_rows")}, indent=1))


if __name__ == "__main__":
    main()
