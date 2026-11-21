#!/usr/bin/env python3
"""Freeze FineWeb-Edu calibration windows (search and held-out) for the search protocol.

Documents come from the first half of FineWeb-Edu sample-10BT, loaded exactly as the final benchmark loads the dataset.
The benchmark's PPL uses the second half (evoprune/data_utils.py, train=False), so calibration and final evaluation never
share a document. Whole documents are packed with a "\\n\\n" separator into 2048-token windows: the search windows
first, then disjoint held-out windows.

    python build_fineweb_calibration.py --tokenizer <Qwen2.5-1.5B model dir> --out <work>/models/qwen15b/fineweb

The result for Qwen2.5-1.5B is the source of the documents for every other model (model_registry.py re-tokenizes them).
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

DATA_SEED = 2026100601
LENGTH = 2048


def pack(dataset, rows, tokenizer, count, start):
    """Pack whole documents from rows[start:] into `count` windows; return windows, used records, next index."""
    separator = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
    tokens, records, index = [], [], start
    while len(tokens) < count * LENGTH:
        row = rows[index]
        text = dataset[row]["text"]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if tokens:
            tokens.extend(separator)
        tokens.extend(ids)
        records.append({"row": row, "tokens": len(ids), "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "text": text})
        index += 1
    windows = [torch.tensor([tokens[i:i + LENGTH]], dtype=torch.long) for i in range(0, count * LENGTH, LENGTH)]
    return windows, records, index, len(tokens) - count * LENGTH


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--search-windows", type=int, default=8)
    parser.add_argument("--heldout-windows", type=int, default=32)
    args = parser.parse_args()
    out = Path(args.out)
    if (out / "manifest.json").exists():
        raise SystemExit(f"{out} is frozen; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")
    n = dataset.num_rows
    rows = random.Random(DATA_SEED).sample(range(n // 2), 4096)   # first half only
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    search, search_docs, nxt, search_tail = pack(dataset, rows, tokenizer, args.search_windows, 0)
    heldout, heldout_docs, _, heldout_tail = pack(dataset, rows, tokenizer, args.heldout_windows, nxt)
    assert not {d["row"] for d in search_docs} & {d["row"] for d in heldout_docs}
    assert max(d["row"] for d in search_docs + heldout_docs) < n // 2

    torch.save(search, out / "search_tokens.pt")
    torch.save(heldout, out / "heldout_tokens.pt")
    with open(out / "documents.json", "w") as handle:
        json.dump({"search": search_docs, "heldout": heldout_docs}, handle)
    manifest = {
        "dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "split": "train",
        "cache_files": sorted({f["filename"] for f in dataset.cache_files}),
        "num_rows": n, "rows_used_from": [0, n // 2], "final_benchmark_rows": [n // 2, n],
        "data_seed": DATA_SEED, "tokenizer": os.path.abspath(args.tokenizer), "window_tokens": LENGTH,
        "search": {"windows": args.search_windows, "documents": len(search_docs), "discarded_tail_tokens": search_tail},
        "heldout": {"windows": args.heldout_windows, "documents": len(heldout_docs), "discarded_tail_tokens": heldout_tail},
        "packing": "whole documents joined by the tokenized '\\n\\n'; only the last document's tail is discarded",
    }
    with open(out / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=1)
    print(json.dumps({k: manifest[k] for k in ("num_rows", "rows_used_from", "search", "heldout")}))


if __name__ == "__main__":
    main()
