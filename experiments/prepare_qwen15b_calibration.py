#!/usr/bin/env python3
"""Pack 8,388,608 FineWeb-Edu tokens (1024 sequences of 8192) from a pinned shard, as calibration data for building the
Qwen2.5-1.5B level database with struct_prune.py.

    python prepare_qwen15b_calibration.py --model <Qwen2.5-1.5B dir> --out <dir>      # writes <dir>/calibration.pt
"""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"     # dataset revision
SHARD = "sample/10BT/000_00000.parquet"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="Qwen2.5-1.5B directory (tokenizer)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    target = out / "calibration.pt"
    assert not target.exists(), "Do not overwrite prepared calibration"
    source = hf_hub_download("HuggingFaceFW/fineweb-edu", SHARD, repo_type="dataset", revision=REVISION,
                             local_dir=str(out / "calibration_source"))
    table = pq.read_table(source, columns=["text"])
    generator = torch.Generator().manual_seed(0)
    order = torch.randperm(len(table), generator=generator).tolist()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    samples, pending, documents = [], [], 0
    for index in order:
        pending.extend(tokenizer(table["text"][index].as_py(), add_special_tokens=False)["input_ids"])
        documents += 1
        while len(pending) >= 8192 and len(samples) < 1024:
            samples.append(torch.tensor([pending[:8192]], dtype=torch.long))
            del pending[:8192]
        if len(samples) == 1024:
            break
    assert len(samples) == 1024 and sum(t.numel() for t in samples) == 8388608
    torch.save(samples, target)
    metadata = {"dataset": "HuggingFaceFW/fineweb-edu", "revision": REVISION, "shard": SHARD,
                "shard_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "documents": documents, "sequence_length": 8192, "sequences": 1024, "tokens": 8388608,
                "token_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "sampling": "Seed 0 torch permutation within the first sample-10BT shard; concatenate documents without additional "
                            "special tokens into full blocks."}
    (out / "calibration_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
