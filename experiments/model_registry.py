#!/usr/bin/env python3
"""Registry and preparation of the models of the study under the fineweb_exact protocol.

Per model (<work>/models/<key>/):
  fineweb/   the frozen FineWeb-Edu documents of the Qwen2.5-1.5B protocol re-tokenized with this model's tokenizer,
             packed like build_fineweb_calibration.py (whole documents joined by the tokenized "\\n\\n", consecutive
             2048-token windows, tail discarded); 8 search windows, up to 32 held-out windows.
             If the search documents fall short of 8 windows, held-out documents are moved to search; search and
             held-out stay disjoint.
  flops_table.json   active heads / KV groups / MLP dims per (layer, level), read from the level database
  teacher_search_8x2048_bf16.pt   dense-model logits on the 8 search windows
Level databases are built with struct_prune.py (see README, "Level databases"); `build_db` below runs it with the
settings of the Llama-3.1-8B database (FineWeb-Edu, 8,388,608 tokens, sequence 4096, bf16, 11 levels with the same
removal fractions) for the models whose registry entry has a "build" field. The databases of Qwen2.5-1.5B/3B/7B and
Llama-3.1-8B were built with the settings listed in their "build_note".

FLOPs: ExactFlops below takes the attention head dimension from the model config. It equals evoprune.flops.ExactFlops
whenever head_dim == hidden / heads (all models except Mistral-Nemo, where head_dim = 128 and hidden / heads = 160).
For Nemo the evaluator reports FLOPs with the wrong head dimension, so the runner replaces that number with ExactFlops
(flag "flops_override"); the evaluator's KL is unaffected because the mask synchronization reads self_attn.head_dim.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
from common import DB_DIR as DB, HF_DIR as HF, MODELS_DIR, PYTHON, REPO, RECOMMENDED_ENV, read_json, save_json, sha256  # noqa: E402

LLAMA8B_HEAD_FRACTION = [0, 4, 7, 11, 14, 18, 22, 26, 28, 31, 32]           # heads removed per level, of 32
MLP8192_COLS = [0, 832, 1632, 2464, 3264, 4096, 4928, 5728, 6560, 7360, 8192]  # ~10% steps, multiples of 32

REGISTRY = {
    "qwen15b": {"family": "qwen", "name": "Qwen2.5-1.5B", "hf_id": "Qwen/Qwen2.5-1.5B",
                "revision": "8faed761d45a263340a0528343f099c05c9a4323", "layers": 28,
                "spec": {"hidden": 1536, "intermediate": 8960, "heads": 12, "kv_heads": 2, "head_dim": 128},
                "model": str(HF / "Qwen2.5-1.5B"), "db": str(DB / "qwen15b"), "need_mib": 8000, "tokenizer_fast": False,
                "build_note": "13 levels; attention columns 0..1536 in steps of 128; MLP columns "
                              "[0, 736, 1504, 2240, 2976, 3744, 4480, 5216, 5984, 6720, 7456, 8224, 8960]; "
                              "calibration: prepare_qwen15b_calibration.py (1024 x 8192 tokens), sequence length 8192"},
    "qwen3b": {"family": "qwen", "name": "Qwen2.5-3B-Instruct", "hf_id": "Qwen/Qwen2.5-3B-Instruct", "layers": 36,
               "spec": {"hidden": 2048, "intermediate": 11008, "heads": 16, "kv_heads": 2, "head_dim": 128},
               "model": str(HF / "Qwen2.5-3B-Instruct"), "db": str(DB / "qwen3b"), "need_mib": 16000, "tokenizer_fast": False,
               "build_note": "17 levels; attention columns 0..2048 in steps of 128; MLP columns 0..11008 in steps of 688; "
                             "FineWeb-Edu, 8,388,608 tokens, sequence length 8192"},
    "qwen7b": {"family": "qwen", "name": "Qwen2.5-7B-Instruct", "hf_id": "Qwen/Qwen2.5-7B-Instruct", "layers": 28,
               "spec": {"hidden": 3584, "intermediate": 18944, "heads": 28, "kv_heads": 4, "head_dim": 128},
               "model": str(HF / "Qwen2.5-7B-Instruct"), "db": str(DB / "qwen7b"), "need_mib": 26000, "tokenizer_fast": False,
               "build_note": "11 levels; attention columns [0, 384, 768, 1152, 1536, 1792, 2176, 2560, 2944, 3200, 3584]; "
                             "MLP columns [0, 1920, 3840, 5760, 7680, 9472, 11392, 13312, 15104, 17024, 18944]; "
                             "FineWeb-Edu, 8,388,608 tokens, sequence length 8192"},
    "llama31": {"family": "llama", "name": "Llama-3.1-8B", "hf_id": "meta-llama/Llama-3.1-8B",
                "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b", "layers": 32,
                "spec": {"hidden": 4096, "intermediate": 14336, "heads": 32, "kv_heads": 8, "head_dim": 128},
                "model": str(HF / "Llama-3.1-8B"), "db": str(DB / "llama31"), "need_mib": 26000, "tokenizer_fast": True,
                "build": {"cols_attn": [0, 512, 896, 1408, 1792, 2304, 2816, 3328, 3584, 3968, 4096],
                          "cols_mlp": [0, 1440, 2880, 4288, 5728, 7168, 8608, 10048, 11456, 12896, 14336]}},
    "llama32_1b": {"family": "llama", "name": "Llama-3.2-1B", "hf_id": "meta-llama/Llama-3.2-1B", "layers": 16,
                   "spec": {"hidden": 2048, "intermediate": 8192, "heads": 32, "kv_heads": 8, "head_dim": 64},
                   "model": str(HF / "Llama-3.2-1B"), "db": str(DB / "llama32_1b"), "need_mib": 10000, "tokenizer_fast": True,
                   "build": {"cols_attn": [h * 64 for h in LLAMA8B_HEAD_FRACTION], "cols_mlp": MLP8192_COLS}},
    "llama32_3b": {"family": "llama", "name": "Llama-3.2-3B", "hf_id": "meta-llama/Llama-3.2-3B", "layers": 28,
                   "spec": {"hidden": 3072, "intermediate": 8192, "heads": 24, "kv_heads": 8, "head_dim": 128},
                   "model": str(HF / "Llama-3.2-3B"), "db": str(DB / "llama32_3b"), "need_mib": 16000, "tokenizer_fast": True,
                   # Llama-8B head-removal fractions scaled to 24 heads (round half up)
                   "build": {"cols_attn": [h * 128 for h in (0, 3, 5, 8, 11, 14, 17, 20, 21, 23, 24)], "cols_mlp": MLP8192_COLS}},
    "mistral7b": {"family": "mistral", "name": "Mistral-7B-v0.3", "hf_id": "mistralai/Mistral-7B-v0.3", "layers": 32,
                  "spec": {"hidden": 4096, "intermediate": 14336, "heads": 32, "kv_heads": 8, "head_dim": 128},
                  "model": str(HF / "Mistral-7B-v0.3"), "db": str(DB / "mistral7b"), "need_mib": 26000, "tokenizer_fast": True,
                  "pre_block": ["model.embed_tokens"],   # MistralModel (transformers 4.45) has no model.rotary_emb
                  # identical shapes to Llama-3.1-8B, so its database levels are reused exactly
                  "build": {"cols_attn": [0, 512, 896, 1408, 1792, 2304, 2816, 3328, 3584, 3968, 4096],
                            "cols_mlp": [0, 1440, 2880, 4288, 5728, 7168, 8608, 10048, 11456, 12896, 14336]}},
    # OLMo: multi-head attention (no grouping), so attention is 25% / 33% of block parameters
    "olmo1b": {"family": "olmo", "name": "OLMo-1B-0724-hf", "hf_id": "allenai/OLMo-1B-0724-hf", "layers": 16,
               "spec": {"hidden": 2048, "intermediate": 8192, "heads": 16, "kv_heads": 16, "head_dim": 128},
               "model": str(HF / "OLMo-1B-0724-hf"), "db": str(DB / "olmo1b"), "need_mib": 10000, "tokenizer_fast": True,
               "pre_block": ["model.embed_tokens"],   # OlmoModel (transformers 4.45) has no model.rotary_emb
               "tokenizer_compat": True,              # struct_prune.py asks for a slow tokenizer; GPT-NeoX has none
               # Llama-3.1-8B head-removal fractions scaled to 16 heads; 15.5 rounded down to keep 11 distinct levels
               "build": {"cols_attn": [h * 128 for h in (0, 2, 4, 6, 7, 9, 11, 13, 14, 15, 16)], "cols_mlp": MLP8192_COLS}},
    "olmo7b": {"family": "olmo", "name": "OLMo-7B-0724-hf", "hf_id": "allenai/OLMo-7B-0724-hf", "layers": 32,
               "spec": {"hidden": 4096, "intermediate": 11008, "heads": 32, "kv_heads": 32, "head_dim": 128},
               "model": str(HF / "OLMo-7B-0724-hf"), "db": str(DB / "olmo7b"), "need_mib": 26000, "tokenizer_fast": True,
               "pre_block": ["model.embed_tokens"], "tokenizer_compat": True,
               "build": {"cols_attn": [0, 512, 896, 1408, 1792, 2304, 2816, 3328, 3584, 3968, 4096],          # = Llama-3.1-8B levels
                         "cols_mlp": [0, 1120, 2208, 3296, 4384, 5504, 6624, 7712, 8800, 9888, 11008]}},     # Llama fractions x 11008
    "nemo12b": {"family": "mistral", "name": "Mistral-Nemo-Instruct-2407", "hf_id": "mistralai/Mistral-Nemo-Instruct-2407",
                "layers": 40,
                "spec": {"hidden": 5120, "intermediate": 14336, "heads": 32, "kv_heads": 8, "head_dim": 128},
                "model": str(HF / "Mistral-Nemo-Instruct-2407"), "db": str(DB / "nemo12b"), "need_mib": 38000,
                "tokenizer_fast": True, "flops_override": True},
}
LENGTH, SEARCH_WINDOWS, HELDOUT_WINDOWS = 2048, 8, 32


def paths(key):
    d = MODELS_DIR / key
    return {"dir": d, "fineweb": d / "fineweb", "flops": d / "flops_table.json",
            "teacher": d / "teacher_search_8x2048_bf16.pt", "control": MODELS_DIR / "control.json"}


class ExactFlops:
    """Real FLOPs reduction of a policy with an explicit attention head dimension (see module docstring)."""

    def __init__(self, table, spec):
        h, m = float(spec["hidden"]), float(spec["intermediate"])
        hd = float(spec.get("head_dim", spec["hidden"] // spec["heads"]))
        self.attn = 2 * h * hd * (np.asarray(table["heads"], float) + np.asarray(table["kv_groups"], float))
        self.mlp = 3 * h * np.asarray(table["mlp_dims"], float)
        self.L = self.attn.shape[0]
        self.dense_total = self.L * (2 * h * hd * (spec["heads"] + spec["kv_heads"]) + 3 * h * m)

    def __call__(self, levels):
        levels = np.atleast_2d(np.asarray(levels, dtype=np.int64))
        layers = np.arange(self.L)
        sparse = self.attn[layers, levels[:, :self.L]].sum(1) + self.mlp[layers, levels[:, self.L:]].sum(1)
        return np.clip(1.0 - sparse / self.dense_total, 0.0, 1.0)


def build_flops_table(key, layers=None):
    """Active query heads, KV groups and MLP dims per (layer, level), inferred as the evaluator's mask synchronization does:
    slices cast to bf16, a head (o_proj column block of head_dim) or MLP dim (down_proj column) is active when its norm
    exceeds 1e-8, and a KV group is active when any of its query heads is."""
    import torch
    r = REGISTRY[key]
    spec, db = r["spec"], r["db"]
    hd, group = spec["head_dim"], spec["heads"] // spec["kv_heads"]
    meta = torch.load(os.path.join(db, "metadata.pth"), map_location="cpu", weights_only=False)
    n_attn, n_mlp = len(meta["cols_removed_attn"]), len(meta["cols_removed_mlp"])

    def weight(module, layer, level):
        return torch.load(os.path.join(db, f"model.layers.{layer}.{module}", f"{level}.pth"), map_location="cpu").to(torch.bfloat16).float()

    table = {"heads": [], "kv_groups": [], "mlp_dims": []}
    for layer in (layers if layers is not None else range(r["layers"])):
        heads, groups, dims = [], [], []
        for level in range(n_attn):
            w = weight("self_attn.o_proj", layer, level)
            assert w.shape[1] == spec["heads"] * hd, (w.shape, spec)
            active = w.view(w.shape[0], spec["heads"], hd).norm(dim=(0, 2)) > 1e-8
            heads.append(int(active.sum()))
            groups.append(int(active.view(spec["kv_heads"], group).any(1).sum()))
        for level in range(n_mlp):
            dims.append(int((weight("mlp.down_proj", layer, level).norm(dim=0) > 1e-8).sum()))
        table["heads"].append(heads)
        table["kv_groups"].append(groups)
        table["mlp_dims"].append(dims)
    return table


def _pack(docs, tokenizer, count):
    separator = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
    tokens, used = [], []
    for d in docs:
        if len(tokens) >= count * LENGTH:
            break
        ids = tokenizer(d["text"], add_special_tokens=False)["input_ids"]
        if tokens:
            tokens.extend(separator)
        tokens.extend(ids)
        used.append({k: d[k] for k in ("row", "text_sha256")})
    full = min(count, len(tokens) // LENGTH)
    import torch
    windows = [torch.tensor([tokens[i:i + LENGTH]], dtype=torch.long) for i in range(0, full * LENGTH, LENGTH)]
    return windows, used, len(tokens) - full * LENGTH


def build_tokens(key):
    p = paths(key)
    if (p["fineweb"] / "manifest.json").exists():
        return
    if key == "qwen15b":
        raise SystemExit("run build_fineweb_calibration.py first: it freezes the documents of the protocol from Qwen2.5-1.5B")
    import torch
    from transformers import AutoTokenizer
    source = paths("qwen15b")["fineweb"] / "documents.json"
    docs = read_json(source)
    search_docs, heldout_docs = list(docs["search"]), list(docs["heldout"])
    fast = REGISTRY[key].get("tokenizer_fast", False)
    tokenizer = AutoTokenizer.from_pretrained(REGISTRY[key]["model"], use_fast=fast)
    moved = 0
    while True:
        search, used, tail = _pack(search_docs, tokenizer, SEARCH_WINDOWS)
        if len(search) == SEARCH_WINDOWS:
            break
        search_docs.append(heldout_docs.pop(0))      # same remedy as the LLaMA preparation
        moved += 1
    heldout, used_h, tail_h = _pack(heldout_docs, tokenizer, HELDOUT_WINDOWS)
    assert not {d["row"] for d in used} & {d["row"] for d in used_h}
    p["fineweb"].mkdir(parents=True, exist_ok=True)
    torch.save(search, p["fineweb"] / "search_tokens.pt")
    torch.save(heldout, p["fineweb"] / "heldout_tokens.pt")
    save_json(p["fineweb"] / "documents.json", {"search": used, "heldout": used_h})
    save_json(p["fineweb"] / "manifest.json", {
        "source_documents": str(source), "source_sha256": sha256(source), "tokenizer": REGISTRY[key]["model"],
        "tokenizer_fast": fast, "window_tokens": LENGTH,
        "packing": "whole documents joined by tokenized '\\n\\n', consecutive windows, tail discarded",
        "search": {"windows": len(search), "documents": len(used), "discarded_tail": tail},
        "heldout": {"windows": len(heldout), "documents": len(used_h), "discarded_tail": tail_h},
        "moved_heldout_documents_to_search": moved})


def build_flops(key):
    p = paths(key)
    if p["flops"].exists():
        return
    table = build_flops_table(key)
    p["dir"].mkdir(parents=True, exist_ok=True)
    save_json(p["flops"], table)


def build_teacher(key, device):
    p = paths(key)
    if p["teacher"].exists():
        return
    import torch
    from evoprune.simplified_candidate_utils import build_teacher_logits_cache
    tokens = torch.load(p["fineweb"] / "search_tokens.pt", map_location="cpu", weights_only=False)
    assert len(tokens) == SEARCH_WINDOWS
    cache = build_teacher_logits_cache(model_name_or_path=REGISTRY[key]["model"], input_ids_list=tokens, dtype="bfloat16",
                                       low_cpu_mem_usage=False, attn_implementation="sdpa", device=device, max_samples=SEARCH_WINDOWS)
    tmp = str(p["teacher"]) + ".tmp"
    torch.save(cache, tmp)
    os.replace(tmp, p["teacher"])


def db_complete(key):
    """struct_prune.py writes metadata.pth before pruning, so completeness means every level file of every layer."""
    import torch
    r = REGISTRY[key]
    db = Path(r["db"])
    if not (db / "metadata.pth").exists():
        return False
    meta = torch.load(db / "metadata.pth", map_location="cpu", weights_only=False)
    for layer in range(r["layers"]):
        for module, levels in (("self_attn.o_proj", meta["cols_removed_attn"]), ("mlp.down_proj", meta["cols_removed_mlp"])):
            d = db / f"model.layers.{layer}.{module}"
            if not all((d / f"{i}.pth").exists() for i in range(len(levels))):
                return False
    return True


def tokenizer_compat_dir(key):
    """struct_prune.py loads AutoTokenizer(use_fast=False). GPT-NeoX (OLMo) has only a fast tokenizer, so we give it a
    copy of the tokenizer files whose config names PreTrainedTokenizerFast (as Llama-3's own config does); the
    tokenizer.json, and therefore every token id, is unchanged."""
    import shutil
    src = Path(REGISTRY[key]["model"])
    dst = HF / f"{src.name}-tokenizer-compat"
    if not (dst / "tokenizer_config.json").exists():
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("tokenizer.json", "special_tokens_map.json"):
            shutil.copy2(src / f, dst / f)
        cfg = read_json(src / "tokenizer_config.json")
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        save_json(dst / "tokenizer_config.json", cfg)
    return dst


def build_db(key, gpus, port=29531):
    """Level database with struct_prune.py, Llama-3.1-8B settings, one or more GPUs."""
    import shutil
    r = REGISTRY[key]
    if "build" not in r:
        raise SystemExit(f"{key}: no generic build recipe; see the README (Level databases) and the registry's build_note")
    db = Path(r["db"])
    if db_complete(key):
        return
    if db.exists():                                   # partial output of a failed build; only ever inside our db/ folder
        assert db.resolve().parent == DB.resolve(), db
        shutil.rmtree(db)
    b = r["build"]
    db.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(Path(PYTHON).parent / "torchrun"), "--nnodes=1", f"--nproc-per-node={len(gpus)}", f"--master_port={port}",
           "struct_prune.py", "--model_name_or_path", r["model"],
           "--prunable_modules", ".*layers.*((q|k|v|o|gate|up|down)_proj)$",
           "--pre_block_modules", *r.get("pre_block", ["model.embed_tokens", "model.rotary_emb"]), "--block_modules", "model.layers",
           "--calibration_data", "fineweb_edu", "--calibration_tokens", "8388608", "--calibration_sequence_length", "4096",
           "--cols_removed_attn", *map(str, b["cols_attn"]), "--cols_removed_mlp", *map(str, b["cols_mlp"]),
           "--mlp_prune_name", "down_proj", "--attn_prune_name", "o_proj",
           "--low_cpu_mem_usage", "--cpu_offload_modules", "--cpu_offload_activations", "--verbose",
           "--dtype", "bfloat16", "--attn_implementation", "sdpa", "--save_dir", str(db), "--rows_in_parallel", "128"]
    if r.get("tokenizer_compat"):
        cmd += ["--tokenizer_name", str(tokenizer_compat_dir(key))]
    env = {**os.environ, **RECOMMENDED_ENV, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "NCCL_TIMEOUT": "10800",
           "PYTHONPATH": str(REPO)}
    log_dir = DB.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"build_db_{key}.log", "a") as log:
        subprocess.run(cmd, cwd=str(REPO / "experiments"), env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    assert db_complete(key), f"database incomplete after build: {db}"


def prepare(key, device="cuda:0"):
    build_tokens(key)
    build_flops(key)
    build_teacher(key, device)
    return {k: str(v) for k, v in paths(key).items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", choices=sorted(REGISTRY))
    parser.add_argument("--build-db", choices=sorted(REGISTRY))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=29531)
    args = parser.parse_args()
    if args.build_db:
        build_db(args.build_db, [int(g) for g in args.gpus.split(",")], args.port)
    if args.prepare:
        print(json.dumps(prepare(args.prepare, args.device), indent=1))
