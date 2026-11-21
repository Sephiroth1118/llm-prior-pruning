#!/usr/bin/env python3
"""Inference speed of the Llama-3.1-8B rows under one implementation .

speed.py ran three code bases (HF for Dense, the ZipLM compact format for four rows, DarwinLM's own code), and
DarwinLM's code leaves its KV cache empty during step-by-step decoding. Here every row is built the same way: the dense
Llama-3.1-8B is sliced by zipprune.ZipLM.compact.compact_model to the row's per-layer shape (kept query heads, kept MLP
dimensions) and timed with the same compact attention, whose cached decoding matches a full forward pass. Speed depends
only on these shapes, not on the weight values. Shapes: Dense = everything kept; Ours, Uniform, ZipLM, LLM-Pruner =
their compact manifests; DarwinLM = its released checkpoint's layout (config lists the pruned heads; MLP width from the
weight shapes; its unused K/V heads are dropped as for every other row). "Dense (HF)" is the unmodified HF model, as a
reference for the implementation gap. Measurements and order as in speed.py; each row is also checked once for
cached-vs-full-forward agreement.
With --gqa every compact row uses compact_gqa.GQACompactAttention instead (no per-step key/value copy),
and round 0 also records the relative difference of its logits from the unconverted compact model.
With --rows and --llmpruner-official: time only the named rows, and add "LLM-Pruner (official)", the official
LLM-Pruner model (<work>/baselines/llmpruner_official/compact_model); Dense and the GQA-corrected LLM-Pruner serve to check
that a session reproduces earlier numbers.
Usage: CUDA_VISIBLE_DEVICES=<gpu> python speed_unified.py --out <json> [--rounds 3] [--gqa] [--rows A,B] [--llmpruner-official]
"""
import argparse
import copy
import gc
import glob
import json
import statistics
import time
from pathlib import Path

import torch

import speed as S

WORK = S.WORK
DARWIN = S.ROWS["DarwinLM"][1]


def manifest_from(path):
    return json.loads((Path(path) / "compact_manifest.json").read_text())


def darwinlm_manifest():
    from safetensors import safe_open
    config = json.loads((DARWIN / "config.json").read_text())
    widths = {}
    for f in sorted(glob.glob(str(DARWIN / "*.safetensors"))):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                if k.endswith("mlp.gate_proj.weight"):
                    widths[int(k.split(".")[2])] = h.get_slice(k).get_shape()[0]
    layers = []
    for i in range(32):
        pruned = set(config["heads_each_attn"][f"{i}.self_attn.o_proj"])
        layers.append({"query_heads": [h for h in range(32) if h not in pruned],
                       "mlp_dimensions": list(range(widths.get(i, 0)))})
    return {"format_version": 1, "layers": layers}


ROWS = ("Dense", "Ours", "Uniform", "ZipLM", "LLM-Pruner", "DarwinLM", "LLM-Pruner (official)")


def manifests(names):
    """Manifests of the rows in names (in the order of ROWS); only the selected baselines are read."""
    out = {}
    for name in (n for n in ROWS if n in names):
        if name == "Dense":
            out[name] = {"format_version": 1,
                         "layers": [{"query_heads": list(range(32)), "mlp_dimensions": list(range(14336))}] * 32}
        elif name == "DarwinLM":
            out[name] = darwinlm_manifest()
        elif name == "LLM-Pruner (official)":
            out[name] = manifest_from(WORK / "baselines/llmpruner_official/compact_model")
        else:
            out[name] = manifest_from(S.ROWS[name][1])
    return out


@torch.inference_mode()
def cache_check(model):
    g = torch.Generator().manual_seed(1)
    prompt = torch.randint(1000, 120000, (2, 64), generator=g).cuda()
    out = model(prompt, use_cache=True)
    past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    seq = torch.cat([prompt, nxt], 1)
    for _ in range(4):
        o = model(nxt, past_key_values=past, use_cache=True)
        past, step, nxt = o.past_key_values, o.logits[:, -1].float(), o.logits[:, -1:].argmax(-1)
        seq = torch.cat([seq, nxt], 1)
    full = model(seq[:, :-1], use_cache=False).logits[:, -1].float()
    return float((step - full).norm() / full.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--gqa", action="store_true", help="GQA-aware attention for every compact row")
    ap.add_argument("--rows", help="comma-separated rows to time (default: all but the official LLM-Pruner)")
    ap.add_argument("--llmpruner-official", action="store_true", help="also time the official LLM-Pruner model")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    from zipprune.ZipLM.compact import compact_model
    base = AutoModelForCausalLM.from_pretrained(S.DENSE, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                low_cpu_mem_usage=True, local_files_only=True).eval()
    keep = [r.strip() for r in a.rows.split(",")] if a.rows else list(ROWS)
    mans = manifests([n for n in keep if n != "LLM-Pruner (official)" or a.llmpruner_official])
    g = torch.Generator().manual_seed(0)
    long_tokens = torch.randint(1000, 120000, (1, 2048), generator=g).cuda()
    prompts = {b: torch.randint(1000, 120000, (b, 512), generator=g).cuda() for b in (1, 16)}
    names = (["Dense (HF)"] if not a.rows or "Dense (HF)" in a.rows.split(",") else []) + list(mans)
    record = {n: {"prefill_ms": [], "decode_b1_ms": [], "decode_b16_ms": []} for n in names}
    meta = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "rounds": a.rounds,
            "implementation": ("compact_gqa (no key/value copy)" if a.gqa else "zipprune.ZipLM.compact (torch SDPA, explicit GQA gather)")
                              + " for every row except Dense (HF)"}
    for rnd in range(a.rounds):
        for name in names:
            t0 = time.time()
            model = copy.deepcopy(base)
            if name != "Dense (HF)":
                compact_model(model, mans[name])
            model = model.to("cuda:0").eval()
            if a.gqa and name != "Dense (HF)":
                from compact_gqa import to_gqa
                if rnd == 0:
                    probe = torch.randint(1000, 120000, (2, 96), generator=torch.Generator().manual_seed(2)).cuda()
                    with torch.inference_mode():
                        before = model(probe, use_cache=False).logits.float()
                to_gqa(model)
                if rnd == 0:
                    with torch.inference_mode():
                        after = model(probe, use_cache=False).logits.float()
                    record[name]["gqa_vs_compact_relative_diff"] = float((after - before).norm() / before.norm())
                    del before, after
            torch.cuda.reset_peak_memory_stats()
            r = record[name]
            r["params"] = sum(p.numel() for p in model.parameters())
            if name != "Dense (HF)":
                layers = mans[name]["layers"]
                r["query_heads"] = sum(len(l["query_heads"]) for l in layers)
                r["kv_heads"] = sum(len({h // 4 for h in l["query_heads"]}) for l in layers)
                r["mlp_fraction"] = sum(len(l["mlp_dimensions"]) for l in layers) / (32 * 14336)
            if rnd == 0:
                r["cache_check_relative_diff"] = cache_check(model)
            for _ in range(2):
                S.prefill(model, long_tokens[:, :256])
                S.decode(model, prompts[1][:, :64], 4)
            for _ in range(5):
                r["prefill_ms"].append(S.prefill(model, long_tokens))
            for _ in range(3):
                r["decode_b1_ms"].append(S.decode(model, prompts[1], 256))
                r["decode_b16_ms"].append(S.decode(model, prompts[16], 256))
            r["peak_mib"] = max(r.get("peak_mib", 0), torch.cuda.max_memory_allocated() / 2 ** 20)
            print(f"round {rnd} {name}: prefill {statistics.median(r['prefill_ms']):.1f} ms, decode b1 "
                  f"{statistics.median(r['decode_b1_ms']):.2f}, b16 {statistics.median(r['decode_b16_ms']):.2f} ms/step"
                  f"{'' if rnd else ', cache check %.4f' % r['cache_check_relative_diff']} ({time.time() - t0:.0f} s)", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            a.out.write_text(json.dumps({"meta": meta, "rows": record}, indent=1))
    ref = record["Dense"]
    summary = {}
    for name, r in record.items():
        s = {k: r[k] for k in ("params", "query_heads", "kv_heads", "mlp_fraction", "cache_check_relative_diff",
                               "gqa_vs_compact_relative_diff") if k in r}
        s["peak_GiB"] = r["peak_mib"] / 1024
        for k in ("prefill_ms", "decode_b1_ms", "decode_b16_ms"):
            s[k] = statistics.median(r[k])
            s["speedup_" + k[:-3]] = statistics.median(ref[k]) / s[k]
        summary[name] = s
    a.out.write_text(json.dumps({"meta": meta, "rows": record, "summary": summary}, indent=1))
    for name, s in summary.items():
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()})


if __name__ == "__main__":
    main()
