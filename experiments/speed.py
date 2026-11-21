#!/usr/bin/env python3
"""Inference speed of the Llama-3.1-8B rows of the compression table.

Rows: Dense, Ours (llama31_llm_nogp_s7), Uniform (level 5), ZipLM (existing database, speed-up 1.6), LLM-Pruner (GQA
corrected), DarwinLM (released 4.6B checkpoint, its own model code). Every compressed row is physically compact, loaded
exactly as for the accuracy benchmark (zipprune.ZipLM.compact.load_compact_model; DarwinLM with trust_remote_code).
Settings: HF transformers, bf16, SDPA where the model code supports it, one GPU, random token ids (speed does not depend
on content). Measured with CUDA events:
  prefill   one forward pass over 2048 tokens at batch 1 (time to first token, cache built)
  decode    256 greedy steps after a 512-token prompt, at batch 1 and at batch 16 (prompt time excluded)
Order: rounds x rows, each row loaded, measured and freed within a round, so drift affects all rows alike; reported
values are medians over all repetitions of all rounds. Peak memory per row is recorded.
Row locations (compact checkpoints and model directories) are under the work directory, see ROWS below.
Usage: CUDA_VISIBLE_DEVICES=<gpu> python speed.py --out <json> [--rounds 3] [--quick]
"""
import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import torch                                         # noqa: E402
from common import WORK                              # noqa: E402

DENSE = WORK / "hf_models" / "Llama-3.1-8B"
ROWS = {
    "Dense": ("hf", DENSE),
    "Ours": ("compact", WORK / "benchmark/llama31/compression_llama31_llm_nogp_s7/compact_model"),
    "Uniform": ("compact", WORK / "benchmark/llama31/uniform_l5/compact_model"),
    "ZipLM": ("compact", WORK / "baselines/ziplm_speedup1.6/compact_model"),
    "LLM-Pruner": ("compact", WORK / "baselines/llmpruner_gqa_corrected/compact_model"),
    "DarwinLM": ("remote", WORK / "hf_models" / "DarwinLM-4.6B-Pruned"),   # Shengkun/DarwinLM-4.6B-Pruned
}


def load(kind, path):
    from transformers import AutoModelForCausalLM
    if kind == "compact":
        from zipprune.ZipLM.compact import load_compact_model
        return load_compact_model(str(path), device="cuda:0", dtype=torch.bfloat16), "sdpa"
    kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)
    if kind == "remote":
        kwargs["trust_remote_code"] = True
    try:
        model = AutoModelForCausalLM.from_pretrained(str(path), attn_implementation="sdpa", **kwargs)
        attn = "sdpa"
    except (ValueError, TypeError):                  # model code without SDPA support: record what it uses
        model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
        attn = getattr(model.config, "_attn_implementation", "default")
    return model.to("cuda:0").eval(), attn


def timed(fn):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)                   # milliseconds


@torch.inference_mode()
def prefill(model, tokens):
    return timed(lambda: model(tokens, use_cache=True))


@torch.inference_mode()
def decode(model, prompt, steps):
    out = model(prompt, use_cache=True)
    past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    state = {"past": past, "nxt": nxt}

    def loop():
        for _ in range(steps):
            o = model(state["nxt"], past_key_values=state["past"], use_cache=True)
            state["past"], state["nxt"] = o.past_key_values, o.logits[:, -1:].argmax(-1)
    return timed(loop) / steps                       # milliseconds per step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="smoke test: Ours only, one short repetition")
    a = ap.parse_args()
    g = torch.Generator().manual_seed(0)
    long_tokens = torch.randint(1000, 120000, (1, 2048), generator=g).cuda()
    prompts = {b: torch.randint(1000, 120000, (b, 512), generator=g).cuda() for b in (1, 16)}
    rows = {"Ours": ROWS["Ours"]} if a.quick else ROWS
    reps = {"prefill": 1 if a.quick else 5, "decode": 1 if a.quick else 3, "steps": 16 if a.quick else 256}
    record = {name: {"prefill_ms": [], "decode_b1_ms": [], "decode_b16_ms": []} for name in rows}
    meta = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "rounds": 1 if a.quick else a.rounds, **reps}
    for rnd in range(meta["rounds"]):
        for name, (kind, path) in rows.items():
            t0 = time.time()
            model, attn = load(kind, path)
            torch.cuda.reset_peak_memory_stats()
            r = record[name]
            r.update(params=sum(p.numel() for p in model.parameters()), attn=attn)
            for _ in range(2):                       # warm-up (kernels, allocator)
                prefill(model, long_tokens[:, :256])
                decode(model, prompts[1][:, :64], 4)
            for _ in range(reps["prefill"]):
                r["prefill_ms"].append(prefill(model, long_tokens))
            for _ in range(reps["decode"]):
                r["decode_b1_ms"].append(decode(model, prompts[1], reps["steps"]))
                r["decode_b16_ms"].append(decode(model, prompts[16], reps["steps"]))
            r["peak_mib"] = max(r.get("peak_mib", 0), torch.cuda.max_memory_allocated() / 2 ** 20)
            print(f"round {rnd} {name}: prefill {statistics.median(r['prefill_ms']):.1f} ms, decode b1 "
                  f"{statistics.median(r['decode_b1_ms']):.2f} ms/step, b16 {statistics.median(r['decode_b16_ms']):.2f} ms/step "
                  f"({time.time() - t0:.0f} s incl. load)", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            a.out.write_text(json.dumps({"meta": meta, "rows": record}, indent=1))
    dense = record.get("Dense")
    summary = {}
    for name, r in record.items():
        s = {"params_B": r["params"] / 1e9, "attn": r["attn"], "peak_GiB": r["peak_mib"] / 1024,
             "prefill_ms": statistics.median(r["prefill_ms"]), "decode_b1_ms": statistics.median(r["decode_b1_ms"]),
             "decode_b16_ms": statistics.median(r["decode_b16_ms"])}
        if dense:
            for k in ("prefill_ms", "decode_b1_ms", "decode_b16_ms"):
                s["speedup_" + k.replace("_ms", "")] = statistics.median(dense[k]) / s[k]
        summary[name] = s
    a.out.write_text(json.dumps({"meta": meta, "rows": record, "summary": summary}, indent=1))
    for name, s in summary.items():
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()})


if __name__ == "__main__":
    main()
