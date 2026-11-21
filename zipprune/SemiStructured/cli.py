#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from .prune_2to4 import (
    DEFAULT_TARGET_SUFFIXES,
    apply_2to4_pruning_,
    load_model_and_tokenizer,
    save_model_and_tokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Apply magnitude 2:4 semi-structured pruning and save a checkpoint.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--keep_pattern_n", type=int, default=2)
    parser.add_argument("--keep_pattern_m", type=int, default=4)
    parser.add_argument(
        "--target_suffixes",
        type=str,
        default=",".join(DEFAULT_TARGET_SUFFIXES),
        help="Comma-separated linear suffixes to prune, e.g. q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    suffixes = tuple(item.strip() for item in args.target_suffixes.split(",") if item.strip())
    print("[2to4][cli] loading model")
    print(f"[2to4][cli] model={args.model_name_or_path}")
    print(f"[2to4][cli] output_dir={args.output_dir}")
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )
    try:
        summary = apply_2to4_pruning_(
            model=model,
            target_suffixes=suffixes,
            n=args.keep_pattern_n,
            m=args.keep_pattern_m,
        )
        print("[2to4][cli] saving checkpoint")
        save_model_and_tokenizer(
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            summary=summary,
        )
        print("[2to4][cli] done")
    finally:
        del model


if __name__ == "__main__":
    main()
