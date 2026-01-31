#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True, help="generation JSONL for ONE model (policy OR base)")
    ap.add_argument("--out_file", required=True, help="pairwise JSONL output (with optional shuffle)")
    ap.add_argument("--seed", type=int, default=42, help="seed for shuffling A/B")
    ap.add_argument("--shuffle_ab", action="store_true", help="randomly swap A/B with p=0.5 per example")
    ap.add_argument("--b_id", default="hh_chosen", help="label for chosen side")

    ap.add_argument("--prompt_key", default="prompt")
    ap.add_argument("--response_key", default="response")
    ap.add_argument("--chosen_key", default="HH_response")
    ap.add_argument("--model_id_key", default="model_id")
    ap.add_argument("--idx_key", default="idx")

    ap.add_argument("--keep_meta", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)

    in_path = Path(args.in_file)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    n_swapped = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for line_no, ex in iter_jsonl(in_path):
            for k in (args.prompt_key, args.response_key, args.chosen_key):
                if k not in ex:
                    raise KeyError(f"Missing key '{k}' at {in_path}:{line_no}. keys={list(ex.keys())}")

            idx = ex.get(args.idx_key, n)
            prompt = ex[args.prompt_key]

            model_resp = ex[args.response_key]
            chosen_resp = ex[args.chosen_key]

            model_id = ex.get(args.model_id_key, "model")
            chosen_id = args.b_id

            # default: A=model, B=chosen
            a, b = model_resp, chosen_resp
            a_id, b_id = model_id, chosen_id
            is_swapped = False

            # optional shuffle: swap A/B with 50% chance
            if args.shuffle_ab and random.random() < 0.5:
                a, b = b, a
                a_id, b_id = b_id, a_id
                is_swapped = True
                n_swapped += 1

            rec = {
                "idx": idx,
                "prompt": prompt,
                "a": a,
                "b": b,
                "a_id": a_id,
                "b_id": b_id,
                "is_swapped": is_swapped,   # IMPORTANT for mapping judge winner back
                "shuffle_seed": args.seed,
            }

            if args.keep_meta:
                for k, v in ex.items():
                    if k not in rec:
                        rec[k] = v

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] wrote {n} lines -> {out_path} | swapped={n_swapped} ({n_swapped/max(1,n):.1%})")


if __name__ == "__main__":
    main()
