"""Export (prompt, response) pairs from a smartrouter store as an SFT dataset.

Rows qualify when both raw texts were retained (log_raw on for that request).
This is the corpus for fine-tuning/distilling your own LLM: the routed model's
answer is the training target, and labels act as quality filters (label=1 rows
are known-bad answers and are dropped unless --keep-bad).

Monitor and load-test traffic is dropped too (--keep-synthetic to retain it).
Those sources repeat one prompt on a timer, so they carry no signal and, left
in, they outnumber the real corpus.

    python train/export_sft.py --store smartrouter.db --out datasets/sft_v1
    python train/export_sft.py --exclude-sensitive          # drop tagged rows
    python train/export_sft.py --tiers cheap frontier       # distill from cloud only

Output JSONL, one object per row (OpenAI messages shape, ready for most
fine-tuning stacks):
    {"messages": [{"role": "user", "content": ...},
                  {"role": "assistant", "content": ...}],
     "meta": {"tier", "model", "source", "sensitive", "label"}}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smartrouter.logging_ import TrainingStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="smartrouter.db")
    ap.add_argument("--out", default="dataset_sft")
    ap.add_argument("--exclude-sensitive", action="store_true",
                    help="drop rows tagged sensitive (e.g. finance traffic)")
    ap.add_argument("--tiers", nargs="*", default=None,
                    help="only these tiers (e.g. `cheap frontier` to distill "
                         "from cloud-model answers)")
    ap.add_argument("--keep-bad", action="store_true",
                    help="keep label=1 rows (known under-served answers)")
    ap.add_argument("--keep-synthetic", action="store_true",
                    help="keep monitor/load-test rows (see _SYNTHETIC_SOURCES)")
    args = ap.parse_args()

    store = TrainingStore(args.store)
    # Ask for the synthetic rows even when dropping them, so the count below is
    # exact. A corpus that silently shrinks by a third is the failure here.
    rows = store.sft_rows(include_sensitive=not args.exclude_sensitive,
                          include_synthetic=True)
    synthetic = set(TrainingStore._SYNTHETIC_SOURCES)
    store.close()

    kept, dropped_bad, dropped_tier, dropped_synth = [], 0, 0, 0
    for r in rows:
        if not args.keep_synthetic and r["source"] in synthetic:
            dropped_synth += 1
            continue
        if not args.keep_bad and r["label"] == 1:
            dropped_bad += 1
            continue
        if args.tiers and r["tier"] not in args.tiers:
            dropped_tier += 1
            continue
        kept.append(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".jsonl", "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps({
                "messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["response"]},
                ],
                "meta": {"tier": r["tier"], "model": r["model"],
                         "source": r["source"], "sensitive": bool(r["sensitive"]),
                         "label": r["label"]},
            }, ensure_ascii=False) + "\n")

    print(f"wrote {args.out}.jsonl ({len(kept)} pairs; "
          f"dropped {dropped_synth} synthetic, {dropped_bad} bad-label, "
          f"{dropped_tier} off-tier)")


if __name__ == "__main__":
    main()
