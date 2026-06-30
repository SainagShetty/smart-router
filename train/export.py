"""Export labeled rows from a smartrouter store into a versioned dataset.

Emits JSONL of {"text"?, "label"} (text only when raw logging was on) plus a
companion .npz of stored embeddings + labels for re-fitting the head without
re-embedding.

    python train/export.py --store smartrouter.db --out datasets/v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from smartrouter.logging_ import TrainingStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="smartrouter.db")
    ap.add_argument("--embedding-model-id", default=None,
                    help="restrict the .npz export to one embedding space")
    ap.add_argument("--out", default="dataset")
    args = ap.parse_args()

    store = TrainingStore(args.store)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    pairs = store.labeled_prompts()
    with open(args.out + ".jsonl", "w", encoding="utf-8") as fh:
        for text, label in pairs:
            fh.write(json.dumps({"text": text, "label": label}) + "\n")
    print(f"wrote {args.out}.jsonl ({len(pairs)} rows with raw text)")

    if args.embedding_model_id:
        X, y = store.training_matrix(args.embedding_model_id)
        np.savez(args.out + ".npz", X=X, y=np.asarray(y))
        print(f"wrote {args.out}.npz ({len(y)} embedded rows)")

    print(f"store totals: {store.count()} rows, {store.count(labeled_only=True)} labeled")
    store.close()


if __name__ == "__main__":
    main()
