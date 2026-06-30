"""Distill a smaller/faster difficulty model from logged (prompt -> label) pairs.

This is the "faster model" path: instead of re-fitting only the linear head, it
re-embeds your logged prompts with a *target* backend (e.g. a small
sentence-transformer or the lightweight hashing space) and fits a fresh head, so
you can swap the whole classifier for one that is cheaper to run at inference.

Requires raw-text logging (logging.log_raw=true) so prompts can be re-embedded.

    python train/distill_embedder.py --store smartrouter.db \
        --backend hashing -o data/distilled.joblib
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smartrouter.classifiers.embedding import EmbeddingClassifier  # noqa: E402
from smartrouter.logging_ import TrainingStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="smartrouter.db")
    ap.add_argument("--backend", default="hashing",
                    choices=["hashing", "sentence-transformer"])
    ap.add_argument("--embedding-model",
                    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("-o", "--out", default="distilled.joblib")
    args = ap.parse_args()

    store = TrainingStore(args.store)
    pairs = store.labeled_prompts()
    if not pairs:
        raise SystemExit(
            "no labeled rows with raw prompt text; enable logging.log_raw to distill"
        )
    texts = [p for p, _ in pairs]
    labels = [l for _, l in pairs]

    clf = EmbeddingClassifier(backend=args.backend,
                              embedding_model=args.embedding_model)
    clf.fit(texts, labels)
    clf.save(args.out)
    print(f"distilled {clf.embedding_model_id} head on {len(labels)} prompts -> {args.out}")
    store.close()


if __name__ == "__main__":
    main()
