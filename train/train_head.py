"""Train (or retrain) the difficulty-classifier head.

Two sources:
  * --exemplars PATH   a JSONL of {"text", "label"} (the bundled seed set)
  * --from-store PATH   a smartrouter SQLite store, using logged embeddings + labels

Writes a joblib head loadable by EmbeddingClassifier.load(). With no args it
rebuilds the bundled default from the packaged exemplars.

    python train/train_head.py                       # rebuild bundled default
    python train/train_head.py --from-store smartrouter.db -o my_head.joblib
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

# macOS Accelerate emits a spurious "divide by zero ... matmul" BLAS warning
# during fit; the result is correct. Quiet it for clean CLI output.
warnings.filterwarnings("ignore", message=".*matmul.*")

# Make src/ importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smartrouter.classifiers.embedding import EmbeddingClassifier  # noqa: E402

DEFAULT_EXEMPLARS = os.path.join(
    os.path.dirname(__file__), "..", "src", "smartrouter", "data",
    "difficulty_exemplars.jsonl",
)
DEFAULT_OUT = os.path.join(
    os.path.dirname(__file__), "..", "src", "smartrouter", "data",
    "default_classifier.joblib",
)


def load_exemplars(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exemplars", default=None)
    ap.add_argument("--from-store", default=None)
    ap.add_argument("--backend", default="hashing",
                    choices=["hashing", "sentence-transformer"])
    ap.add_argument("--embedding-model",
                    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    clf = EmbeddingClassifier(backend=args.backend,
                              embedding_model=args.embedding_model)

    if args.from_store:
        import numpy as np
        from smartrouter.logging_ import TrainingStore

        store = TrainingStore(args.from_store)
        vectors, labels = store.training_matrix(clf.embedding_model_id)
        if len(labels) == 0:
            raise SystemExit(
                "no labeled rows with matching embedding_model_id in the store"
            )
        clf.fit_vectors(np.asarray(vectors), labels)
        print(f"trained head on {len(labels)} labeled rows from {args.from_store}")
    else:
        path = args.exemplars or DEFAULT_EXEMPLARS
        texts, labels = load_exemplars(path)
        clf.fit(texts, labels)
        print(f"trained head on {len(labels)} exemplars from {path}")

    clf.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
