"""Offline evaluation of the difficulty classifier + tier policy.

Scores a labeled prompt set, reports the tier distribution, and (since the set
carries easy/hard gold labels) the separation quality: ROC-AUC plus accuracy at
the local/cheap boundary. No network or provider calls.

    python eval/run_eval.py                       # bundled classifier, default bands
    python eval/run_eval.py --config router.yaml  # use a real config's classifier
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

# macOS Accelerate emits a spurious "divide by zero ... matmul" warning from
# BLAS; the computation is correct. Quiet it for clean CLI output.
warnings.filterwarnings("ignore", message=".*matmul.*")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smartrouter.features import extract  # noqa: E402

DEFAULT_SET = os.path.join(os.path.dirname(__file__), "labeled_prompts.jsonl")
DEFAULT_BANDS = [("local", 0.0), ("cheap", 0.40), ("frontier", 0.75)]


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tier_for(score, bands):
    name = bands[0][0]
    for n, lo in bands:
        if score >= lo:
            name = n
    return name


def build_classifier(config_path):
    if config_path:
        from smartrouter.classifiers import build_classifier as bc
        from smartrouter.config import RouterConfig

        cfg = RouterConfig.from_yaml(config_path)
        bands = [(t.name, t.min_score) for t in cfg.tiers]
        return bc(cfg.classifier), bands
    from smartrouter.classifiers.embedding import EmbeddingClassifier

    return EmbeddingClassifier.bundled(), DEFAULT_BANDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=DEFAULT_SET)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    clf, bands = build_classifier(args.config)
    rows = load(args.set)

    scores, labels, tiers = [], [], {}
    for r in rows:
        s = clf.score(extract([{"role": "user", "content": r["text"]}]))
        scores.append(s)
        labels.append(int(r.get("label", -1)))
        tiers[tier_for(s, bands)] = tiers.get(tier_for(s, bands), 0) + 1

    print(f"prompts: {len(rows)}")
    print("tier distribution:")
    for name, _ in bands:
        print(f"  {name:10} {tiers.get(name, 0)}")

    if all(l in (0, 1) for l in labels) and len(set(labels)) == 2:
        import numpy as np
        from sklearn.metrics import accuracy_score, roc_auc_score

        y = np.asarray(labels)
        s = np.asarray(scores)
        auc = roc_auc_score(y, s)
        boundary = bands[1][1] if len(bands) > 1 else 0.5
        acc = accuracy_score(y, (s >= boundary).astype(int))
        print(f"\nseparation: ROC-AUC={auc:.3f} | acc@{boundary:.2f}={acc:.3f}")
        easy = s[y == 0]
        hard = s[y == 1]
        print(f"  easy score  mean={easy.mean():.3f} max={easy.max():.3f}")
        print(f"  hard score  mean={hard.mean():.3f} min={hard.min():.3f}")


if __name__ == "__main__":
    main()
