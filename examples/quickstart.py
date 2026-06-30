"""Quickstart: route a few prompts and see which tier each lands in.

This only makes routing *decisions* (no provider calls), so it runs without any
API keys or a local model. Swap route() for the drop-in client to actually call
the chosen model.

    python examples/quickstart.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*matmul.*")  # macOS Accelerate noise

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smartrouter import RouterConfig, RouterCore  # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), "router.yaml")

PROMPTS = [
    "What's the capital of France?",
    "Convert 10 miles to kilometers.",
    "Explain why my Kubernetes pod keeps getting OOMKilled and how to debug it.",
    "Prove the halting problem is undecidable via diagonalization.",
]


def main():
    core = RouterCore(RouterConfig.from_yaml(CONFIG))
    for p in PROMPTS:
        d = core.route([{"role": "user", "content": p}])
        print(f"[{d.tier:8}] score={d.score:.2f}  {d.model:30}  <- {p[:45]}")

    # Force a single call to stay on-device, even though it's a hard query.
    d = core.route(
        [{"role": "user", "content": "design a distributed rate limiter, prove it"}],
        local_only=True,
    )
    print(f"\nlocal_only -> [{d.tier}] {d.model}  ({d.reason})")
    core.close()


if __name__ == "__main__":
    main()
