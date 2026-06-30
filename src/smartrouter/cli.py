"""smartrouter command-line entry point.

    smartrouter serve --config router.yaml --host 127.0.0.1 --port 4000

The bearer token (optional) is read from the SMARTROUTER_API_KEY env var so it
never lands in a process list or shell history.
"""
from __future__ import annotations

import argparse
import os
import sys

from .config import RouterConfig


def _serve(args):
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - optional extra
        sys.exit("the server requires `pip install smartrouter[server]`")

    from .server import create_app

    config = RouterConfig.from_yaml(args.config)
    app = create_app(config, api_key=os.environ.get("SMARTROUTER_API_KEY"))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="smartrouter")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the shared router HTTP server")
    serve.add_argument("--config", default="router.yaml")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=4000)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
