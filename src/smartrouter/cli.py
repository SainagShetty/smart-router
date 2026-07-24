"""smartrouter command-line entry point.

    smartrouter ask "what is 2+2?"           # route + answer + one-line receipt
    smartrouter route "prove the theorem"    # dry-run: show the decision, no call
    smartrouter stats                        # decision-log totals + free-local %
    smartrouter init                         # autodetect Ollama -> router.yaml
    smartrouter serve --config router.yaml   # shared HTTP server

Config is resolved from --config, then $SMARTROUTER_CONFIG, then
~/.config/smartrouter/router.yaml, then ./router.yaml. The bearer token
(optional) is read from SMARTROUTER_API_KEY so it never lands in shell history.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .config import RouterConfig

def _resolve_config_path(explicit=None):
    # Resolved at call time so SMARTROUTER_CONFIG set after import is honored.
    locations = [
        explicit,
        os.environ.get("SMARTROUTER_CONFIG"),
        os.path.expanduser("~/.config/smartrouter/router.yaml"),
        "router.yaml",
    ]
    for path in locations:
        if path and os.path.exists(path):
            return path
    return None


def _resolve_config(explicit=None):
    path = _resolve_config_path(explicit)
    if not path:
        sys.exit(
            "no config found. Pass --config, set SMARTROUTER_CONFIG, or run "
            "`smartrouter init` to create one."
        )
    return RouterConfig.from_yaml(path)


def _read_prompt(args):
    text = " ".join(args.prompt).strip()
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if not text:
        sys.exit("no prompt given (pass it as an argument or pipe via stdin)")
    return [{"role": "user", "content": text}]


def _receipt(d):
    bits = [f"tier={d.tier}", f"model={d.model}", f"score={d.score}"]
    if getattr(d, "cascade", False):
        bits.append("cascade")
    return "[" + "  ".join(bits) + "]"


def _ask(args):
    from .core import RouterCore

    core = RouterCore(_resolve_config(args.config))
    messages = _read_prompt(args)
    resp, d = core.complete(messages, cascade=args.cascade or None,
                            local_only=args.local_only, sensitive=args.sensitive)
    content = resp["choices"][0]["message"].get("content") or ""
    print(content)
    print(_receipt(d), file=sys.stderr)
    core.close()


def _route(args):
    from .core import RouterCore

    core = RouterCore(_resolve_config(args.config))
    d = core.route(_read_prompt(args), cascade=args.cascade or None,
                   local_only=args.local_only, sensitive=args.sensitive)
    print(_receipt(d))
    if d.rejected:
        print("rejected:", file=sys.stderr)
        for r in d.rejected:
            print(f"  - {r}", file=sys.stderr)
    core.close()


def _stats(args):
    from .core import RouterCore

    config = _resolve_config(args.config)
    core = RouterCore(config)
    if not core.store:
        sys.exit("logging is disabled in this config; no stats to show.")
    local_providers = {n for n, p in config.providers.items() if p.is_local()}
    local_models = {m.id for m in config.models if m.provider in local_providers}
    s = core.store.savings_summary(local_models=local_models)
    print(f"requests:        {s['total']}")
    print(f"served on-device: {s['on_device']} ({s['on_device_pct']}%)")
    print(f"free (no cost):   {s['free']}")
    print(f"observed spend:   ${s['total_cost']}")
    print(f"by tier:          {s['by_tier']}")
    core.close()


def _init(args):
    import subprocess

    if os.path.exists(args.output) and not args.force:
        sys.exit(f"{args.output} already exists (use --force to overwrite)")

    models = []
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True,
                             timeout=5)
        for line in out.stdout.splitlines()[1:]:
            name = line.split()[0] if line.split() else None
            if name:
                models.append(name)
    except Exception:
        pass

    local_models = "\n".join(
        f'  - {{ id: "{m}", provider: ollama, tier: local, '
        f'context_window: 8192, capabilities: {{ tools: true, json: true }} }}'
        for m in models
    )
    if local_models:
        local_models += "\n"

    config = _INIT_TEMPLATE.format(local_models=local_models)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(config)
    found = f"detected {len(models)} Ollama model(s)" if models else \
        "no Ollama models detected — add local models under `models:` to use " \
        "the free on-device floor"
    print(f"wrote {args.output} ({found}).")


_INIT_TEMPLATE = """# smartrouter config. Local-first: the free on-device floor is preferred and
# requests escalate to cloud tiers only when needed. Edit ids/costs to taste.
providers:
  ollama: {{ type: ollama, base_url: http://localhost:11434/v1, local: true }}
  openrouter: {{ type: openrouter, base_url: https://openrouter.ai/api/v1, api_key_env: OPENROUTER_API_KEY }}

tiers:
  - {{ name: local,    min_score: 0.0 }}
  - {{ name: cheap,    min_score: 0.40 }}
  - {{ name: frontier, min_score: 0.75 }}

models:
{local_models}  # cloud tiers — edit ids/costs for your OpenRouter account:
  - {{ id: "openai/gpt-4o-mini", provider: openrouter, tier: cheap, cost_per_1k: 0.0006, context_window: 128000, capabilities: {{ vision: true, tools: true, json: true }} }}
  - {{ id: "anthropic/claude-3.5-sonnet", provider: openrouter, tier: frontier, cost_per_1k: 0.009, context_window: 200000, capabilities: {{ vision: true, tools: true, json: true }} }}

policy:
  default_tier: local
  fallback: down
  # cascade: true   # run cheapest-first and escalate on a bad answer

logging:
  enabled: true
  db_path: smartrouter.db
"""


def _serve(args):
    try:
        import uvicorn
    except Exception:  # pragma: no cover - optional extra
        sys.exit("the server requires `pip install smartrouter[server]`")

    from .server import create_app

    config = _resolve_config(args.config)
    app = create_app(config, api_key=os.environ.get("SMARTROUTER_API_KEY"))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


def _ui(args):
    try:
        import uvicorn
    except Exception:  # pragma: no cover - optional extra
        sys.exit("the UI requires `pip install smartrouter[server]`")

    config_path = _resolve_config_path(args.config)
    if not config_path:
        sys.exit(
            "no config found. Pass --config, set SMARTROUTER_CONFIG, or run "
            "`smartrouter init` to create one."
        )

    from .ui_server import create_app

    app = create_app(config_path)
    uvicorn.run(app, host=args.host, port=args.port)


def _add_route_flags(p):
    p.add_argument("prompt", nargs="*", help="the prompt (or pipe via stdin)")
    p.add_argument("--config")
    p.add_argument("--cascade", action="store_true",
                   help="run cheapest-first and escalate on a bad answer")
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--sensitive", action="store_true",
                   help="pin to on-device models; never egress")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="smartrouter")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="route a prompt, call the model, print the answer")
    _add_route_flags(ask)
    ask.set_defaults(func=_ask)

    route = sub.add_parser("route", help="show the routing decision without calling")
    _add_route_flags(route)
    route.set_defaults(func=_route)

    stats = sub.add_parser("stats", help="decision-log totals + free-local savings")
    stats.add_argument("--config")
    stats.set_defaults(func=_stats)

    init = sub.add_parser("init", help="autodetect Ollama and write a router.yaml")
    init.add_argument("--output", default="router.yaml")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=_init)

    serve = sub.add_parser("serve", help="run the shared router HTTP server")
    serve.add_argument("--config")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=4000)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=_serve)

    ui = sub.add_parser("ui", help="run the router configuration UI")
    ui.add_argument("--config")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=4001)
    ui.set_defaults(func=_ui)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
