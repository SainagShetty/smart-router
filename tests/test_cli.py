"""CLI: ask / route / stats / init and config resolution."""
import os

import pytest
import yaml

from smartrouter import cli

from conftest import make_config


def _write_config(tmp_path, db_path=None):
    cfg = make_config(db_path=db_path)
    p = tmp_path / "router.yaml"
    p.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return str(p)


def _stub_cli_providers(monkeypatch, content="cli answer"):
    # patch RouterCore so every provider returns a canned completion, no network
    from smartrouter.core import RouterCore
    orig_init = RouterCore.__init__

    def patched(self, config):
        orig_init(self, config)
        for name, prov in self.providers.items():
            prov.complete = (lambda m, msgs, **p: {
                "model": m,
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"total_tokens": 10}})
    monkeypatch.setattr(RouterCore, "__init__", patched)


def test_ask_prints_answer_and_receipt(tmp_path, capsys, monkeypatch):
    _stub_cli_providers(monkeypatch, content="forty two")
    cfg = _write_config(tmp_path)
    cli.main(["ask", "--config", cfg, "what", "is", "6x7?"])
    out = capsys.readouterr()
    assert "forty two" in out.out
    assert "tier=" in out.err  # receipt on stderr


def test_route_is_dry_run(tmp_path, capsys, monkeypatch):
    # route must NOT call a provider; use a config and assert a decision prints
    cfg = _write_config(tmp_path)
    cli.main(["route", "--config", cfg, "what is 2+2?"])
    out = capsys.readouterr()
    assert "tier=local" in out.out


def test_route_sensitive_flag(tmp_path, capsys, monkeypatch):
    cfg = _write_config(tmp_path)
    cli.main(["route", "--config", cfg, "--sensitive", "a private note"])
    out = capsys.readouterr()
    assert "model=llama3.1:8b" in out.out


def test_stats_reads_log(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "sr.db")
    _stub_cli_providers(monkeypatch)
    cfg = _write_config(tmp_path, db_path=db)
    cli.main(["ask", "--config", cfg, "hello"])
    cli.main(["stats", "--config", cfg])
    out = capsys.readouterr()
    assert "served on-device" in out.out
    assert "requests:" in out.out


def test_init_writes_valid_config(tmp_path, capsys, monkeypatch):
    # no ollama binary -> template still written AND must be a valid loadable config
    from smartrouter import RouterConfig
    out_path = str(tmp_path / "router.yaml")
    cli.main(["init", "--output", out_path])
    assert os.path.exists(out_path)
    body = open(out_path).read()
    assert "default_tier: local" in body
    cfg = RouterConfig.from_yaml(out_path)  # must not raise
    assert cfg.models  # never empty, even with no local models detected
    captured = capsys.readouterr()
    assert "wrote" in captured.out


def test_init_refuses_overwrite(tmp_path):
    out_path = tmp_path / "router.yaml"
    out_path.write_text("existing")
    with pytest.raises(SystemExit):
        cli.main(["init", "--output", str(out_path)])


def test_config_resolution_env(tmp_path, capsys, monkeypatch):
    cfg = _write_config(tmp_path)
    monkeypatch.setenv("SMARTROUTER_CONFIG", cfg)
    cli.main(["route", "what is 2+2?"])
    out = capsys.readouterr()
    assert "tier=local" in out.out


def test_missing_config_exits(tmp_path, monkeypatch):
    monkeypatch.delenv("SMARTROUTER_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["route", "hi"])
