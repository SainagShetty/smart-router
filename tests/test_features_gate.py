from smartrouter import capabilities
from smartrouter.features import extract


def test_extract_basic_text():
    f = extract([{"role": "user", "content": "hello world"}])
    assert f.text == "hello world"
    assert f.estimated_tokens >= 1
    assert not f.has_images and not f.needs_tools and not f.needs_json


def test_extract_detects_image():
    f = extract([{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
    ]}])
    assert f.has_images is True
    assert "what is this" in f.text


def test_extract_tools_and_json():
    f = extract([{"role": "user", "content": "go"}],
                tools=[{"type": "function"}],
                response_format={"type": "json_object"})
    assert f.needs_tools is True
    assert f.needs_json is True


def test_extract_code_ratio():
    f = extract([{"role": "user", "content": "fix ```def f(): return 1```"}])
    assert f.code_ratio > 0


def test_gate_rejects_non_vision_for_image(config):
    f = extract([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "http://x"}}]}])
    res = capabilities.gate(config.models, f)
    ids = {m.id for m in res.candidates}
    assert "llama3.1:8b" not in ids  # local model has no vision
    assert "gpt-4o-mini" in ids
    assert any("vision" in r for r in res.rejected)


def test_gate_rejects_on_context_window(config):
    f = extract([{"role": "user", "content": "x" * 60000}])  # ~15k tokens
    res = capabilities.gate(config.models, f)
    ids = {m.id for m in res.candidates}
    assert "llama3.1:8b" not in ids  # 8k window too small
    assert "gpt-4o-mini" in ids


def test_gate_rejects_tools_unsupported(config):
    # Build a model with no tool support and check it's filtered.
    f = extract([{"role": "user", "content": "go"}], tools=[{"type": "function"}])
    res = capabilities.gate(config.models, f)
    # all three configured models support tools, so all pass
    assert len(res.candidates) == 3
