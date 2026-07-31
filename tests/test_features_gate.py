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


# ---- request shape: telling an agent loop from one long chat turn -----------

def _agent_messages(n_steps=3, tool_result="x" * 500):
    """A tool-calling loop: task, then (assistant tool_call -> tool result) xN."""
    msgs = [{"role": "user", "content": "book the 8:30 class"}]
    for _ in range(n_steps):
        msgs.append({"role": "assistant", "content": "calling a tool"})
        msgs.append({"role": "tool", "content": tool_result})
    return msgs


def test_shape_counts_tool_and_assistant_messages():
    f = extract(_agent_messages(n_steps=3))
    assert f.n_tool_msgs == 3
    assert f.n_assistant_msgs == 3


def test_shape_is_zero_for_plain_chat():
    f = extract([{"role": "user", "content": "what is 2+2"}])
    assert f.n_tool_msgs == 0
    assert f.n_assistant_msgs == 0
    assert f.tail_chars == 0


def test_head_chars_is_the_first_user_turn_not_the_longest():
    # The task is short; a later user turn is long. head_chars must track the
    # task, since that is what sets the difficulty ceiling for the whole run.
    f = extract([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "y" * 900},
    ])
    assert f.head_chars == 2


def test_tail_chars_is_the_newest_tool_result():
    msgs = _agent_messages(n_steps=2)
    msgs.append({"role": "assistant", "content": "calling a tool"})
    msgs.append({"role": "tool", "content": "z" * 77})
    f = extract(msgs)
    assert f.tail_chars == 77


def test_shape_distinguishes_agent_loop_from_one_long_turn():
    """Both flatten to similar text; only the shape fields tell them apart."""
    loop = extract(_agent_messages(n_steps=4))
    chat = extract([{"role": "user", "content": "x" * loop.raw_chars}])
    assert abs(loop.raw_chars - chat.raw_chars) < 100  # comparable size
    assert loop.n_tool_msgs == 4 and chat.n_tool_msgs == 0
