import pytest

from smartrouter import RouterClient, RouterCore
from smartrouter.errors import ProviderError

from conftest import make_config, stub_providers


def test_route_easy_local(config):
    core = RouterCore(config)
    d = core.route([{"role": "user", "content": "what is 2+2?"}])
    assert d.tier == "local"
    assert d.model == "llama3.1:8b"
    assert d.decision_id


def test_route_hard_escalates(config):
    core = RouterCore(config)
    d = core.route([{"role": "user", "content":
                     "derive backpropagation for a softmax network step by step "
                     "and prove convergence"}])
    assert d.tier in ("cheap", "frontier")


def test_complete_calls_provider_and_returns_response(config):
    core = RouterCore(config)
    stub_providers(core, content="42")
    resp, d = core.complete([{"role": "user", "content": "2+2?"}])
    assert resp["choices"][0]["message"]["content"] == "42"
    assert d.tier == "local"


def test_local_only_in_complete(config):
    core = RouterCore(config)
    stub_providers(core)
    resp, d = core.complete(
        [{"role": "user", "content":
          "design a distributed lock-free rate limiter and prove linearizability"}],
        local_only=True,
    )
    assert d.tier == "local"


def test_fallback_chain_on_provider_error(config):
    core = RouterCore(config)
    # hard query routes frontier (openrouter); make openrouter fail -> falls to ollama
    stub_providers(core, fail_providers=("openrouter",))
    resp, d = core.complete(
        [{"role": "user", "content":
          "prove the theorem, derive the bound, analyze the distributed algorithm"}],
    )
    # frontier+cheap are openrouter (fail); only local (ollama) succeeds
    assert resp["model"] == "llama3.1:8b"


def test_all_providers_fail_raises(config):
    core = RouterCore(config)
    stub_providers(core, fail_providers=("openrouter", "ollama"))
    with pytest.raises(ProviderError):
        core.complete([{"role": "user", "content": "hi"}])


def test_client_dropin_shape(config):
    client = RouterClient(config)
    stub_providers(client._core, content="hello")
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": "2+2?"}])
    assert resp.choices[0].message.content == "hello"
    assert resp.decision_id
    assert resp.routing.tier == "local"


def test_client_passes_overrides(config):
    client = RouterClient(config)
    stub_providers(client._core)
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content":
                   "architect a fault-tolerant distributed system, prove correctness"}],
        local_only=True,
    )
    assert resp.routing.tier == "local"


def test_client_streaming(config):
    client = RouterClient(config)
    stub_providers(client._core, content="streamed")
    chunks = list(client.chat.completions.create(
        messages=[{"role": "user", "content": "hi"}], stream=True))
    assert chunks
    assert chunks[0].choices[0].delta.content == "streamed"
    assert chunks[0].decision_id


def test_sensitive_keeps_hard_prompt_on_device(config):
    core = RouterCore(config)
    stub_providers(core)
    resp, d = core.complete(
        [{"role": "user", "content":
          "design a distributed lock-free rate limiter and prove linearizability"}],
        sensitive=True,
    )
    assert d.tier == "local"
    assert d.provider == "ollama"


def test_sensitive_raises_rather_than_egress(config):
    core = RouterCore(config)
    stub_providers(core)
    # an image forces vision (cloud-only here); sensitive must raise, not egress
    from smartrouter.errors import NoEligibleModel
    with pytest.raises(NoEligibleModel):
        core.complete(
            [{"role": "user", "content": [
                {"type": "text", "text": "what is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:x"}}]}],
            sensitive=True,
        )


def test_classifier_failure_degrades_gracefully(config):
    core = RouterCore(config)

    class Boom:
        embedding_model_id = "boom"

        def score_and_embed(self, f):
            raise RuntimeError("classifier down")

    core.classifier = Boom()
    stub_providers(core)
    resp, d = core.complete([{"role": "user", "content": "hi"}])
    # score degrades to 0 -> cheapest tier, request still served
    assert d.tier == "local"
    assert resp["choices"][0]["message"]["content"]
