"""Provider transport contracts that need no running model server."""

import pytest

from engine import providers
from engine.assistant import Assistant
from engine.providers import (
    AnthropicProvider,
    OpenAICompatProvider,
    ProviderUnavailable,
    build_provider,
    model_provider_configured,
)


def test_openai_compatible_summary_uses_the_versioned_route(monkeypatch):
    provider = OpenAICompatProvider(base_url="http://local.test", model="local-model")
    seen = []

    def fake_post(path, payload):
        seen.append((path, payload))
        return {"choices": [{"message": {"content": "42"}}]}

    monkeypatch.setattr(provider, "_post", fake_post)
    response = provider.complete(
        system="summarise", messages=[{"role": "user", "content": "result"}],
        max_tokens=50,
    )

    assert response.text == "42"
    assert seen[0][0] == "/v1/chat/completions"


def test_openai_fallbacks_share_one_decreasing_request_budget(monkeypatch):
    provider = OpenAICompatProvider(
        base_url="http://local.test", model="local-model", mode="auto"
    )
    clock = iter([100.0, 100.0, 104.0, 108.5])
    timeouts = []

    monkeypatch.setattr(providers.time, "monotonic", lambda: next(clock))

    def fake_post(_path, _payload, timeout_s):
        timeouts.append(timeout_s)
        if len(timeouts) < 3:
            raise ProviderUnavailable("mode unsupported")
        return {
            "choices": [{
                "message": {
                    "content": '{"name":"cannot_answer","input":{"reason":"scope"}}'
                }
            }]
        }

    monkeypatch.setattr(provider, "_post", fake_post)
    response = provider.create_tool_call(
        system=[{"type": "text", "text": "system"}],
        messages=[{"role": "user", "content": "question"}],
        tools=[{
            "name": "cannot_answer",
            "description": "refuse",
            "input_schema": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        }],
        max_tokens=100,
        timeout_s=10,
    )

    assert response.tool_call is not None
    assert response.tool_call.name == "cannot_answer"
    assert timeouts == [10.0, 6.0, 1.5]


def test_local_provider_is_enabled_without_an_anthropic_key():
    assert model_provider_configured({
        "ASK_PROVIDER": "ollama",
        "ASK_LOCAL_BASE_URL": "http://model.internal:11434",
    })
    assert not model_provider_configured({"ASK_PROVIDER": "anthropic"})
    assert model_provider_configured({
        "ASK_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "test-key",
    })


def test_provider_configuration_typos_fail_closed(monkeypatch):
    with pytest.raises(ProviderUnavailable, match="unsupported ASK_PROVIDER"):
        model_provider_configured({"ASK_PROVIDER": "ollmaa"})
    with pytest.raises(ProviderUnavailable, match="unsupported model provider"):
        build_provider(name="claud")

    monkeypatch.setenv("ASK_LOCAL_MODE", "grammer")
    with pytest.raises(ProviderUnavailable, match="ASK_LOCAL_MODE"):
        OpenAICompatProvider(base_url="http://local.test")

    monkeypatch.delenv("ASK_LOCAL_MODE", raising=False)
    with pytest.raises(ProviderUnavailable, match="ASK_LOCAL_BASE_URL"):
        OpenAICompatProvider(base_url="file:///etc/passwd")
    with pytest.raises(ProviderUnavailable, match="embedded credentials"):
        OpenAICompatProvider(base_url="https://user:secret@model.example")


def test_explicit_supported_provider_names_build_expected_adapters():
    assert isinstance(build_provider(name="local"), OpenAICompatProvider)
    # Supplying a client avoids constructing a real SDK client while proving
    # the normalized selector reaches the intended adapter.
    assert isinstance(build_provider(name="claude", client=object()), AnthropicProvider)


def test_assistant_does_not_override_the_local_model_with_anthropic_default(con, monkeypatch):
    monkeypatch.setenv("ASK_PROVIDER", "ollama")
    monkeypatch.setenv("ASK_LOCAL_BASE_URL", "http://model.internal:11434")
    monkeypatch.setenv("ASK_LOCAL_MODEL", "qwen-local-sql")

    assistant = Assistant(con)

    assert isinstance(assistant.provider, OpenAICompatProvider)
    assert assistant.provider.model == "qwen-local-sql"
    assert assistant.model == "qwen-local-sql"
