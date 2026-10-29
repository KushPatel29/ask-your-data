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


def test_anthropic_provider_through_the_installed_sdk():
    """The provider against the real `anthropic` package, over a mocked transport.

    The other provider tests use stand-in clients, so a major SDK upgrade could
    change the request or response shapes without one of them noticing. This
    sends the provider's actual request through the installed SDK and parses an
    API-shaped reply back, then checks that a transport failure still maps to
    ProviderUnavailable.
    """
    import json

    import anthropic
    import httpx2 as httpx  # the SDK's own HTTP layer; it rejects plain httpx clients

    seen = {}

    def reply(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-test",
            "content": [{"type": "text", "text": "Checking."},
                        {"type": "tool_use", "id": "toolu_1", "name": "run_sql",
                         "input": {"sql": "SELECT 1"}}],
            "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 5},
        })

    client = anthropic.Anthropic(api_key="test-key", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(reply)))
    provider = providers.AnthropicProvider(client=client, model="claude-test")
    tools = [{"name": "run_sql", "description": "Run one read-only query.",
              "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}}}]
    out = provider.create_tool_call(system="Answer from the warehouse.",
                                    messages=[{"role": "user", "content": "How many rows?"}],
                                    tools=tools, max_tokens=64)

    assert seen["path"] == "/v1/messages"
    assert seen["body"]["tool_choice"] == {"type": "any"}
    assert seen["body"]["tools"][0]["name"] == "run_sql"
    assert out.tool_call.name == "run_sql" and out.tool_call.input == {"sql": "SELECT 1"}
    assert out.usage == {"input_tokens": 12, "output_tokens": 5}
    assert out.text == "Checking."

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    down = anthropic.Anthropic(api_key="test-key", max_retries=0,
                               http_client=httpx.Client(transport=httpx.MockTransport(refuse)))
    with pytest.raises(providers.ProviderUnavailable):
        providers.AnthropicProvider(client=down, model="claude-test").create_tool_call(
            system="s", messages=[{"role": "user", "content": "q"}], tools=tools, max_tokens=8)
