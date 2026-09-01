"""Provider transport contracts that need no running model server."""

from engine import providers
from engine.providers import OpenAICompatProvider, ProviderUnavailable


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
