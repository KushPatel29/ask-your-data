"""Provider transport contracts that need no running model server."""

from engine.providers import OpenAICompatProvider


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
