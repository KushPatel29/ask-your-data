"""Voice is an input/output edge, not a path around the governed query engine."""

from types import SimpleNamespace

import pytest

from engine import voice


class _Call:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


class _Audio:
    def __init__(self, transcript="How many denied claims?", audio=b"ID3audio"):
        self.transcriptions = _Call(SimpleNamespace(text=transcript))
        self.speech = _Call(SimpleNamespace(read=lambda: audio))


class _Client:
    def __init__(self, **kwargs):
        self.audio = _Audio(**kwargs)


def test_transcription_uses_the_model_context_and_language_hint():
    client = _Client()
    service = voice.OpenAIVoice("unused-with-injected-client", client=client)
    got = service.transcribe(b"RIFF....", filename="question.wav", language="hi")

    assert got.text == "How many denied claims?"
    assert got.bytes_received == 8
    sent = client.audio.transcriptions.kwargs
    assert sent["model"] == voice.STT_MODEL
    assert sent["file"].name.endswith(".wav")
    assert "denial rate" in sent["extra_body"]["keywords"]
    assert sent["extra_body"]["languages"] == ["hi"]


def test_speech_is_prompted_for_precise_analytics_narration():
    client = _Client(audio=b"mp3-bytes")
    service = voice.OpenAIVoice("unused", client=client)
    got = service.synthesize("Denial rate is **8.2%**.", voice="cedar")

    assert got.audio == b"mp3-bytes" and got.voice == "cedar"
    sent = client.audio.speech.kwargs
    assert sent["model"] == voice.TTS_MODEL
    assert sent["input"] == "Denial rate is 8.2%."
    assert "percentages" in sent["instructions"]
    assert sent["response_format"] == "mp3"


def test_markdown_sql_urls_and_excess_space_are_not_read_aloud():
    raw = """**Result:** 42. ```sql\nSELECT secret FROM t\n``` See https://example.com/x"""
    assert voice.speakable_text(raw) == "Result: 42. See"


def test_recordings_are_bounded_before_any_provider_call():
    client = _Client()
    service = voice.OpenAIVoice("unused", client=client)
    with pytest.raises(voice.VoiceUnavailable, match="voice limit"):
        service.transcribe(b"x" * (voice.MAX_AUDIO_BYTES + 1))
    assert client.audio.transcriptions.kwargs is None


def test_empty_audio_and_empty_transcripts_have_actionable_failures():
    service = voice.OpenAIVoice("unused", client=_Client())
    with pytest.raises(voice.VoiceUnavailable, match="Record a question"):
        service.transcribe(b"")

    silent = voice.OpenAIVoice("unused", client=_Client(transcript="   "))
    with pytest.raises(voice.VoiceUnavailable, match="No speech was detected"):
        silent.transcribe(b"RIFF")


def test_provider_errors_never_echo_credentials_or_response_payloads():
    key = "sk-example-secret-that-must-not-leak"
    client = _Client()
    client.audio.transcriptions = _Call(error=RuntimeError(f"invalid key {key}"))
    service = voice.OpenAIVoice(key, client=client)
    with pytest.raises(voice.VoiceUnavailable) as caught:
        service.transcribe(b"RIFF")
    assert key not in str(caught.value)
    assert "Check the OpenAI key" in str(caught.value)


def test_server_key_is_optional_and_explicit():
    assert voice.server_api_key({}) == ""
    assert voice.server_api_key({"OPENAI_API_KEY": " sk-live "}) == "sk-live"
