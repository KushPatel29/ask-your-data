"""In-process speech: what it promises, and what it refuses to do.

Nothing here downloads a model. The two engines are injected, because a test
that fetches 103 MB to prove a size bound is a test nobody runs twice — and the
things worth asserting about this module are its bounds, its failure modes and
which engine gets chosen, none of which need real weights.

The one thing weights WOULD prove — that the transcript is accurate — is not a
unit test's job and is not claimed here. It was measured by round-tripping the
TTS half into the STT half, and the numbers live in the module docstring.
"""

from __future__ import annotations

import io
import sys
import wave

import pytest

from engine import local_voice, voice


class FakeSTT:
    """Stands in for faster-whisper's WhisperModel."""

    def __init__(self, text="how many denied claims are there?"):
        self.text = text
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)

        class Segment:
            def __init__(self, t):
                self.text = t

        return [Segment(self.text)], object()


class FakeTTS:
    """Stands in for piper's PiperVoice, which writes RIFF into a wave handle."""

    def __init__(self, frames=b"\x00\x01" * 800):
        self.frames = frames
        self.spoken: list[str] = []

    def synthesize_wav(self, text, handle):
        self.spoken.append(text)
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22050)
        handle.writeframes(self.frames)


def make(**kw):
    return local_voice.LocalVoice(stt=FakeSTT(), tts=FakeTTS(), **kw)


# ---------------------------------------------------------------------------
# The prompt, and the negative result behind it
# ---------------------------------------------------------------------------

def test_the_prompt_names_the_terms_whose_spelling_is_load_bearing():
    """`Self-Pay` is a literal the planner binds a WHERE clause against, so its
    casing is not cosmetic. Unprompted, the model writes `Self-pay`."""
    prompt = local_voice.build_prompt()
    assert "Self-Pay" in prompt
    assert "denial rate" in prompt


def test_the_prompt_is_curated_rather_than_derived_from_the_lexicon():
    """The tempting version was tried three ways and measured worse each time;
    the reason is structural. `engine/semantics.py` normalises its 797 value
    phrases to lower case so the compiler can match them, and CASE is the whole
    thing the prompt buys. A derived list cannot teach `Self-Pay` when it only
    knows `self pay`."""
    assert local_voice.build_prompt() == local_voice.build_prompt(())
    doc = local_voice.build_prompt.__doc__ or ""
    assert "lower case" in doc, "the reason it is curated must stay written down"


def test_extra_terms_extend_the_prompt_without_duplicating_it():
    prompt = local_voice.build_prompt(["Self-Pay", "Ambulatory"])
    assert prompt.count("Self-Pay") == 1
    assert "Ambulatory" in prompt


# ---------------------------------------------------------------------------
# Bounds. The recording arrives from a browser and the answer from a model.
# ---------------------------------------------------------------------------

def test_an_empty_recording_is_refused_before_a_model_is_touched():
    with pytest.raises(voice.VoiceUnavailable):
        local_voice.LocalVoice(stt=None, tts=None).transcribe(b"")


def test_an_oversized_recording_is_refused_at_the_documented_ceiling():
    oversized = b"\x00" * (local_voice.MAX_AUDIO_BYTES + 1)
    with pytest.raises(voice.VoiceUnavailable) as caught:
        local_voice.LocalVoice(stt=None, tts=None).transcribe(oversized)
    assert "12 MB" in str(caught.value)


def test_narration_is_bounded_well_below_the_remote_ceiling():
    """Synthesis is 20-50x real time, so the remote 4,096-character ceiling is
    about three minutes of speech nobody listens to."""
    assert local_voice.MAX_TTS_CHARS < voice.MAX_TTS_CHARS
    engine = FakeTTS()
    client = local_voice.LocalVoice(stt=FakeSTT(), tts=engine)
    client.synthesize("word " * 5000)
    assert len(engine.spoken[0]) <= local_voice.MAX_TTS_CHARS


def test_an_answer_with_nothing_sayable_in_it_is_refused():
    """`speakable_text` strips code fences and URLs; a SQL-only answer reduces
    to nothing, and synthesizing silence would be a player that does not play."""
    with pytest.raises(voice.VoiceUnavailable):
        make().synthesize("```sql\nSELECT 1\n```")


# ---------------------------------------------------------------------------
# What comes back
# ---------------------------------------------------------------------------

def test_speech_is_a_complete_wav_and_says_so():
    """Piper writes a RIFF header, the remote provider returns MP3. The app
    plays whatever `mime_type` says — hard-coding audio/mpeg played a RIFF file
    as an MP3, which some browsers refuse and others render as a dead player."""
    speech = make().synthesize("Ten departments, from Fresh and Produce to Electronics.")
    assert speech.mime_type == "audio/wav"
    assert speech.audio[:4] == b"RIFF"
    with wave.open(io.BytesIO(speech.audio)) as handle:
        assert handle.getnframes() > 0


def test_the_transcript_carries_the_engine_that_produced_it():
    result = make().transcribe(b"\x00" * 64)
    assert result.text
    assert "faster-whisper" in result.model
    assert result.bytes_received == 64


def test_the_decoder_is_given_the_prompt_and_pinned_to_english():
    engine = FakeSTT()
    local_voice.LocalVoice(stt=engine, tts=FakeTTS()).transcribe(b"\x00" * 32)
    kwargs = engine.calls[0]
    assert kwargs["initial_prompt"] == local_voice.build_prompt()
    assert kwargs["language"] == "en"


def test_silence_is_reported_rather_than_answered():
    engine = FakeSTT(text="   ")
    with pytest.raises(voice.VoiceUnavailable):
        local_voice.LocalVoice(stt=engine, tts=FakeTTS()).transcribe(b"\x00" * 32)


# ---------------------------------------------------------------------------
# Choosing an engine
# ---------------------------------------------------------------------------

def test_an_explicit_endpoint_beats_the_in_process_models():
    """Someone who set ASK_VOICE_BASE_URL has said what they want, and quietly
    preferring a local model over an operator's choice reads as a bug."""
    _client, label = voice.resolve(environ={"ASK_VOICE_BASE_URL": "http://localhost:8000"})
    assert label == "self-hosted"


def test_with_nothing_configured_the_in_process_models_answer():
    """This is the case that matters: the public deployment has no key and no
    second container, and voice used to be unreachable there."""
    _client, label = voice.resolve(environ={})
    assert label == ("local" if local_voice.available() else "")


def test_the_preference_can_be_forced_in_either_direction():
    """A deployment proving which path runs should not have to uninstall a
    package to do it."""
    assert voice.engine_preference({"ASK_VOICE_ENGINE": "remote"}) == "remote"
    assert voice.engine_preference({"ASK_VOICE_ENGINE": "nonsense"}) == "auto"
    _client, label = voice.resolve(environ={"ASK_VOICE_ENGINE": "remote"})
    assert label == "", "forced remote with nothing configured resolves to nothing"


def test_the_rail_names_the_engine_that_really_resolved():
    """The rail used to answer "is an endpoint or key set", so on the public
    deployment it said voice was unconfigured while speech was running inside
    the process."""
    assert "no key" in voice.describe_engine("local")
    assert "cloud" in voice.describe_engine("cloud")
    assert "unavailable" in voice.describe_engine("")


def test_importing_the_module_never_loads_a_model():
    """`installed()` is called while drawing the status rail on every rerun. A
    probe that downloaded 100 MB to answer "is voice available" would be a
    worse bug than the missing feature."""
    local_voice.reset()
    engines = local_voice.installed()
    assert isinstance(engines.stt, bool) and isinstance(engines.tts, bool)
    assert local_voice._STT is None and local_voice._TTS is None


def test_a_model_that_fails_its_checksum_is_discarded_not_cached(tmp_path):
    """Left in place, a corrupt or substituted file would be re-verified and
    re-rejected forever — one bad download becoming a permanent outage."""
    bad = tmp_path / "en_US-lessac-low.onnx"
    bad.write_bytes(b"not a model")
    with pytest.raises(voice.VoiceUnavailable):
        local_voice._verify(bad, local_voice.TTS_SHA256[bad.name])
    assert not bad.exists()


def test_the_app_survives_the_voice_packages_being_absent(monkeypatch):
    """The load-bearing safety claim of this module, asserted rather than hoped.

    These are two heavy optional wheels. A deployment that cannot install them
    — a platform without a manylinux build, an air-gapped mirror, a pinned
    resolver — must still answer questions. So the failure mode has to be a
    quiet False and a rail that says "unavailable", never an exception on a
    page that was only ever going to show a Listen button.

    The remote seam has to keep working through the same failure, because a
    self-hosted endpoint has nothing to do with whether CTranslate2 imports.
    """
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in ("faster_whisper", "piper"):
            raise ImportError(f"simulated: {name} is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for module in [m for m in list(sys.modules) if m.split(".")[0] in ("faster_whisper", "piper")]:
        monkeypatch.delitem(sys.modules, module, raising=False)

    assert local_voice.installed() == local_voice.Engines(stt=False, tts=False)
    assert local_voice.available() is False

    client, label = voice.resolve(environ={})
    assert (client, label) == (None, "")
    assert "unavailable" in voice.describe_engine(label)

    remote, remote_label = voice.resolve(
        environ={"ASK_VOICE_BASE_URL": "http://localhost:8000"})
    assert remote_label == "self-hosted" and remote is not None
