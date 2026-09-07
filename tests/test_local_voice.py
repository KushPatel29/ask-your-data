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

import hashlib
import io
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine import local_voice, voice

ROOT = Path(__file__).resolve().parent.parent


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


def test_the_default_voice_is_male_permissive_and_checksum_pinned():
    """The better-scoring hfc voice is non-commercial, so it cannot quietly
    become the default of an app presented as enterprise-ready."""
    assert local_voice.DEFAULT_TTS_VOICE == "en_US-joe-medium"
    files = local_voice._pinned_tts_files(local_voice.DEFAULT_TTS_VOICE)
    assert set(files) == {
        "en_US-joe-medium.onnx", "en_US-joe-medium.onnx.json",
    }
    assert all(len(digest) == 64 for digest in files.values())


def test_an_unreviewed_voice_fails_before_download():
    with pytest.raises(voice.VoiceUnavailable, match="not checksum-pinned"):
        local_voice._pinned_tts_files("en_US-some-new-voice")


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


def test_local_speech_reports_the_voice_that_actually_generated_it():
    speech = make().synthesize("There are twelve claims.", voice="a-remote-voice")
    assert speech.voice == local_voice.TTS_VOICE


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


# ---------------------------------------------------------------------------
# Supply chain. Both halves execute weights fetched over the network.
# ---------------------------------------------------------------------------

def test_both_speech_models_are_pinned_not_just_the_voice():
    """The interface told every visitor "both models ... are cached with a
    pinned checksum" while only the voice was.

    `WhisperModel("tiny.en")` resolves through huggingface_hub to whatever the
    repository's `main` points at today, so the weights this process executes
    could change with no commit here. The claim covered both halves; now the
    control does.
    """
    assert local_voice.STT_REVISION and len(local_voice.STT_REVISION) == 40
    assert local_voice.STT_REPO == "Systran/faster-whisper-tiny.en"
    assert set(local_voice.STT_SHA256) == {
        "config.json", "model.bin", "tokenizer.json", "vocabulary.txt",
    }
    assert all(len(d) == 64 for d in local_voice.STT_SHA256.values())


def test_stt_verifies_artifacts_before_initializing_the_model(monkeypatch, tmp_path):
    calls = []
    downloaded = []
    payload = b"reviewed model artifact"
    expected = {name: hashlib.sha256(payload).hexdigest() for name in local_voice.STT_SHA256}
    monkeypatch.setattr(local_voice, "STT_SHA256", expected)
    monkeypatch.setattr(local_voice, "STT_MODEL", local_voice.DEFAULT_STT_MODEL)
    monkeypatch.setattr(local_voice, "_whisper_dir", lambda: tmp_path)

    def download(path, digest, url):
        downloaded.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def construct(path, **kwargs):
        calls.append((path, kwargs))
        for name, digest in expected.items():
            assert hashlib.sha256((Path(path) / name).read_bytes()).hexdigest() == digest
        return object()

    monkeypatch.setattr(local_voice, "_download_verified", download)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=construct))
    local_voice._load_stt()
    assert all(f"/resolve/{local_voice.STT_REVISION}/" in url for url in downloaded)
    assert calls[0][1]["local_files_only"] is True

    # Even a buggy downloader cannot hand unchecked artifacts to CTranslate2.
    calls.clear()

    def tampered(path, digest, url):
        path.write_bytes(b"substituted graph")

    monkeypatch.setattr(local_voice, "_download_verified", tampered)
    with pytest.raises(voice.VoiceUnavailable, match="pinned checksum"):
        local_voice._load_stt()
    assert not calls


def test_an_overridden_stt_model_fails_before_download_or_instantiation(monkeypatch):
    monkeypatch.setattr(local_voice, "STT_MODEL", "unreviewed/model")
    assert not local_voice.stt_is_pinned()
    with pytest.raises(voice.VoiceUnavailable, match="not checksum-pinned"):
        local_voice._load_stt()


def test_a_tampered_snapshot_file_is_rejected_and_discarded(tmp_path):
    """Same rule the voice already follows: a mismatch is a hard failure, and
    the file is removed so one bad download does not become a permanent
    outage by being re-verified and re-rejected forever."""
    root = (tmp_path / f"models--{local_voice.STT_REPO.replace('/', '--')}"
            / "snapshots" / local_voice.STT_REVISION)
    root.mkdir(parents=True)
    bad = root / "config.json"
    bad.write_bytes(b'{"tampered": true}')

    with pytest.raises(voice.VoiceUnavailable, match="did not match its pinned checksum"):
        local_voice._verify_stt_snapshot(tmp_path)
    assert not bad.exists()


def test_an_incomplete_cache_cannot_skip_integrity_checks(tmp_path):
    with pytest.raises(voice.VoiceUnavailable, match="cache is incomplete"):
        local_voice._verify_stt_snapshot(tmp_path)


def test_a_zero_frame_wav_is_not_returned_as_successful_speech():
    with pytest.raises(voice.VoiceUnavailable, match="produced no audio"):
        local_voice.LocalVoice(tts=FakeTTS(frames=b"")).synthesize("There are twelve claims.")


@pytest.mark.parametrize("fault", ["header", "truncated"])
def test_malformed_speech_is_contained_in_an_actionable_voice_error(fault):
    class BrokenTTS(FakeTTS):
        def synthesize_wav(self, text, handle):
            super().synthesize_wav(text, handle)
            if fault == "header":
                handle._file.seek(0)
                handle._file.write(b"NOPE")
            else:
                handle._file.truncate(48)

    with pytest.raises(voice.VoiceUnavailable, match="Try again"):
        local_voice.LocalVoice(tts=BrokenTTS()).synthesize("There are twelve claims.")


def test_an_english_only_model_does_not_silently_accept_other_languages():
    engine = FakeSTT()
    with pytest.raises(voice.VoiceUnavailable, match="English only"):
        local_voice.LocalVoice(stt=engine).transcribe(b"recording", language="hi")
    assert not engine.calls


def test_an_interrupted_download_leaves_no_final_or_partial_artifact(monkeypatch, tmp_path):
    class Interrupted(io.BytesIO):
        def read1(self, *args):
            if self.tell():
                raise TimeoutError("connection interrupted")
            return super().read(4)

    monkeypatch.setattr(local_voice, "urlopen", lambda *args, **kw: Interrupted(b"model bytes"))
    destination = tmp_path / "voice.onnx"
    with pytest.raises(TimeoutError):
        local_voice._download_verified(destination, "0" * 64, "https://example.invalid/model")
    assert not destination.exists()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("byte_limit,deadline", [(3, 120), (1024, -1)])
def test_download_size_and_deadline_limits_discard_partial_artifacts(
        monkeypatch, tmp_path, byte_limit, deadline):
    payload = b"model bytes"
    monkeypatch.setattr(local_voice, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    monkeypatch.setattr(local_voice, "MAX_MODEL_BYTES", byte_limit)
    monkeypatch.setattr(local_voice, "MODEL_DOWNLOAD_DEADLINE", deadline)
    with pytest.raises(voice.VoiceUnavailable, match="exceeded its limit"):
        local_voice._download_verified(
            tmp_path / "voice.onnx", hashlib.sha256(payload).hexdigest(),
            "https://example.invalid/model",
        )
    assert not list(tmp_path.iterdir())


def test_corrupt_cache_is_repaired_and_verified_cache_needs_no_network(monkeypatch, tmp_path):
    payload = b"verified model bytes"
    expected = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "voice.onnx"
    destination.write_bytes(b"an interrupted earlier download")
    requests = []

    def fetch(url, *, timeout):
        assert not destination.exists()
        requests.append(timeout)
        return io.BytesIO(payload)

    monkeypatch.setattr(local_voice, "urlopen", fetch)
    local_voice._download_verified(destination, expected, "https://example.invalid/model")
    assert destination.read_bytes() == payload
    assert requests == [local_voice.MODEL_DOWNLOAD_TIMEOUT]
    local_voice._download_verified(destination, expected, "https://example.invalid/model")
    assert len(requests) == 1
    assert list(tmp_path.iterdir()) == [destination]


def test_bad_download_never_reaches_the_final_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(local_voice, "urlopen", lambda *a, **kw: io.BytesIO(b"substituted"))
    with pytest.raises(voice.VoiceUnavailable, match="pinned checksum"):
        local_voice._download_verified(
            tmp_path / "voice.onnx", "0" * 64, "https://example.invalid/model",
        )
    assert not list(tmp_path.iterdir())


def test_simultaneous_speech_requests_do_not_run_inference_concurrently():
    class SharedEngine(FakeTTS):
        active = 0
        maximum = 0

        def synthesize_wav(self, text, handle):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            time.sleep(0.02)
            super().synthesize_wav(text, handle)
            self.active -= 1

    engine = SharedEngine()
    client = local_voice.LocalVoice(tts=engine)
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(client.synthesize, ["There are twelve claims."] * 4))
    assert len(responses) == 4
    assert engine.maximum == 1


def test_busy_voice_has_a_bounded_wait_and_an_actionable_retry(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    engine = FakeTTS()
    monkeypatch.setattr(local_voice, "INFERENCE_WAIT_SECONDS", 0.01)

    def occupy():
        with local_voice._TTS_LOCK:
            entered.set()
            release.wait(2)

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(occupy)
        assert entered.wait(1)
        try:
            with pytest.raises(voice.VoiceUnavailable, match="busy"):
                local_voice.LocalVoice(tts=engine).synthesize("There are twelve claims.")
            assert not engine.spoken
        finally:
            release.set()
        worker.result(timeout=2)


def test_readiness_is_nonblocking_and_failed_loading_can_be_retried(monkeypatch):
    local_voice.reset()
    monkeypatch.setattr(local_voice, "installed", lambda: local_voice.Engines(True, True))
    entered = threading.Event()
    release = threading.Event()

    def failing_load():
        entered.set()
        release.wait(2)
        raise TimeoutError("internal host information")

    monkeypatch.setattr(local_voice, "_load_tts", failing_load)
    assert local_voice.readiness().tts == "not_loaded"
    with ThreadPoolExecutor(max_workers=2) as pool:
        worker = pool.submit(local_voice.LocalVoice()._text_to_speech)
        assert entered.wait(1)
        try:
            snapshot = pool.submit(local_voice.readiness).result(timeout=1)
            assert snapshot.tts == "loading"
        finally:
            release.set()
        with pytest.raises(voice.VoiceUnavailable):
            worker.result(timeout=2)
    assert local_voice.readiness().tts == "error"
    assert "internal host information" not in local_voice.readiness().tts_message
    monkeypatch.setattr(local_voice, "_load_tts", FakeTTS)
    local_voice.LocalVoice()._text_to_speech()
    assert local_voice.readiness().tts == "ready"
    assert not local_voice.readiness().tts_message
    local_voice.reset()


def test_the_interface_reads_the_pin_from_the_engine_rather_than_restating_it():
    """The caption must not be able to claim a control the engine is not
    applying — which is exactly the drift this whole finding was."""
    source = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "local_voice.stt_is_pinned()" in source
    assert "local_voice_pinned()" in source


# ---------------------------------------------------------------------------
# Warming the voice, which is what makes autoplay work at all.
# ---------------------------------------------------------------------------

def test_prewarm_does_not_run_at_import():
    """21.5s in front of the first page render would be worse than the problem
    it solves. The module must stay inert until something asks."""
    import ast
    from pathlib import Path

    source = (Path(local_voice.__file__)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_calls = [n for n in tree.body if isinstance(n, ast.Expr)
                       and isinstance(n.value, ast.Call)]
    assert not top_level_calls, "nothing may execute at import"


def test_prewarm_is_a_background_thread_and_never_blocks(monkeypatch):
    """It returns whether a warm-up was STARTED, not whether it finished.
    Nothing waits on it; a question that arrives first pays the load itself on
    the same lock, exactly as before."""
    import inspect

    source = inspect.getsource(local_voice.prewarm)
    assert "daemon=True" in source
    assert ".join(" not in source, "a warm-up that blocks is not a warm-up"


def test_prewarm_can_be_declined_by_a_deployment(monkeypatch):
    """It costs 135 MB in every container that serves anybody, including
    visitors who never ask for sound."""
    local_voice.reset()
    monkeypatch.setenv("ASK_VOICE_PREWARM", "0")
    assert local_voice.prewarm() is False


def test_prewarm_is_skipped_when_the_engine_is_not_installed(monkeypatch):
    monkeypatch.setenv("ASK_VOICE_PREWARM", "1")
    monkeypatch.setattr(local_voice, "installed",
                        lambda: local_voice.Engines(stt=False, tts=False))
    local_voice.reset()
    assert local_voice.prewarm() is False


def test_the_app_warms_the_voice_once_per_container():
    """@st.cache_resource rather than per session: the weights are read-only
    and identical for every visitor."""
    source = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "local_voice.prewarm()" in source
    assert "def _warm_voice()" in source
    index = source.index("def _warm_voice()")
    assert "@st.cache_resource" in source[max(0, index - 200):index]
