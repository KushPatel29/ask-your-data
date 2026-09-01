"""
Speech that needs no API key, no account, and no second server.

WHY THIS EXISTS
`engine/voice.py` already spoke to an OpenAI-compatible endpoint, and one of
its two modes was genuinely free: point `ASK_VOICE_BASE_URL` at a self-hosted
Speaches container and you get faster-whisper and Kokoro with nothing billed.
That mode is real and it is still here. It is also unreachable on the public
deployment, because Streamlit Community Cloud runs one process and there is
nowhere to put a second container — so every visitor to the demo saw
"Recordings are sent only after voice is enabled" and voice was, in practice, a
feature you had to already be an operator to use.

This module removes the server. Both halves run in THIS process, on models
whose weights are open:

  * STT - faster-whisper `tiny.en`, CTranslate2, int8 on CPU. ~40 MB.
  * TTS - Piper `en_US-joe-medium`, a male VITS model on onnxruntime. ~63 MB.

WHICH VOICE, AND THE PREMISE THAT TURNED OUT TO BE WRONG
Five en_US voices were downloaded and driven through 30 synthesis takes each,
then transcribed back by the STT half above and scored on exact round-trip:

    en_US-hfc_male-medium   30/30   +87.0 MB   34x real time
    en_US-lessac-low        27/30   +87.3 MB   "about 19% of IT allowed amount"
    en_US-ryan-low          27/30   +87.1 MB   "tonight claims" for "denied claims"
    en_US-joe-medium        26/30   +87.0 MB   "of the amount" for "allowed amount"
    en_US-norman-medium      2/10   +94.4 MB   corrupted the NUMBER, 5 takes of 5

The premise going in was that a `medium` voice would cost memory a 611 MB app
cannot spare. It does not: in Piper, `low` and `medium` are the SAME model at
different sample rates — every single-speaker en_US voice at both tiers is the
same ~63 MB file, and only `high` is larger. There was no trade-off to make.

One take is not a measurement; VITS sampling is stochastic, which is why the
score is out of 30. norman is disqualified on its own: it turned "876" into
"the 76 to 9" and "19%" into "99%", and a voice that corrupts the number is
unusable in an app whose entire argument is the number.

LICENCE, STATED BECAUSE IT IS A REAL CONSTRAINT
The best round-trip score belonged to hfc_male, but its source dataset is
CC BY-NC-SA 4.0 and therefore unsuitable as an enterprise default. Joe's source
dataset is CC0, its Piper model repository is MIT, it is explicitly male, and
it costs the same memory. The default therefore prefers deployability over four
extra exact transcripts in a synthetic round-trip test. A user reviews every
STT transcript before it runs, while the TTS answer is already visible beside
its audio control.

Measured on this machine, warm: synthesis 0.086-0.123 s for 3.0-4.2 s of speech
(34-35x faster than real time), transcription 0.02-0.32 s for a short question.
Cold, each pays a one-time model download and a first-call warm-up, which is
why both are lazy and neither is touched at import.

ONNXRUNTIME WAS ALREADY HERE
Piper is a VITS graph on the same onnxruntime the schema index uses, so TTS
adds a model rather than a runtime. faster-whisper does bring CTranslate2.

WHAT IT COSTS, MEASURED RATHER THAN ESTIMATED
Resident set of one process, each step cumulative:

    warehouse + semantic layer + schema index      330 MB
    + Piper TTS loaded                             465 MB
    + faster-whisper STT loaded                    611 MB

So speech roughly doubles the floor, and that is the honest headline. Two
things keep it acceptable rather than reckless. Both engines are LAZY, so a
visitor who never presses Listen and never records pays none of it — the app
sits at 330 MB exactly as it did before this module existed. And the ceiling
that actually bit this project was Render's 512 MB free tier, which the app had
already outgrown at a 564 MB peak; the public deployment is Streamlit Community
Cloud, which has several times that.

On disk it is about 80 MB of wheels and 103 MB of weights, the weights fetched
once per container into the same cache the schema index uses.

EVERYTHING DEGRADES, NOTHING BREAKS
`available()` is the only gate the app needs. If the optional packages are not
installed, or a model cannot be fetched, it returns False and the interface
behaves exactly as it did before this module existed — the remote seam, or a
notice that voice is unconfigured. A deployment that cannot install
CTranslate2 must still answer questions, so no import here is allowed to run at
module scope.

WHY tiny.en AND NOT base.en
Measured against three utterances synthesized by the TTS half above, with the
repo's own DOMAIN_KEYWORDS supplied as the decoder prompt, `tiny.en` and
`base.en` produced the SAME transcripts — and `base.en` took 41.8 s to load
against 0.9 s. The prompt is what earns the accuracy: without it `tiny.en`
heard "Self-bay" for "Self-Pay", and with it the value binds correctly. Paying
75 MB and 40 s for a model that does not read better is not a quality decision,
it is an unexamined default.

The transcript is still shown for confirmation before it can become a question.
That is not a workaround for a small model; it is the same rule the remote path
follows, because speech is an edge around the pipeline and never a way into it.
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

from engine.voice import (
    DOMAIN_KEYWORDS,
    MAX_AUDIO_BYTES,
    Speech,
    Transcript,
    VoiceUnavailable,
    speakable_text,
)

# The interpreter-visible names of the two open models. Pinned, because "the
# latest tiny model" is not a reproducible claim and this repository does not
# make those.
STT_MODEL = os.environ.get("ASK_LOCAL_STT_MODEL", "tiny.en")
DEFAULT_TTS_VOICE = "en_US-joe-medium"
TTS_VOICE = os.environ.get("ASK_LOCAL_TTS_VOICE", DEFAULT_TTS_VOICE)

# Checksums for the Piper voice, in the same spirit as engine/vector_index.py:
# the download is over HTTPS from a host this project does not control, and a
# model file is code as much as a wheel is. A mismatch is a hard failure rather
# than a warning, because a voice model that is not the one measured above is
# not the one this module documents.
TTS_SHA256 = {
    "en_US-joe-medium.onnx":
        "58afce0321b8d9c46d7cdf9c16500cc55a793b4220212dba6b70fb788b3baf06",
    "en_US-joe-medium.onnx.json":
        "3d6d5410b3795cb1950595247ef8f06190719e6fdbfa3a2356d8ec368e1aad33",
    # The previous voice, kept so an existing cache and ASK_LOCAL_TTS_VOICE
    # still verify rather than falling through the check unpinned.
    "en_US-lessac-low.onnx":
        "f7d01dde371555732c4c314111ac79672b1a5ce2fc19266ab42178fd8df7f375",
    "en_US-lessac-low.onnx.json":
        "45754dfdebb3b8661c3fc564713772deec6e064feeb5b4e9594857dc7305193a",
}

# The decoder prompt is built by build_prompt() rather than stored, so there is
# exactly one place that decides what the model is biased toward.


def build_prompt(extra=()) -> str:
    """The decoder prompt, and the negative result behind it.

    Measured on speech synthesized by the TTS half of this module: with no
    prompt, `tiny.en` writes `Self-pay`; with `Self-Pay` named in the prompt it
    writes `Self-Pay`. The casing is not cosmetic — it is a literal the planner
    binds against — so a prompt is worth having.

    THE TEMPTING VERSION DOES NOT WORK, AND IT WAS TRIED THREE WAYS.
    `engine/semantics.py` already extracts 797 value phrases from the warehouse
    by `SELECT DISTINCT`, and biasing the decoder with them would have been
    derived rather than hand-written, which is this repository's whole
    preference. Every ranking made transcription WORSE than the curated list:

      * longest first pulled a 200-character clinical query narrative in, ate
        the budget, and produced "about 19% of it allowed amount"
      * longest-within-a-length-bound spent it on `models staging stg
        warehousessql` — dbt model paths, which nobody says out loud
      * source-column cardinality ascending, the most principled of the three,
        filled up on columns holding ONE distinct value: `true`, `pass`,
        `success`, `error`. It heard "Softpay".

    The reason is structural, not a tuning failure. The lexicon is normalised to
    lower case so the compiler can match it, and CASE is exactly what the
    prompt was for. A derived list cannot teach `Self-Pay` when it only knows
    `self pay`.

    So the vocabulary here is curated, in `engine/voice.DOMAIN_KEYWORDS`, and
    that is what that constant is FOR. It is a short list of terms whose
    spelling matters in this product, which is a different thing from every
    value in the warehouse.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for phrase in list(DOMAIN_KEYWORDS) + list(extra):
        text = " ".join(str(phrase).split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        terms.append(text)
    return (
        "A short analytics question about an enterprise data warehouse. Terms: "
        + ", ".join(terms)
        + "."
    )


# Narration is bounded well below the remote ceiling. Synthesis is ~34x real
# time, so 1,200 characters is 62 s of audio and 1.7 s of work; the remote 4,096
# would be nearer three minutes of speech nobody listens to.
MAX_TTS_CHARS = 1200

_LOCK = threading.RLock()
_STT = None
_TTS = None


def _cache_root() -> Path:
    """The same cache the schema index uses, so one directory holds every model."""
    override = os.environ.get("ASK_MODEL_CACHE", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".cache" / "ask-your-data" / "models"


def _voice_dir() -> Path:
    return _cache_root() / "piper"


def _whisper_dir() -> Path:
    return _cache_root() / "whisper"


@dataclass(frozen=True)
class Engines:
    """What this process can actually do, as opposed to what it imports."""

    stt: bool
    tts: bool

    @property
    def any(self) -> bool:
        return self.stt or self.tts


def installed() -> Engines:
    """Whether the optional packages import. No model is fetched here.

    Deliberately cheap and side-effect free: the app calls it while drawing the
    status rail on every rerun, and a probe that downloaded 100 MB to answer
    "is voice available" would be a worse bug than the missing feature.
    """
    try:
        import faster_whisper  # noqa: F401
        stt = True
    except Exception:  # noqa: BLE001 - an optional dependency, not an error
        stt = False
    try:
        import piper  # noqa: F401
        tts = True
    except Exception:  # noqa: BLE001
        tts = False
    return Engines(stt=stt, tts=tts)


def available() -> bool:
    return installed().any


def _verify(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        # Remove it: a corrupt or substituted file left in the cache would be
        # re-verified and re-rejected on every call, turning one bad download
        # into a permanently broken feature.
        path.unlink(missing_ok=True)
        raise VoiceUnavailable(
            f"The speech model {path.name} did not match its pinned checksum and "
            "was discarded. Retry, or configure a speech endpoint instead."
        )


def _pinned_tts_files(voice_id: str) -> dict[str, str]:
    """The two artifacts a supported Piper voice must verify against.

    This check intentionally runs before importing Piper or touching the
    network. An operator typo must fail closed, not download an unreviewed
    executable graph and silently skip the checksum because the mapping did
    not contain its name.
    """
    names = (f"{voice_id}.onnx", f"{voice_id}.onnx.json")
    expected = {name: TTS_SHA256.get(name, "") for name in names}
    if not all(expected.values()):
        raise VoiceUnavailable(
            f"The configured Piper voice {voice_id!r} is not checksum-pinned by "
            "this release. Choose a supported voice or deploy a self-hosted "
            "speech endpoint."
        )
    return expected


def _load_tts():
    """Piper, downloaded once and checksum-verified."""
    expected = _pinned_tts_files(TTS_VOICE)
    from piper import PiperVoice
    from piper.download_voices import download_voice

    directory = _voice_dir()
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / f"{TTS_VOICE}.onnx"
    config = directory / f"{TTS_VOICE}.onnx.json"
    if not (model.is_file() and config.is_file()):
        download_voice(TTS_VOICE, directory)
    for path in (model, config):
        _verify(path, expected[path.name])
    return PiperVoice.load(model)


def _load_stt():
    from faster_whisper import WhisperModel

    directory = _whisper_dir()
    directory.mkdir(parents=True, exist_ok=True)
    # int8 on CPU is the whole reason this fits: float16 has no CPU path and
    # float32 triples the resident model for a transcript a human then reads
    # and confirms before anything runs.
    return WhisperModel(
        STT_MODEL, device="cpu", compute_type="int8", download_root=str(directory)
    )


class LocalVoice:
    """The same two methods `engine.voice.OpenAIVoice` offers, with no network.

    Interface-compatible on purpose. app/streamlit_app.py should not contain a
    branch for which engine is speaking: the transcript is confirmed and the
    answer is read aloud the same way either way, and the only thing that
    differs is what the rail reports.
    """

    local = True

    def __init__(self, *, stt=None, tts=None, extra_terms=()) -> None:
        self._stt = stt
        self._tts = tts
        self.prompt = build_prompt(extra_terms)
        self.stt_model = f"faster-whisper {STT_MODEL}"
        self.tts_model = f"piper {TTS_VOICE}"

    # -- lazy singletons -------------------------------------------------
    # Process-global rather than per-session, because the weights are read-only
    # and identical for every visitor; a model per Streamlit session would
    # multiply 100 MB by the number of open tabs.
    def _speech_to_text(self):
        global _STT
        if self._stt is not None:
            return self._stt
        with _LOCK:
            if _STT is None:
                try:
                    globals()["_STT"] = _load_stt()
                except VoiceUnavailable:
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise VoiceUnavailable(
                        "The local speech-to-text model could not be loaded. It "
                        "downloads once on first use; check the network and retry."
                    ) from exc
        return _STT

    def _text_to_speech(self):
        global _TTS
        if self._tts is not None:
            return self._tts
        with _LOCK:
            if _TTS is None:
                try:
                    globals()["_TTS"] = _load_tts()
                except VoiceUnavailable:
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise VoiceUnavailable(
                        "The local text-to-speech voice could not be loaded. It "
                        "downloads once on first use; check the network and retry."
                    ) from exc
        return _TTS

    # -- the seam --------------------------------------------------------
    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "question.wav",
        mime_type: str = "audio/wav",
        language: str = "",
    ) -> Transcript:
        payload = bytes(audio or b"")
        if not payload:
            raise VoiceUnavailable("Record a question before transcribing it.")
        if len(payload) > MAX_AUDIO_BYTES:
            raise VoiceUnavailable(
                "That recording is larger than the "
                f"{MAX_AUDIO_BYTES // (1024 * 1024)} MB voice limit."
            )
        model = self._speech_to_text()
        try:
            segments, _info = model.transcribe(
                io.BytesIO(payload),
                beam_size=1,
                language=(language or "en"),
                initial_prompt=self.prompt,
                vad_filter=True,
            )
            text = " ".join(segment.text for segment in segments).strip()
        except Exception as exc:  # noqa: BLE001 - engine errors must be contained
            raise VoiceUnavailable(
                "Transcription failed on the local model. Re-record and try again."
            ) from exc
        if not text:
            raise VoiceUnavailable(
                "No speech was detected. Record a clear, short question and try again."
            )
        return Transcript(text=text, model=self.stt_model, bytes_received=len(payload))

    def synthesize(self, text: str, *, voice: str = "") -> Speech:
        narration = speakable_text(text, limit=MAX_TTS_CHARS)
        if not narration:
            raise VoiceUnavailable("There is no answer text to read aloud.")
        engine = self._text_to_speech()
        buffer = io.BytesIO()
        try:
            # Piper writes a RIFF header, so the bytes are a complete WAV file
            # and the browser can play them without a container guess. The
            # remote path returns MP3; `Speech.mime_type` is what keeps the
            # player honest about which it received.
            with wave.open(buffer, "wb") as handle:
                engine.synthesize_wav(narration, handle)
        except Exception as exc:  # noqa: BLE001
            raise VoiceUnavailable(
                "Speech generation failed on the local voice. Try again."
            ) from exc
        audio = buffer.getvalue()
        if not audio:
            raise VoiceUnavailable("The local voice produced no audio. Try again.")
        return Speech(
            audio=audio,
            model=self.tts_model,
            voice=voice or TTS_VOICE,
            mime_type="audio/wav",
        )


def reset() -> None:
    """Drop the cached engines. For tests; the app never needs it."""
    global _STT, _TTS
    with _LOCK:
        _STT = None
        _TTS = None
