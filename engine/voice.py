"""Bounded speech-to-text and text-to-speech for the Streamlit interface.

Voice is deliberately a pair of edges around the existing analytics pipeline:

    recording -> transcript -> the same planner/model/guard/executor -> answer -> speech

It is not a second assistant and it never gets a path around the SQL boundary.
The transcript is shown for confirmation before it can become a question, and
speech is generated only when the visitor asks to hear a completed answer.

The OpenAI SDK is imported lazily so the keyless compiler and the full offline
test suite remain usable when the optional voice provider is unavailable.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass

STT_MODEL = os.environ.get("ASK_STT_MODEL", "gpt-transcribe")
TTS_MODEL = os.environ.get("ASK_TTS_MODEL", "gpt-4o-mini-tts")
DEFAULT_VOICE = os.environ.get("ASK_TTS_VOICE", "cedar")
ENV_BASE_URL = "ASK_VOICE_BASE_URL"
LOCAL_STT_MODEL = "Systran/faster-whisper-small"
LOCAL_TTS_MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"
LOCAL_DEFAULT_VOICE = "af_heart"

# OpenAI accepts larger files, but a spoken analytics question should not need
# anything close to that ceiling. A local bound keeps one browser session from
# retaining a meeting-sized upload in the Streamlit process.
MAX_AUDIO_BYTES = 12 * 1024 * 1024
MAX_TTS_CHARS = 4096

SUPPORTED_AUDIO_TYPES = {
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/m4a": ".m4a",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
}

# Terms whose spelling matters in this product. They are context, not an
# instruction to invent them: the transcription endpoint treats keywords as
# hints and the visitor still confirms the resulting text before it is run.
DOMAIN_KEYWORDS = (
    "DuckDB",
    "SQL",
    "denial rate",
    "net collection rate",
    "gross margin",
    "attrition rate",
    "order fill rate",
    "payer",
    "subledger",
    "reconciliation",
    "dbt",
    # Values, not concepts, and the distinction earns its place: these are
    # literals the planner binds a WHERE clause against, so their spelling is
    # load-bearing in a way that a topic word's is not. Measured with the local
    # model: unprompted it writes `Self-pay`, and `self pay` binds nothing.
    "Self-Pay",
    "Medicare",
    "Medicaid",
    "Commercial",
    "Denied",
    "Voluntary",
)


class VoiceUnavailable(RuntimeError):
    """A safe, user-facing voice-provider failure."""


@dataclass(frozen=True)
class Transcript:
    text: str
    model: str
    bytes_received: int


@dataclass(frozen=True)
class Speech:
    audio: bytes
    model: str
    voice: str
    mime_type: str = "audio/mpeg"


def server_api_key(environ: dict[str, str] | None = None) -> str:
    """The server-owned voice key, when deployment explicitly configured one."""
    env = os.environ if environ is None else environ
    return str(env.get("OPENAI_API_KEY", "") or "").strip()


def base_url(environ: dict[str, str] | None = None) -> str:
    """An optional self-hosted OpenAI-compatible speech endpoint."""
    env = os.environ if environ is None else environ
    return str(env.get(ENV_BASE_URL, "") or "").strip().rstrip("/")


def local_endpoint_configured(environ: dict[str, str] | None = None) -> bool:
    return bool(base_url(environ))


def configured_stt_model(environ: dict[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    fallback = LOCAL_STT_MODEL if local_endpoint_configured(env) else "gpt-transcribe"
    return str(env.get("ASK_STT_MODEL", fallback) or fallback)


def configured_tts_model(environ: dict[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    fallback = LOCAL_TTS_MODEL if local_endpoint_configured(env) else "gpt-4o-mini-tts"
    return str(env.get("ASK_TTS_MODEL", fallback) or fallback)


def _filename(name: str, mime_type: str) -> str:
    """Give the SDK a supported extension without trusting an uploaded path."""
    suffix = SUPPORTED_AUDIO_TYPES.get((mime_type or "").lower(), ".wav")
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(name or "question").rsplit(".", 1)[0])
    return f"{stem[:60] or 'question'}{suffix}"


def speakable_text(text: str, *, limit: int = MAX_TTS_CHARS) -> str:
    """Turn a rendered answer into clean narration, never SQL or raw markup."""
    clean = str(text or "")
    clean = re.sub(r"```.*?```", " ", clean, flags=re.S)
    clean = re.sub(r"`([^`]*)`", r"\1", clean)
    clean = re.sub(r"https?://\S+", "", clean)
    clean = re.sub(r"[*_#>|]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    clean = re.sub(r"\s+([.,!?;:%])", r"\1", clean)
    return clean[:limit].strip()


class OpenAIVoice:
    """Small provider seam around OpenAI's transcription and speech endpoints."""

    def __init__(
        self,
        api_key: str = "",
        *,
        base_url: str = "",
        client=None,
        stt_model: str | None = None,
        tts_model: str | None = None,
    ) -> None:
        endpoint = str(base_url or "").strip().rstrip("/")
        self.local = bool(endpoint)
        if not str(api_key or "").strip() and not self.local and client is None:
            raise VoiceUnavailable("Configure a local speech endpoint or add an OpenAI API key.")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on installation
                raise VoiceUnavailable(
                    "Voice needs the optional OpenAI SDK. Install the project requirements."
                ) from exc
            # Speaches needs no credential, but OpenAI's SDK rejects an empty
            # key before making a local request. This sentinel is sent only to
            # the explicitly configured self-hosted endpoint.
            client = OpenAI(
                api_key=str(api_key or "local-speech"),
                base_url=endpoint or None,
                timeout=120.0,
            )
        self.client = client
        self.base_url = endpoint
        self.stt_model = stt_model or (
            os.environ.get("ASK_STT_MODEL", LOCAL_STT_MODEL) if self.local else STT_MODEL
        )
        self.tts_model = tts_model or (
            os.environ.get("ASK_TTS_MODEL", LOCAL_TTS_MODEL) if self.local else TTS_MODEL
        )

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

        file_obj = io.BytesIO(payload)
        file_obj.name = _filename(filename, mime_type)
        kwargs: dict = {
            "model": self.stt_model,
            "file": file_obj,
            "prompt": (
                "A short analytics question about an enterprise data warehouse. "
                "Preserve metric names, acronyms, table concepts, and numbers exactly."
            ),
        }
        context = {"keywords": list(DOMAIN_KEYWORDS)}
        if language:
            context["languages"] = [language]
        # OpenAI's GPT transcription models accept richer context. Speaches'
        # faster-whisper route follows the ordinary OpenAI fields and rejects
        # provider-specific extra_body values, so local calls use `language`.
        if not self.local and self.stt_model.startswith("gpt-transcribe"):
            kwargs["extra_body"] = context
        elif language:
            kwargs["language"] = language

        try:
            response = self.client.audio.transcriptions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - provider errors must be contained
            raise VoiceUnavailable(
                "Transcription failed. Check the configured speech service, "
                "model availability, and recording, then try again."
            ) from exc
        text = str(getattr(response, "text", "") or "").strip()
        if not text:
            raise VoiceUnavailable(
                "No speech was detected. Record a clear, short question and try again."
            )
        return Transcript(text=text, model=self.stt_model, bytes_received=len(payload))

    def synthesize(self, text: str, *, voice: str = DEFAULT_VOICE) -> Speech:
        narration = speakable_text(text)
        if not narration:
            raise VoiceUnavailable("There is no answer text to read aloud.")
        try:
            request = {
                "model": self.tts_model,
                "voice": voice,
                "input": narration,
                "response_format": "mp3",
            }
            if not self.local:
                request["instructions"] = (
                    "Speak like a calm senior data analyst briefing an executive. "
                    "Be clear and measured. Articulate numbers, percentages, "
                    "and acronyms precisely."
                )
            response = self.client.audio.speech.create(
                **request,
            )
            if hasattr(response, "read"):
                audio = response.read()
            else:
                audio = getattr(response, "content", b"")
        except Exception as exc:  # noqa: BLE001 - provider errors must be contained
            raise VoiceUnavailable(
                "Speech generation failed. Check the configured speech service and model, "
                "then try again."
            ) from exc
        audio = bytes(audio or b"")
        if not audio:
            raise VoiceUnavailable("The speech provider returned no audio. Try again.")
        return Speech(audio=audio, model=self.tts_model, voice=voice)


# ---------------------------------------------------------------------------
# Choosing an engine
# ---------------------------------------------------------------------------

ENV_ENGINE = "ASK_VOICE_ENGINE"          # local | remote | auto (default)
ENGINE_LOCAL = "local"
ENGINE_REMOTE = "remote"
ENGINE_AUTO = "auto"


def engine_preference(environ: dict[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    mode = str(env.get(ENV_ENGINE, ENGINE_AUTO) or ENGINE_AUTO).strip().lower()
    return mode if mode in (ENGINE_LOCAL, ENGINE_REMOTE, ENGINE_AUTO) else ENGINE_AUTO


def resolve(api_key: str = "", *, environ: dict[str, str] | None = None):
    """Return (client, label) for the best speech engine available, or (None, "").

    Precedence, and the reasoning for each step:

    1. An explicitly configured REMOTE endpoint or key wins. Someone who set
       `ASK_VOICE_BASE_URL` or pasted a key has told the app what they want,
       and silently preferring a local model over an operator's choice is the
       kind of helpfulness that reads as a bug.
    2. Otherwise the in-process open models, if their packages are installed.
       This is what makes voice reachable on the public deployment, where
       there is no second container to run and no key to spend.
    3. Otherwise nothing, and the interface says voice is unconfigured — the
       behaviour this app had before local speech existed.

    `ASK_VOICE_ENGINE` overrides the whole thing in either direction, because a
    deployment that wants to prove one path is running should not have to
    uninstall a package to do it.
    """
    env = os.environ if environ is None else environ
    preference = engine_preference(env)
    key = str(api_key or "").strip() or server_api_key(env)
    endpoint = base_url(env)
    remote_configured = bool(key or endpoint)

    def _local():
        # Imported here, never at module scope: engine/local_voice.py pulls in
        # optional packages, and this module has to keep working on a
        # deployment that could not install them.
        from engine import local_voice

        if not local_voice.available():
            return None, ""
        return local_voice.LocalVoice(), "local"

    if preference == ENGINE_LOCAL:
        return _local()
    if preference == ENGINE_REMOTE or remote_configured:
        if not remote_configured:
            return None, ""
        return OpenAIVoice(api_key=key, base_url=endpoint), (
            "self-hosted" if endpoint else "cloud"
        )
    return _local()


def describe_engine(label: str) -> str:
    """One line for the status rail, naming what is really running."""
    if label == "local":
        from engine import local_voice

        return (
            f"in-process · open models<br><em>{local_voice.STT_MODEL} · "
            f"{local_voice.TTS_VOICE} · no key</em>"
        )
    if label == "self-hosted":
        return "stt → review → tts<br><em>faster-whisper · Kokoro · self-hosted</em>"
    if label == "cloud":
        return "stt → review → tts<br><em>OpenAI · cloud</em>"
    return "unavailable<br><em>install the voice extras or configure an endpoint</em>"
