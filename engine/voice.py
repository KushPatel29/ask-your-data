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
        api_key: str,
        *,
        client=None,
        stt_model: str = STT_MODEL,
        tts_model: str = TTS_MODEL,
    ) -> None:
        if not str(api_key or "").strip() and client is None:
            raise VoiceUnavailable("Add an OpenAI API key to enable voice.")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on installation
                raise VoiceUnavailable(
                    "Voice needs the optional OpenAI SDK. Install the project requirements."
                ) from exc
            client = OpenAI(api_key=api_key)
        self.client = client
        self.stt_model = stt_model
        self.tts_model = tts_model

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
        # gpt-transcribe accepts richer context in the request body. Keeping it
        # in extra_body also leaves the SDK version independent of those fields.
        if self.stt_model.startswith("gpt-transcribe"):
            kwargs["extra_body"] = context
        elif language:
            # Older transcription models use one language hint.
            kwargs["language"] = language

        try:
            response = self.client.audio.transcriptions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - provider errors must be contained
            raise VoiceUnavailable(
                "Transcription failed. Check the OpenAI key, account access, "
                "and recording, then try again."
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
            response = self.client.audio.speech.create(
                model=self.tts_model,
                voice=voice,
                input=narration,
                instructions=(
                    "Speak like a calm senior data analyst briefing an executive. "
                    "Be clear and measured. Articulate numbers, percentages, "
                    "and acronyms precisely."
                ),
                response_format="mp3",
            )
            if hasattr(response, "read"):
                audio = response.read()
            else:
                audio = getattr(response, "content", b"")
        except Exception as exc:  # noqa: BLE001 - provider errors must be contained
            raise VoiceUnavailable(
                "Speech generation failed. Check the OpenAI key and account access, then try again."
            ) from exc
        audio = bytes(audio or b"")
        if not audio:
            raise VoiceUnavailable("The speech provider returned no audio. Try again.")
        return Speech(audio=audio, model=self.tts_model, voice=voice)
