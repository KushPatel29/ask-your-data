"""Exercise real playback state transitions without a browser or a speech model."""

import ast
import hashlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from engine.voice import VoiceUnavailable


class Surface:
    def __init__(self):
        self.session_state = {}
        self.players = []
        self.clicked = False

    def spinner(self, *_args, **_kwargs):
        return nullcontext()

    def button(self, *_args, **_kwargs):
        clicked, self.clicked = self.clicked, False
        return clicked

    def audio(self, data, **kwargs):
        self.players.append((data, kwargs))

    def caption(self, *_args):
        pass

    def error(self, *_args):
        pass


class Speaker:
    tts_model = "piper en_US-joe-medium"

    def __init__(self):
        self.calls = 0
        self.fail = False

    def synthesize(self, text, **_kwargs):
        self.calls += 1
        if self.fail:
            raise VoiceUnavailable("Temporary speech failure")
        return SimpleNamespace(audio=b"RIFF" + text.encode(), mime_type="audio/wav")


def renderer():
    source = (Path(__file__).parents[1] / "app" / "streamlit_app.py").read_text("utf-8")
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_render_answer_audio")
    surface, speaker = Surface(), Speaker()
    namespace = {
        "st": surface, "hashlib": hashlib,
        "voice": SimpleNamespace(VoiceUnavailable=VoiceUnavailable,
                                 configured_tts_model=lambda: speaker.tts_model),
        "_voice_ready": lambda: True, "_voice_name": lambda: "en_US-joe-medium",
        "_voice_models": lambda: ("tiny.en", speaker.tts_model),
        "_voice_client": lambda: speaker, "_voice_label": lambda: "local",
        "_autospeak_on": lambda: surface.session_state.get("_autospeak_pref", True),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), source, "exec"), namespace)
    return namespace["_render_answer_audio"], surface, speaker


def test_repeat_answer_plays_for_new_turn_but_not_for_rerender():
    render, surface, speaker = renderer()
    render("There are 876 denied claims.", 0, autospeak=True)
    render("There are 876 denied claims.", 0, autospeak=True)
    render("There are 876 denied claims.", 1, autospeak=True)
    assert [options["autoplay"] for _, options in surface.players] == [True, False, True]
    assert speaker.calls == 1, "same speech bytes can be reused across turns"


def test_manual_listen_plays_even_when_automatic_speech_is_off():
    render, surface, _ = renderer()
    surface.session_state["_autospeak_pref"] = False
    surface.clicked = True
    render("There are 876 denied claims.", 0)
    assert surface.players[-1][1] == {"format": "audio/wav", "autoplay": True}


def test_audio_eviction_keeps_payload_and_mime_paired():
    render, surface, speaker = renderer()
    for index in range(5):
        render(f"Result {index}.", index, autospeak=True)
    keys = [key for key in surface.session_state
            if key.startswith("voice_audio_") and not key.startswith("voice_audio_mime_")]
    assert len(keys) == 3
    for key in keys:
        identity = key.removeprefix("voice_audio_")
        assert surface.session_state[f"voice_audio_mime_{identity}"] == "audio/wav"
    render("Result 4.", 4, autospeak=True)
    assert surface.players[-1][1]["format"] == "audio/wav"
    assert not surface.players[-1][1]["autoplay"]
    render("Result 0.", 0, autospeak=True)
    assert speaker.calls == 5, "an evicted old answer must not synthesize itself again"


def test_streamlit_unordered_state_never_evicts_the_clip_just_generated():
    class UnorderedState(dict):
        def __iter__(self):
            return iter(reversed(list(super().__iter__())))

    render, surface, speaker = renderer()
    surface.session_state = UnorderedState()
    for index in range(7):
        render(f"Result {index}.", index, autospeak=True)
        assert surface.players[-1][0] == b"RIFF" + f"Result {index}.".encode()
    assert speaker.calls == len(surface.players) == 7


def test_removing_a_recording_clears_its_transcript_before_confirming():
    source = (Path(__file__).parents[1] / "app" / "streamlit_app.py").read_text("utf-8")
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_voice_question")
    surface = Surface()
    surface.session_state.update(voice_last_digest="old", voice_draft="Old question",
                                 voice_meta="Old recording")
    surface.toggle = lambda *_args, **_kwargs: False
    surface.audio_input = lambda *_args, **_kwargs: None
    namespace = {
        "st": surface, "_voice_ready": lambda: True,
        "_voice_models": lambda: ("stt", "tts"), "_voice_label": lambda: "self-hosted",
        "_remember_autospeak": lambda: None, "_autospeak_on": lambda: False,
        "ui": SimpleNamespace(voice_dock=lambda **_kwargs: None),
    }
    surface.session_state["voice_autospeak"] = False
    exec(compile(ast.Module(body=[function], type_ignores=[]), source, "exec"), namespace)
    assert namespace["_voice_question"]() == ""
    assert "voice_draft" not in surface.session_state
    assert "voice_last_digest" not in surface.session_state


def test_failure_waits_for_explicit_retry_instead_of_retrying_each_rerun():
    render, surface, speaker = renderer()
    speaker.fail = True
    render("Result.", 0, autospeak=True)
    render("Result.", 0, autospeak=True)
    assert speaker.calls == 1
    speaker.fail = False
    surface.clicked = True
    render("Result.", 0, autospeak=True)
    assert speaker.calls == 2
    assert surface.players[-1][1]["autoplay"]
    assert not any(key.startswith("voice_audio_error_") for key in surface.session_state)
