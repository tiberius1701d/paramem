"""Unit tests for POST /voice.

CPU-only — no real model, no ffmpeg, no STT model loaded.

Covers:
- 404 when mobile_pwa.enabled is False.
- 503 when STT is not loaded (stt=None or stt.is_loaded=False).
- 401 without a bearer token (auth middleware active, tested via minimal app).
- Happy path: decode → process_utterance → handle_chat → {transcript, reply}.
- Empty transcript → returns {"transcript": "", "reply": ""} without calling handle_chat.
- 400 on ffmpeg decode failure.
- TTS happy path: tts_manager.synthesize returns PCM → response contains a non-empty
  base64 ``audio`` field that decodes to a valid WAV (RIFF header), ``audio_format="wav"``.
- TTS unavailable (tts_manager=None) → ``audio=""`` but transcript+reply present.
- TTS raises → text-only graceful fallback (``audio=""``).

All heavy callables are mocked:
- ``_decode_audio_to_pcm`` — avoid needing a real ffmpeg subprocess.
- ``process_utterance``    — avoid needing a real STT model or GPU.
- ``handle_chat``          — avoid needing a real LLM.

Auth note: ``app_module.app`` has ``BearerTokenMiddleware`` wired at module-load
time against ``_state["user_token_store"]`` — ``None`` in these tests (no store
configured) means auth OFF.  Tests that exercise the 404/503/400 gates do NOT
need auth; the 401 test builds a separate minimal app with a real
``UserTokenStore`` (matching the pattern in ``tests/server/test_auth.py``).
"""

from __future__ import annotations

import base64
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
)
from paramem.server.auth import BearerTokenMiddleware
from paramem.server.session_buffer import SessionBuffer
from paramem.server.user_tokens import UserTokenStore
from paramem.server.voice_pipeline import UtteranceResult

# ---------------------------------------------------------------------------
# State factory
# ---------------------------------------------------------------------------


def _make_config(pwa_enabled: bool = True) -> MagicMock:
    cfg = MagicMock()
    cfg.mobile_pwa.enabled = pwa_enabled
    cfg.mobile_pwa.cookie_name = "paramem_token"
    cfg.debug = False
    cfg.consolidation.abort_quiesce_timeout_s = 5.0
    # _resolve_and_enroll_speaker reads these for language resolution and
    # greeting; configure them as concrete values so comparisons don't fail.
    cfg.tts.language_confidence_threshold = 0.85
    cfg.voice.greeting_interval_hours = 0  # disable greeting in unit tests
    return cfg


def _make_stt(loaded: bool = True) -> MagicMock:
    stt = MagicMock()
    stt.is_loaded = loaded
    return stt


def _make_speaker_store(
    *,
    known_ids: dict[str, str | None] | None = None,
    match_speaker_id: str | None = None,
    match_tentative: bool = True,
    anon_id: str = "speaker3",
    is_anonymous: bool = True,
) -> MagicMock:
    """Return a minimal SpeakerStore mock for voice-enrollment tests.

    Parameters
    ----------
    known_ids:
        Maps speaker_id → display_name for ``get_name`` and
        ``resolve_speaker_name``.  Both methods are wired to the same lookup
        so callers can use either without distinguishing which internal
        path reached them.
    match_speaker_id:
        The speaker_id ``match()`` returns (``None`` = no match).
    match_tentative:
        Whether the ``match()`` result is tentative (tentative → not used).
    anon_id:
        The value returned by ``register_anonymous``.
    is_anonymous:
        Return value of ``is_anonymous()``.
    """
    store = MagicMock()

    # get_name
    def _get_name(sid):
        if known_ids is None:
            return None
        return (known_ids or {}).get(sid)

    store.get_name.side_effect = _get_name
    # resolve_speaker_name is now called by _resolve_speaker for the
    # auth-speaker-id path.  Wire it to the same lookup so tests that
    # construct stores via this helper work regardless of which internal
    # method is invoked.
    store.resolve_speaker_name.side_effect = _get_name

    # match
    match_result = MagicMock()
    match_result.speaker_id = match_speaker_id
    match_result.name = (known_ids or {}).get(match_speaker_id) if match_speaker_id else None
    match_result.tentative = match_tentative
    store.match.return_value = match_result

    # register_anonymous
    store.register_anonymous.return_value = anon_id

    # is_anonymous
    store.is_anonymous.return_value = is_anonymous

    # get_preferred_language
    store.get_preferred_language.return_value = None

    return store


def _make_state(
    tmp_path,
    pwa_enabled: bool = True,
    stt_loaded: bool = True,
    mode: str = "local",
    speaker_store=None,
) -> dict:
    cfg = _make_config(pwa_enabled)
    sessions = tmp_path / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    buffer = SessionBuffer(session_dir=sessions, state_dir=sessions.parent / "state", debug=False)

    return {
        "config": cfg,
        "stt": _make_stt(stt_loaded),
        "session_buffer": buffer,
        "speaker_store": speaker_store,
        "model": MagicMock(),
        "tokenizer": MagicMock(),
        "router": MagicMock(),
        "memory_store": MagicMock(),
        "cloud_agent": None,
        "ha_client": None,
        "background_trainer": None,
        "consolidation_loop": None,
        "mode": mode,
        "effective_mode": None,
        "latest_embedding": None,
        "latest_language_detection": None,
        "unknown_speakers": {},
        "pending_enrollments": set(),
        "user_token_store": None,
        # Required by _run_chat_turn's relay leg (_notice_once) whenever
        # serving is RELAY — which now includes local-mode requests with no
        # resolved speaker, not just server-wide cloud-only mode.
        "relay_notice_conversations": set(),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def state(tmp_path, monkeypatch):
    """Default state: PWA enabled, STT loaded, local mode."""
    fresh = _make_state(tmp_path)
    monkeypatch.setattr(app_module, "_state", fresh)
    return fresh


@pytest.fixture()
def client(state):
    """TestClient against app_module.app.

    ``app_module.app`` has ``BearerTokenMiddleware`` wired at module-load
    time against ``_state["user_token_store"]``.  ``state`` (this fixture's
    dependency) sets ``user_token_store: None``, so auth is OFF and no
    header is needed.
    """
    return TestClient(app_module.app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Helper — post voice
# ---------------------------------------------------------------------------


def _post_voice(
    client: TestClient,
    audio: bytes = b"\x00\x00" * 100,
    content_type: str = "audio/webm",
    headers: dict | None = None,
) -> dict:
    _headers = {"content-type": content_type}
    if headers:
        _headers.update(headers)
    return client.post("/voice", content=audio, headers=_headers)


def _resolved_speaker_patch(speaker_id: str = "speaker0", speaker: str = "Alex"):
    """Patch ``_resolve_and_enroll_speaker`` to return a resolved
    (``serving=PERSONAL``) speaker.

    Several ``/voice`` tests are about LOCAL-mode dispatch specifics
    (training abort, TTS synthesis) that require ``handle_chat`` to be
    reached — which now only happens for a resolved speaker (a speakerless
    caller is served by the relay path instead, see
    ``test_voice_relay_route``).  The default ``state``/``client`` fixtures
    carry no speaker_store and the mocked STT result carries no embedding,
    so nothing resolves a speaker without this patch.
    """
    from paramem.server.app import ResolvedSpeaker

    return patch(
        "paramem.server.app._resolve_and_enroll_speaker",
        new=AsyncMock(
            return_value=ResolvedSpeaker(
                speaker_id=speaker_id,
                speaker=speaker,
                follow_up=None,
                greeting_prefix=None,
                effective_language="en",
            )
        ),
    )


# ---------------------------------------------------------------------------
# Tests: gating
# ---------------------------------------------------------------------------


def _authed_client(monkeypatch, tmp_path, pwa_enabled=True, stt_loaded=True):
    """Helper: build a TestClient against app_module.app.

    Auth is OFF: ``_make_state`` sets ``user_token_store: None`` and no
    header is needed.
    """
    fresh = _make_state(tmp_path, pwa_enabled=pwa_enabled, stt_loaded=stt_loaded)
    monkeypatch.setattr(app_module, "_state", fresh)
    return TestClient(app_module.app, raise_server_exceptions=True), fresh


def test_voice_404_when_pwa_disabled(tmp_path, monkeypatch):
    """POST /voice returns 404 when mobile_pwa.enabled is False."""
    c, _ = _authed_client(monkeypatch, tmp_path, pwa_enabled=False)
    resp = c.post("/voice", content=b"data", headers={"content-type": "audio/webm"})
    assert resp.status_code == 404


def test_voice_503_when_stt_not_loaded(tmp_path, monkeypatch):
    """POST /voice returns 503 when stt.is_loaded is False."""
    c, _ = _authed_client(monkeypatch, tmp_path, stt_loaded=False)
    resp = c.post("/voice", content=b"data", headers={"content-type": "audio/webm"})
    assert resp.status_code == 503
    assert resp.json()["error"] == "stt_unavailable"


def test_voice_503_when_stt_none(tmp_path, monkeypatch):
    """POST /voice returns 503 when _state['stt'] is None."""
    c, fresh = _authed_client(monkeypatch, tmp_path)
    fresh["stt"] = None
    resp = c.post("/voice", content=b"data", headers={"content-type": "audio/webm"})
    assert resp.status_code == 503


def test_voice_401_without_token(tmp_path):
    """/voice returns 401 when a bearer token is required but not sent.

    Uses a minimal app with BearerTokenMiddleware wired against a real
    UserTokenStore with an active token — same pattern as test_auth.py —
    rather than trying to mutate the module-level middleware instance in
    app_module.app.
    """
    store = UserTokenStore(tmp_path / "user_tokens.json")
    store.mint("speaker0", "Device", scope="chat")

    mini = FastAPI()
    mini.add_middleware(BearerTokenMiddleware, user_token_getter=lambda: store)

    @mini.post("/voice")
    async def _stub_voice(req: Request) -> dict:
        return {"transcript": "", "reply": ""}

    c = TestClient(mini, raise_server_exceptions=True)
    resp = c.post("/voice", content=b"data", headers={"content-type": "audio/webm"})
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------


def test_voice_happy_path(client, state):
    """decode → process_utterance → handle_chat → {transcript, reply}."""
    fake_pcm = b"\x00\x00" * 800
    fake_result = UtteranceResult(text="what time is it", language="en")
    fake_chat = MagicMock()
    fake_chat.text = "It is noon."
    fake_chat.escalated = False

    with (
        _resolved_speaker_patch(),
        patch(
            "paramem.server.app._decode_audio_to_pcm",
            return_value=fake_pcm,
        ),
        patch(
            "paramem.server.app.process_utterance",
            new=AsyncMock(return_value=fake_result),
        ),
        patch(
            "paramem.server.app.handle_chat",
            return_value=fake_chat,
        ),
    ):
        resp = _post_voice(client, audio=b"fake-container-audio")

    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == "what time is it"
    assert body["reply"] == "It is noon."


def test_voice_empty_transcript_skips_chat(client, state):
    """Empty transcript → return {"transcript": "", "reply": ""} without handle_chat."""
    fake_pcm = b"\x00\x00" * 400
    empty_result = UtteranceResult(text="")

    with (
        patch(
            "paramem.server.app._decode_audio_to_pcm",
            return_value=fake_pcm,
        ),
        patch(
            "paramem.server.app.process_utterance",
            new=AsyncMock(return_value=empty_result),
        ),
        patch(
            "paramem.server.app.handle_chat",
        ) as mock_chat,
    ):
        resp = _post_voice(client, audio=b"fake-silent-audio")

    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == ""
    assert body["reply"] == ""
    mock_chat.assert_not_called()


def test_voice_empty_body_returns_empty(client, state):
    """Empty request body → {"transcript": "", "reply": ""} without decode."""
    with (
        patch("paramem.server.app._decode_audio_to_pcm") as mock_decode,
        patch("paramem.server.app.process_utterance") as mock_proc,
    ):
        resp = _post_voice(client, audio=b"")

    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == ""
    assert body["reply"] == ""
    mock_decode.assert_not_called()
    mock_proc.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: decode failure
# ---------------------------------------------------------------------------


def test_voice_400_on_decode_failure(client, state):
    """ffmpeg decode failure → HTTP 400 with error=audio_decode_failed."""
    with patch(
        "paramem.server.app._decode_audio_to_pcm",
        side_effect=RuntimeError("ffmpeg produced empty output"),
    ):
        resp = _post_voice(client, audio=b"garbage")

    assert resp.status_code == 400
    assert resp.json()["error"] == "audio_decode_failed"


# ---------------------------------------------------------------------------
# Tests: body-size cap
# ---------------------------------------------------------------------------


def test_voice_413_when_content_length_exceeds_cap(client, state):
    """Content-Length header > 25 MB → HTTP 413 before body is read."""
    large_content_length = str(26 * 1024 * 1024)  # 26 MB
    resp = _post_voice(
        client,
        audio=b"\x00\x00" * 100,  # small body — gate fires on header, not body
        headers={"content-length": large_content_length},
    )
    assert resp.status_code == 413
    assert resp.json()["error"] == "audio_too_large"


def test_voice_413_when_body_exceeds_cap_no_header(client, state, monkeypatch):
    """Body exceeding 25 MB without Content-Length → HTTP 413."""
    oversized = b"\x00" * (26 * 1024 * 1024)

    with patch("paramem.server.app._decode_audio_to_pcm") as mock_decode:
        resp = _post_voice(client, audio=oversized)

    assert resp.status_code == 413
    assert resp.json()["error"] == "audio_too_large"
    mock_decode.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: cloud-only branch
# ---------------------------------------------------------------------------


def test_voice_relay_route(tmp_path, monkeypatch):
    """POST /voice in cloud-only mode routes through _relay_route."""
    fresh = _make_state(tmp_path, mode="cloud-only")
    monkeypatch.setattr(app_module, "_state", fresh)

    headers = {}

    fake_pcm = b"\x00\x00" * 800
    fake_result = UtteranceResult(text="turn off the lights", language="en")
    fake_cloud_result = MagicMock()
    fake_cloud_result.text = "Lights turned off."
    fake_cloud_result.escalated = True

    with (
        patch("paramem.server.app._decode_audio_to_pcm", return_value=fake_pcm),
        patch("paramem.server.app.process_utterance", new=AsyncMock(return_value=fake_result)),
        patch("paramem.server.app._relay_route", return_value=fake_cloud_result) as mock_cloud,
        patch("paramem.server.app.handle_chat") as mock_local,
    ):
        from fastapi.testclient import TestClient

        tc = TestClient(
            app_module.app,
            raise_server_exceptions=True,
            headers=headers,
        )
        resp = tc.post(
            "/voice",
            content=b"fake-audio",
            headers={"content-type": "audio/webm"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == "turn off the lights"
    # No speaker resolves for this call (no store, no embedding match) —
    # serving is RELAY, so the once-per-conversation speakerless notice
    # prepends the answer, same mechanism as the degraded-serving notice.
    assert body["reply"] == f"{app_module._SPEAKERLESS_RELAY_NOTICE}Lights turned off."
    mock_cloud.assert_called_once()
    mock_local.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: training-abort branch
# ---------------------------------------------------------------------------


def test_voice_aborts_training_before_inference(client, state):
    """POST /voice calls abort_for_inference when bg_trainer.is_training is True."""
    fake_pcm = b"\x00\x00" * 800
    fake_result = UtteranceResult(text="hello", language="en")
    fake_chat = MagicMock()
    fake_chat.text = "Hi there."
    fake_chat.escalated = False

    bg_trainer = MagicMock()
    bg_trainer.is_training = True
    bg_trainer.abort_for_inference.return_value = True
    state["background_trainer"] = bg_trainer

    with (
        _resolved_speaker_patch(),
        patch("paramem.server.app._decode_audio_to_pcm", return_value=fake_pcm),
        patch("paramem.server.app.process_utterance", new=AsyncMock(return_value=fake_result)),
        patch("paramem.server.app.handle_chat", return_value=fake_chat),
    ):
        resp = _post_voice(client, audio=b"fake-audio")

    assert resp.status_code == 200
    bg_trainer.abort_for_inference.assert_called_once()


def test_voice_forces_trainer_stop_when_abort_times_out(client, state):
    """When abort_for_inference returns False, trainer is force-stopped."""
    fake_pcm = b"\x00\x00" * 800
    fake_result = UtteranceResult(text="hello", language="en")
    fake_chat = MagicMock()
    fake_chat.text = "Hi."
    fake_chat.escalated = False

    bg_trainer = MagicMock()
    bg_trainer.is_training = True
    bg_trainer.abort_for_inference.return_value = False
    state["background_trainer"] = bg_trainer

    with (
        _resolved_speaker_patch(),
        patch("paramem.server.app._decode_audio_to_pcm", return_value=fake_pcm),
        patch("paramem.server.app.process_utterance", new=AsyncMock(return_value=fake_result)),
        patch("paramem.server.app.handle_chat", return_value=fake_chat),
    ):
        resp = _post_voice(client, audio=b"fake-audio")

    assert resp.status_code == 200
    assert bg_trainer._shutdown_requested is True
    assert bg_trainer._is_training is False


# ---------------------------------------------------------------------------
# Tests: TTS synthesis in /voice response
# ---------------------------------------------------------------------------


def _make_tts_manager(loaded: bool = True, pcm: bytes = b"\x00\x01" * 200, sr: int = 22050):
    """Return a MagicMock TTSManager whose synthesize() returns known PCM."""
    mgr = MagicMock()
    mgr.is_loaded = loaded
    mgr.synthesize.return_value = (pcm, sr)
    return mgr


def _is_valid_wav(wav_bytes: bytes) -> bool:
    """Return True when wav_bytes starts with a well-formed RIFF/WAVE header."""
    if len(wav_bytes) < 44:
        return False
    riff, chunk_size, wave = struct.unpack_from("<4sI4s", wav_bytes, 0)
    fmt_marker, fmt_size, audio_fmt, channels, sr = struct.unpack_from("<4sIHHI", wav_bytes, 12)
    data_marker = wav_bytes[36:40]
    return (
        riff == b"RIFF"
        and wave == b"WAVE"
        and fmt_marker == b"fmt "
        and fmt_size == 16
        and audio_fmt == 1  # PCM
        and channels == 1
        and data_marker == b"data"
    )


def test_voice_tts_happy_path(client, state):
    """tts_manager.synthesize returns PCM → audio field is base64 WAV, format="wav".

    Speaker resolution is stubbed to PERSONAL (a resolved speaker_id is now
    required to reach ``handle_chat``/the local dispatch this test targets
    — a speakerless caller is served by the relay path instead) with
    ``effective_language="en"`` forwarded to TTS.
    """
    fake_pcm_input = b"\x00\x00" * 800
    fake_result = UtteranceResult(text="hello", language="en", language_probability=0.95)
    fake_chat = MagicMock()
    fake_chat.text = "Hi there."
    fake_chat.escalated = False

    synth_pcm = b"\x10\x20" * 400
    synth_sr = 22050
    tts_mgr = _make_tts_manager(loaded=True, pcm=synth_pcm, sr=synth_sr)
    state["tts_manager"] = tts_mgr

    with (
        _resolved_speaker_patch(),
        patch("paramem.server.app._decode_audio_to_pcm", return_value=fake_pcm_input),
        patch("paramem.server.app.process_utterance", new=AsyncMock(return_value=fake_result)),
        patch("paramem.server.app.handle_chat", return_value=fake_chat),
    ):
        resp = _post_voice(client, audio=b"fake-audio")

    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == "hello"
    assert body["reply"] == "Hi there."
    assert body["audio_format"] == "wav"
    assert body["audio"], "audio field must be non-empty"

    # Verify the base64 payload decodes to a valid WAV with the expected PCM tail.
    wav_bytes = base64.b64decode(body["audio"])
    assert _is_valid_wav(wav_bytes), "decoded audio must start with a valid RIFF/WAV header"
    assert wav_bytes[44:] == synth_pcm, "WAV body must equal the PCM returned by synthesize()"

    # Confirm the manager was called with the resolved effective language.
    tts_mgr.synthesize.assert_called_once_with("Hi there.", "en")


def test_voice_tts_unavailable_returns_text_only(tmp_path, monkeypatch):
    """tts_manager=None → audio="" but transcript and reply are present."""
    fresh = _make_state(tmp_path)
    fresh["tts_manager"] = None
    monkeypatch.setattr(app_module, "_state", fresh)

    tc = TestClient(
        app_module.app,
        raise_server_exceptions=True,
        headers={},
    )

    fake_pcm = b"\x00\x00" * 800
    fake_result = UtteranceResult(text="good morning", language="en")
    fake_chat = MagicMock()
    fake_chat.text = "Good morning!"
    fake_chat.escalated = False

    with (
        _resolved_speaker_patch(),
        patch("paramem.server.app._decode_audio_to_pcm", return_value=fake_pcm),
        patch("paramem.server.app.process_utterance", new=AsyncMock(return_value=fake_result)),
        patch("paramem.server.app.handle_chat", return_value=fake_chat),
    ):
        resp = tc.post("/voice", content=b"fake-audio", headers={"content-type": "audio/webm"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == "good morning"
    assert body["reply"] == "Good morning!"
    assert body["audio"] == ""
    assert body["audio_format"] == ""


def test_voice_tts_raises_returns_text_only(client, state):
    """TTS synthesis raises an exception → audio="" graceful fallback, no 500."""
    fake_pcm = b"\x00\x00" * 800
    fake_result = UtteranceResult(text="test query", language="en")
    fake_chat = MagicMock()
    fake_chat.text = "Test reply."
    fake_chat.escalated = False

    tts_mgr = MagicMock()
    tts_mgr.is_loaded = True
    tts_mgr.synthesize.side_effect = RuntimeError("synthesis device error")
    state["tts_manager"] = tts_mgr

    with (
        _resolved_speaker_patch(),
        patch("paramem.server.app._decode_audio_to_pcm", return_value=fake_pcm),
        patch("paramem.server.app.process_utterance", new=AsyncMock(return_value=fake_result)),
        patch("paramem.server.app.handle_chat", return_value=fake_chat),
    ):
        resp = _post_voice(client, audio=b"fake-audio")

    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == "test query"
    assert body["reply"] == "Test reply."
    assert body["audio"] == ""
    assert body["audio_format"] == ""


# ---------------------------------------------------------------------------
# Tests: shared-device PWA voice identity
# ---------------------------------------------------------------------------
# All tests in this section cover the token-type selector introduced in
# _resolve_and_enroll_speaker:
#   per-user token (auth_speaker_id set) → skip embedding, token identity
#   shared token (auth_speaker_id None) → compute embedding, run enrollment
# ---------------------------------------------------------------------------


def test_voice_personal_token_skips_embedding(tmp_path, monkeypatch):
    """Per-user token path: process_utterance called with compute_embedding=False.

    The token-type selector sets compute_embedding = (auth_speaker_id is None).
    When auth_speaker_id is set (per-user token), compute_embedding is False.

    The speaker_id is injected via real BearerTokenMiddleware + UserTokenStore:
    a per-user token minted for speaker0 is sent in the Authorization header so
    auth.py:180 sets scope["state"]["speaker_id"] = "speaker0" — the same real
    code path used in production.
    """
    # Set up the daily identity so UserTokenStore.mint() can sign the token.
    ident = mint_daily_identity()
    key_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(ident, "pw"), key_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
    _clear_daily_identity_cache()

    user_store = UserTokenStore(tmp_path / "user_tokens.json")
    per_user_token = user_store.mint("speaker0", "TestDevice")

    # Wire the store into _state so the middleware's lambda resolves it.
    fresh = _make_state(tmp_path)
    fresh["user_token_store"] = user_store
    monkeypatch.setattr(app_module, "_state", fresh)

    # app_module.app wires BearerTokenMiddleware with user_token_getter pointing at
    # _state["user_token_store"].  Presenting the per-user token causes auth.py:180
    # to set scope["state"]["speaker_id"] = "speaker0" on the real request scope.
    tc = TestClient(app_module.app, raise_server_exceptions=True)

    fake_pcm = b"\x00\x00" * 800
    fake_utterance = UtteranceResult(text="hello", language="en")
    fake_chat_result = MagicMock()
    fake_chat_result.text = "Hi."
    fake_chat_result.escalated = False

    captured = {}

    async def mock_process_utterance(*args, **kwargs):
        captured["compute_embedding"] = kwargs.get("compute_embedding")
        return fake_utterance

    from paramem.server.app import ResolvedSpeaker

    resolved = ResolvedSpeaker(
        speaker_id="speaker0",
        speaker="Alice",
        follow_up=None,
        greeting_prefix=None,
        effective_language="en",
    )

    with (
        patch("paramem.server.app._decode_audio_to_pcm", return_value=fake_pcm),
        patch("paramem.server.app.process_utterance", new=mock_process_utterance),
        patch(
            "paramem.server.app._resolve_and_enroll_speaker",
            new=AsyncMock(return_value=resolved),
        ),
        patch(
            "paramem.server.app._run_chat_turn",
            new=AsyncMock(return_value=(fake_chat_result, "Hi.")),
        ),
    ):
        resp = tc.post(
            "/voice",
            content=b"audio-data",
            headers={
                "content-type": "audio/webm",
                "x-conversation-id": "my-conv",
                "Authorization": f"Bearer {per_user_token}",
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "Hi."
    # Per-user token → compute_embedding=False
    assert captured.get("compute_embedding") is False
    assert body.get("follow_up") is None


def test_voice_shared_token_computes_embedding(tmp_path, monkeypatch):
    """Shared token path: process_utterance called with compute_embedding=True.

    Known-voice embedding match → speaker resolved, no follow_up.
    auth_speaker_id is None (no per-user token in test environment).
    """
    store = _make_speaker_store(
        known_ids={"speaker0": "Alice"},
        match_speaker_id="speaker0",
        match_tentative=False,  # non-tentative → matched identity
        is_anonymous=False,
    )
    fresh = _make_state(tmp_path, speaker_store=store)
    monkeypatch.setattr(app_module, "_state", fresh)
    tc = TestClient(
        app_module.app,
        raise_server_exceptions=True,
        headers={},
    )

    fake_pcm = b"\x00\x00" * 800
    # Embedding provided — store will match it
    fake_utterance = UtteranceResult(
        text="hello", language="en", language_probability=0.95, embedding=[0.1, 0.2]
    )
    fake_chat_result = MagicMock()
    fake_chat_result.text = "Hello Alice."
    fake_chat_result.escalated = False

    captured = {}

    async def mock_process_utterance(*args, **kwargs):
        captured["compute_embedding"] = kwargs.get("compute_embedding")
        return fake_utterance

    with (
        patch("paramem.server.app._decode_audio_to_pcm", return_value=fake_pcm),
        patch("paramem.server.app.process_utterance", new=mock_process_utterance),
        patch(
            "paramem.server.app._run_chat_turn",
            new=AsyncMock(return_value=(fake_chat_result, "Hello Alice.")),
        ),
    ):
        resp = tc.post(
            "/voice",
            content=b"audio-data",
            headers={"content-type": "audio/webm"},
        )

    assert resp.status_code == 200
    body = resp.json()
    # auth_speaker_id is None (no per-user token) → compute_embedding=True
    assert captured["compute_embedding"] is True
    assert body["reply"] == "Hello Alice."
    assert body.get("follow_up") is None


def test_voice_shared_token_unknown_voice_enrolls(tmp_path, monkeypatch):
    """Shared token + unknown voice → register_anonymous called, follow_up returned."""
    store = _make_speaker_store(
        anon_id="speaker3",
        is_anonymous=True,
    )
    # Default: no match (match_speaker_id=None, match_tentative=True)
    fresh = _make_state(tmp_path, speaker_store=store)
    monkeypatch.setattr(app_module, "_state", fresh)
    fresh["config"].speaker.enrollment_prompt = "What's your name?"
    fresh["config"].speaker.enrollment_reprompt_interval = 3600
    tc = TestClient(
        app_module.app,
        raise_server_exceptions=True,
        headers={},
    )

    fake_pcm = b"\x00\x00" * 800
    fake_utterance = UtteranceResult(
        text="what is the weather",
        language="en",
        language_probability=0.9,
        embedding=[0.5, 0.5],
    )
    fake_chat_result = MagicMock()
    fake_chat_result.text = "It is sunny."
    fake_chat_result.escalated = False

    with (
        patch("paramem.server.app._decode_audio_to_pcm", return_value=fake_pcm),
        patch(
            "paramem.server.app.process_utterance",
            new=AsyncMock(return_value=fake_utterance),
        ),
        patch(
            "paramem.server.app._run_chat_turn",
            new=AsyncMock(return_value=(fake_chat_result, "It is sunny.")),
        ),
        patch(
            "paramem.server.app._run_enrollment_for_speaker",
            new=AsyncMock(return_value=None),  # name not yet disclosed
        ),
    ):
        resp = tc.post(
            "/voice",
            content=b"audio-data",
            headers={"content-type": "audio/webm"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "It is sunny."
    # Unknown voice → promoted to canonical anonymous ID
    store.register_anonymous.assert_called_once()
    # Enrollment prompt surfaced as follow_up
    assert body["follow_up"] == "What's your name?"


def test_voice_name_disclosure_binds(tmp_path, monkeypatch):
    """_run_enrollment_for_speaker returns a name → follow_up cleared to None."""
    store = _make_speaker_store(
        anon_id="speaker3",
        is_anonymous=True,
    )
    fresh = _make_state(tmp_path, speaker_store=store)
    monkeypatch.setattr(app_module, "_state", fresh)
    fresh["config"].speaker.enrollment_prompt = "What's your name?"
    fresh["config"].speaker.enrollment_reprompt_interval = 3600
    tc = TestClient(
        app_module.app,
        raise_server_exceptions=True,
        headers={},
    )

    fake_pcm = b"\x00\x00" * 800
    fake_utterance = UtteranceResult(
        text="My name is Alex",
        language="en",
        language_probability=0.9,
        embedding=[0.1, 0.2],
    )
    fake_chat_result = MagicMock()
    fake_chat_result.text = "Nice to meet you, Alex."
    fake_chat_result.escalated = False

    with (
        patch("paramem.server.app._decode_audio_to_pcm", return_value=fake_pcm),
        patch(
            "paramem.server.app.process_utterance",
            new=AsyncMock(return_value=fake_utterance),
        ),
        patch(
            "paramem.server.app._run_chat_turn",
            new=AsyncMock(return_value=(fake_chat_result, "Nice to meet you, Alex.")),
        ),
        patch(
            "paramem.server.app._run_enrollment_for_speaker",
            new=AsyncMock(return_value="Alex"),  # name disclosed
        ),
    ):
        resp = tc.post(
            "/voice",
            content=b"audio-data",
            headers={"content-type": "audio/webm"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "Nice to meet you, Alex."
    # Disclosure → follow_up cleared (no enrollment prompt needed)
    assert body.get("follow_up") is None


def test_run_enrollment_turns_chronological_order(tmp_path, monkeypatch):
    """Regression: prior session turns come BEFORE the live turn.

    ``all_turns`` previously prepended ``extra_turns`` (the live turn) ahead
    of ``buffer.get_session_turns()``'s output, inverting chronological order
    for the LLM name-extractor.  Pins: buffer turns first, the live turn
    last.
    """
    import asyncio

    fresh = _make_state(tmp_path)
    fresh["speaker_store"] = MagicMock()
    fresh["model"] = MagicMock()
    fresh["tokenizer"] = MagicMock()
    buffer = MagicMock()
    prior_turns = [
        {"role": "user", "text": "Hi there"},
        {"role": "assistant", "text": "Hello! How can I help?"},
    ]
    buffer.get_session_turns.return_value = list(prior_turns)
    fresh["session_buffer"] = buffer
    monkeypatch.setattr(app_module, "_state", fresh)

    captured = {}

    def fake_extract(turns, model, tokenizer):
        captured["turns"] = list(turns)
        return None, "raw"

    monkeypatch.setattr(app_module, "extract_name_via_llm", fake_extract)

    live_turn = {"role": "user", "text": "My name is Priya"}
    result = asyncio.run(
        app_module._run_enrollment_for_speaker(
            "speaker3",
            "conv-order-test",
            [0.1, 0.2],
            extra_turns=[live_turn],
        )
    )

    assert result is None  # extractor returned None → no side effects
    assert captured["turns"] == prior_turns + [live_turn]


def test_voice_shared_token_fresh_conversation_id(tmp_path, monkeypatch):
    """Two shared-token /voice requests get distinct conversation_ids.

    Each push-to-talk press is a separate POST /voice, and because auth_speaker_id
    is None (shared token), each gets a fresh voice-* id.  The retro-claim
    across conversation_ids preserves two-turn enrollment.
    """
    fresh = _make_state(tmp_path)
    monkeypatch.setattr(app_module, "_state", fresh)
    tc = TestClient(
        app_module.app,
        raise_server_exceptions=True,
        headers={},
    )

    fake_pcm = b"\x00\x00" * 800
    fake_chat_result = MagicMock()
    fake_chat_result.text = "Reply."
    fake_chat_result.escalated = False

    captured_conv_ids: list[str] = []

    async def mock_turn(**kwargs):
        captured_conv_ids.append(kwargs["conversation_id"])
        return (fake_chat_result, "Reply.")

    fake_utterance = UtteranceResult(text="hello", language="en")

    with (
        patch("paramem.server.app._decode_audio_to_pcm", return_value=fake_pcm),
        patch(
            "paramem.server.app.process_utterance",
            new=AsyncMock(return_value=fake_utterance),
        ),
        patch("paramem.server.app._run_chat_turn", new=mock_turn),
    ):
        tc.post("/voice", content=b"a1", headers={"content-type": "audio/webm"})
        tc.post("/voice", content=b"a2", headers={"content-type": "audio/webm"})

    assert len(captured_conv_ids) == 2
    assert captured_conv_ids[0] != captured_conv_ids[1], (
        "Shared-token /voice requests must get distinct conversation_ids"
    )
    assert all(cid.startswith("voice-") for cid in captured_conv_ids)


def test_voice_shared_token_resolved_speaker_does_not_leak_open_entries(tmp_path, monkeypatch):
    """Many shared-token utterances from the SAME resolved (known) speaker
    must not accumulate one _open entry per utterance.

    Regression for the orphan-leak: _resolve_and_enroll_speaker calls
    set_speaker(transport_id, ...) internally (correct for its own
    unknown-speaker grouping), but the buffer conversation_key
    ("voice-{speaker_id}") is a DIFFERENT key. Without reconciliation the
    transport-id _open entry never gets a session_id, so retirement-based
    pruning can never evict it — one leaked entry per push-to-talk press.
    """
    store = _make_speaker_store(
        known_ids={"speaker0": "Alice"},
        match_speaker_id="speaker0",
        match_tentative=False,  # non-tentative → matched identity every time
        is_anonymous=False,
    )
    fresh = _make_state(tmp_path, speaker_store=store)
    monkeypatch.setattr(app_module, "_state", fresh)
    buffer: SessionBuffer = fresh["session_buffer"]
    tc = TestClient(
        app_module.app,
        raise_server_exceptions=True,
        headers={},
    )

    fake_chat_result = MagicMock()
    fake_chat_result.text = "Reply."
    fake_chat_result.escalated = False

    async def mock_turn(**kwargs):
        return (fake_chat_result, "Reply.")

    fake_pcm = b"\x00\x00" * 800

    with (
        patch("paramem.server.app._decode_audio_to_pcm", return_value=fake_pcm),
        patch(
            "paramem.server.app.process_utterance",
            new=AsyncMock(
                return_value=UtteranceResult(
                    text="hello", language="en", language_probability=0.95, embedding=[0.1, 0.2]
                )
            ),
        ),
        patch("paramem.server.app._run_chat_turn", new=mock_turn),
    ):
        for _ in range(10):
            resp = tc.post("/voice", content=b"a", headers={"content-type": "audio/webm"})
            assert resp.status_code == 200

    # 10 utterances, each with a fresh per-utterance transport id, all
    # resolving to the same known speaker (speaker0) → exactly ONE _open
    # entry ("voice-speaker0"), not 10 (one per leaked transport id).
    assert list(buffer._open.keys()) == ["voice-speaker0"]
    assert buffer.get_speaker_id("voice-speaker0") == "speaker0"


# ---------------------------------------------------------------------------
# Tests: /chat speaker vs ServingPath distinction
# ---------------------------------------------------------------------------


def test_chat_anonymous_speaker_raw_id_and_personal_serving(tmp_path, monkeypatch):
    """POST /chat for an anonymous-promoted speaker: raw canonical id in
    ChatResponse, and the resolved ``speaker_id`` reaches ``_run_chat_turn``
    unchanged (a speaker_id resolved, even though the display name has not
    been disclosed yet, is full speaker treatment — ``_run_chat_turn``
    derives ``ServingPath.PERSONAL`` from it via ``ServingPath.for_speaker``,
    see ``TestRunChatTurnServingFork`` for that derivation's own coverage).

    Re-spec: the former ``display_speaker`` suppression is gone.  An
    anonymous-promoted speaker still has a resolved speaker_id, so this is
    full speaker treatment (B). Suppression of the raw token from
    HUMAN-facing text is now the reply-boundary resolver's job
    (``resolve_speaker_tokens``), not a pre-emptive None field threaded
    through the whole call chain.
    """
    from paramem.server.app import ResolvedSpeaker

    # An anonymous speaker: store.is_anonymous returns True, speaker is the
    # canonical "speaker3" ID (the store's convention pre-disclosure).
    anon_resolved = ResolvedSpeaker(
        speaker_id="speaker3",
        speaker="speaker3",
        follow_up="What's your name?",
        greeting_prefix=None,
        effective_language=None,
    )

    fake_chat_result = MagicMock()
    fake_chat_result.text = "Hello."
    fake_chat_result.escalated = False

    fresh = _make_state(tmp_path)
    # Disable text-side language detection so the handler does not attempt to
    # load the fastText model (which is not present in CI).
    fresh["config"].text_lang_detection.enabled = False
    monkeypatch.setattr(app_module, "_state", fresh)

    tc = TestClient(
        app_module.app,
        raise_server_exceptions=True,
        headers={},
    )

    mock_run_turn = AsyncMock(return_value=(fake_chat_result, "Hello."))

    with (
        patch(
            "paramem.server.app._resolve_and_enroll_speaker",
            new=AsyncMock(return_value=anon_resolved),
        ),
        patch("paramem.server.app._run_chat_turn", new=mock_run_turn),
    ):
        resp = tc.post(
            "/chat",
            json={"text": "hi", "conversation_id": "test-anon-conv"},
        )

    assert resp.status_code == 200
    body = resp.json()
    # Raw canonical ID propagates to ChatResponse.speaker (attribution)
    assert body["speaker"] == "speaker3"
    # _run_chat_turn must receive the resolved speaker_id (even undisclosed)
    # unchanged — it derives ServingPath.PERSONAL from it internally, not a
    # relay turn.
    call_kwargs = mock_run_turn.call_args.kwargs
    assert call_kwargs["speaker_id"] == "speaker3"


# ---------------------------------------------------------------------------
# Tests: ChatRequest.history retirement — legacy key accepted and ignored
# ---------------------------------------------------------------------------


def test_chat_legacy_history_key_accepted_and_ignored(tmp_path, monkeypatch):
    """A /chat body carrying a legacy ``history`` key (old HA client, or any
    caller unaware of the field's retirement) is accepted — Pydantic's
    ``extra="ignore"`` model default silently drops it — and the turn is
    served normally rather than rejected with a 422.

    Uses a resolved speaker (derives ``ServingPath.PERSONAL`` inside
    ``_run_chat_turn``) so the turn reaches ``handle_chat`` — this test is
    about the legacy body field being dropped, not about the speakerless
    relay fork (covered separately)."""
    from paramem.server.app import ResolvedSpeaker

    resolved = ResolvedSpeaker(
        speaker_id="speaker0",
        speaker="Alex",
        follow_up=None,
        greeting_prefix=None,
        effective_language=None,
    )
    fake_chat_result = MagicMock()
    fake_chat_result.text = "OK."
    fake_chat_result.escalated = False

    fresh = _make_state(tmp_path)
    fresh["config"].text_lang_detection.enabled = False
    monkeypatch.setattr(app_module, "_state", fresh)

    tc = TestClient(
        app_module.app,
        raise_server_exceptions=True,
        headers={},
    )

    with (
        patch(
            "paramem.server.app._resolve_and_enroll_speaker",
            new=AsyncMock(return_value=resolved),
        ),
        patch("paramem.server.app.handle_chat", return_value=fake_chat_result) as mock_handle_chat,
    ):
        resp = tc.post(
            "/chat",
            json={
                "text": "hi",
                "conversation_id": "legacy-history-conv",
                "history": [{"role": "user", "text": "an old client would have sent this"}],
            },
        )

    assert resp.status_code == 200
    assert resp.json()["text"] == "OK."
    # Discriminating assertion: history threaded to handle_chat comes from
    # SessionBuffer.get_session_turns (empty for a brand-new conversation
    # id), never from the legacy request body — the dropped field's content
    # must never reach the dispatcher.
    mock_handle_chat.assert_called_once()
    call_kwargs = mock_handle_chat.call_args.kwargs
    assert call_kwargs["history"] == []
    assert "an old client would have sent this" not in str(call_kwargs["history"])


# ---------------------------------------------------------------------------
# Tests: persist-before-resolve — token-space persist, resolved reply
# ---------------------------------------------------------------------------


def test_persist_before_resolve_reply_boundary(tmp_path, monkeypatch):
    """Order is load-bearing: the persisted assistant turn keeps the raw
    ``speaker{N}`` token (token-space, as generated); only the text returned
    to the caller has it resolved to a display name.  Resolution happens
    strictly after ``SessionBuffer.append``, never before."""
    from paramem.server.app import ResolvedSpeaker

    store = _make_speaker_store(known_ids={"speaker0": "Alice"}, is_anonymous=False)
    fresh = _make_state(tmp_path, speaker_store=store)
    fresh["config"].text_lang_detection.enabled = False
    monkeypatch.setattr(app_module, "_state", fresh)

    resolved = ResolvedSpeaker(
        speaker_id="speaker0",
        speaker="Alice",
        follow_up=None,
        greeting_prefix=None,
        effective_language=None,
    )
    fake_chat_result = MagicMock()
    fake_chat_result.text = "speaker0 likes hiking."
    fake_chat_result.escalated = False

    tc = TestClient(
        app_module.app,
        raise_server_exceptions=True,
        headers={},
    )

    with (
        patch(
            "paramem.server.app._resolve_and_enroll_speaker",
            new=AsyncMock(return_value=resolved),
        ),
        patch("paramem.server.app.handle_chat", return_value=fake_chat_result),
    ):
        resp = tc.post(
            "/chat",
            json={"text": "what do I like", "conversation_id": "persist-test-conv"},
        )

    assert resp.status_code == 200
    # Reply boundary: the display name, not the token, reaches the caller.
    assert resp.json()["text"] == "Alice likes hiking."

    # Persisted turn: token-space, unresolved — the buffer never sees the name.
    turns = fresh["session_buffer"].get_session_turns("persist-test-conv")
    assistant_turns = [t for t in turns if t["role"] == "assistant"]
    assert len(assistant_turns) == 1
    assert assistant_turns[0]["text"] == "speaker0 likes hiking."


# ---------------------------------------------------------------------------
# Tests: ResolvedSpeaker seam unit tests
# ---------------------------------------------------------------------------


class TestResolvedSpeakerSeam:
    """Unit tests for _resolve_and_enroll_speaker contract.

    Calls the seam directly to verify the return contract under three
    representative scenarios: token path, anonymous+embedding, disclosure.
    """

    def _make_chat_request(self, embedding=None, conversation_id="test-conv", text="hello"):
        req = MagicMock()
        req.speaker_embedding = embedding
        req.conversation_id = conversation_id
        req.text = text
        return req

    def _make_buffer(self, speaker_id=None, speaker_name=None):
        buf = MagicMock()
        buf.get_speaker_id.return_value = speaker_id
        buf.get_speaker.return_value = speaker_name
        return buf

    def _patch_state(self, monkeypatch, store=None):
        """Patch _state with a minimal config sufficient for the seam."""
        cfg = _make_config()
        cfg.speaker.enrollment_reprompt_interval = 3600
        cfg.speaker.enrollment_prompt = "What's your name?"
        state = {
            "config": cfg,
            "unknown_speakers": {},
            "pending_enrollments": set(),
            "speaker_store": store,
        }
        monkeypatch.setattr(app_module, "_state", state)
        return state

    def test_token_path_returns_token_identity(self, tmp_path, monkeypatch):
        """auth_speaker_id set → identity from token, register_anonymous not called."""
        store = _make_speaker_store(known_ids={"speaker0": "Alice"}, is_anonymous=False)
        self._patch_state(monkeypatch, store=store)

        req = self._make_chat_request()
        buf = self._make_buffer()

        import asyncio

        result = asyncio.run(
            app_module._resolve_and_enroll_speaker(
                request=req,
                auth_speaker_id="speaker0",
                buffer=buf,
                store=store,
                detected_language=None,
                detected_language_prob=0.0,
            )
        )

        assert result.speaker_id == "speaker0"
        assert result.speaker == "Alice"
        assert result.follow_up is None
        store.register_anonymous.assert_not_called()

    def test_anonymous_embedding_triggers_registration(self, tmp_path, monkeypatch):
        """Unknown voice with embedding → register_anonymous called, follow_up set."""
        store = _make_speaker_store(anon_id="speaker3", is_anonymous=True)
        self._patch_state(monkeypatch, store=store)

        req = self._make_chat_request(embedding=[0.1, 0.2, 0.3])
        buf = self._make_buffer()

        import asyncio

        with patch(
            "paramem.server.app._run_enrollment_for_speaker",
            new=AsyncMock(return_value=None),
        ):
            result = asyncio.run(
                app_module._resolve_and_enroll_speaker(
                    request=req,
                    auth_speaker_id=None,
                    buffer=buf,
                    store=store,
                    detected_language=None,
                    detected_language_prob=0.0,
                )
            )

        assert result.speaker_id == "speaker3"
        assert result.follow_up == "What's your name?"
        store.register_anonymous.assert_called_once_with([0.1, 0.2, 0.3])

    def test_disclosure_clears_follow_up(self, tmp_path, monkeypatch):
        """_run_enrollment_for_speaker returns a name → speaker set, follow_up None."""
        store = _make_speaker_store(anon_id="speaker3", is_anonymous=True)
        self._patch_state(monkeypatch, store=store)

        req = self._make_chat_request(embedding=[0.1, 0.2], text="My name is Alex")
        buf = self._make_buffer()

        import asyncio

        with patch(
            "paramem.server.app._run_enrollment_for_speaker",
            new=AsyncMock(return_value="Alex"),
        ):
            result = asyncio.run(
                app_module._resolve_and_enroll_speaker(
                    request=req,
                    auth_speaker_id=None,
                    buffer=buf,
                    store=store,
                    detected_language=None,
                    detected_language_prob=0.0,
                )
            )

        assert result.speaker == "Alex"
        assert result.follow_up is None


# ---------------------------------------------------------------------------
# Tests: /debug/probe — reply-boundary resolve on both exits
# ---------------------------------------------------------------------------


def test_debug_probe_cloud_only_exit_resolves_speaker_tokens(tmp_path, monkeypatch):
    """Cloud-only branch of ``debug_probe`` (app.py, the
    ``resolve_speaker_tokens(cloud_result.text, store)`` call): the funnel's
    raw answer is resolved to a display name before the response is built —
    same reply-boundary contract as every other exit, previously untested
    for this endpoint."""
    import asyncio

    from paramem.server.app import DebugProbeRequest

    store = _make_speaker_store(
        known_ids={"speaker0": "Alex", "speaker1": "Bob"}, is_anonymous=False
    )
    fresh = _make_state(tmp_path, speaker_store=store, mode="cloud-only")
    fresh["config"].debug = True
    fresh["config"].text_lang_detection.enabled = False
    fresh["config"].cloud.allow_degraded_serving = False
    fresh["cloud_only_reason"] = None
    monkeypatch.setattr(app_module, "_state", fresh)

    with patch.object(
        app_module,
        "_relay_route",
        return_value=MagicMock(text="speaker1 asked that too.", escalated=True),
    ):
        resp = asyncio.run(
            app_module.debug_probe(DebugProbeRequest(text="hi", speaker_id="speaker0"))
        )

    assert resp.text == "Bob asked that too."


def test_debug_probe_local_mode_exit_resolves_speaker_tokens(tmp_path, monkeypatch):
    """Local-mode branch of ``debug_probe`` (app.py, the
    ``resolve_speaker_tokens(result.text, store)`` call): ``handle_chat``'s
    raw answer is resolved to a display name before the response is built."""
    import asyncio

    from paramem.server.app import DebugProbeRequest

    store = _make_speaker_store(
        known_ids={"speaker0": "Alex", "speaker1": "Bob"}, is_anonymous=False
    )
    fresh = _make_state(tmp_path, speaker_store=store, mode="local")
    fresh["config"].debug = True
    fresh["config"].text_lang_detection.enabled = False
    monkeypatch.setattr(app_module, "_state", fresh)

    fake_result = MagicMock()
    fake_result.text = "speaker1 asked that too."
    fake_result.escalated = False

    with patch("paramem.server.app.handle_chat", return_value=fake_result):
        resp = asyncio.run(
            app_module.debug_probe(DebugProbeRequest(text="hi", speaker_id="speaker0"))
        )

    assert resp.text == "Bob asked that too."


def test_debug_probe_anonymous_speaker_display_name_stays_none(tmp_path, monkeypatch):
    """An anonymous-enrolled profile's ``get_name`` returns the RAW
    ``speaker{N}`` token (proving the id exists) while
    ``resolve_speaker_name`` returns ``None`` (anonymous suppression —
    ``SpeakerStore.resolve_speaker_name``'s documented contract).
    ``debug_probe`` must thread ``resolve_speaker_name``'s display name
    (``None`` pre-disclosure) as the ``speaker=`` argument to
    ``handle_chat`` and the response's ``speaker`` field — never
    ``get_name``'s raw token, which exists only for the 404 existence
    check.

    Re-spec note: the local system-prompt identity line
    (:func:`~paramem.server.inference._build_system_prompt`) no longer
    gates on ``speaker`` at all — it is driven by ``speaker_id`` presence
    directly (B-form prefix, anonymous included), so this test is scoped
    to the ``speaker=`` display-name plumbing only, not to whether an
    identity line is emitted.
    """
    import asyncio

    from paramem.server.app import DebugProbeRequest

    store = MagicMock()
    store.get_name.side_effect = lambda sid: sid if sid == "speaker3" else None
    store.resolve_speaker_name.side_effect = lambda sid: None
    fresh = _make_state(tmp_path, speaker_store=store, mode="local")
    fresh["config"].debug = True
    fresh["config"].text_lang_detection.enabled = False
    monkeypatch.setattr(app_module, "_state", fresh)

    fake_result = MagicMock()
    fake_result.text = "hi there"
    fake_result.escalated = False

    captured = {}

    def fake_handle_chat(**kwargs):
        captured["speaker"] = kwargs.get("speaker")
        return fake_result

    with patch("paramem.server.app.handle_chat", side_effect=fake_handle_chat):
        resp = asyncio.run(
            app_module.debug_probe(DebugProbeRequest(text="hi", speaker_id="speaker3"))
        )

    assert captured["speaker"] is None, (
        "debug_probe must thread resolve_speaker_name's anonymous-suppressed "
        "None into handle_chat's speaker= argument, never get_name's raw token"
    )
    assert resp.speaker is None
