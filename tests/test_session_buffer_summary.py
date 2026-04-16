"""Unit tests for SessionBuffer.get_summary — pending attribution stats."""

import pytest

from paramem.server.session_buffer import SessionBuffer


@pytest.fixture
def buf(tmp_path):
    return SessionBuffer(session_dir=tmp_path / "sessions", debug=False)


def test_summary_empty(buf):
    s = buf.get_summary()
    assert s == {"total": 0, "orphaned": 0, "oldest_age_seconds": None, "per_speaker": {}}


def test_summary_orphaned_session(buf):
    buf.append("conv-1", "user", "hello")
    s = buf.get_summary()
    assert s["total"] == 1
    assert s["orphaned"] == 1
    assert s["per_speaker"] == {}
    assert s["oldest_age_seconds"] is not None
    assert s["oldest_age_seconds"] >= 0


def test_summary_attributed_session(buf):
    buf.set_speaker("conv-1", "spk-abc", "Alice")
    buf.append("conv-1", "user", "hello")
    s = buf.get_summary()
    assert s["total"] == 1
    assert s["orphaned"] == 0
    assert s["per_speaker"] == {"spk-abc": 1}


def test_retro_claim_attributes_matching_orphan(buf, tmp_path):
    """Orphan sessions with matching voice embeddings get claimed by existing profiles."""
    import math

    from paramem.server.speaker import SpeakerStore

    v = [0.5, 0.3, 0.7, 0.1, 0.4, 0.6, 0.2, 0.8]
    norm = math.sqrt(sum(x * x for x in v))
    embedding = [x / norm for x in v]

    store = SpeakerStore(tmp_path / "profiles.json")
    speaker_id = store.enroll("Alex", embedding)

    # Orphan session with matching voice
    buf.append("conv-orphan", "user", "hello there", embedding=embedding)
    assert buf.get_summary()["orphaned"] == 1

    claimed = buf.claim_sessions_for_speaker(speaker_id, "Alex", store)
    assert claimed == 1

    s = buf.get_summary()
    assert s["orphaned"] == 0
    assert s["per_speaker"] == {speaker_id: 1}


def test_summary_mixed_orphaned_and_attributed(buf):
    buf.set_speaker("conv-a", "spk-1", "Alice")
    buf.append("conv-a", "user", "hi")
    buf.append("conv-b", "user", "anonymous")  # no speaker
    buf.set_speaker("conv-c", "spk-1", "Alice")
    buf.append("conv-c", "user", "hi again")
    s = buf.get_summary()
    assert s["total"] == 3
    assert s["orphaned"] == 1
    assert s["per_speaker"] == {"spk-1": 2}
