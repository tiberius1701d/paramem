"""Tests for voice-based speaker identification."""

import json
import math

import pytest

from paramem.server.speaker import (
    SpeakerStore,
    _l2_normalize,
    compute_centroid,
    cosine_similarity,
)

# --- Cosine similarity ---


def test_cosine_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_opposite_vectors():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_empty_vectors():
    assert cosine_similarity([], []) == 0.0


def test_cosine_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_different_lengths():
    assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0


# --- Centroid computation ---


def test_centroid_single_embedding():
    v = [3.0, 4.0]
    c = compute_centroid([v])
    # Should be L2-normalized
    assert cosine_similarity(c, v) == pytest.approx(1.0)
    norm = math.sqrt(sum(x * x for x in c))
    assert norm == pytest.approx(1.0)


def test_centroid_identical_embeddings():
    v = [1.0, 0.0, 0.0]
    c = compute_centroid([v, v, v])
    assert cosine_similarity(c, v) == pytest.approx(1.0)


def test_centroid_symmetric():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    c = compute_centroid([a, b])
    # Centroid should be equidistant from both
    assert cosine_similarity(c, a) == pytest.approx(cosine_similarity(c, b))


def test_centroid_empty():
    assert compute_centroid([]) == []


def test_l2_normalize():
    v = [3.0, 4.0]
    n = _l2_normalize(v)
    assert n == pytest.approx([0.6, 0.8])


def test_l2_normalize_zero():
    n = _l2_normalize([0.0, 0.0])
    assert n == [0.0, 0.0]


# --- SpeakerStore ---


@pytest.fixture
def store(tmp_path):
    return SpeakerStore(tmp_path / "profiles.json")


@pytest.fixture
def sample_embedding():
    """A normalized 8-dim embedding for testing."""
    v = [0.5, 0.3, 0.7, 0.1, 0.4, 0.6, 0.2, 0.8]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


@pytest.fixture
def different_embedding():
    """A distinctly different normalized 8-dim embedding (low similarity to sample)."""
    v = [0.0, 0.9, 0.1, 0.8, 0.0, 0.1, 0.9, 0.0]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def test_empty_store_no_match(store):
    result = store.match([0.1, 0.2, 0.3])
    assert result.speaker_id is None
    assert result.name is None
    assert result.confidence == 0.0


def test_list_profiles_empty(store):
    assert store.list_profiles() == []


def test_list_profiles_after_enroll(store, sample_embedding, different_embedding):
    alice_id = store.enroll("Alice", sample_embedding)
    bob_id = store.enroll("Bob", different_embedding, method="manual")
    profiles = {p["id"]: p for p in store.list_profiles()}
    assert set(profiles.keys()) == {alice_id, bob_id}
    assert profiles[alice_id]["name"] == "Alice"
    assert profiles[alice_id]["embeddings"] == 1
    assert profiles[alice_id]["enroll_method"] == "self_introduced"
    assert profiles[bob_id]["embeddings"] == 1
    assert profiles[bob_id]["enroll_method"] == "manual"


def test_list_profiles_legacy_default_method(store, sample_embedding):
    """Legacy profiles loaded without enroll_method report 'unknown'."""
    sid = store.enroll("Alice", sample_embedding)
    # Simulate a legacy profile missing the field
    store._profiles[sid].pop("enroll_method", None)
    profiles = store.list_profiles()
    assert profiles[0]["enroll_method"] == "unknown"


def test_enroll_and_exact_match(store, sample_embedding):
    speaker_id = store.enroll("Alice", sample_embedding)
    assert speaker_id is not None
    assert speaker_id == "speaker0"  # first enroll into empty store gets speaker0
    result = store.match(sample_embedding)
    assert result.speaker_id == speaker_id
    assert result.name == "Alice"
    assert result.confidence == pytest.approx(1.0, abs=0.01)
    assert not result.tentative


def test_high_confidence_match(store, sample_embedding):
    speaker_id = store.enroll("Alice", sample_embedding)
    noisy = [x + 0.05 for x in sample_embedding]
    result = store.match(noisy)
    assert result.speaker_id == speaker_id
    assert result.name == "Alice"
    assert result.confidence >= store.high_threshold
    assert not result.tentative


def test_low_confidence_tentative(store):
    store_low = SpeakerStore(store.store_path, high_threshold=0.95, low_threshold=0.50)
    emb_a = [1.0, 0.0, 0.0, 0.0]
    emb_b = [0.7, 0.7, 0.0, 0.0]  # ~0.71 cosine similarity
    store_low.enroll("Bob", emb_a)
    result = store_low.match(emb_b)
    assert result.name == "Bob"
    assert result.speaker_id is not None
    assert 0.50 <= result.confidence < 0.95
    assert result.tentative


def test_no_match_below_threshold(store, sample_embedding, different_embedding):
    store_strict = SpeakerStore(store.store_path, high_threshold=0.99, low_threshold=0.98)
    store_strict.enroll("Alice", sample_embedding)
    result = store_strict.match(different_embedding)
    assert result.speaker_id is None
    assert result.name is None


def test_persistence_round_trip(tmp_path, sample_embedding):
    path = tmp_path / "profiles.json"
    store1 = SpeakerStore(path)
    speaker_id = store1.enroll("Alice", sample_embedding)
    assert store1.profile_count == 1

    store2 = SpeakerStore(path)
    assert store2.profile_count == 1
    result = store2.match(sample_embedding)
    assert result.speaker_id == speaker_id
    assert result.name == "Alice"
    assert result.confidence == pytest.approx(1.0, abs=0.01)


def test_enroll_same_voice_rejected(store, sample_embedding):
    """Same voice enrolling under a different name is rejected."""
    speaker_id = store.enroll("Alice", sample_embedding)
    assert speaker_id is not None
    # Same embedding, different name — rejected (voice already enrolled)
    result = store.enroll("Bob", sample_embedding)
    assert result is None
    assert store.profile_count == 1


def test_enroll_same_name_different_voice(tmp_path, sample_embedding, different_embedding):
    """Two different voices can share the same display name."""
    # Use strict thresholds so test embeddings aren't rejected as same voice
    store = SpeakerStore(tmp_path / "profiles.json", high_threshold=0.90, low_threshold=0.70)
    id1 = store.enroll("Alex", sample_embedding)
    id2 = store.enroll("Alex", different_embedding)
    assert id1 is not None
    assert id2 is not None
    assert id1 != id2
    assert store.profile_count == 2

    result1 = store.match(sample_embedding)
    assert result1.speaker_id == id1
    assert result1.name == "Alex"

    result2 = store.match(different_embedding)
    assert result2.speaker_id == id2
    assert result2.name == "Alex"


def test_remove_profile(store, sample_embedding):
    speaker_id = store.enroll("Alice", sample_embedding)
    assert store.profile_count == 1
    assert store.remove(speaker_id)
    assert store.profile_count == 0
    assert not store.remove(speaker_id)  # Already removed


def test_remove_nonexistent(store):
    assert not store.remove("nonexistent_id")


def test_multiple_speakers(store):
    emb_a = [1.0, 0.0, 0.0, 0.0]
    emb_b = [0.0, 1.0, 0.0, 0.0]
    id_a = store.enroll("Alice", emb_a)
    id_b = store.enroll("Bob", emb_b)
    assert store.profile_count == 2

    result_a = store.match(emb_a)
    assert result_a.speaker_id == id_a
    assert result_a.name == "Alice"

    result_b = store.match(emb_b)
    assert result_b.speaker_id == id_b
    assert result_b.name == "Bob"


def test_speaker_names(store, sample_embedding, different_embedding):
    store.enroll("Alice", sample_embedding)
    store.enroll("Bob", different_embedding)
    names = store.speaker_names
    assert "Alice" in names
    assert "Bob" in names


def test_speaker_names_deduplication(store, sample_embedding, different_embedding):
    """speaker_names returns unique names even with multiple profiles sharing a name."""
    store.enroll("Alex", sample_embedding)
    store.enroll("Alex", different_embedding)
    assert store.speaker_names == ["Alex"]


def test_get_name(store, sample_embedding):
    speaker_id = store.enroll("Alice", sample_embedding)
    assert store.get_name(speaker_id) == "Alice"
    assert store.get_name("nonexistent") is None


# ---------------------------------------------------------------------------
# P3 — resolve_speaker_name (anonymous-suppressed name resolver)
# ---------------------------------------------------------------------------


class TestResolveSpeakerName:
    """P3: ``resolve_speaker_name`` returns the display name or ``None``.

    Miss cases:
    - Unknown speaker_id → ``None`` (no such profile).
    - Anonymous profile (``enroll_method == "anonymous_voice"``) → ``None``
      (name equals speaker_id, not a salutation).

    Hit case:
    - Known, named profile → display name.

    No exception path: the method is a dict lookup on ``_profiles`` and must
    never raise, so removing the ``try/except get_name failed`` wrappers in
    app.py is safe.
    """

    def test_known_named_speaker_returns_name(self, tmp_path, sample_embedding):
        store = SpeakerStore(tmp_path / "profiles.json")
        sid = store.enroll("Tobias", sample_embedding)
        assert store.resolve_speaker_name(sid) == "Tobias"

    def test_unknown_speaker_id_returns_none(self, tmp_path):
        store = SpeakerStore(tmp_path / "profiles.json")
        assert store.resolve_speaker_name("speaker99") is None

    def test_anonymous_profile_returns_none(self, tmp_path, sample_embedding):
        """Anonymous profiles must NOT surface their speaker_id as a display name."""
        store = SpeakerStore(tmp_path / "profiles.json")
        sid = store.register_anonymous(sample_embedding)
        # Verify the profile was created with anonymous_voice method.
        assert store.is_anonymous(sid)
        # P3 must suppress the anonymous token.
        assert store.resolve_speaker_name(sid) is None

    def test_no_exception_on_unknown(self, tmp_path):
        """resolve_speaker_name must not raise for any speaker_id value.

        This proves the deleted try/except blocks in app.py were dead
        defensive code: a dict lookup on _profiles cannot raise.
        """
        store = SpeakerStore(tmp_path / "profiles.json")
        # None-like inputs, missing keys, empty string — all return None safely.
        assert store.resolve_speaker_name("nonexistent") is None
        assert store.resolve_speaker_name("speaker0") is None
        assert store.resolve_speaker_name("") is None


def test_empty_embedding_no_match(store, sample_embedding):
    store.enroll("Alice", sample_embedding)
    result = store.match([])
    assert result.speaker_id is None


def test_enroll_empty_name_rejected(store, sample_embedding):
    assert store.enroll("", sample_embedding) is None
    assert store.profile_count == 0


def test_enroll_empty_embedding_rejected(store):
    assert store.enroll("Alice", []) is None
    assert store.profile_count == 0


def test_corrupted_file_handled(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text("not json{{{")
    store = SpeakerStore(path)
    assert store.profile_count == 0


def test_file_created_on_first_enroll(tmp_path, sample_embedding):
    path = tmp_path / "subdir" / "profiles.json"
    assert not path.exists()
    store = SpeakerStore(path)
    store.enroll("Alice", sample_embedding)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["version"] == 6
    assert len(data["speakers"]) == 1
    profile = list(data["speakers"].values())[0]
    assert "embeddings" in profile
    assert len(profile["embeddings"]) == 1


def test_legacy_v1_load_raises(tmp_path, sample_embedding):
    """v1 store (retired migration) raises ValueError on load."""
    path = tmp_path / "profiles.json"
    legacy_data = {"speakers": {"Alice": sample_embedding}}
    path.write_text(json.dumps(legacy_data))

    import pytest

    with pytest.raises(ValueError, match="Unsupported speaker store version"):
        SpeakerStore(path)


def test_legacy_v2_load_raises(tmp_path, sample_embedding):
    """v2 store (retired migration) raises ValueError on load."""
    path = tmp_path / "profiles.json"
    v2_data = {
        "speakers": {"abc12345": {"name": "Alice", "embedding": sample_embedding}},
        "version": 2,
    }
    path.write_text(json.dumps(v2_data))

    import pytest

    with pytest.raises(ValueError, match="Unsupported speaker store version"):
        SpeakerStore(path)


def test_legacy_v3_load_raises(tmp_path, sample_embedding):
    """v3 store (retired migration) raises ValueError on load."""
    path = tmp_path / "profiles.json"
    v3_payload = {
        "version": 3,
        "next_anon_index": 1,
        "last_greeted": {},
        "speakers": {
            "Speaker0": {
                "name": "Alice",
                "embeddings": [sample_embedding],
                "preferred_language": "en",
            },
        },
    }
    path.write_text(json.dumps(v3_payload))

    import pytest

    with pytest.raises(ValueError, match="Unsupported speaker store version"):
        SpeakerStore(path)


# --- Multi-embedding / centroid ---


def test_add_embedding_enriches_profile(store, sample_embedding):
    speaker_id = store.enroll("Alice", sample_embedding)
    # Perturb enough to be below 0.95 redundancy threshold but still same-ish
    perturbed = [x + (0.5 if i % 2 == 0 else -0.3) for i, x in enumerate(sample_embedding)]
    assert store.add_embedding(speaker_id, perturbed)
    store.flush()  # deferred save

    # Verify persistence
    data = json.loads(store.store_path.read_text())
    assert len(data["speakers"][speaker_id]["embeddings"]) == 2


def test_add_embedding_rejects_redundant(store, sample_embedding):
    speaker_id = store.enroll("Alice", sample_embedding)
    # Same embedding is > 0.95 similar to centroid — skipped
    assert not store.add_embedding(speaker_id, sample_embedding)


def test_add_embedding_nonexistent_profile(store, sample_embedding):
    assert not store.add_embedding("nonexistent", sample_embedding)


def test_add_embedding_cap_enforced(tmp_path, sample_embedding):
    """Max embeddings cap is enforced."""
    store = SpeakerStore(tmp_path / "profiles.json", max_embeddings=2)
    sid = store.enroll("Alice", sample_embedding)
    # First add succeeds (different enough from centroid)
    emb2 = [x + (0.5 if i % 2 == 0 else -0.3) for i, x in enumerate(sample_embedding)]
    assert store.add_embedding(sid, emb2)
    # Second add rejected — cap reached (1 from enroll + 1 from add = 2)
    emb3 = [-x for x in sample_embedding]
    assert not store.add_embedding(sid, emb3)


def test_deferred_save_not_persisted_before_flush(tmp_path, sample_embedding):
    """Enrichment is not written to disk until flush() is called."""
    path = tmp_path / "profiles.json"
    store = SpeakerStore(path)
    sid = store.enroll("Alice", sample_embedding)
    perturbed = [x + (0.5 if i % 2 == 0 else -0.3) for i, x in enumerate(sample_embedding)]
    store.add_embedding(sid, perturbed)
    # Read file before flush — should still have 1 embedding (from enroll)
    data = json.loads(path.read_text())
    assert len(data["speakers"][sid]["embeddings"]) == 1
    store.flush()
    data = json.loads(path.read_text())
    assert len(data["speakers"][sid]["embeddings"]) == 2


def test_centroid_improves_with_samples(store):
    """Centroid from two directions matches both better than either alone."""
    emb_a = [1.0, 0.2, 0.0, 0.0]
    emb_b = [0.8, 0.0, 0.3, 0.0]
    test_mid = [0.9, 0.1, 0.15, 0.0]

    # Single embedding profile
    store_single = SpeakerStore(store.store_path, high_threshold=0.60, low_threshold=0.45)
    sid = store_single.enroll("Alice", emb_a)
    result_single = store_single.match(test_mid)

    # Add second embedding
    store_single.add_embedding(sid, emb_b)
    result_centroid = store_single.match(test_mid)

    assert result_centroid.confidence >= result_single.confidence


# --- Greeting logic ---


class TestTimePeriod:
    def test_morning(self):
        assert SpeakerStore._time_period(6) == "morning"
        assert SpeakerStore._time_period(0) == "morning"
        assert SpeakerStore._time_period(11) == "morning"

    def test_afternoon(self):
        assert SpeakerStore._time_period(12) == "afternoon"
        assert SpeakerStore._time_period(15) == "afternoon"
        assert SpeakerStore._time_period(17) == "afternoon"

    def test_evening(self):
        assert SpeakerStore._time_period(18) == "evening"
        assert SpeakerStore._time_period(21) == "evening"
        assert SpeakerStore._time_period(23) == "evening"


_GREETINGS = {
    "en": {"morning": "Good morning", "afternoon": "Good afternoon", "evening": "Good evening"},
    "de": {"morning": "Guten Morgen", "afternoon": "Guten Tag", "evening": "Guten Abend"},
}


class TestGreetingText:
    def test_english(self):
        assert SpeakerStore._greeting_text("morning", _GREETINGS, "en") == "Good morning"
        assert SpeakerStore._greeting_text("evening", _GREETINGS, "en") == "Good evening"

    def test_german_localized(self):
        assert SpeakerStore._greeting_text("afternoon", _GREETINGS, "de") == "Guten Tag"

    def test_unknown_language_falls_back_to_english(self):
        assert SpeakerStore._greeting_text("morning", _GREETINGS, "xx") == "Good morning"


class TestShouldGreet:
    def test_disabled_when_interval_zero(self, tmp_path):
        store = SpeakerStore(tmp_path / "profiles.json")
        assert store.should_greet("speaker1", 0, _GREETINGS) is None

    def test_first_greeting(self, tmp_path):
        store = SpeakerStore(tmp_path / "profiles.json")
        result = store.should_greet("speaker1", 24, _GREETINGS)
        assert result is not None
        assert "Good" in result

    def test_german_greeting(self, tmp_path):
        store = SpeakerStore(tmp_path / "profiles.json")
        result = store.should_greet("speaker1", 24, _GREETINGS, language="de")
        assert result in ("Guten Morgen", "Guten Tag", "Guten Abend")

    def test_no_repeat_within_period(self, tmp_path):
        store = SpeakerStore(tmp_path / "profiles.json")
        result1 = store.should_greet("speaker1", 24, _GREETINGS)
        assert result1 is not None
        store.confirm_greeting("speaker1")
        result2 = store.should_greet("speaker1", 24, _GREETINGS)
        assert result2 is None

    def test_different_speakers_independent(self, tmp_path):
        store = SpeakerStore(tmp_path / "profiles.json")
        store.should_greet("speaker1", 24, _GREETINGS)
        store.confirm_greeting("speaker1")
        result = store.should_greet("speaker2", 24, _GREETINGS)
        assert result is not None


# ---------------------------------------------------------------------------
# speaker_embedding module — unit tests (no pyannote model download required)
# ---------------------------------------------------------------------------


def _make_pcm(duration_seconds: float, sample_rate: int = 16000, amplitude: float = 0.5) -> bytes:
    """Generate a constant-amplitude int16 mono PCM buffer for testing."""
    import numpy as np

    n_samples = int(duration_seconds * sample_rate)
    value = int(amplitude * 32767)
    samples = np.full(n_samples, value, dtype=np.int16)
    return samples.tobytes()


def test_rms_normalization_applied(monkeypatch):
    """RMS normalisation equalises gain: loud and quiet inputs reach ~0.1 RMS.

    The pyannote Inference call is monkeypatched to capture the waveform
    actually passed to the model and return a dummy embedding.
    """
    import numpy as np
    import torch

    import paramem.server.speaker_embedding as mod

    captured_waveforms = []

    class _FakeInference:
        def __call__(self, inputs):
            captured_waveforms.append(inputs["waveform"].clone())
            return torch.zeros(1, 256)  # dummy 256-dim embedding

    mod._embedding_model = _FakeInference()
    sample_rate = 16000

    try:
        # Loud: amplitude 0.9
        loud_pcm = _make_pcm(2.0, sample_rate, amplitude=0.9)
        mod.compute_speaker_embedding(loud_pcm, sample_rate)

        # Quiet: amplitude 0.05
        quiet_pcm = _make_pcm(2.0, sample_rate, amplitude=0.05)
        mod.compute_speaker_embedding(quiet_pcm, sample_rate)
    finally:
        mod._embedding_model = None

    assert len(captured_waveforms) == 2, "Expected two inference calls"

    for wf in captured_waveforms:
        audio = wf.squeeze(0).numpy()
        rms = float(np.sqrt(np.mean(audio**2)))
        assert abs(rms - 0.1) / 0.1 < 0.10, f"RMS after normalisation expected ≈0.1, got {rms:.4f}"


def test_duration_gate_rejects_short_audio():
    """Utterances shorter than min_duration_seconds return an empty list.

    Uses a 0.5s buffer and the default threshold of 1.0s.  When the
    Inference object is set, the duration gate must still reject before
    calling it; for the longer-buffer assertion, Inference is monkeypatched
    to return a dummy embedding so the test does not require a real model.
    """
    import torch

    import paramem.server.speaker_embedding as mod

    sample_rate = 16000
    short_pcm = _make_pcm(0.5, sample_rate)  # 0.5s — below 1.0s threshold

    class _FakeInference:
        def __call__(self, inputs):
            return torch.zeros(1, 256)

    # With no model loaded, duration check is never reached — both paths return [].
    # Load a fake model so the duration gate itself is exercised.
    mod._embedding_model = _FakeInference()
    try:
        result_short = mod.compute_speaker_embedding(
            short_pcm, sample_rate, min_duration_seconds=1.0
        )
        assert result_short == [], f"Expected [] for 0.5s audio, got {result_short}"

        long_pcm = _make_pcm(1.2, sample_rate)
        result_long = mod.compute_speaker_embedding(long_pcm, sample_rate, min_duration_seconds=1.0)
        assert isinstance(result_long, list), "Expected list from 1.2s audio"
        assert len(result_long) > 0, "Expected non-empty embedding for 1.2s audio"
    finally:
        mod._embedding_model = None


def test_min_embedding_duration_seconds_config_default():
    """SpeakerConfig default for min_embedding_duration_seconds is 1.0."""
    from paramem.server.config import SpeakerConfig

    cfg = SpeakerConfig()
    assert cfg.min_embedding_duration_seconds == 1.0


# ---------------------------------------------------------------------------
# register_anonymous — anonymous speaker ID promotion to a named identity
# ---------------------------------------------------------------------------


class TestRegisterAnonymous:
    """Tests for SpeakerStore.register_anonymous."""

    def test_first_registration_returns_speaker0(self, tmp_path, sample_embedding):
        """First unrecognized voice gets speaker0."""
        store = SpeakerStore(tmp_path / "profiles.json")
        anon_id = store.register_anonymous(sample_embedding)
        assert anon_id == "speaker0"

    def test_second_distinct_embedding_gets_speaker1(
        self, tmp_path, sample_embedding, different_embedding
    ):
        """Two distinct voices get incrementing IDs."""
        store = SpeakerStore(tmp_path / "profiles.json")
        id0 = store.register_anonymous(sample_embedding)
        id1 = store.register_anonymous(different_embedding)
        assert id0 == "speaker0"
        assert id1 == "speaker1"

    def test_same_embedding_returns_same_id(self, tmp_path, sample_embedding):
        """Re-registering the same embedding is idempotent (centroid match)."""
        store = SpeakerStore(tmp_path / "profiles.json")
        id_first = store.register_anonymous(sample_embedding)
        id_second = store.register_anonymous(sample_embedding)
        assert id_first == id_second
        # Counter must not have advanced on the second call.
        assert store._next_anon_index == 1

    def test_counter_persists_across_reload(self, tmp_path, sample_embedding, different_embedding):
        """next_anon_index survives a SpeakerStore reload from disk."""
        path = tmp_path / "profiles.json"
        store1 = SpeakerStore(path)
        store1.register_anonymous(sample_embedding)
        store1.register_anonymous(different_embedding)
        assert store1._next_anon_index == 2

        # Reload from disk.
        store2 = SpeakerStore(path)
        assert store2._next_anon_index == 2
        # Next allocation continues from 2. Use strict thresholds so emb3 is
        # definitely below low_threshold and gets a fresh allocation rather than
        # tentatively matching one of the anonymous profiles under the default
        # thresholds (high=0.60, low=0.45).
        store2_strict = SpeakerStore(path, high_threshold=0.90, low_threshold=0.80)
        emb3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        id_new = store2_strict.register_anonymous(emb3)
        assert id_new == "speaker2"

    def test_disk_has_version6_and_next_anon_index(self, tmp_path, sample_embedding):
        """JSON on disk has version: 6 and next_anon_index field after registration."""
        path = tmp_path / "profiles.json"
        store = SpeakerStore(path)
        store.register_anonymous(sample_embedding)
        data = json.loads(path.read_text())
        assert data["version"] == 6
        assert "next_anon_index" in data
        assert data["next_anon_index"] == 1

    def test_v4_load_raises(self, tmp_path, sample_embedding):
        """v4 store (retired migration) raises ValueError on load."""
        path = tmp_path / "profiles.json"
        v4_data = {
            "speakers": {
                "abc12345": {
                    "name": "Alice",
                    "embeddings": [sample_embedding],
                    "preferred_language": "",
                    "enroll_method": "self_introduced",
                }
            },
            "last_greeted": {},
            "version": 4,
        }
        path.write_text(json.dumps(v4_data))

        with pytest.raises(ValueError, match="Unsupported speaker store version"):
            SpeakerStore(path)

    def test_empty_embedding_raises(self, tmp_path):
        """register_anonymous rejects an empty embedding."""
        store = SpeakerStore(tmp_path / "profiles.json")
        with pytest.raises(ValueError, match="non-empty"):
            store.register_anonymous([])

    def test_named_speaker_not_overwritten(self, tmp_path, sample_embedding):
        """A voice already enrolled under a real name returns the named ID."""
        store = SpeakerStore(tmp_path / "profiles.json")
        named_id = store.enroll("Alice", sample_embedding)
        assert named_id is not None
        # register_anonymous should return the existing named profile via centroid match.
        anon_id = store.register_anonymous(sample_embedding)
        assert anon_id == named_id
        # enroll() now uses _mint_anon_speaker_id() — the shared monotonic counter
        # advances for ALL mints, named or anonymous. Alice's enroll bumped it to 1.
        assert store._next_anon_index == 1
        profiles = store.list_profiles()
        assert len(profiles) == 1
        assert profiles[0]["name"] == "Alice"

    def test_anonymous_profile_uses_enroll_method_anonymous_voice(self, tmp_path, sample_embedding):
        """Anonymous profiles record enroll_method='anonymous_voice'."""
        store = SpeakerStore(tmp_path / "profiles.json")
        anon_id = store.register_anonymous(sample_embedding)
        profiles = {p["id"]: p for p in store.list_profiles()}
        assert profiles[anon_id]["enroll_method"] == "anonymous_voice"

    def test_get_name_returns_speaker_id_for_anonymous(self, tmp_path, sample_embedding):
        """For anonymous profiles name==speaker_id so get_name returns the ID."""
        store = SpeakerStore(tmp_path / "profiles.json")
        anon_id = store.register_anonymous(sample_embedding)
        assert store.get_name(anon_id) == anon_id


class TestIsAnonymous:
    """``is_anonymous`` lets user-facing text suppress the canonical
    ``Speaker{N}`` token until the profile is name-disclosed."""

    def test_anonymous_profile_is_anonymous(self, tmp_path, sample_embedding):
        store = SpeakerStore(tmp_path / "profiles.json")
        anon_id = store.register_anonymous(sample_embedding)
        assert store.is_anonymous(anon_id) is True

    def test_named_profile_is_not_anonymous(self, tmp_path, sample_embedding):
        store = SpeakerStore(tmp_path / "profiles.json")
        named_id = store.enroll("Alice", sample_embedding)
        assert store.is_anonymous(named_id) is False

    def test_upgrade_flips_anonymous_to_named(self, tmp_path, sample_embedding):
        """After register_anonymous → enroll, the same speaker_id stops
        being reported as anonymous. This is the path that lets the
        greeting prefix start using the real name on the very next turn."""
        store = SpeakerStore(tmp_path / "profiles.json")
        anon_id = store.register_anonymous(sample_embedding)
        assert store.is_anonymous(anon_id) is True
        upgraded_id = store.enroll("Alice", sample_embedding)
        assert upgraded_id == anon_id
        assert store.is_anonymous(anon_id) is False

    def test_unknown_or_none_id_is_not_anonymous(self, tmp_path):
        store = SpeakerStore(tmp_path / "profiles.json")
        assert store.is_anonymous(None) is False
        assert store.is_anonymous("") is False
        assert store.is_anonymous("nonexistent") is False


# ---------------------------------------------------------------------------
# enroll() upgrade-in-place for anonymous profiles (Fix 1)
# ---------------------------------------------------------------------------


class TestEnrollUpgradesAnonymous:
    """enroll() must upgrade anonymous Speaker{N} profiles rather than reject."""

    def test_enroll_after_register_anonymous_upgrades_profile(self, tmp_path, sample_embedding):
        """After register_anonymous creates speaker0, enrolling with the same voice
        and a real name returns speaker0 (not a new UUID), upgrades the name, and
        sets enroll_method to the requested method.
        """
        store = SpeakerStore(tmp_path / "profiles.json")
        anon_id = store.register_anonymous(sample_embedding)
        assert anon_id == "speaker0"

        result_id = store.enroll("Alice", sample_embedding)
        assert result_id == "speaker0", "enroll() must return the existing anonymous ID"
        assert store.get_name("speaker0") == "Alice"
        profiles = {p["id"]: p for p in store.list_profiles()}
        assert profiles["speaker0"]["enroll_method"] == "self_introduced"
        # Profile count must not increase — upgrade in-place only.
        assert store.profile_count == 1

    def test_enroll_against_named_profile_still_rejected(
        self, tmp_path, sample_embedding, different_embedding
    ):
        """Enrolling a voice already present as a named profile still returns None.

        Prevents the same voice registering under two different names.
        Uses strict thresholds so the two test embeddings are well-separated.
        """
        store = SpeakerStore(tmp_path / "profiles.json", high_threshold=0.90, low_threshold=0.70)
        alice_id = store.enroll("Alice", sample_embedding)
        assert alice_id is not None

        # Attempt to enroll the same voice (sample_embedding) under a new name.
        result = store.enroll("Bob", sample_embedding)
        assert result is None
        assert store.profile_count == 1


# ---------------------------------------------------------------------------
# register_anonymous tentative-match semantics (Fix 2)
# ---------------------------------------------------------------------------


class TestRegisterAnonymousTentative:
    """Tentative matches behave correctly: reuse anonymous, isolate named."""

    def _make_tentative_store(self, tmp_path: object) -> "SpeakerStore":
        """Return a store with high_threshold=0.95, low_threshold=0.50 so that
        embeddings with cosine similarity in [0.50, 0.95) are tentative.
        """
        return SpeakerStore(
            tmp_path / "profiles.json",
            high_threshold=0.95,
            low_threshold=0.50,
        )

    def _make_tentative_pair(self) -> tuple[list[float], list[float]]:
        """Return two embeddings whose cosine similarity is ~0.71 (tentative zone
        for thresholds high=0.95, low=0.50).

        [1,0,0,0] and [0.7,0.7,0,0] have cosine similarity ≈ 0.707.
        """

        a = [1.0, 0.0, 0.0, 0.0]
        b_raw = [0.7, 0.7, 0.0, 0.0]
        norm_b = math.sqrt(sum(x * x for x in b_raw))
        b = [x / norm_b for x in b_raw]
        return a, b

    def test_tentative_against_anonymous_reuses_same_id(self, tmp_path):
        """A tentative match against an anonymous profile reuses that profile.

        Avoids split-identity when the same speaker's embedding varies slightly
        across sessions but remains within the tentative zone.
        """
        store = self._make_tentative_store(tmp_path)
        emb_a, emb_b = self._make_tentative_pair()

        id_a = store.register_anonymous(emb_a)
        assert id_a == "speaker0"

        # emb_b is tentatively similar to emb_a / speaker0 (anonymous).
        id_b = store.register_anonymous(emb_b)
        assert id_b == "speaker0", (
            "Tentative match against anonymous profile must reuse the existing ID"
        )
        # Counter must not advance — no new profile created.
        assert store._next_anon_index == 1

    def test_tentative_against_named_allocates_new_speaker(self, tmp_path):
        """A tentative match against a named profile allocates a new Speaker{N}.

        Protects named-profile centroids from contamination by ambiguous voices.
        """
        store = self._make_tentative_store(tmp_path)
        emb_a, emb_b = self._make_tentative_pair()

        # Enroll emb_a as a named speaker — now uses the shared counter, so
        # Alice gets speaker0 and the counter advances to 1.
        alice_id = store.enroll("Alice", emb_a)
        assert alice_id is not None
        assert alice_id == "speaker0"

        # emb_b is tentative against Alice's profile — must not contaminate it.
        # register_anonymous falls through and allocates the next counter slot.
        anon_id = store.register_anonymous(emb_b)
        assert anon_id != alice_id
        assert anon_id == "speaker1"
        # Alice's profile must remain unchanged.
        assert store.get_name(alice_id) == "Alice"
        profiles = {p["id"]: p for p in store.list_profiles()}
        assert profiles[alice_id]["enroll_method"] != "anonymous_voice"


# ---------------------------------------------------------------------------
# enroll() tentative-band upgrade — closes the anonymous→named identity split
# ---------------------------------------------------------------------------


class TestEnrollTentativeUpgrade:
    """Regression tests for the anonymous→named identity split fix.

    Previously, enroll() only upgraded anonymous profiles at high-confidence
    (conf >= high_threshold).  A tentative-band self-introduction minted a new
    uuid id, splitting the speaker's identity from their existing Speaker{N}.

    After the fix, enroll() upgrades anonymous profiles in BOTH bands
    (mirrors register_anonymous's tentative-reuse of anonymous profiles).
    Named profiles at tentative confidence still mint a fresh Speaker{N}
    (protects the named centroid from ambiguous-voice contamination).
    """

    def _make_tentative_store(self, tmp_path: object) -> "SpeakerStore":
        return SpeakerStore(
            tmp_path / "profiles.json",
            high_threshold=0.95,
            low_threshold=0.50,
        )

    def _make_tentative_pair(self) -> tuple[list[float], list[float]]:

        a = [1.0, 0.0, 0.0, 0.0]
        b_raw = [0.7, 0.7, 0.0, 0.0]
        norm_b = math.sqrt(sum(x * x for x in b_raw))
        b = [x / norm_b for x in b_raw]
        return a, b

    def test_tentative_enroll_upgrades_anonymous_profile(self, tmp_path):
        """The split-fix regression: an anonymous Speaker{N} followed by a
        tentative-confidence self-introduction MUST return the SAME Speaker{N}
        and upgrade the profile (not mint a new id).

        This test MUST fail on pre-fix code (enroll mints a uuid at tentative
        confidence) and pass after.
        """
        store = self._make_tentative_store(tmp_path)
        emb_a, emb_b = self._make_tentative_pair()

        anon_id = store.register_anonymous(emb_a)
        assert anon_id == "speaker0"

        # emb_b is tentative against speaker0 (anonymous).  After the fix,
        # enroll should upgrade speaker0 in-place (not mint a new id).
        result_id = store.enroll("Alice", emb_b)
        assert result_id == "speaker0", (
            "enroll() with a tentative anonymous match must return the existing speaker{N}, "
            "not a new id — the anonymous→named identity split is the root-cause bug"
        )
        assert store.get_name("speaker0") == "Alice"
        assert store.is_anonymous("speaker0") is False
        # Upgrade in-place: no new profile created, counter did NOT advance again.
        assert store.profile_count == 1
        assert store._next_anon_index == 1

    def test_two_new_enrollments_get_sequential_speaker_ids(
        self, tmp_path, sample_embedding, different_embedding
    ):
        """Two new named enrollments (no prior match) get sequential speaker{N} ids."""
        store = SpeakerStore(tmp_path / "profiles.json", high_threshold=0.90, low_threshold=0.70)
        alice_id = store.enroll("Alice", sample_embedding)
        bob_id = store.enroll("Bob", different_embedding)
        assert alice_id == "speaker0"
        assert bob_id == "speaker1"
        assert alice_id != bob_id
        assert store.profile_count == 2

    def test_tentative_enroll_against_named_mints_new_speaker_id(self, tmp_path):
        """A tentative match against a NAMED profile mints a new Speaker{N}
        (protects the named centroid from ambiguous-voice contamination).

        This mirrors ``TestRegisterAnonymousTentative.
        test_tentative_against_named_allocates_new_speaker``
        but exercises enroll() instead of register_anonymous().
        """
        store = self._make_tentative_store(tmp_path)
        emb_a, emb_b = self._make_tentative_pair()

        alice_id = store.enroll("Alice", emb_a)
        assert alice_id == "speaker0"

        # emb_b is tentative against Alice's named profile — must mint a NEW id.
        bob_id = store.enroll("Bob", emb_b)
        assert bob_id is not None
        assert bob_id != alice_id
        assert bob_id == "speaker1"
        # Alice's profile must be unchanged.
        assert store.get_name(alice_id) == "Alice"
        assert store.is_anonymous(alice_id) is False
        assert store.profile_count == 2

    def test_named_rejection_still_holds_at_high_confidence(self, tmp_path, sample_embedding):
        """High-confidence match against a named profile still returns None (no change)."""
        store = SpeakerStore(tmp_path / "profiles.json", high_threshold=0.90, low_threshold=0.70)
        alice_id = store.enroll("Alice", sample_embedding)
        assert alice_id is not None
        # Same embedding — high-confidence → reject.
        result = store.enroll("Bob", sample_embedding)
        assert result is None
        assert store.profile_count == 1


# ---------------------------------------------------------------------------
# _mint_anon_speaker_id — lowercase guarantee
# ---------------------------------------------------------------------------


class TestMintAnonSpeakerId:
    """_mint_anon_speaker_id must return lowercase speaker{N}."""

    def test_mint_returns_lowercase(self, tmp_path):
        """First minted id is 'speaker0' (lowercase)."""
        store = SpeakerStore(tmp_path / "sp.json")
        with store._lock:
            sid = store._mint_anon_speaker_id()
        assert sid == "speaker0", f"Expected 'speaker0', got {sid!r}"

    def test_mint_increments_counter(self, tmp_path):
        """Successive mints produce 'speaker0', 'speaker1', …"""
        store = SpeakerStore(tmp_path / "sp.json")
        ids = []
        with store._lock:
            for _ in range(3):
                ids.append(store._mint_anon_speaker_id())
        assert ids == ["speaker0", "speaker1", "speaker2"], ids


# ---------------------------------------------------------------------------
# v5 and v6 store behaviour
# ---------------------------------------------------------------------------


class TestSpeakerStoreMigrationV5ToV6:
    """v5 store raises on load (migration retired); v6 loads without modification."""

    def test_v5_load_raises(self, tmp_path):
        """A v5 store (retired migration) raises ValueError on load."""
        import json

        store_path = tmp_path / "sp.json"
        emb = [1.0, 0.0, 0.0]
        v5_payload = {
            "version": 5,
            "next_anon_index": 2,
            "last_greeted": {"Speaker0": "2026-01-01T00:00:00+00:00"},
            "speakers": {
                "Speaker0": {
                    "name": "Tobias",
                    "embeddings": [emb],
                    "preferred_language": "en",
                    "enroll_method": "self_introduced",
                },
            },
        }
        store_path.write_text(json.dumps(v5_payload))

        with pytest.raises(ValueError, match="Unsupported speaker store version"):
            SpeakerStore(store_path)

    def test_v6_store_not_re_migrated(self, tmp_path):
        """A v6 store (already lowercase) is loaded without modification."""
        import json

        store_path = tmp_path / "sp.json"
        emb = [1.0, 0.0, 0.0]
        v6_payload = {
            "version": 6,
            "next_anon_index": 1,
            "last_greeted": {"speaker0": "2026-01-01T00:00:00+00:00"},
            "speakers": {
                "speaker0": {
                    "name": "Tobias",
                    "embeddings": [emb],
                    "preferred_language": "en",
                    "enroll_method": "self_introduced",
                }
            },
        }
        store_path.write_text(json.dumps(v6_payload))

        store = SpeakerStore(store_path)

        assert "speaker0" in store._profiles
        assert store._profiles["speaker0"]["name"] == "Tobias"


# ---------------------------------------------------------------------------
# Ingest safety-net: _normalize_extraction lowercases speaker-id tokens
# ---------------------------------------------------------------------------


class TestIngestSafetyNet:
    """_normalize_extraction lowercases speaker-id tokens at the ingest boundary.

    This is a scoped exception to the 'extraction only .strip()s' rule: ONLY
    tokens matching is_speaker_id() are lowercased; display names are untouched.
    """

    def test_entity_name_speaker0_lowercased(self):
        """Entity with name='Speaker0' is lowercased to 'speaker0'."""
        from paramem.graph.extractor import _normalize_extraction

        data = {
            "entities": [{"name": "Speaker0", "entity_type": "person"}],
            "relations": [],
        }
        result = _normalize_extraction(data)
        assert result["entities"][0]["name"] == "speaker0", (
            f"Ingest safety-net must lowercase 'Speaker0' → 'speaker0'; "
            f"got {result['entities'][0]['name']!r}"
        )

    def test_entity_name_display_not_lowercased(self):
        """Non-speaker-id entity names (display names) are NOT lowercased."""
        from paramem.graph.extractor import _normalize_extraction

        data = {
            "entities": [{"name": "Tobias Becker", "entity_type": "person"}],
            "relations": [],
        }
        result = _normalize_extraction(data)
        # Display name must be preserved verbatim (only .strip() applied).
        assert result["entities"][0]["name"] == "Tobias Becker"

    def test_relation_subject_speaker0_lowercased(self):
        """Relation subject='Speaker0' is lowercased to 'speaker0'."""
        from paramem.graph.extractor import _normalize_extraction

        data = {
            "entities": [],
            "relations": [
                {
                    "subject": "Speaker0",
                    "predicate": "works_at",
                    "object": "Acme",
                    "relation_type": "factual",
                    "confidence": 1.0,
                }
            ],
        }
        result = _normalize_extraction(data)
        assert result["relations"][0]["subject"] == "speaker0", (
            f"Ingest safety-net must lowercase subject 'Speaker0' → 'speaker0'; "
            f"got {result['relations'][0]['subject']!r}"
        )

    def test_relation_object_speaker_lowercased(self):
        """Relation object='Speaker1' is lowercased to 'speaker1'."""
        from paramem.graph.extractor import _normalize_extraction

        data = {
            "entities": [],
            "relations": [
                {
                    "subject": "Acme",
                    "predicate": "employs",
                    "object": "Speaker1",
                    "relation_type": "factual",
                    "confidence": 1.0,
                }
            ],
        }
        result = _normalize_extraction(data)
        assert result["relations"][0]["object"] == "speaker1"

    def test_already_lowercase_passthrough(self):
        """Already-lowercase speaker0 is a no-op (idempotent)."""
        from paramem.graph.extractor import _normalize_extraction

        data = {
            "entities": [{"name": "speaker0", "entity_type": "person"}],
            "relations": [
                {
                    "subject": "speaker0",
                    "predicate": "works_at",
                    "object": "Acme",
                    "relation_type": "factual",
                    "confidence": 1.0,
                }
            ],
        }
        result = _normalize_extraction(data)
        assert result["entities"][0]["name"] == "speaker0"
        assert result["relations"][0]["subject"] == "speaker0"


# ---------------------------------------------------------------------------
# Phase B — household_display_names (B5)
# ---------------------------------------------------------------------------


class TestHouseholdDisplayNames:
    """SpeakerStore.household_display_names() returns display names of
    non-anonymous profiles only.
    """

    def _make_store(self, tmp_path) -> SpeakerStore:
        return SpeakerStore(tmp_path / "profiles.json")

    def test_empty_store_returns_empty(self, tmp_path) -> None:
        store = self._make_store(tmp_path)
        assert store.household_display_names() == []

    def test_enrolled_name_returned(self, tmp_path) -> None:
        store = self._make_store(tmp_path)
        embedding = [0.1] * 192
        store.enroll("Alice", embedding)
        names = store.household_display_names()
        assert "Alice" in names

    def test_anonymous_excluded(self, tmp_path) -> None:
        """Anonymous profiles (enroll_method == 'anonymous_voice') are NOT returned."""
        store = self._make_store(tmp_path)
        embedding = [0.1] * 192
        anon_id = store.register_anonymous(embedding)
        names = store.household_display_names()
        # The anonymous speaker's id is stored as its name; must not appear.
        assert anon_id not in names
        assert names == []

    def test_mixed_returns_only_named(self, tmp_path) -> None:
        """Mix of enrolled + anonymous: only enrolled names returned."""
        store = self._make_store(tmp_path)
        embedding = [0.1] * 192
        store.enroll("Alice", embedding)
        embedding2 = [0.2] * 192
        store.register_anonymous(embedding2)
        names = store.household_display_names()
        assert "Alice" in names
        assert len(names) == 1

    def test_multiple_enrolled_all_returned(self, tmp_path) -> None:
        store = self._make_store(tmp_path)
        # Use orthogonal embeddings so the second enroll is not rejected as a
        # duplicate of the first (cosine similarity near 0 avoids the dedup gate).
        emb1 = [1.0] + [0.0] * 191
        emb2 = [0.0, 1.0] + [0.0] * 190
        store.enroll("Alice", emb1)
        store.enroll("Bob", emb2)
        names = store.household_display_names()
        assert set(names) == {"Alice", "Bob"}
