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
    assert len(speaker_id) == 8
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
    assert data["version"] == 5
    assert len(data["speakers"]) == 1
    profile = list(data["speakers"].values())[0]
    assert "embeddings" in profile
    assert len(profile["embeddings"]) == 1


def test_legacy_v1_migration(tmp_path, sample_embedding):
    """Legacy v1 name-keyed format is auto-migrated to v5."""
    path = tmp_path / "profiles.json"
    legacy_data = {"speakers": {"Alice": sample_embedding}}
    path.write_text(json.dumps(legacy_data))

    store = SpeakerStore(path)
    assert store.profile_count == 1
    assert store._next_anon_index == 0

    result = store.match(sample_embedding)
    assert result.name == "Alice"
    assert result.speaker_id is not None
    assert len(result.speaker_id) == 8

    data = json.loads(path.read_text())
    assert data["version"] == 5
    assert all("name" in v and "embeddings" in v for v in data["speakers"].values())


def test_legacy_v2_migration(tmp_path, sample_embedding):
    """v2 single-embedding format is auto-migrated to v5 multi-embedding."""
    path = tmp_path / "profiles.json"
    v2_data = {
        "speakers": {"abc12345": {"name": "Alice", "embedding": sample_embedding}},
        "version": 2,
    }
    path.write_text(json.dumps(v2_data))

    store = SpeakerStore(path)
    assert store.profile_count == 1
    assert store._next_anon_index == 0

    result = store.match(sample_embedding)
    assert result.name == "Alice"
    assert result.speaker_id == "abc12345"

    data = json.loads(path.read_text())
    assert data["version"] == 5
    profile = data["speakers"]["abc12345"]
    assert "embeddings" in profile
    assert "embedding" not in profile
    assert len(profile["embeddings"]) == 1


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


class TestGreetingText:
    def test_morning_text(self):
        assert SpeakerStore._greeting_text("morning") == "Good morning"

    def test_afternoon_text(self):
        assert SpeakerStore._greeting_text("afternoon") == "Good afternoon"

    def test_evening_text(self):
        assert SpeakerStore._greeting_text("evening") == "Good evening"


class TestShouldGreet:
    def test_disabled_when_interval_zero(self, tmp_path):
        store = SpeakerStore(tmp_path / "profiles.json")
        assert store.should_greet("speaker1", interval_hours=0) is None

    def test_first_greeting(self, tmp_path):
        store = SpeakerStore(tmp_path / "profiles.json")
        result = store.should_greet("speaker1", interval_hours=24)
        assert result is not None
        assert "Good" in result

    def test_no_repeat_within_period(self, tmp_path):
        store = SpeakerStore(tmp_path / "profiles.json")
        result1 = store.should_greet("speaker1", interval_hours=24)
        assert result1 is not None
        store.confirm_greeting("speaker1")
        result2 = store.should_greet("speaker1", interval_hours=24)
        assert result2 is None

    def test_different_speakers_independent(self, tmp_path):
        store = SpeakerStore(tmp_path / "profiles.json")
        store.should_greet("speaker1", interval_hours=24)
        store.confirm_greeting("speaker1")
        result = store.should_greet("speaker2", interval_hours=24)
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
# register_anonymous — anonymous speaker ID promotion (Slice 3-pre)
# ---------------------------------------------------------------------------


class TestRegisterAnonymous:
    """Tests for SpeakerStore.register_anonymous."""

    def test_first_registration_returns_speaker0(self, tmp_path, sample_embedding):
        """First unrecognized voice gets Speaker0."""
        store = SpeakerStore(tmp_path / "profiles.json")
        anon_id = store.register_anonymous(sample_embedding)
        assert anon_id == "Speaker0"

    def test_second_distinct_embedding_gets_speaker1(
        self, tmp_path, sample_embedding, different_embedding
    ):
        """Two distinct voices get incrementing IDs."""
        store = SpeakerStore(tmp_path / "profiles.json")
        id0 = store.register_anonymous(sample_embedding)
        id1 = store.register_anonymous(different_embedding)
        assert id0 == "Speaker0"
        assert id1 == "Speaker1"

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
        assert id_new == "Speaker2"

    def test_disk_has_version5_and_next_anon_index(self, tmp_path, sample_embedding):
        """JSON on disk has version: 5 and next_anon_index field after registration."""
        path = tmp_path / "profiles.json"
        store = SpeakerStore(path)
        store.register_anonymous(sample_embedding)
        data = json.loads(path.read_text())
        assert data["version"] == 5
        assert "next_anon_index" in data
        assert data["next_anon_index"] == 1

    def test_v4_migration_sets_next_anon_index_zero(self, tmp_path, sample_embedding):
        """Loading a v4-format file sets next_anon_index=0 and bumps version to 5."""
        path = tmp_path / "profiles.json"
        # Craft a valid v4 store.
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

        store = SpeakerStore(path)
        # Migration sets the counter to 0.
        assert store._next_anon_index == 0
        # Disk is rewritten as v5 immediately on load.
        data = json.loads(path.read_text())
        assert data["version"] == 5
        assert data.get("next_anon_index") == 0

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
        # No anonymous profile should have been created.
        assert store._next_anon_index == 0
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


# ---------------------------------------------------------------------------
# enroll() upgrade-in-place for anonymous profiles (Fix 1)
# ---------------------------------------------------------------------------


class TestEnrollUpgradesAnonymous:
    """enroll() must upgrade anonymous Speaker{N} profiles rather than reject."""

    def test_enroll_after_register_anonymous_upgrades_profile(self, tmp_path, sample_embedding):
        """After register_anonymous creates Speaker0, enrolling with the same voice
        and a real name returns Speaker0 (not a new UUID), upgrades the name, and
        sets enroll_method to the requested method.
        """
        store = SpeakerStore(tmp_path / "profiles.json")
        anon_id = store.register_anonymous(sample_embedding)
        assert anon_id == "Speaker0"

        result_id = store.enroll("Alice", sample_embedding)
        assert result_id == "Speaker0", "enroll() must return the existing anonymous ID"
        assert store.get_name("Speaker0") == "Alice"
        profiles = {p["id"]: p for p in store.list_profiles()}
        assert profiles["Speaker0"]["enroll_method"] == "self_introduced"
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
        import math

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
        assert id_a == "Speaker0"

        # emb_b is tentatively similar to emb_a / Speaker0 (anonymous).
        id_b = store.register_anonymous(emb_b)
        assert id_b == "Speaker0", (
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

        # Enroll emb_a as a named speaker.
        alice_id = store.enroll("Alice", emb_a)
        assert alice_id is not None

        # emb_b is tentative against Alice's profile — must not contaminate it.
        anon_id = store.register_anonymous(emb_b)
        assert anon_id != alice_id
        assert anon_id == "Speaker0"
        # Alice's profile must remain unchanged.
        assert store.get_name(alice_id) == "Alice"
        profiles = {p["id"]: p for p in store.list_profiles()}
        assert profiles[alice_id]["enroll_method"] != "anonymous_voice"
