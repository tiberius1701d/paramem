"""Round-trip tests for the core infrastructure artifacts.

Each artifact below gets exercised twice: once without a master key
(plaintext on disk, backward-compatible) and once with a master key set
(PMEM1-wrapped ciphertext on disk).  Verifies:

- Write succeeds in both modes.
- Disk bytes carry the PMEM1 magic only when a key is set.
- Reader round-trips to the original content regardless of mode.
- Plaintext and ciphertext differ on disk when a key is set.

Covers:
- Post-session queue (``PostSessionQueue``)
- Speaker store (``SpeakerStore``)
- BG-trainer resume state (``_write_resume_state_atomic`` / ``_read_resume_state``)
- Knowledge graph (``GraphMerger.save_graph`` / ``load_graph``)
- Registry writer / loader (``save_registry`` / ``load_registry`` in
  ``paramem.training.indexed_memory``)
- The ``registry_sha256`` stability invariant — the hash of the registry
  must be identical in plaintext and ciphertext modes (it is the hash of
  the *content*, not of the on-disk bytes).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from paramem.backup.encryption import (
    MASTER_KEY_ENV_VAR,
    PMEM1_MAGIC,
    _clear_cipher_cache,
    is_pmem1_envelope,
    read_maybe_encrypted,
)


def _make_key() -> str:
    return Fernet.generate_key().decode()


def _clean_env() -> None:
    os.environ.pop(MASTER_KEY_ENV_VAR, None)
    _clear_cipher_cache()


@pytest.fixture(autouse=True)
def _env_isolation():
    """Isolate the master-key env var and cipher cache per test."""
    _clean_env()
    yield
    _clean_env()


# ---------------------------------------------------------------------------
# PostSessionQueue
# ---------------------------------------------------------------------------


class TestQueueRoundtrip:
    def _entry(self, conv_id: str) -> dict:
        return {
            "session_id": conv_id,
            "speaker_id": "spk-1",
            "speaker_name": "Alex",
            "transcript": "hello",
            "enqueued_at": "2026-04-23T00:00:00Z",
        }

    def test_plaintext_without_key(self, tmp_path):
        from paramem.server.post_session_queue import PostSessionQueue

        path = tmp_path / "queue.json"
        q = PostSessionQueue(path)
        q.enqueue(self._entry("conv-a"))

        assert not is_pmem1_envelope(path)
        # Plaintext JSON is directly loadable.
        assert isinstance(json.loads(path.read_text()), list)

    def test_encrypted_with_key(self, tmp_path):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()

        from paramem.server.post_session_queue import PostSessionQueue

        path = tmp_path / "queue.json"
        q = PostSessionQueue(path)
        q.enqueue(self._entry("conv-a"))

        # On-disk carries the envelope.
        assert is_pmem1_envelope(path)
        raw = path.read_bytes()
        assert raw.startswith(PMEM1_MAGIC)
        assert b"conv-a" not in raw  # plaintext must not leak

        # Reconstructing a queue from the same path transparently decrypts.
        q2 = PostSessionQueue(path)
        entries = q2.peek()
        assert len(entries) == 1
        assert entries[0]["session_id"] == "conv-a"


# ---------------------------------------------------------------------------
# SpeakerStore
# ---------------------------------------------------------------------------


class TestSpeakerStoreRoundtrip:
    def test_plaintext_without_key(self, tmp_path):
        from paramem.server.speaker import SpeakerStore

        store_path = tmp_path / "speakers.json"
        s = SpeakerStore(store_path=store_path)
        s.enroll(name="Alex", embedding=[0.1] * 192)

        assert not is_pmem1_envelope(store_path)
        # Plaintext layout survives a reload.
        s2 = SpeakerStore(store_path=store_path)
        assert s2.profile_count == 1
        assert "Alex" in s2.speaker_names

    def test_encrypted_with_key(self, tmp_path):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()

        from paramem.server.speaker import SpeakerStore

        store_path = tmp_path / "speakers.json"
        s = SpeakerStore(store_path=store_path)
        s.enroll(name="Alex", embedding=[0.1] * 192)

        assert is_pmem1_envelope(store_path)
        raw = store_path.read_bytes()
        assert raw.startswith(PMEM1_MAGIC)
        assert b"Alex" not in raw  # name must not leak in ciphertext

        # Reload reads through the envelope.
        s2 = SpeakerStore(store_path=store_path)
        assert "Alex" in s2.speaker_names


# ---------------------------------------------------------------------------
# BG-trainer resume state
# ---------------------------------------------------------------------------


class TestResumeStateRoundtrip:
    def test_plaintext_without_key(self, tmp_path):
        from paramem.server.background_trainer import (
            _read_resume_state,
            _write_resume_state_atomic,
        )

        path = tmp_path / "resume_state.json"
        state = {"last_completed_epoch": 5, "checkpoint": "/tmp/ckpt-5"}
        _write_resume_state_atomic(path, state)

        assert not is_pmem1_envelope(path)
        assert _read_resume_state(path) == state

    def test_encrypted_with_key(self, tmp_path):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()

        from paramem.server.background_trainer import (
            _read_resume_state,
            _write_resume_state_atomic,
        )

        path = tmp_path / "resume_state.json"
        state = {"last_completed_epoch": 7, "checkpoint": "/tmp/ckpt-7"}
        _write_resume_state_atomic(path, state)

        assert is_pmem1_envelope(path)
        assert path.read_bytes().startswith(PMEM1_MAGIC)
        assert _read_resume_state(path) == state


# ---------------------------------------------------------------------------
# Graph merger
# ---------------------------------------------------------------------------


class TestGraphRoundtrip:
    def _merger_with_data(self):
        from paramem.graph.merger import GraphMerger

        merger = GraphMerger()
        # Add a simple triple via the graph's NetworkX surface.
        merger.graph.add_node("Alice")
        merger.graph.add_node("Seattle")
        merger.graph.add_edge("Alice", "Seattle", predicate="lives_in")
        return merger

    def test_plaintext_without_key(self, tmp_path):
        merger = self._merger_with_data()
        path = tmp_path / "graph.json"
        merger.save_graph(path)

        assert not is_pmem1_envelope(path)

        from paramem.graph.merger import GraphMerger

        other = GraphMerger()
        other.load_graph(path)
        assert other.graph.number_of_nodes() == 2
        assert other.graph.has_edge("Alice", "Seattle")

    def test_encrypted_with_key(self, tmp_path):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()

        merger = self._merger_with_data()
        path = tmp_path / "graph.json"
        merger.save_graph(path)

        assert is_pmem1_envelope(path)
        raw = path.read_bytes()
        assert raw.startswith(PMEM1_MAGIC)
        assert b"Alice" not in raw  # entity name must not leak

        from paramem.graph.merger import GraphMerger

        other = GraphMerger()
        other.load_graph(path)
        assert other.graph.has_edge("Alice", "Seattle")


# ---------------------------------------------------------------------------
# Registry writer / loader
# ---------------------------------------------------------------------------


class TestRegistryRoundtrip:
    def test_plaintext_without_key(self, tmp_path):
        from paramem.training.indexed_memory import load_registry, save_registry

        path = tmp_path / "registry.json"
        data = {"graph1": "abc", "graph2": "def"}
        save_registry(data, path)

        assert not is_pmem1_envelope(path)
        assert load_registry(path) == data

    def test_encrypted_with_key(self, tmp_path):
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()

        from paramem.training.indexed_memory import load_registry, save_registry

        path = tmp_path / "registry.json"
        data = {"graph1": "abc", "graph2": "def"}
        save_registry(data, path)

        assert is_pmem1_envelope(path)
        assert path.read_bytes().startswith(PMEM1_MAGIC)
        assert load_registry(path) == data


# ---------------------------------------------------------------------------
# registry_sha256 stability — hash of content, not of on-disk bytes
# ---------------------------------------------------------------------------


class TestRegistrySha256Stability:
    """The registry fingerprint used for adapter-manifest live-slot matching
    must be stable across plaintext ↔ ciphertext transitions.  Fernet rewrites
    the ciphertext with a fresh IV on every encryption, so a naive hash of
    on-disk bytes would produce a different fingerprint each time and break
    drift detection.
    """

    def _hash_content(self, path: Path) -> str:
        return hashlib.sha256(read_maybe_encrypted(path)).hexdigest()

    def test_same_content_same_hash_in_both_modes(self, tmp_path):
        from paramem.training.indexed_memory import save_registry

        data = {"graph1": "abc"}
        path_pt = tmp_path / "plain.json"
        save_registry(data, path_pt)
        pt_hash = self._hash_content(path_pt)

        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        path_ct = tmp_path / "encrypted.json"
        save_registry(data, path_ct)
        ct_hash = self._hash_content(path_ct)

        assert pt_hash == ct_hash, "content hash must not depend on on-disk encoding"

    def test_hash_stable_across_two_writes_with_same_key(self, tmp_path):
        """Re-encrypting the same content under the same key produces the
        same content hash even though the ciphertext differs (fresh IV).
        """
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        from paramem.training.indexed_memory import save_registry

        data = {"graph1": "abc"}
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        save_registry(data, path_a)
        save_registry(data, path_b)

        # On-disk ciphertexts differ (different IVs).
        assert path_a.read_bytes() != path_b.read_bytes()

        # Content hashes are identical.
        assert self._hash_content(path_a) == self._hash_content(path_b)

    def test_hash_changes_when_content_changes(self, tmp_path):
        """Sanity: a genuine content change must produce a different hash."""
        os.environ[MASTER_KEY_ENV_VAR] = _make_key()
        from paramem.training.indexed_memory import save_registry

        path = tmp_path / "r.json"
        save_registry({"graph1": "abc"}, path)
        h1 = self._hash_content(path)
        save_registry({"graph1": "def"}, path)
        h2 = self._hash_content(path)
        assert h1 != h2
