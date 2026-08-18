"""Tests for paramem.adapters.manifest -- the v5 AdapterManifest record.

Covers the v5 record's design: the kind<->fingerprint invariant (a
``train`` payload carries base_model/tokenizer/lora, any other kind
carries none of them -- checked both ways), the closed ``payload.kind``
vocabulary, and the current-schema-only read boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from paramem.adapters.manifest import (
    MANIFEST_SCHEMA_VERSION,
    AdapterManifest,
    BaseModelFingerprint,
    LoRAShape,
    ManifestSchemaError,
    PayloadFingerprint,
    TokenizerFingerprint,
    _lookup_hash_from_manifests,
    graph_payload_manifest,
    read_manifest,
    write_manifest,
)
from tests._manifest_fixtures import (
    BASE_MODEL_HASH,
    BASE_MODEL_ID,
    BASE_MODEL_SHA,
    make_train_manifest,
)


class TestPayloadKindFingerprintInvariant:
    def test_weight_payload_manifest_carries_base_tokenizer_and_lora_fingerprints(self):
        manifest = make_train_manifest(payload_sha256="c" * 64)

        assert manifest.payload.kind == "train"
        assert manifest.base_model == BaseModelFingerprint(
            repo=BASE_MODEL_ID, sha=BASE_MODEL_SHA, hash=BASE_MODEL_HASH
        )
        assert manifest.tokenizer is not None
        assert manifest.lora is not None

    def test_graph_payload_manifest_omits_weight_fingerprints(self):
        manifest = graph_payload_manifest(
            name="semantic", key_count=5, registry_sha256="deadbeef", window_stamp=""
        )

        assert manifest.payload.kind == "simulate"
        assert manifest.base_model is None
        assert manifest.tokenizer is None
        assert manifest.lora is None

    def test_manifest_rejects_weight_fingerprints_on_a_graph_payload(self):
        """A simulate payload carrying all three weight fingerprints violates
        the kind<->fingerprint invariant just as much as a train payload
        missing one -- __post_init__ enforces both directions."""
        with pytest.raises(ManifestSchemaError):
            AdapterManifest(
                schema_version=MANIFEST_SCHEMA_VERSION,
                name="semantic",
                trained_at="2026-01-01T00:00:00Z",
                payload=PayloadFingerprint(kind="simulate", sha256="a" * 64),
                registry_sha256="",
                key_count=0,
                base_model=BaseModelFingerprint(
                    repo=BASE_MODEL_ID, sha=BASE_MODEL_SHA, hash=BASE_MODEL_HASH
                ),
                tokenizer=TokenizerFingerprint(
                    name_or_path=BASE_MODEL_ID, vocab_size=32000, merges_hash="b" * 64
                ),
                lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=("q_proj",)),
            )

    def test_manifest_rejects_a_simulate_payload_with_a_partial_fingerprint_set(self):
        """A simulate payload carrying ONE of the three fingerprints (the
        other two None) must be refused just as loudly as all three present
        -- the invariant is "all None", not "not all three non-None"."""
        with pytest.raises(ManifestSchemaError):
            AdapterManifest(
                schema_version=MANIFEST_SCHEMA_VERSION,
                name="semantic",
                trained_at="2026-01-01T00:00:00Z",
                payload=PayloadFingerprint(kind="simulate", sha256="a" * 64),
                registry_sha256="",
                key_count=0,
                base_model=BaseModelFingerprint(
                    repo=BASE_MODEL_ID, sha=BASE_MODEL_SHA, hash=BASE_MODEL_HASH
                ),
                tokenizer=None,
                lora=None,
            )

    def test_manifest_rejects_a_weight_payload_missing_its_fingerprints(self):
        """A train payload missing even one of the three fingerprints
        (tokenizer, here) must be refused -- the invariant is all-or-nothing,
        not per-field optional."""
        with pytest.raises(ManifestSchemaError):
            AdapterManifest(
                schema_version=MANIFEST_SCHEMA_VERSION,
                name="episodic",
                trained_at="2026-01-01T00:00:00Z",
                payload=PayloadFingerprint(kind="train", sha256="a" * 64),
                registry_sha256="",
                key_count=0,
                base_model=BaseModelFingerprint(
                    repo=BASE_MODEL_ID, sha=BASE_MODEL_SHA, hash=BASE_MODEL_HASH
                ),
                tokenizer=None,
                lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=("q_proj",)),
            )


class TestSimulatePartialFingerprintRefusedAtParse:
    def test_parse_rejects_a_simulate_payload_dict_carrying_one_fingerprint_block(self, tmp_path):
        """The construction-side partial-fingerprint refusal
        (TestPayloadKindFingerprintInvariant) also fires from the read
        boundary -- a hand-edited or drifted on-disk dict carrying a
        simulate payload with one non-null fingerprint block must not
        silently parse."""
        slot = tmp_path / "slot"
        slot.mkdir()
        raw = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "name": "semantic",
            "trained_at": "2026-01-01T00:00:00Z",
            "window_stamp": "",
            "payload": {"kind": "simulate", "sha256": "a" * 64},
            "base_model": {"repo": BASE_MODEL_ID, "sha": BASE_MODEL_SHA, "hash": BASE_MODEL_HASH},
            "registry_sha256": "",
            "key_count": 0,
        }
        (slot / "meta.json").write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(ManifestSchemaError):
            read_manifest(slot)


class TestPayloadKindClosedVocabulary:
    def test_construction_rejects_an_out_of_vocabulary_kind(self):
        with pytest.raises(ManifestSchemaError):
            PayloadFingerprint(kind="bogus", sha256="a" * 64)

    def test_parse_rejects_an_out_of_vocabulary_kind(self, tmp_path):
        slot = tmp_path / "slot"
        slot.mkdir()
        raw = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "name": "episodic",
            "trained_at": "2026-01-01T00:00:00Z",
            "window_stamp": "",
            "payload": {"kind": "bogus", "sha256": "a" * 64},
            "registry_sha256": "",
            "key_count": 0,
        }
        (slot / "meta.json").write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(ManifestSchemaError):
            read_manifest(slot)


class TestWronglyTypedSubObjectsRaiseSchemaErrorNotTypeError:
    """A JSON-valid but wrongly-typed sub-object (e.g. ``"payload": 5``)
    must raise :class:`ManifestSchemaError` at the parse boundary, not a
    bare ``TypeError`` from an ``in`` check against a non-container -- a
    bare ``TypeError`` is outside every caller's ``ManifestSchemaError``
    catch (``find_live_slot``, ``verify_tier_binding`` steps 5/6) and would
    crash the whole tree walk for one malformed manifest."""

    def test_a_scalar_payload_raises_manifest_schema_error(self, tmp_path):
        slot = tmp_path / "slot"
        slot.mkdir()
        raw = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "name": "episodic",
            "trained_at": "2026-01-01T00:00:00Z",
            "window_stamp": "",
            "payload": 5,
            "registry_sha256": "",
            "key_count": 0,
        }
        (slot / "meta.json").write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(ManifestSchemaError):
            read_manifest(slot)

    def test_a_scalar_base_model_raises_manifest_schema_error(self, tmp_path):
        slot = tmp_path / "slot"
        slot.mkdir()
        raw = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "name": "episodic",
            "trained_at": "2026-01-01T00:00:00Z",
            "window_stamp": "",
            "payload": {"kind": "train", "sha256": "a" * 64},
            "base_model": 5,
            "tokenizer": {"name_or_path": "x", "vocab_size": 1, "merges_hash": "b" * 64},
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": []},
            "registry_sha256": "",
            "key_count": 0,
        }
        (slot / "meta.json").write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(ManifestSchemaError):
            read_manifest(slot)

    def test_a_scalar_tokenizer_raises_manifest_schema_error(self, tmp_path):
        slot = tmp_path / "slot"
        slot.mkdir()
        raw = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "name": "episodic",
            "trained_at": "2026-01-01T00:00:00Z",
            "window_stamp": "",
            "payload": {"kind": "train", "sha256": "a" * 64},
            "base_model": {"repo": BASE_MODEL_ID, "sha": BASE_MODEL_SHA, "hash": BASE_MODEL_HASH},
            "tokenizer": ["not", "a", "dict"],
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": []},
            "registry_sha256": "",
            "key_count": 0,
        }
        (slot / "meta.json").write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(ManifestSchemaError):
            read_manifest(slot)

    def test_a_scalar_lora_raises_manifest_schema_error(self, tmp_path):
        slot = tmp_path / "slot"
        slot.mkdir()
        raw = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "name": "episodic",
            "trained_at": "2026-01-01T00:00:00Z",
            "window_stamp": "",
            "payload": {"kind": "train", "sha256": "a" * 64},
            "base_model": {"repo": BASE_MODEL_ID, "sha": BASE_MODEL_SHA, "hash": BASE_MODEL_HASH},
            "tokenizer": {"name_or_path": "x", "vocab_size": 1, "merges_hash": "b" * 64},
            "lora": "not a dict either",
            "registry_sha256": "",
            "key_count": 0,
        }
        (slot / "meta.json").write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(ManifestSchemaError):
            read_manifest(slot)


class TestSimulateManifestOmitsFingerprintKeysOnDisk:
    def test_simulate_manifest_serializes_without_base_model_tokenizer_lora_keys(self, tmp_path):
        manifest = graph_payload_manifest(
            name="semantic", key_count=3, registry_sha256="", window_stamp=""
        )
        manifest = replace(manifest, payload=replace(manifest.payload, sha256="a" * 64))
        slot = tmp_path / "slot"
        slot.mkdir()

        write_manifest(slot, manifest)

        raw = json.loads((slot / "meta.json").read_text())
        assert "base_model" not in raw
        assert "tokenizer" not in raw
        assert "lora" not in raw


class TestGraphPayloadRoundTrip:
    def test_graph_payload_manifest_round_trips_through_json(self, tmp_path):
        manifest = graph_payload_manifest(
            name="semantic", key_count=7, registry_sha256="a" * 64, window_stamp="20260101T0000"
        )
        manifest = replace(manifest, payload=replace(manifest.payload, sha256="f" * 64))
        slot = tmp_path / "slot"
        slot.mkdir()

        write_manifest(slot, manifest)
        read_back = read_manifest(slot)

        assert read_back == manifest


class TestSchemaVersionGate:
    def test_a_manifest_of_any_other_schema_version_is_refused(self, tmp_path):
        manifest = make_train_manifest(payload_sha256="d" * 64)
        slot = tmp_path / "slot"
        slot.mkdir()

        older = replace(manifest, schema_version=MANIFEST_SCHEMA_VERSION - 1)
        write_manifest(slot, older)
        with pytest.raises(ManifestSchemaError):
            read_manifest(slot)

        newer = replace(manifest, schema_version=MANIFEST_SCHEMA_VERSION + 1)
        write_manifest(slot, newer)
        with pytest.raises(ManifestSchemaError):
            read_manifest(slot)

    def test_a_read_manifest_is_written_back_at_the_current_schema_version(self, tmp_path):
        """A manifest read off disk, then written back unchanged in shape
        (the restamp write-back shape at ``persistence.py``'s
        ``replace(read_manifest(slot), ...)``), must still carry
        ``schema_version == MANIFEST_SCHEMA_VERSION`` on the re-read file --
        never a stale or pass-through value."""
        manifest = graph_payload_manifest(
            name="semantic", key_count=3, registry_sha256="a" * 64, window_stamp=""
        )
        manifest = replace(manifest, payload=replace(manifest.payload, sha256="e" * 64))
        slot = tmp_path / "slot"
        slot.mkdir()
        write_manifest(slot, manifest)

        read_back = read_manifest(slot)
        assert read_back.schema_version == MANIFEST_SCHEMA_VERSION

        # Write-back, as a restamp would (only a data field changes).
        write_manifest(slot, replace(read_back, key_count=9))

        on_disk = json.loads((slot / "meta.json").read_text())
        assert on_disk["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert read_manifest(slot).schema_version == MANIFEST_SCHEMA_VERSION


class TestReadManifestDecodeFailure:
    """A ``meta.json`` present but not valid UTF-8 (binary garbage) must
    raise the same documented :class:`ManifestSchemaError` every other
    present-but-unparseable shape raises -- never an uncaught
    ``UnicodeDecodeError`` escaping the documented contract."""

    def test_non_utf8_meta_json_raises_manifest_schema_error(self, tmp_path):
        slot = tmp_path / "slot"
        slot.mkdir()
        (slot / "meta.json").write_bytes(b"\xff\xfe\x00garbage-not-utf8\x80\x81")

        with pytest.raises(ManifestSchemaError):
            read_manifest(slot)


class TestPayloadDigestSurvivesReEncryption:
    def test_payload_digest_survives_a_re_encryption_of_the_payload_file(
        self, tmp_path, monkeypatch
    ):
        from pyrage import x25519

        from paramem.backup.encryption import envelope_encrypt_bytes
        from paramem.backup.hashing import plaintext_sha256
        from paramem.backup.key_store import (
            DAILY_PASSPHRASE_ENV_VAR,
            _clear_daily_identity_cache,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
            write_recovery_pub_file,
        )

        daily_a = mint_daily_identity()
        recovery = x25519.Identity.generate()
        daily_path = tmp_path / "daily_key.age"
        recovery_path = tmp_path / "recovery.pub"
        write_daily_key_file(wrap_daily_identity(daily_a, "pw"), daily_path)
        write_recovery_pub_file(recovery.to_public(), recovery_path)
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")
        monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", daily_path)
        monkeypatch.setattr("paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT", recovery_path)

        payload_path = tmp_path / "graph.json"
        plaintext = b'{"nodes": ["a", "b"]}'
        payload_path.write_bytes(envelope_encrypt_bytes(plaintext))

        before = plaintext_sha256(payload_path)
        assert before == hashlib.sha256(plaintext).hexdigest()

        # Simulate `rotate-daily`: decrypt the SAME plaintext under a fresh
        # daily identity and re-encrypt in place.
        daily_b = mint_daily_identity()
        write_daily_key_file(wrap_daily_identity(daily_b, "pw"), daily_path)
        _clear_daily_identity_cache()
        payload_path.write_bytes(envelope_encrypt_bytes(plaintext))

        after = plaintext_sha256(payload_path)
        assert after == before, (
            "the plaintext-content digest must survive re-encryption under a new daily key"
        )


class TestBaseHashLookupSkipsGraphManifests:
    def test_base_hash_lookup_skips_graph_manifests_on_a_mixed_tree(self, tmp_path):
        adapter_root = tmp_path / "adapters"
        train_slot = adapter_root / "episodic" / "20260101-000000"
        graph_slot = adapter_root / "semantic" / "20260101-000000"
        train_slot.mkdir(parents=True)
        graph_slot.mkdir(parents=True)

        train_manifest = make_train_manifest(
            name="episodic", payload_sha256="1" * 64, trained_at="2026-01-01T00:00:00Z"
        )
        write_manifest(train_slot, train_manifest)

        graph_manifest = graph_payload_manifest(
            name="semantic", key_count=3, registry_sha256="", window_stamp=""
        )
        graph_manifest = replace(
            graph_manifest, payload=replace(graph_manifest.payload, sha256="2" * 64)
        )
        write_manifest(graph_slot, graph_manifest)

        result = _lookup_hash_from_manifests(adapter_root, BASE_MODEL_ID, BASE_MODEL_SHA)

        assert result == BASE_MODEL_HASH
