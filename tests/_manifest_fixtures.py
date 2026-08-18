"""Shared fixture builders for the manifest / slot / migration test suite.

``tests/adapters/test_manifest.py``, ``tests/adapters/test_slot.py``,
``tests/migrate/test_stamp_slot_manifests_v5.py``, ``tests/test_donor.py``
and ``tests/test_key_registry.py`` all need the same two shapes on disk: a
current (v5) :class:`~paramem.adapters.manifest.AdapterManifest` train
payload, and a prior-shape (schema_version=4, no ``payload`` field)
``meta.json`` dict for the migration script's raw reader.  One place for
both, rather than four near-identical inline copies.

The pre-v5 shape mirrors ``tests/backup/_slot_fixtures.py``'s
``_make_adapter_slot`` meta dict (schema_version=4, ``base_model`` /
``tokenizer`` / ``lora`` present, no ``payload`` field) -- the real prior
shape ``scripts/migrate/stamp_slot_manifests_v5.py`` exists to migrate away
from.
"""

from __future__ import annotations

import json
from pathlib import Path

from paramem.adapters.manifest import (
    AdapterManifest,
    BaseModelFingerprint,
    LoRAShape,
    PayloadFingerprint,
    TokenizerFingerprint,
)
from paramem.training.key_registry import KeyRegistry

BASE_MODEL_ID = "test-org/test-base-model"
BASE_MODEL_SHA = "abc123commit"
BASE_MODEL_HASH = "sha256:" + "a" * 64
TOKENIZER_NAME = "test-org/test-base-model"
TOKENIZER_MERGES_HASH = "b" * 64
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ("q_proj", "v_proj")

# The lora_shape dict shape donor.py's donor_topology_id/donor_store_dir
# consume -- kept in lockstep with the LoRAShape fixture above so a donor
# fixture slot's manifest.lora always matches the topology its store
# directory is keyed by.
LORA_SHAPE_DICT = {
    "r": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "target_modules": list(LORA_TARGET_MODULES),
}


def make_train_manifest(
    *,
    name: str = "episodic",
    registry_sha256: str = "",
    key_count: "int | str" = 0,
    payload_sha256: str = "",
    window_stamp: str = "",
    trained_at: str = "2026-01-01T00:00:00Z",
    synthesized: bool = False,
) -> AdapterManifest:
    """A current-schema ``train``-payload manifest with fixed fingerprints."""
    return AdapterManifest(
        schema_version=5,
        name=name,
        trained_at=trained_at,
        payload=PayloadFingerprint(kind="train", sha256=payload_sha256),
        registry_sha256=registry_sha256,
        key_count=key_count,
        base_model=BaseModelFingerprint(
            repo=BASE_MODEL_ID, sha=BASE_MODEL_SHA, hash=BASE_MODEL_HASH
        ),
        tokenizer=TokenizerFingerprint(
            name_or_path=TOKENIZER_NAME, vocab_size=32000, merges_hash=TOKENIZER_MERGES_HASH
        ),
        lora=LoRAShape(
            rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.0, target_modules=LORA_TARGET_MODULES
        ),
        synthesized=synthesized,
        window_stamp=window_stamp,
    )


def v4_train_meta_dict(
    *,
    name: str = "episodic",
    registry_sha256: str = "",
    key_count: "int | str" = 0,
    synthesized: bool = False,
    window_stamp: str = "",
    trained_at: str = "2026-01-01T00:00:00Z",
) -> dict:
    """A prior-shape (schema_version=4, no ``payload`` field) train
    ``meta.json`` dict -- the shape ``read_manifest`` refuses outright and
    the migration script's raw ``json.loads`` reader is exercised against."""
    return {
        "schema_version": 4,
        "name": name,
        "trained_at": trained_at,
        "window_stamp": window_stamp,
        "base_model": {"repo": BASE_MODEL_ID, "sha": BASE_MODEL_SHA, "hash": BASE_MODEL_HASH},
        "tokenizer": {
            "name_or_path": TOKENIZER_NAME,
            "vocab_size": 32000,
            "merges_hash": TOKENIZER_MERGES_HASH,
        },
        "lora": {
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": 0.0,
            "target_modules": list(LORA_TARGET_MODULES),
        },
        "registry_sha256": registry_sha256,
        "key_count": key_count,
        "synthesized": synthesized,
    }


def write_slot_files(
    slot: Path,
    *,
    weight_bytes: bytes = b"fake-plaintext-weights",
    write_config: bool = True,
) -> None:
    """Write the train-payload weight file (+ optional adapter_config.json)
    directly into *slot* -- plain bytes, no PEFT/model dependency."""
    slot.mkdir(parents=True, exist_ok=True)
    (slot / "adapter_model.safetensors").write_bytes(weight_bytes)
    if write_config:
        (slot / "adapter_config.json").write_bytes(b'{"peft_type": "LORA"}')


def write_raw_meta(slot: Path, meta: dict) -> None:
    """Write *meta* verbatim as ``<slot>/meta.json`` -- bypasses
    ``write_manifest``, which only accepts an :class:`AdapterManifest`."""
    slot.mkdir(parents=True, exist_ok=True)
    (slot / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def canonical_registry_bytes(keys: list[str]) -> bytes:
    """The exact bytes :meth:`KeyRegistry.save_bytes` produces for *keys*."""
    reg = KeyRegistry()
    for k in keys:
        reg.add(k)
    return reg.save_bytes()


def write_canonical_registry(tier_root: Path, keys: list[str]) -> bytes:
    """Write an already-canonical registry (bytes identical to what
    :meth:`KeyRegistry.save_bytes` produces) at *tier_root* and return the
    bytes written."""
    payload = canonical_registry_bytes(keys)
    tier_root.mkdir(parents=True, exist_ok=True)
    (tier_root / "indexed_key_registry.json").write_bytes(payload)
    return payload


def write_pre_change_stale_registry(
    tier_root: Path,
    *,
    active_keys: list[str],
    stale_records: dict[str, dict],
    simhash: "dict[str, int] | None" = None,
) -> bytes:
    """Write a pre-change-shape registry at *tier_root*: a ``"stale"`` section
    that is a dict of per-id records (each optionally carrying a
    ``stale_since`` timestamp and/or a withheld-id fingerprint), rather than
    the current bare sorted-id-list shape.

    ``KeyRegistry.load``/``load_from_bytes`` refuse this shape outright — it
    is exactly the input ``scripts/migrate/stamp_slot_manifests_v5.py``'s
    ``_migrate_registry_bytes`` reads directly as raw JSON to rewrite into
    the current shape. *simhash* is the top-level ``"simhash"`` map (active
    keys' fingerprints, plus any duplicate a caller wants on a withheld id to
    exercise the "'simhash'-section duplicate" migration case).
    """
    data = {
        "active_keys": list(active_keys),
        "stale": dict(stale_records),
        "simhash": dict(simhash or {}),
    }
    payload = json.dumps(data, indent=2).encode("utf-8")
    tier_root.mkdir(parents=True, exist_ok=True)
    (tier_root / "indexed_key_registry.json").write_bytes(payload)
    return payload


def write_noncanonical_registry(tier_root: Path, keys: list[str]) -> bytes:
    """Write a KeyRegistry-shaped but NON-canonically-formatted registry
    (compact, no indentation) at *tier_root*.

    ``KeyRegistry.load_from_bytes`` requires a list-valued ``active_keys``,
    a list-valued ``stale`` of string ids, and a dict-valued ``simhash`` --
    all three present here (``stale`` empty, since this fixture has no
    withheld ids) -- so this parses cleanly but re-serializes to different
    bytes via :meth:`KeyRegistry.save_bytes` (``indent=2``) -- the
    pre-migration on-disk shape a genuine re-serialization pass changes,
    deliberately distinct from :func:`write_canonical_registry`'s
    already-final bytes.
    """
    data = {"active_keys": list(keys), "stale": [], "simhash": {}}
    payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
    tier_root.mkdir(parents=True, exist_ok=True)
    (tier_root / "indexed_key_registry.json").write_bytes(payload)
    return payload
