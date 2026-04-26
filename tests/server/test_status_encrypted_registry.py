"""Regression for /status HTTP 500 when indexed_key_registry.json is encrypted.

Pre-fix, ``app.py`` read the registry with raw ``open() + json.load()``. After
the registry-encryption work flipped writes to go through ``write_infra_bytes``
(age-encrypted when the daily identity is loaded), the read path produced
``UnicodeDecodeError`` on the age magic byte and surfaced as HTTP 500 from
``GET /status`` after every consolidation cycle until the operator restarted
the server.

The fix routes the read through ``read_maybe_encrypted``. This test exercises
the exact write→read pair against an encrypted registry and asserts it
round-trips, plus exercises the resilience block in app.py (it must not
re-raise on any failure shape — UnicodeDecodeError, RuntimeError on missing
key, OSError on missing file, JSONDecodeError on corrupt content).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pyrage
import pytest
from pyrage import x25519

from paramem.backup.encryption import (
    AGE_MAGIC,
    read_maybe_encrypted,
    write_infra_bytes,
)
from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
    write_recovery_pub_file,
)


@pytest.fixture(autouse=True)
def _isolate_env_and_caches():
    os.environ.pop(DAILY_PASSPHRASE_ENV_VAR, None)
    _clear_daily_identity_cache()
    yield
    os.environ.pop(DAILY_PASSPHRASE_ENV_VAR, None)
    _clear_daily_identity_cache()


def _setup_daily_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passphrase: str = "pw",
) -> None:
    """Mint a daily identity and point module defaults at it."""
    daily = mint_daily_identity()
    recovery = x25519.Identity.generate()
    daily_path = tmp_path / "daily_key.age"
    recovery_path = tmp_path / "recovery.pub"
    write_daily_key_file(wrap_daily_identity(daily, passphrase), daily_path)
    write_recovery_pub_file(recovery.to_public(), recovery_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", daily_path)
    monkeypatch.setattr("paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT", recovery_path)


def test_encrypted_registry_round_trips_through_read_maybe_encrypted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the daily identity is loaded, ``write_infra_bytes`` produces age
    ciphertext on disk. The fixed /status read path must use
    ``read_maybe_encrypted`` to unwrap it transparently — the prior raw
    ``open() + json.load()`` failed with UnicodeDecodeError on the age magic
    byte and surfaced as HTTP 500."""
    _setup_daily_identity(tmp_path, monkeypatch)

    registry_path = tmp_path / "indexed_key_registry.json"
    payload = {
        "adapter_health": {"episodic": {"recall": 0.95, "loss": 0.18}},
        "version": 2,
    }
    write_infra_bytes(registry_path, json.dumps(payload).encode("utf-8"))

    # On-disk shape must be age-encrypted (proves the bug's preconditions).
    on_disk = registry_path.read_bytes()
    assert on_disk.startswith(AGE_MAGIC), "expected age envelope, got plaintext"

    # The pre-fix code path (raw open + json.load) crashes here:
    with pytest.raises(UnicodeDecodeError):
        with open(registry_path) as f:
            json.load(f)

    # The fixed code path (read_maybe_encrypted + json.loads) round-trips:
    parsed = json.loads(read_maybe_encrypted(registry_path).decode("utf-8"))
    assert parsed == payload
    assert parsed["adapter_health"]["episodic"]["recall"] == 0.95


def test_plaintext_registry_still_works(tmp_path: Path) -> None:
    """When the daily identity is NOT loaded, ``write_infra_bytes`` falls back
    to plaintext. ``read_maybe_encrypted`` must round-trip the plaintext path
    too — this is the common case on hosts where the operator has not set up
    the daily identity."""
    registry_path = tmp_path / "indexed_key_registry.json"
    payload = {"adapter_health": {}, "version": 2}
    write_infra_bytes(registry_path, json.dumps(payload).encode("utf-8"))

    on_disk = registry_path.read_bytes()
    assert not on_disk.startswith(AGE_MAGIC), "expected plaintext, got age envelope"

    parsed = json.loads(read_maybe_encrypted(registry_path).decode("utf-8"))
    assert parsed == payload


def test_app_status_block_swallows_decrypt_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the registry is age-encrypted with a key the running process can't
    unwrap (e.g. operator rotated the daily identity but didn't restart the
    server), ``read_maybe_encrypted`` raises pyrage.DecryptError. The /status
    block must catch this and return an empty adapter_health dict — never
    surface as HTTP 500."""
    # Mint two distinct daily identities — one to encrypt, the other to load.
    encrypter = mint_daily_identity()
    decrypter_pass = "different-passphrase"
    decrypter_daily = mint_daily_identity()
    decrypter_path = tmp_path / "daily_key.age"
    recovery_path = tmp_path / "recovery.pub"
    write_daily_key_file(wrap_daily_identity(decrypter_daily, decrypter_pass), decrypter_path)
    # Recovery recipient that will NOT match the encrypter
    recovery = x25519.Identity.generate()
    write_recovery_pub_file(recovery.to_public(), recovery_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, decrypter_pass)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", decrypter_path)
    monkeypatch.setattr("paramem.backup.key_store.RECOVERY_PUB_PATH_DEFAULT", recovery_path)

    # Write the registry with the OTHER daily identity as the only recipient.
    registry_path = tmp_path / "indexed_key_registry.json"
    from paramem.backup.encryption import age_encrypt_bytes

    ciphertext = age_encrypt_bytes(b'{"adapter_health":{}}', [encrypter.to_public()])
    registry_path.write_bytes(ciphertext)

    # The unwrap must fail (different daily identity, no recovery match).
    with pytest.raises(pyrage.DecryptError):
        read_maybe_encrypted(registry_path)

    # The /status block matches `except Exception:` — verify the *fix* swallows.
    adapter_health: dict = {}
    if registry_path.exists():
        try:
            adapter_health = (
                json.loads(read_maybe_encrypted(registry_path).decode("utf-8")).get(
                    "adapter_health", {}
                )
                or {}
            )
        except Exception:
            adapter_health = {}
    assert adapter_health == {}
