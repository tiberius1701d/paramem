"""Tests for the age-format envelope helpers (``pyrage`` backend).

Covers:

1. Bytes round-trip with a single recipient.
2. Multi-recipient bytes envelope — either identity decrypts independently.
3. File round-trip (streaming ``encrypt_io`` / ``decrypt_io``).
4. Atomic-write crash safety: ``<dst>.tmp`` is cleaned up on error and no
   partial file lands at the destination path.
5. Plugin-recipient refusal at every public entry point.
6. Tampered ciphertext surfaces :class:`pyrage.DecryptError`.
7. ``is_age_envelope`` correctly distinguishes age from arbitrary bytes.
8. Bech32 round-trip through :func:`identity_from_bech32` and
   :func:`recipient_from_bech32`.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pyrage
import pytest
from pyrage import x25519

from paramem.backup.age_envelope import (
    AGE_MAGIC,
    PluginRecipientRejected,
    age_decrypt_bytes,
    age_decrypt_file,
    age_encrypt_bytes,
    age_encrypt_file,
    identity_from_bech32,
    is_age_envelope,
    recipient_from_bech32,
)


def _mint_pair() -> tuple[x25519.Identity, x25519.Recipient]:
    ident = x25519.Identity.generate()
    return ident, ident.to_public()


class TestBytesRoundTrip:
    def test_single_recipient(self) -> None:
        ident, recip = _mint_pair()
        ct = age_encrypt_bytes(b"hello world", [recip])
        assert ct.startswith(AGE_MAGIC), "envelope must carry age v1 magic"
        assert age_decrypt_bytes(ct, [ident]) == b"hello world"

    def test_multi_recipient_either_decrypts(self) -> None:
        ident_a, recip_a = _mint_pair()
        ident_b, recip_b = _mint_pair()
        plaintext = b"shared" * 100
        ct = age_encrypt_bytes(plaintext, [recip_a, recip_b])
        assert age_decrypt_bytes(ct, [ident_a]) == plaintext
        assert age_decrypt_bytes(ct, [ident_b]) == plaintext

    def test_wrong_identity_raises_decrypt_error(self) -> None:
        ident, recip = _mint_pair()
        wrong_ident, _ = _mint_pair()
        ct = age_encrypt_bytes(b"secret", [recip])
        with pytest.raises(pyrage.DecryptError):
            age_decrypt_bytes(ct, [wrong_ident])

    def test_tampered_ciphertext_raises(self) -> None:
        ident, recip = _mint_pair()
        ct = age_encrypt_bytes(b"payload", [recip])
        # Flip bytes deep inside the envelope past the header.
        tampered = ct[:100] + bytes(len(ct) - 100)
        with pytest.raises(pyrage.DecryptError):
            age_decrypt_bytes(tampered, [ident])


class TestFileRoundTrip:
    def test_encrypt_decrypt_file(self, tmp_path: Path) -> None:
        ident, recip = _mint_pair()
        src = tmp_path / "plain.bin"
        mid = tmp_path / "cipher.age"
        dst = tmp_path / "roundtrip.bin"
        body = os.urandom(64 * 1024 + 17)  # crosses one age chunk boundary
        src.write_bytes(body)

        age_encrypt_file(src, mid, [recip])
        assert is_age_envelope(mid)

        age_decrypt_file(mid, dst, [ident])
        assert dst.read_bytes() == body

    def test_multi_recipient_file_round_trip(self, tmp_path: Path) -> None:
        ident_a, recip_a = _mint_pair()
        ident_b, recip_b = _mint_pair()
        src = tmp_path / "plain.bin"
        mid = tmp_path / "cipher.age"
        src.write_bytes(b"multi" * 1000)

        age_encrypt_file(src, mid, [recip_a, recip_b])
        # Decrypt twice — once per identity, into distinct outputs.
        dst_a = tmp_path / "out_a.bin"
        dst_b = tmp_path / "out_b.bin"
        age_decrypt_file(mid, dst_a, [ident_a])
        age_decrypt_file(mid, dst_b, [ident_b])
        assert dst_a.read_bytes() == src.read_bytes()
        assert dst_b.read_bytes() == src.read_bytes()


class TestAtomicWriteCrashSafety:
    def test_encrypt_file_crash_leaves_no_partial(self, tmp_path: Path) -> None:
        """If encrypt_io raises mid-write, <dst>.tmp is cleaned and dst is absent."""
        _, recip = _mint_pair()
        src = tmp_path / "plain.bin"
        dst = tmp_path / "cipher.age"
        src.write_bytes(b"payload" * 100)

        with patch("pyrage.encrypt_io", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                age_encrypt_file(src, dst, [recip])

        assert not dst.exists(), "dst must not be created on failure"
        assert not (tmp_path / "cipher.age.tmp").exists(), "tmp must be cleaned"

    def test_decrypt_file_crash_leaves_no_partial(self, tmp_path: Path) -> None:
        ident, recip = _mint_pair()
        src = tmp_path / "plain.bin"
        mid = tmp_path / "cipher.age"
        dst = tmp_path / "plain2.bin"
        src.write_bytes(b"payload")

        age_encrypt_file(src, mid, [recip])

        with patch("pyrage.decrypt_io", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                age_decrypt_file(mid, dst, [ident])

        assert not dst.exists()
        assert not (tmp_path / "plain2.bin.tmp").exists()


class TestPluginRecipientRefusal:
    def test_encrypt_bytes_refuses_non_x25519_recipient(self) -> None:
        with pytest.raises(PluginRecipientRejected):
            age_encrypt_bytes(b"x", [object()])  # type: ignore[list-item]

    def test_decrypt_bytes_refuses_non_x25519_identity(self) -> None:
        with pytest.raises(PluginRecipientRejected):
            age_decrypt_bytes(b"x", [object()])  # type: ignore[list-item]

    def test_empty_recipients_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one recipient"):
            age_encrypt_bytes(b"x", [])

    def test_empty_identities_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one identity"):
            age_decrypt_bytes(b"x", [])

    def test_recipient_from_bech32_refuses_plugin_style(self) -> None:
        # A synthetic plugin-style recipient that survives the age1 prefix
        # check but fails the x25519 bech32 decode.
        with pytest.raises(PluginRecipientRejected):
            recipient_from_bech32(
                "age1yubikey1qtq8dk6mcyrq33vk6d5atxu0sdgw6qarax5y2jc6g4ej3kw7m6y8g2q9qf"
            )

    def test_recipient_from_bech32_refuses_non_age_prefix(self) -> None:
        with pytest.raises(ValueError, match="not an age recipient"):
            recipient_from_bech32("AGE-SECRET-KEY-1dummy")

    def test_identity_from_bech32_refuses_non_age_prefix(self) -> None:
        with pytest.raises(ValueError, match="not an age secret key"):
            identity_from_bech32("age1dummy")


class TestIsAgeEnvelope:
    def test_true_for_written_envelope(self, tmp_path: Path) -> None:
        _, recip = _mint_pair()
        src = tmp_path / "plain.bin"
        dst = tmp_path / "cipher.age"
        src.write_bytes(b"x")
        age_encrypt_file(src, dst, [recip])
        assert is_age_envelope(dst) is True

    def test_false_for_arbitrary_bytes(self, tmp_path: Path) -> None:
        p = tmp_path / "random.bin"
        p.write_bytes(b"OTHER\n" + os.urandom(100))
        assert is_age_envelope(p) is False

    def test_false_for_missing_file(self, tmp_path: Path) -> None:
        assert is_age_envelope(tmp_path / "does-not-exist") is False


class TestBech32RoundTrip:
    def test_identity_round_trip(self) -> None:
        ident = x25519.Identity.generate()
        bech = str(ident)
        assert bech.startswith("AGE-SECRET-KEY-1")
        assert len(bech) == 74
        parsed = identity_from_bech32(bech)
        assert str(parsed) == bech

    def test_recipient_round_trip(self) -> None:
        ident = x25519.Identity.generate()
        recip = ident.to_public()
        bech = str(recip)
        assert bech.startswith("age1")
        parsed = recipient_from_bech32(bech)
        assert str(parsed) == bech

    def test_lowercase_bech32_still_parses(self) -> None:
        """pyrage emits uppercase; confirm case-insensitive ingest still works."""
        ident = x25519.Identity.generate()
        bech = str(ident)
        parsed = identity_from_bech32(bech.lower())
        assert str(parsed).upper() == bech.upper()


class TestEdgeCases:
    def test_empty_plaintext_round_trips(self) -> None:
        """age's chunked format handles zero-byte payloads; confirm at our boundary."""
        ident, recip = _mint_pair()
        ct = age_encrypt_bytes(b"", [recip])
        assert ct.startswith(AGE_MAGIC)
        assert age_decrypt_bytes(ct, [ident]) == b""

    def test_encrypt_file_missing_source_raises_file_not_found(self, tmp_path: Path) -> None:
        """Missing source path surfaces FileNotFoundError for operator-facing clarity."""
        _, recip = _mint_pair()
        with pytest.raises(FileNotFoundError):
            age_encrypt_file(tmp_path / "nope.bin", tmp_path / "out.age", [recip])
        # Nothing is written — no tmp, no dst.
        assert not (tmp_path / "out.age").exists()
        assert not (tmp_path / "out.age.tmp").exists()
