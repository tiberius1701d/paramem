"""End-to-end integration tests for the security CLI chain.

Invokes the installed ``paramem`` binary via :mod:`subprocess` with a
hermetic env — each test gets its own scratch ``HOME`` so
``~/.config/paramem/{daily_key.age,recovery.pub}`` resolve into the
test's ``tmp_path``. The whole chain is exercised without touching the
systemd service, the GPU, or real operator key material.

Covers the flows unit tests structurally can't:

- CLI argparse dispatch reaches every subcommand under the real binary.
- Subprocess boundary keeps module-attribute late-binding honest (the
  default-argument / module-default pitfalls I hit multiple times during
  the WP2b arc would have been caught here immediately).
- The env-pollution class of bug that pytest-dotenv + a live deployment
  introduced — here the subprocess env is built explicitly, not
  inherited from pytest's polluted one.
- Real filesystem atomicity across multiple subprocess invocations.

Each scenario is hermetic. Total runtime ~60 s (dominated by pyrage's
scrypt KDF unwraps — ~1.5 s per daily unlock, paid every subprocess
that reads the daily envelope).

Gated behind ``@pytest.mark.integration`` so quick unit runs skip it.
To run only this file::

    pytest -m integration tests/integration/test_security_chain.py
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

PARAMEM_BINARY = os.environ.get(
    "PARAMEM_BINARY",
    "/home/tiberius/miniforge3/envs/paramem/bin/paramem",
)

AGE_MAGIC = b"age-encryption.org/v1\n"
RECOVERY_BECH32_RE = re.compile(r"AGE-SECRET-KEY-1[A-Z0-9]+")


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _build_env(home: Path, **extra: str) -> dict[str, str]:
    """Return a subprocess env with HOME → *home* and PARAMEM_* keys cleared.

    Starting from ``os.environ`` inherits the system essentials (PATH,
    PYTHONPATH, LANG, etc.) without carrying any paramem-specific pollution
    from the pytest session (which auto-loads .env via python-dotenv's
    plugin). Any ``PARAMEM_*`` the caller wants is set explicitly via
    *extra*.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("PARAMEM_")}
    env["HOME"] = str(home)
    # Point XDG_CONFIG_HOME too so any code that honours it lands in the
    # scratch tree instead of the real ~/.config.
    env["XDG_CONFIG_HOME"] = str(home / ".config")
    env.update(extra)
    return env


def _run(
    *args: str,
    env: dict[str, str],
    cwd: Path | None = None,
    input_: bytes | None = None,
    check: bool = True,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """Invoke the paramem binary with *args* under *env*.

    Captures stdout + stderr. On non-zero exit with ``check=True``, raises
    :class:`AssertionError` with both streams decoded for debuggability.
    """
    proc = subprocess.run(
        [PARAMEM_BINARY, *args],
        env=env,
        cwd=str(cwd) if cwd else None,
        input=input_,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"paramem {' '.join(args)} failed (exit {proc.returncode})\n"
            f"stdout: {proc.stdout.decode('utf-8', errors='replace')}\n"
            f"stderr: {proc.stderr.decode('utf-8', errors='replace')}"
        )
    return proc


def _skip_if_binary_missing():
    if not Path(PARAMEM_BINARY).exists():
        pytest.skip(f"paramem binary not at {PARAMEM_BINARY}; set PARAMEM_BINARY to override")


# ---------------------------------------------------------------------------
# Shared setup: generate keys and seed age-encrypted infra files
# ---------------------------------------------------------------------------


def _setup_age_store(
    scratch_home: Path,
    data_dir: Path,
    tmp_path: Path,
    passphrase: str = "integration-pw",
) -> tuple[dict[str, str], str, Path]:
    """Run ``generate-key`` then seed age-encrypted infra files.

    Builds the store the fresh-install way: mint daily + recovery identities
    with ``paramem generate-key``, then write the infra files as age envelopes
    in-process (using the recipients from the newly-generated key files) so
    downstream subprocess invocations encounter the full age-encrypted state.

    Returns the env dict (with the daily passphrase set), the captured
    recovery bech32, and the passphrase file path so downstream scenarios
    can re-use them.
    """
    env = _build_env(scratch_home)

    # Mint daily + recovery under the scratch HOME.
    pw_file = tmp_path / "integration-pw.txt"
    pw_file.write_text(passphrase + "\n", encoding="utf-8")
    gen = _run(
        "generate-key",
        "--passphrase-file",
        str(pw_file),
        "--yes",
        env=env,
    )
    err = gen.stderr.decode("utf-8", errors="replace")
    match = RECOVERY_BECH32_RE.search(err)
    assert match, f"recovery bech32 not printed to stderr: {err[:400]}"
    recovery_bech32 = match.group(0)

    # Set the daily passphrase in the env for subsequent subprocess calls.
    env["PARAMEM_DAILY_PASSPHRASE"] = passphrase

    # Seed age-encrypted infra files in-process using the generated key
    # material so the store is ready for rotate / restore / dump commands.
    from paramem.backup.age_envelope import age_encrypt_bytes
    from paramem.backup.key_store import load_daily_identity, load_recovery_recipient

    config_dir = scratch_home / ".config" / "paramem"
    daily_ident = load_daily_identity(config_dir / "daily_key.age", passphrase=passphrase)
    recovery_rec = load_recovery_recipient(config_dir / "recovery.pub")
    recipients = [daily_ident.to_public(), recovery_rec]

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "registry.json").write_bytes(age_encrypt_bytes(b'{"integration": 1}', recipients))
    (data_dir / "speaker_profiles.json").write_bytes(
        age_encrypt_bytes(b'{"speakers": {}, "version": 5}', recipients)
    )

    return env, recovery_bech32, pw_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scratch_home(tmp_path: Path) -> Path:
    _skip_if_binary_missing()
    home = tmp_path / "home"
    home.mkdir()
    (home / ".config" / "paramem").mkdir(parents=True)
    return home


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Scenario 1: fresh install → age-encrypted store
# ---------------------------------------------------------------------------


class TestFreshInstallToAge:
    def test_fresh_install_produces_age_store(self, scratch_home, data_dir, tmp_path):
        env, recovery_bech32, _ = _setup_age_store(scratch_home, data_dir, tmp_path)

        # Every seeded file carries the age magic.
        for name in ("registry.json", "speaker_profiles.json"):
            head = (data_dir / name).read_bytes()[: len(AGE_MAGIC)]
            assert head == AGE_MAGIC, f"{name} not age-encrypted: head={head!r}"

        # Recovery bech32 parses and yields a deterministic recipient string —
        # confirms generate-key's stderr output is machine-extractable.
        assert recovery_bech32.startswith("AGE-SECRET-KEY-1")
        assert len(recovery_bech32) == 74

        # Key files on disk at expected modes.
        daily_path = scratch_home / ".config" / "paramem" / "daily_key.age"
        recovery_path = scratch_home / ".config" / "paramem" / "recovery.pub"
        assert stat.S_IMODE(daily_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(recovery_path.stat().st_mode) == 0o644

        # End-to-end read smoke via `paramem dump` — universal reader dispatches
        # age → age_decrypt_bytes with the cached daily identity.
        dumped = _run(
            "dump",
            str(data_dir / "registry.json"),
            env=env,
        )
        assert dumped.stdout == b'{"integration": 1}', (
            f"dump round-trip failed, got: {dumped.stdout!r}"
        )


# ---------------------------------------------------------------------------
# Scenario 2: rotate-daily then rotate-recovery
# ---------------------------------------------------------------------------


class TestRotations:
    def test_rotate_daily_then_rotate_recovery(self, scratch_home, data_dir, tmp_path):
        env, recovery_bech32_v1, _ = _setup_age_store(scratch_home, data_dir, tmp_path)

        # Capture the original daily + recovery recipients for comparison.
        daily_path = scratch_home / ".config" / "paramem" / "daily_key.age"
        recovery_path = scratch_home / ".config" / "paramem" / "recovery.pub"
        recovery_pub_before = recovery_path.read_text("utf-8").strip()
        daily_bytes_before = daily_path.read_bytes()

        # --- rotate-daily --------------------------------------------------
        _run(
            "rotate-daily",
            "--data-dir",
            str(data_dir),
            "--config",
            "/nonexistent",
            "--verbose",
            env=env,
        )

        # daily_key.age changed, recovery.pub unchanged.
        assert daily_path.read_bytes() != daily_bytes_before, (
            "rotate-daily must replace daily_key.age"
        )
        assert recovery_path.read_text("utf-8").strip() == recovery_pub_before, (
            "rotate-daily must preserve the recovery recipient"
        )
        # No manifest, no pending file left behind.
        assert not (scratch_home / ".config" / "paramem" / "rotation.manifest.json").exists()
        assert not (scratch_home / ".config" / "paramem" / "daily_key.age.pending").exists()

        # Data still readable with the new daily.
        dumped = _run("dump", str(data_dir / "registry.json"), env=env)
        assert dumped.stdout == b'{"integration": 1}'

        # --- rotate-recovery ----------------------------------------------
        rotate_rec = _run(
            "rotate-recovery",
            "--data-dir",
            str(data_dir),
            "--config",
            "/nonexistent",
            "--yes",
            env=env,
        )
        err = rotate_rec.stderr.decode("utf-8", errors="replace")
        # A NEW recovery bech32 was emitted (print-once).
        new_match = RECOVERY_BECH32_RE.search(err)
        assert new_match, f"rotate-recovery did not print the new bech32: {err[:400]}"
        assert new_match.group(0) != recovery_bech32_v1, (
            "rotate-recovery must mint a DIFFERENT recovery identity"
        )

        # recovery.pub changed, daily_key.age unchanged this phase.
        assert recovery_path.read_text("utf-8").strip() != recovery_pub_before
        # No manifest / pending files left behind.
        assert not (scratch_home / ".config" / "paramem" / "rotation.manifest.json").exists()
        assert not (scratch_home / ".config" / "paramem" / "recovery.pub.pending").exists()

        # Data still readable with the (unchanged-in-this-phase) daily.
        dumped = _run("dump", str(data_dir / "speaker_profiles.json"), env=env)
        assert dumped.stdout == b'{"speakers": {}, "version": 5}'


# ---------------------------------------------------------------------------
# Scenario 3: simulated hardware loss → restore with recovery bech32
# ---------------------------------------------------------------------------


class TestHardwareLossRestore:
    def test_restore_with_captured_recovery_bech32(self, scratch_home, data_dir, tmp_path):
        env, recovery_bech32, _ = _setup_age_store(scratch_home, data_dir, tmp_path)

        # Snapshot original envelope bytes for post-restore comparison.
        original_envelopes = {
            name: (data_dir / name).read_bytes()
            for name in ("registry.json", "speaker_profiles.json")
        }

        # Simulate hardware loss: delete daily + recovery.pub.
        (scratch_home / ".config" / "paramem" / "daily_key.age").unlink()
        (scratch_home / ".config" / "paramem" / "recovery.pub").unlink()

        # Operator supplies the recovery bech32 (from paper) + a fresh passphrase.
        recovery_file = tmp_path / "recovery.txt"
        recovery_file.write_text(recovery_bech32, encoding="utf-8")
        new_pw_file = tmp_path / "restore-pw.txt"
        new_pw_file.write_text("restored-pw\n", encoding="utf-8")

        # The env must NOT carry the old PARAMEM_DAILY_PASSPHRASE — simulate a
        # true hardware-loss scenario where the operator has only paper.
        env_for_restore = _build_env(scratch_home)

        _run(
            "restore",
            "--data-dir",
            str(data_dir),
            "--config",
            "/nonexistent",
            "--recovery-key-file",
            str(recovery_file),
            "--passphrase-file",
            str(new_pw_file),
            "--verbose",
            env=env_for_restore,
        )

        # Both key files re-created at the expected modes.
        daily_path = scratch_home / ".config" / "paramem" / "daily_key.age"
        recovery_path = scratch_home / ".config" / "paramem" / "recovery.pub"
        assert daily_path.exists()
        assert recovery_path.exists()
        assert stat.S_IMODE(daily_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(recovery_path.stat().st_mode) == 0o644

        # The envelopes have been re-encrypted (bytes differ from pre-restore)
        # but still read back as the original plaintext.
        for name, before_bytes in original_envelopes.items():
            after_bytes = (data_dir / name).read_bytes()
            assert after_bytes != before_bytes, f"{name} must be re-keyed"
            assert after_bytes.startswith(AGE_MAGIC)

        # Round-trip through dump using the new passphrase.
        env_for_restore["PARAMEM_DAILY_PASSPHRASE"] = "restored-pw"
        dumped = _run("dump", str(data_dir / "registry.json"), env=env_for_restore)
        assert dumped.stdout == b'{"integration": 1}'


# ---------------------------------------------------------------------------
# Scenario 4: rotate-daily resumes from a pre-staged manifest (post-crash)
# ---------------------------------------------------------------------------


class TestRotateDailyResumeFromManifest:
    def test_resume_finalises_from_partial_state(self, scratch_home, data_dir, tmp_path):
        """Pre-stage the state a prior rotate-daily run would leave after a
        crash: daily_key.age.pending + rotation.manifest.json listing the
        remaining files. A fresh `paramem rotate-daily` invocation must
        finalise without re-minting — proof that the resume path is wired
        through the CLI entry point, not just the rotation primitive."""
        env, _, _ = _setup_age_store(scratch_home, data_dir, tmp_path)

        config_dir = scratch_home / ".config" / "paramem"
        daily_path = config_dir / "daily_key.age"
        manifest_path = config_dir / "rotation.manifest.json"
        pending_path = config_dir / "daily_key.age.pending"

        # Extract the current daily's public recipient for manifest metadata —
        # use an in-process helper. Also mint the NEW daily via the same helper
        # and wrap it at the pending path. This emulates the state paramem
        # rotate-daily would leave mid-run.
        from paramem.backup.key_store import (
            load_daily_identity,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
        )
        from paramem.backup.rotation import RotationManifest, write_manifest_atomic

        passphrase = env["PARAMEM_DAILY_PASSPHRASE"]
        old_daily = load_daily_identity(daily_path, passphrase=passphrase)  # sanity
        _ = old_daily
        new_daily = mint_daily_identity()
        write_daily_key_file(wrap_daily_identity(new_daily, passphrase), pending_path)

        # Build the manifest: zero files done, everything pending.
        age_files = [data_dir / "registry.json", data_dir / "speaker_profiles.json"]
        manifest = RotationManifest(
            operation="rotate-daily",
            started_at="2026-04-24T12:00:00Z",
            new_daily_pub=str(new_daily.to_public()),
            files_pending=[str(p) for p in age_files],
            files_done=[],
        )
        write_manifest_atomic(manifest_path, manifest)

        # Invoke rotate-daily — must RESUME, not start fresh.
        before_daily_bytes = daily_path.read_bytes()
        _run(
            "rotate-daily",
            "--data-dir",
            str(data_dir),
            "--config",
            "/nonexistent",
            "--verbose",
            env=env,
        )

        # Pending file promoted; manifest deleted.
        assert not pending_path.exists()
        assert not manifest_path.exists()
        assert daily_path.read_bytes() != before_daily_bytes, (
            "resume must promote the pre-staged pending daily into place"
        )

        # The resulting daily identity equals the one we pre-staged.
        final = load_daily_identity(daily_path, passphrase=passphrase)
        assert str(final) == str(new_daily), (
            "resume must NOT re-mint — it must finalise the pre-existing pending"
        )


# ---------------------------------------------------------------------------
# Regression smoke: `--dry-run` cleanup (bug caught mid-D3 arc)
# ---------------------------------------------------------------------------


class TestDryRunCleanup:
    def test_rotate_daily_dry_run_leaves_no_artefacts(self, scratch_home, data_dir, tmp_path):
        """rotate-daily --dry-run on a fresh-start must not leave a pending
        key file or a manifest behind (they would confuse a later non-dry-run
        into thinking a prior crash occurred)."""
        env, _, _ = _setup_age_store(scratch_home, data_dir, tmp_path)

        _run(
            "rotate-daily",
            "--data-dir",
            str(data_dir),
            "--config",
            "/nonexistent",
            "--dry-run",
            env=env,
        )

        config_dir = scratch_home / ".config" / "paramem"
        assert not (config_dir / "rotation.manifest.json").exists()
        assert not (config_dir / "daily_key.age.pending").exists()

        # A subsequent real rotate-daily now behaves as fresh-start, not resume.
        rc = _run(
            "rotate-daily",
            "--data-dir",
            str(data_dir),
            "--config",
            "/nonexistent",
            env=env,
            check=False,
        )
        assert rc.returncode == 0


# ---------------------------------------------------------------------------
# Sanity: subcommand dispatch survives a `paramem --help`
# ---------------------------------------------------------------------------


class TestCliSurface:
    def test_help_lists_all_security_subcommands(self, scratch_home):
        proc = _run("--help", env=_build_env(scratch_home), check=False)
        combined = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
        for sub in (
            "generate-key",
            "dump",
            "rotate-daily",
            "rotate-recovery",
            "restore",
            "change-passphrase",
        ):
            assert sub in combined, f"subcommand missing from --help: {sub}"


# ---------------------------------------------------------------------------
# Scenario 7: change-passphrase rewraps daily_key.age without touching data
# ---------------------------------------------------------------------------


class TestChangePassphrase:
    def test_rewrap_preserves_identity_and_data_readability(self, scratch_home, data_dir, tmp_path):
        """Seed an age store, change the passphrase, verify that the same age
        envelopes still decrypt via the new passphrase through `paramem dump`
        — i.e. the X25519 identity survived the rewrap and existing data did
        not need re-encryption."""
        env, _, _ = _setup_age_store(scratch_home, data_dir, tmp_path)

        # Snapshot current daily_key.age bytes + recovery.pub.
        daily_before = (scratch_home / ".config" / "paramem" / "daily_key.age").read_bytes()
        recovery_before = (scratch_home / ".config" / "paramem" / "recovery.pub").read_text("utf-8")
        registry_before = (data_dir / "registry.json").read_bytes()

        old_pw_file = tmp_path / "old-pw.txt"
        old_pw_file.write_text("integration-pw\n", encoding="utf-8")  # matches _setup_age_store
        new_pw_file = tmp_path / "new-pw.txt"
        new_pw_file.write_text("rewrapped-pw\n", encoding="utf-8")

        _run(
            "change-passphrase",
            "--old-passphrase-file",
            str(old_pw_file),
            "--new-passphrase-file",
            str(new_pw_file),
            env=env,
        )

        # daily_key.age bytes changed (new wrapping); recovery.pub and all
        # data envelopes untouched.
        daily_after = (scratch_home / ".config" / "paramem" / "daily_key.age").read_bytes()
        assert daily_after != daily_before, "change-passphrase must rewrite daily_key.age"
        assert (scratch_home / ".config" / "paramem" / "recovery.pub").read_text(
            "utf-8"
        ) == recovery_before, "recovery.pub must not be touched"
        assert (data_dir / "registry.json").read_bytes() == registry_before, (
            "data envelopes must not be touched by a passphrase rewrap"
        )

        # Old passphrase no longer works; new passphrase does.
        env_new = dict(env)
        env_new["PARAMEM_DAILY_PASSPHRASE"] = "rewrapped-pw"
        dumped = _run("dump", str(data_dir / "registry.json"), env=env_new)
        assert dumped.stdout == b'{"integration": 1}'

        # Try with the OLD passphrase — must fail. The CLI's `dump` wraps the
        # unwrap error as a non-zero exit.
        env_stale = dict(env)
        env_stale["PARAMEM_DAILY_PASSPHRASE"] = "integration-pw"
        fail = _run(
            "dump",
            str(data_dir / "registry.json"),
            env=env_stale,
            check=False,
        )
        assert fail.returncode != 0, "old passphrase must no longer unlock daily_key.age"

    def test_rewrap_refuses_when_old_equals_new(self, scratch_home, data_dir, tmp_path):
        env, _, _ = _setup_age_store(scratch_home, data_dir, tmp_path)
        same_pw_file = tmp_path / "same-pw.txt"
        same_pw_file.write_text("integration-pw\n", encoding="utf-8")

        fail = _run(
            "change-passphrase",
            "--old-passphrase-file",
            str(same_pw_file),
            "--new-passphrase-file",
            str(same_pw_file),
            env=env,
            check=False,
        )
        assert fail.returncode == 1
        assert b"identical" in fail.stderr
