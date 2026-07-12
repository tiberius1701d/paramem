"""Candidate construction, validation, and promotion of the live server.yaml.

Covers the config pipeline split (``build_server_config`` is the construction stage
boot itself runs) and the three migration primitives built on it:

- ``validate_candidate`` — construct the candidate **at the live config path**, so
  validation cannot drift from boot and path-anchored ``paths.*`` resolve to the
  values the promoted config will actually have.
- ``promote_config`` — the only route by which the live config file changes:
  read → hash-check → construct → rename.  Every rejection happens before the
  first mutation.
- ``backup_live_config`` — the ``pre_migration`` restore point, written before the
  promotion.

Privacy is load-bearing here: the stash and the diffs keep ``${VAR}`` templates
verbatim, so a secret is never persisted or echoed.  Construction interpolates env
vars into a **copy** (``_interpolate_env_vars`` returns new containers), and the
resulting ``ServerConfig`` is discarded by every caller.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.backup.types import FatalConfigError
from paramem.server.config import ServerConfig, build_server_config, load_server_config
from paramem.server.migration import (
    CandidateChanged,
    CandidateConfigInvalid,
    backup_live_config,
    initial_migration_state,
    promote_config,
    validate_candidate,
)

FIXTURE_CONFIG = Path(__file__).resolve().parents[1] / "fixtures" / "server.yaml"

# A minimal candidate that constructs cleanly.
_VALID_CANDIDATE = b"model: mistral\ndebug: true\n"


# ---------------------------------------------------------------------------
# The split: build_server_config is the construction stage boot performs
# ---------------------------------------------------------------------------


class TestBuildServerConfigIsBoot:
    def test_build_from_parsed_equals_load_from_path(self):
        """Boot is unchanged by the split: same file → identical config both ways.

        ``load_server_config`` now delegates construction to ``build_server_config``;
        this pins that the delegation is behaviour-preserving on the production-shaped
        fixture (adapters, retention tiers, agents, voice — the whole surface).
        """
        raw = yaml.safe_load(FIXTURE_CONFIG.read_bytes())
        built = build_server_config(raw, source_path=FIXTURE_CONFIG)
        loaded = load_server_config(FIXTURE_CONFIG)
        assert built == loaded

    def test_construction_is_path_anchored(self, tmp_path):
        """Same bytes, different source_path → different resolved paths.*.

        This is *why* candidate validation must pass the LIVE config path: relative
        ``paths.*`` anchor to the project root of the YAML's own directory, so
        validating a staged file at its staging path would construct a different
        config than the one that goes live.
        """
        raw = yaml.safe_load(FIXTURE_CONFIG.read_bytes())

        at_live = build_server_config(raw, source_path=FIXTURE_CONFIG)
        at_staging = build_server_config(raw, source_path=tmp_path / "candidate.yaml")

        assert at_live.paths.data != at_staging.paths.data
        assert at_live != at_staging

    def test_adapter_guard_error_names_the_source_path(self, tmp_path):
        """The loader guard's remediation text names source_path, as it does at boot."""
        raw = {"adapters": {"episodic": {"enabled": True, "rank": 8}}}
        live = tmp_path / "server.yaml"
        with pytest.raises(FatalConfigError, match=str(live)):
            build_server_config(raw, source_path=live)


# ---------------------------------------------------------------------------
# validate_candidate
# ---------------------------------------------------------------------------


class TestValidateCandidate:
    def test_valid_candidate_returns_server_config(self, tmp_path):
        cfg = validate_candidate(_VALID_CANDIDATE, tmp_path / "server.yaml")
        assert isinstance(cfg, ServerConfig)
        assert cfg.model_name == "mistral"

    def test_production_fixture_is_a_valid_candidate(self, tmp_path):
        cfg = validate_candidate(FIXTURE_CONFIG.read_bytes(), tmp_path / "server.yaml")
        assert isinstance(cfg, ServerConfig)

    def test_mode_typo_rejected(self, tmp_path):
        """``mode: simulated`` — the typo that boots clean and silently picks simulate."""
        with pytest.raises(CandidateConfigInvalid, match="consolidation.mode"):
            validate_candidate(b"consolidation:\n  mode: simulated\n", tmp_path / "server.yaml")

    def test_stalled_ingestion_combination_rejected(self, tmp_path):
        """max_interim_count=0 + mode=simulate stalls ingestion — rejected at load."""
        with pytest.raises(CandidateConfigInvalid, match="max_interim_count"):
            validate_candidate(
                b"consolidation:\n  mode: simulate\n  max_interim_count: 0\n",
                tmp_path / "server.yaml",
            )

    def test_unknown_key_rejected(self, tmp_path):
        """An unknown YAML key raises TypeError from **kwargs → 4xx, not 500."""
        with pytest.raises(CandidateConfigInvalid):
            validate_candidate(
                b"consolidation:\n  no_such_knob: 3\n",
                tmp_path / "server.yaml",
            )

    def test_env_template_in_typed_field_rejected(self, tmp_path, monkeypatch):
        """``max_interim_count: ${VAR}`` survives interpolation as a str → TypeError.

        The str reaches ``__post_init__``'s ``self.max_interim_count < 0`` comparison
        and raises TypeError, not ValueError.  Catching only ValueError would surface
        this operator typo as an unhandled 500.
        """
        monkeypatch.setenv("PARAMEM_TEST_INTERIM", "not_an_int")
        with pytest.raises(CandidateConfigInvalid):
            validate_candidate(
                b"consolidation:\n  max_interim_count: ${PARAMEM_TEST_INTERIM}\n",
                tmp_path / "server.yaml",
            )

    def test_unparseable_yaml_rejected(self, tmp_path):
        with pytest.raises(CandidateConfigInvalid):
            validate_candidate(b"model: [unclosed\n", tmp_path / "server.yaml")


# ---------------------------------------------------------------------------
# Privacy — the stash keeps ${VAR} verbatim because construction copies
# ---------------------------------------------------------------------------


class TestValidateCandidateDoesNotMutateInput:
    def test_parsed_input_is_not_interpolated_in_place(self, tmp_path, monkeypatch):
        """Construction must not substitute secrets into the caller's parsed dict.

        ``_interpolate_env_vars`` returns new containers rather than mutating in
        place — that single property is what lets the migration stash, the unified
        diff, and the tier diff carry ``${VAR}`` templates verbatim.  Pin it: build
        a config from a parsed dict holding a secret template and prove the dict is
        byte-identical afterwards.
        """
        monkeypatch.setenv("PARAMEM_TEST_SECRET", "sentinel-value-must-not-leak")
        raw = yaml.safe_load(
            b"tools:\n  ha:\n    token: ${PARAMEM_TEST_SECRET}\n    url: http://x\n"
        )
        before = copy.deepcopy(raw)

        cfg = build_server_config(raw, source_path=tmp_path / "server.yaml")

        # The constructed config DID interpolate (it is the boot-time config)...
        assert cfg.tools.ha.token == "sentinel-value-must-not-leak"
        # ...but the caller's dict is untouched.
        assert raw == before
        assert raw["tools"]["ha"]["token"] == "${PARAMEM_TEST_SECRET}"


# ---------------------------------------------------------------------------
# promote_config
# ---------------------------------------------------------------------------


def _sha256_file(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()


class TestPromoteConfig:
    def test_valid_candidate_is_promoted(self, tmp_path):
        live = tmp_path / "server.yaml"
        live.write_bytes(b"model: mistral\ndebug: false\n")
        cand = tmp_path / "candidate.yaml"
        cand.write_bytes(_VALID_CANDIDATE)

        result = promote_config(cand, live, expected_sha256=_sha256_file(cand))

        assert result is None
        assert live.read_bytes() == _VALID_CANDIDATE
        assert not cand.exists()

    def test_invalid_candidate_leaves_the_filesystem_untouched(self, tmp_path):
        """Validation precedes the rename: zero mutations on the reject path."""
        live = tmp_path / "server.yaml"
        live_bytes = b"model: mistral\ndebug: false\n"
        live.write_bytes(live_bytes)
        cand = tmp_path / "candidate.yaml"
        cand_bytes = b"consolidation:\n  mode: simulated\n"
        cand.write_bytes(cand_bytes)

        with pytest.raises(CandidateConfigInvalid):
            promote_config(cand, live, expected_sha256=_sha256_file(cand))

        assert live.read_bytes() == live_bytes, "live config was mutated by a rejected promote"
        assert cand.read_bytes() == cand_bytes, "candidate was consumed by a rejected promote"

    def test_candidate_changed_on_disk_is_rejected(self, tmp_path):
        """What was previewed is what gets promoted — or nothing is."""
        live = tmp_path / "server.yaml"
        live_bytes = b"model: mistral\ndebug: false\n"
        live.write_bytes(live_bytes)
        cand = tmp_path / "candidate.yaml"
        cand.write_bytes(_VALID_CANDIDATE)
        staged_hash = _sha256_file(cand)

        # Operator edits the candidate after previewing it.
        cand.write_bytes(b"model: mistral\ndebug: false\ncloud_only: true\n")

        with pytest.raises(CandidateChanged):
            promote_config(cand, live, expected_sha256=staged_hash)

        assert live.read_bytes() == live_bytes
        assert cand.exists()

    def test_empty_expected_hash_is_rejected(self, tmp_path):
        """No production caller has nothing to compare against — reject, don't skip.

        Every candidate is staged via ``/migration/preview``, which always computes
        a 64-hex digest.  An empty hash reaching here would promote without ever
        checking that the file on disk is still the one the operator previewed.
        """
        live = tmp_path / "server.yaml"
        live.write_bytes(b"model: mistral\n")
        cand = tmp_path / "candidate.yaml"
        cand.write_bytes(_VALID_CANDIDATE)

        with pytest.raises(ValueError, match="expected_sha256"):
            promote_config(cand, live, expected_sha256="")

        assert live.read_bytes() == b"model: mistral\n"
        assert cand.exists()


class TestBackupLiveConfig:
    def test_writes_pre_migration_slot_with_pre_trial_hash(self, tmp_path):
        import json

        live = tmp_path / "server.yaml"
        live.write_bytes(b"model: mistral\n")
        backups_root = tmp_path / "backups"

        pre_hash, slot = backup_live_config(live, backups_root)

        assert pre_hash == _sha256_file(live)
        assert slot.is_dir()
        assert slot.parent == backups_root / "config"
        meta_files = list(slot.glob("*.meta.json"))
        assert meta_files
        meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
        assert meta["tier"] == "pre_migration"
        assert meta["pre_trial_hash"] == pre_hash

    def test_missing_live_config_yields_empty_hash(self, tmp_path):
        pre_hash, slot = backup_live_config(tmp_path / "absent.yaml", tmp_path / "backups")
        assert pre_hash == ""
        assert slot.is_dir()


# ---------------------------------------------------------------------------
# Endpoint privacy — a secret template never leaves the stash
# ---------------------------------------------------------------------------

_SECRET_SENTINEL = "sentinel-value-must-not-leak"
_SECRET_CANDIDATE = (
    b"model: mistral\ndebug: true\ntools:\n  ha:\n"
    b"    url: http://ha.local\n    token: ${PARAMEM_TEST_SECRET}\n"
)


class TestPreviewKeepsSecretsOutOfEverything:
    @pytest.fixture()
    def state(self, tmp_path, monkeypatch):
        live = tmp_path / "server.yaml"
        live.write_bytes(b"model: mistral\ndebug: false\n")
        config = MagicMock()
        config.paths.data = tmp_path / "data" / "ha"
        config.paths.data.mkdir(parents=True, exist_ok=True)
        config.adapter_dir = tmp_path / "data" / "ha" / "adapters"
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        fresh = {
            "model": None,
            "config": config,
            "config_path": str(live),
            "consolidating": False,
            "migration": initial_migration_state(),
            "server_started_at": "2026-07-11T00:00:00+00:00",
        }
        monkeypatch.setattr(app_module, "_state", fresh)
        return fresh

    def test_secret_appears_nowhere_after_preview(self, state, tmp_path, monkeypatch, caplog):
        """Env var set to a sentinel; preview must not echo, log, or persist it.

        The candidate's ``tools.ha.token`` is a ``${VAR}`` template.  Preview
        constructs the candidate (which interpolates the sentinel into a throwaway
        ``ServerConfig``) — and then discards it.  The response, the logs, the
        backups root, and the stash must all still carry the template, not the value.
        """
        monkeypatch.setenv("PARAMEM_TEST_SECRET", _SECRET_SENTINEL)
        cand = tmp_path / "candidate.yaml"
        cand.write_bytes(_SECRET_CANDIDATE)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        with caplog.at_level(logging.DEBUG):
            resp = client.post("/migration/preview", json={"candidate_path": str(cand)})

        assert resp.status_code == 200, resp.text
        assert _SECRET_SENTINEL not in resp.text, "sentinel leaked into the preview response"

        for record in caplog.records:
            assert _SECRET_SENTINEL not in record.getMessage(), "sentinel leaked into the logs"

        backups_root = state["config"].paths.data / "backups"
        for path in backups_root.rglob("*"):
            if path.is_file():
                assert _SECRET_SENTINEL.encode() not in path.read_bytes(), (
                    f"sentinel leaked into {path}"
                )

        stash = state["migration"]
        assert stash["state"] == "STAGING"
        assert stash["parsed_candidate"]["tools"]["ha"]["token"] == "${PARAMEM_TEST_SECRET}"
        assert b"${PARAMEM_TEST_SECRET}" in stash["candidate_bytes"]
        assert _SECRET_SENTINEL not in stash["unified_diff"]
        assert all(_SECRET_SENTINEL not in str(row) for row in stash["tier_diff"])


class TestConfirmKeepsSecretsOutOfTheBackupSlot:
    """Confirm is the branch that actually writes a file (the pre_migration slot).

    Preview writes nothing, so a file-sweep assertion after preview never executes
    its loop body.  This class stages a secret-bearing candidate directly (bypassing
    preview) and confirms it, so the sweep below is exercised against real bytes on
    disk.
    """

    @pytest.fixture()
    def state(self, tmp_path, monkeypatch):
        import asyncio

        live = tmp_path / "server.yaml"
        live.write_bytes(b"model: mistral\ndebug: false\n")
        cand = tmp_path / "candidate.yaml"
        cand.write_bytes(_SECRET_CANDIDATE)

        config = MagicMock()
        config.paths.data = tmp_path / "data" / "ha"
        config.paths.data.mkdir(parents=True, exist_ok=True)
        config.adapter_dir = tmp_path / "data" / "ha" / "adapters"
        config.adapter_dir.mkdir(parents=True, exist_ok=True)

        loop_mock = MagicMock()
        loop_mock.merger.save_bytes.return_value = b'{"nodes":[],"links":[]}'

        staging = initial_migration_state()
        staging["state"] = "STAGING"
        staging["candidate_path"] = str(cand)
        staging["candidate_hash"] = __import__("hashlib").sha256(_SECRET_CANDIDATE).hexdigest()
        staging["candidate_bytes"] = _SECRET_CANDIDATE
        staging["candidate_text"] = _SECRET_CANDIDATE.decode("utf-8")
        # Non-mode-switch, non-base-swap diff — routes through the general trial
        # branch, which writes the pre_migration config backup slot.
        staging["tier_diff"] = [
            {"dotted_path": "debug", "old_value": False, "new_value": True, "tier": "operational"}
        ]

        fresh = {
            "model": None,
            "tokenizer": None,
            "config": config,
            "config_path": str(live),
            "consolidating": False,
            "migration": staging,
            "migration_lock": asyncio.Lock(),
            "server_started_at": "2026-07-11T00:00:00+00:00",
            "mode": "normal",
            "background_trainer": None,
            "consolidation_loop": loop_mock,
            "session_buffer": None,
        }
        monkeypatch.setattr(app_module, "_state", fresh)

        async def _noop_trial():
            pass

        monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)
        return fresh

    def test_secret_appears_nowhere_on_disk_after_confirm(self, state, monkeypatch, caplog):
        monkeypatch.setenv("PARAMEM_TEST_SECRET", _SECRET_SENTINEL)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        with caplog.at_level(logging.DEBUG):
            resp = client.post("/migration/confirm", json={})

        assert resp.status_code == 200, resp.text
        assert _SECRET_SENTINEL not in resp.text, "sentinel leaked into the confirm response"

        for record in caplog.records:
            assert _SECRET_SENTINEL not in record.getMessage(), "sentinel leaked into the logs"

        backups_root = state["config"].paths.data / "backups"
        swept = list(backups_root.rglob("*"))
        files = [p for p in swept if p.is_file()]
        assert files, "expected the pre_migration config backup slot to write at least one file"
        for path in files:
            assert _SECRET_SENTINEL.encode() not in path.read_bytes(), (
                f"sentinel leaked into {path}"
            )

        # The promoted live config also keeps the template verbatim — promote_config
        # renames the candidate's own bytes, it does not write an interpolated copy.
        live_bytes = Path(state["config_path"]).read_bytes()
        assert _SECRET_SENTINEL.encode() not in live_bytes
        assert b"${PARAMEM_TEST_SECRET}" in live_bytes
