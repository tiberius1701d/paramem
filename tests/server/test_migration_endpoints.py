"""Tests for the Slice 3b.1 migration server endpoints.

Bypass convention
-----------------
All tests use ``fastapi.testclient.TestClient`` with the real FastAPI app but
without triggering the lifespan (no model load, no GPU required).

Pattern:
1. Import ``paramem.server.app`` as ``app_module``.
2. Build a fresh ``_state`` dict per test via fixture.
3. ``monkeypatch.setattr(app_module, "_state", fresh_state)``.
4. Pre-populate ``_state`` with the minimum required keys.
5. Use ``TestClient(app_module.app)``.

FastAPI's ``TestClient`` automatically handles the lifespan; since no
lifespan-triggering routes are called and the lifespan itself is not run by
the TestClient unless you use it as a context manager *without* the
``raise_server_exceptions=True`` flag, we use the client without entering it
as a context manager — this avoids model-load side-effects.

This pattern is the canonical convention for future server endpoint tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.server.migration import initial_migration_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_LIVE_YAML = (
    b"model: mistral\ndebug: false\nadapters:\n"
    b"  episodic:\n    enabled: true\n    rank: 8\n    alpha: 16\n"
)
_CAND_YAML = (
    b"model: mistral\ndebug: true\nadapters:\n"
    b"  episodic:\n    enabled: true\n    rank: 8\n    alpha: 16\n"
)


def _make_state(tmp_path: Path) -> dict:
    """Build a minimal fresh _state dict for endpoint tests."""
    live_yaml = tmp_path / "server.yaml"
    live_yaml.write_bytes(_LIVE_YAML)
    config = MagicMock()
    config.adapter_dir = tmp_path / "adapters"
    config.adapter_dir.mkdir(parents=True, exist_ok=True)
    config.paths.data = tmp_path / "data"
    config.paths.data.mkdir(parents=True, exist_ok=True)
    return {
        "model": None,
        "config": config,
        "config_path": str(live_yaml),
        "consolidating": False,
        "migration": initial_migration_state(),
        "server_started_at": "2026-04-22T00:00:00+00:00",
    }


@pytest.fixture()
def state(tmp_path, monkeypatch):
    """Monkeypatched _state dict — reset before each test."""
    fresh = _make_state(tmp_path)
    monkeypatch.setattr(app_module, "_state", fresh)
    return fresh


@pytest.fixture()
def client(state):
    """TestClient bound to the monkeypatched state (no lifespan)."""
    # Use raise_server_exceptions=False so 4xx/5xx are returned as responses,
    # not re-raised.  We check status codes explicitly.
    return TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helper: write a candidate YAML to a temp file
# ---------------------------------------------------------------------------


def _write_candidate(tmp_path: Path, content: bytes = _CAND_YAML) -> Path:
    p = tmp_path / "candidate.yaml"
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
# POST /migration/preview — happy path
# ---------------------------------------------------------------------------


class TestPreviewHappyPath:
    def test_preview_returns_200_and_staging(self, client, state, tmp_path):
        """Valid absolute candidate → 200, state=STAGING."""
        cand = _write_candidate(tmp_path)
        resp = client.post("/migration/preview", json={"candidate_path": str(cand)})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["state"] == "STAGING"
        assert body["candidate_path"] == str(cand)
        assert body["candidate_hash"]  # non-empty
        assert body["staged_at"]

    def test_preview_sets_stash_in_server_state(self, client, state, tmp_path):
        """After 200, _state['migration'] reflects the staged candidate."""
        cand = _write_candidate(tmp_path)
        client.post("/migration/preview", json={"candidate_path": str(cand)})
        migration = state["migration"]
        assert migration["state"] == "STAGING"
        assert migration["candidate_path"] == str(cand)

    def test_preview_includes_unified_diff(self, client, state, tmp_path):
        """Response includes a non-empty unified_diff when files differ."""
        cand = _write_candidate(tmp_path)
        resp = client.post("/migration/preview", json={"candidate_path": str(cand)})
        body = resp.json()
        assert isinstance(body["unified_diff"], str)
        # Files differ (debug: false → debug: true), so diff is non-empty.
        assert body["unified_diff"]

    def test_preview_includes_tier_diff(self, client, state, tmp_path):
        """Response includes tier_diff rows for changed fields."""
        cand = _write_candidate(tmp_path)
        resp = client.post("/migration/preview", json={"candidate_path": str(cand)})
        body = resp.json()
        assert isinstance(body["tier_diff"], list)
        # debug changed — at least one row.
        assert body["tier_diff"]

    def test_preview_includes_shape_changes_field(self, client, state, tmp_path):
        """Response always includes shape_changes list (may be empty)."""
        cand = _write_candidate(tmp_path)
        resp = client.post("/migration/preview", json={"candidate_path": str(cand)})
        body = resp.json()
        assert isinstance(body["shape_changes"], list)

    def test_preview_pre_flight_fail_is_none_in_3b1(self, client, state, tmp_path):
        """pre_flight_fail is always None in Slice 3b.1 (Condition 3)."""
        cand = _write_candidate(tmp_path)
        resp = client.post("/migration/preview", json={"candidate_path": str(cand)})
        body = resp.json()
        assert "pre_flight_fail" in body, "pre_flight_fail must always be present (Condition 3)"
        assert body["pre_flight_fail"] is None

    def test_preview_simulate_mode_override_false_by_default(self, client, state, tmp_path):
        """simulate_mode_override is False when candidate is not simulate mode."""
        cand = _write_candidate(tmp_path)
        resp = client.post("/migration/preview", json={"candidate_path": str(cand)})
        body = resp.json()
        assert body["simulate_mode_override"] is False

    def test_preview_simulate_mode_override_true_when_candidate_is_simulate(
        self, client, state, tmp_path
    ):
        """simulate_mode_override is True when candidate sets consolidation.mode: simulate."""
        content = b"model: mistral\nconsolidation:\n  mode: simulate\n"
        cand = tmp_path / "sim.yaml"
        cand.write_bytes(content)
        resp = client.post("/migration/preview", json={"candidate_path": str(cand)})
        body = resp.json()
        assert body["simulate_mode_override"] is True


# ---------------------------------------------------------------------------
# POST /migration/preview — error cases
# ---------------------------------------------------------------------------


class TestPreviewErrors:
    def test_preview_400_on_relative_path(self, client, state):
        """Relative candidate_path → 400 candidate_path_invalid."""
        resp = client.post("/migration/preview", json={"candidate_path": "relative/path.yaml"})
        assert resp.status_code == 400
        detail = resp.json().get("detail", {})
        assert detail.get("error") == "candidate_path_invalid"

    def test_preview_400_on_missing_file(self, client, state, tmp_path):
        """Non-existent absolute path → 400 candidate_path_invalid."""
        resp = client.post(
            "/migration/preview", json={"candidate_path": str(tmp_path / "no_such.yaml")}
        )
        assert resp.status_code == 400
        detail = resp.json().get("detail", {})
        assert detail.get("error") == "candidate_path_invalid"

    def test_preview_400_on_unparseable_yaml(self, client, state, tmp_path):
        """Invalid YAML content → 400 candidate_unparseable (Condition 7)."""
        cand = tmp_path / "bad.yaml"
        cand.write_bytes(b":\nnot valid: {\nyaml\n")
        resp = client.post("/migration/preview", json={"candidate_path": str(cand)})
        assert resp.status_code == 400
        detail = resp.json().get("detail", {})
        assert detail.get("error") == "candidate_unparseable"

    def test_preview_400_on_yaml_root_not_dict(self, client, state, tmp_path):
        """YAML root that is a list, not a dict → 400 candidate_unparseable."""
        cand = tmp_path / "list.yaml"
        cand.write_bytes(b"- item1\n- item2\n")
        resp = client.post("/migration/preview", json={"candidate_path": str(cand)})
        assert resp.status_code == 400
        detail = resp.json().get("detail", {})
        assert detail.get("error") == "candidate_unparseable"

    def test_preview_409_when_consolidating(self, client, state, tmp_path):
        """_state["consolidating"]=True → 409 consolidating."""
        state["consolidating"] = True
        cand = _write_candidate(tmp_path)
        resp = client.post("/migration/preview", json={"candidate_path": str(cand)})
        assert resp.status_code == 409
        detail = resp.json().get("detail", {})
        assert detail.get("error") == "consolidating"

    def test_preview_409_when_already_staging(self, client, state, tmp_path):
        """STAGING state → 409 already_staging."""
        # First preview to enter STAGING
        cand = _write_candidate(tmp_path)
        client.post("/migration/preview", json={"candidate_path": str(cand)})
        # Second preview should fail
        resp = client.post("/migration/preview", json={"candidate_path": str(cand)})
        assert resp.status_code == 409
        detail = resp.json().get("detail", {})
        assert detail.get("error") == "already_staging"


# ---------------------------------------------------------------------------
# POST /migration/cancel
# ---------------------------------------------------------------------------


class TestCancel:
    def test_cancel_after_preview_returns_live(self, client, state, tmp_path):
        """Preview then cancel → 200, state=LIVE, cleared_path set."""
        cand = _write_candidate(tmp_path)
        client.post("/migration/preview", json={"candidate_path": str(cand)})
        resp = client.post("/migration/cancel")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["state"] == "LIVE"
        assert body["cleared_path"] == str(cand)

    def test_cancel_resets_migration_state_to_live(self, client, state, tmp_path):
        """After cancel, _state['migration']['state'] is LIVE."""
        cand = _write_candidate(tmp_path)
        client.post("/migration/preview", json={"candidate_path": str(cand)})
        client.post("/migration/cancel")
        assert state["migration"]["state"] == "LIVE"

    def test_cancel_409_when_not_staging(self, client, state):
        """Cancel when already LIVE → 409 not_staging."""
        resp = client.post("/migration/cancel")
        assert resp.status_code == 409
        detail = resp.json().get("detail", {})
        assert detail.get("error") == "not_staging"

    def test_cancel_allows_new_preview_afterward(self, client, state, tmp_path):
        """Cancel → can immediately POST /migration/preview again."""
        cand = _write_candidate(tmp_path)
        client.post("/migration/preview", json={"candidate_path": str(cand)})
        client.post("/migration/cancel")
        resp2 = client.post("/migration/preview", json={"candidate_path": str(cand)})
        assert resp2.status_code == 200
        assert resp2.json()["state"] == "STAGING"


# ---------------------------------------------------------------------------
# GET /migration/status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_returns_live_initially(self, client, state):
        """Initial state is LIVE with null candidate fields."""
        resp = client.get("/migration/status")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["state"] == "LIVE"
        assert body["candidate_path"] is None
        assert body["candidate_hash"] is None
        assert body["staged_at"] is None

    def test_status_reflects_staging_after_preview(self, client, state, tmp_path):
        """After preview, /migration/status shows STAGING."""
        cand = _write_candidate(tmp_path)
        client.post("/migration/preview", json={"candidate_path": str(cand)})
        resp = client.get("/migration/status")
        body = resp.json()
        assert body["state"] == "STAGING"
        assert body["candidate_path"] == str(cand)
        assert body["candidate_hash"]

    def test_status_includes_server_started_at(self, client, state):
        """server_started_at is populated (Condition 6)."""
        resp = client.get("/migration/status")
        body = resp.json()
        assert "server_started_at" in body
        assert body["server_started_at"] == "2026-04-22T00:00:00+00:00"

    def test_status_includes_consolidating_flag(self, client, state):
        """consolidating mirrors _state['consolidating']."""
        state["consolidating"] = True
        resp = client.get("/migration/status")
        assert resp.json()["consolidating"] is True

    def test_status_returns_to_live_after_cancel(self, client, state, tmp_path):
        """Cancel → status reverts to LIVE."""
        cand = _write_candidate(tmp_path)
        client.post("/migration/preview", json={"candidate_path": str(cand)})
        client.post("/migration/cancel")
        resp = client.get("/migration/status")
        assert resp.json()["state"] == "LIVE"


# ---------------------------------------------------------------------------
# GET /migration/diff
# ---------------------------------------------------------------------------


class TestDiff:
    def test_diff_returns_preview_shape_when_staging(self, client, state, tmp_path):
        """After preview, /migration/diff returns the same shape as PreviewResponse."""
        cand = _write_candidate(tmp_path)
        client.post("/migration/preview", json={"candidate_path": str(cand)})
        resp = client.get("/migration/diff")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["state"] == "STAGING"
        assert "unified_diff" in body
        assert "tier_diff" in body
        assert "shape_changes" in body
        assert "pre_flight_fail" in body

    def test_diff_409_when_not_staging(self, client, state):
        """No candidate staged → 409 not_staging."""
        resp = client.get("/migration/diff")
        assert resp.status_code == 409
        detail = resp.json().get("detail", {})
        assert detail.get("error") == "not_staging"

    def test_diff_matches_preview_response(self, client, state, tmp_path):
        """diff response equals preview response for the same candidate."""
        cand = _write_candidate(tmp_path)
        preview_resp = client.post("/migration/preview", json={"candidate_path": str(cand)})
        diff_resp = client.get("/migration/diff")
        preview = preview_resp.json()
        diff = diff_resp.json()
        # Core fields must match.
        assert diff["candidate_hash"] == preview["candidate_hash"]
        assert diff["unified_diff"] == preview["unified_diff"]
        assert diff["tier_diff"] == preview["tier_diff"]

    def test_diff_409_after_cancel(self, client, state, tmp_path):
        """Cancel clears stash → diff returns 409."""
        cand = _write_candidate(tmp_path)
        client.post("/migration/preview", json={"candidate_path": str(cand)})
        client.post("/migration/cancel")
        resp = client.get("/migration/diff")
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Tier classification in diff
# ---------------------------------------------------------------------------


class TestTierDiffClassification:
    def test_destructive_change_shows_destructive_tier(self, client, state, tmp_path):
        """Changing adapters.episodic.rank triggers a destructive tier row."""
        live_yaml = Path(state["config_path"])
        live_yaml.write_bytes(
            b"model: mistral\nadapters:\n  episodic:\n"
            b"    enabled: true\n    rank: 8\n    alpha: 16\n"
        )
        cand = tmp_path / "rank_change.yaml"
        cand.write_bytes(
            b"model: mistral\nadapters:\n  episodic:\n"
            b"    enabled: true\n    rank: 16\n    alpha: 16\n"
        )
        resp = client.post("/migration/preview", json={"candidate_path": str(cand)})
        body = resp.json()
        assert any(
            r["dotted_path"] == "adapters.episodic.rank" and r["tier"] == "destructive"
            for r in body["tier_diff"]
        ), f"Expected destructive rank row, got: {body['tier_diff']}"

    def test_alpha_change_shows_destructive_tier(self, client, state, tmp_path):
        """Changing adapters.episodic.alpha → destructive tier row (Condition 1)."""
        live_yaml = Path(state["config_path"])
        live_yaml.write_bytes(
            b"model: mistral\nadapters:\n  episodic:\n"
            b"    enabled: true\n    rank: 8\n    alpha: 16\n"
        )
        cand = tmp_path / "alpha_change.yaml"
        cand.write_bytes(
            b"model: mistral\nadapters:\n  episodic:\n"
            b"    enabled: true\n    rank: 8\n    alpha: 32\n"
        )
        resp = client.post("/migration/preview", json={"candidate_path": str(cand)})
        body = resp.json()
        assert any(
            r["dotted_path"] == "adapters.episodic.alpha" and r["tier"] == "destructive"
            for r in body["tier_diff"]
        ), f"Expected destructive alpha row, got: {body['tier_diff']}"


# ---------------------------------------------------------------------------
# Fix 4 — registry_path graceful degradation when paths.data is None
# ---------------------------------------------------------------------------


class TestPreviewRegistryPathNoneGracefulDegradation:
    def test_preview_succeeds_when_paths_data_is_none(self, tmp_path, monkeypatch):
        """config.paths.data = None → preview returns 200, no 500.

        Guards against the TypeError that arises from ``None / "registry" / ...``
        when ``config.paths.data`` is None but ``hasattr(config, "paths")`` is True.
        """
        from fastapi.testclient import TestClient

        import paramem.server.app as app_module

        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(_LIVE_YAML)
        config = MagicMock()
        config.adapter_dir = tmp_path / "adapters"
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        # Deliberately set paths.data to None to trigger the original bug.
        config.paths.data = None

        fresh_state = {
            "model": None,
            "config": config,
            "config_path": str(live_yaml),
            "consolidating": False,
            "migration": initial_migration_state(),
            "server_started_at": "2026-04-22T00:00:00+00:00",
        }
        monkeypatch.setattr(app_module, "_state", fresh_state)
        test_client = TestClient(app_module.app, raise_server_exceptions=False)

        cand = tmp_path / "candidate.yaml"
        cand.write_bytes(_CAND_YAML)
        resp = test_client.post("/migration/preview", json={"candidate_path": str(cand)})
        assert resp.status_code == 200, f"Expected 200 but got {resp.status_code}: {resp.text}"
        body = resp.json()
        # Graceful degradation: shape_changes may be empty (no registry found) but no crash.
        assert isinstance(body["shape_changes"], list)
