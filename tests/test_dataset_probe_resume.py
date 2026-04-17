"""Unit tests for _find_resume_dir and _validate_resume_accumulator
in experiments/dataset_probe.py.

Verifies that the resume directory search looks under the correct
{base_dir}/{dataset}/{model}/{ts} layout introduced when model_output_dir
was called with ``base_dir=base_dir / args.dataset``.

Also verifies that _validate_resume_accumulator correctly reloads
accumulated QA from per-session raw_qa.json files and raises RuntimeError
on missing or corrupt files.

No GPU, no model load, no network access.
"""

import json
from pathlib import Path

import pytest

from experiments.dataset_probe import (
    _find_resume_dir,
    _validate_no_train_resume,
    _validate_resume_accumulator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(parent: Path, ts: str, dataset: str, completed: bool = False) -> Path:
    """Create a minimal run directory with a state.json.

    Args:
        parent: The {base_dir}/{dataset}/{model} directory.
        ts: Timestamp string used as the run directory name.
        dataset: Value stored in state.json["dataset"].
        completed: Value stored in state.json["completed"].

    Returns:
        Path to the created run directory.
    """
    run_dir = parent / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "dataset": dataset,
        "completed": completed,
        "training_started": False,
        "processed_session_ids": [],
    }
    (run_dir / "state.json").write_text(json.dumps(state))
    return run_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFindResumeDir:
    """_find_resume_dir must search {base_dir}/{dataset}/{model}/."""

    def test_finds_run_under_dataset_subdir(self, tmp_path):
        """Newest incomplete run is found when the directory layout is correct."""
        base_dir = tmp_path / "outputs" / "dataset_probe"
        model = "mistral"
        dataset = "perltqa"
        model_dir = base_dir / dataset / model
        _make_run(model_dir, "20260416_120000", dataset=dataset)

        result = _find_resume_dir(base_dir, model, dataset)

        assert result is not None, "_find_resume_dir should find the run"
        assert result.parent == model_dir

    def test_returns_none_when_no_dataset_subdir_exists(self, tmp_path):
        """Returns None when {base_dir}/{dataset}/{model}/ does not exist at all."""
        base_dir = tmp_path / "outputs" / "dataset_probe"
        # Do NOT create any subdirectory.
        result = _find_resume_dir(base_dir, "mistral", "perltqa")
        assert result is None

    def test_ignores_flat_model_dir_without_dataset_prefix(self, tmp_path):
        """A run placed directly at {base_dir}/{model}/ (old layout) is NOT found."""
        base_dir = tmp_path / "outputs" / "dataset_probe"
        old_model_dir = base_dir / "mistral"
        _make_run(old_model_dir, "20260416_120000", dataset="perltqa")

        result = _find_resume_dir(base_dir, "mistral", "perltqa")
        assert result is None, (
            "_find_resume_dir must not find runs placed at {base_dir}/{model}/ (old flat layout)"
        )

    def test_skips_completed_run(self, tmp_path):
        """Completed runs are never returned as resume targets."""
        base_dir = tmp_path / "outputs" / "dataset_probe"
        model_dir = base_dir / "perltqa" / "mistral"
        _make_run(model_dir, "20260416_120000", dataset="perltqa", completed=True)

        result = _find_resume_dir(base_dir, "mistral", "perltqa")
        assert result is None

    def test_skips_run_with_wrong_dataset(self, tmp_path):
        """Run with a different dataset tag is not returned."""
        base_dir = tmp_path / "outputs" / "dataset_probe"
        model_dir = base_dir / "perltqa" / "mistral"
        _make_run(model_dir, "20260416_120000", dataset="longmemeval")

        result = _find_resume_dir(base_dir, "mistral", "perltqa")
        assert result is None

    def test_returns_newest_incomplete_run(self, tmp_path):
        """When multiple incomplete runs exist, the lexicographically latest is chosen."""
        base_dir = tmp_path / "outputs" / "dataset_probe"
        model_dir = base_dir / "perltqa" / "mistral"
        older = _make_run(model_dir, "20260416_100000", dataset="perltqa")
        newer = _make_run(model_dir, "20260416_120000", dataset="perltqa")

        result = _find_resume_dir(base_dir, "mistral", "perltqa")
        assert result == newer, f"Expected newest run {newer}, got {result}"
        assert result != older

    def test_different_datasets_do_not_cross_contaminate(self, tmp_path):
        """Runs for dataset A are not returned when searching for dataset B."""
        base_dir = tmp_path / "outputs" / "dataset_probe"
        _make_run(base_dir / "longmemeval" / "mistral", "20260416_120000", dataset="longmemeval")
        _make_run(base_dir / "perltqa" / "mistral", "20260416_110000", dataset="perltqa")

        result_perltqa = _find_resume_dir(base_dir, "mistral", "perltqa")
        result_longmemeval = _find_resume_dir(base_dir, "mistral", "longmemeval")

        assert result_perltqa is not None
        assert "perltqa" in str(result_perltqa)
        assert result_longmemeval is not None
        assert "longmemeval" in str(result_longmemeval)


# ---------------------------------------------------------------------------
# Helpers for _validate_resume_accumulator tests
# ---------------------------------------------------------------------------


def _write_raw_qa(run_dir: Path, sid: str, episodic_qa: list, procedural_rels: list) -> Path:
    """Write a raw_qa.json file under {run_dir}/diagnostics/{sid}.raw_qa.json.

    Args:
        run_dir: The probe run directory.
        sid: Session identifier used as the filename stem.
        episodic_qa: List of episodic QA pair dicts.
        procedural_rels: List of procedural relation dicts.

    Returns:
        Path to the written file.
    """
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    raw_qa_path = diag_dir / f"{sid}.raw_qa.json"
    raw_qa_path.write_text(
        json.dumps({"episodic_qa": episodic_qa, "procedural_rels": procedural_rels})
    )
    return raw_qa_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidateResumeAccumulator:
    """_validate_resume_accumulator must reload and concatenate raw_qa files."""

    def test_happy_path_two_sessions_concatenated(self, tmp_path):
        """Episodic QA from two sessions is concatenated in order.

        The returned tuple contains all QA pairs from both files,
        in the order of the processed_ids list.
        """
        sid1 = "session_001"
        sid2 = "session_002"

        ep1 = [{"q": "Q1", "a": "A1"}, {"q": "Q2", "a": "A2"}]
        pr1 = [{"subject": "X", "predicate": "likes", "object": "Y"}]
        ep2 = [{"q": "Q3", "a": "A3"}]
        pr2 = []

        _write_raw_qa(tmp_path, sid1, ep1, pr1)
        _write_raw_qa(tmp_path, sid2, ep2, pr2)

        all_episodic, all_procedural = _validate_resume_accumulator(tmp_path, [sid1, sid2])

        assert all_episodic == ep1 + ep2, (
            f"Episodic QA must be concatenated in sid order; got {all_episodic}"
        )
        assert all_procedural == pr1 + pr2, (
            f"Procedural rels must be concatenated in sid order; got {all_procedural}"
        )

    def test_empty_processed_ids_returns_empty_lists(self, tmp_path):
        """An empty processed_ids list returns ([], []) without touching disk."""
        all_episodic, all_procedural = _validate_resume_accumulator(tmp_path, [])
        assert all_episodic == []
        assert all_procedural == []

    def test_missing_raw_qa_raises_runtime_error(self, tmp_path):
        """A processed session_id with no raw_qa.json raises RuntimeError.

        The error message must contain the missing session_id and the word
        'missing', so callers can identify which file is absent.
        """
        # Write one valid file; the second is intentionally absent.
        _write_raw_qa(tmp_path, "session_present", [], [])

        with pytest.raises(RuntimeError, match="session_absent") as exc_info:
            _validate_resume_accumulator(tmp_path, ["session_present", "session_absent"])
        assert "missing" in str(exc_info.value).lower(), (
            "RuntimeError message must mention 'missing' for the absent file"
        )

    def test_corrupt_raw_qa_raises_runtime_error(self, tmp_path):
        """A raw_qa.json with invalid JSON content raises RuntimeError.

        The error message must reference 'parse' or 'corrupted' so callers
        understand the file is unreadable.
        """
        sid = "session_corrupt"
        diag_dir = tmp_path / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        (diag_dir / f"{sid}.raw_qa.json").write_text("{ this is not valid json !!!")

        with pytest.raises(RuntimeError) as exc_info:
            _validate_resume_accumulator(tmp_path, [sid])

        msg = str(exc_info.value).lower()
        assert "parse" in msg or "corrupted" in msg, (
            f"RuntimeError message must contain 'parse' or 'corrupted', got: {exc_info.value!r}"
        )

    def test_order_of_concatenation_matches_processed_ids_order(self, tmp_path):
        """QA pairs are appended in processed_ids order, not filesystem order."""
        sid_a = "zzz_last_alpha"
        sid_b = "aaa_first_alpha"

        # Unique marker QA pairs so we can verify ordering.
        ep_a = [{"q": "QA", "a": "AA"}]
        ep_b = [{"q": "QB", "a": "AB"}]

        _write_raw_qa(tmp_path, sid_a, ep_a, [])
        _write_raw_qa(tmp_path, sid_b, ep_b, [])

        # Request in sid_b-first order (alphabetically reversed from filenames).
        all_episodic, _ = _validate_resume_accumulator(tmp_path, [sid_b, sid_a])

        # sid_b comes first in the result because it is first in processed_ids.
        assert all_episodic[0] == ep_b[0], (
            "First QA pair must come from the first session in processed_ids"
        )
        assert all_episodic[1] == ep_a[0], (
            "Second QA pair must come from the second session in processed_ids"
        )


# ---------------------------------------------------------------------------
# Tests for _validate_no_train_resume
# ---------------------------------------------------------------------------


class TestNoTrainResumeValidation:
    """_validate_no_train_resume must catch --no-train mismatches on resume.

    Old runs without the ``no_train`` key are treated as False (backward
    compatible). Mismatches between the saved flag and the current invocation
    must raise ValueError so the caller can log and exit 1.
    """

    def test_resume_mismatch_on_no_train_errors(self):
        """Resuming a no_train=True run without --no-train raises ValueError.

        If the saved state has ``no_train=True`` but the current invocation
        passes ``no_train=False``, the validation must raise ValueError with
        a message that mentions the conflict.
        """
        state_with_no_train = {
            "dataset": "perltqa",
            "completed": False,
            "training_started": False,
            "no_train": True,
        }

        with pytest.raises(ValueError, match="no-train"):
            _validate_no_train_resume(state_with_no_train, no_train=False)

    def test_resume_allows_missing_no_train_field(self):
        """Old runs without no_train key resume fine when --no-train is absent.

        A state.json that predates the --no-train flag has no ``no_train``
        key. That absence must be treated as False, so resuming without
        --no-train (also False) raises no error.
        """
        legacy_state = {
            "dataset": "perltqa",
            "completed": False,
            "training_started": False,
            # Intentionally omit "no_train" to simulate an old-format state.
        }

        # Must not raise.
        _validate_no_train_resume(legacy_state, no_train=False)
