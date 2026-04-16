"""Tests for _commit_session write-order contract in experiments/dataset_probe.py.

The commit path must write in this order:
  1. {sid}.raw_qa.json      — accumulator input, needed for crash-resume
  2. {sid}.json             — per-session diagnostics
  3. state.json             — commit point; session is "done" only after this

A crash between steps 2 and 3 leaves the session un-committed in state.json
but with raw_qa.json present, so _validate_resume_accumulator can recover.
Reversing the order would cause unrecoverable data loss on partial crashes.

No GPU, no model load, no network access.
"""

from pathlib import Path

from experiments.dataset_probe import _commit_session
from experiments.utils.dataset_types import DatasetSession

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(session_id: str = "sess_001") -> DatasetSession:
    """Return a minimal DatasetSession with the given session_id.

    Args:
        session_id: Identifier for the session.

    Returns:
        DatasetSession with empty transcript and minimal metadata.
    """
    return DatasetSession(
        session_id=session_id,
        transcript="User: hi\nAssistant: hello",
        speaker_id="sp1",
        speaker_name="Bob",
        metadata={"dataset": "perltqa"},
    )


def _base_state(dataset: str = "perltqa") -> dict:
    """Return a minimal mutable state dict, as _load_state would produce.

    Args:
        dataset: Dataset name stored in state.

    Returns:
        State dict with the expected skeleton fields.
    """
    return {
        "dataset": dataset,
        "completed": False,
        "training_started": False,
        "processed_session_ids": [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCommitSessionWriteOrder:
    """_commit_session must write raw_qa → diagnostics → state in that order."""

    def test_three_writes_in_correct_order(self, tmp_path, monkeypatch):
        """_atomic_json_write is called exactly three times in the required order.

        Write 1: path ends with .raw_qa.json
        Write 2: path ends with .json  (diagnostics, without .raw_qa prefix)
        Write 3: path ends with state.json
        """
        write_log: list[tuple[Path, object]] = []

        import experiments.dataset_probe as dp

        monkeypatch.setattr(
            dp, "_atomic_json_write", lambda data, path: write_log.append((path, data))
        )

        session = _make_session("sess_001")
        state = _base_state()
        diag_data = {"session_id": "sess_001", "extraction": {}}

        _commit_session(
            run_dir=tmp_path,
            session=session,
            episodic_qa=[{"q": "Q1", "a": "A1"}],
            procedural_rels=[],
            diag_data=diag_data,
            state=state,
        )

        assert len(write_log) == 3, (
            f"Expected exactly 3 _atomic_json_write calls, got {len(write_log)}: "
            f"{[str(p) for p, _ in write_log]}"
        )

        path1, path2, path3 = (p for p, _ in write_log)

        # Write 1: raw QA file
        assert str(path1).endswith(".raw_qa.json"), (
            f"First write must be *.raw_qa.json, got {path1}"
        )

        # Write 2: per-session diagnostics (ends in .json but NOT .raw_qa.json)
        assert str(path2).endswith(".json") and not str(path2).endswith(".raw_qa.json"), (
            f"Second write must be a *.json diagnostics file, got {path2}"
        )

        # Write 3: state.json commit point
        assert path3.name == "state.json", f"Third write must be state.json, got {path3}"

    def test_raw_qa_contains_episodic_and_procedural(self, tmp_path, monkeypatch):
        """The raw_qa payload written in step 1 has the expected shape."""
        captured: list[tuple[Path, object]] = []

        import experiments.dataset_probe as dp

        monkeypatch.setattr(
            dp, "_atomic_json_write", lambda data, path: captured.append((path, data))
        )

        episodic = [{"q": "Q1", "a": "A1"}, {"q": "Q2", "a": "A2"}]
        proc = [{"subject": "X", "predicate": "likes", "object": "Y"}]

        _commit_session(
            run_dir=tmp_path,
            session=_make_session(),
            episodic_qa=episodic,
            procedural_rels=proc,
            diag_data={},
            state=_base_state(),
        )

        path1, data1 = captured[0]
        assert str(path1).endswith(".raw_qa.json")
        assert data1["episodic_qa"] == episodic
        assert data1["procedural_rels"] == proc

    def test_session_id_in_processed_session_ids_after_commit(self, tmp_path, monkeypatch):
        """After _commit_session, the session_id appears in state['processed_session_ids']."""
        import experiments.dataset_probe as dp

        monkeypatch.setattr(dp, "_atomic_json_write", lambda data, path: None)

        state = _base_state()
        _commit_session(
            run_dir=tmp_path,
            session=_make_session("sess_abc"),
            episodic_qa=[],
            procedural_rels=[],
            diag_data={},
            state=state,
        )

        assert "sess_abc" in state["processed_session_ids"]

    def test_duplicate_commit_does_not_duplicate_session_id(self, tmp_path, monkeypatch):
        """Calling _commit_session twice with the same session_id is idempotent.

        The session_id must appear exactly once in processed_session_ids.
        """
        import experiments.dataset_probe as dp

        monkeypatch.setattr(dp, "_atomic_json_write", lambda data, path: None)

        state = _base_state()

        _commit_session(
            run_dir=tmp_path,
            session=_make_session("sess_dup"),
            episodic_qa=[],
            procedural_rels=[],
            diag_data={},
            state=state,
        )
        _commit_session(
            run_dir=tmp_path,
            session=_make_session("sess_dup"),
            episodic_qa=[],
            procedural_rels=[],
            diag_data={},
            state=state,
        )

        count = state["processed_session_ids"].count("sess_dup")
        assert count == 1, (
            f"session_id 'sess_dup' should appear exactly once in processed_session_ids, "
            f"but appeared {count} times: {state['processed_session_ids']}"
        )

    def test_raw_qa_path_uses_session_id(self, tmp_path, monkeypatch):
        """The raw_qa file path includes the session_id as filename stem."""
        captured_paths: list[Path] = []

        import experiments.dataset_probe as dp

        monkeypatch.setattr(
            dp, "_atomic_json_write", lambda data, path: captured_paths.append(path)
        )

        _commit_session(
            run_dir=tmp_path,
            session=_make_session("my_unique_session"),
            episodic_qa=[],
            procedural_rels=[],
            diag_data={},
            state=_base_state(),
        )

        raw_qa_path = captured_paths[0]
        assert "my_unique_session" in raw_qa_path.name, (
            f"raw_qa filename must contain the session_id, got {raw_qa_path.name}"
        )
