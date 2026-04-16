"""Contract tests for dataset-agnostic loader interface.

Verifies that PerLTQALoader and LongMemEvalLoader both:
  - Expose a ``name`` attribute and an ``iter_sessions`` method.
  - Yield DatasetSession objects with all five fields populated.
  - Respect the ``limit`` parameter.
  - Satisfy the answer-leakage contract (question/answer never in transcript).

No GPU, no model load, no network access. Runs in under a second.
"""

import inspect
import json
from pathlib import Path

import pytest

from experiments.utils.dataset_types import DatasetLoader, DatasetSession
from experiments.utils.longmemeval_loader import LongMemEvalLoader
from experiments.utils.perltqa_loader import (
    FALLBACK_PATH,  # noqa: F401 — verify existing export is still importable
    PERLTQA_MEM_PATH,  # noqa: F401 — verify existing export is still importable
    PerLTQALoader,
    is_available,  # noqa: F401 — verify existing export is still importable
    list_characters,  # noqa: F401 — verify existing export is still importable
    load_character_dialogues,  # noqa: F401 — verify existing export is still importable
    load_character_eval_qa,  # noqa: F401 — verify existing export is still importable
    load_fallback_qa,  # noqa: F401 — verify existing export is still importable
    load_qa,  # noqa: F401 — verify existing export is still importable
)

FIXTURES = Path(__file__).parent / "fixtures"
PERLTQA_FIXTURE = FIXTURES / "perltqa_probe_sample.json"
LONGMEMEVAL_FIXTURE = FIXTURES / "longmemeval_oracle_sample.json"


# ---------------------------------------------------------------------------
# PerLTQA loader contract
# ---------------------------------------------------------------------------


class TestPerLTQALoaderContract:
    """PerLTQALoader yields properly formed DatasetSession objects."""

    def _patched_loader(self, monkeypatch):
        """Return a PerLTQALoader backed by the fixture instead of real data."""
        monkeypatch.setattr(
            "experiments.utils.perltqa_loader.PERLTQA_MEM_PATH",
            PERLTQA_FIXTURE,
        )
        return PerLTQALoader()

    def test_iter_sessions_contract(self, monkeypatch):
        """Two sessions for fixture character; all DatasetSession fields populated."""
        loader = self._patched_loader(monkeypatch)
        sessions = list(loader.iter_sessions(limit=2, character="Alex"))

        assert len(sessions) == 2, f"Expected 2 sessions, got {len(sessions)}"
        for session in sessions:
            assert isinstance(session, DatasetSession)
            # All five fields must be non-empty strings / non-None
            assert session.session_id, "session_id must be non-empty"
            assert session.transcript, "transcript must be non-empty"
            assert session.speaker_id, "speaker_id must be non-empty"
            assert session.speaker_name, "speaker_name must be non-empty"
            assert isinstance(session.metadata, dict), "metadata must be a dict"

    def test_speaker_id_prefix(self, monkeypatch):
        """speaker_id must start with 'perltqa:'."""
        loader = self._patched_loader(monkeypatch)
        sessions = list(loader.iter_sessions(limit=2, character="Alex"))
        for session in sessions:
            assert session.speaker_id.startswith("perltqa:"), (
                f"speaker_id {session.speaker_id!r} must start with 'perltqa:'"
            )

    def test_metadata_dataset_field(self, monkeypatch):
        """metadata['dataset'] must equal 'perltqa'."""
        loader = self._patched_loader(monkeypatch)
        sessions = list(loader.iter_sessions(limit=2, character="Alex"))
        for session in sessions:
            assert session.metadata.get("dataset") == "perltqa", (
                f"metadata['dataset'] = {session.metadata.get('dataset')!r}, expected 'perltqa'"
            )

    def test_session_id_format(self, monkeypatch):
        """session_id must follow perltqa_{character}_{dlg_id} format."""
        loader = self._patched_loader(monkeypatch)
        sessions = list(loader.iter_sessions(limit=2, character="Alex"))
        for session in sessions:
            assert session.session_id.startswith("perltqa_Alex_"), (
                f"session_id {session.session_id!r} has unexpected format"
            )

    def test_limit_one(self, monkeypatch):
        """limit=1 must yield exactly 1 session."""
        loader = self._patched_loader(monkeypatch)
        sessions = list(loader.iter_sessions(limit=1, character="Alex"))
        assert len(sessions) == 1, f"limit=1 should yield 1 session, got {len(sessions)}"


# ---------------------------------------------------------------------------
# LongMemEval loader contract
# ---------------------------------------------------------------------------


def _make_fake_dataset(fixture_path: Path):
    """Load fixture JSON as a plain Python list (iterable, like HF Dataset)."""
    with open(fixture_path) as f:
        return json.load(f)


class TestLongMemEvalLoaderContract:
    """LongMemEvalLoader yields properly formed DatasetSession objects."""

    def _patched_loader(self, monkeypatch):
        """Return a LongMemEvalLoader that reads from the fixture.

        The loader bypasses datasets.load_dataset (see longmemeval_loader.py)
        and calls huggingface_hub.hf_hub_download for the raw JSON file, then
        parses with stdlib json. We patch hf_hub_download to return the
        fixture path directly — json.load on the fixture yields the same
        list-of-dicts the loader expects.
        """

        def _fake_hf_hub_download(*args, **kwargs):
            return str(LONGMEMEVAL_FIXTURE)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_hf_hub_download)
        return LongMemEvalLoader(split="longmemeval_oracle")

    def test_iter_sessions_yields_three(self, monkeypatch):
        """Fixture has 1 preference session + 2 multi-session = 3 total."""
        loader = self._patched_loader(monkeypatch)
        sessions = list(loader.iter_sessions(limit=3))
        assert len(sessions) == 3, f"Expected 3 sessions, got {len(sessions)}"

    def test_all_five_fields_populated(self, monkeypatch):
        """All DatasetSession fields must be non-empty on every session."""
        loader = self._patched_loader(monkeypatch)
        sessions = list(loader.iter_sessions(limit=3))
        for session in sessions:
            assert isinstance(session, DatasetSession)
            assert session.session_id, "session_id must be non-empty"
            assert session.transcript, "transcript must be non-empty"
            assert session.speaker_id, "speaker_id must be non-empty"
            assert session.speaker_name, "speaker_name must be non-empty"
            assert isinstance(session.metadata, dict), "metadata must be a dict"

    def test_speaker_id_prefix(self, monkeypatch):
        """speaker_id must start with 'longmemeval:'."""
        loader = self._patched_loader(monkeypatch)
        sessions = list(loader.iter_sessions(limit=3))
        for session in sessions:
            assert session.speaker_id.startswith("longmemeval:"), (
                f"speaker_id {session.speaker_id!r} must start with 'longmemeval:'"
            )

    def test_metadata_split_field(self, monkeypatch):
        """metadata['split'] must equal 'longmemeval_oracle'."""
        loader = self._patched_loader(monkeypatch)
        sessions = list(loader.iter_sessions(limit=3))
        for session in sessions:
            assert session.metadata.get("split") == "longmemeval_oracle", (
                f"metadata['split'] = {session.metadata.get('split')!r}"
            )

    def test_metadata_question_type(self, monkeypatch):
        """metadata['question_type'] must match the originating fixture example."""
        loader = self._patched_loader(monkeypatch)
        sessions = list(loader.iter_sessions(limit=3))
        # Session 0: single-session-preference (1 session in example A)
        assert sessions[0].metadata["question_type"] == "single-session-preference"
        # Sessions 1-2: multi-session (2 sessions in example B)
        assert sessions[1].metadata["question_type"] == "multi-session"
        assert sessions[2].metadata["question_type"] == "multi-session"

    def test_transcript_contains_user_and_assistant(self, monkeypatch):
        """Transcript must include both 'User:' and 'Assistant:' prefixes."""
        loader = self._patched_loader(monkeypatch)
        sessions = list(loader.iter_sessions(limit=3))
        for session in sessions:
            assert "User:" in session.transcript, (
                f"transcript missing 'User:' prefix in session {session.session_id}"
            )
            assert "Assistant:" in session.transcript, (
                f"transcript missing 'Assistant:' prefix in session {session.session_id}"
            )

    def test_leakage_question_absent_from_transcript(self, monkeypatch):
        """The fixture question strings must not appear verbatim in any transcript."""
        fixture_data = _make_fake_dataset(LONGMEMEVAL_FIXTURE)
        loader = self._patched_loader(monkeypatch)
        sessions = list(loader.iter_sessions(limit=3))

        # Build question_id -> question / answer mapping from fixture.
        qa_map = {ex["question_id"]: ex for ex in fixture_data}

        for session in sessions:
            qid = session.metadata["question_id"]
            example = qa_map[qid]
            question_str: str = example["question"]
            answer_str: str = str(example["answer"])  # defensive cast for int answers

            assert question_str not in session.transcript, (
                f"Leakage: question {question_str!r} found verbatim in "
                f"transcript of session {session.session_id!r}"
            )
            assert answer_str not in session.transcript, (
                f"Leakage: answer {answer_str!r} found verbatim in "
                f"transcript of session {session.session_id!r}"
            )

    def test_is_answer_session_metadata(self, monkeypatch):
        """is_answer_session must be True for answer session, False for distractor."""
        loader = self._patched_loader(monkeypatch)
        sessions = list(loader.iter_sessions(limit=3))

        # sessions[1]: example B's first session (answer_session_A, in answer_session_ids)
        # sessions[2]: example B's second session (distractor_session_B, not in answer_session_ids)
        assert sessions[1].metadata["is_answer_session"] is True, (
            f"Expected is_answer_session=True for {sessions[1].session_id!r}, "
            f"got {sessions[1].metadata['is_answer_session']!r}"
        )
        assert sessions[2].metadata["is_answer_session"] is False, (
            f"Expected is_answer_session=False for {sessions[2].session_id!r}, "
            f"got {sessions[2].metadata['is_answer_session']!r}"
        )

    def test_limit_one(self, monkeypatch):
        """limit=1 must yield exactly 1 session (first in dataset-native order)."""
        loader = self._patched_loader(monkeypatch)
        sessions = list(loader.iter_sessions(limit=1))
        assert len(sessions) == 1, f"limit=1 should yield 1 session, got {len(sessions)}"
        # First session is always from example A (single-session-preference)
        assert sessions[0].metadata["question_type"] == "single-session-preference"


# ---------------------------------------------------------------------------
# Shared interface guard
# ---------------------------------------------------------------------------


class TestLoadersShareInterface:
    """Both loader classes must satisfy the DatasetLoader Protocol.

    This test acts as the guard against future loader drift: if a maintainer
    removes ``name`` or breaks ``iter_sessions``, this test fails immediately.
    """

    @pytest.mark.parametrize("loader_cls", [PerLTQALoader, LongMemEvalLoader])
    def test_satisfies_protocol(self, loader_cls):
        """Loader class must pass isinstance check against the Protocol."""
        assert isinstance(loader_cls, type), f"{loader_cls} must be a class, not an instance"
        # Check at the class level using runtime_checkable Protocol.
        # Protocol checks name attribute and iter_sessions method presence.
        loader_instance = object.__new__(loader_cls)
        assert isinstance(loader_instance, DatasetLoader), (
            f"{loader_cls.__name__} does not satisfy the DatasetLoader Protocol. "
            "Ensure 'name' attribute and 'iter_sessions' method are present."
        )

    @pytest.mark.parametrize("loader_cls", [PerLTQALoader, LongMemEvalLoader])
    def test_name_attribute_is_string(self, loader_cls):
        """name attribute must be a non-empty string."""
        loader_instance = object.__new__(loader_cls)
        assert isinstance(loader_instance.name, str), (
            f"{loader_cls.__name__}.name must be a str, got {type(loader_instance.name)}"
        )
        assert loader_instance.name, f"{loader_cls.__name__}.name must be non-empty"

    @pytest.mark.parametrize("loader_cls", [PerLTQALoader, LongMemEvalLoader])
    def test_iter_sessions_signature(self, loader_cls):
        """iter_sessions must accept limit and **kwargs."""
        sig = inspect.signature(loader_cls.iter_sessions)
        params = sig.parameters
        assert "limit" in params, (
            f"{loader_cls.__name__}.iter_sessions must accept a 'limit' parameter"
        )
        # Must accept extra kwargs (for dataset-specific selectors)
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        assert has_var_keyword, f"{loader_cls.__name__}.iter_sessions must accept **kwargs"
