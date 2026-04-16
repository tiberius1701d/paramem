"""Contract tests for dataset-agnostic loader interface.

Verifies that PerLTQALoader and LongMemEvalLoader both:
  - Expose a ``name`` attribute and an ``iter_sessions`` method.
  - Yield DatasetSession objects with all five fields populated.
  - Respect the ``limit`` parameter.
  - Satisfy the answer-leakage contract (question/answer never in transcript).

Also verifies stratified sampling behaviour for LongMemEvalLoader:
  - Correct total count, per-bucket balance, and determinism across seeds.
  - Short-bucket redistribution with logged warnings.
  - Validation errors for bad constructor arguments.

No GPU, no model load, no network access. Runs in under a second.
"""

import inspect
import json
import logging
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
LONGMEMEVAL_UNEVEN_FIXTURE = FIXTURES / "longmemeval_oracle_uneven.json"


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
        """limit=3 yields exactly 3 sessions from the fixture (which has 9 total)."""
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


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


class TestLongMemEvalStratified:
    """LongMemEvalLoader stratified sampling — correctness, determinism, edge cases.

    The extended fixture has 8 examples across 4 question_types:
      - single-session-preference: 2 examples, 1 haystack session each (2 sessions)
      - multi-session:             2 examples, 2 and 1 haystack sessions   (3 sessions)
      - single-session-user:       2 examples, 1 haystack session each (2 sessions)
      - temporal-reasoning:        2 examples, 1 haystack session each (2 sessions)
    Total: 9 sessions across 4 buckets.

    All tests use _patched_loader to bypass HF network access.
    """

    def _patched_loader(self, monkeypatch, **kwargs):
        """Return a LongMemEvalLoader backed by the fixture.

        Args:
            monkeypatch: pytest monkeypatch fixture.
            **kwargs: Forwarded to LongMemEvalLoader constructor alongside the
                mandatory ``split="longmemeval_oracle"`` argument.

        Returns:
            LongMemEvalLoader instance wired to the local fixture file.
        """

        def _fake_hf_hub_download(*args, **kw):
            return str(LONGMEMEVAL_FIXTURE)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_hf_hub_download)
        return LongMemEvalLoader(split="longmemeval_oracle", **kwargs)

    # ------------------------------------------------------------------
    # 1. Correct total count
    # ------------------------------------------------------------------

    def test_stratified_returns_requested_count(self, monkeypatch):
        """sample_size=4 with 4 buckets yields exactly 4 sessions."""
        loader = self._patched_loader(
            monkeypatch, sample_strategy="stratified", sample_size=4, sample_seed=42
        )
        sessions = list(loader.iter_sessions())
        assert len(sessions) == 4, f"Expected 4 sessions, got {len(sessions)}"

    # ------------------------------------------------------------------
    # 2. Balance across buckets
    # ------------------------------------------------------------------

    def test_stratified_balanced_across_buckets(self, monkeypatch):
        """sample_size=4, 4 buckets → exactly 1 session per question_type."""
        loader = self._patched_loader(
            monkeypatch, sample_strategy="stratified", sample_size=4, sample_seed=42
        )
        sessions = list(loader.iter_sessions())
        counts: dict[str, int] = {}
        for s in sessions:
            qt = s.metadata["question_type"]
            counts[qt] = counts.get(qt, 0) + 1

        assert len(counts) == 4, f"Expected sessions from 4 question_types, got {counts}"
        for qt, count in counts.items():
            assert count == 1, f"Bucket {qt!r} got {count} sessions, expected 1"

    # ------------------------------------------------------------------
    # 3. Determinism: same seed → identical session_id sequence
    # ------------------------------------------------------------------

    def test_stratified_deterministic_same_seed(self, monkeypatch):
        """Two calls with the same seed yield identical session_id sequences."""
        loader_a = self._patched_loader(
            monkeypatch, sample_strategy="stratified", sample_size=4, sample_seed=42
        )
        loader_b = self._patched_loader(
            monkeypatch, sample_strategy="stratified", sample_size=4, sample_seed=42
        )
        ids_a = [s.session_id for s in loader_a.iter_sessions()]
        ids_b = [s.session_id for s in loader_b.iter_sessions()]
        assert ids_a == ids_b, (
            f"Same seed produced different session_id sequences:\n{ids_a}\nvs\n{ids_b}"
        )

    # ------------------------------------------------------------------
    # 5. Uneven distribution (remainder allocation)
    # ------------------------------------------------------------------

    def test_stratified_uneven_distribution(self, monkeypatch):
        """sample_size=5, 4 buckets → sum==5 and max-min<=1 (fair remainder)."""
        loader = self._patched_loader(
            monkeypatch, sample_strategy="stratified", sample_size=5, sample_seed=42
        )
        sessions = list(loader.iter_sessions())
        assert len(sessions) == 5, f"Expected 5 sessions, got {len(sessions)}"

        counts: dict[str, int] = {}
        for s in sessions:
            qt = s.metadata["question_type"]
            counts[qt] = counts.get(qt, 0) + 1

        assert len(counts) == 4, f"Expected all 4 buckets represented, got {counts}"
        max_c = max(counts.values())
        min_c = min(counts.values())
        assert max_c - min_c <= 1, (
            f"Uneven distribution exceeds 1: max={max_c}, min={min_c}, counts={counts}"
        )

    # ------------------------------------------------------------------
    # 6. Short bucket redistribution + warning
    # ------------------------------------------------------------------

    def test_stratified_short_bucket_redistribution(self, monkeypatch):
        """When sample_size exceeds total available sessions, shortfall is redistributed.

        Fixture buckets (sorted alphabetically):
          multi-session             : 3 sessions
          single-session-preference : 2 sessions
          single-session-user       : 2 sessions
          temporal-reasoning        : 2 sessions
        Total: 9 sessions.

        With sample_size=10 and 4 buckets: divmod(10,4) = (2, 2), so the
        first two sorted buckets (multi-session, single-session-preference)
        each get quota=3.  multi-session has 3 — satisfied.  But
        single-session-preference has only 2, creating a shortfall of 1.
        That shortfall is redistributed to multi-session (already exhausted),
        which cannot absorb it.  The loader must not raise; it yields all 9
        available sessions instead of the requested 10.

        Warning emission is also verified via a logging handler installed
        directly on the loader's named logger, bypassing the unreliable
        caplog fixture in this test environment.
        """
        loader_logger = logging.getLogger("experiments.utils.longmemeval_loader")
        warning_messages: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                warning_messages.append(record.getMessage())

        handler = _Capture(level=logging.WARNING)
        loader_logger.addHandler(handler)
        try:
            loader = self._patched_loader(
                monkeypatch, sample_strategy="stratified", sample_size=10, sample_seed=42
            )
            sessions = list(loader.iter_sessions())
        finally:
            loader_logger.removeHandler(handler)

        # Fixture has 9 sessions total — all should be yielded (sample_size > available).
        assert len(sessions) == 9, f"Expected 9 sessions (all available), got {len(sessions)}"
        # At least one "short" warning must have been emitted.
        short_messages = [m for m in warning_messages if "short" in m.lower()]
        assert short_messages, (
            "Expected at least one 'short' warning when quota exceeds bucket size. "
            f"Emitted warnings: {warning_messages}"
        )

    # ------------------------------------------------------------------
    # 7. Requires sample_size
    # ------------------------------------------------------------------

    def test_stratified_requires_sample_size(self, monkeypatch):
        """sample_strategy='stratified' without sample_size raises ValueError."""
        with pytest.raises(ValueError, match="sample_size"):
            self._patched_loader(monkeypatch, sample_strategy="stratified")

    # ------------------------------------------------------------------
    # 8. Rejects non-positive sample_size
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("bad_size", [0, -1])
    def test_stratified_rejects_non_positive_size(self, monkeypatch, bad_size):
        """sample_size=0 or sample_size=-1 raises ValueError."""
        with pytest.raises(ValueError, match="sample_size"):
            self._patched_loader(monkeypatch, sample_strategy="stratified", sample_size=bad_size)

    # ------------------------------------------------------------------
    # 9. None strategy is identical to default
    # ------------------------------------------------------------------

    def test_none_strategy_byte_identical_to_default(self, monkeypatch):
        """Explicit sample_strategy='none' yields same sessions as default constructor."""
        loader_explicit = self._patched_loader(monkeypatch, sample_strategy="none")
        loader_default = self._patched_loader(monkeypatch)
        ids_explicit = [s.session_id for s in loader_explicit.iter_sessions()]
        ids_default = [s.session_id for s in loader_default.iter_sessions()]
        assert ids_explicit == ids_default, (
            "sample_strategy='none' diverges from default constructor path.\n"
            f"explicit: {ids_explicit}\ndefault:  {ids_default}"
        )

    # ------------------------------------------------------------------
    # 10. Unknown strategy raises ValueError
    # ------------------------------------------------------------------

    def test_unknown_strategy_raises(self, monkeypatch):
        """sample_strategy='foo' raises ValueError at construction time."""
        with pytest.raises(ValueError, match="sample_strategy"):
            self._patched_loader(monkeypatch, sample_strategy="foo")

    # ------------------------------------------------------------------
    # 11. limit × stratified interaction (secondary cap)
    # ------------------------------------------------------------------

    def test_stratified_with_limit_cap(self, monkeypatch):
        """limit=3 applied on top of sample_size=4 yields exactly 3 sessions.

        Exercises the secondary-cap code path in _iter_stratified that
        applies ``limit`` after the stratified sort step.
        """
        loader = self._patched_loader(
            monkeypatch, sample_strategy="stratified", sample_size=4, sample_seed=42
        )
        sessions = list(loader.iter_sessions(limit=3))
        assert len(sessions) == 3, (
            f"Expected 3 sessions with limit=3 on a sample_size=4 stratified loader, "
            f"got {len(sessions)}"
        )

    # ------------------------------------------------------------------
    # 12. All buckets short: yield all available + WARNING logged
    # ------------------------------------------------------------------

    def test_stratified_all_buckets_short(self, monkeypatch):
        """sample_size=100 on 9-session fixture yields all 9 and logs a WARNING.

        When every bucket is exhausted before reaching its quota, the loader
        must yield all available sessions (not raise) and emit at least one
        WARNING-level log record about the shortfall.

        Uses the direct-handler pattern from test_stratified_short_bucket_redistribution
        to capture log output reliably.
        """
        loader_logger = logging.getLogger("experiments.utils.longmemeval_loader")
        warning_messages: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                warning_messages.append(record.getMessage())

        handler = _Capture(level=logging.WARNING)
        loader_logger.addHandler(handler)
        try:
            loader = self._patched_loader(
                monkeypatch, sample_strategy="stratified", sample_size=100, sample_seed=42
            )
            sessions = list(loader.iter_sessions())
        finally:
            loader_logger.removeHandler(handler)

        # All 9 sessions in the fixture must be yielded.
        assert len(sessions) == 9, (
            f"Expected 9 sessions (all available) when sample_size=100 on 9-session "
            f"fixture, got {len(sessions)}"
        )
        # At least one WARNING about the shortfall must have been emitted.
        short_messages = [m for m in warning_messages if "short" in m.lower()]
        assert short_messages, (
            "Expected at least one 'short' WARNING when sample_size exceeds total "
            f"available sessions. Emitted warnings: {warning_messages}"
        )

    # ------------------------------------------------------------------
    # 13. Different seeds → different picks (robust regression guard)
    # ------------------------------------------------------------------

    def test_stratified_different_seed_different_picks(self, monkeypatch):
        """At least one seed pair in a small set must yield different session_id sequences.

        Iterates over several seed pairs. If all pairs coincidentally produce
        identical session_id sequences, the shuffle is broken; pytest.fail is
        raised with a descriptive message. The multi-session bucket has 3
        sessions, making collisions extremely unlikely with any working shuffle.
        """
        seed_pairs = [(42, 1), (42, 2), (42, 7), (42, 99), (42, 123)]
        found_difference = False
        for seed_a, seed_b in seed_pairs:
            loader_a = self._patched_loader(
                monkeypatch, sample_strategy="stratified", sample_size=4, sample_seed=seed_a
            )
            loader_b = self._patched_loader(
                monkeypatch, sample_strategy="stratified", sample_size=4, sample_seed=seed_b
            )
            ids_a = [s.session_id for s in loader_a.iter_sessions()]
            ids_b = [s.session_id for s in loader_b.iter_sessions()]
            if ids_a != ids_b:
                found_difference = True
                break
        if not found_difference:
            pytest.fail(
                "All seed pairs produced identical session_id sequences. "
                "The per-bucket shuffle may be broken. Tested pairs: " + str(seed_pairs)
            )

    # ------------------------------------------------------------------
    # 14. Multi-pass redistribution: residual shortfall is fully resolved
    # ------------------------------------------------------------------

    def test_stratified_multipass_redistribution(self, monkeypatch):
        """Multi-pass redistribution fills residual shortfall left by the first pass.

        Fixture ``longmemeval_oracle_uneven.json`` has 4 question_type buckets
        (alphabetical order) with available session counts [1, 3, 3, 3]:
          - multi-session             : 1 session
          - single-session-preference : 3 sessions
          - single-session-user       : 3 sessions
          - temporal-reasoning        : 3 sessions
        Total: 10 sessions.

        With sample_size=10, divmod(10, 4) = (2, 2), so quotas are [3, 3, 2, 2]
        (first two sorted buckets get the +1 remainder).

        First pass:
          - multi-session (quota=3, available=1): takes 1, shortfall=2.
          - single-session-preference (quota=3, available=3): takes 3.
          - single-session-user (quota=2, available=3): takes 2, headroom=1.
          - temporal-reasoning (quota=2, available=3): takes 2, headroom=1.
          total_picked=8, shortfall=2.

        Second pass (first redistribution):
          Eligible: single-session-user (headroom 1), temporal-reasoning (headroom 1).
          extra_base=1, extra_rem=0 → each absorbs 1.
          total_picked=10, shortfall=0.  Done — no residual.

        A single-pass implementation would have attempted single-session-preference
        first (already exhausted → 0 absorbed), then single-session-user (+1), then
        temporal-reasoning (+0 due to rounding) → yields 9, not 10.

        Assertion: total yielded == min(sample_size, total_available) == 10.
        """

        def _fake_hf_hub_download(*args, **kw):
            return str(LONGMEMEVAL_UNEVEN_FIXTURE)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_hf_hub_download)

        loader = LongMemEvalLoader(
            split="longmemeval_oracle",
            sample_strategy="stratified",
            sample_size=10,
            sample_seed=42,
        )
        sessions = list(loader.iter_sessions())

        total_available = 10  # fixture has exactly 10 sessions
        assert len(sessions) == min(10, total_available), (
            f"Multi-pass redistribution failed: expected {min(10, total_available)} sessions, "
            f"got {len(sessions)}. A single-pass implementation would have yielded 9."
        )
