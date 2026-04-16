"""LongMemEval dataset loader for ParaMem dataset-agnostic probing.

Loads the xiaowu0162/longmemeval-cleaned HuggingFace dataset and yields
DatasetSession objects normalized to the common loader interface.

Pinned revision ensures reproducibility:
    LONGMEMEVAL_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"

Three splits are supported:
  - longmemeval_oracle (default) — 500 examples, 948 sessions, evidence-only
  - longmemeval_s_cleaned        — schema parity with oracle unconfirmed
  - longmemeval_m_cleaned        — schema parity with oracle unconfirmed

Session iteration order: dataset-native example order × parallel-array index
within each example. ``limit`` caps the total number of yielded sessions.

When ``sample_strategy="stratified"`` is used, sessions are bucketed by
``question_type`` and a balanced per-bucket sample of ``sample_size`` total
sessions is yielded in (example_idx, session_idx_in_example) order.

Cache location: data/external/longmemeval/ (gitignored; datasets library
manages nested directories internally).

Answer-leakage contract:
  The example's top-level ``question`` and ``answer`` fields are audit-only
  metadata and MUST NOT appear in the rendered transcript. The transcript is
  built exclusively from ``haystack_sessions`` turns (including has_answer=True
  turns, which are valid conversation content, not leakage).

References:
  - LongMemEval paper: https://arxiv.org/abs/2410.10813
  - HF dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
  - License: MIT (ICLR 2025)
"""

import logging
import random
from pathlib import Path
from typing import Iterator

from experiments.utils.dataset_types import DatasetSession

logger = logging.getLogger(__name__)

# Pinned revision for reproducibility. Unpinned loaders turn HF-side
# schema drift into extraction-quality drift with no test failure.
LONGMEMEVAL_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"

# Valid split values (only oracle schema parity is confirmed).
_VALID_SPLITS = frozenset(
    {
        "longmemeval_oracle",
        "longmemeval_s_cleaned",
        "longmemeval_m_cleaned",
    }
)
_UNCONFIRMED_SPLITS = frozenset({"longmemeval_s_cleaned", "longmemeval_m_cleaned"})

# Valid sample strategy values.
_VALID_STRATEGIES = frozenset({"none", "stratified"})

# Default cache location alongside the PerLTQA cache under data/external/.
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / "data" / "external" / "longmemeval"


class LongMemEvalLoader:
    """HuggingFace-backed loader yielding DatasetSession for LongMemEval.

    Args:
        split: HF split name. One of ``longmemeval_oracle`` (default),
            ``longmemeval_s_cleaned``, or ``longmemeval_m_cleaned``.
            S/M splits log a WARNING about unconfirmed schema parity.
        cache_dir: Override the local HF cache directory. If None, uses
            ``data/external/longmemeval/`` under the project root.
        sample_strategy: Sampling strategy. One of ``"none"`` (default,
            yields all sessions in dataset order) or ``"stratified"``
            (yields a balanced sample across ``question_type`` buckets).
            Unknown values raise ``ValueError``.
        sample_size: Total number of sessions to yield when
            ``sample_strategy="stratified"``. Must be a positive integer.
            Required when ``sample_strategy="stratified"``; ignored when
            ``sample_strategy="none"``.
        sample_seed: Random seed for deterministic per-bucket shuffling
            when ``sample_strategy="stratified"``. Default: 42.

    Raises:
        ValueError: If ``split`` is not a recognised split name.
        ValueError: If ``sample_strategy`` is not a recognised strategy.
        ValueError: If ``sample_strategy="stratified"`` and ``sample_size``
            is ``None`` or not a positive integer.

    Attributes:
        name: Loader identifier for the dataset registry ("longmemeval").
    """

    name: str = "longmemeval"

    def __init__(
        self,
        split: str = "longmemeval_oracle",
        cache_dir: Path | None = None,
        sample_strategy: str = "none",
        sample_size: int | None = None,
        sample_seed: int = 42,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"Unknown split {split!r}. Valid splits: {sorted(_VALID_SPLITS)}")
        if split in _UNCONFIRMED_SPLITS:
            logger.warning(
                "LongMemEvalLoader: split %r — schema parity with oracle is "
                "unconfirmed. First run must be manually reviewed before "
                "trusting diagnostics from this split.",
                split,
            )
        if sample_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown sample_strategy {sample_strategy!r}. "
                f"Valid strategies: {sorted(_VALID_STRATEGIES)}"
            )
        if sample_strategy == "stratified" and (sample_size is None or sample_size <= 0):
            raise ValueError(
                f"sample_size must be a positive integer when "
                f"sample_strategy='stratified', got {sample_size!r}"
            )
        self._split = split
        self._cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
        self._sample_strategy = sample_strategy
        self._sample_size = sample_size
        self._sample_seed = sample_seed

    def iter_sessions(
        self,
        limit: int | None = None,
        **kwargs,  # absorb dataset_probe kwargs (e.g. character — not used here)
    ) -> Iterator[DatasetSession]:
        """Yield DatasetSession objects from the LongMemEval dataset.

        When ``sample_strategy="none"`` (the default): sessions are yielded in
        dataset-native example order × parallel-array index within each
        example's haystack. ``limit`` caps total yielded sessions.

        When ``sample_strategy="stratified"``: sessions are bucketed by
        ``question_type``. Each bucket is deterministically shuffled with
        ``sample_seed``. A balanced quota of ``sample_size`` total sessions is
        drawn from all buckets (round-robin remainder allocation across buckets
        sorted by name). Short buckets trigger a logged warning and redistribute
        their shortfall. The final pick list is sorted by
        (example_idx, session_idx_in_example) before yielding so output order
        is deterministic and reproducible. ``limit`` is applied on top of the
        stratified set as a secondary cap.

        The rendered transcript includes all turns (including has_answer=True
        turns, which are conversation content, not leakage). The top-level
        ``question`` and ``answer`` fields are never concatenated into the
        transcript.

        Args:
            limit: Maximum number of sessions to yield. None = all. When
                ``sample_strategy="stratified"``, acts as a secondary cap on
                top of ``sample_size``.
            **kwargs: Ignored; present so the loader satisfies the Protocol
                without keyword mismatches from dataset_probe.py.

        Yields:
            DatasetSession with all five fields populated.
        """
        # NOTE: We bypass `datasets.load_dataset` because its JSON-array
        # code path sets `pyarrow._json.ReadOptions.block_size = len(batch)`
        # on this repo's files (single-JSON-array layout) and overflows
        # int32_t on files larger than ~2 GiB — and crashes earlier on
        # smaller files if batch growth triggers the x2 backoff loop.
        # `hf_hub_download` preserves the pinned-revision promise while
        # letting us parse with stdlib json. See huggingface/datasets#6501.
        try:
            import json as _json

            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "The 'huggingface_hub' package is required for LongMemEvalLoader. "
                "Install via: pip install huggingface_hub"
            ) from exc

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Loading LongMemEval split=%r revision=%s cache=%s",
            self._split,
            LONGMEMEVAL_REVISION[:8],
            self._cache_dir,
        )
        filename = f"{self._split}.json"
        local_path = hf_hub_download(
            repo_id="xiaowu0162/longmemeval-cleaned",
            filename=filename,
            revision=LONGMEMEVAL_REVISION,
            repo_type="dataset",
            cache_dir=str(self._cache_dir),
        )
        with open(local_path, encoding="utf-8") as f:
            ds = _json.load(f)

        if self._sample_strategy == "stratified":
            yield from self._iter_stratified(ds, limit)
        else:
            yield from self._iter_all(ds, limit)

    def _iter_all(
        self,
        ds: list,
        limit: int | None,
    ) -> Iterator[DatasetSession]:
        """Yield all sessions in dataset-native order, capped by ``limit``.

        This is the original iteration path; ``sample_strategy="none"`` routes
        here. Behaviour is identical to the pre-stratified implementation.

        Args:
            ds: Parsed dataset list (one dict per example).
            limit: Maximum number of sessions to yield. None = all.

        Yields:
            DatasetSession with all five fields populated.
        """
        yielded = 0
        for example in ds:
            question_id: str = example["question_id"]
            question_type: str = example["question_type"]
            question_date: str = example["question_date"]
            answer_session_ids: list[str] = example["answer_session_ids"]

            haystack_session_ids: list[str] = example["haystack_session_ids"]
            haystack_dates: list[str] = example["haystack_dates"]
            haystack_sessions: list[list] = example["haystack_sessions"]

            for i, (session_id_raw, session_date, turn_list) in enumerate(
                zip(haystack_session_ids, haystack_dates, haystack_sessions)
            ):
                if limit is not None and yielded >= limit:
                    return

                yield _make_session(
                    question_id=question_id,
                    question_type=question_type,
                    question_date=question_date,
                    answer_session_ids=answer_session_ids,
                    session_id_raw=session_id_raw,
                    session_date=session_date,
                    turn_list=turn_list,
                    session_index_in_example=i,
                    split=self._split,
                )
                yielded += 1

    def _iter_stratified(
        self,
        ds: list,
        limit: int | None,
    ) -> Iterator[DatasetSession]:
        """Yield a balanced stratified sample across ``question_type`` buckets.

        Algorithm:
          1. Enumerate all (example_idx, session_idx, ...) tuples in native order.
          2. Bucket by ``question_type``.
          3. Deterministically shuffle each bucket with ``self._sample_seed``.
          4. Compute per-bucket quotas: ``base = sample_size // num_buckets``,
             remainder allocated round-robin across sorted bucket names.
          5. Short buckets: take all available, redistribute shortfall round-robin
             to remaining buckets; log a warning.
          6. Concatenate picks across buckets, then sort by (example_idx, session_idx_in_example).
          7. Apply ``limit`` as a secondary cap.

        Args:
            ds: Parsed dataset list (one dict per example).
            limit: Secondary cap applied after stratified selection.

        Yields:
            DatasetSession with all five fields populated.
        """
        # Step 1: enumerate all session records in native order.
        # Each record is (example_idx, session_idx, question_type, example, sid_raw, date, turns)
        all_records: list[tuple] = []
        for example_idx, example in enumerate(ds):
            question_id: str = example["question_id"]
            question_type: str = example["question_type"]
            question_date: str = example["question_date"]
            answer_session_ids: list[str] = example["answer_session_ids"]

            haystack_session_ids: list[str] = example["haystack_session_ids"]
            haystack_dates: list[str] = example["haystack_dates"]
            haystack_sessions: list[list] = example["haystack_sessions"]

            for session_idx, (session_id_raw, session_date, turn_list) in enumerate(
                zip(haystack_session_ids, haystack_dates, haystack_sessions)
            ):
                all_records.append(
                    (
                        example_idx,
                        session_idx,
                        question_type,
                        question_id,
                        question_date,
                        answer_session_ids,
                        session_id_raw,
                        session_date,
                        turn_list,
                    )
                )

        # Step 2: bucket by question_type.
        buckets: dict[str, list[tuple]] = {}
        for record in all_records:
            qt = record[2]
            buckets.setdefault(qt, []).append(record)

        sorted_types = sorted(buckets.keys())
        num_buckets = len(sorted_types)

        # Step 3: deterministically shuffle each bucket.
        # Re-seed per bucket so each bucket's permutation is a pure function of
        # (seed, bucket_contents), independent of sibling-bucket lengths.  A
        # single shared Random instance would produce an equally valid but
        # different permutation — changing this pattern would shift which sessions
        # are picked for a given seed and would break existing deterministic tests.
        for qt in sorted_types:
            random.Random(self._sample_seed).shuffle(buckets[qt])

        # Step 4: compute per-bucket quotas.
        assert self._sample_size is not None  # validated at construction
        base_quota, remainder = divmod(self._sample_size, num_buckets)
        quotas: dict[str, int] = {}
        for rank, qt in enumerate(sorted_types):
            quotas[qt] = base_quota + (1 if rank < remainder else 0)

        # Step 5: pick from each bucket; redistribute shortfalls via multi-pass
        # round-robin.  A single pass can leave residual shortfall when a
        # redistribution target bucket is also exhausted.  We repeat until
        # total_picked == sample_size (done) or no bucket has remaining headroom
        # (all buckets short; log once and break).
        picks: list[tuple] = []
        # already_taken tracks how many sessions we have consumed from each bucket
        # across all passes.  It starts at 0 and grows monotonically.
        already_taken: dict[str, int] = {qt: 0 for qt in sorted_types}
        total_picked = 0
        target = self._sample_size

        # --- First pass: honour initial quotas and record shortfalls ---
        shortfall = 0
        for qt in sorted_types:
            available = buckets[qt]
            quota = quotas[qt]
            take = min(quota, len(available))
            picks.extend(available[:take])
            already_taken[qt] = take
            total_picked += take
            if take < quota:
                short = quota - take
                logger.warning("Bucket %r short %d item(s), redistributing", qt, short)
                shortfall += short

        # --- Subsequent passes: redistribute shortfall round-robin ---
        while total_picked < target and shortfall > 0:
            # Identify buckets that still have headroom (not fully consumed).
            eligible = [qt for qt in sorted_types if already_taken[qt] < len(buckets[qt])]
            if not eligible:
                # Every bucket is exhausted; emit warning and stop.
                logger.warning(
                    "All buckets short; only %d of %d sessions available",
                    total_picked,
                    target,
                )
                break

            # Distribute remaining shortfall round-robin across eligible buckets.
            extra_base, extra_rem = divmod(shortfall, len(eligible))
            new_shortfall = 0
            for rank, qt in enumerate(eligible):
                extra = extra_base + (1 if rank < extra_rem else 0)
                if extra == 0:
                    continue
                available = buckets[qt]
                start = already_taken[qt]
                additional = available[start : start + extra]
                picks.extend(additional)
                already_taken[qt] += len(additional)
                total_picked += len(additional)
                residual = extra - len(additional)
                if residual > 0:
                    new_shortfall += residual
            shortfall = new_shortfall

        # Step 6: sort by (example_idx, session_idx_in_example).
        picks.sort(key=lambda r: (r[0], r[1]))

        # Step 7: yield, applying secondary limit cap.
        yielded = 0
        for record in picks:
            if limit is not None and yielded >= limit:
                return
            (
                _example_idx,
                session_idx,
                question_type,
                question_id,
                question_date,
                answer_session_ids,
                session_id_raw,
                session_date,
                turn_list,
            ) = record
            yield _make_session(
                question_id=question_id,
                question_type=question_type,
                question_date=question_date,
                answer_session_ids=answer_session_ids,
                session_id_raw=session_id_raw,
                session_date=session_date,
                turn_list=turn_list,
                session_index_in_example=session_idx,
                split=self._split,
            )
            yielded += 1


def _make_session(
    *,
    question_id: str,
    question_type: str,
    question_date: str,
    answer_session_ids: list[str],
    session_id_raw: str,
    session_date: str,
    turn_list: list[dict],
    session_index_in_example: int,
    split: str,
) -> DatasetSession:
    """Build a DatasetSession from the raw per-session fields.

    This helper centralises the metadata construction and session_id /
    speaker_id formatting that was previously inline in ``iter_sessions``.

    Args:
        question_id: Top-level ``question_id`` from the dataset example.
        question_type: Top-level ``question_type`` from the dataset example.
        question_date: Top-level ``question_date`` from the dataset example.
        answer_session_ids: List of haystack session IDs that contain the answer.
        session_id_raw: Raw session ID from the ``haystack_session_ids`` array.
        session_date: Date string from the ``haystack_dates`` array.
        turn_list: List of turn dicts from the ``haystack_sessions`` array.
        session_index_in_example: Zero-based index of this session within its
            parent example's haystack (used for ``session_index_in_example``
            metadata and sort-key stability in stratified mode).
        split: The HF split name (e.g. ``"longmemeval_oracle"``).

    Returns:
        Fully-populated DatasetSession.
    """
    transcript = _build_transcript(turn_list)
    session_id = f"longmemeval:{question_id}:{session_id_raw}"
    speaker_id = f"longmemeval:{question_id}"

    metadata: dict = {
        "dataset": "longmemeval",
        "split": split,
        "question_id": question_id,
        "question_type": question_type,
        "haystack_session_id": session_id_raw,
        "session_index_in_example": session_index_in_example,
        "question_date": question_date,
        "haystack_date": session_date,
        "has_answer_turn_count": sum(1 for t in turn_list if t.get("has_answer", False)),
        "is_answer_session": session_id_raw in answer_session_ids,
    }

    return DatasetSession(
        session_id=session_id,
        transcript=transcript,
        speaker_id=speaker_id,
        speaker_name="User",
        metadata=metadata,
    )


def _build_transcript(turn_list: list[dict]) -> str:
    """Render a list of LongMemEval turns into a User/Assistant transcript.

    Each turn has ``role`` (``"user"`` or ``"assistant"``) and ``content``.
    The ``has_answer`` flag is conversation metadata and is not emitted.

    Args:
        turn_list: List of turn dicts from the haystack_sessions field.

    Returns:
        Newline-separated multi-turn string (``User: ...\\nAssistant: ...``).
    """
    lines = []
    for turn in turn_list:
        role: str = turn["role"]
        content: str = turn["content"]
        prefix = role.capitalize()  # "user" -> "User", "assistant" -> "Assistant"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)
