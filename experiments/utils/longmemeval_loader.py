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

    Attributes:
        name: Loader identifier for the dataset registry ("longmemeval").
    """

    name: str = "longmemeval"

    def __init__(
        self,
        split: str = "longmemeval_oracle",
        cache_dir: Path | None = None,
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
        self._split = split
        self._cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR

    def iter_sessions(
        self,
        limit: int | None = None,
        **kwargs,  # absorb dataset_probe kwargs (e.g. character — not used here)
    ) -> Iterator[DatasetSession]:
        """Yield DatasetSession objects from the LongMemEval dataset.

        Session ordering: dataset-native example order × parallel-array index
        within each example's haystack. ``limit`` caps total yielded sessions.

        The rendered transcript includes all turns (including has_answer=True
        turns, which are conversation content, not leakage). The top-level
        ``question`` and ``answer`` fields are never concatenated into the
        transcript.

        Args:
            limit: Maximum number of sessions to yield. None = all.
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

        yielded = 0
        for example in ds:
            question_id: str = example["question_id"]
            question_type: str = example["question_type"]
            question_date: str = example["question_date"]
            # Defensive cast: answer field is heterogeneous (str | int in oracle).
            # Never concatenated into transcript — only used for the leakage guard.
            answer_session_ids: list[str] = example["answer_session_ids"]

            haystack_session_ids: list[str] = example["haystack_session_ids"]
            haystack_dates: list[str] = example["haystack_dates"]
            haystack_sessions: list[list] = example["haystack_sessions"]

            for i, (session_id_raw, session_date, turn_list) in enumerate(
                zip(haystack_session_ids, haystack_dates, haystack_sessions)
            ):
                if limit is not None and yielded >= limit:
                    return

                transcript = _build_transcript(turn_list)
                session_id = f"longmemeval:{question_id}:{session_id_raw}"
                speaker_id = f"longmemeval:{question_id}"

                metadata: dict = {
                    "dataset": "longmemeval",
                    "split": self._split,
                    "question_id": question_id,
                    "question_type": question_type,
                    "haystack_session_id": session_id_raw,
                    "session_index_in_example": i,
                    "question_date": question_date,
                    "haystack_date": session_date,
                    "has_answer_turn_count": sum(
                        1 for t in turn_list if t.get("has_answer", False)
                    ),
                    "is_answer_session": session_id_raw in answer_session_ids,
                }

                yield DatasetSession(
                    session_id=session_id,
                    transcript=transcript,
                    speaker_id=speaker_id,
                    speaker_name="User",
                    metadata=metadata,
                )
                yielded += 1


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
