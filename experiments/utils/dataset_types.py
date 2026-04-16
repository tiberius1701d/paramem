"""Common loader interface types for dataset-agnostic probing.

Defines the DatasetSession dataclass and the DatasetLoader Protocol
shared by all dataset adapters (PerLTQA, LongMemEval, and future loaders).

Every loader must yield DatasetSession objects with all five fields
populated; loaders that lack a natural value synthesize a deterministic
placeholder (e.g. speaker_id = f"{dataset_name}:{user_id}").
"""

from dataclasses import dataclass
from typing import Any, Iterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class DatasetSession:
    """A single normalized conversation session from any dataset.

    Attributes:
        session_id: Globally unique identifier within the dataset.
            Synthesized by the loader (e.g. "perltqa_Alex_dlg001" or
            "longmemeval:gpt4_abc123:answer_session_1").
        transcript: Multi-turn conversation in "User: ...\nAssistant: ..."
            format, ready for extraction pipeline ingestion.
        speaker_id: Stable user identifier for this session (not a display
            name). Used for speaker tagging in QA pairs downstream.
        speaker_name: Display name; may equal speaker_id when anonymous.
            Passed to extract_session() as speaker_name.
        metadata: Dataset-specific extras (timestamp, category, QA IDs, etc.).
            Schema varies per loader; always a plain dict.
    """

    session_id: str
    transcript: str
    speaker_id: str
    speaker_name: str
    metadata: dict[str, Any]


@runtime_checkable
class DatasetLoader(Protocol):
    """Protocol for all dataset adapters.

    Implementations must expose a ``name`` attribute (e.g. "perltqa") and
    an ``iter_sessions`` generator. Any new dataset loader slots in by
    implementing this protocol; no changes to dataset_probe.py beyond
    adding one registry entry and one CLI choice.
    """

    name: str

    def iter_sessions(
        self,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Iterator[DatasetSession]:
        """Yield normalized DatasetSession objects.

        Args:
            limit: Maximum number of sessions to yield across all examples.
                None yields all available sessions. Deterministic ordering
                is required (dataset-native order).
            **kwargs: Dataset-specific selectors (e.g. ``character`` for
                PerLTQA, ``split`` already bound at construction time for
                LongMemEval).

        Yields:
            DatasetSession with all five fields populated.
        """
        ...
