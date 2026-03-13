"""PerLTQA dataset loader for ParaMem evaluation.

Loads character profiles and QA pairs from the PerLTQA dataset
(https://github.com/Elvin-Yiming-Du/PerLTQA) and converts them
to ParaMem's indexed key format.

Expected local layout (clone to data/external/PerLTQA/):
  data/external/PerLTQA/dataset/
    ├── train.json
    ├── valid.json
    └── test.json

Each entry has: q (question), a (answer), r (reference memory), m (anchors).

Falls back to synthetic data if PerLTQA is not available locally.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
PERLTQA_DIR = PROJECT_ROOT / "data" / "external" / "PerLTQA" / "dataset"
FALLBACK_PATH = PROJECT_ROOT / "data" / "synthetic" / "personal_facts.json"


def is_available() -> bool:
    """Check if PerLTQA dataset is available locally."""
    return any(PERLTQA_DIR.glob("*.json"))


def _load_raw(splits: list[str] | None = None) -> list[dict]:
    """Load raw PerLTQA entries from all splits."""
    if splits is None:
        splits = ["train", "valid", "test"]

    entries = []
    for split in splits:
        path = PERLTQA_DIR / f"{split}.json"
        if not path.exists():
            logger.warning("PerLTQA split not found: %s", path)
            continue
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            entries.extend(data)
        elif isinstance(data, dict):
            for character_entries in data.values():
                if isinstance(character_entries, list):
                    entries.extend(character_entries)
    return entries


def list_characters() -> dict[str, int]:
    """List available characters with their QA pair counts.

    Returns dict mapping character_id -> count.
    """
    entries = _load_raw()
    counts: dict[str, int] = {}
    for entry in entries:
        char_id = entry.get("character_id", entry.get("character", "unknown"))
        counts[char_id] = counts.get(char_id, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def load_character_qa(
    character_id: str,
    max_pairs: int = 100,
    max_answer_words: int = 30,
) -> list[dict]:
    """Load QA pairs for a specific character.

    Args:
        character_id: Character identifier in the dataset.
        max_pairs: Maximum number of QA pairs to return.
        max_answer_words: Filter out QA pairs with longer answers
            (indexed keys need concise answers).

    Returns:
        List of {"question": str, "answer": str} dicts.
    """
    entries = _load_raw()
    qa_pairs = []

    for entry in entries:
        char_id = entry.get("character_id", entry.get("character", "unknown"))
        if char_id != character_id:
            continue

        question = entry.get("q", "").strip()
        answer = entry.get("a", "").strip()

        if not question or not answer:
            continue

        if len(answer.split()) > max_answer_words:
            continue

        qa_pairs.append({"question": question, "answer": answer})

        if len(qa_pairs) >= max_pairs:
            break

    return qa_pairs


def load_fallback_qa(max_pairs: int = 100) -> list[dict]:
    """Load QA pairs from synthetic personal_facts.json as fallback.

    Returns one QA pair per fact (the first pair).
    """
    if not FALLBACK_PATH.exists():
        raise FileNotFoundError(f"Fallback data not found: {FALLBACK_PATH}")

    with open(FALLBACK_PATH) as f:
        facts = json.load(f)

    qa_pairs = []
    for fact in facts:
        for qa in fact.get("qa_pairs", []):
            qa_pairs.append({"question": qa["question"], "answer": qa["answer"]})
            break  # One per fact
        if len(qa_pairs) >= max_pairs:
            break

    return qa_pairs


def load_qa(
    character_id: str | None = None,
    max_pairs: int = 100,
    max_answer_words: int = 30,
) -> tuple[list[dict], str]:
    """Load QA pairs, preferring PerLTQA with synthetic fallback.

    Returns:
        (qa_pairs, source_label) where source_label is "perltqa:<char_id>"
        or "synthetic:personal_facts".
    """
    if is_available() and character_id is not None:
        pairs = load_character_qa(character_id, max_pairs, max_answer_words)
        if pairs:
            logger.info(
                "Loaded %d QA pairs from PerLTQA character '%s'",
                len(pairs),
                character_id,
            )
            return pairs, f"perltqa:{character_id}"

    logger.info("Falling back to synthetic personal_facts.json")
    pairs = load_fallback_qa(max_pairs)
    return pairs, "synthetic:personal_facts"
