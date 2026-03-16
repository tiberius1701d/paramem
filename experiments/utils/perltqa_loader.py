"""PerLTQA dataset loader for ParaMem evaluation.

Loads character profiles, dialogues, and QA pairs from the PerLTQA dataset
(https://github.com/Elvin-Yiming-Du/PerLTQA) for use in ParaMem experiments.

Expected local layout (clone to data/external/PerLTQA/):
  data/external/PerLTQA/Dataset/en_v2/
    ├── perltqa_en_v2.json    # Ground-truth QA evaluation pairs (32 characters)
    └── perltmem_en_v2.json   # Character profiles + dialogues (141 characters)

Two main use cases:
  1. load_character_dialogues() — raw dialogue transcripts for pipeline input
     (graph extraction → QA generation → indexed key training)
  2. load_character_eval_qa() — ground-truth QA pairs for evaluation

Falls back to synthetic data if PerLTQA is not available locally.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
PERLTQA_DIR = PROJECT_ROOT / "data" / "external" / "PerLTQA" / "Dataset" / "en_v2"
PERLTQA_QA_PATH = PERLTQA_DIR / "perltqa_en_v2.json"
PERLTQA_MEM_PATH = PERLTQA_DIR / "perltmem_en_v2.json"
FALLBACK_PATH = PROJECT_ROOT / "data" / "synthetic" / "personal_facts.json"


def is_available() -> bool:
    """Check if PerLTQA dataset is available locally."""
    return PERLTQA_MEM_PATH.exists()


def _load_mem_data() -> dict:
    """Load the full memory data (profiles + dialogues) for all 141 characters."""
    with open(PERLTQA_MEM_PATH) as f:
        return json.load(f)


def _load_qa_data() -> list[dict]:
    """Load ground-truth QA evaluation data (32 characters)."""
    if not PERLTQA_QA_PATH.exists():
        return []
    with open(PERLTQA_QA_PATH) as f:
        return json.load(f)


def list_characters() -> dict[str, dict]:
    """List available characters with dialogue and QA counts.

    Returns dict mapping character_name -> {dialogues, events, eval_qa}.
    """
    mem_data = _load_mem_data()
    qa_data = _load_qa_data()

    # Build eval QA counts (handle nested structure)
    eval_counts: dict[str, int] = {}
    for entry in qa_data:
        for name, info in entry.items():
            total = 0
            for cat, items in info.items():
                for item in items:
                    if "Question" in item:
                        total += 1
                    else:
                        for sub_list in item.values():
                            if isinstance(sub_list, list):
                                total += len(sub_list)
            eval_counts[name] = total

    stats = {}
    for name, info in mem_data.items():
        stats[name] = {
            "dialogues": len(info.get("dialogues", {})),
            "events": len(info.get("events", {})),
            "eval_qa": eval_counts.get(name, 0),
        }

    return dict(sorted(stats.items(), key=lambda x: -x[1]["dialogues"]))


def load_character_dialogues(
    character_name: str,
    max_sessions: int | None = None,
) -> list[dict]:
    """Load dialogues for a character as session transcripts.

    Each dialogue becomes a session dict with:
      - session_id: unique identifier
      - transcript: multi-turn conversation text (User/Assistant format)
      - timestamp: first timestamp from the dialogue

    These transcripts are meant to be fed directly into extract_graph().

    Args:
        character_name: Character name in the dataset (e.g. "Liang Xin").
        max_sessions: Maximum number of sessions to return. None = all.

    Returns:
        List of session dicts sorted by timestamp.
    """
    mem_data = _load_mem_data()
    if character_name not in mem_data:
        logger.warning("Character '%s' not found in PerLTQA", character_name)
        return []

    char = mem_data[character_name]
    dialogues = char.get("dialogues", {})

    sessions = []
    for dlg_id, dlg in dialogues.items():
        contents = dlg.get("contents", {})
        if not contents:
            continue

        # Build transcript from all timestamped turns
        transcript_lines = []
        first_timestamp = None
        for timestamp in sorted(contents.keys()):
            if first_timestamp is None:
                first_timestamp = timestamp
            for line in contents[timestamp]:
                # Lines are "Name: message" — convert to User/Assistant
                # The protagonist's lines become User, others become Assistant
                if line.startswith(f"{character_name}:"):
                    msg = line[len(character_name) + 1:].strip()
                    transcript_lines.append(f"User: {msg}")
                else:
                    # Other character's line — treat as context
                    transcript_lines.append(f"Assistant: {line}")

        if not transcript_lines:
            continue

        sessions.append({
            "session_id": f"perltqa_{character_name}_{dlg_id}",
            "transcript": "\n".join(transcript_lines),
            "timestamp": first_timestamp or "",
        })

    sessions.sort(key=lambda s: s["timestamp"])

    if max_sessions is not None:
        sessions = sessions[:max_sessions]

    logger.info(
        "Loaded %d dialogue sessions for '%s'",
        len(sessions),
        character_name,
    )
    return sessions


def load_character_eval_qa(
    character_name: str,
    max_pairs: int = 200,
) -> list[dict]:
    """Load ground-truth QA pairs for a character from the evaluation set.

    These are human-authored QA pairs across profile, social, events, and
    dialogue categories. Use for evaluation, NOT for training input — the
    pipeline generates its own training QA from dialogues via graph extraction.

    Args:
        character_name: Character name in the dataset.
        max_pairs: Maximum number of QA pairs to return.

    Returns:
        List of {"question": str, "answer": str, "category": str} dicts.
    """
    qa_data = _load_qa_data()

    for entry in qa_data:
        if character_name not in entry:
            continue

        info = entry[character_name]
        qa_pairs = []
        for category, items in info.items():
            for item in items:
                # Profile items are flat dicts with Question/Answer.
                # Other categories are nested: {id_key: [QA_list]}.
                if "Question" in item:
                    qa_items = [item]
                else:
                    qa_items = []
                    for sub_list in item.values():
                        if isinstance(sub_list, list):
                            qa_items.extend(sub_list)

                for qa_item in qa_items:
                    question = qa_item.get("Question", "").strip()
                    answer = qa_item.get("Answer", "").strip()
                    if not question or not answer:
                        continue
                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "category": category,
                    })
                    if len(qa_pairs) >= max_pairs:
                        return qa_pairs

        return qa_pairs

    logger.warning("Character '%s' not found in PerLTQA eval set", character_name)
    return []


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
) -> tuple[list[dict], str]:
    """Load QA pairs, preferring PerLTQA eval set with synthetic fallback.

    This is the legacy interface for backwards compatibility with tests that
    pass pre-made QA pairs directly to training. For the full pipeline
    (dialogue → graph extraction → QA gen), use load_character_dialogues().

    Returns:
        (qa_pairs, source_label) where source_label is "perltqa:<name>"
        or "synthetic:personal_facts".
    """
    if is_available() and character_id is not None:
        pairs = load_character_eval_qa(character_id, max_pairs)
        if pairs:
            # Strip category field for backward compat
            clean = [{"question": p["question"], "answer": p["answer"]} for p in pairs]
            logger.info(
                "Loaded %d eval QA pairs from PerLTQA character '%s'",
                len(clean),
                character_id,
            )
            return clean, f"perltqa:{character_id}"

    logger.info("Falling back to synthetic personal_facts.json")
    pairs = load_fallback_qa(max_pairs)
    return pairs, "synthetic:personal_facts"
