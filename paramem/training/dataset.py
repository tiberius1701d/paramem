"""Dataset utilities for personal memory training."""

from paramem.graph.prompts import _load_prompt_section

_TRAINED_RECALL_FILE = "trained_recall.txt"


def trained_recall_system_prompt() -> str:
    """Load the trained-recall system prompt (``trained_recall.txt`` § SYSTEM).

    This is one half of the weight-coupled training/probe interface — every
    adapter in production was trained with this exact text as the system
    message.  Loaded at call time (never cached), so it participates in
    :func:`~paramem.graph.prompts.prompt_overrides` like every other prompt.

    Returns:
        The system prompt string, verbatim.
    """
    return _load_prompt_section(_TRAINED_RECALL_FILE, "SYSTEM")


def trained_recall_template() -> str:
    """Load the trained-recall user template (``trained_recall.txt`` § RECALL).

    Returns the raw ``{key}``-bearing template — callers format it with the
    specific key being recalled.  The other half of the weight-coupled
    training/probe interface; see :func:`trained_recall_system_prompt`.

    Returns:
        The template string with one ``{key}`` slot, e.g.
        ``"Recall the fact stored under key '{key}'."``.
    """
    return _load_prompt_section(_TRAINED_RECALL_FILE, "RECALL")


def build_inference_prompts(questions: list[str], tokenizer) -> list[str]:
    """Render inference prompts for N questions, sharing one system-prompt load.

    No fact context is provided — the model must recall from its adapted
    weights.  The trained-recall system prompt is loaded exactly once
    regardless of ``len(questions)``, so a caller building N prompts (batched
    probing, batched recall evaluation) does not re-read the prompt file —
    and does not emit N ``record_prompt`` provenance entries — for what is
    a single, shared load.  This is the ONE message-construction site in the
    package; every caller, single-question or batched, renders through here.

    Args:
        questions: User-turn question strings, one per prompt to render.
        tokenizer: HuggingFace tokenizer supporting ``apply_chat_template``.

    Returns:
        Rendered prompt strings, same order and length as *questions*.
    """
    from paramem.models.loader import adapt_messages

    system_prompt = trained_recall_system_prompt()
    return [
        tokenizer.apply_chat_template(
            adapt_messages(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                tokenizer,
            ),
            tokenize=False,
            add_generation_prompt=True,
        )
        for question in questions
    ]


def _tokenize_with_prompt_masking(messages: list[dict], tokenizer, max_length: int) -> dict:
    """Tokenize a chat message list with prompt token masking.

    Returns ``{"input_ids", "attention_mask", "labels"}`` with prompt tokens
    masked to ``-100`` in labels so the model learns to predict only the
    assistant turn.

    Used by both the entry-format training path
    (:func:`paramem.memory.entry.format_entry_training`) and the legacy
    QA-format archive (:func:`archive.legacy_qa.format_indexed_training`).

    Args:
        messages: List of chat message dicts (role/content pairs), with the
            last message being the assistant response.
        tokenizer: HuggingFace tokenizer supporting ``apply_chat_template``.
        max_length: Maximum token length; truncation applied to both full and
            prompt encodings.

    Returns:
        Dict with ``input_ids``, ``attention_mask``, and ``labels`` tensors
        (prompt tokens masked to ``-100``).
    """
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )

    full_enc = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors="pt")
    prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_length, return_tensors="pt")

    input_ids = full_enc["input_ids"].squeeze()
    attention_mask = full_enc["attention_mask"].squeeze()
    prompt_length = prompt_enc["input_ids"].shape[1]

    labels = input_ids.clone()
    labels[:prompt_length] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
