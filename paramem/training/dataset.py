"""Dataset utilities for personal memory training."""

from paramem.graph.prompts import _load_prompt_section
from paramem.utils.tokens import RenderedPrompt

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


def build_inference_prompts(questions: list[str], tokenizer) -> list[RenderedPrompt]:
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
        Rendered :class:`~paramem.utils.tokens.RenderedPrompt` strings, same
        order and length as *questions*.
    """
    from paramem.models.loader import render_chat_prompt

    system_prompt = trained_recall_system_prompt()
    return [
        render_chat_prompt(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            tokenizer,
            add_generation_prompt=True,
        )
        for question in questions
    ]


def _tokenize_with_prompt_masking(messages: list[dict], tokenizer, max_length: int) -> dict:
    """Tokenize a chat message list with prompt token masking.

    Returns ``{"input_ids", "attention_mask", "labels"}`` with prompt tokens
    masked to ``-100`` in labels so the model learns to predict only the
    assistant turn.

    Used by the entry-format training path
    (:func:`paramem.memory.entry.format_entry_training`).

    Args:
        messages: List of chat message dicts (role/content pairs), with the
            last message being the assistant response. UN-adapted — this
            function renders through :func:`~paramem.models.loader.render_chat_prompt`,
            which applies :func:`~paramem.models.loader.adapt_messages`
            internally; callers must not adapt a second time.
        tokenizer: HuggingFace tokenizer supporting ``apply_chat_template``.
        max_length: Maximum token length; truncation applied to both full and
            prompt encodings.

    Returns:
        Dict with ``input_ids``, ``attention_mask``, and ``labels`` tensors
        (prompt tokens masked to ``-100``).

    Raises:
        ValueError: if truncation at *max_length* cuts into the assistant
            turn (``prompt_length >= len(input_ids)``) — every label would
            be masked, so the example would carry zero training signal.
    """
    from paramem.models.loader import render_chat_prompt
    from paramem.utils.tokens import encode_rendered

    full_text = render_chat_prompt(messages, tokenizer, add_generation_prompt=False)
    prompt_text = render_chat_prompt(messages[:-1], tokenizer, add_generation_prompt=True)

    full_enc = encode_rendered(
        tokenizer, full_text, truncation=True, max_length=max_length, return_tensors="pt"
    )
    prompt_enc = encode_rendered(
        tokenizer, prompt_text, truncation=True, max_length=max_length, return_tensors="pt"
    )

    input_ids = full_enc["input_ids"].squeeze()
    attention_mask = full_enc["attention_mask"].squeeze()
    prompt_length = prompt_enc["input_ids"].shape[1]

    if prompt_length >= len(input_ids):
        raise ValueError(
            f"max_length={max_length} truncates the prompt itself "
            f"(prompt_length={prompt_length}, encoded_length={len(input_ids)}); "
            "every label would be masked, leaving zero training signal. "
            "Raise training_max_seq_length."
        )

    labels = input_ids.clone()
    labels[:prompt_length] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
