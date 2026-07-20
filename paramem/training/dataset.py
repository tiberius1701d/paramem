"""Dataset utilities for personal memory training."""

SYSTEM_PROMPT = (
    "You are a personal assistant with memory of your user's life. "
    "Answer questions about the user based on what you know about them."
)


def format_inference_prompt(question: str, tokenizer) -> str:
    """Format a question for inference using the model's native chat template.

    No fact context is provided — the model must recall from its adapted weights.
    """
    from paramem.models.loader import adapt_messages

    messages = adapt_messages(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenizer,
    )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
