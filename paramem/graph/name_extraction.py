"""LLM-based speaker name extraction for enrollment.

Graph-layer module: no imports from paramem.server.  Only uses
paramem.evaluation.recall, paramem.graph.prompts, and stdlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def extract_name_via_llm(
    turns: list[dict],
    model: Any,
    tokenizer: Any,
    *,
    prompts_dir: str | Path | None = None,
    prompt_filename: str = "name_extraction.txt",
    system_filename: str = "name_extraction_system.txt",
    user_turns_only: bool = True,
    params: dict | None = None,
) -> tuple[str | None, str]:
    """Extract a speaker's self-introduced name from conversation turns.

    Loads the system and user prompts from external files via
    :func:`paramem.graph.prompts._load_prompt` with ``required=True``.
    A missing prompt file raises :exc:`FileNotFoundError` immediately —
    no inline fallback, no silent empty-prompt production path.

    Args:
        turns: List of turn dicts, each with ``"role"`` and ``"text"`` keys.
        model: The loaded HuggingFace model.
        tokenizer: The loaded HuggingFace tokenizer.
        prompts_dir: Optional override for the prompt directory.  Defaults to
            ``configs/prompts/`` (``_load_prompt``'s default).  Calibration
            callers point this at a scratch directory so prompt edits are
            picked up without a server restart (read-fresh semantics).
        prompt_filename: User-turn prompt file name.  Defaults to
            ``"name_extraction.txt"``.  Calibration passes the operator's
            override here so provenance and execution are always in sync.
        system_filename: System prompt file name.  Defaults to
            ``"name_extraction_system.txt"``.  Same override semantics as
            *prompt_filename*.
        user_turns_only: When ``True`` (default), only ``role == "user"``
            turns are included in the transcript passed to the model.  This
            prevents assistant salutations (e.g. "Good evening, user") from
            being mis-classified as name introductions.
        params: Optional inference overrides (``temperature``, ``seed``,
            ``max_tokens``).  Unset fields fall back to production defaults
            (``temperature=0.0``, ``max_new_tokens=64``).

    Returns:
        A ``(name, raw_output)`` tuple.  ``name`` is the extracted name
        string (1–3 words, ≤30 chars), or ``None`` when no valid
        self-introduction was found (includes the ``NONE`` sentinel and
        all post-filter rejections).  ``raw_output`` is the verbatim model
        string before any post-filter is applied — always populated, even
        when ``name`` is ``None``.

    Raises:
        FileNotFoundError: When the system or user prompt file is absent from
            all searched directories.
    """
    from paramem.evaluation.recall import generate_answer
    from paramem.graph.prompts import _load_prompt

    # Build transcript — filter to user turns only when requested.
    lines = []
    for turn in turns:
        role = turn.get("role", "unknown")
        if user_turns_only and role != "user":
            continue
        text = turn.get("text", "")
        lines.append(f"{role}: {text}")
    transcript_text = "\n".join(lines)

    prompts_dir_path: Path | None = Path(prompts_dir) if prompts_dir is not None else None

    # required=True: a missing prompt file surfaces immediately rather than
    # silently yielding an empty prompt in production.
    system_msg = _load_prompt(
        system_filename,
        prompts_dir=prompts_dir_path,
        required=True,
    )
    user_template = _load_prompt(
        prompt_filename,
        prompts_dir=prompts_dir_path,
        required=True,
    )

    user_msg = user_template.format(transcript=transcript_text)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Resolve inference params.
    p = params or {}
    temperature = p.get("temperature", 0.0)
    if temperature is None:
        temperature = 0.0
    max_new_tokens = p.get("max_tokens", 64)
    if max_new_tokens is None:
        max_new_tokens = 64
    seed = p.get("seed", None)

    result = generate_answer(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=float(temperature),
        seed=seed,
    )
    raw_output = result.strip().strip('"').strip("'").strip(".")

    if not raw_output or raw_output.upper() == "NONE" or len(raw_output) > 30:
        return None, raw_output

    words = raw_output.split()
    if len(words) > 3 or len(words) == 0:
        return None, raw_output

    return raw_output, raw_output
