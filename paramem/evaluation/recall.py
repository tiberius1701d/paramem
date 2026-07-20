"""Evaluation harness for personal memory recall."""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

_DEFAULT_REPETITION_PENALTY = 1.1


def generate_answer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    repetition_penalty: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
) -> str:
    """Generate an answer from the model given a prompt.

    When repetition_penalty is None, uses the module-level default (1.1).
    Call sites can override per-objective when needed.

    ``top_p`` / ``top_k`` / ``seed`` are optional sampling overrides used by
    the calibration tool to probe LLM-compliance variance.  ``seed`` is
    applied via ``torch.manual_seed`` immediately before generation (global
    torch RNG), which makes sampling reproducible at temperature > 0 and is
    a no-op at temperature 0 (greedy).  This affects the global RNG, which
    is acceptable for the serialized calibration use case.  All three default
    to ``None`` — production paths preserve the current temperature-driven
    sampling behaviour.
    """
    if repetition_penalty is None:
        repetition_penalty = _DEFAULT_REPETITION_PENALTY
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Build stop token list: eos_token + chat template end tokens (e.g. <|im_end|>)
    stop_ids = [tokenizer.eos_token_id]
    for token_name in ["<|im_end|>", "<|eot_id|>"]:
        encoded = tokenizer.encode(token_name, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] not in stop_ids:
            stop_ids.append(encoded[0])

    extra_kwargs: dict = {}
    if top_p is not None:
        extra_kwargs["top_p"] = top_p
    if top_k is not None:
        extra_kwargs["top_k"] = top_k

    if seed is not None:
        torch.manual_seed(int(seed))
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_ids,
            repetition_penalty=repetition_penalty,
            **extra_kwargs,
        )

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
