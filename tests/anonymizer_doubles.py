"""Shared test doubles for the anonymize chain: a ready-made
``AnonymizerPrompts`` construction and a bare ``ScrubCategory``.

Not a test module itself — imported by the anonymizer test files so none
of them re-implements the same constructions.

Never imported by production code under ``paramem/``.
"""

from __future__ import annotations

from paramem.cloud.anonymize import AnonymizerPrompts
from paramem.config.taxonomy import ScrubCategory
from paramem.graph.anonymizer_prompts import load_anonymizer_prompts


def basic_prompts(
    *,
    anchor_system: str | None = None,
    anchor: str | None = None,
) -> AnonymizerPrompts:
    """A well-shaped, full four-field :class:`AnonymizerPrompts`.

    The ``SCAN-SYSTEM``/``SCAN`` sections are always the shipped
    ``configs/prompts/anonymization.txt`` sections, loaded through the one
    production composer (:func:`~paramem.graph.anonymizer_prompts.
    load_anonymizer_prompts`) — the same sections
    :func:`~paramem.cloud.anonymize_steps.render_scan_section` formats at
    runtime, once per scanned payload slice, so a caller exercising the
    full ``anonymize()`` chain gets a real,
    well-shaped ``{keywords}``/``{text}`` SCAN template without hand-typing
    one. ``anchor_system``/``anchor`` default to the shipped ANCHOR
    sections the same way, and may be overridden with a minimal custom
    template when a test needs direct control over
    :func:`~paramem.cloud.anonymize_steps.ask_speaker_anchor`'s own
    ``{speaker_id}``/``{values}``/``{text}`` rendering.
    """
    loaded = load_anonymizer_prompts()
    return AnonymizerPrompts(
        scan_system=loaded.scan_system,
        scan=loaded.scan,
        anchor_system=anchor_system if anchor_system is not None else loaded.anchor_system,
        anchor=anchor if anchor is not None else loaded.anchor,
    )


def basic_category(prefix: str) -> ScrubCategory:
    """A bare active :class:`~paramem.config.taxonomy.ScrubCategory` for
    *prefix* — the one field the current design carries."""
    return ScrubCategory(prefix=prefix)


class ScriptedTokenizer:
    """A deterministic, model-free tokenizer double for driving the full
    ``anonymize()`` chain on CPU.

    ``apply_chat_template`` joins ``role:content`` per message (a real
    enough render that ``supports_system_role``'s own marker probe reports
    the system role as supported, so ``adapt_messages`` leaves a
    system+user pair unfolded — the same shape
    ``tests/test_anonymize_steps.py``'s ``_StubChatTemplateTokenizer``
    relies on). ``__call__`` tokenizes by whitespace split, so
    :func:`~paramem.utils.tokens.estimate_tokens` is EXACT (word count, not
    the words-per-token fallback estimate) — a test that needs a precise
    token budget (to force :class:`~paramem.cloud.anonymize_steps.
    AnonymizeBudgetRefused` on one slice but not another) can compute the
    exact number up front rather than guess at the fallback ratio.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = "\n".join(f"{m['role']}:{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "\nassistant:"
        return rendered

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class ScriptedGenerate:
    """A callable double for
    :func:`paramem.cloud.anonymize_steps.generate_answer` — returns one
    scripted raw reply per call, in the order given, and records every
    rendered prompt it was called with.

    Patch it in at the import site the production call resolves through
    (``anonymize_steps`` imports ``generate_answer`` directly, so
    ``monkeypatch.setattr(anonymize_steps_module, "generate_answer", ...)``
    is the one correct patch site — patching
    ``paramem.evaluation.recall.generate_answer`` would not be seen by
    ``anonymize_steps``'s own already-bound name).

    Raises ``AssertionError`` if a call is made after every scripted reply
    is consumed — a test driving more slices than it scripted replies for
    is a test bug, not a double that should silently keep answering.
    """

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[str] = []

    def __call__(
        self, model, tokenizer, prompt, *, max_new_tokens=None, temperature=None, seed=None
    ):
        self.calls.append(prompt)
        if not self._replies:
            raise AssertionError("ScriptedGenerate: no more scripted replies")
        return self._replies.pop(0)
