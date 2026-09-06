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
    :func:`~paramem.cloud.anonymize_steps.scan_values` formats at runtime,
    so a caller exercising the full ``anonymize()`` chain gets a real,
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
