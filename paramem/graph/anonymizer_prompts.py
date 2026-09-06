"""Prompt composer for local anonymization.

The ONE place that turns the sectioned ``configs/prompts/anonymization.txt``
home into the ``AnonymizerPrompts`` value the anonymizer chain's two local
model calls — SCAN and ANCHOR — consume. No model-facing prompt TEXT lives
in this module — composition logic only; every string a model sees is
loaded from the home via :func:`paramem.graph.prompts._load_prompt_sections`.

The home carries the configured categories only through the SCAN section's
own ``{keywords}`` render slot (formatted by
:func:`~paramem.cloud.anonymize_steps.scan_values` from
:func:`~paramem.config.taxonomy.prefix_descriptions`, every row whether
scrubbed or not) — this module takes no category argument and does no
category resolution.

This module cannot live in ``paramem.cloud`` (that package must not do
prompt IO or import ``paramem.graph``) and cannot live in
``paramem.graph.prompts`` (deliberately dependency-light so
``paramem.graph.merger`` can import the section loader without the
extractor's heavyweight transitive chain).

Placement note: :class:`~paramem.cloud.anonymize.AnonymizerPrompts` is
declared in ``paramem.cloud.anonymize`` and imported here rather than
redefined — this module composes the type, it does not own it. No import
cycle results: this module imports FROM ``paramem.cloud.anonymize`` (one
direction, matching the existing rule that ``paramem.graph`` depends on
``paramem.cloud``, never the reverse); ``paramem.cloud.anonymize`` does not
import this module or anything else under ``paramem.graph``.
"""

from __future__ import annotations

from pathlib import Path

from paramem.cloud.anonymize import AnonymizerPrompts
from paramem.graph.prompts import _load_prompt_sections

_ANONYMIZATION_PROMPT_FILE = "anonymization.txt"


def load_anonymizer_prompts(*, prompts_dir: str | Path | None = None) -> AnonymizerPrompts:
    """Compose the prompt sections one ``anonymize()`` run needs.

    Loads all four sections of the sectioned ``anonymization.txt`` home
    (:func:`~paramem.graph.prompts._load_prompt_sections`) into an
    :class:`AnonymizerPrompts`. The SCAN section carries a ``{keywords}``
    render slot, formatted per call by
    :func:`~paramem.cloud.anonymize_steps.scan_values` — this composer
    itself does no per-call formatting, no ``scrub`` or resolved-category
    argument.

    Args:
        prompts_dir: Optional operator ``paths.prompts`` override, threaded
            unchanged to the section loader. ``None`` (default) resolves
            only the shipped ``configs/prompts/`` copy.

    Returns:
        A fully composed :class:`AnonymizerPrompts`.

    Raises:
        FileNotFoundError: When ``anonymization.txt`` is absent from every
            searched directory.
        KeyError: When a required section (``SCAN-SYSTEM``, ``SCAN``,
            ``ANCHOR-SYSTEM``, ``ANCHOR``) is absent from the home —
            normally caught earlier at boot by
            :func:`paramem.graph.prompts.ensure_prompt_assets`.
    """
    resolved_dir = Path(prompts_dir) if prompts_dir is not None else None
    sections = _load_prompt_sections(_ANONYMIZATION_PROMPT_FILE, prompts_dir=resolved_dir)

    return AnonymizerPrompts(
        scan_system=sections["SCAN-SYSTEM"],
        scan=sections["SCAN"],
        anchor_system=sections["ANCHOR-SYSTEM"],
        anchor=sections["ANCHOR"],
    )
