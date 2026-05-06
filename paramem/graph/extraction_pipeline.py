"""Extraction pipeline — single-topology entry to ``extract_graph`` /
``extract_procedural_graph``.

This module owns the extraction CHOKEPOINT that all orchestrators
(consolidation, calibration, ad-hoc tools, tests) go through.  It
centralises the kwarg assembly, prompt-filename resolution, adapter
guard, and gradient-checkpointing discipline that previously lived as
``ConsolidationLoop._{extraction_kwargs,run_extract_graph,
run_extract_procedural_graph}`` — methods whose names announced their
mishousing in a class whose actual purpose is training orchestration.

Architecture
------------

Two public objects::

    @dataclass
    class ExtractionConfig:
        # 15 tunables, sourced from configs/server.yaml::consolidation.extraction_*

    class ExtractionPipeline:
        def __init__(self, model, tokenizer, *, config, prompts_dir): ...
        def kwargs(self, *, source_type, **overrides) -> dict: ...
        def run(self, transcript, session_id, *, source_type, **overrides) -> SessionGraph: ...
        def run_procedural(self, transcript, session_id, *, speaker_id, ...) -> SessionGraph: ...

Consolidation and calibration *consume* this module — they don't own it.
``ConsolidationLoop`` holds an ``ExtractionPipeline`` instance.  The
calibration endpoint constructs one directly (no ConsolidationLoop
needed just to reach the extractors).

The class belongs in the graph layer (it's a pipeline over graph
extraction primitives), not the training layer.

Single-topology rule
--------------------

Direct calls to :func:`paramem.graph.extractor.extract_graph` and
:func:`paramem.graph.extractor.extract_procedural_graph` are allowed
only inside this module (and inside ``extractor.py`` itself, which
defines them).  Every orchestrator routes through
:class:`ExtractionPipeline` so the pipeline cannot diverge by accident.

Enforced by ``tests/test_extraction_pipeline_guard.py`` (AST scan) and
``tests/test_consolidation.py::TestExtractionPathParity``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from paramem.graph.extractor import (
    DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME,
    DEFAULT_SYSTEM_PROMPT_FILENAME,
    DEFAULT_USER_PROMPT_FILENAME,
    extract_graph,
    extract_procedural_graph,
)

if TYPE_CHECKING:
    from paramem.graph.schema import SessionGraph


@dataclass
class ExtractionConfig:
    """Tunables for the extraction pipeline.

    Field names drop the ``extraction_`` prefix that
    ``ConsolidationScheduleConfig`` carries — the prefix is redundant
    inside an ``ExtractionConfig`` context.  All defaults match the
    production ``configs/server.yaml`` under
    ``consolidation.extraction_*``; the server lifespan constructs an
    instance from yaml-loaded values at startup.

    The yaml schema does NOT change as part of this dataclass — yaml
    keys remain ``consolidation.extraction_*`` for operator
    compatibility.  Only the in-memory representation drops the prefix.
    """

    temperature: float = 0.0
    max_tokens: int = 8192
    plausibility_max_tokens: int = 8192
    stt_correction: bool = True
    ha_validation: bool = True
    noise_filter: str = "anthropic"
    noise_filter_model: str = "claude-sonnet-4-6"
    noise_filter_endpoint: str | None = None
    ner_check: bool = False
    ner_model: str = "en_core_web_sm"
    plausibility_judge: str = "auto"
    plausibility_stage: str = "deanon"
    verify_anonymization: bool = True
    role_aware_grounding: str = "off"
    pii_scope: set[str] | frozenset[str] | None = None


def _require_speaker_id(overrides: dict) -> str:
    """Return ``overrides["speaker_id"]``, raising on missing/empty.

    A forgotten or empty ``speaker_id`` would silently propagate as
    ``""`` into every ``Relation.speaker_id`` downstream — Pydantic
    accepts the empty string as a valid str.  This guard surfaces the
    omission as a programming error at the chokepoint instead.
    """
    sid = overrides.get("speaker_id")
    if sid is None or sid == "":
        raise ValueError(
            "speaker_id is required for extraction. Pass it explicitly to "
            "ExtractionPipeline.run(...) or .run_procedural(...). The empty "
            "string is rejected to surface a missing-arg programming error "
            "rather than silently stamp '' onto every Relation downstream."
        )
    return sid


class ExtractionPipeline:
    """Single-topology entry to factual + procedural graph extraction.

    Owns
    ----
    - ``model`` + ``tokenizer`` (set once at construction; assumed not
      to change for the lifetime of the pipeline).
    - ``config: ExtractionConfig`` (the 15 tunables).
    - ``prompts_dir`` (override for ``configs/prompts/``; ``None`` falls
      back to the project default).

    Exposes
    -------
    - :meth:`kwargs(source_type, **overrides) -> dict` — builds the
      kwarg dict for ``extract_graph``.  Every override flows through
      ``pick()`` so calibration callers can swap any field per-call.
      Source-type-derived prompt filenames are themselves overridable
      (calibration relies on this to swap a ``calib_*.txt`` variant).

    - :meth:`run(transcript, session_id, source_type, **overrides) ->
      SessionGraph` — factual extraction.  Wraps ``extract_graph`` with
      the ``disable_adapter()`` context manager when the model is a
      ``PeftModel`` (extraction must run on the base model, never the
      adapter being trained), and disables gradient checkpointing on
      the model first (HF Trainer silently disables KV cache when
      checkpointing is active; KV cache is required for
      ``model.generate()``).

    - :meth:`run_procedural(transcript, session_id, *, speaker_id, ...)
      -> SessionGraph` — procedural-extraction sibling.  Same
      adapter-guard + checkpointing discipline.  ``source_type``
      selects between dialogue and document procedural prompts.

    Not in scope
    ------------

    Debug-mode persistence of the per-session graph (``cycle_<N>/sessions/<id>/<kind>.json``)
    stays with the consolidation orchestrator that owns the cycle context.
    The pipeline returns the SessionGraph; callers persist it if they want.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        config: ExtractionConfig | None = None,
        prompts_dir: str | Path | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ExtractionConfig()
        self.prompts_dir = prompts_dir

    def kwargs(self, *, source_type: str = "transcript", **overrides) -> dict:
        """Build the kwarg dict passed into :func:`extract_graph`.

        ``source_type`` only governs runtime gate defaults; **prompt
        files are the same for every source type**.  One prompt-pair
        (``extraction.txt`` + ``extraction_system.txt``) is the single
        ground truth for extraction; document chunks land in the same
        ``{transcript}`` slot at the chat-template layer (1:1 wrap into
        the user message), and the model adapts to the surface form of
        the slot content.  This is a deliberate tradeoff against the
        former two-prompt design: that design produced silent drift on
        schema-shape rules (speaker fragmentation NEGATIVE, concept
        POSITIVE) between the transcript and document variants.

        Gate defaults still differ by ``source_type``:

        * ``"transcript"`` (default) — ``stt_correction`` and
          ``ha_validation`` default to the config values (typically
          ``True``).

        * ``"document"`` — ``stt_correction`` defaults to ``False``
          (written text has no STT artefact surface) and
          ``ha_validation`` defaults to ``False`` (Home Assistant
          entity grounding is a dialogue-context concern).  Operators
          may still override either explicitly via *overrides*.

        ``speaker_id`` is **required** in *overrides* and must be a
        non-empty string.  An absent or empty value raises
        :exc:`ValueError`.
        """
        cfg = self.config

        def pick(name: str, fallback):
            val = overrides.get(name, None)
            return fallback if val is None else val

        # Single prompt-pair regardless of source_type.  Per-source
        # extension goes via overrides (system_prompt_filename /
        # user_prompt_filename) or via prepend/append on the slot
        # content at the caller layer — never via a parallel file pair.
        system_prompt_filename = DEFAULT_SYSTEM_PROMPT_FILENAME
        user_prompt_filename = DEFAULT_USER_PROMPT_FILENAME
        if source_type == "document":
            stt_correction_default = False
            ha_validation_default = False
        else:
            stt_correction_default = cfg.stt_correction
            ha_validation_default = cfg.ha_validation

        return dict(
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            plausibility_max_tokens=cfg.plausibility_max_tokens,
            prompts_dir=self.prompts_dir,
            ha_context=overrides.get("ha_context"),
            stt_correction=pick("stt_correction", stt_correction_default),
            ha_validation=pick("ha_validation", ha_validation_default),
            noise_filter=pick("noise_filter", cfg.noise_filter),
            noise_filter_model=pick("noise_filter_model", cfg.noise_filter_model),
            noise_filter_endpoint=pick("noise_filter_endpoint", cfg.noise_filter_endpoint),
            speaker_name=overrides.get("speaker_name"),
            ner_check=pick("ner_check", cfg.ner_check),
            ner_model=pick("ner_model", cfg.ner_model),
            plausibility_judge=pick("plausibility_judge", cfg.plausibility_judge),
            plausibility_stage=pick("plausibility_stage", cfg.plausibility_stage),
            verify_anonymization=pick("verify_anonymization", cfg.verify_anonymization),
            role_aware_grounding=pick("role_aware_grounding", cfg.role_aware_grounding),
            pii_scope=pick("pii_scope", cfg.pii_scope),
            speaker_id=_require_speaker_id(overrides),
            system_prompt_filename=pick("system_prompt_filename", system_prompt_filename),
            user_prompt_filename=pick("user_prompt_filename", user_prompt_filename),
            stop_phase=overrides.get("stop_phase"),
        )

    def run(
        self,
        session_transcript: str,
        session_id: str,
        *,
        source_type: str = "transcript",
        **overrides,
    ) -> "SessionGraph":
        """Single entry point for factual extraction.

        See class docstring for invariants.  ``overrides`` accepts every
        kwarg :meth:`kwargs` returns plus the chokepoint-only fields
        ``speaker_id``, ``speaker_name``, ``ha_context``, ``stop_phase``.
        """
        from peft import PeftModel as _PeftModel

        self.model.gradient_checkpointing_disable()
        kwargs = self.kwargs(source_type=source_type, **overrides)
        if isinstance(self.model, _PeftModel):
            with self.model.disable_adapter():
                return extract_graph(
                    self.model,
                    self.tokenizer,
                    session_transcript,
                    session_id,
                    **kwargs,
                )
        return extract_graph(
            self.model,
            self.tokenizer,
            session_transcript,
            session_id,
            **kwargs,
        )

    def run_procedural(
        self,
        session_transcript: str,
        session_id: str,
        *,
        speaker_id: str,
        speaker_name: str | None = None,
        stt_correction: bool | None = None,
        source_type: str = "transcript",
    ) -> "SessionGraph":
        """Single entry point for procedural (preferences/habits) extraction.

        ``source_type`` mirrors :meth:`run` — only governs the
        ``stt_correction`` default (``False`` for documents).  The
        prompt-pair is the same regardless of source type; document
        chunks land in the ``{transcript}`` slot like transcripts do.
        Explicit *stt_correction* overrides still win.

        ``speaker_id`` is stamped onto every ``Relation`` produced by the
        procedural extractor as provenance.  Required — empty / ``None``
        raises :exc:`ValueError`.
        """
        from peft import PeftModel as _PeftModel

        if not speaker_id:
            raise ValueError(
                "speaker_id is required for procedural extraction (no empty-string default)."
            )

        self.model.gradient_checkpointing_disable()
        cfg = self.config
        stt_default = False if source_type == "document" else cfg.stt_correction
        stt = stt_default if stt_correction is None else stt_correction
        # Single prompt-pair for both source types — see :meth:`kwargs`.
        system_prompt_filename = DEFAULT_SYSTEM_PROMPT_FILENAME
        user_prompt_filename = DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME
        call_kwargs = dict(
            max_tokens=cfg.max_tokens,
            prompts_dir=self.prompts_dir,
            stt_correction=stt,
            speaker_name=speaker_name,
            speaker_id=speaker_id,
            system_prompt_filename=system_prompt_filename,
            user_prompt_filename=user_prompt_filename,
        )
        if isinstance(self.model, _PeftModel):
            with self.model.disable_adapter():
                return extract_procedural_graph(
                    self.model,
                    self.tokenizer,
                    session_transcript,
                    session_id,
                    **call_kwargs,
                )
        return extract_procedural_graph(
            self.model,
            self.tokenizer,
            session_transcript,
            session_id,
            **call_kwargs,
        )
