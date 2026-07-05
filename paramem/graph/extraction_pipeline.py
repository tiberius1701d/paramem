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
from paramem.models.loader import base_model_inference

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
    ha_validation: bool = True
    noise_filter: str = "anthropic"
    noise_filter_model: str = "claude-sonnet-4-6"
    noise_filter_endpoint: str | None = None
    sota_enabled: bool = False  # master gate for ALL SOTA; mirrors ConsolidationConfig.sota_enabled
    ner_check: bool = False
    ner_model: str = "en_core_web_sm"
    plausibility_judge: str = "auto"
    plausibility_stage: str = "deanon"
    verify_anonymization: bool = True
    pii_scope: set[str] | frozenset[str] | None = None
    correction_entity_types: set[str] | frozenset[str] | None = None


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
      SessionGraph` — factual extraction.  Runs ``extract_graph`` inside
      :func:`~paramem.models.loader.base_model_inference`, which disables
      any active LoRA adapter (extraction must run on the base model,
      never the adapter being trained) and gradient checkpointing (HF
      silently disables the KV cache when checkpointing is active; the
      cache is required for ``model.generate()``) for the call, then
      restores both to their pre-scope state on exit.

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
        model_name: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ExtractionConfig()
        self.prompts_dir = prompts_dir
        # Model alias for per-file prompt resolution.  When set, the prompt
        # loader checks prompts_dir/<model_name>/<filename> before the shared
        # prompts_dir and the default configs/prompts/ directory.  A model
        # overrides only the files it provides; everything else inherits the
        # shared default.  Only local-model extraction prompts are per-model;
        # sota_* prompts are model-independent by design.
        self.model_name = model_name

    def kwargs(self, *, source_type: str = "transcript", **overrides) -> dict:
        """Build the kwarg dict passed into :func:`extract_graph`.

        ``source_type`` governs runtime gate defaults and is forwarded
        into the returned dict as ``source_type`` so :func:`extract_graph`
        (and, inside it, :func:`_stamp_speaker_entity`) can gate the
        document-only exact-full-name speaker rewrite on it. **Prompt
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

        * ``"transcript"`` (default) — ``ha_validation`` defaults to
          the config value (typically ``True``).

        * ``"document"`` — ``ha_validation`` defaults to ``False``
          (Home Assistant entity grounding is a dialogue-context
          concern). Operators may still override it explicitly via
          *overrides*.

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
            ha_validation_default = False
        else:
            ha_validation_default = cfg.ha_validation

        return dict(
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            plausibility_max_tokens=cfg.plausibility_max_tokens,
            prompts_dir=pick("prompts_dir", self.prompts_dir),
            # model_alias drives per-file prompt resolution in extract_graph /
            # extract_procedural_graph.  sota_* prompts are model-independent
            # by design and ignore this value.
            model_alias=self.model_name,
            ha_context=overrides.get("ha_context"),
            ha_validation=pick("ha_validation", ha_validation_default),
            noise_filter=pick("noise_filter", cfg.noise_filter),
            noise_filter_model=pick("noise_filter_model", cfg.noise_filter_model),
            noise_filter_endpoint=pick("noise_filter_endpoint", cfg.noise_filter_endpoint),
            sota_enabled=pick("sota_enabled", cfg.sota_enabled),
            speaker_name=overrides.get("speaker_name"),
            ner_check=pick("ner_check", cfg.ner_check),
            ner_model=pick("ner_model", cfg.ner_model),
            plausibility_judge=pick("plausibility_judge", cfg.plausibility_judge),
            plausibility_stage=pick("plausibility_stage", cfg.plausibility_stage),
            verify_anonymization=pick("verify_anonymization", cfg.verify_anonymization),
            pii_scope=pick("pii_scope", cfg.pii_scope),
            correction_entity_types=pick("correction_entity_types", cfg.correction_entity_types),
            speaker_id=_require_speaker_id(overrides),
            system_prompt_filename=pick("system_prompt_filename", system_prompt_filename),
            user_prompt_filename=pick("user_prompt_filename", user_prompt_filename),
            stop_phase=overrides.get("stop_phase"),
            seed=overrides.get("seed"),
            timestamp=overrides.get("timestamp"),
            source_type=source_type,
        )

    def _run_extractor(
        self,
        extract_fn,
        session_transcript: str,
        session_id: str,
        call_kwargs: dict,
    ) -> "SessionGraph":
        """Shared execution primitive for :meth:`run` and :meth:`run_procedural`.

        Runs the extractor inside :func:`base_model_inference`, which disables
        gradient checkpointing for the call (HF silently disables the KV cache
        when checkpointing is active; the cache is required for
        ``model.generate()``), disables any active LoRA adapter when
        ``self.model`` is a ``PeftModel`` (extraction must run on the base
        model, never the adapter being trained), and restores both to their
        entry state on exit.  ``extract_fn`` is ``extract_graph`` or
        ``extract_procedural_graph``.
        """
        with base_model_inference(self.model):
            return extract_fn(
                self.model,
                self.tokenizer,
                session_transcript,
                session_id,
                **call_kwargs,
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
        kwargs = self.kwargs(source_type=source_type, **overrides)
        return self._run_extractor(extract_graph, session_transcript, session_id, kwargs)

    def run_procedural(
        self,
        session_transcript: str,
        session_id: str,
        *,
        speaker_id: str,
        speaker_name: str | None = None,
        source_type: str = "transcript",
        timestamp: str | None = None,
        prompts_dir: str | Path | None = None,
        seed: int | None = None,
        system_prompt_filename: str | None = None,
        user_prompt_filename: str | None = None,
    ) -> "SessionGraph":
        """Single entry point for procedural (preferences/habits) extraction.

        ``source_type`` mirrors :meth:`run`.  The prompt-pair is the
        same regardless of source type; document chunks land in the
        ``{transcript}`` slot like transcripts do.

        ``speaker_id`` is stamped onto every ``Relation`` produced by the
        procedural extractor as provenance.  Required — empty / ``None``
        raises :exc:`ValueError`.

        ``timestamp``: session-start assertion time (ISO 8601), forwarded to
        :func:`~paramem.graph.extractor.extract_procedural_graph` so
        ``last_seen`` on newly-merged edges reflects when the facts were
        asserted.  ``None`` (default) falls back to ``now()``.

        ``prompts_dir``/``seed``/``system_prompt_filename``/
        ``user_prompt_filename`` mirror the same-named overrides on
        :meth:`run` (via :meth:`kwargs`): each defaults to ``None``, which
        preserves the pre-existing production behaviour (``self.prompts_dir``,
        no seed, the two ``DEFAULT_*`` prompt filenames) unchanged for every
        existing caller. ``stop_phase`` is intentionally NOT exposed here —
        procedural extraction is a single-phase call, so a stop point is
        inapplicable.
        """
        if not speaker_id:
            raise ValueError(
                "speaker_id is required for procedural extraction (no empty-string default)."
            )

        cfg = self.config
        # Single prompt-pair for both source types — see :meth:`kwargs`.
        resolved_system_prompt_filename = system_prompt_filename or DEFAULT_SYSTEM_PROMPT_FILENAME
        resolved_user_prompt_filename = (
            user_prompt_filename or DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME
        )
        call_kwargs = dict(
            max_tokens=cfg.max_tokens,
            prompts_dir=prompts_dir if prompts_dir is not None else self.prompts_dir,
            # model_alias drives per-file prompt resolution — see kwargs() comment.
            model_alias=self.model_name,
            speaker_name=speaker_name,
            speaker_id=speaker_id,
            system_prompt_filename=resolved_system_prompt_filename,
            user_prompt_filename=resolved_user_prompt_filename,
            seed=seed,
            timestamp=timestamp,
            source_type=source_type,
        )
        return self._run_extractor(
            extract_procedural_graph, session_transcript, session_id, call_kwargs
        )
