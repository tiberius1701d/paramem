#!/usr/bin/env python3
"""Compare extraction quality: Claude vs Mistral vs Gemma 4 — same pipeline.

All models go through the same code path:
1. Same extraction prompt
2. Same JSON parsing
3. Same normalization
4. Same STT correction

Each model pass saves results immediately. Re-running skips completed passes.
Saved passes are not validated against the current session set — when running
with a different --path, either delete stale passes first or use --rerun.

Usage:
    # Stop the server first (frees GPU VRAM)
    systemctl --user stop paramem-server

    export $(grep -v '^#' .env | xargs)

    # Default: all sessions (data/ha/sessions/ + archive/)
    python scripts/compare_extraction.py

    # Curated evaluation set (folder of transcripts):
    python scripts/compare_extraction.py --path data/ha/debug/eval_set_v1/

    # Debug a single transcript:
    python scripts/compare_extraction.py --path data/ha/sessions/01KN....jsonl

    # Force re-run a specific model:
    python scripts/compare_extraction.py --rerun gemma4
"""

import argparse
import gc
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paramem.graph.extractor import (
    _correct_entity_names,
    _extract_json_block,
    _normalize_extraction,
    load_extraction_prompts,
)
from paramem.graph.schema import SessionGraph

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
for name in ("httpx", "anthropic", "urllib3", "transformers", "accelerate", "bitsandbytes"):
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Defaults — overridable via CLI.
DEFAULT_PROMPT_DIR = Path("configs/prompts")
DEFAULT_SESSION_DIR = Path("data/ha/sessions")
DEFAULT_OUTPUT_DIR = Path("data/ha/debug/extraction_comparison")
DEFAULT_MAX_TOKENS = 2048
DEFAULT_ANON_TEMPERATURE = 0.0  # keep JSON deterministic regardless of extraction temp
DEFAULT_VALIDATOR_TEMPERATURE = 0.0  # keep validator deterministic by default

# These globals hold the effective values for the current run. main() sets
# them from CLI args before any pass runs. Kept module-level so the existing
# generate_*/save_pass/load_pass functions don't need threading changes
# every time a new knob is added.
PROMPT_DIR = DEFAULT_PROMPT_DIR
SESSION_DIR = DEFAULT_SESSION_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
MAX_TOKENS = DEFAULT_MAX_TOKENS
VALIDATOR_TEMPERATURE = DEFAULT_VALIDATOR_TEMPERATURE


def _load_transcript_file(f: Path, skip_test_lang: bool = True) -> dict | None:
    """Parse one session JSONL into a {session_id, transcript} dict. None if skipped."""
    if skip_test_lang and f.stem == "test-lang":
        return None
    lines = f.read_text().strip().split("\n")
    turns = [json.loads(line) for line in lines]
    text_parts = [f"[{t.get('role', 'user')}] {t.get('text', '')}" for t in turns]
    transcript = "\n".join(text_parts)
    if len(transcript) <= 50:
        return None
    return {"session_id": f.stem, "transcript": transcript}


def load_transcripts(path: Path | None = None) -> list[dict]:
    """Load transcripts.

    - path=None: default — all sessions under SESSION_DIR + archive/
    - path is a file: load that single transcript (no test-lang skip — explicit)
    - path is a directory: load every *.jsonl directly inside it (no recursion)
    """
    if path is not None and path.is_file():
        t = _load_transcript_file(path, skip_test_lang=False)
        return [t] if t is not None else []

    search_dirs: list[Path]
    if path is None:
        search_dirs = [SESSION_DIR, SESSION_DIR / "archive"]
    elif path.is_dir():
        search_dirs = [path]
    else:
        raise FileNotFoundError(f"--path not found: {path}")

    files: list[Path] = []
    for d in search_dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.jsonl")))

    transcripts = []
    for f in files:
        t = _load_transcript_file(f)
        if t is not None:
            transcripts.append(t)
    return transcripts


def run_extraction(raw_output: str, transcript: str, session_id: str) -> SessionGraph:
    """Shared extraction pipeline: parse -> normalize -> STT correct."""
    try:
        json_str = _extract_json_block(raw_output)
        data = json.loads(json_str)
        data["session_id"] = session_id
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        data = _normalize_extraction(data)
        graph = SessionGraph.model_validate(data)
    except Exception as e:
        logger.warning("Parse error: %s", e)
        return SessionGraph(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    if not graph.relations:
        return graph

    graph = _correct_entity_names(graph, transcript)
    return graph


# ---------------------------------------------------------------------------
# Cloud provider dispatch — generic across Anthropic + OpenAI-compatible
# endpoints. Used by both the extraction path and the anonymization path so
# any cloud extractor can self-anonymize with no Claude-specific branches.
# ---------------------------------------------------------------------------


def _cloud_chat(
    provider: str,
    api_key: str,
    model_id: str,
    system: str,
    user: str,
    temperature: float,
    endpoint: str | None = None,
) -> str:
    """Single entry point for any cloud chat completion.

    Dispatches by provider name. Supports Anthropic (native SDK) and any
    OpenAI-compatible endpoint (OpenAI, Groq, Mistral, Ollama, …).
    """
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_id,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in response.content if hasattr(b, "text"))

    from paramem.graph.extractor import OPENAI_COMPAT_ENDPOINTS, OPENAI_COMPAT_PROVIDERS

    if provider in OPENAI_COMPAT_PROVIDERS:
        import httpx

        url = endpoint or OPENAI_COMPAT_ENDPOINTS.get(provider, "")
        if not url:
            raise RuntimeError(
                f"No endpoint configured for provider {provider!r}; pass endpoint= explicitly"
            )
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise RuntimeError(f"Unsupported cloud provider {provider!r}")


def _speaker_context(speaker_name: str | None) -> str:
    """Thin wrapper over the shared `build_speaker_context` in the extractor module."""
    from paramem.graph.extractor import build_speaker_context

    return build_speaker_context(speaker_name)


def _generate_with_cloud(
    info: dict, transcript: str, temperature: float, speaker_name: str | None = None
) -> str:
    """Extraction via a cloud provider registered in EXTRACTORS_CLOUD.

    On API failure logs and returns an empty string — the pipeline parses
    that as an empty extraction for this session and continues. We do NOT
    fail the whole sweep on a transient cloud error.
    """
    system, prompt = load_extraction_prompts(PROMPT_DIR)
    try:
        return _cloud_chat(
            info["provider"],
            os.environ[info["key_env"]],
            info["model_id"],
            system,
            prompt.format(transcript=transcript, speaker_context=_speaker_context(speaker_name)),
            temperature,
            info.get("endpoint"),
        )
    except Exception as e:
        cause = e.__cause__ or e.__context__
        detail = f"{type(e).__name__}: {e}"
        if cause:
            detail += f" (caused by {type(cause).__name__}: {cause})"
        logger.warning("%s extraction call failed — %s", info["provider"], detail)
        return ""


def generate_with_local(
    transcript: str,
    model,
    tokenizer,
    temperature: float = 0.0,
    speaker_name: str | None = None,
) -> str:
    """Get raw extraction output from a local model."""
    from paramem.evaluation.recall import generate_answer
    from paramem.models.loader import adapt_messages

    system, prompt = load_extraction_prompts(PROMPT_DIR)
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": prompt.format(
                transcript=transcript, speaker_context=_speaker_context(speaker_name)
            ),
        },
    ]
    formatted = tokenizer.apply_chat_template(
        adapt_messages(messages, tokenizer),
        tokenize=False,
        add_generation_prompt=True,
    )
    return generate_answer(
        model, tokenizer, formatted, max_new_tokens=MAX_TOKENS, temperature=temperature
    )


def serialize_relations(graph: SessionGraph) -> list[dict]:
    return [
        {
            "subject": r.subject,
            "predicate": r.predicate,
            "object": r.object,
            "relation_type": r.relation_type,
        }
        for r in graph.relations
    ]


def print_relations(graph: SessionGraph):
    if graph.relations:
        for r in graph.relations:
            print(
                f"    {r.subject} --[{r.predicate}]--> {r.object}"
                f"  ({r.relation_type}, conf={r.confidence})"
            )
    else:
        print("    (none)")


def _pass_path(
    extractor: str,
    temperature: float,
    validator: str,
    plausibility_judge: str = "auto",
    plausibility_stage: str = "deanon",
    verify_anonymization: bool = True,
) -> Path:
    """Filename scoped by every config axis that affects results."""
    va = "on" if verify_anonymization else "off"
    return OUTPUT_DIR / (
        f"{extractor}_t{temperature:.2f}_v{validator}_vt{VALIDATOR_TEMPERATURE:.2f}"
        f"_pj{plausibility_judge}_ps{plausibility_stage}_va{va}.json"
    )


def save_pass(
    extractor: str,
    temperature: float,
    validator: str,
    plausibility_judge: str,
    plausibility_stage: str,
    verify_anonymization: bool,
    results: dict,
):
    """Save a single pass to disk immediately."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _pass_path(
        extractor,
        temperature,
        validator,
        plausibility_judge,
        plausibility_stage,
        verify_anonymization,
    )
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [{extractor}] Results saved to {path}")


def load_pass(
    extractor: str,
    temperature: float,
    validator: str,
    plausibility_judge: str,
    plausibility_stage: str,
    verify_anonymization: bool,
    expected_session_ids: set[str] | None = None,
) -> dict | None:
    """Load a previously completed pass for this exact configuration.

    If `expected_session_ids` is provided, verify the cached pass covers
    the same session set. On mismatch, treat as a cache miss and warn — a
    silent apples-to-oranges comparison has cost us full GPU/API runs in
    the past.
    """
    path = _pass_path(
        extractor,
        temperature,
        validator,
        plausibility_judge,
        plausibility_stage,
        verify_anonymization,
    )
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        if expected_session_ids is not None:
            cached_ids = set(data.keys())
            if cached_ids != expected_session_ids:
                only_cached = cached_ids - expected_session_ids
                only_current = expected_session_ids - cached_ids
                print(
                    f"  WARNING: cached {path.name} covers a different session set — "
                    f"{len(only_cached)} only in cache, {len(only_current)} only in "
                    f"current run. Ignoring cache; will rerun {extractor}."
                )
                return None
        return data
    return None


def unload_model(model, tokenizer):
    """Free GPU memory after a model pass."""
    import torch

    if hasattr(model, "cpu"):
        model.cpu()
    del tokenizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# --- Uniform pipeline — identical treatment for every extractor ---
#
# Every extractor (Claude API or any local model) goes through the same
# three-stage SOTA pipeline: anonymize → validate (enrich) → de-anonymize.
# The anonymizer and validator are deliberate, separately-configurable
# choices, held CONSTANT across all extractors in a sweep. The only
# variable across extractors is the raw extraction quality.


def _anonymize_via_cloud(
    graph: SessionGraph, info: dict, transcript: str = ""
) -> tuple[list[dict] | None, dict, str]:
    """Call a cloud provider with the anonymization prompt.

    Returns (anon_facts, mapping, anon_transcript) — same shape as the local
    anonymizer. Uses `load_anonymization_prompt()` so a file override applies
    symmetrically to both paths. Backward-compat: if the model returns the old
    schema (`anonymized` key, no `anonymized_transcript`), the transcript slot
    comes back empty and the caller falls back to mechanical replacement.
    """
    from paramem.graph.extractor import _extract_json_block, load_anonymization_prompt

    facts = [
        {
            "subject": r.subject,
            "predicate": r.predicate,
            "object": r.object,
            "relation_type": r.relation_type,
            "confidence": r.confidence,
        }
        for r in graph.relations
    ]
    prompt = load_anonymization_prompt().format(
        facts_json=json.dumps(facts, indent=2),
        transcript=transcript or "(no transcript provided)",
    )
    try:
        raw = _cloud_chat(
            info["provider"],
            os.environ[info["key_env"]],
            info["model_id"],
            "You anonymize data. Output valid JSON only.",
            prompt,
            DEFAULT_ANON_TEMPERATURE,
            info.get("endpoint"),
        )
    except Exception as e:
        cause = e.__cause__ or e.__context__
        detail = f"{type(e).__name__}: {e}"
        if cause:
            detail += f" (caused by {type(cause).__name__}: {cause})"
        logger.warning("%s anonymization call failed — %s", info["provider"], detail)
        return None, {}, ""
    try:
        from paramem.graph.extractor import _normalize_anonymization_mapping

        data = json.loads(_extract_json_block(raw))
        if isinstance(data, dict) and "mapping" in data:
            normalized, _ = _normalize_anonymization_mapping(data["mapping"])
            if "anonymized_facts" in data:
                return (
                    data["anonymized_facts"],
                    normalized,
                    data.get("anonymized_transcript", ""),
                )
            if "anonymized" in data:  # backward compat
                return data["anonymized"], normalized, ""
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("%s anonymization parse failed: %s", info["provider"], e)
    return None, {}, ""


# ---------------------------------------------------------------------------
# Registries — add new entries here, no other code changes needed.
# ---------------------------------------------------------------------------

# Cloud extractors: any model reachable via API. Each entry: provider, model_id,
# env var holding the API key, optional endpoint override for OpenAI-compat hosts.
EXTRACTORS_CLOUD: dict[str, dict] = {
    "claude": {
        "type": "cloud",
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "key_env": "ANTHROPIC_API_KEY",
    },
    # Example additions (uncomment when needed):
    # "gpt4": {"type": "cloud", "provider": "openai",
    #          "model_id": "gpt-4-turbo", "key_env": "OPENAI_API_KEY"},
}

# SOTA validators: cloud services that see only anonymized data. Same shape as
# EXTRACTORS_CLOUD entries so anything listed here can also serve as a cloud
# extractor (by also adding it to EXTRACTORS_CLOUD).
VALIDATORS: dict[str, dict] = {
    "claude": {
        "type": "cloud",
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "key_env": "ANTHROPIC_API_KEY",
    },
}


def list_extractors() -> dict[str, dict]:
    """Combined registry: cloud extractors + local models from MODEL_REGISTRY.

    Returns a mapping of name → info dict. Cloud entries carry provider/model
    metadata. Local entries carry {"type": "local"} and are resolved to a
    ModelConfig from paramem.server.config.MODEL_REGISTRY when a pass runs.
    """
    from paramem.server.config import MODEL_REGISTRY

    combined: dict[str, dict] = dict(EXTRACTORS_CLOUD)
    for name in MODEL_REGISTRY:
        if name not in combined:
            combined[name] = {"type": "local"}
    return combined


def _validator_call(
    func_name: str, anon_facts: list[dict], anon_transcript: str, validator: str
) -> list[dict] | None:
    """Generic call to a SOTA pipeline stage using the configured validator.

    Validator temperature is `VALIDATOR_TEMPERATURE` (default 0.0). Override
    via `--validator-temperature` if a study deliberately calls for sampling
    on the validator side — same value applies to both enrichment and
    plausibility for parity.
    """
    from paramem.graph import extractor as ext

    info = VALIDATORS[validator]
    api_key = os.environ.get(info["key_env"], "")
    if not api_key:
        logger.warning("No %s set for validator %r", info["key_env"], validator)
        return None
    return getattr(ext, func_name)(
        anon_facts,
        api_key,
        info["provider"],
        info["model_id"],
        anon_transcript,
        endpoint=info.get("endpoint"),
        max_tokens=MAX_TOKENS,
        temperature=VALIDATOR_TEMPERATURE,
    )


def enrich(
    anon_facts: list[dict], anon_transcript: str, validator: str
) -> tuple[list[dict] | None, str | None, str | None]:
    """SOTA enrichment — returns `(facts, updated_transcript, raw_response)`.

    Updated transcript carries any new placeholders SOTA introduced (braced
    form). Callers should diff it against the input `anon_transcript` to
    recover SOTA's new bindings. Raw response is preserved for diagnostics —
    essential when binding recovery fails and we need to see what SOTA
    actually emitted. Legacy responses (bare array, no transcript) return
    `(facts, None, raw)`.
    """
    from paramem.graph import extractor as ext

    info = VALIDATORS[validator]
    api_key = os.environ.get(info["key_env"], "")
    if not api_key:
        logger.warning("No %s set for validator %r", info["key_env"], validator)
        return None, None, None
    return ext._filter_with_sota(
        anon_facts,
        api_key,
        info["provider"],
        info["model_id"],
        anon_transcript,
        endpoint=info.get("endpoint"),
        max_tokens=MAX_TOKENS,
        temperature=VALIDATOR_TEMPERATURE,
    )


def plausibility_filter_sota(
    facts: list[dict], transcript: str, validator: str
) -> tuple[list[dict] | None, str | None]:
    """SOTA plausibility filter — drops invalid relations only.

    Returns `(facts, raw_response)`. Raw preserved for audit of judge's
    verdict when questioning drop decisions.
    """
    from paramem.graph import extractor as ext

    info = VALIDATORS[validator]
    api_key = os.environ.get(info["key_env"], "")
    if not api_key:
        logger.warning("No %s set for validator %r", info["key_env"], validator)
        return None, None
    return ext._plausibility_filter_with_sota(
        facts,
        api_key,
        info["provider"],
        info["model_id"],
        transcript,
        endpoint=info.get("endpoint"),
        max_tokens=MAX_TOKENS,
        temperature=VALIDATOR_TEMPERATURE,
    )


def plausibility_filter_local(
    facts: list[dict], transcript: str, model, tokenizer
) -> list[dict] | None:
    """Local-model plausibility filter — drops invalid relations only.

    Operates on whatever data the caller passes — typically de-anonymized
    real-name facts + real-name transcript. Returns filtered list or None.
    """
    from paramem.graph.extractor import _local_plausibility_filter

    return _local_plausibility_filter(
        facts,
        transcript,
        model,
        tokenizer,
        max_tokens=MAX_TOKENS,
        temperature=VALIDATOR_TEMPERATURE,
    )


def _self_anonymize_with_transcript(
    graph: SessionGraph,
    transcript: str,
    extractor: str,
    local_model,
    local_tokenizer,
) -> tuple[list[dict] | None, dict, str]:
    """Self-anonymize facts AND transcript in one call.

    Returns (anon_facts, mapping, anon_transcript). Falls back to a mechanical
    transcript anonymization if the model didn't return one (older anonymizer).
    """
    from paramem.graph.extractor import _anonymize_transcript

    info = list_extractors()[extractor]
    if info["type"] == "cloud":
        anon_facts, mapping, anon_transcript = _anonymize_via_cloud(
            graph, info, transcript=transcript
        )
        if anon_facts is not None and not anon_transcript:
            # Cloud model used the older schema — fall back to mechanical replacement
            anon_transcript = _anonymize_transcript(transcript, mapping)
        return anon_facts, mapping, anon_transcript

    from paramem.graph.extractor import _anonymize_with_local_model

    anon_facts, mapping, anon_transcript = _anonymize_with_local_model(
        graph,
        local_model,
        local_tokenizer,
        transcript=transcript,
        max_tokens=MAX_TOKENS,
        temperature=DEFAULT_ANON_TEMPERATURE,
    )
    if anon_facts is not None and not anon_transcript:
        # Local model used the older schema — fall back to mechanical replacement
        anon_transcript = _anonymize_transcript(transcript, mapping)
    return anon_facts, mapping, anon_transcript


def _enrich_uniformly(
    extracted: SessionGraph,
    transcript: str,
    extractor: str,
    local_model,
    local_tokenizer,
    validator: str,
    plausibility_judge: str = "auto",
    plausibility_stage: str = "deanon",
    verify_anonymization: bool = True,
    ner_check: bool = False,
    ner_model: str = "en_core_web_sm",
) -> tuple[SessionGraph, dict]:
    """Anonymize → SOTA enrich → de-anonymize, with configurable plausibility filter.

    Forward-path privacy guard (`verify_anonymization`): after self-anonymization,
    verify no real name from the extracted graph leaked into the SOTA-bound data.
    On a leak, abort the SOTA call and return the un-enriched graph. Defaults to
    ON; the test harness can turn it off for deliberate leak-investigation runs.
    Production callers must leave it on.

    Plausibility filter is composable:
      - judge ∈ {"auto", "off", "<extractor-name>", "<any-validator-name>"}.
        "auto" means "same model as the extractor" (cross-family with the SOTA enricher).
        "off" skips the plausibility step entirely.
      - stage ∈ {"anon", "deanon"} — operate on anonymized or real-name data.
        "deanon" works for any local plausibility judge AND surfaces leaked
        placeholders as obvious drops. "anon" is required for SOTA judges
        (privacy: SOTA must never see real names).

    Returns (enriched_graph, pipeline_status) with per-stage outcomes
    ("ok" / "failed" / "skipped" / "off" / "leaked").
    """
    from paramem.graph.extractor import (
        extract_pii_names_with_ner,
        verify_anonymization_completeness,
    )
    from paramem.graph.schema import Relation

    # Initialize ALL status keys in one place so both the anon-stage and
    # deanon-stage branches update rather than a later block overwriting.
    status: dict = {
        "anonymize": "skipped",
        "repair": "skipped",
        "enrich": "skipped",
        "plausibility": "skipped",
        "grounding_gate": "skipped",
        "plausibility_judge": plausibility_judge,
        "plausibility_stage": plausibility_stage,
        "plausibility_judge_actual": None,
        "plausibility_dropped": 0,
        "facts_pre_plausibility": [],
        "facts_post_plausibility": [],
        "anonymization_verified": verify_anonymization,
        "fallback_path": None,
    }

    def _fallback_plausibility_on_raw(reason: str) -> tuple[SessionGraph, dict]:
        """Run plausibility on raw extraction (real names) when SOTA path unavailable.

        The raw graph was produced by the local extractor from the real
        transcript, so facts are already in real names — no de-anonymization
        needed. A PA never aborts; the pipeline always produces a final
        result, even if SOTA enrichment was unsafe or failed.
        """
        from paramem.graph.extractor import (
            _drop_ungrounded_facts as _gate,
        )
        from paramem.graph.extractor import (
            _strip_residual_placeholders as _sweep,
        )

        status["fallback_path"] = reason
        raw_facts = serialize_relations(extracted)
        # Defense-in-depth: if the local extractor produced placeholder-like
        # artifacts (very rare), drop them before plausibility.
        raw_facts, sweep_dropped = _sweep(raw_facts)
        if sweep_dropped:
            status["residual_placeholders_dropped"] = len(sweep_dropped)
            status["residual_dropped_facts"] = sweep_dropped
        # Privacy gate also applies in fallback: local extractors can
        # hallucinate world-knowledge entities too (not exclusive to SOTA).
        # `known_names=set()` is deliberate: the main path trusts
        # `mapping.keys()` because the anonymizer is guaranteed to have only
        # mapped names present in the transcript. In fallback there's no
        # mapping to trust, and `extracted.entities` can itself contain
        # hallucinations (e.g. Qwen's generic "Speaker"). Every entity must
        # therefore be proven by transcript grounding alone.
        raw_facts, fallback_ungrounded = _gate(raw_facts, transcript, set())
        if fallback_ungrounded:
            status["ungrounded_dropped_facts"] = fallback_ungrounded
            status["grounding_gate"] = f"dropped_{len(fallback_ungrounded)}"
            logger.warning("Fallback: dropped %d ungrounded fact(s).", len(fallback_ungrounded))
        else:
            status["grounding_gate"] = "ok"
        status["facts_pre_plausibility"] = list(raw_facts)
        if not raw_facts or plausibility_judge == "off":
            status["plausibility"] = "off" if plausibility_judge == "off" else "skipped"
            status["facts_post_plausibility"] = list(raw_facts)
            return extracted, status
        judge = plausibility_judge if plausibility_judge != "auto" else extractor
        plausible = _run_plausibility_judge(
            judge,
            raw_facts,
            transcript,
            local_model,
            local_tokenizer,
            expected_extractor=extractor,
        )
        if plausible is None:
            logger.warning("Fallback plausibility (%s, raw) failed — keeping raw extraction", judge)
            status["plausibility"] = "failed"
            status["facts_post_plausibility"] = list(raw_facts)
            return extracted, status
        pre, post = len(raw_facts), len(plausible)
        status["plausibility_dropped"] = pre - post
        status["plausibility"] = "ok_but_empty" if (post == 0 and pre > 0) else "ok"
        status["plausibility_judge_actual"] = judge
        status["facts_post_plausibility"] = list(plausible)
        from paramem.graph.extractor import _canonicalize_symmetric_predicates

        kept = []
        for f in plausible:
            try:
                kept.append(Relation(**f))
            except Exception:
                continue
        kept = _canonicalize_symmetric_predicates(kept)
        kept_names = {n for r in kept for n in (r.subject, r.object)}
        kept_entities = [e for e in extracted.entities if e.name in kept_names]
        return extracted.model_copy(update={"relations": kept, "entities": kept_entities}), status

    # Stage 1: anonymize facts + transcript via extractor (one mapping)
    anon_facts, mapping, anon_transcript = _self_anonymize_with_transcript(
        extracted, transcript, extractor, local_model, local_tokenizer
    )
    if anon_facts is None:
        logger.warning(
            "Anonymization failed — running local plausibility on raw extraction (no SOTA call)"
        )
        status["anonymize"] = "failed"
        return _fallback_plausibility_on_raw("anon_failed")
    if not anon_facts:
        logger.info("Anonymization produced 0 facts — skipping pipeline")
        status["anonymize"] = "ok"
        status["grounding_gate"] = "no_input"
        return (
            extracted.model_copy(update={"relations": [], "entities": []}),
            status,
        )

    # Canonicalize mapping direction before any downstream use.
    from paramem.graph.extractor import (
        _mapping_is_canonical,
        _normalize_anonymization_mapping,
        _repair_anonymization_leaks,
        _strip_residual_placeholders,
    )

    mapping, norm_stats = _normalize_anonymization_mapping(mapping)
    if norm_stats["dropped"]:
        status["mapping_ambiguous_dropped"] = norm_stats["dropped"]

    # Forward-path privacy guard.
    extra_pii = extract_pii_names_with_ner(transcript, ner_model) if ner_check else None
    if ner_check and extra_pii is not None:
        status["ner_pii_detected"] = sorted(extra_pii)[:20]
    leaked = verify_anonymization_completeness(
        extracted, mapping, anon_facts, anon_transcript, extra_pii_names=extra_pii
    )
    status["repair"] = "skipped"
    _skip_sota = False
    if leaked:
        status["anonymization_leaks"] = leaked[:10]
        if not verify_anonymization:
            logger.warning(
                "Anonymization incomplete — real names leaked: %s. "
                "Proceeding anyway because --no-verify-anonymization is set. "
                "Real names WILL be sent to the SOTA validator.",
                leaked[:5],
            )
        elif _mapping_is_canonical(mapping):
            logger.info("Repairing %d leaked name(s): %s", len(leaked), leaked[:5])
            anon_facts, mapping, anon_transcript, repair_stats = _repair_anonymization_leaks(
                extracted, mapping, anon_facts, anon_transcript, transcript, leaked
            )
            status["repair"] = "ok"
            status["repair_stats"] = repair_stats
            leaked = verify_anonymization_completeness(
                extracted, mapping, anon_facts, anon_transcript, extra_pii_names=extra_pii
            )
            if leaked:
                # Residual leak after repair. Do fact-level filtering rather
                # than aborting the whole session: drop only triples that
                # reference residually-leaked names, keep the rest. Skip SOTA
                # (transcript is still dirty — cannot send to cloud), but
                # de-anonymize and return the clean remaining subset locally.
                status["repair"] = "partial"
                repair_stats["residual_leaked"] = leaked[:10]
                leaked_lc = {n.lower() for n in leaked}
                pre_filter = len(anon_facts)
                anon_facts = [
                    f
                    for f in anon_facts
                    if isinstance(f, dict)
                    and str(f.get("subject", "")).lower() not in leaked_lc
                    and str(f.get("object", "")).lower() not in leaked_lc
                ]
                repair_stats["residual_leaked_triples_dropped"] = pre_filter - len(anon_facts)
                logger.warning(
                    "Residual leaks after repair (%s); dropped %d referencing triple(s), "
                    "skipping SOTA (dirty transcript), continuing locally.",
                    leaked[:5],
                    pre_filter - len(anon_facts),
                )
                status["anonymize"] = "leaked_repaired"
                # Signal downstream to skip SOTA enrichment and plausibility
                # on anon (both would expose the leaked real-name). Local
                # plausibility + grounding gate still apply.
                _skip_sota = True
        else:
            logger.warning(
                "Mapping non-canonical and leak detected — cannot safely repair. "
                "Falling back to raw plausibility on original extraction."
            )
            status["anonymize"] = "leaked"
            return _fallback_plausibility_on_raw("anon_leaked_noncanonical")
    # Preserve leaked_repaired state set during residual-leak handling; only
    # mark fully-ok when no leak path was taken.
    if status.get("anonymize") not in ("leaked_repaired",):
        status["anonymize"] = "ok"

    # Stage 2: SOTA enrichment. Returns (facts, updated_transcript, raw_response).
    # Skipped when the transcript carries a residual real-name leak we can't
    # scrub — sending it would violate privacy. Fall through to local
    # plausibility + grounding gate on the leak-filtered anon_facts.
    if _skip_sota:
        enriched_anon, updated_anon_transcript, sota_raw = anon_facts, None, None
        status["enrich"] = "skipped_dirty_transcript"
    else:
        enriched_anon, updated_anon_transcript, sota_raw = enrich(
            anon_facts, anon_transcript, validator
        )
        status["sota_raw_response"] = sota_raw
        if enriched_anon is None:
            logger.warning("Enrichment failed — falling back to anonymized extraction")
            status["enrich"] = "failed"
            enriched_anon = anon_facts
        else:
            status["enrich"] = "ok"

    # Diagnostic: preserve all four transcripts for post-hoc inspection.
    # Diffing original vs recovered reveals SOTA compliance at a glance.
    status["transcripts"] = {
        "original": transcript,
        "anonymized": anon_transcript,
        "sota_updated": updated_anon_transcript,
    }

    # Defensive brace-strip: always runs, even when SOTA was skipped. The
    # local anonymizer may emit braced placeholders in some model outputs,
    # and the de-anon lookup expects bare tokens. No-op when facts are
    # already bare.
    from paramem.graph.extractor import (
        _extract_sota_bindings,
        _strip_placeholder_braces,
    )

    # Recover SOTA-introduced placeholder bindings via transcript diff.
    # Ungrounded placeholders (in facts but not in updated_transcript) fall
    # through to the residual-placeholder sweep post de-anon.
    if updated_anon_transcript:
        bindings = _extract_sota_bindings(anon_transcript, updated_anon_transcript)
        for real_span, placeholder in bindings.items():
            mapping.setdefault(real_span, placeholder)
        if bindings:
            status["sota_bindings"] = len(bindings)
            logger.info("SOTA introduced %d new binding(s)", len(bindings))

        # Recover the real-name transcript by reverse-substituting every
        # placeholder in SOTA's output using the extended mapping. Word-
        # boundary regex + longest-first ordering prevents Person_1 from
        # eating the Person_1 prefix of Person_10. If SOTA only substituted
        # spans (no rewording), recovered == original.
        reverse = {v: k for k, v in mapping.items()}
        recovered = updated_anon_transcript
        for placeholder, real in sorted(reverse.items(), key=lambda kv: -len(kv[0] or "")):
            if not placeholder:
                continue
            recovered = re.sub(rf"\{{{re.escape(placeholder)}\}}", real, recovered)
            recovered = re.sub(rf"\b{re.escape(placeholder)}\b", real, recovered)
        status["transcripts"]["recovered"] = recovered
        if recovered.strip() != transcript.strip():
            # Coarse length-based change signal. Full text diff in the saved
            # status lets the user inspect divergence offline.
            status["transcripts"]["length_ratio"] = round(
                len(recovered) / max(len(transcript), 1), 3
            )
    enriched_anon = _strip_placeholder_braces(enriched_anon)

    # De-anonymize helper — used for both the pre-plausibility snapshot and the
    # post-plausibility result. Stage 3a (anon-stage plausibility) filters in
    # anon space; we de-anonymize both the pre and post lists so status keys
    # are consistent across modes (always in real-name space).
    def _deanon_list(items: list[dict]) -> list[dict]:
        reverse = {v: k for k, v in mapping.items()}
        out = []
        for f in items:
            if not isinstance(f, dict):
                continue
            subj = reverse.get(f.get("subject", ""), f.get("subject", ""))
            obj = reverse.get(f.get("object", ""), f.get("object", ""))
            out.append(
                {
                    "subject": subj,
                    "predicate": f.get("predicate", ""),
                    "object": obj,
                    "relation_type": f.get("relation_type", "factual"),
                    "confidence": float(f.get("confidence", 1.0)),
                }
            )
        return out

    # Stage 3a: plausibility on anonymized data (SOTA judge only — privacy-safe).
    # Skipped when transcript has residual leaks (SOTA plausibility also sends
    # the transcript to the cloud).
    if (
        not _skip_sota
        and enriched_anon
        and plausibility_judge != "off"
        and plausibility_stage == "anon"
        and plausibility_judge in VALIDATORS
    ):
        status["plausibility_judge_actual"] = plausibility_judge
        status["facts_pre_plausibility"] = _deanon_list(enriched_anon)
        pre_count = len(enriched_anon)
        plausible_anon, plausibility_raw = plausibility_filter_sota(
            enriched_anon, anon_transcript, plausibility_judge
        )
        status["sota_plausibility_raw_response"] = plausibility_raw
        if plausible_anon is None:
            logger.warning("Plausibility (SOTA, anon) failed — keeping enriched output")
            status["plausibility"] = "failed"
            status["facts_post_plausibility"] = list(status["facts_pre_plausibility"])
        else:
            post_count = len(plausible_anon)
            status["plausibility_dropped"] = pre_count - post_count
            if post_count == 0 and pre_count > 0:
                status["plausibility"] = "ok_but_empty"
            else:
                status["plausibility"] = "ok"
            enriched_anon = plausible_anon
            status["facts_post_plausibility"] = _deanon_list(enriched_anon)

    if not enriched_anon:
        # Distinguish three empty states: (a) nothing to extract from the
        # start, (b) all SOTA-added facts were filtered pre-de-anon, (c)
        # residual-leak filter scrubbed everything. Each preserves its own
        # upstream status; the gate didn't actually run in any case.
        status["grounding_gate"] = "no_input"
        return (
            extracted.model_copy(update={"relations": [], "entities": []}),
            status,
        )

    # Stage 3b: de-anonymize (final result path) + placeholder sweep.
    deanon_facts_pre_sweep = _deanon_list(enriched_anon)
    # Preserve the pre-sweep de-anonymized output for diagnostics. When the
    # all-dropped fallback later overwrites facts_pre/post_plausibility with
    # raw-extraction data, this field still shows what the primary pipeline
    # actually produced before the residual sweep ran.
    status["primary_deanon_facts"] = list(deanon_facts_pre_sweep)
    deanon_facts, dropped_facts = _strip_residual_placeholders(deanon_facts_pre_sweep)
    if dropped_facts:
        status["residual_placeholders_dropped"] = len(dropped_facts)
        status["residual_dropped_facts"] = dropped_facts
        logger.warning(
            "Dropped %d fact(s) with residual placeholder strings post-de-anon.",
            len(dropped_facts),
        )

    # Final privacy gate: drop SOTA world-knowledge inferences by requiring
    # every surviving fact endpoint to appear in the original transcript
    # (or be a known real-name placeholder from the mapping).
    from paramem.graph.extractor import _drop_ungrounded_facts

    known_real_names = set(mapping.keys())
    deanon_facts, ungrounded_facts = _drop_ungrounded_facts(
        deanon_facts, transcript, known_real_names
    )
    if ungrounded_facts:
        status["ungrounded_dropped_facts"] = ungrounded_facts
        status["grounding_gate"] = f"dropped_{len(ungrounded_facts)}"
        logger.warning(
            "Dropped %d fact(s) ungrounded in the transcript (likely SOTA "
            "world-knowledge inference).",
            len(ungrounded_facts),
        )
    else:
        status["grounding_gate"] = "ok"

    # If no plausibility runs (any reason — off, stage mismatch, empty), the
    # snapshot keys reflect the de-anonymized post-enrichment state.
    if status["plausibility"] == "skipped":
        status["facts_pre_plausibility"] = list(deanon_facts)
        status["facts_post_plausibility"] = list(deanon_facts)

    # Stage 3c: plausibility on de-anonymized data (local judges — real names).
    if deanon_facts and plausibility_judge != "off" and plausibility_stage == "deanon":
        judge = plausibility_judge if plausibility_judge != "auto" else extractor
        # Record the effective judge — may differ from the requested one if the
        # requested local model isn't loaded (silent downgrade in
        # `_run_plausibility_judge`). The requested name stays in
        # status["plausibility_judge"]; this one is what actually ran.
        status["plausibility_judge_actual"] = (
            judge if judge in VALIDATORS or judge == extractor else extractor
        )
        status["facts_pre_plausibility"] = list(deanon_facts)
        plausible_deanon = _run_plausibility_judge(
            judge,
            deanon_facts,
            transcript,
            local_model,
            local_tokenizer,
            expected_extractor=extractor,
        )
        if plausible_deanon is None:
            logger.warning("Plausibility (%s, deanon) failed — keeping enriched output", judge)
            status["plausibility"] = "failed"
            status["facts_post_plausibility"] = list(deanon_facts)
        else:
            pre_count = len(deanon_facts)
            post_count = len(plausible_deanon)
            status["plausibility_dropped"] = pre_count - post_count
            # Distinguish "judge kept valid facts" from "judge dropped everything"
            # — both show as `plausibility == "ok"` without this flag.
            if post_count == 0 and pre_count > 0:
                status["plausibility"] = "ok_but_empty"
            else:
                status["plausibility"] = "ok"
            deanon_facts = plausible_deanon
            status["facts_post_plausibility"] = list(deanon_facts)

    # "off" mode — explicit skip. Applies regardless of stage. Set here so the
    # status is not the ambiguous "skipped" (which means "no plausibility branch
    # matched"); "off" means "user asked us not to run plausibility".
    if plausibility_judge == "off":
        status["plausibility"] = "off"

    # Stage 4: build SessionGraph + deterministic symmetric-predicate dedup
    from paramem.graph.extractor import _canonicalize_symmetric_predicates

    kept = []
    for f in deanon_facts:
        try:
            kept.append(Relation(**f))
        except Exception:
            continue
    kept = _canonicalize_symmetric_predicates(kept)
    kept_names = {n for r in kept for n in (r.subject, r.object)}
    kept_entities = [e for e in extracted.entities if e.name in kept_names]

    # Safety net: if the pipeline dropped everything (typically because the
    # local anonymizer emitted placeholders in facts that weren't in its own
    # returned mapping → de-anonymization failed → residual sweep dropped all)
    # but raw extraction had content, fall back to raw plausibility. A PA
    # session should not yield zero facts just because the anonymizer was
    # inconsistent.
    if not kept and extracted.relations:
        logger.warning(
            "Pipeline dropped all facts; falling back to raw plausibility on extraction."
        )
        return _fallback_plausibility_on_raw("all_dropped")

    return (
        extracted.model_copy(update={"relations": kept, "entities": kept_entities}),
        status,
    )


def _run_plausibility_judge(
    judge: str,
    facts: list[dict],
    transcript: str,
    current_local_model,
    current_local_tokenizer,
    expected_extractor: str | None = None,
) -> list[dict] | None:
    """Dispatch the plausibility filter to whichever model the user picked.

    Invoked from both the de-anonymized branch of `_enrich_uniformly` and
    from `_fallback_plausibility_on_raw` (operates on real names).
    Dispatches by judge type:
    - SOTA judge → `plausibility_filter_sota` (sends the given data to the
      registered cloud provider). Privacy enforcement is the CLI layer's job.
    - Local judge → `plausibility_filter_local` on the currently-loaded model.

    Local judges that don't match the currently-loaded extractor model are
    silently downgraded to the loaded model (logged) — loading a second
    local model mid-pass is out of scope for this script today.
    """
    if judge in VALIDATORS:
        # Drop the raw response — callers of this helper don't persist it.
        # The anon-stage path calls plausibility_filter_sota directly and
        # captures the raw response there.
        facts_out, _raw = plausibility_filter_sota(facts, transcript, judge)
        return facts_out
    # Local judge — only the currently-loaded model is supported.
    if current_local_model is None or current_local_tokenizer is None:
        logger.warning(
            "Plausibility judge %r requires a loaded local model that isn't available in this pass",
            judge,
        )
        return None
    if judge != "auto" and expected_extractor is not None and judge != expected_extractor:
        logger.warning(
            "Plausibility judge %r requested but only %r is loaded for this pass; "
            "using %r instead. Cross-extractor judging would require loading a "
            "second local model, which is not supported here.",
            judge,
            expected_extractor,
            expected_extractor,
        )
    return plausibility_filter_local(
        facts, transcript, current_local_model, current_local_tokenizer
    )


def _speaker_from_session_id(sid: str) -> str | None:
    """Derive speaker name from a PerLTQA session_id.

    `perltqa_Ye Jie_40_7_0#21`       → 'Ye Jie' (space-separated).
    `perltqa_Ye_Jie_40_7_0#21`       → 'Ye Jie' (underscore-separated).
    `perltqa_John_2_40_7_0#21`       → 'John 2' (numeric component in name).

    Anchors on the PerLTQA session-counter tail `_<int>_<int>_<int>(?:#<int>)?`
    so numeric components inside the speaker's own name (rare but possible)
    are preserved. Returns None for non-PerLTQA ids, leaving the extractor's
    generic Speaker handling in place.
    """
    m = re.match(r"^perltqa_(.+?)_\d+_\d+_\d+(?:#\d+)?$", sid or "")
    if not m:
        return None
    return m.group(1).replace("_", " ").strip()


def _generate_raw(
    extractor: str,
    transcript: str,
    temperature: float,
    local_model=None,
    local_tokenizer=None,
    speaker_name: str | None = None,
) -> str:
    info = list_extractors()[extractor]
    if info["type"] == "cloud":
        return _generate_with_cloud(info, transcript, temperature, speaker_name=speaker_name)
    if local_model is None or local_tokenizer is None:
        raise RuntimeError(f"Extractor {extractor!r} requires a loaded local model")
    return generate_with_local(
        transcript,
        local_model,
        local_tokenizer,
        temperature=temperature,
        speaker_name=speaker_name,
    )


def run_pass(
    extractor: str,
    transcripts: list[dict],
    temperature: float,
    validator: str,
    plausibility_judge: str = "auto",
    plausibility_stage: str = "deanon",
    verify_anonymization: bool = True,
    ner_check: bool = False,
    ner_model: str = "en_core_web_sm",
) -> dict:
    """Extract + self-anonymize + SOTA-enrich + plausibility + de-anonymize, per session."""
    info = list_extractors()[extractor]
    model = tokenizer = None
    if info["type"] == "local":
        from paramem.models.loader import load_base_model
        from paramem.server.config import MODEL_REGISTRY

        print(f"  Loading {extractor}...")
        model, tokenizer = load_base_model(MODEL_REGISTRY[extractor])
        print(f"  {extractor} loaded\n")

    results = {}
    try:
        for i, t in enumerate(transcripts):
            sid = t["session_id"]
            preview = t["transcript"].split("\n")[0][:80]
            print(f"  [{extractor}] Session {i + 1}/{len(transcripts)}: {sid[:20]}")
            print(f"    {preview}")
            # PerLTQA filenames denote the memory OWNER, not necessarily the
            # first-person speaker of the transcript. Injecting a filename-
            # derived name as the mandatory subject confuses the extractor
            # on sessions narrating third-person context. Keep the harness
            # speaker-neutral; production gets the real speaker from
            # `speaker_store.get_name(speaker_id)` via the server path.
            raw = _generate_raw(
                extractor,
                t["transcript"],
                temperature,
                model,
                tokenizer,
                speaker_name=None,
            )
            extracted = run_extraction(raw, t["transcript"], sid)
            print(f"    -> {len(extracted.relations)} relations (extracted)")
            print_relations(extracted)
            # Serialize BEFORE enrichment — some enrichment paths mutate in place.
            extracted_serialized = serialize_relations(extracted)
            enriched = extracted
            pipeline_status = {
                "anonymize": "skipped",
                "enrich": "skipped",
                "plausibility": "skipped",
                "grounding_gate": "skipped",
                "plausibility_judge": plausibility_judge,
                "plausibility_stage": plausibility_stage,
                "anonymization_verified": verify_anonymization,
            }
            if extracted.relations:
                enriched, pipeline_status = _enrich_uniformly(
                    extracted,
                    t["transcript"],
                    extractor,
                    model,
                    tokenizer,
                    validator,
                    plausibility_judge=plausibility_judge,
                    plausibility_stage=plausibility_stage,
                    verify_anonymization=verify_anonymization,
                    ner_check=ner_check,
                    ner_model=ner_model,
                )
                print(
                    f"    -> {len(enriched.relations)} relations (enriched) "
                    f"[anon={pipeline_status['anonymize']} "
                    f"enr={pipeline_status['enrich']} "
                    f"plaus={pipeline_status['plausibility']}]"
                )
            results[sid] = {
                "raw_output": raw,
                "relations_extracted": extracted_serialized,
                "relations_enriched": serialize_relations(enriched),
                "pipeline_status": pipeline_status,
            }
    finally:
        if model is not None:
            print(f"\n  Unloading {extractor}...")
            unload_model(model, tokenizer)
            print(f"  {extractor} unloaded")
    return results


def print_summary(transcripts: list[dict], results_by_model: dict[str, dict]):
    """Print comparison summary: extracted + enriched counts per model.

    - extracted: raw facts the model produced directly.
    - enriched: facts remaining after the uniform anonymize → validate →
      de-anonymize pipeline. The anonymizer and validator are held constant
      across all extractors in a sweep, so the only variable driving
      differences is extraction quality. Enrichment performs coreference
      resolution, compound-fact splitting, and noise removal, so the
      enriched count may exceed the extracted count.
    """
    models = list(results_by_model.keys())

    # Detect legacy schema (pre-rename used relations_raw / relations_validated / relations).
    # Mixing old and new shapes silently produces zero columns — warn loudly instead.
    def _has_legacy_schema(pass_data: dict) -> bool:
        for sess in pass_data.values():
            if not isinstance(sess, dict):
                continue
            keys = sess.keys()
            if "relations_extracted" not in keys and (
                "relations_raw" in keys or "relations" in keys
            ):
                return True
        return False

    legacy = [m for m in models if _has_legacy_schema(results_by_model[m])]
    if legacy:
        print(
            f"\nWARNING: legacy-schema passes detected for {legacy} — counts "
            f"will be 0 for these. Delete or `--rerun` them to refresh."
        )

    header_cols = ["Session"] + [f"{m}_ext" for m in models] + [f"{m}_enr" for m in models]
    widths = [25] + [10] * (len(header_cols) - 1)
    fmt = "  " + "".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(widths))
    total_width = sum(widths) + 2
    print("\n" + "=" * total_width)
    print("SUMMARY  (ext=extracted, enr=after SOTA enrich + plausibility filter)")
    print(fmt.format(*header_cols))
    print("-" * total_width)
    totals_ext = {m: 0 for m in models}
    totals_enr = {m: 0 for m in models}
    for t in transcripts:
        sid = t["session_id"]
        row_label = sid[:22]
        ext = [len(results_by_model[m].get(sid, {}).get("relations_extracted", [])) for m in models]
        enr = [len(results_by_model[m].get(sid, {}).get("relations_enriched", [])) for m in models]
        for m, e, n in zip(models, ext, enr):
            totals_ext[m] += e
            totals_enr[m] += n
        print(fmt.format(row_label, *ext, *enr))
    print("-" * total_width)
    print(fmt.format("TOTAL", *totals_ext.values(), *totals_enr.values()))

    # Stage-failure tally — surfaces silent fallbacks in the SOTA pipeline.
    failures = {m: {"anonymize": 0, "enrich": 0, "plausibility": 0} for m in models}
    for m in models:
        for sess in results_by_model[m].values():
            status = sess.get("pipeline_status") if isinstance(sess, dict) else None
            if not status:
                continue
            for stage in failures[m]:
                if status.get(stage) == "failed":
                    failures[m][stage] += 1
    if any(any(v.values()) for v in failures.values()):
        print("\nSTAGE FAILURES (silent fallbacks in the SOTA pipeline):")
        for m, counts in failures.items():
            if any(counts.values()):
                print(
                    f"  {m}: anon={counts['anonymize']} "
                    f"enr={counts['enrich']} plaus={counts['plausibility']}"
                )


def main():
    global PROMPT_DIR, OUTPUT_DIR, MAX_TOKENS, VALIDATOR_TEMPERATURE

    all_extractors = sorted(list_extractors())

    parser = argparse.ArgumentParser(
        description="Compare extraction quality across models. Every extractor "
        "is treated identically: extract → self-anonymize → cloud validate → "
        "de-anonymize. The extractor performs its own anonymization — for a "
        "local extractor that means on-device (raw data never leaves); for a "
        "cloud extractor it means re-using the same cloud provider that "
        "already saw the raw data at extraction time. The validator only "
        "ever receives anonymized data. Extractors and validators come from "
        "registries (EXTRACTORS_CLOUD, VALIDATORS, paramem.server.config."
        "MODEL_REGISTRY) — add entries there, no pipeline code changes."
    )
    parser.add_argument(
        "--rerun",
        default=None,
        help="Force re-run a specific extractor at the current config "
        "(leaves saved passes at other temperatures/configs untouched).",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to a single transcript (.jsonl) or a folder of transcripts "
        "(no subdirectory recursion). Default: all sessions under "
        f"{DEFAULT_SESSION_DIR}/ (+ archive).",
    )
    parser.add_argument(
        "--models",
        default=",".join(all_extractors),
        help=f"Comma-separated subset of {all_extractors} to run as extractors (default: all).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Extraction temperature. Default: 0.0.",
    )
    parser.add_argument(
        "--validator",
        default=sorted(VALIDATORS)[0],
        choices=sorted(VALIDATORS),
        help=f"SOTA provider for the noise-filter / enrichment pass "
        f"(default: {sorted(VALIDATORS)[0]}). The validator sees ONLY "
        f"anonymized data; anonymization is always performed by the "
        f"extractor itself (locally for local extractors, by the same "
        f"cloud provider for cloud extractors — no extra trust boundary).",
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=DEFAULT_PROMPT_DIR,
        help=f"Directory containing prompt files (default: {DEFAULT_PROMPT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for saved passes (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per generation call (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--validator-temperature",
        type=float,
        default=DEFAULT_VALIDATOR_TEMPERATURE,
        help=f"Temperature for the SOTA validator (enrichment + plausibility). "
        f"Default: {DEFAULT_VALIDATOR_TEMPERATURE}. Keep at 0.0 for reproducible "
        f"runs; raise only for deliberate validator-sampling studies.",
    )
    parser.add_argument(
        "--plausibility-judge",
        default=sorted(VALIDATORS)[0],
        help=f"Model role to judge plausibility. Default: {sorted(VALIDATORS)[0]} "
        f"(SOTA judge, reliable quality, operates on anonymized data — combined "
        f"with --plausibility-stage=anon this is privacy-safe). 'auto' = use the "
        f"extractor itself (requires --plausibility-stage=deanon; local judge "
        f"quality varies by model — see sweep results). 'off' skips the "
        f"plausibility step. Or any other registered validator name.",
    )
    parser.add_argument(
        "--plausibility-stage",
        default="anon",
        choices=["deanon", "anon"],
        help="Where in the pipeline plausibility runs. 'anon' (default): on "
        "placeholder facts before de-anonymization — SOTA judges only "
        "(privacy-safe). 'deanon': on real-name facts after de-anonymization — "
        "local judges only, can detect leaked placeholders but SOTA judges here "
        "would send real names to the cloud.",
    )
    parser.add_argument(
        "--verify-anonymization",
        dest="verify_anonymization",
        action="store_true",
        default=True,
        help="Forward-path privacy guard (default ON): verify no real name "
        "leaked past anonymization before sending to SOTA; abort on leak.",
    )
    parser.add_argument(
        "--no-verify-anonymization",
        dest="verify_anonymization",
        action="store_false",
        help="DANGER: disable the forward-path privacy guard. Allows the "
        "SOTA call to proceed even when real names leaked past anonymization. "
        "For leak-investigation tests only — never in production.",
    )
    parser.add_argument(
        "--ner-check",
        action="store_true",
        default=False,
        help="Belt-and-braces PII detection via spaCy NER. When enabled, run "
        "NER on the raw transcript and add its person/place findings to the "
        "guard's PII set. Catches cases where the extractor mislabeled or "
        "missed a PII entity. Requires spaCy + model installed (default: off).",
    )
    parser.add_argument(
        "--ner-model",
        default="en_core_web_sm",
        help="spaCy model name for --ner-check. For multilingual coverage "
        "use e.g. de_core_news_sm (German) or xx_sent_ud_sm (multi).",
    )
    args = parser.parse_args()

    # Apply overridable defaults to module globals so downstream helpers see them.
    PROMPT_DIR = args.prompt_dir
    OUTPUT_DIR = args.output_dir
    MAX_TOKENS = args.max_tokens
    VALIDATOR_TEMPERATURE = args.validator_temperature

    extractors = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in extractors if m not in all_extractors]
    if unknown:
        print(f"ERROR: unknown model(s): {unknown}. Choose from {all_extractors}.")
        sys.exit(1)

    # Validate plausibility judge against {auto, off, extractor names, validator names}.
    valid_judges = {"auto", "off"} | set(all_extractors) | set(VALIDATORS)
    if args.plausibility_judge not in valid_judges:
        print(
            f"ERROR: --plausibility-judge {args.plausibility_judge!r} not recognised. "
            f"Choose from: {sorted(valid_judges)}"
        )
        sys.exit(1)
    # Advisory: SOTA judges on de-anonymized data send real names to the cloud.
    # For most configurations this is undesirable (new trust boundary crossed).
    # For self-judgement of a cloud extractor (extractor and judge are the
    # same cloud provider) it's the same trust boundary already crossed at
    # extraction time, so it's informational. We warn and proceed — the user
    # configured it deliberately.
    if args.plausibility_judge in VALIDATORS and args.plausibility_stage != "anon":
        print(
            f"WARNING: SOTA judge {args.plausibility_judge!r} on --plausibility-stage "
            f"{args.plausibility_stage!r} sends real names to the cloud. Proceed only "
            f"if the extractor shares the same trust boundary."
        )

    # Guard against silent no-op combinations. Plausibility runs in exactly one
    # of two branches:
    #   anon-stage  → requires judge ∈ VALIDATORS (SOTA only)
    #   deanon-stage → requires judge ∈ {"auto"} ∪ extractors (local only)
    # Any other combo would silently skip plausibility with status="skipped".
    if args.plausibility_judge != "off":
        judge_is_sota = args.plausibility_judge in VALIDATORS
        judge_is_local = args.plausibility_judge == "auto" or (
            args.plausibility_judge in all_extractors and args.plausibility_judge not in VALIDATORS
        )
        if args.plausibility_stage == "anon" and not judge_is_sota:
            print(
                f"ERROR: --plausibility-stage anon requires a SOTA judge (one of "
                f"{sorted(VALIDATORS)}). Got {args.plausibility_judge!r}. "
                f"Use --plausibility-stage deanon for local judges, or --plausibility-judge off."
            )
            sys.exit(1)
        if args.plausibility_stage == "deanon" and not judge_is_local:
            print(
                f"ERROR: --plausibility-stage deanon requires a local judge "
                f"('auto' or an extractor name: {sorted(set(all_extractors) - set(VALIDATORS))}). "
                f"Got {args.plausibility_judge!r}. Use --plausibility-stage anon for SOTA judges."
            )
            sys.exit(1)

    # Lazy API-key validation — only require keys for providers actually used
    # by this run (selected cloud extractors + the chosen validator + plausibility judge).
    needed_keys: set[str] = set()
    ex_info = list_extractors()
    for name in extractors:
        info = ex_info[name]
        if info["type"] == "cloud":
            needed_keys.add(info["key_env"])
    needed_keys.add(VALIDATORS[args.validator]["key_env"])
    if args.plausibility_judge in VALIDATORS:
        needed_keys.add(VALIDATORS[args.plausibility_judge]["key_env"])
    missing = [k for k in sorted(needed_keys) if not os.environ.get(k)]
    if missing:
        print(f"ERROR: required env var(s) not set: {missing}")
        sys.exit(1)

    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

    transcripts = load_transcripts(args.path)
    print(
        f"Processing {len(transcripts)} sessions | temp={args.temperature} | "
        f"validator={args.validator} | plausibility={args.plausibility_judge}/"
        f"{args.plausibility_stage} (anonymization: self by each extractor)\n"
    )
    if not transcripts:
        print("No sessions found")
        sys.exit(0)

    verify_tag = "on" if args.verify_anonymization else "off"
    cfg_tag = (
        f"t{args.temperature:.2f}_v{args.validator}_vt{VALIDATOR_TEMPERATURE:.2f}"
        f"_pj{args.plausibility_judge}_ps{args.plausibility_stage}_va{verify_tag}"
    )
    if not args.verify_anonymization:
        print(
            "WARNING: --no-verify-anonymization is set. Forward-path privacy "
            "guard is OFF. Real names may leak to the SOTA validator. This is "
            "for leak-investigation tests only."
        )
    expected_ids = {t["session_id"] for t in transcripts}
    results: dict[str, dict] = {}
    for idx, extractor in enumerate(extractors, 1):
        cached = (
            load_pass(
                extractor,
                args.temperature,
                args.validator,
                args.plausibility_judge,
                args.plausibility_stage,
                args.verify_anonymization,
                expected_session_ids=expected_ids,
            )
            if args.rerun != extractor
            else None
        )
        if cached is not None:
            print(f"PASS {idx}: {extractor} @ {cfg_tag} — loaded from previous run")
            results[extractor] = cached
            continue
        print("\n" + "=" * 70)
        print(f"PASS {idx}: {extractor} @ {cfg_tag}")
        print("=" * 70)
        results[extractor] = run_pass(
            extractor,
            transcripts,
            args.temperature,
            args.validator,
            plausibility_judge=args.plausibility_judge,
            plausibility_stage=args.plausibility_stage,
            verify_anonymization=args.verify_anonymization,
            ner_check=args.ner_check,
            ner_model=args.ner_model,
        )
        save_pass(
            extractor,
            args.temperature,
            args.validator,
            args.plausibility_judge,
            args.plausibility_stage,
            args.verify_anonymization,
            results[extractor],
        )

    print_summary(transcripts, results)


if __name__ == "__main__":
    main()
