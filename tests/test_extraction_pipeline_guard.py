"""Structural guard: forbid direct extract_graph / extract_procedural_graph calls.

The extraction pipeline has exactly one callable topology. All orchestrators
(server, experiments, tests) must reach the extractors through
:class:`paramem.graph.extraction_pipeline.ExtractionPipeline` (the
:meth:`run` / :meth:`run_procedural` chokepoints), never via direct calls
to ``extract_graph(...)`` / ``extract_procedural_graph(...)``.

This test scans the tracked codebase and fails if a new call site appears
outside the whitelist. Pair with the parity tests in test_consolidation.py.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paramem.memory.store import MemoryStore as _MS

# Files allowed to call the extractors directly:
# - extractor.py: the module defining them.
# - extraction_pipeline.py: the chokepoint module that wraps them.
# - tests/: may patch, import, or assert on their names.
# - archive/: historical, not executed.
_ALLOWED_SUFFIXES = (
    "paramem/graph/extractor.py",
    "paramem/graph/extraction_pipeline.py",
)

_ALLOWED_PREFIXES = (
    "tests/",
    "archive/",
)

# Grandfathered standalone experiment probes. These pre-date the
# ExtractionPipeline chokepoint and run outside the production orchestrator
# path (one-shot GPU evals, not a live extract→train loop). New experiment
# code MUST route through ExtractionPipeline instead. If one of these is
# rewritten to use the chokepoint, remove it from this set.
_GRANDFATHERED_FILES = frozenset(
    {
        "experiments/chained_adapter_smoke.py",
        "experiments/test8_large_scale.py",
        "experiments/test_early_stopping.py",
        "experiments/utils/perltqa_loader.py",
        "experiments/utils/test_harness.py",
    }
)

_FORBIDDEN_CALL_NAMES = frozenset({"extract_graph", "extract_procedural_graph"})


def _is_allowed(rel: str) -> bool:
    if rel in _GRANDFATHERED_FILES:
        return True
    if rel.endswith(_ALLOWED_SUFFIXES):
        return True
    return any(rel.startswith(p) for p in _ALLOWED_PREFIXES)


def _find_forbidden_call_sites(py_file: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, source)`` for every direct call to a forbidden
    extractor name in ``py_file``.

    Uses :mod:`ast` so the guard counts actual call sites, not strings
    that happen to contain ``extract_graph(`` (docstrings, comments,
    function definitions, log messages, …).  An ``ast.Call`` whose
    ``func`` is an ``ast.Name`` matching the whitelist is the canonical
    "direct call" we forbid.  Attribute calls like
    ``self.extraction.run(...)`` are different syntax and not caught
    here — by design, they are the routed-through pattern we encourage.
    """
    import ast

    try:
        text = py_file.read_text()
    except UnicodeDecodeError:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FORBIDDEN_CALL_NAMES
        ):
            line = lines[node.lineno - 1] if 0 < node.lineno <= len(lines) else ""
            out.append((node.lineno, line.strip()))
    return out


def test_no_direct_extract_graph_calls_outside_whitelist():
    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[tuple[str, int, str]] = []

    for py_file in repo_root.rglob("*.py"):
        rel = py_file.relative_to(repo_root).as_posix()
        if _is_allowed(rel):
            continue
        for lineno, src in _find_forbidden_call_sites(py_file):
            offenders.append((rel, lineno, src))

    assert not offenders, (
        "Direct extract_graph/extract_procedural_graph calls found outside the "
        "allowed orchestrator. Route new callers through "
        "ExtractionPipeline.run / ExtractionPipeline.run_procedural so "
        "extraction behavior cannot diverge between production and tests:\n"
        + "\n".join(f"  {path}:{line} — {src}" for path, line, src in offenders)
    )


# Positional params the chokepoint supplies itself — not user-facing kwargs.
_EXTRACTOR_POSITIONAL = {"model", "tokenizer", "transcript", "session_id"}


def _make_pipeline(model=None, tokenizer=None, **config_overrides):
    """Build a stand-in :class:`ExtractionPipeline` for chokepoint contract tests.

    Mirrors the attribute shape the contract tests need without a real
    model load.  The pipeline's :meth:`kwargs` and :meth:`run` /
    :meth:`run_procedural` methods are exercised against ``MagicMock``
    stand-ins for the extractor functions via :func:`monkeypatch`.
    """
    from paramem.graph.extraction_pipeline import ExtractionConfig, ExtractionPipeline

    return ExtractionPipeline(
        model=model if model is not None else MagicMock(),
        tokenizer=tokenizer if tokenizer is not None else MagicMock(),
        config=ExtractionConfig(**config_overrides),
        prompts_dir=None,
    )


@pytest.mark.parametrize("source_type", ["transcript", "document"])
def test_extraction_pipeline_kwargs_match_extract_graph_signature(source_type):
    """Every kwarg the chokepoint passes must exist on extract_graph.

    Catches rename drift: if the chokepoint grows a new flag but the extractor
    hasn't caught up (or vice versa), production crashes with TypeError.
    This test fails first.  Parametrized over both source_type values so
    both the transcript and document kwarg sets are validated.
    """
    from paramem.graph.extractor import extract_graph

    pipeline = _make_pipeline()
    pipeline_keys = set(pipeline.kwargs(source_type=source_type, speaker_id="Speaker0").keys())
    extractor_params = set(inspect.signature(extract_graph).parameters) - _EXTRACTOR_POSITIONAL

    unknown = pipeline_keys - extractor_params
    assert not unknown, (
        f"ExtractionPipeline.kwargs(source_type={source_type!r}) passes kwargs "
        f"extract_graph does not accept: {sorted(unknown)}. "
        "Either rename the chokepoint key to match the extractor signature, or "
        "add the parameter to extract_graph."
    )


def test_kwargs_emits_model_alias():
    """ExtractionPipeline.kwargs() must emit ``model_alias`` tied to self.model_name.

    Regression guard: if the ``model_alias`` key is accidentally dropped from
    kwargs(), per-model prompt resolution silently falls back to the shared
    default for every model — new model-specific prompt files become dead
    config that is never loaded.
    """
    from paramem.graph.extraction_pipeline import ExtractionConfig, ExtractionPipeline

    pipeline = ExtractionPipeline(
        model=MagicMock(),
        tokenizer=MagicMock(),
        config=ExtractionConfig(),
        model_name="qwen3-4b",
    )
    kw = pipeline.kwargs(source_type="transcript", speaker_id="Speaker0")
    assert "model_alias" in kw, "kwargs() must emit 'model_alias'"
    assert kw["model_alias"] == "qwen3-4b"

    # model_name=None → model_alias=None (base resolution, unchanged)
    pipeline_base = ExtractionPipeline(
        model=MagicMock(),
        tokenizer=MagicMock(),
        config=ExtractionConfig(),
    )
    kw_base = pipeline_base.kwargs(source_type="transcript", speaker_id="Speaker0")
    assert kw_base["model_alias"] is None


def test_load_extraction_prompts_per_model_resolution(tmp_path):
    """load_extraction_prompts(..., model='qwen3-4b') picks the per-model file.

    Guards the path from ExtractionPipeline.kwargs model_alias → extract_graph
    → _generate_extraction → load_extraction_prompts → _load_prompt.
    """
    from paramem.graph.extractor import load_extraction_prompts

    (tmp_path / "qwen3-4b").mkdir()
    (tmp_path / "qwen3-4b" / "extraction.txt").write_text("qwen-body")
    (tmp_path / "extraction.txt").write_text("base-body")
    (tmp_path / "extraction_system.txt").write_text("shared-system")

    system, prompt = load_extraction_prompts(tmp_path, model="qwen3-4b")
    assert prompt == "qwen-body", "Per-model extraction.txt must be selected"
    assert system == "shared-system", "Shared extraction_system.txt must be inherited"


def test_procedural_kwargs_match_extract_procedural_graph_signature():
    """Same contract for the procedural chokepoint — the inline kwargs dict
    inside :meth:`ExtractionPipeline.run_procedural` must line up with the
    ``extract_procedural_graph`` signature."""
    from paramem.graph.extractor import extract_procedural_graph

    # Mirror the inline kwargs shape inside ExtractionPipeline.run_procedural.
    # If that method ever diverges, update this set alongside.
    # speaker_id is passed as a keyword arg from call_kwargs (even though it is
    # a required positional-or-keyword param in the extractor signature).
    procedural_keys = {
        "max_tokens",
        "prompts_dir",
        "model_alias",
        "speaker_name",
        "speaker_id",
        "system_prompt_filename",
        "user_prompt_filename",
    }
    extractor_params = (
        set(inspect.signature(extract_procedural_graph).parameters) - _EXTRACTOR_POSITIONAL
    )

    unknown = procedural_keys - extractor_params
    assert not unknown, (
        f"ExtractionPipeline.run_procedural passes kwargs extract_procedural_graph "
        f"does not accept: {sorted(unknown)}"
    )


def _peft_model_mock() -> MagicMock:
    """Return a ``MagicMock(spec=PeftModel)`` with the chokepoint's required
    methods explicitly attached.

    ``spec=PeftModel`` enables the ``isinstance(model, PeftModel)`` branch
    inside :meth:`ExtractionPipeline.run` while the explicit assignments
    expose the attributes the chokepoint actually calls
    (``disable_adapter`` for the adapter guard,
    ``gradient_checkpointing_disable`` for the KV-cache fix).  Without
    these assignments, ``MagicMock(spec=...)`` raises ``AttributeError`` on
    calls that aren't directly on PeftModel's own surface.
    """
    from peft import PeftModel

    m = MagicMock(spec=PeftModel)
    m.disable_adapter = MagicMock()
    m.gradient_checkpointing_disable = MagicMock()
    return m


def test_run_wraps_peft_model_in_disable_adapter(monkeypatch):
    """Gap B: when the model is a PeftModel, :meth:`ExtractionPipeline.run`
    MUST enter the ``disable_adapter()`` context manager before calling
    extract_graph.

    Regression guard: if the guard is ever refactored away, extraction
    silently runs with the training-active adapter, contaminating output.
    """
    fake_peft = _peft_model_mock()
    pipeline = _make_pipeline(model=fake_peft)

    monkeypatch.setattr(
        "paramem.graph.extraction_pipeline.extract_graph",
        lambda *a, **kw: MagicMock(),
    )
    pipeline.run("transcript", "s001", speaker_id="Speaker0")

    fake_peft.disable_adapter.assert_called_once()


def test_run_skips_disable_adapter_for_plain_model(monkeypatch):
    """Gap B (negative): plain (non-PeftModel) models must NOT be wrapped."""
    plain_model = MagicMock()  # no spec — fails isinstance(_, PeftModel)
    pipeline = _make_pipeline(model=plain_model)

    monkeypatch.setattr(
        "paramem.graph.extraction_pipeline.extract_graph",
        lambda *a, **kw: MagicMock(),
    )
    pipeline.run("transcript", "s001", speaker_id="Speaker0")

    plain_model.disable_adapter.assert_not_called()


def test_run_threads_positional_args(monkeypatch):
    """Gap C: :meth:`ExtractionPipeline.run` must pass
    ``(model, tokenizer, transcript, session_id)`` to the real extractor
    unchanged. If the chokepoint ever reshapes the call, this test fails
    before production notices.
    """
    captured = {}

    def spy(model, tokenizer, transcript, session_id, **kwargs):
        captured["model"] = model
        captured["tokenizer"] = tokenizer
        captured["transcript"] = transcript
        captured["session_id"] = session_id
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr("paramem.graph.extraction_pipeline.extract_graph", spy)

    model_sentinel = MagicMock(name="model_sentinel")
    tokenizer_sentinel = MagicMock(name="tokenizer_sentinel")
    pipeline = _make_pipeline(
        model=model_sentinel,
        tokenizer=tokenizer_sentinel,
        noise_filter="claude",
    )
    pipeline.prompts_dir = "/custom/prompts"

    pipeline.run("alex lives here", "s042", speaker_id="Speaker0")

    assert captured["model"] is model_sentinel
    assert captured["tokenizer"] is tokenizer_sentinel
    assert captured["transcript"] == "alex lives here"
    assert captured["session_id"] == "s042"
    assert captured["kwargs"]["prompts_dir"] == "/custom/prompts"
    assert captured["kwargs"]["noise_filter"] == "claude"


def _collect_extract_session_kwargs(py_file: Path) -> list[set[str]]:
    """Return the kwarg name-sets for every `*.extract_session(...)` call in py_file."""
    import ast

    tree = ast.parse(py_file.read_text())
    out: list[set[str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "extract_session"
        ):
            out.append({kw.arg for kw in node.keywords if kw.arg is not None})
    return out


def _collect_extract_session_callsites(
    py_file: Path,
) -> list[tuple[int, set[str]]]:
    """Return ``(lineno, kwarg_name_set)`` for every ``*.extract_session(...)``
    call in ``py_file``."""
    import ast

    tree = ast.parse(py_file.read_text())
    out: list[tuple[int, set[str]]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "extract_session"
        ):
            out.append((node.lineno, {kw.arg for kw in node.keywords if kw.arg is not None}))
    return out


def test_server_extract_session_callsites_pass_identical_kwargs():
    """Gap D: every `*.extract_session(...)` call site under `paramem/server/`
    must pass the same kwarg names as every other server caller.

    Silently dropping flags on one path (as app.py:1276 did before this test
    landed) means the same consolidation operation runs with different SOTA
    pipeline behavior depending on which code path triggered it.

    Scans the whole `paramem/server/` tree at call-site granularity.  Both
    consolidation call sites live in ``app.py``; per-file deduplication
    would hide a divergence between them.  This version tracks each site
    independently so parity is enforced regardless of file distribution.
    """
    repo_root = Path(__file__).resolve().parent.parent
    server_dir = repo_root / "paramem" / "server"

    # key: "<rel_path>:<lineno>", value: kwarg name set
    all_sites: dict[str, set[str]] = {}
    for py_file in server_dir.rglob("*.py"):
        for lineno, kwset in _collect_extract_session_callsites(py_file):
            label = f"{py_file.relative_to(repo_root).as_posix()}:{lineno}"
            all_sites[label] = kwset

    assert len(all_sites) >= 2, (
        "expected at least two server-layer extract_session call sites; found: "
        f"{sorted(all_sites)}. If extract_session was refactored out of the server, "
        "update or remove this test.  Both orchestration paths in app.py "
        "(_extract_and_start_training and _run_extraction_phase) must be present."
    )

    reference_label, reference_kwargs = sorted(all_sites.items())[0]
    divergent: dict[str, tuple[set[str], set[str]]] = {}
    for label, kwargs in all_sites.items():
        if label == reference_label:
            continue
        missing_here = reference_kwargs - kwargs
        extra_here = kwargs - reference_kwargs
        if missing_here or extra_here:
            divergent[label] = (missing_here, extra_here)

    assert not divergent, (
        "extract_session kwargs diverge across server-layer call sites — same "
        f"operation, different flags. Reference: {reference_label}\n"
        + "\n".join(
            f"  {label}: missing={sorted(miss)} extra={sorted(extra)}"
            for label, (miss, extra) in sorted(divergent.items())
        )
    )


def test_server_extract_session_callsites_pass_event_time():
    """Every `paramem/server/` extract_session(...) call site must pass
    `event_time` explicitly.

    `extract_session`'s `event_time` param defaults to ``None`` (falls back
    to ``now()`` at the extractor layer) so the ~19 test/experiment call
    sites outside `paramem/server/` don't need updating. Production call
    sites MUST NOT rely on that fallback — a session's facts must carry the
    real session-start assertion time, not consolidation-time. This guard
    locks the production invariant without touching tests/experiments.
    """
    repo_root = Path(__file__).resolve().parent.parent
    server_dir = repo_root / "paramem" / "server"

    missing: list[str] = []
    for py_file in server_dir.rglob("*.py"):
        for lineno, kwset in _collect_extract_session_callsites(py_file):
            if "event_time" not in kwset:
                label = f"{py_file.relative_to(repo_root).as_posix()}:{lineno}"
                missing.append(label)

    assert not missing, (
        "extract_session call site(s) under paramem/server/ omit event_time — "
        "the session's real started_at time would silently fall back to "
        "now() at the extractor layer, recording consolidation-time instead "
        "of assertion-time on newly-merged edges:\n" + "\n".join(f"  {m}" for m in missing)
    )


def test_server_yaml_extraction_flags_round_trip(tmp_path):
    """Gap (b): values set in server.yaml must reach `config.consolidation.*`
    unchanged.

    Catches YAML-schema drift: if a field is renamed or dropped in
    ConsolidationScheduleConfig but the YAML still carries the old key, loading
    silently ignores it and the server runs on defaults. This test writes a
    minimal YAML with every `extraction_*` flag flipped away from its default,
    loads it, and asserts every value arrives where the server call sites read
    from.
    """
    from paramem.server.config import ConsolidationScheduleConfig, load_server_config

    flipped = {
        "extraction_max_tokens": 4096,
        "extraction_ha_validation": False,
        "extraction_noise_filter": "",
        "extraction_noise_filter_model": "claude-other",
        "extraction_noise_filter_endpoint": "http://custom:8080/v1",
        "extraction_plausibility_judge": "off",
        "extraction_plausibility_stage": "anon",
        "extraction_verify_anonymization": False,
        "extraction_ner_check": True,
        "extraction_ner_model": "xx_ent_wiki_sm",
    }

    defaults = ConsolidationScheduleConfig()
    for key, val in flipped.items():
        assert getattr(defaults, key) != val, (
            f"Test YAML uses default value for {key!r} — would not detect parsing "
            "drift. Pick a non-default value."
        )

    yaml_path = tmp_path / "server.yaml"
    body = "consolidation:\n" + "\n".join(f"  {k}: {_yaml_literal(v)}" for k, v in flipped.items())
    yaml_path.write_text(body)

    loaded = load_server_config(yaml_path)
    for key, expected in flipped.items():
        actual = getattr(loaded.consolidation, key)
        assert actual == expected, (
            f"server.yaml→ConsolidationScheduleConfig drift on {key}: "
            f"yaml={expected!r}, loaded={actual!r}"
        )


def _yaml_literal(v):
    """Render Python values as YAML scalars the parser will round-trip.

    Always quote strings — YAML 1.1 parses bare "off"/"on"/"no"/"yes" as
    booleans, which corrupts string fields like `extraction_plausibility_judge`.
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        escaped = v.replace('"', '\\"')
        return f'"{escaped}"'
    return repr(v)


def test_server_extract_session_kwargs_map_to_consolidation_config():
    """Gap (b) sibling: every kwarg at a server-layer `extract_session(...)` call
    must be backed by a `config.consolidation.extraction_<kwarg>` field.

    If a new SOTA flag is added at a server call site but the corresponding
    ConsolidationScheduleConfig field is missing, `config.consolidation...` on
    that line raises AttributeError the first time a session consolidates.
    This test catches it offline.
    """
    from paramem.server.config import ConsolidationScheduleConfig

    repo_root = Path(__file__).resolve().parent.parent
    server_dir = repo_root / "paramem" / "server"

    known_non_extraction_kwargs = {
        "speaker_id",
        "speaker_name",
        "ha_context",
        # Session-metadata-derived; no extraction_<kwarg> field in
        # ConsolidationScheduleConfig because it is resolved per call,
        # not stored as a loop-level default.
        "source_type",
        # Session-start assertion time (session["started_at"]); same
        # per-call, not-a-loop-default reasoning as source_type.
        "event_time",
    }
    config_fields = {f for f in vars(ConsolidationScheduleConfig()).keys()}

    orphans: list[tuple[str, str]] = []
    for py_file in server_dir.rglob("*.py"):
        for kwset in _collect_extract_session_kwargs(py_file):
            for kw in kwset:
                if kw in known_non_extraction_kwargs:
                    continue
                if f"extraction_{kw}" not in config_fields:
                    orphans.append((py_file.relative_to(repo_root).as_posix(), kw))

    assert not orphans, (
        "extract_session kwargs at server call sites with no matching "
        "ConsolidationScheduleConfig field — will AttributeError at runtime:\n"
        + "\n".join(f"  {path}: {kw} (expected field: extraction_{kw})" for path, kw in orphans)
    )


def test_run_procedural_wraps_peft_model_in_disable_adapter(monkeypatch):
    """Procedural mirror of Gap B: PeftModel must enter `disable_adapter()`
    before the procedural extractor is called.

    Without this guard, procedural extraction runs with the training-active
    adapter, contaminating preference capture.
    """
    fake_peft = _peft_model_mock()
    pipeline = _make_pipeline(model=fake_peft)

    monkeypatch.setattr(
        "paramem.graph.extraction_pipeline.extract_procedural_graph",
        lambda *a, **kw: MagicMock(),
    )
    pipeline.run_procedural("transcript", "s001", speaker_id="Speaker0")

    fake_peft.disable_adapter.assert_called_once()


def test_run_procedural_skips_disable_adapter_for_plain_model(monkeypatch):
    """Procedural negative case: plain models must NOT be wrapped."""
    plain_model = MagicMock()
    pipeline = _make_pipeline(model=plain_model)

    monkeypatch.setattr(
        "paramem.graph.extraction_pipeline.extract_procedural_graph",
        lambda *a, **kw: MagicMock(),
    )
    pipeline.run_procedural("transcript", "s001", speaker_id="Speaker0")

    plain_model.disable_adapter.assert_not_called()


def test_consolidation_loop_constructor_threads_extraction_flags(tmp_path):
    """Experiment-path mirror of (b): kwargs forwarded to
    ``ConsolidationLoop.__init__`` must land on
    ``loop.extraction.config.<field>`` unchanged.

    Experiment entrypoints configure the pipeline by passing ``extraction_*``
    kwargs to the loop constructor.  If the constructor ever silently drops or
    renames one of them, experiments run on defaults while tests that check
    ``ExtractionPipeline.kwargs`` output still pass — because those tests
    seed a config by hand. This closes that loop.
    """
    from peft import PeftModel

    from paramem.training.consolidation import ConsolidationLoop
    from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

    flipped = {
        "extraction_temperature": 0.7,
        "extraction_max_tokens": 4096,
        "extraction_ha_validation": False,
        "extraction_noise_filter": "claude",
        "extraction_noise_filter_model": "claude-other",
        "extraction_noise_filter_endpoint": "http://custom:8080/v1",
        "extraction_ner_check": True,
        "extraction_ner_model": "xx_ent_wiki_sm",
        "extraction_plausibility_judge": "off",
        "extraction_plausibility_stage": "anon",
        "extraction_verify_anonymization": False,
    }

    # Skip adapter wiring — we only care about flag storage on
    # loop.extraction.config.  __class__ = PeftModel so _ensure_adapters'
    # isinstance check short-circuits without restricting the mock's
    # attribute surface.
    model = MagicMock()
    model.__class__ = PeftModel
    model.peft_config = {
        "episodic": MagicMock(),
        "semantic": MagicMock(),
        "in_training": MagicMock(),
    }

    loop = ConsolidationLoop(
        model=model,
        tokenizer=MagicMock(),
        consolidation_config=ConsolidationConfig(),
        training_config=TrainingConfig(),
        episodic_adapter_config=AdapterConfig(),
        semantic_adapter_config=AdapterConfig(),
        memory_store=_MS(replay_enabled=False),
        output_dir=tmp_path,
        **flipped,
    )

    cfg = loop.extraction.config
    for ctor_key, expected in flipped.items():
        # Constructor key drops the ``extraction_`` prefix on the dataclass field.
        field = ctor_key.removeprefix("extraction_")
        actual = getattr(cfg, field)
        assert actual == expected, (
            f"ConsolidationLoop.__init__ dropped kwarg {ctor_key!r}: "
            f"passed={expected!r}, stored on extraction.config.{field}={actual!r}"
        )


def test_consolidation_loop_threads_model_name_to_extraction_pipeline(tmp_path):
    """ConsolidationLoop.__init__ must store model_name on loop.extraction.model_name.

    Regression guard: if the constructor drops this kwarg, per-model prompt
    resolution silently uses None for every production server cycle.
    """
    from peft import PeftModel

    from paramem.training.consolidation import ConsolidationLoop
    from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

    model = MagicMock()
    model.__class__ = PeftModel
    model.peft_config = {
        "episodic": MagicMock(),
        "semantic": MagicMock(),
        "in_training": MagicMock(),
    }

    loop = ConsolidationLoop(
        model=model,
        tokenizer=MagicMock(),
        consolidation_config=ConsolidationConfig(),
        training_config=TrainingConfig(),
        episodic_adapter_config=AdapterConfig(),
        semantic_adapter_config=AdapterConfig(),
        memory_store=_MS(replay_enabled=False),
        output_dir=tmp_path,
        model_name="qwen3-4b",
    )

    assert loop.extraction.model_name == "qwen3-4b", (
        "ConsolidationLoop must thread model_name into ExtractionPipeline.model_name"
    )

    # Verify model_alias flows through into kwargs() as well.
    kw = loop.extraction.kwargs(source_type="transcript", speaker_id="Speaker0")
    assert kw["model_alias"] == "qwen3-4b"


def test_run_threads_positional_args_procedural(monkeypatch):
    """Gap C (procedural): same positional-arg contract for the procedural
    chokepoint."""
    captured = {}

    def spy(model, tokenizer, transcript, session_id, **kwargs):
        captured["model"] = model
        captured["tokenizer"] = tokenizer
        captured["transcript"] = transcript
        captured["session_id"] = session_id
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr("paramem.graph.extraction_pipeline.extract_procedural_graph", spy)

    model_sentinel = MagicMock(name="model_sentinel")
    pipeline = _make_pipeline(model=model_sentinel)
    pipeline.prompts_dir = "/custom/prompts"

    pipeline.run_procedural(
        "alex prefers coffee",
        "s042",
        speaker_id="Speaker0",
        speaker_name="Alex",
    )

    assert captured["model"] is model_sentinel
    assert captured["transcript"] == "alex prefers coffee"
    assert captured["session_id"] == "s042"
    assert captured["kwargs"]["speaker_name"] == "Alex"
    assert captured["kwargs"]["prompts_dir"] == "/custom/prompts"


def test_run_uses_default_prompts_for_document(monkeypatch):
    """When called with source_type='document', :meth:`ExtractionPipeline.run`
    must still use ``DEFAULT_SYSTEM_PROMPT_FILENAME`` and
    ``DEFAULT_USER_PROMPT_FILENAME``.

    The two-prompt design (separate document-variant files) was retired
    after it produced silent drift on schema-shape rules.  One prompt-pair
    is the single ground truth for every source type; document chunks land
    in the same ``{transcript}`` slot at the chat-template layer.

    Regression guard: if any future edit re-introduces a source-type-driven
    prompt-file fork, this test fails before drift can re-emerge.
    """
    from paramem.graph.extractor import (
        DEFAULT_SYSTEM_PROMPT_FILENAME,
        DEFAULT_USER_PROMPT_FILENAME,
    )

    captured = {}

    def spy(model, tokenizer, transcript, session_id, **kwargs):
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr("paramem.graph.extraction_pipeline.extract_graph", spy)

    pipeline = _make_pipeline()
    pipeline.run("some document text", "doc-001", source_type="document", speaker_id="Speaker0")

    got_system = captured["kwargs"].get("system_prompt_filename")
    assert got_system == DEFAULT_SYSTEM_PROMPT_FILENAME, (
        f"Expected system_prompt_filename={DEFAULT_SYSTEM_PROMPT_FILENAME!r}, got {got_system!r}"
    )
    got_user = captured["kwargs"].get("user_prompt_filename")
    assert got_user == DEFAULT_USER_PROMPT_FILENAME, (
        f"Expected user_prompt_filename={DEFAULT_USER_PROMPT_FILENAME!r}, got {got_user!r}"
    )


def test_run_document_flips_gate_defaults(monkeypatch):
    """``source_type='document'`` survives as a runtime distinction for
    gate defaults — ``ha_validation`` defaults to ``False`` because HA
    grounding is dialogue-only.  Prompt selection no longer differs.
    """
    captured = {}

    def spy(model, tokenizer, transcript, session_id, **kwargs):
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr("paramem.graph.extraction_pipeline.extract_graph", spy)

    pipeline = _make_pipeline()
    pipeline.run("doc text", "doc-001", source_type="document", speaker_id="Speaker0")
    assert captured["kwargs"]["ha_validation"] is False


def test_run_honors_prompt_filename_overrides(monkeypatch):
    """Explicit ``user_prompt_filename`` / ``system_prompt_filename`` overrides
    passed via :meth:`ExtractionPipeline.run` MUST win over the source-type
    defaults.

    Regression guard for the silent-override-drop bug: previously
    ``_extraction_kwargs`` returned the locally-computed source-type defaults
    unconditionally, bypassing the ``pick`` helper that handles every other
    override.

    Without this, calibration probes (which inject ``calib_*.txt``
    variants via overrides) silently get the production prompt — making
    every "candidate vs baseline" diff a same-prompt comparison.
    """
    captured = {}

    def spy(model, tokenizer, transcript, session_id, **kwargs):
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr("paramem.graph.extraction_pipeline.extract_graph", spy)

    pipeline = _make_pipeline()
    pipeline.run(
        "transcript text",
        "calib-001",
        source_type="transcript",
        speaker_id="Speaker0",
        # Operator-supplied calibration overrides.
        user_prompt_filename="calib_extraction.txt",
        system_prompt_filename="calib_extraction_system.txt",
    )

    assert captured["kwargs"]["user_prompt_filename"] == "calib_extraction.txt", (
        "user_prompt_filename override was dropped — production prompt "
        "would have been used. See ExtractionPipeline.kwargs in "
        "paramem/graph/extraction_pipeline.py."
    )
    assert captured["kwargs"]["system_prompt_filename"] == "calib_extraction_system.txt"


def test_kwargs_honors_prompts_dir_override():
    """A per-call ``prompts_dir`` override passed via :meth:`kwargs` MUST win
    over ``self.prompts_dir`` (the construction-time default).

    Regression guard: ``prompts_dir`` was hardcoded as
    ``prompts_dir=self.prompts_dir`` inside :meth:`ExtractionPipeline.kwargs`,
    unlike every other tunable (``noise_filter``, ``system_prompt_filename``,
    etc.) which routes through the local ``pick`` helper. A caller passing
    ``prompts_dir=...`` into ``ExtractionPipeline.run(...)`` — e.g.
    ``POST /calibrate/extract`` swapping in a candidate prompt directory —
    had the override silently dropped; the model always ran against the
    construction-time ``self.prompts_dir`` while the response still echoed
    the candidate file contents, making every prompts_dir A/B invalid.
    """
    from paramem.graph.extraction_pipeline import ExtractionConfig, ExtractionPipeline

    pipeline = ExtractionPipeline(
        model=MagicMock(),
        tokenizer=MagicMock(),
        config=ExtractionConfig(),
        prompts_dir="configs/prompts",
    )

    # Override wins.
    kw = pipeline.kwargs(
        source_type="transcript",
        speaker_id="Speaker0",
        prompts_dir="/tmp/override_dir",
    )
    assert kw["prompts_dir"] == "/tmp/override_dir"

    # No override → falls back to self.prompts_dir.
    kw_default = pipeline.kwargs(source_type="transcript", speaker_id="Speaker0")
    assert kw_default["prompts_dir"] == "configs/prompts"


def test_run_procedural_uses_default_prompts_for_document(monkeypatch):
    """:meth:`run_procedural` with ``source_type='document'`` must still use
    ``DEFAULT_SYSTEM_PROMPT_FILENAME`` and
    ``DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME``.

    Symmetric to ``test_run_uses_default_prompts_for_document`` for the
    procedural rail.  Regression guard against re-introducing the
    document-variant procedural prompt file.
    """
    from paramem.graph.extractor import (
        DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME,
        DEFAULT_SYSTEM_PROMPT_FILENAME,
    )

    captured = {}

    def spy(model, tokenizer, transcript, session_id, **kwargs):
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr("paramem.graph.extraction_pipeline.extract_procedural_graph", spy)

    pipeline = _make_pipeline()
    pipeline.run_procedural(
        "some document text", "doc-001", speaker_id="Speaker0", source_type="document"
    )

    got_system = captured["kwargs"].get("system_prompt_filename")
    assert got_system == DEFAULT_SYSTEM_PROMPT_FILENAME, (
        f"Expected system_prompt_filename={DEFAULT_SYSTEM_PROMPT_FILENAME!r}, got {got_system!r}"
    )
    got_user = captured["kwargs"].get("user_prompt_filename")
    assert got_user == DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME, (
        f"Expected user_prompt_filename={DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME!r}, "
        f"got {got_user!r}"
    )


def test_run_procedural_threads_source_type_transcript(monkeypatch):
    """When called with source_type='transcript' (default),
    :meth:`run_procedural` must use ``DEFAULT_SYSTEM_PROMPT_FILENAME`` and
    ``DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME``.

    Ensures the transcript path continues to use dialogue prompts after the
    document-path plumbing was added.
    """
    from paramem.graph.extractor import (
        DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME,
        DEFAULT_SYSTEM_PROMPT_FILENAME,
    )

    captured = {}

    def spy(model, tokenizer, transcript, session_id, **kwargs):
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr("paramem.graph.extraction_pipeline.extract_procedural_graph", spy)

    pipeline = _make_pipeline()
    pipeline.run_procedural(
        "some transcript text", "s001", speaker_id="Speaker0", source_type="transcript"
    )

    got_system = captured["kwargs"].get("system_prompt_filename")
    assert got_system == DEFAULT_SYSTEM_PROMPT_FILENAME, (
        f"Expected system_prompt_filename={DEFAULT_SYSTEM_PROMPT_FILENAME!r}, got {got_system!r}"
    )
    got_user = captured["kwargs"].get("user_prompt_filename")
    assert got_user == DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME, (
        f"Expected user_prompt_filename={DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME!r}, "
        f"got {got_user!r}"
    )


def _build_loop_with_session_dump(tmp_path, monkeypatch, *, fake_graph):
    """Build a real ConsolidationLoop with snapshot dumping enabled.

    Patches ExtractionPipeline.extract_graph / extract_procedural_graph to
    return a fixed SessionGraph so the orchestrator's
    ``DebugSnapshotWriter.on_session_extracted`` runs against deterministic
    input.
    """
    from peft import PeftModel

    from paramem.training.consolidation import ConsolidationLoop
    from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

    monkeypatch.setattr(
        "paramem.graph.extraction_pipeline.extract_graph",
        lambda *a, **kw: fake_graph,
    )
    monkeypatch.setattr(
        "paramem.graph.extraction_pipeline.extract_procedural_graph",
        lambda *a, **kw: fake_graph,
    )

    model = MagicMock()
    model.__class__ = PeftModel
    model.peft_config = {
        "episodic": MagicMock(),
        "semantic": MagicMock(),
        "in_training": MagicMock(),
    }

    loop = ConsolidationLoop(
        model=model,
        tokenizer=MagicMock(),
        consolidation_config=ConsolidationConfig(),
        training_config=TrainingConfig(),
        episodic_adapter_config=AdapterConfig(),
        semantic_adapter_config=AdapterConfig(),
        memory_store=_MS(replay_enabled=False),
        output_dir=tmp_path,
        save_cycle_snapshots=True,
        snapshot_dir=tmp_path,
    )
    loop.cycle_count = 7
    return loop


def test_consolidation_dumps_per_session_graph_with_diagnostics(monkeypatch, tmp_path):
    """The consolidation orchestrator must persist the per-session
    SessionGraph (with diagnostics intact) under debug mode after each
    extraction chokepoint call, before downstream merging flattens it
    into the cumulative graph.

    Verifies the architectural seam: extraction returns a SessionGraph;
    the orchestrator calls
    ``self._debug_writer.on_session_extracted(graph, session_id, kind)``
    immediately afterward.  ``kind`` reflects which extractor produced the
    graph — never an adapter name (allocation is downstream).

    Regression guard for the debug-mode persistence gap discovered when
    diagnosing the first end-to-end document-ingest cycle.
    """
    from paramem.graph.schema import Entity, Relation, SessionGraph

    fake_graph = SessionGraph(
        session_id="sess-A",
        timestamp="2026-04-26T22:30:00Z",
        entities=[Entity(name="Alice", entity_type="person", attributes={})],
        relations=[
            Relation(
                subject="Alice",
                predicate="works_at",
                object="Acme",
                relation_type="factual",
                confidence=1.0,
                speaker_id="Speaker0",
            )
        ],
        summary="Alice works at Acme.",
        diagnostics={
            "sota_raw_response": "{'malformed': json}",
            "fallback_path": "all_dropped",
            "residual_dropped_facts": [{"subject": "Person_1"}],
        },
    )

    loop = _build_loop_with_session_dump(tmp_path, monkeypatch, fake_graph=fake_graph)
    # 2026-05-15 layout: dumps land under
    # paths.debug/episodic/[interim_<stamp>/]cycle_<N>/run_<run_id>/sessions/<id>/.
    snapshot_root = loop.snapshot_dir_for()
    assert snapshot_root is not None

    # --- Episodic chokepoint ---
    graph = loop.extraction.run("transcript text", "sess-A", speaker_id="Speaker0")
    loop._debug_writer.on_session_extracted(graph, "sess-A", "graph")

    main_path = snapshot_root / "sessions" / "sess-A" / "graph_snapshot.json"
    assert main_path.exists(), f"episodic dump missing at {main_path}"
    main_dump = json.loads(main_path.read_text())
    assert main_dump["diagnostics"]["fallback_path"] == "all_dropped"
    assert main_dump["diagnostics"]["sota_raw_response"] == "{'malformed': json}"
    assert main_dump["relations"][0]["predicate"] == "works_at"

    # --- Procedural chokepoint ---
    proc_graph = loop.extraction.run_procedural("transcript text", "sess-A", speaker_id="Speaker0")
    loop._debug_writer.on_session_extracted(proc_graph, "sess-A", "procedural_graph")

    proc_path = snapshot_root / "sessions" / "sess-A" / "procedural_graph_snapshot.json"
    assert proc_path.exists(), f"procedural dump missing at {proc_path}"
    proc_dump = json.loads(proc_path.read_text())
    assert proc_dump["diagnostics"]["fallback_path"] == "all_dropped"

    # Same session, two distinct dumps — operator can compare both extractors
    # for one session by `cat sessions/<session_id>/*_snapshot.json`.
    assert main_path.parent == proc_path.parent
    assert main_path.name != proc_path.name


def test_on_session_extracted_short_circuits_when_debug_off(tmp_path):
    """``DebugSnapshotWriter.on_session_extracted`` must short-circuit when
    either ``save_cycle_snapshots`` is False or the debug base
    (``_debug_base``) is None.  Production runs without debug mode must not
    pay any disk cost from this debug-only diagnostic.
    """
    from paramem.training.consolidation import ConsolidationLoop
    from paramem.training.debug_snapshot import DebugSnapshotWriter

    loop = ConsolidationLoop.__new__(ConsolidationLoop)
    loop.cycle_count = 0
    loop.run_id = "20260515T000000Z_aaaaaa"
    writer = DebugSnapshotWriter(loop)
    fake_graph = MagicMock(model_dump_json=lambda **kw: "{}")

    # save_cycle_snapshots=False → no write.
    loop.save_cycle_snapshots = False
    loop._debug_base = tmp_path
    writer.on_session_extracted(fake_graph, "s001", "graph")
    assert not (tmp_path / "episodic").exists(), "dump fired despite save_cycle_snapshots=False"

    # _debug_base=None → no write (no AttributeError on ``None / "<tier>"``).
    loop.save_cycle_snapshots = True
    loop._debug_base = None
    writer.on_session_extracted(fake_graph, "s002", "graph")


# ---------------------------------------------------------------------------
# Phase 2a regression: no speaker_id empty-string defaults
# ---------------------------------------------------------------------------


_SPEAKER_ID_DEFAULT_PATTERN = re.compile(r'speaker_id:\s*str\s*=\s*""')

_SPEAKER_ID_DEFAULT_FILES = (
    "paramem/graph/extractor.py",
    "paramem/graph/extraction_pipeline.py",
    "paramem/training/consolidation.py",
)


def test_no_speaker_id_empty_string_defaults():
    """Regression guard: ``speaker_id: str = ""`` must not appear in the
    extraction or consolidation modules.

    The empty-string default is a soft seam that lets a caller omit
    speaker_id and silently propagate ``""`` into the extraction chain.
    Pydantic accepts ``""`` as a valid string, so the schema does not catch
    the omission — only this test does.

    To verify the guard works: temporarily add ``speaker_id: str = ""`` to
    one of the listed files, run this test, and confirm it fails.  Then
    revert.

    Phase 1 (commit 1b7eba2) left the defaults intentionally as a deferred
    cleanup; Phase 2a removes them (this test is the gating CI assertion).
    """
    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[tuple[str, int, str]] = []

    for rel in _SPEAKER_ID_DEFAULT_FILES:
        py_file = repo_root / rel
        try:
            text = py_file.read_text()
        except FileNotFoundError:
            offenders.append((rel, 0, "<file not found>"))
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if _SPEAKER_ID_DEFAULT_PATTERN.search(line):
                offenders.append((rel, lineno, line.strip()))

    assert not offenders, (
        "speaker_id empty-string default found — Phase 2a regression.\n"
        "Every caller must supply a non-empty speaker_id; the SpeakerStore\n"
        "anonymous-group ID covers the no-named-speaker case.\n"
        "Offending lines:\n"
        + "\n".join(f"  {path}:{lineno} — {src}" for path, lineno, src in offenders)
    )


def test_kwargs_raises_on_missing_speaker_id():
    """:meth:`ExtractionPipeline.kwargs` must raise ``ValueError`` when
    ``speaker_id`` is absent from the overrides dict.

    This enforces the contract at the extraction chokepoint: a caller that
    forgets to pass speaker_id gets an immediate, clear error instead of
    silently stamping ``""`` onto every Relation it produces.
    """
    pipeline = _make_pipeline()
    with pytest.raises(ValueError, match="speaker_id is required"):
        pipeline.kwargs()


def test_kwargs_raises_on_empty_speaker_id():
    """:meth:`ExtractionPipeline.kwargs` must raise ``ValueError`` when
    ``speaker_id`` is explicitly passed as an empty string.

    An empty string is semantically indistinguishable from a missing value
    from the schema's perspective (Pydantic accepts both), so the guard must
    treat it the same way as an absent key.
    """
    pipeline = _make_pipeline()
    with pytest.raises(ValueError, match="speaker_id is required"):
        pipeline.kwargs(speaker_id="")


# ---------------------------------------------------------------------------
# Structural guard — BackgroundTrainer single-literal invariant
# ---------------------------------------------------------------------------


def _find_background_trainer_constructor_calls(py_file: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, source)`` for every ``BackgroundTrainer(...)`` AST call node.

    Uses :mod:`ast` to count actual constructor call sites (``ast.Call`` whose
    ``func`` is an ``ast.Name`` with id ``"BackgroundTrainer"``).  String
    occurrences in docstrings, comments, or log messages are not counted.
    """
    import ast

    try:
        text = py_file.read_text()
    except UnicodeDecodeError:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "BackgroundTrainer"
        ):
            line = lines[node.lineno - 1] if 0 < node.lineno <= len(lines) else ""
            out.append((node.lineno, line.strip()))
    return out


def test_background_trainer_single_constructor_literal_in_app():
    """``BackgroundTrainer(...)`` must appear EXACTLY ONCE in ``paramem/server/app.py``.

    The single-constructor invariant enforces the singleton pattern:
    ``_build_bg_trainer`` is the sole factory; all dispatch sites call
    ``_active_bg_trainer`` (get-or-create) or ``_build_bg_trainer``
    (deliberate fresh-build after release).  A raw ``BackgroundTrainer(...)``
    at a dispatch site orphans the prior worker thread and leaks its VRAM.

    If this test fails, the new call site must be moved inside
    ``_build_bg_trainer`` or replaced with ``_active_bg_trainer``.
    """
    import ast

    repo_root = Path(__file__).resolve().parent.parent
    app_file = repo_root / "paramem" / "server" / "app.py"

    text = app_file.read_text()
    tree = ast.parse(text)
    lines = text.splitlines()

    # Locate all BackgroundTrainer(...) calls together with the enclosing
    # function name so the assertion can confirm the sole call is inside
    # _build_bg_trainer.
    hits: list[tuple[int, str, str | None]] = []  # (lineno, source, enclosing_fn)

    # Build a mapping: lineno → enclosing function name (nearest FunctionDef ancestor).
    fn_ranges: list[tuple[int, int, str]] = []  # (start, end, name)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # end_lineno is available for Python 3.8+ AST nodes.
            end = getattr(node, "end_lineno", node.lineno)
            fn_ranges.append((node.lineno, end, node.name))

    def _enclosing_fn(lineno: int) -> str | None:
        # Find the innermost function whose range contains lineno.
        best: tuple[int, str] | None = None
        for start, end, name in fn_ranges:
            if start <= lineno <= end:
                if best is None or start > best[0]:
                    best = (start, name)
        return best[1] if best else None

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "BackgroundTrainer"
        ):
            line = lines[node.lineno - 1] if 0 < node.lineno <= len(lines) else ""
            hits.append((node.lineno, line.strip(), _enclosing_fn(node.lineno)))

    assert len(hits) == 1, (
        f"Expected EXACTLY ONE BackgroundTrainer(...) constructor literal in "
        f"paramem/server/app.py (inside _build_bg_trainer); found {len(hits)}.\n"
        "Each extra literal is a potential orphaned-worker leak. Move new "
        "dispatch sites to _active_bg_trainer (singleton reuse) or "
        "_build_bg_trainer (deliberate fresh-build after release):\n"
        + "\n".join(f"  app.py:{lineno} in {fn or '<module>'} — {src}" for lineno, src, fn in hits)
    )

    lineno, src, enclosing_fn = hits[0]
    assert enclosing_fn == "_build_bg_trainer", (
        f"The single BackgroundTrainer(...) literal must live inside "
        f"_build_bg_trainer; found it in {enclosing_fn!r} at app.py:{lineno}. "
        "Move the construction into _build_bg_trainer and have the caller "
        "use _active_bg_trainer or _build_bg_trainer."
    )
