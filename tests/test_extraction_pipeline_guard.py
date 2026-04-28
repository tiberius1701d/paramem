"""Structural guard: forbid direct extract_graph / extract_procedural_graph calls.

The extraction pipeline has exactly one callable topology. All orchestrators
(server, experiments, tests) must reach the extractors through the
ConsolidationLoop helpers `_run_extract_graph` / `_run_extract_procedural_graph`,
never via direct calls to `extract_graph(...)` / `extract_procedural_graph(...)`.

This test scans the tracked codebase and fails if a new call site appears
outside the whitelist. Pair with the parity tests in test_consolidation.py.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

# Files allowed to call the extractors directly:
# - extractor.py: the module defining them.
# - consolidation.py: the only orchestrator that wraps them.
# - tests/: may patch, import, or assert on their names.
# - archive/: historical, not executed.
_ALLOWED_SUFFIXES = (
    "paramem/graph/extractor.py",
    "paramem/training/consolidation.py",
)

_ALLOWED_PREFIXES = (
    "tests/",
    "archive/",
)

# Grandfathered standalone experiment probes. These pre-date the
# ConsolidationLoop helpers and run outside the production orchestrator path
# (one-shot GPU evals, not a live extract→train loop). New experiment code
# MUST route through ConsolidationLoop._run_extract_graph instead. If one of
# these is rewritten to use the helpers, remove it from this set.
_GRANDFATHERED_FILES = frozenset(
    {
        "experiments/chained_adapter_smoke.py",
        "experiments/test8_large_scale.py",
        "experiments/test_early_stopping.py",
        "experiments/utils/perltqa_loader.py",
        "experiments/utils/test_harness.py",
    }
)

_FORBIDDEN_CALL = re.compile(r"\bextract(?:_procedural)?_graph\s*\(")


def _is_allowed(rel: str) -> bool:
    if rel in _GRANDFATHERED_FILES:
        return True
    if rel.endswith(_ALLOWED_SUFFIXES):
        return True
    return any(rel.startswith(p) for p in _ALLOWED_PREFIXES)


def test_no_direct_extract_graph_calls_outside_whitelist():
    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[tuple[str, int, str]] = []

    for py_file in repo_root.rglob("*.py"):
        rel = py_file.relative_to(repo_root).as_posix()
        if _is_allowed(rel):
            continue
        try:
            text = py_file.read_text()
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            # Skip comments and import statements — only flag call sites.
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith(("import ", "from ")):
                continue
            if _FORBIDDEN_CALL.search(line):
                offenders.append((rel, lineno, line.strip()))

    assert not offenders, (
        "Direct extract_graph/extract_procedural_graph calls found outside the "
        "allowed orchestrator. Route new callers through "
        "ConsolidationLoop._run_extract_graph / _run_extract_procedural_graph "
        "so extraction behavior cannot diverge between production and tests:\n"
        + "\n".join(f"  {path}:{line} — {src}" for path, line, src in offenders)
    )


# Positional params the helpers supply themselves — not user-facing kwargs.
_EXTRACTOR_POSITIONAL = {"model", "tokenizer", "transcript", "session_id"}


def _extraction_kwargs_namespace():
    """Build a minimal namespace with just the attrs _extraction_kwargs reads.

    Avoids constructing a full ConsolidationLoop (which requires a real model
    for _ensure_adapters). Contract tests don't need the full object graph —
    only that the method's output keys line up with the extractor signature.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        extraction_temperature=0.0,
        extraction_max_tokens=2048,
        prompts_dir=None,
        extraction_stt_correction=True,
        extraction_ha_validation=True,
        extraction_noise_filter="",
        extraction_noise_filter_model="claude-sonnet-4-6",
        extraction_noise_filter_endpoint=None,
        extraction_ner_check=False,
        extraction_ner_model="en_core_web_sm",
        extraction_plausibility_judge="auto",
        extraction_plausibility_stage="deanon",
        extraction_verify_anonymization=True,
        # Per-session debug-dump gate read by _dump_session_graph (called
        # from both extraction chokepoints).  Wrappers short-circuit when
        # either attr is falsy, keeping these tests file-system-free.
        save_cycle_snapshots=False,
        snapshot_dir=None,
    )


@pytest.mark.parametrize("source_type", ["transcript", "document"])
def test_extraction_kwargs_match_extract_graph_signature(source_type):
    """Every kwarg the helper passes must exist on extract_graph.

    Catches rename drift: if the helper grows a new flag but the extractor
    hasn't caught up (or vice versa), production crashes with TypeError.
    This test fails first.  Parametrized over both source_type values so
    both the transcript and document kwarg sets are validated.
    """
    from paramem.graph.extractor import extract_graph
    from paramem.training.consolidation import ConsolidationLoop

    ns = _extraction_kwargs_namespace()
    helper_keys = set(ConsolidationLoop._extraction_kwargs(ns, source_type=source_type).keys())
    extractor_params = set(inspect.signature(extract_graph).parameters) - _EXTRACTOR_POSITIONAL

    unknown = helper_keys - extractor_params
    assert not unknown, (
        f"_extraction_kwargs(source_type={source_type!r}) passes kwargs extract_graph does "
        f"not accept: {sorted(unknown)}. "
        "Either rename the helper key to match the extractor signature, or add the "
        "parameter to extract_graph."
    )


def test_procedural_kwargs_match_extract_procedural_graph_signature():
    """Same contract for the procedural helper — its inline kwargs dict must
    line up with the extract_procedural_graph signature."""
    from paramem.graph.extractor import extract_procedural_graph

    # Mirror the inline kwargs shape inside _run_extract_procedural_graph.
    # If that helper ever diverges, update this set alongside.
    procedural_keys = {
        "max_tokens",
        "prompts_dir",
        "stt_correction",
        "speaker_name",
        "system_prompt_filename",
        "user_prompt_filename",
    }
    extractor_params = (
        set(inspect.signature(extract_procedural_graph).parameters) - _EXTRACTOR_POSITIONAL
    )

    unknown = procedural_keys - extractor_params
    assert not unknown, (
        f"_run_extract_procedural_graph passes kwargs extract_procedural_graph does not "
        f"accept: {sorted(unknown)}"
    )


def _loop_ns_with_model(model, tokenizer=None):
    """Namespace usable as `self` for bound-call tests of _run_extract_graph."""
    import types
    from unittest.mock import MagicMock

    from paramem.training.consolidation import ConsolidationLoop

    ns = _extraction_kwargs_namespace()
    ns.model = model
    ns.tokenizer = tokenizer if tokenizer is not None else MagicMock()
    ns._disable_gradient_checkpointing = lambda: None
    # _run_extract_graph calls self._extraction_kwargs(...) — bind it.
    ns._extraction_kwargs = types.MethodType(ConsolidationLoop._extraction_kwargs, ns)
    # _run_extract_graph also calls self._dump_session_graph(...); bind it
    # so the wrapper can resolve the attr — short-circuits via the namespace's
    # save_cycle_snapshots=False / snapshot_dir=None gate.
    ns._dump_session_graph = types.MethodType(ConsolidationLoop._dump_session_graph, ns)
    return ns


def test_run_extract_graph_wraps_peft_model_in_disable_adapter(monkeypatch):
    """Gap B: when the model is a PeftModel, the helper MUST enter the
    `disable_adapter()` context manager before calling extract_graph.

    Regression guard: if the guard is ever refactored away, extraction
    silently runs with the training-active adapter, contaminating output.
    """
    from unittest.mock import MagicMock

    from peft import PeftModel

    from paramem.training.consolidation import ConsolidationLoop

    fake_peft = MagicMock(spec=PeftModel)  # isinstance(fake_peft, PeftModel) is True
    ns = _loop_ns_with_model(fake_peft)

    monkeypatch.setattr(
        "paramem.training.consolidation.extract_graph",
        lambda *a, **kw: MagicMock(),
    )
    ConsolidationLoop._run_extract_graph(ns, "transcript", "s001")

    fake_peft.disable_adapter.assert_called_once()


def test_run_extract_graph_skips_disable_adapter_for_plain_model(monkeypatch):
    """Gap B (negative): plain (non-PeftModel) models must NOT be wrapped."""
    from unittest.mock import MagicMock

    from paramem.training.consolidation import ConsolidationLoop

    plain_model = MagicMock()  # no spec — fails isinstance(_, PeftModel)
    ns = _loop_ns_with_model(plain_model)

    monkeypatch.setattr(
        "paramem.training.consolidation.extract_graph",
        lambda *a, **kw: MagicMock(),
    )
    ConsolidationLoop._run_extract_graph(ns, "transcript", "s001")

    plain_model.disable_adapter.assert_not_called()


def test_run_extract_graph_threads_positional_args(monkeypatch):
    """Gap C: _run_extract_graph must pass (model, tokenizer, transcript, session_id)
    to the real extractor unchanged. If the helper ever reshapes the call, this
    test fails before production notices.
    """
    from unittest.mock import MagicMock

    from paramem.training.consolidation import ConsolidationLoop

    captured = {}

    def spy(model, tokenizer, transcript, session_id, **kwargs):
        captured["model"] = model
        captured["tokenizer"] = tokenizer
        captured["transcript"] = transcript
        captured["session_id"] = session_id
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr("paramem.training.consolidation.extract_graph", spy)

    model_sentinel = MagicMock(name="model_sentinel")
    tokenizer_sentinel = MagicMock(name="tokenizer_sentinel")
    ns = _loop_ns_with_model(model_sentinel, tokenizer_sentinel)
    ns.prompts_dir = "/custom/prompts"
    ns.extraction_noise_filter = "claude"

    ConsolidationLoop._run_extract_graph(ns, "tobias lives here", "s042")

    assert captured["model"] is model_sentinel
    assert captured["tokenizer"] is tokenizer_sentinel
    assert captured["transcript"] == "tobias lives here"
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


def test_server_extract_session_callsites_pass_identical_kwargs():
    """Gap D: every `*.extract_session(...)` call site under `paramem/server/`
    must pass the same kwarg names as every other server caller.

    Silently dropping flags on one path (as app.py:1276 did before this test
    landed) means the same consolidation operation runs with different SOTA
    pipeline behavior depending on which code path triggered it.

    Scans the whole `paramem/server/` tree, so a new orchestrator that starts
    calling `extract_session` is audited automatically — no test edit needed.
    """
    repo_root = Path(__file__).resolve().parent.parent
    server_dir = repo_root / "paramem" / "server"

    per_file: dict[str, set[str]] = {}
    for py_file in server_dir.rglob("*.py"):
        call_kwargs = _collect_extract_session_kwargs(py_file)
        if not call_kwargs:
            continue
        per_file[py_file.relative_to(repo_root).as_posix()] = set().union(*call_kwargs)

    assert len(per_file) >= 2, (
        "expected at least two server-layer extract_session call sites; found: "
        f"{sorted(per_file)}. If extract_session was refactored out of the server, "
        "update or remove this test. If a new caller replaced an old one, the "
        "single remaining site still needs a peer for parity to be meaningful."
    )

    reference_path, reference_kwargs = sorted(per_file.items())[0]
    divergent: dict[str, tuple[set[str], set[str]]] = {}
    for path, kwargs in per_file.items():
        if path == reference_path:
            continue
        missing_here = reference_kwargs - kwargs
        extra_here = kwargs - reference_kwargs
        if missing_here or extra_here:
            divergent[path] = (missing_here, extra_here)

    assert not divergent, (
        "extract_session kwargs diverge across server-layer call sites — same "
        f"operation, different flags. Reference: {reference_path}\n"
        + "\n".join(
            f"  {path}: missing={sorted(miss)} extra={sorted(extra)}"
            for path, (miss, extra) in sorted(divergent.items())
        )
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
        "extraction_stt_correction": False,
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


def test_run_extract_procedural_graph_wraps_peft_model_in_disable_adapter(monkeypatch):
    """Procedural mirror of Gap B: PeftModel must enter `disable_adapter()`
    before the procedural extractor is called.

    Without this guard, procedural extraction runs with the training-active
    adapter, contaminating preference capture.
    """
    from unittest.mock import MagicMock

    from peft import PeftModel

    from paramem.training.consolidation import ConsolidationLoop

    fake_peft = MagicMock(spec=PeftModel)
    ns = _loop_ns_with_model(fake_peft)

    monkeypatch.setattr(
        "paramem.training.consolidation.extract_procedural_graph",
        lambda *a, **kw: MagicMock(),
    )
    ConsolidationLoop._run_extract_procedural_graph(ns, "transcript", "s001")

    fake_peft.disable_adapter.assert_called_once()


def test_run_extract_procedural_graph_skips_disable_adapter_for_plain_model(monkeypatch):
    """Procedural negative case: plain models must NOT be wrapped."""
    from unittest.mock import MagicMock

    from paramem.training.consolidation import ConsolidationLoop

    plain_model = MagicMock()
    ns = _loop_ns_with_model(plain_model)

    monkeypatch.setattr(
        "paramem.training.consolidation.extract_procedural_graph",
        lambda *a, **kw: MagicMock(),
    )
    ConsolidationLoop._run_extract_procedural_graph(ns, "transcript", "s001")

    plain_model.disable_adapter.assert_not_called()


def test_consolidation_loop_constructor_stores_extraction_flags(tmp_path):
    """Experiment-path mirror of (b): kwargs forwarded to
    `ConsolidationLoop.__init__` must land on `self.extraction_*` unchanged.

    `run_multi_session` (and any other experiment entrypoint) configures the
    pipeline by passing `extraction_*` kwargs to the loop constructor. If the
    constructor ever silently drops or renames one of them, experiments run on
    defaults while tests that check `_extraction_kwargs` output still pass —
    because those tests seed a namespace by hand. This closes that loop.
    """
    from unittest.mock import MagicMock

    from peft import PeftModel

    from paramem.training.consolidation import ConsolidationLoop
    from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

    flipped = {
        "extraction_temperature": 0.7,
        "extraction_max_tokens": 4096,
        "extraction_stt_correction": False,
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

    # Skip adapter wiring — we only care about flag storage on self.
    # __class__ = PeftModel so _ensure_adapters' isinstance check short-circuits
    # without restricting the mock's attribute surface.
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
        output_dir=tmp_path,
        persist_graph=False,
        **flipped,
    )

    for key, expected in flipped.items():
        actual = getattr(loop, key)
        assert actual == expected, (
            f"ConsolidationLoop.__init__ dropped kwarg {key!r}: "
            f"passed={expected!r}, stored={actual!r}"
        )


def test_run_extract_procedural_graph_threads_positional_args(monkeypatch):
    """Gap C (procedural): same positional-arg contract for the procedural helper."""
    from unittest.mock import MagicMock

    from paramem.training.consolidation import ConsolidationLoop

    captured = {}

    def spy(model, tokenizer, transcript, session_id, **kwargs):
        captured["model"] = model
        captured["tokenizer"] = tokenizer
        captured["transcript"] = transcript
        captured["session_id"] = session_id
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr("paramem.training.consolidation.extract_procedural_graph", spy)

    model_sentinel = MagicMock(name="model_sentinel")
    ns = _loop_ns_with_model(model_sentinel)
    ns.prompts_dir = "/custom/prompts"

    ConsolidationLoop._run_extract_procedural_graph(
        ns, "tobias prefers coffee", "s042", speaker_name="Tobias", stt_correction=False
    )

    assert captured["model"] is model_sentinel
    assert captured["transcript"] == "tobias prefers coffee"
    assert captured["session_id"] == "s042"
    assert captured["kwargs"]["speaker_name"] == "Tobias"
    assert captured["kwargs"]["stt_correction"] is False
    assert captured["kwargs"]["prompts_dir"] == "/custom/prompts"


def test_extraction_system_document_prompt_present():
    """Verify extraction_system_document.txt exists under configs/prompts/.

    The document extraction path resolves this filename at runtime via
    load_extraction_prompts; a missing file causes a silent fallback to the
    hardcoded default system prompt, which has no narrator-binding directives.
    """
    repo_root = Path(__file__).resolve().parent.parent
    doc_prompt = repo_root / "configs" / "prompts" / "extraction_system_document.txt"
    assert doc_prompt.exists(), (
        "configs/prompts/extraction_system_document.txt is missing. "
        "The document extraction path will silently fall back to the dialogue "
        "system prompt, losing narrator binding."
    )


def test_run_extract_graph_threads_source_type(monkeypatch):
    """When called with source_type='document', _run_extract_graph must resolve
    BOTH system_prompt_filename to DOCUMENT_SYSTEM_PROMPT_FILENAME and
    user_prompt_filename to DOCUMENT_USER_PROMPT_FILENAME.

    Regression guard: if _extraction_kwargs ever stops routing source_type to
    the correct filenames, document extraction silently uses the dialogue prompts.
    """
    from unittest.mock import MagicMock

    from paramem.graph.extractor import (
        DOCUMENT_SYSTEM_PROMPT_FILENAME,
        DOCUMENT_USER_PROMPT_FILENAME,
    )
    from paramem.training.consolidation import ConsolidationLoop

    captured = {}

    def spy(model, tokenizer, transcript, session_id, **kwargs):
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr("paramem.training.consolidation.extract_graph", spy)

    ns = _loop_ns_with_model(MagicMock())
    ConsolidationLoop._run_extract_graph(
        ns, "some document text", "doc-001", source_type="document"
    )

    got_system = captured["kwargs"].get("system_prompt_filename")
    assert got_system == DOCUMENT_SYSTEM_PROMPT_FILENAME, (
        f"Expected system_prompt_filename={DOCUMENT_SYSTEM_PROMPT_FILENAME!r}, got {got_system!r}"
    )
    got_user = captured["kwargs"].get("user_prompt_filename")
    assert got_user == DOCUMENT_USER_PROMPT_FILENAME, (
        f"Expected user_prompt_filename={DOCUMENT_USER_PROMPT_FILENAME!r}, got {got_user!r}"
    )


def test_extraction_document_prompt_present():
    """Verify extraction_document.txt exists under configs/prompts/.

    The document extraction path resolves this filename at runtime via
    load_extraction_prompts; a missing file causes a silent fallback to the
    hardcoded default user template, which has dialogue-shaped few-shots
    incompatible with document extraction.
    """
    repo_root = Path(__file__).resolve().parent.parent
    doc_prompt = repo_root / "configs" / "prompts" / "extraction_document.txt"
    assert doc_prompt.exists(), (
        "configs/prompts/extraction_document.txt is missing. "
        "The document extraction path will silently fall back to the dialogue "
        "user template, producing dialogue-shaped few-shots for document input."
    )


def _loop_ns_for_procedural():
    """Namespace usable as `self` for bound-call tests of _run_extract_procedural_graph."""
    import types
    from unittest.mock import MagicMock

    from paramem.training.consolidation import ConsolidationLoop

    ns = _extraction_kwargs_namespace()
    ns.model = MagicMock()
    ns.tokenizer = MagicMock()
    ns._disable_gradient_checkpointing = lambda: None
    # Bind _dump_session_graph so the wrapper resolves the attr; short-circuits
    # on the namespace's save_cycle_snapshots=False / snapshot_dir=None gate.
    ns._dump_session_graph = types.MethodType(ConsolidationLoop._dump_session_graph, ns)
    return ns


def test_run_extract_procedural_graph_threads_source_type_document(monkeypatch):
    """When called with source_type='document', _run_extract_procedural_graph must
    resolve BOTH system_prompt_filename to DOCUMENT_SYSTEM_PROMPT_FILENAME AND
    user_prompt_filename to DOCUMENT_PROCEDURAL_USER_PROMPT_FILENAME.

    Symmetric to test_run_extract_graph_threads_source_type for the episodic rail.
    Regression guard: if _run_extract_procedural_graph ever stops routing
    source_type to the correct procedural filenames, document procedural
    extraction silently receives dialogue-shaped few-shots that reference a
    non-existent assistant response.
    """
    from unittest.mock import MagicMock

    from paramem.graph.extractor import (
        DOCUMENT_PROCEDURAL_USER_PROMPT_FILENAME,
        DOCUMENT_SYSTEM_PROMPT_FILENAME,
    )
    from paramem.training.consolidation import ConsolidationLoop

    captured = {}

    def spy(model, tokenizer, transcript, session_id, **kwargs):
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr("paramem.training.consolidation.extract_procedural_graph", spy)

    ns = _loop_ns_for_procedural()
    ConsolidationLoop._run_extract_procedural_graph(
        ns, "some document text", "doc-001", source_type="document"
    )

    got_system = captured["kwargs"].get("system_prompt_filename")
    assert got_system == DOCUMENT_SYSTEM_PROMPT_FILENAME, (
        f"Expected system_prompt_filename={DOCUMENT_SYSTEM_PROMPT_FILENAME!r}, got {got_system!r}"
    )
    got_user = captured["kwargs"].get("user_prompt_filename")
    assert got_user == DOCUMENT_PROCEDURAL_USER_PROMPT_FILENAME, (
        f"Expected user_prompt_filename={DOCUMENT_PROCEDURAL_USER_PROMPT_FILENAME!r}, "
        f"got {got_user!r}"
    )


def test_run_extract_procedural_graph_threads_source_type_transcript(monkeypatch):
    """When called with source_type='transcript' (default), _run_extract_procedural_graph
    must use DEFAULT_SYSTEM_PROMPT_FILENAME and DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME.

    Ensures the transcript path continues to use dialogue prompts after the
    document-path plumbing was added.
    """
    from unittest.mock import MagicMock

    from paramem.graph.extractor import (
        DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME,
        DEFAULT_SYSTEM_PROMPT_FILENAME,
    )
    from paramem.training.consolidation import ConsolidationLoop

    captured = {}

    def spy(model, tokenizer, transcript, session_id, **kwargs):
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr("paramem.training.consolidation.extract_procedural_graph", spy)

    ns = _loop_ns_for_procedural()
    ConsolidationLoop._run_extract_procedural_graph(
        ns, "some transcript text", "s001", source_type="transcript"
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


def test_extraction_chokepoints_dump_per_session_graph_with_diagnostics(monkeypatch, tmp_path):
    """Both extraction chokepoints must persist the per-session SessionGraph
    (with diagnostics intact) under debug mode, before downstream merging
    flattens it into the cumulative graph.

    Verifies the architectural seam: extraction is upstream of adapter
    allocation, so the dump fires from ``_run_extract_graph`` and
    ``_run_extract_procedural_graph`` (the single-topology chokepoints),
    not from any orchestrator.  ``kind`` reflects which extractor produced
    the graph — never an adapter name (allocation is downstream).

    Regression guard for the debug-mode persistence gap discovered when
    diagnosing the first end-to-end document-ingest cycle.
    """
    from unittest.mock import MagicMock

    from paramem.graph.schema import Entity, Relation, SessionGraph
    from paramem.training.consolidation import ConsolidationLoop

    def _fake_session_graph(session_id: str):
        return SessionGraph(
            session_id=session_id,
            timestamp="2026-04-26T22:30:00Z",
            entities=[Entity(name="Alice", entity_type="person", attributes={})],
            relations=[
                Relation(
                    subject="Alice",
                    predicate="works_at",
                    object="Acme",
                    relation_type="factual",
                    confidence=1.0,
                )
            ],
            summary="Alice works at Acme.",
            diagnostics={
                "sota_raw_response": "{'malformed': json}",
                "fallback_path": "all_dropped",
                "residual_dropped_facts": [{"subject": "Person_1"}],
            },
        )

    # --- Episodic chokepoint ---
    ns_main = _loop_ns_with_model(MagicMock())
    ns_main.save_cycle_snapshots = True
    ns_main.snapshot_dir = tmp_path
    ns_main.cycle_count = 7
    monkeypatch.setattr(
        "paramem.training.consolidation.extract_graph",
        lambda *a, **kw: _fake_session_graph("sess-A"),
    )
    ConsolidationLoop._run_extract_graph(ns_main, "transcript text", "sess-A")

    main_path = tmp_path / "cycle_7" / "sessions" / "sess-A" / "graph_snapshot.json"
    assert main_path.exists(), f"episodic dump missing at {main_path}"
    main_dump = json.loads(main_path.read_text())
    assert main_dump["diagnostics"]["fallback_path"] == "all_dropped"
    assert main_dump["diagnostics"]["sota_raw_response"] == "{'malformed': json}"
    assert main_dump["relations"][0]["predicate"] == "works_at"

    # --- Procedural chokepoint ---
    ns_proc = _loop_ns_for_procedural()
    ns_proc.save_cycle_snapshots = True
    ns_proc.snapshot_dir = tmp_path
    ns_proc.cycle_count = 7
    monkeypatch.setattr(
        "paramem.training.consolidation.extract_procedural_graph",
        lambda *a, **kw: _fake_session_graph("sess-A"),
    )
    ConsolidationLoop._run_extract_procedural_graph(ns_proc, "transcript text", "sess-A")

    proc_path = tmp_path / "cycle_7" / "sessions" / "sess-A" / "procedural_graph_snapshot.json"
    assert proc_path.exists(), f"procedural dump missing at {proc_path}"
    proc_dump = json.loads(proc_path.read_text())
    assert proc_dump["diagnostics"]["fallback_path"] == "all_dropped"

    # Same session, two distinct dumps — operator can compare both extractors
    # for one session by `cat sessions/<session_id>/*_snapshot.json`.
    assert main_path.parent == proc_path.parent
    assert main_path.name != proc_path.name


def test_dump_session_graph_short_circuits_when_debug_off(monkeypatch, tmp_path):
    """The dump gate must short-circuit when either save_cycle_snapshots is
    False or snapshot_dir is None.  Production runs without debug mode must
    not pay any disk cost from this debug-only diagnostic.
    """
    from unittest.mock import MagicMock

    from paramem.training.consolidation import ConsolidationLoop

    monkeypatch.setattr(
        "paramem.training.consolidation.extract_graph",
        lambda *a, **kw: MagicMock(model_dump_json=lambda **kw: "{}"),
    )

    # save_cycle_snapshots=False — default in _loop_ns_with_model.
    ns = _loop_ns_with_model(MagicMock())
    ns.snapshot_dir = tmp_path  # set, but gate is on save_cycle_snapshots
    ConsolidationLoop._run_extract_graph(ns, "t", "s001")
    assert not (tmp_path / "cycle_0").exists(), "dump fired despite save_cycle_snapshots=False"

    # snapshot_dir=None — equally short-circuits.
    ns2 = _loop_ns_with_model(MagicMock())
    ns2.save_cycle_snapshots = True
    ns2.snapshot_dir = None
    # Would raise AttributeError on `None / "cycle_X"` if the gate didn't fire.
    ConsolidationLoop._run_extract_graph(ns2, "t", "s002")
