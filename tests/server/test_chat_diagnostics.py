"""Tests for the serving-turn phase trace and the ``ChatResult.diagnostics``
surface.

Covers:

* ``handle_chat`` opens its own ``extraction_trace()`` / ``phase_trace``
  scope around the whole dispatch, recording exactly one ``serve_turn``
  phase per call whose ``raw_output``/``parsed`` fields carry the returned
  reply text and diagnostics dict.
* Every routing branch (personal-probe, HA, cloud, abstention, base model)
  stamps the handle_chat-level diagnostics keys onto the result, and only
  the personal-probe leg additionally carries the probe/temporal keys
  ``_probe_and_reason`` builds.
* ``is_residual`` is computed unconditionally, not gated on
  ``config.debug``.
* ``_probe_and_reason``'s own diagnostics: per-adapter probe/recall
  counts, the recalled-fact total, and the date-group selection outcome
  (including the zero-survivor early return, which never reaches
  probing).
* A dispatch that raises still leaves a failed-outcome phase record
  behind and re-raises the original exception.

Mock patterns mirror ``tests/test_inference_response_shaping.py`` and
``tests/test_abstention.py`` — no model, no GPU, no network. The autouse
``_extraction_trace_scope`` fixture (``tests/conftest.py``) supplies the
outer trace scope; ``with extraction_trace() as trace:`` inside a test
re-enters that same scope (re-entry is a no-op by design) so the test can
read back the records ``handle_chat`` appended.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from paramem.graph.phase_trace import extraction_trace
from paramem.memory.store import MemoryStore as _MS
from paramem.server.config import ServerConfig
from paramem.server.inference import ChatResult, _probe_and_reason, handle_chat
from paramem.server.router import Intent, RoutingPlan, RoutingStep

# The diagnostics keys handle_chat guarantees on every non-relay result,
# regardless of which branch produced it.
_HANDLE_CHAT_KEYS = {
    "conversation_id",
    "intent",
    "paths_attempted",
    "fallthrough_reason",
    "exit_via",
    "is_residual",
    "is_self_referential",
}


class TestServeTurnPhaseRecord:
    """``handle_chat`` records exactly one ``serve_turn`` phase per call,
    with ``raw_output``/``parsed`` tied to the returned result."""

    def test_records_exactly_one_serve_turn_with_matching_raw_and_parsed(self):
        config = ServerConfig()
        with (
            patch(
                "paramem.server.inference._base_model_answer",
                return_value=ChatResult(text="base reply"),
            ),
            extraction_trace() as trace,
        ):
            result = handle_chat(
                text="hello there",
                conversation_id="c1",
                speaker=None,
                speaker_id="speaker0",
                history=None,
                model=MagicMock(),
                tokenizer=MagicMock(),
                config=config,
            )
            serve_records = [r for r in trace.records if r.name == "serve_turn"]

        assert len(serve_records) == 1
        record = serve_records[0]
        assert record.outcome == "ok"
        assert record.raw_output == result.text == "base reply"
        assert record.parsed == result.diagnostics
        assert result.diagnostics["exit_via"] == "base_model"

    def test_raising_dispatch_records_failed_outcome_and_reraises(self):
        """A dispatch that raises still leaves a ``failed`` phase record
        behind (the exception propagates from inside
        ``with extraction_trace(), phase_trace(...):``), and the original
        exception is re-raised rather than swallowed."""
        config = ServerConfig()
        router = MagicMock()
        router.route.side_effect = RuntimeError("boom")

        with extraction_trace() as trace:
            with pytest.raises(RuntimeError, match="boom"):
                handle_chat(
                    text="hello",
                    conversation_id="c1",
                    speaker=None,
                    speaker_id="speaker0",
                    history=None,
                    model=MagicMock(),
                    tokenizer=MagicMock(),
                    config=config,
                    router=router,
                )
            serve_records = [r for r in trace.records if r.name == "serve_turn"]

        assert len(serve_records) == 1
        assert serve_records[0].outcome == "failed"
        assert "boom" in (serve_records[0].reason or "")


class TestIsResidualUnconditional:
    """``is_residual`` is computed regardless of ``config.debug`` — a
    regression pin for the fix that moved the computation out of the
    debug-gated log block."""

    def test_computed_true_with_debug_off_when_plan_has_no_signal(self):
        config = ServerConfig()
        config.debug = False
        router = MagicMock()
        router.route.return_value = RoutingPlan(
            strategy="direct", intent=Intent.GENERAL, steps=[], ha_domains=[]
        )

        with patch(
            "paramem.server.inference._base_model_answer",
            return_value=ChatResult(text="base reply"),
        ):
            result = handle_chat(
                text="What's the weather like generally?",
                conversation_id="c1",
                speaker=None,
                speaker_id="speaker0",
                history=None,
                model=MagicMock(),
                tokenizer=MagicMock(),
                config=config,
                router=router,
            )

        assert result.diagnostics["is_residual"] is True


class TestRoutingBranchDiagnostics:
    """Every routing branch stamps the handle_chat-level diagnostics keys;
    only the personal-probe leg additionally carries the probe/temporal
    keys ``_probe_and_reason`` builds."""

    def test_personal_probe_branch(self):
        router = MagicMock()
        router.route.return_value = RoutingPlan(
            strategy="targeted_probe",
            intent=Intent.PERSONAL,
            steps=[RoutingStep(adapter_name="episodic", keys_to_probe=["graph0001"])],
        )
        config = ServerConfig()
        probe_diagnostics = {
            "temporal": None,
            "probes": {"episodic": {"probed": 1, "recalled": 1}},
            "facts_recalled": 1,
        }

        with patch(
            "paramem.server.inference._probe_and_reason",
            return_value=ChatResult(text="probe reply", diagnostics=dict(probe_diagnostics)),
        ):
            result = handle_chat(
                text="What do I like?",
                conversation_id="c1",
                speaker=None,
                speaker_id="speaker0",
                history=None,
                model=MagicMock(),
                tokenizer=MagicMock(),
                config=config,
                router=router,
                memory_store=_MS(replay_enabled=False),
            )

        assert result.diagnostics["exit_via"] == "personal_probe"
        assert _HANDLE_CHAT_KEYS <= result.diagnostics.keys()
        assert result.diagnostics["temporal"] is None
        assert result.diagnostics["probes"] == {"episodic": {"probed": 1, "recalled": 1}}
        assert result.diagnostics["facts_recalled"] == 1

    def test_ha_branch(self):
        router = MagicMock()
        router.route.return_value = RoutingPlan(strategy="direct", intent=Intent.GENERAL)
        ha_client = MagicMock()
        ha_client.conversation_process.return_value = "HA handled it"
        config = ServerConfig()

        result = handle_chat(
            text="turn on the kitchen light",
            conversation_id="c1",
            speaker=None,
            speaker_id="speaker0",
            history=None,
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=config,
            router=router,
            ha_client=ha_client,
        )

        assert result.diagnostics["exit_via"] == "general_ha"
        assert _HANDLE_CHAT_KEYS <= result.diagnostics.keys()
        assert "probes" not in result.diagnostics
        assert "temporal" not in result.diagnostics
        assert "facts_recalled" not in result.diagnostics

    def test_cloud_branch(self):
        router = MagicMock()
        router.route.return_value = RoutingPlan(strategy="direct", intent=Intent.GENERAL)
        config = ServerConfig()

        with patch(
            "paramem.server.inference.answer_via_cloud",
            return_value=ChatResult(text="cloud answer", escalated=True),
        ):
            result = handle_chat(
                text="What's the capital of France?",
                conversation_id="c1",
                speaker=None,
                speaker_id="speaker0",
                history=None,
                model=MagicMock(),
                tokenizer=MagicMock(),
                config=config,
                router=router,
                cloud_agent=MagicMock(),
            )

        assert result.diagnostics["exit_via"] == "general_cloud"
        assert _HANDLE_CHAT_KEYS <= result.diagnostics.keys()
        assert "probes" not in result.diagnostics
        assert "temporal" not in result.diagnostics
        assert "facts_recalled" not in result.diagnostics

    def test_abstention_branch(self):
        """Mirrors tests/test_abstention.py's proven self-referential +
        no-steps setup: PERSONAL intent with an empty plan.steps takes
        the defensive no-probe path, HA/cloud are both absent, and the
        canned response fires before the base model is ever reached."""
        router = MagicMock()
        router.route.return_value = RoutingPlan(strategy="direct", intent=Intent.PERSONAL)
        router._speaker_key_index = {}
        config = ServerConfig()
        assert config.abstention.enabled is True

        with (
            patch("paramem.server.inference.is_self_referential", return_value=True),
            patch("paramem.server.inference._base_model_answer") as mock_base_model,
        ):
            result = handle_chat(
                text="Where do I live?",
                conversation_id="c1",
                speaker=None,
                speaker_id="speaker0",
                history=None,
                model=MagicMock(),
                tokenizer=MagicMock(),
                config=config,
                router=router,
                memory_store=_MS(replay_enabled=False),
            )

        mock_base_model.assert_not_called()
        assert result.diagnostics["exit_via"].startswith("abstention")
        assert _HANDLE_CHAT_KEYS <= result.diagnostics.keys()
        assert "probes" not in result.diagnostics
        assert "temporal" not in result.diagnostics
        assert "facts_recalled" not in result.diagnostics

    def test_base_model_branch(self):
        config = ServerConfig()

        with patch(
            "paramem.server.inference._base_model_answer",
            return_value=ChatResult(text="base reply"),
        ):
            result = handle_chat(
                text="hello there",
                conversation_id="c1",
                speaker=None,
                speaker_id="speaker0",
                history=None,
                model=MagicMock(),
                tokenizer=MagicMock(),
                config=config,
                router=None,
            )

        assert result.diagnostics["exit_via"] == "base_model"
        assert _HANDLE_CHAT_KEYS <= result.diagnostics.keys()
        assert "probes" not in result.diagnostics
        assert "temporal" not in result.diagnostics
        assert "facts_recalled" not in result.diagnostics


class _ProbeAndReasonHelpers:
    """Shared plan/model builders, mirroring
    tests/test_inference_response_shaping.py::_PlanBuilder's pattern."""

    @staticmethod
    def make_plan(steps):
        return RoutingPlan(
            steps=[RoutingStep(adapter_name=a, keys_to_probe=list(k)) for a, k in steps],
            strategy="direct",
            intent=Intent.PERSONAL,
        )

    @staticmethod
    def make_model(adapter_names):
        model = MagicMock()
        model.peft_config = {name: MagicMock() for name in adapter_names}
        return model


class TestProbeAndReasonProbeDiagnostics(_ProbeAndReasonHelpers):
    """Per-adapter probe/recall counts and the recalled-fact total, read
    from the same values the existing log lines render — one derivation,
    two renderings."""

    @staticmethod
    def _stub_probe(monkeypatch, recalled_keys):
        def fake_grouped(model, tokenizer, keys_by_adapter, **kwargs):
            results = {}
            for keys in keys_by_adapter.values():
                for k in keys:
                    if k in recalled_keys:
                        results[k] = {"key": k, "fact_text": f"fact about {k}", "confidence": 1.0}
                    else:
                        results[k] = {"key": k, "failure_reason": "no_match"}
            return results

        monkeypatch.setattr("paramem.memory.probe.probe_keys_grouped_by_adapter", fake_grouped)
        monkeypatch.setattr("paramem.models.loader.switch_adapter", lambda model, name: None)
        monkeypatch.setattr(
            "paramem.memory.store.MemoryStore.read_simhash_registry_from_disk",
            staticmethod(lambda path, cached=False: {}),
        )
        monkeypatch.setattr(
            "paramem.server.inference.is_self_referential", lambda text, **kwargs: False
        )

    def test_probes_and_facts_recalled_match_the_probe_outcome(self, monkeypatch):
        self._stub_probe(monkeypatch, recalled_keys={"e1"})
        monkeypatch.setattr(
            "paramem.server.inference.generate_answer", lambda *a, **kw: "final answer."
        )

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self.make_model(["episodic"])

        config = ServerConfig()
        # Disabled — this test is about probe counts, not the date-group
        # selection stage (covered separately below).
        config.inference.temporal_selection_enabled = False

        result = _probe_and_reason(
            text="What do I like?",
            plan=self.make_plan([("episodic", ["e1", "e2"])]),
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            memory_store=_MS(replay_enabled=False),
        )

        assert result.diagnostics["probes"] == {"episodic": {"probed": 2, "recalled": 1}}
        assert result.diagnostics["facts_recalled"] == 1
        assert result.diagnostics["temporal"] is None

    def test_no_layers_branch_carries_probes_but_not_facts_recalled(self, monkeypatch):
        """Every probed key fails: the ``not layers`` fallback sets
        ``probes`` (probing did happen) but never reaches
        ``facts_recalled`` (no layer was ever assembled)."""
        self._stub_probe(monkeypatch, recalled_keys=set())
        monkeypatch.setattr("paramem.server.inference._escalate_to_ha_agent", lambda *a, **kw: None)
        monkeypatch.setattr("paramem.server.inference.answer_via_cloud", lambda *a, **kw: None)
        monkeypatch.setattr(
            "paramem.server.inference._base_model_answer",
            lambda *a, **kw: ChatResult(text="base fallback"),
        )

        tokenizer = MagicMock()
        model = self.make_model(["episodic"])

        config = ServerConfig()
        config.abstention.enabled = False
        config.inference.temporal_selection_enabled = False

        result = _probe_and_reason(
            text="What do I like?",
            plan=self.make_plan([("episodic", ["e1"])]),
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            memory_store=_MS(replay_enabled=False),
        )

        assert result.diagnostics["probes"] == {"episodic": {"probed": 1, "recalled": 0}}
        assert "facts_recalled" not in result.diagnostics
        assert result.text == "base fallback"


class TestProbeAndReasonTemporalDiagnostics(_ProbeAndReasonHelpers):
    """The date-group selection stage's outcome, mirrored into
    ``result.diagnostics["temporal"]``: ``None`` when the stage did not
    run, a populated dict when it did — including on the zero-survivor
    early return, which never reaches probing (so ``probes`` stays
    absent there)."""

    @staticmethod
    def _stub_probe_ok(monkeypatch):
        monkeypatch.setattr(
            "paramem.memory.probe.probe_keys_grouped_by_adapter",
            lambda model, tokenizer, keys_by_adapter, **kw: {
                k: {"key": k, "fact_text": f"fact about {k}"}
                for keys in keys_by_adapter.values()
                for k in keys
            },
        )
        monkeypatch.setattr("paramem.models.loader.switch_adapter", lambda model, name: None)
        monkeypatch.setattr(
            "paramem.memory.store.MemoryStore.read_simhash_registry_from_disk",
            staticmethod(lambda path, cached=False: {}),
        )
        monkeypatch.setattr(
            "paramem.server.inference.generate_answer", lambda *a, **kw: "final answer."
        )

    def test_temporal_is_none_when_stage_disabled(self, monkeypatch):
        self._stub_probe_ok(monkeypatch)
        monkeypatch.setattr("paramem.server.inference.is_self_referential", lambda *a, **kw: False)

        def exploding_select(*args, **kwargs):
            raise AssertionError("select_date_groups must not be called when the stage is disabled")

        monkeypatch.setattr("paramem.server.inference.select_date_groups", exploding_select)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self.make_model(["episodic"])

        config = ServerConfig()
        config.inference.temporal_selection_enabled = False

        result = _probe_and_reason(
            text="What do I like?",
            plan=self.make_plan([("episodic", ["e1"])]),
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            memory_store=_MS(replay_enabled=False),
        )

        assert result.diagnostics["temporal"] is None

    def test_temporal_dict_populated_when_stage_runs(self, monkeypatch):
        from paramem.server.temporal_selection import DateSelection

        self._stub_probe_ok(monkeypatch)

        def fake_select(text, date_by_key, *, model, tokenizer, config, today):
            return DateSelection(all=True, ranges=(), include_undated=True, fail_open=False)

        monkeypatch.setattr("paramem.server.inference.select_date_groups", fake_select)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self.make_model(["episodic"])

        config = ServerConfig()

        memory_store = _MS(replay_enabled=False)
        memory_store.set_bookkeeping(
            "e1",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="2026-08-01T09:00:00",
            last_seen="2026-08-01T09:00:00",
        )

        result = _probe_and_reason(
            text="What do you know about me?",
            plan=self.make_plan([("episodic", ["e1"])]),
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            memory_store=memory_store,
        )

        assert result.diagnostics["temporal"] == {
            "all": True,
            "ranges": 0,
            "include_undated": True,
            "fail_open": False,
            "keys_before": 1,
            "keys_after": 1,
            "period_note": False,
        }
        assert result.diagnostics["probes"] == {"episodic": {"probed": 1, "recalled": 1}}

    def test_zero_survivor_early_return_carries_temporal_without_probes(self, monkeypatch):
        from paramem.server.temporal import DateWindow
        from paramem.server.temporal_selection import DateSelection

        def exploding_probe(self, *args, **kwargs):
            raise AssertionError("MemoryStore.probe must not be called when zero keys survive")

        monkeypatch.setattr("paramem.memory.store.MemoryStore.probe", exploding_probe)
        monkeypatch.setattr(
            "paramem.server.inference._generate_local_reply",
            lambda *a, **kw: ("Nothing to add.", False),
        )

        def fake_select(text, date_by_key, *, model, tokenizer, config, today):
            return DateSelection(
                all=False,
                ranges=(DateWindow(start=date(2026, 8, 10), end=date(2026, 8, 10)),),
                include_undated=False,
            )

        monkeypatch.setattr("paramem.server.inference.select_date_groups", fake_select)

        tokenizer = MagicMock()
        model = self.make_model(["episodic"])

        config = ServerConfig()

        memory_store = _MS(replay_enabled=False)
        memory_store.set_bookkeeping(
            "e1",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="2026-08-01T09:00:00",
            last_seen="2026-08-01T09:00:00",
        )

        result = _probe_and_reason(
            text="What did we discuss on August 10th?",
            plan=self.make_plan([("episodic", ["e1"])]),
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            memory_store=memory_store,
        )

        assert result.diagnostics["temporal"]["keys_after"] == 0
        assert result.diagnostics["temporal"]["keys_before"] == 1
        assert result.diagnostics["temporal"]["period_note"] is True
        assert "probes" not in result.diagnostics
        assert "facts_recalled" not in result.diagnostics
        assert result.text == "Nothing to add."
