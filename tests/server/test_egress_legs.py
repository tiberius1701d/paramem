"""External egress: ``paramem.server.egress`` — the per-leg record and its
stamping rule, the ``cloud_permitted=False`` refusal ordering, the probe
door's forced-routing facility, ``ChatRequest``'s own lack of a route
field, and the one-way module boundary between egress and local routing.

CPU-only: no model, no GPU, no network.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from paramem.cloud.anonymize import AnonymizedContract, failed_contract
from paramem.cloud.providers.base import CloudAgent
from paramem.config.taxonomy import ScrubCategory
from paramem.server.chat_result import ChatResult
from paramem.server.config import HAToolsConfig, TextLangDetectionConfig
from paramem.server.egress import (
    _EGRESS_VALUES,
    _REFUSAL_VALUES,
    LEG_NAMES,
    OutboundText,
    _refuse_failed_contract,
    _stamp_leg,
    answer_via_cloud,
    answer_via_ha,
)

# ---------------------------------------------------------------------------
# Per-leg record
# ---------------------------------------------------------------------------


class TestLegRecord:
    def test_a_leg_carries_at_most_one_of_its_two_keys(self):
        diagnostics: dict = {"cloud_egress": "verbatim"}

        _stamp_leg(diagnostics, "ha", egress="scrubbed")
        assert diagnostics == {"cloud_egress": "verbatim", "ha_egress": "scrubbed"}

        _stamp_leg(diagnostics, "ha", refusal="guard")
        assert diagnostics == {"cloud_egress": "verbatim", "ha_refusal": "guard"}

    def test_stamp_leg_rejects_an_unknown_leg_or_value(self):
        assert LEG_NAMES == ("cloud", "ha")
        with pytest.raises(ValueError):
            _stamp_leg({}, "voice", egress="scrubbed")
        with pytest.raises(ValueError):
            _stamp_leg({}, "ha", egress="not-a-real-value")
        assert "not-a-real-value" not in _EGRESS_VALUES
        with pytest.raises(ValueError):
            _stamp_leg({}, "ha", refusal="not-a-real-cause")
        assert "not-a-real-cause" not in _REFUSAL_VALUES
        with pytest.raises(ValueError):
            _stamp_leg({}, "ha")  # neither given
        with pytest.raises(ValueError):
            _stamp_leg({}, "ha", egress="scrubbed", refusal="guard")  # both given


# ---------------------------------------------------------------------------
# cloud_permitted=False refuses ahead of every other policy read
# ---------------------------------------------------------------------------


class _ExplodingSanitization:
    @property
    def cloud_mode(self):
        raise AssertionError("cloud_mode must not be read before the not_permitted refusal")


class _CloudModeGuardConfig:
    sanitization = _ExplodingSanitization()


class TestCloudPermittedFalseRefusesBeforeAnyProviderCall:
    """``cloud_permitted=False`` refuses before ``cloud_mode`` is ever read
    or a provider called — a config whose ``sanitization.cloud_mode``
    raises on access proves the ordering directly: reading it before the
    refusal would fail this test, not merely leave it unexercised."""

    def test_refuses_before_reading_cloud_mode_or_calling_the_provider(self):
        cloud_agent = MagicMock(spec=CloudAgent)
        outbound = OutboundText(
            "What's the population of Berlin?", _CloudModeGuardConfig(), diagnostics={}
        )

        result = answer_via_cloud(outbound, cloud_agent, cloud_permitted=False)

        assert result is None
        cloud_agent.call.assert_not_called()


# ---------------------------------------------------------------------------
# The HA leg — verbatim, no contract, no restore
# ---------------------------------------------------------------------------


def _ha_config(*, ha_agent_id: str = "agent.ha") -> SimpleNamespace:
    return SimpleNamespace(
        ha_agent_id=ha_agent_id,
        tools=SimpleNamespace(ha=SimpleNamespace(supported_languages=["en"])),
    )


class TestAnswerViaHaCarriesTheTurnVerbatim:
    def test_sends_the_text_unchanged_and_returns_the_reply_as_it_came(self, monkeypatch):
        def _explode(*args, **kwargs):
            raise AssertionError("the HA leg must never build an anonymize contract")

        monkeypatch.setattr("paramem.graph.flows.anonymize_turn", _explode)

        ha_client = MagicMock()
        ha_client.conversation_process.return_value = "the reply, verbatim"
        outbound = OutboundText("My name is Alex.", _ha_config(), diagnostics={}, language="en")

        result = answer_via_ha(outbound, ha_client)

        ha_client.conversation_process.assert_called_once_with(
            "My name is Alex.",
            agent_id="agent.ha",
            language="en",
            supported_languages=["en"],
        )
        assert result.text == "the reply, verbatim"
        assert result.escalated is True

    def test_stamps_verbatim_egress_and_never_a_refusal(self):
        outbound = OutboundText("hi", _ha_config(), diagnostics={})
        ha_client = MagicMock()
        ha_client.conversation_process.return_value = "ok"

        answer_via_ha(outbound, ha_client)

        assert outbound.diagnostics == {"ha_egress": "verbatim"}

    def test_no_ha_client_or_no_agent_id_returns_none_and_stamps_nothing(self):
        outbound = OutboundText("hi", _ha_config(ha_agent_id=""), diagnostics={})
        assert answer_via_ha(outbound, MagicMock()) is None
        assert outbound.diagnostics == {}

        outbound2 = OutboundText("hi", _ha_config(), diagnostics={})
        assert answer_via_ha(outbound2, None) is None
        assert outbound2.diagnostics == {}

    def test_a_none_reply_falls_through_without_a_result(self):
        ha_client = MagicMock()
        ha_client.conversation_process.return_value = None
        outbound = OutboundText("hi", _ha_config(), diagnostics={})
        assert answer_via_ha(outbound, ha_client) is None

    def test_a_personal_verdict_never_closes_the_ha_leg(self):
        """Unlike the cloud leg's ``personal_blocked`` refusal, the HA leg
        is never gated on ``is_personal`` — function over privacy on this
        leg, by design."""
        ha_client = MagicMock()
        ha_client.conversation_process.return_value = "ok"
        outbound = OutboundText("My phone number is 555-0100.", _ha_config(), diagnostics={})
        outbound.is_personal = True

        result = answer_via_ha(outbound, ha_client)

        assert result is not None
        assert result.text == "ok"


# ---------------------------------------------------------------------------
# The cloud leg under anonymize — contract build, substitute, restore, and
# the one-to-one failure -> refusal mapping.
# ---------------------------------------------------------------------------


def _anonymize_config(*, cloud_mode: str = "anonymize") -> SimpleNamespace:
    return SimpleNamespace(
        sanitization=SimpleNamespace(
            cloud_mode=cloud_mode, scrub_categories=(ScrubCategory("Person"),)
        ),
        consolidation=SimpleNamespace(extraction_anonymize_token_envelope=8192),
    )


def _ok_contract(forward: dict[str, str]) -> AnonymizedContract:
    reverse = {v: k for k, v in forward.items()}
    return AnonymizedContract(
        status="ok",
        forward=forward,
        reverse=reverse,
        anon_transcript="",
        declared=frozenset(reverse.keys()),
        rekey_dropped=0,
        raw="{}",
        failure=None,
        facts=[],
        model_calls=1,
    )


class TestAnswerViaCloudUnderAnonymize:
    def test_builds_the_contract_substitutes_and_restores_the_reply(self, monkeypatch):
        contract = _ok_contract({"Alex": "Person_1"})
        monkeypatch.setattr("paramem.graph.flows.anonymize_turn", lambda *a, **k: contract)

        cloud_agent = MagicMock(spec=CloudAgent)
        cloud_agent.call.return_value = SimpleNamespace(text="Nice to meet you, Person_1!")

        outbound = OutboundText(
            "Alex said hi.",
            _anonymize_config(),
            diagnostics={},
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

        result = answer_via_cloud(outbound, cloud_agent)

        sent_query = cloud_agent.call.call_args.kwargs["query"]
        assert sent_query == "Person_1 said hi."
        assert result.text == "Nice to meet you, Alex!"
        assert outbound.diagnostics["cloud_egress"] == "scrubbed"

    def test_forward_empty_stamps_verbatim_egress(self, monkeypatch):
        contract = _ok_contract({})
        monkeypatch.setattr("paramem.graph.flows.anonymize_turn", lambda *a, **k: contract)
        cloud_agent = MagicMock(spec=CloudAgent)
        cloud_agent.call.return_value = SimpleNamespace(text="ok")
        outbound = OutboundText(
            "Nothing personal here.",
            _anonymize_config(),
            diagnostics={},
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

        answer_via_cloud(outbound, cloud_agent)

        assert outbound.diagnostics["cloud_egress"] == "verbatim"

    @pytest.mark.parametrize("failure", ["guard", "model_unavailable", "scan_failed"])
    def test_each_contract_failure_maps_one_to_one_onto_its_own_refusal_stamp(
        self, monkeypatch, failure
    ):
        contract = failed_contract(failure=failure)
        monkeypatch.setattr("paramem.graph.flows.anonymize_turn", lambda *a, **k: contract)
        cloud_agent = MagicMock(spec=CloudAgent)
        outbound = OutboundText(
            "Alex said hi.",
            _anonymize_config(),
            diagnostics={},
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

        result = answer_via_cloud(outbound, cloud_agent)

        assert result is None
        assert outbound.diagnostics["cloud_refusal"] == failure
        cloud_agent.call.assert_not_called()


class TestRefuseFailedContractRaisesOnAnUnrecognisedFailure:
    def test_an_unknown_failure_value_raises_runtime_error(self):
        class _FakeContract:
            failure = "not-a-real-failure"

        class _StubOutbound:
            diagnostics: dict = {}

            def contract(self):
                return _FakeContract()

        with pytest.raises(RuntimeError):
            _refuse_failed_contract(_StubOutbound(), "cloud")


class TestRefusalValuesClosedVocabulary:
    def test_the_six_values_are_exactly_this_set(self):
        assert set(_REFUSAL_VALUES) == {
            "not_permitted",
            "personal_blocked",
            "guard",
            "model_unavailable",
            "scan_failed",
            "unresolved_placeholder",
        }


class TestOutboundHistorySubstitutesTheForwardTable:
    def test_a_kept_value_is_placeholdered_in_every_history_turn(self, monkeypatch) -> None:
        contract = _ok_contract({"Alex": "Person_1"})
        monkeypatch.setattr("paramem.graph.flows.anonymize_turn", lambda *a, **k: contract)
        outbound = OutboundText(
            "hi",
            _anonymize_config(),
            diagnostics={},
            model=MagicMock(),
            tokenizer=MagicMock(),
            history=[{"role": "user", "text": "Alex called earlier."}],
        )

        history = outbound.outbound_history()

        assert history == [{"role": "user", "text": "Person_1 called earlier."}]


class TestOutboundTextSurface:
    def test_outbound_text_takes_no_retain_argument(self):
        assert "retain" not in inspect.signature(OutboundText.outbound_text).parameters

    def test_contract_is_memoised_once_per_object(self, monkeypatch):
        calls: list[int] = []

        def _counting_anonymize_turn(*args, **kwargs):
            calls.append(1)
            return _ok_contract({})

        monkeypatch.setattr("paramem.graph.flows.anonymize_turn", _counting_anonymize_turn)
        outbound = OutboundText(
            "hello",
            _anonymize_config(),
            diagnostics={},
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

        outbound.contract()
        outbound.contract()
        outbound.outbound_text()

        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


class TestHaToolsConfigConfigured:
    @pytest.mark.parametrize(
        "url, token, expected",
        [
            ("http://ha.local:8123", "tok", True),
            ("", "tok", False),
            ("http://ha.local:8123", "", False),
            ("", "", False),
        ],
    )
    def test_configured_truth_table(self, url, token, expected):
        assert HAToolsConfig(url=url, token=token).configured is expected


# ---------------------------------------------------------------------------
# Probe door
# ---------------------------------------------------------------------------


def _probe_state(*, mode: str = "local", cloud_providers: dict | None = None) -> dict:
    store = MagicMock()
    store.get_name.return_value = "TestUser"
    store.resolve_speaker_name.return_value = "TestUser"
    cfg = MagicMock()
    cfg.debug = True
    cfg.text_lang_detection = TextLangDetectionConfig(enabled=False)
    cfg.consolidation.abort_quiesce_timeout_s = 1.0
    session_buffer = MagicMock()
    session_buffer.get_conversation_turns.return_value = []
    return {
        "config": cfg,
        "mode": mode,
        "speaker_store": store,
        "session_buffer": session_buffer,
        "router": MagicMock(),
        "ha_client": None,
        "cloud_agent": MagicMock(name="default_cloud_agent"),
        "cloud_providers": cloud_providers or {},
        "model": MagicMock(),
        "tokenizer": MagicMock(),
        "memory_store": MagicMock(),
        "background_trainer": None,
        "effective_mode": None,
        "last_model_use_monotonic": 0.0,
        "user_token_store": None,
        "ha_graph": None,
    }


class TestDebugProbeRoute:
    def test_route_ha_uses_the_ha_leg_only(self, monkeypatch):
        import paramem.server.app as app_module

        state = _probe_state(mode="local")
        monkeypatch.setattr(app_module, "_state", state)

        with patch(
            "paramem.server.app.handle_chat", return_value=ChatResult(text="ok")
        ) as mock_handle:
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post(
                "/debug/probe", json={"text": "hi", "speaker_id": "spk-1", "route": "ha"}
            )

        assert resp.status_code == 200
        assert mock_handle.call_args.kwargs["forced_leg"] == "ha"
        # The forced leg alone closes the cloud leg — the agent object
        # itself is not nulled as a second mechanism.
        assert mock_handle.call_args.kwargs["cloud_agent"] is state["cloud_agent"]

    def test_route_cloud_uses_the_cloud_leg_only(self, monkeypatch):
        import paramem.server.app as app_module

        state = _probe_state(mode="local")
        monkeypatch.setattr(app_module, "_state", state)

        with patch(
            "paramem.server.app.handle_chat", return_value=ChatResult(text="ok")
        ) as mock_handle:
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post(
                "/debug/probe", json={"text": "hi", "speaker_id": "spk-1", "route": "cloud"}
            )

        assert resp.status_code == 200
        assert mock_handle.call_args.kwargs["forced_leg"] == "cloud"
        assert mock_handle.call_args.kwargs["cloud_agent"] is state["cloud_agent"]

    def test_route_cloud_provider_selects_that_provider_agent(self, monkeypatch):
        import paramem.server.app as app_module

        provider_agent = MagicMock(name="anthropic_agent")
        state = _probe_state(mode="local", cloud_providers={"anthropic": provider_agent})
        monkeypatch.setattr(app_module, "_state", state)

        with patch(
            "paramem.server.app.handle_chat", return_value=ChatResult(text="ok")
        ) as mock_handle:
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post(
                "/debug/probe",
                json={"text": "hi", "speaker_id": "spk-1", "route": "cloud:anthropic"},
            )

        assert resp.status_code == 200
        assert mock_handle.call_args.kwargs["forced_leg"] == "cloud"
        assert mock_handle.call_args.kwargs["cloud_agent"] is provider_agent

    def test_unknown_provider_404(self, monkeypatch):
        import paramem.server.app as app_module

        state = _probe_state(mode="local", cloud_providers={})
        monkeypatch.setattr(app_module, "_state", state)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post(
            "/debug/probe",
            json={"text": "hi", "speaker_id": "spk-1", "route": "cloud:bogus"},
        )

        assert resp.status_code == 404
        assert resp.json()["detail"]["status"] == "provider_not_configured"

    def test_unknown_route_value_422(self, monkeypatch):
        import paramem.server.app as app_module

        state = _probe_state(mode="local")
        monkeypatch.setattr(app_module, "_state", state)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post(
            "/debug/probe",
            json={"text": "hi", "speaker_id": "spk-1", "route": "teleport"},
        )

        assert resp.status_code == 422
        assert resp.json()["detail"]["status"] == "unknown_route"

    def test_probe_still_appends_nothing_to_the_session_buffer(self, monkeypatch):
        import paramem.server.app as app_module

        state = _probe_state(mode="local")
        monkeypatch.setattr(app_module, "_state", state)

        with patch("paramem.server.app.handle_chat", return_value=ChatResult(text="ok")):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/debug/probe", json={"text": "hi", "speaker_id": "spk-1"})

        assert resp.status_code == 200
        state["session_buffer"].append.assert_not_called()


class TestChatRequestHasNoRoute:
    def test_route_in_the_body_is_ignored(self):
        from paramem.server.app import ChatRequest

        assert "route" not in ChatRequest.model_fields

        request = ChatRequest(text="hi", route="cloud")

        assert not hasattr(request, "route")
        assert request.model_dump() == {
            "text": "hi",
            "conversation_id": "default",
            "speaker_embedding": None,
        }


# ---------------------------------------------------------------------------
# One-way import boundary
# ---------------------------------------------------------------------------


class TestImportBoundary:
    def test_egress_module_imports_nothing_from_inference(self):
        import paramem.server.egress as egress_module

        tree = ast.parse(Path(egress_module.__file__).read_text())
        offending: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "paramem.server.inference" in node.module
            ):
                offending.append(node.module)
            elif isinstance(node, ast.Import):
                offending.extend(
                    alias.name for alias in node.names if "paramem.server.inference" in alias.name
                )
        assert offending == []

    def test_inference_module_imports_from_egress_confirming_the_direction(self):
        import paramem.server.inference as inference_module

        tree = ast.parse(Path(inference_module.__file__).read_text())
        found = any(
            isinstance(node, ast.ImportFrom) and node.module == "paramem.server.egress"
            for node in ast.walk(tree)
        )
        assert found
