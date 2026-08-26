"""External egress: ``paramem.server.egress`` — one primitive
(:class:`OutboundText`), two doors (:func:`answer_via_ha`,
:func:`answer_via_cloud`), the per-leg record and its stamping rule, the HA
retain filter, the admission composition that gates the span tagger's load,
the probe door's forced-routing facility, and the one-way module boundary
between egress and local routing.

Covered:

* The HA leg's own policy — always reachable, scrubs always, never closed
  by a personal verdict, refuses only on its own three causes.
* The retain filter: an HA-registered entity/area name stays readable in
  the HA-bound payload while every other detected span is scrubbed; a key
  occurring outside a retained span is substituted everywhere; a
  fuzzy-only index match retains nothing; no entity graph retains nothing.
* ``HAEntityGraph.retained_spans`` directly — case-insensitive literal
  matching, overlap merging, and the exact-offset invariant.
* The reply exit gate — a declared placeholder in the reply resolves back;
  a declared-but-unobserved token in the reply refuses the whole reply.
* The one anonymize-chain call per :class:`OutboundText`, including the
  forwarded query behind ``[ESCALATE]``, which is a distinct artifact and
  gets its own object.
* The per-leg record: both legs may carry a key on one turn, a leg carries
  at most one of its own two keys, and the stamp write itself is validated
  against the closed vocabulary.
* ``scrubbing_reachable``'s HA term and ``HAToolsConfig.configured``.
* The probe door (``/debug/probe``'s ``route``) and ``ChatRequest``'s own
  lack of a route field.
* The relay leg's HA-answered result carrying ``ha_egress``, and the
  egress record being stamped at the send boundary rather than on
  transport success.
* The one-way import boundary between ``egress`` and ``inference``.

CPU-only: no model, no GPU, no network. The span tagger is stubbed by
scanning whatever text is actually handed to it, so the same stub serves
every distinct payload (turn text, forwarded query, or history) an
:class:`OutboundText` assembles, mirroring
``tests/test_anonymize_chain.py``'s ``_install_tag`` pattern.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from paramem.cloud import span_tagger
from paramem.cloud.admission import scrubbing_reachable
from paramem.cloud.anonymize import failed_contract
from paramem.cloud.providers.base import CloudAgent, CloudResponse
from paramem.cloud.span_tagger import TaggedSpan, TaggerUnavailable, TagResult
from paramem.config.taxonomy import ScrubCategory
from paramem.server.chat_result import ChatResult
from paramem.server.config import HAToolsConfig, TextLangDetectionConfig
from paramem.server.egress import (
    _EGRESS_VALUES,
    _REFUSAL_VALUES,
    LEG_NAMES,
    OutboundText,
    _stamp_leg,
    answer_via_cloud,
    answer_via_ha,
)
from paramem.server.ha_graph import HAEntityGraph
from paramem.server.incidents import read_incidents
from paramem.server.tools.ha_client import HAClient
from paramem.training.stage_ledger import data_state_dir

PERSON = ScrubCategory(
    name="Person", prefix="Person", hints=("person name",), tagger_labels=("person",)
)


def _install_tag(monkeypatch, terms: list[tuple[str, str, float]]) -> None:
    """Stub the span tagger to report *terms* wherever they literally occur
    in whatever text is handed to it.

    Scanning the ACTUAL text at call time (rather than pre-baking a fixed
    span tuple against one precomputed payload) is what lets one
    installation serve every distinct text an :class:`OutboundText` may
    assemble in a test — a turn, a forwarded query, or a text that
    contains none of *terms* at all.
    """

    def _fake_tag(text, labels):
        spans: list[TaggedSpan] = []
        for value, label, score in terms:
            pos = 0
            while True:
                idx = text.find(value, pos)
                if idx == -1:
                    break
                spans.append(
                    TaggedSpan(
                        start=idx, end=idx + len(value), text=value, label=label, score=score
                    )
                )
                pos = idx + 1
        return TagResult(spans=tuple(spans), windows=1)

    monkeypatch.setattr(span_tagger, "tag", _fake_tag)


def _config(
    *,
    cloud_mode: str = "block",
    ha_agent_id: str = "conversation.test_agent",
    data_dir: Path | None = None,
) -> MagicMock:
    config = MagicMock()
    config.sanitization.cloud_mode = cloud_mode
    config.sanitization.scrub_categories = (PERSON,)
    config.consolidation.extraction_anonymize_token_envelope = 8192
    config.ha_agent_id = ha_agent_id
    config.tools.ha.supported_languages = []
    config.personal_referent = None
    config.paths.data = data_dir if data_dir is not None else Path("/tmp/paramem-egress-legs-test")
    return config


def _outbound(text: str, config, *, diagnostics: dict | None = None, **kwargs) -> OutboundText:
    return OutboundText(
        text, config, diagnostics=diagnostics if diagnostics is not None else {}, **kwargs
    )


def _ha_client(reply: str | None) -> MagicMock:
    client = MagicMock(spec=HAClient)
    client.conversation_process.return_value = reply
    return client


def _ha_graph(*friendly_names: str) -> HAEntityGraph:
    states = [
        {"entity_id": f"light.entity{i}", "attributes": {"friendly_name": name}}
        for i, name in enumerate(friendly_names)
    ]
    return HAEntityGraph.build(states)


# ---------------------------------------------------------------------------
# HA leg reachability and policy
# ---------------------------------------------------------------------------


class TestHaLegPolicy:
    def test_personal_turn_reaches_ha_under_cloud_mode_block(self, monkeypatch):
        """The HA leg is never closed by a personal verdict, whatever
        ``cloud_mode`` is configured to. The cloud agent is not touched at
        all — the HA leg needs no cloud interaction to answer."""
        _install_tag(monkeypatch, [("Alex", "person", 0.9)])
        config = _config(cloud_mode="block")
        text = "My name is Alex, turn off the lights"
        diagnostics: dict = {}
        outbound = _outbound(text, config, diagnostics=diagnostics, is_personal=True)
        ha_client = _ha_client("Lights off.")
        cloud_agent = MagicMock(spec=CloudAgent)

        result = answer_via_ha(outbound, ha_client)

        assert result is not None
        assert result.text == "Lights off."
        assert diagnostics == {"ha_egress": "scrubbed"}
        cloud_agent.call.assert_not_called()

    @pytest.mark.parametrize(
        "ha_client, ha_agent_id",
        [(None, "conversation.test_agent"), ("mock", "")],
        ids=["no_client", "empty_agent_id"],
    )
    def test_no_agent_id_writes_no_record_and_runs_no_anonymize(
        self, monkeypatch, ha_client, ha_agent_id
    ):
        """Either half of ``ha_client is None or not ha_agent_id`` means
        there was no leg to refuse: nothing is written to diagnostics and
        the anonymize chain never runs."""
        spy = MagicMock()
        monkeypatch.setattr("paramem.graph.flows.anonymize_turn", spy)
        config = _config(ha_agent_id=ha_agent_id)
        diagnostics: dict = {}
        outbound = _outbound("hi there", config, diagnostics=diagnostics)
        client = MagicMock(spec=HAClient) if ha_client == "mock" else None

        result = answer_via_ha(outbound, client)

        assert result is None
        assert diagnostics == {}
        spy.assert_not_called()


# ---------------------------------------------------------------------------
# Payload scrubbing and HA-registered-name retention
# ---------------------------------------------------------------------------


class TestHaRetainedSurfaces:
    def test_registered_entity_name_reaches_ha_while_a_person_is_scrubbed(self, monkeypatch):
        _install_tag(monkeypatch, [("Alex", "person", 0.9), ("Mira", "person", 0.9)])
        config = _config()
        text = "Turn on Alex's lamp and text Mira."
        ha_graph = _ha_graph("Alex's Lamp")
        outbound = _outbound(text, config)
        ha_client = _ha_client("OK, done.")

        answer_via_ha(outbound, ha_client, ha_graph=ha_graph)

        sent = ha_client.conversation_process.call_args.args[0]
        assert "Alex's lamp" in sent
        assert "Mira" not in sent

    def test_a_key_occurring_outside_a_retained_span_is_still_substituted(self, monkeypatch):
        _install_tag(monkeypatch, [("Alex", "person", 0.9)])
        config = _config()
        text = "Turn on Alex's lamp. Alex likes jazz."
        ha_graph = _ha_graph("Alex's Lamp")
        outbound = _outbound(text, config)
        ha_client = _ha_client("OK, done.")

        answer_via_ha(outbound, ha_client, ha_graph=ha_graph)

        sent = ha_client.conversation_process.call_args.args[0]
        assert "Alex" not in sent

    def test_fuzzy_only_index_match_retains_nothing(self, monkeypatch):
        """The registered name only fuzzy-matches this text (no space in
        ``livingroom``); ``retained_spans`` is literal-only, so it finds no
        occurrence and nothing is retained."""
        _install_tag(monkeypatch, [("Alex", "person", 0.9)])
        config = _config()
        text = "Turn on the livingroom lamp and my name is Alex."
        ha_graph = _ha_graph("Living Room Lamp")
        assert ha_graph.retained_spans(text) == ()
        outbound = _outbound(text, config)
        ha_client = _ha_client("OK, done.")

        answer_via_ha(outbound, ha_client, ha_graph=ha_graph)

        sent = ha_client.conversation_process.call_args.args[0]
        assert "Alex" not in sent

    def test_no_ha_graph_retains_nothing(self, monkeypatch):
        _install_tag(monkeypatch, [("Alex", "person", 0.9)])
        config = _config()
        text = "My name is Alex."
        diagnostics: dict = {}
        outbound = _outbound(text, config, diagnostics=diagnostics)
        ha_client = _ha_client("Hi there.")

        answer_via_ha(outbound, ha_client, ha_graph=None)

        sent = ha_client.conversation_process.call_args.args[0]
        assert "Alex" not in sent
        assert diagnostics == {"ha_egress": "scrubbed"}


class TestRetainedSpans:
    def test_case_insensitive_literal_occurrences_merge_overlaps(self):
        graph = _ha_graph("Living Room Lamp")
        # A second entity contributes the shorter, overlapping area name to
        # the index.
        graph.refresh(
            [
                {
                    "entity_id": "light.living_room_lamp",
                    "attributes": {"friendly_name": "Living Room Lamp"},
                },
                {
                    "entity_id": "sensor.living_room_marker",
                    "attributes": {"friendly_name": "Room Sensor", "area_name": "Living Room"},
                },
            ]
        )
        text = "Turn ON the LIVING ROOM Lamp please"

        spans = graph.retained_spans(text)

        assert len(spans) == 1
        start, end = spans[0]
        assert text[start:end] == "LIVING ROOM Lamp"


# ---------------------------------------------------------------------------
# Reply exit gate
# ---------------------------------------------------------------------------


class TestHaReplyExitGate:
    def test_placeholder_in_the_reply_is_restored(self, monkeypatch):
        _install_tag(monkeypatch, [("Alex", "person", 0.9)])
        config = _config()
        diagnostics: dict = {}
        outbound = _outbound("My name is Alex.", config, diagnostics=diagnostics)
        ha_client = _ha_client("Nice to meet you, Person_1!")

        result = answer_via_ha(outbound, ha_client)

        assert result is not None
        assert result.text == "Nice to meet you, Alex!"
        assert diagnostics == {"ha_egress": "scrubbed"}

    def test_surviving_declared_token_refuses(self, monkeypatch):
        """``Mira`` is retained (an HA-registered name) so its placeholder
        never reaches HA; when the reply nonetheless carries that
        placeholder token, it is declared but unobserved and the whole
        reply is refused."""
        _install_tag(monkeypatch, [("Alex", "person", 0.9), ("Mira", "person", 0.9)])
        config = _config()
        ha_graph = _ha_graph("Mira")
        diagnostics: dict = {}
        outbound = _outbound("My name is Alex and Mira is home.", config, diagnostics=diagnostics)
        ha_client = _ha_client("Got it, Person_2.")

        result = answer_via_ha(outbound, ha_client, ha_graph=ha_graph)

        assert result is None
        assert diagnostics == {"ha_refusal": "unresolved_placeholder"}


# ---------------------------------------------------------------------------
# Refusal causes
# ---------------------------------------------------------------------------


class TestHaRefusalCauses:
    def test_tagger_unavailable_refuses_and_records_the_incident(self, monkeypatch, tmp_path):
        def _raising_tag(text, labels):
            raise TaggerUnavailable("no handle loaded")

        monkeypatch.setattr(span_tagger, "tag", _raising_tag)
        config = _config(data_dir=tmp_path / "data")
        ha_client = _ha_client("should not be reached")

        for _ in range(2):
            diagnostics: dict = {}
            outbound = _outbound("hi there", config, diagnostics=diagnostics)
            result = answer_via_ha(outbound, ha_client)
            assert result is None
            assert diagnostics == {"ha_refusal": "tagger_unavailable"}

        ha_client.conversation_process.assert_not_called()
        incidents = read_incidents(data_state_dir(config.paths.data))
        span_tagger_incidents = [i for i in incidents if i.type == "span_tagger_unavailable"]
        assert len(span_tagger_incidents) == 1, "a second refusal must dedup, not append"
        incident = span_tagger_incidents[0]
        assert incident.count == 2
        # The record side, keyed by leg rather than by wording: the
        # dedup key is the leg itself, so an HA refusal and a cloud
        # refusal never collide on one row.
        assert incident.id == "span_tagger_unavailable:ha"
        assert incident.detail.get("leg") == "ha"

    def test_guard_refuses_without_an_incident(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "paramem.graph.flows.anonymize_turn",
            lambda *a, **k: failed_contract(failure="guard"),
        )
        config = _config(data_dir=tmp_path / "data")
        ha_client = _ha_client("should not be reached")
        diagnostics: dict = {}
        outbound = _outbound("hi there", config, diagnostics=diagnostics)

        result = answer_via_ha(outbound, ha_client)

        assert result is None
        assert diagnostics == {"ha_refusal": "guard"}
        ha_client.conversation_process.assert_not_called()
        incidents = read_incidents(data_state_dir(config.paths.data))
        assert not [i for i in incidents if i.type == "span_tagger_unavailable"]


# ---------------------------------------------------------------------------
# One anonymize call per OutboundText
# ---------------------------------------------------------------------------


class TestOneAnonymizePerOutboundText:
    def test_ha_miss_then_cloud_fallback_scrubs_once(self, monkeypatch):
        from paramem.graph import flows as flows_module

        real_anonymize_turn = flows_module.anonymize_turn
        calls: list[int] = []

        def _counting(*a, **k):
            calls.append(1)
            return real_anonymize_turn(*a, **k)

        monkeypatch.setattr(flows_module, "anonymize_turn", _counting)
        _install_tag(monkeypatch, [("Alex", "person", 0.9)])

        config = _config(cloud_mode="anonymize")
        text = "My name is Alex, what's 2 + 2?"
        outbound = _outbound(text, config)
        ha_client = _ha_client(None)  # HA miss
        cloud_agent = MagicMock(spec=CloudAgent)
        cloud_agent.call.return_value = CloudResponse(text="4")

        ha_result = answer_via_ha(outbound, ha_client)
        assert ha_result is None

        cloud_result = answer_via_cloud(outbound, cloud_agent)
        assert cloud_result is not None

        assert len(calls) == 1
        ha_sent = ha_client.conversation_process.call_args.args[0]
        cloud_sent = cloud_agent.call.call_args.kwargs["query"]
        assert ha_sent == cloud_sent
        assert "Alex" not in ha_sent

    def test_forwarded_query_gets_its_own_contract(self, monkeypatch):
        """The [ESCALATE] hop's forwarded query is a distinct artifact from
        the turn: each gets its own :class:`OutboundText`, so the chain
        runs once for the turn and once more for the forwarded query — the
        query HA receives carries no real person name."""
        from paramem.graph import flows as flows_module
        from paramem.server.inference import _maybe_escalate

        real_anonymize_turn = flows_module.anonymize_turn
        seen_texts: list[str] = []

        def _counting(text, *a, **k):
            seen_texts.append(text)
            return real_anonymize_turn(text, *a, **k)

        monkeypatch.setattr(flows_module, "anonymize_turn", _counting)
        _install_tag(monkeypatch, [("Alex", "person", 0.9)])

        config = _config(cloud_mode="anonymize")
        turn_text = "What's the weather like?"
        forwarded_query = "Turn off Alex's lights"

        turn_diagnostics: dict = {}
        turn_outbound = _outbound(turn_text, config, diagnostics=turn_diagnostics)
        turn_outbound.contract()  # the turn's own contract, already computed upstream

        ha_client = _ha_client("Lights off.")
        response = f"Let me check. [ESCALATE]: {forwarded_query}"

        result = _maybe_escalate(
            response,
            config,
            diagnostics=turn_diagnostics,
            ha_client=ha_client,
            cloud_agent=MagicMock(spec=CloudAgent),
        )

        assert seen_texts == [turn_text, forwarded_query]
        sent = ha_client.conversation_process.call_args.args[0]
        assert "Alex" not in sent
        assert result is not None

    def test_personal_forwarded_query_is_not_sent_to_ha(self, monkeypatch):
        """A forwarded query carrying its own self-referential content
        suppresses the HA door outright — the door is never called, so no
        ``ha_refusal`` is stamped either; the union'd personal verdict then
        closes the cloud leg too under ``cloud_mode: block``."""
        from paramem.server.inference import _maybe_escalate

        _install_tag(monkeypatch, [])
        config = _config(cloud_mode="block")
        ha_client = _ha_client("should never be called")
        cloud_agent = MagicMock(spec=CloudAgent)
        diagnostics: dict = {}
        response = "Let me check. [ESCALATE]: My name is Alex, remember that"

        _maybe_escalate(
            response,
            config,
            diagnostics=diagnostics,
            ha_client=ha_client,
            cloud_agent=cloud_agent,
        )

        ha_client.conversation_process.assert_not_called()
        cloud_agent.call.assert_not_called()
        assert "ha_refusal" not in diagnostics
        assert diagnostics.get("cloud_refusal") == "personal_blocked"


# ---------------------------------------------------------------------------
# cloud_mode governs the cloud door only
# ---------------------------------------------------------------------------


class TestCloudModeMatrixAndHaIndependence:
    """``cloud_mode`` selects the cloud door's outcome over the full
    (mode, personal-verdict) matrix; the HA door's own outcome never
    depends on it at all."""

    @pytest.mark.parametrize(
        "cloud_mode, is_personal, expect",
        [
            ("block", False, "verbatim"),
            ("block", True, "refused"),
            ("anonymize", False, "scrubbed"),
            ("anonymize", True, "scrubbed"),
            ("both", False, "scrubbed"),
            ("both", True, "refused"),
        ],
    )
    def test_cloud_door_matrix(self, monkeypatch, cloud_mode, is_personal, expect):
        _install_tag(monkeypatch, [("Alex", "person", 0.9)])
        config = _config(cloud_mode=cloud_mode)
        text = "My name is Alex, what's the weather?"
        diagnostics: dict = {}
        outbound = _outbound(text, config, diagnostics=diagnostics, is_personal=is_personal)
        cloud_agent = MagicMock(spec=CloudAgent)
        cloud_agent.call.return_value = CloudResponse(text="cloud reply")

        result = answer_via_cloud(outbound, cloud_agent)

        if expect == "refused":
            assert result is None
            cloud_agent.call.assert_not_called()
            assert diagnostics == {"cloud_refusal": "personal_blocked"}
        elif expect == "verbatim":
            assert result is not None
            sent = cloud_agent.call.call_args.kwargs["query"]
            assert sent == text
            assert diagnostics == {"cloud_egress": "verbatim"}
        else:  # scrubbed
            assert result is not None
            sent = cloud_agent.call.call_args.kwargs["query"]
            assert "Alex" not in sent
            assert diagnostics == {"cloud_egress": "scrubbed"}

    @pytest.mark.parametrize("cloud_mode", ["block", "anonymize", "both"])
    @pytest.mark.parametrize("is_personal", [False, True])
    def test_ha_door_outcome_is_independent_of_cloud_mode(
        self, monkeypatch, cloud_mode, is_personal
    ):
        _install_tag(monkeypatch, [("Alex", "person", 0.9)])
        config = _config(cloud_mode=cloud_mode)
        text = "My name is Alex, turn off the lights"
        diagnostics: dict = {}
        outbound = _outbound(text, config, diagnostics=diagnostics, is_personal=is_personal)
        ha_client = _ha_client("Lights off.")

        result = answer_via_ha(outbound, ha_client)

        assert result is not None
        assert result.text == "Lights off."
        assert diagnostics == {"ha_egress": "scrubbed"}


# ---------------------------------------------------------------------------
# Per-leg record
# ---------------------------------------------------------------------------


class TestLegRecord:
    def test_both_legs_record_on_one_turn(self, monkeypatch):
        _install_tag(monkeypatch, [])
        config = _config(cloud_mode="anonymize")
        diagnostics: dict = {}
        outbound = _outbound("Turn on the lights.", config, diagnostics=diagnostics)
        ha_client = _ha_client("Lights on.")
        cloud_agent = MagicMock(spec=CloudAgent)
        cloud_agent.call.return_value = CloudResponse(text="cloud reply")

        answer_via_ha(outbound, ha_client)
        answer_via_cloud(outbound, cloud_agent)

        assert "ha_egress" in diagnostics
        assert "cloud_egress" in diagnostics

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
# Admission
# ---------------------------------------------------------------------------


class TestScrubbingReachableHaTerm:
    def test_ha_only_deployment_is_reachable(self):
        assert (
            scrubbing_reachable(
                scrub_enabled=True,
                cloud_enabled=False,
                cloud_mode="block",
                provider="",
                model="",
                endpoint=None,
                ha_agent_id="conversation.home",
                ha_tools_configured=True,
            )
            is True
        )

    def test_neither_leg_needs_it_is_not_reachable(self):
        assert (
            scrubbing_reachable(
                scrub_enabled=True,
                cloud_enabled=False,
                cloud_mode="block",
                provider="",
                model="",
                endpoint=None,
                ha_agent_id="",
                ha_tools_configured=False,
            )
            is False
        )

    def test_ha_agent_id_without_a_buildable_client_is_not_reachable(self):
        assert (
            scrubbing_reachable(
                scrub_enabled=True,
                cloud_enabled=False,
                cloud_mode="block",
                provider="",
                model="",
                endpoint=None,
                ha_agent_id="conversation.home",
                ha_tools_configured=False,
            )
            is False
        )


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
        "last_chat_monotonic": 0.0,
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
# Relay-leg egress record
# ---------------------------------------------------------------------------


class TestRelayLegEgressRecord:
    def test_relay_leg_ha_answer_carries_ha_egress(self, monkeypatch):
        from paramem.server.app import _relay_route

        _install_tag(monkeypatch, [])
        config = _config()
        ha_client = _ha_client("Lights on.")

        result = _relay_route(
            text="Turn on the lights.",
            history=None,
            config=config,
            cloud_permitted=True,
            ha_client=ha_client,
            cloud_agent=None,
        )

        assert result.text == "Lights on."
        assert result.diagnostics.get("ha_egress") in {"scrubbed", "verbatim"}

    def test_egress_is_stamped_before_transport_so_a_send_failure_still_leaves_the_record(
        self, monkeypatch
    ):
        _install_tag(monkeypatch, [])
        config = _config()
        ha_client = MagicMock(spec=HAClient)
        ha_client.conversation_process.side_effect = RuntimeError("transport boom")
        diagnostics: dict = {}
        outbound = _outbound("Turn on the lights.", config, diagnostics=diagnostics)

        with pytest.raises(RuntimeError, match="transport boom"):
            answer_via_ha(outbound, ha_client)

        assert diagnostics == {"ha_egress": "verbatim"}


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
