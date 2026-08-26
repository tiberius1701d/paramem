"""Tests for the last-resort abstention short-circuit.

When the sanitizer blocks cloud escalation (self-referential / personal
query) AND no local parametric-memory match was found, ``handle_chat``
must return the configured canned response rather than invoking the bare
base model, which would otherwise confabulate plausible-sounding personal
data (observed: untrained adapter + "Where do I live?" → "New York City").
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from peft import PeftModel

from paramem.memory.store import MemoryStore as _MS
from paramem.server.config import (
    AbstentionConfig,
    ServerConfig,
    load_server_config,
)
from tests._serving_door import live_door_config, stub_live_door_probe


class TestAbstentionConfig:
    def test_defaults(self):
        cfg = AbstentionConfig()
        assert cfg.enabled is True
        # The standard, cold-start, and no-identity messages live in
        # configs/prompts/abstention_*.txt; load_*() reads them.
        assert cfg.load_response().strip() == "I don't have that information stored yet."
        assert (
            cfg.load_cold_start_response().strip()
            == "I'm still getting to know you, but I don't have that information yet."
        )
        assert (
            cfg.load_no_identity_response().strip()
            == "I don't know who I'm talking to, so I can't answer that."
        )

    def test_server_config_includes_abstention(self):
        config = ServerConfig()
        assert isinstance(config.abstention, AbstentionConfig)
        assert config.abstention.enabled is True

    def test_yaml_override_via_inline_string(self, tmp_path):
        config_file = tmp_path / "server.yaml"
        config_file.write_text(
            'abstention:\n  enabled: false\n  response_override: "Custom abstention message."\n'
        )
        config = load_server_config(config_file)
        assert config.abstention.enabled is False
        # Override beats file beats fallback.
        assert config.abstention.load_response() == "Custom abstention message."

    def test_yaml_override_cold_start(self, tmp_path):
        config_file = tmp_path / "server.yaml"
        config_file.write_text(
            "abstention:\n  cold_start_response_override: 'Hi! Tell me about yourself.'\n"
        )
        config = load_server_config(config_file)
        assert config.abstention.load_cold_start_response() == "Hi! Tell me about yourself."

    def test_yaml_override_no_identity(self, tmp_path):
        config_file = tmp_path / "server.yaml"
        config_file.write_text("abstention:\n  no_identity_response_override: 'Who is this?'\n")
        config = load_server_config(config_file)
        assert config.abstention.load_no_identity_response() == "Who is this?"

    def test_yaml_partial_override_keeps_file_default(self, tmp_path):
        config_file = tmp_path / "server.yaml"
        config_file.write_text("abstention:\n  enabled: false\n")
        config = load_server_config(config_file)
        assert config.abstention.enabled is False
        # No override + default file path → reads from
        # configs/prompts/abstention_response.txt.
        assert (
            config.abstention.load_response().strip() == "I don't have that information stored yet."
        )

    def test_missing_file_falls_back_to_module_default(self, tmp_path):
        # Point all three files at non-existent paths; the loader must fall
        # back to the module-level constants.
        config_file = tmp_path / "server.yaml"
        config_file.write_text(
            "abstention:\n"
            f"  response_file: '{tmp_path}/missing.txt'\n"
            f"  cold_start_response_file: '{tmp_path}/missing_cold.txt'\n"
            f"  no_identity_response_file: '{tmp_path}/missing_no_identity.txt'\n"
        )
        config = load_server_config(config_file)
        assert config.abstention.load_response() == "I don't have that information stored yet."
        assert (
            config.abstention.load_cold_start_response()
            == "I'm still getting to know you, but I don't have that information yet."
        )
        assert (
            config.abstention.load_no_identity_response()
            == "I don't know who I'm talking to, so I can't answer that."
        )

    @pytest.mark.skipif(
        not Path("configs/server.yaml").exists(),
        reason="operator-local configs/server.yaml absent (CI / fresh clone)",
    )
    def test_project_server_yaml_has_abstention_enabled(self):
        config = load_server_config("configs/server.yaml")
        assert config.abstention.enabled is True
        assert config.abstention.load_response()
        assert config.abstention.load_cold_start_response()
        assert config.abstention.load_no_identity_response()


class TestAbstentionShortCircuit:
    """Verify the short-circuit fires at the correct decision point in
    ``handle_chat`` and does not perturb paths that shouldn't be affected.
    """

    def _make_none_match_router(self, intent=None):
        """Router stub with no PA steps and no HA domains.

        ``intent`` defaults to PERSONAL because every abstention test in
        this class deals with personal-class queries — that is the
        signal that drives abstention.  Pass ``Intent.UNKNOWN`` /
        ``Intent.GENERAL`` when testing non-personal paths.
        """
        from paramem.server.router import Intent, RoutingPlan

        if intent is None:
            intent = Intent.PERSONAL

        router = MagicMock()
        router.route = lambda text, speaker=None, speaker_id=None: RoutingPlan(
            strategy="direct", intent=intent
        )
        router._speaker_key_index = {}
        return router

    def _minimal_mock_model(self):
        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        return model

    def _make_router_with_facts(self, speaker_id: str, intent=None):
        """Router whose _speaker_key_index has at least one key for ``speaker_id``."""
        from paramem.server.router import Intent, RoutingPlan

        if intent is None:
            intent = Intent.PERSONAL

        router = MagicMock()
        router.route = lambda text, speaker=None, speaker_id=None: RoutingPlan(
            strategy="direct", intent=intent
        )
        router._speaker_key_index = {speaker_id: {"graph0001"}}
        return router

    def _make_router_with_steps(self, speaker_id: str, intent=None):
        """Router that yields a non-empty plan.steps so the chat handler
        dispatches into ``_probe_and_reason`` rather than the no-steps
        abstention branch in ``handle_chat``.
        """
        from paramem.server.router import Intent, RoutingPlan, RoutingStep

        if intent is None:
            intent = Intent.PERSONAL

        router = MagicMock()
        router.route = lambda text, speaker=None, speaker_id=None: RoutingPlan(
            strategy="direct",
            intent=intent,
            steps=[
                RoutingStep(adapter_name="episodic", keys_to_probe=["graph0001"]),
            ],
        )
        router._speaker_key_index = {speaker_id: {"graph0001"}}
        return router

    def test_fires_when_sanitizer_blocks_and_no_match(self):
        """Self-referential query + speaker has facts but query missed →
        canned ``response``, never invokes ``_base_model_answer``."""
        from paramem.server.inference import handle_chat

        config = ServerConfig()
        # Explicit defaults for defensive clarity
        assert config.abstention.enabled is True

        with (
            patch(
                "paramem.server.inference.is_self_referential",
                return_value=True,
            ),
            patch("paramem.server.inference._base_model_answer") as mock_base_model,
        ):
            result = handle_chat(
                text="Where do I live?",
                conversation_id="test",
                speaker="Alex",
                history=None,
                model=self._minimal_mock_model(),
                tokenizer=MagicMock(),
                config=config,
                router=self._make_router_with_facts("spk-abc123"),
                speaker_id="spk-abc123",
                memory_store=_MS(),
            )

        assert result.text == config.abstention.load_response()
        mock_base_model.assert_not_called()

    def test_fires_for_anonymous_speaker_with_id(self):
        """speaker_id present but real name absent — fires the cold-start
        variant (per deferred-identity-binding design: id is sufficient for
        attribution, and an anonymous-promoted speaker has no facts yet)."""
        from paramem.server.inference import handle_chat

        config = ServerConfig()

        with (
            patch(
                "paramem.server.inference.is_self_referential",
                return_value=True,
            ),
            patch("paramem.server.inference._base_model_answer") as mock_base_model,
        ):
            result = handle_chat(
                text="What's my birthday?",
                conversation_id="test",
                speaker=None,  # name not yet disclosed
                history=None,
                model=self._minimal_mock_model(),
                tokenizer=MagicMock(),
                config=config,
                router=self._make_none_match_router(),
                speaker_id="spk-anon-42",
                memory_store=_MS(),
            )

        assert result.text == config.abstention.load_cold_start_response()
        mock_base_model.assert_not_called()

    def test_cold_start_when_speaker_has_no_facts(self):
        """Identified speaker but router has no keys for them → return the
        cold_start_response (not the canned ``response``).  This is the
        between-enrollment-and-consolidation state; the canned message
        reads as confused there because the system *can't* know facts
        about a freshly enrolled speaker yet.
        """
        from paramem.server.inference import handle_chat

        config = ServerConfig()

        with (
            patch(
                "paramem.server.inference.is_self_referential",
                return_value=True,
            ),
            patch("paramem.server.inference._base_model_answer") as mock_base_model,
        ):
            result = handle_chat(
                text="What do you know about me?",
                conversation_id="test",
                speaker="Alex",
                history=None,
                model=self._minimal_mock_model(),
                tokenizer=MagicMock(),
                config=config,
                router=self._make_none_match_router(),  # empty _speaker_key_index
                speaker_id="spk-fresh-1",
                memory_store=_MS(),
            )

        assert result.text == config.abstention.load_cold_start_response()
        assert result.text != config.abstention.load_response()  # explicit distinguish
        mock_base_model.assert_not_called()

    def test_self_introduction_falls_through_to_base_model(self):
        """Statement-form personal content (self-introduction, fact-sharing)
        is not a confabulation risk — the user is the source of the facts in
        the same turn — so the abstention must NOT fire even though the
        sanitizer blocks the cloud path. Without this, the user gets
        ``"I don't have that information stored yet."`` in response to
        ``"I'm Alex. I live in Kelkham."`` instead of a conversational
        acknowledgement.
        """
        from paramem.server.chat_result import ChatResult
        from paramem.server.inference import handle_chat

        config = ServerConfig()

        with (
            patch(
                "paramem.server.inference.is_self_referential",
                return_value=True,
            ),
            patch(
                "paramem.server.inference._base_model_answer",
                return_value=ChatResult(text="Nice to meet you, Alex."),
            ) as mock_base_model,
        ):
            result = handle_chat(
                text="I'm Alex. I live in Kelkham with my wife Pat.",
                conversation_id="test",
                speaker=None,
                history=None,
                model=self._minimal_mock_model(),
                tokenizer=MagicMock(),
                config=config,
                router=self._make_none_match_router(),
                speaker_id="spk-anon-1",
                memory_store=_MS(),
            )

        mock_base_model.assert_called_once()
        assert result.text != config.abstention.load_response()

    def test_disabled_falls_through_to_base_model(self):
        """With abstention.enabled=False, ``_base_model_answer`` still runs
        as the last resort."""
        from paramem.server.chat_result import ChatResult
        from paramem.server.inference import handle_chat

        config = ServerConfig()
        config.abstention.enabled = False

        with (
            patch(
                "paramem.server.inference.is_self_referential",
                return_value=True,
            ),
            patch(
                "paramem.server.inference._base_model_answer",
                return_value=ChatResult(text="base model answer"),
            ) as mock_base_model,
        ):
            result = handle_chat(
                text="Where do I live?",
                conversation_id="test",
                speaker="Alex",
                history=None,
                model=self._minimal_mock_model(),
                tokenizer=MagicMock(),
                config=config,
                router=self._make_none_match_router(),
                speaker_id="spk-abc123",
                memory_store=_MS(),
            )

        mock_base_model.assert_called_once()
        assert result.text == "base model answer"

    def test_speakerless_call_raises_value_error(self):
        """The speaker-present contract: handle_chat requires a resolved
        speaker_id.  Speakerless requests never reach here in production —
        the ServingPath boundary in paramem.server.app forks them to the
        relay path — but the contract must fail loud, not silently proceed
        as anonymous, if it is ever violated."""
        from paramem.server.inference import handle_chat

        config = ServerConfig()

        with pytest.raises(ValueError, match="speaker_id"):
            handle_chat(
                text="Where do I live?",
                conversation_id="test",
                speaker=None,
                history=None,
                model=self._minimal_mock_model(),
                tokenizer=MagicMock(),
                config=config,
                router=self._make_none_match_router(),
                speaker_id=None,
                memory_store=_MS(),
            )

    def test_probe_and_reason_disabled_falls_through_to_base_model(self, monkeypatch):
        """With abstention.enabled=False, ``_probe_and_reason``'s
        sanitizer-blocked + no probes + no HA path falls through to the
        base model. Locks the toggle as a real opt-out for both abstention
        sites (handle_chat AND _probe_and_reason)."""
        from paramem.server.chat_result import ChatResult
        from paramem.server.inference import handle_chat

        config = live_door_config()
        config.abstention.enabled = False

        memory_store = _MS()
        memory_store.set_bookkeeping(
            "graph0001",
            speaker_id="spk-abc123",
            relation_type="factual",
            first_seen="2026-08-01T09:00:00",
            last_seen="2026-08-01T09:00:00",
            promoted=False,
        )

        stub_live_door_probe(monkeypatch, failing_keys={"graph0001"})
        import paramem.memory.probe as _probe_mod

        mock_probe = MagicMock(wraps=_probe_mod.probe_keys_grouped_by_adapter)
        monkeypatch.setattr("paramem.memory.probe.probe_keys_grouped_by_adapter", mock_probe)

        with (
            patch(
                "paramem.server.inference.is_self_referential",
                return_value=True,
            ),
            patch(
                "paramem.server.inference._base_model_answer",
                return_value=ChatResult(text="base model answer"),
            ) as mock_base_model,
        ):
            result = handle_chat(
                text="Where do I live?",
                conversation_id="test",
                speaker="Alex",
                history=None,
                model=self._minimal_mock_model(),
                tokenizer=MagicMock(),
                config=config,
                router=self._make_router_with_steps("spk-abc123"),
                speaker_id="spk-abc123",
                memory_store=memory_store,
            )

        # The probe really ran and missed — not "no door was ever
        # consulted", which reaches the same fallback for another reason.
        mock_probe.assert_called_once()
        mock_base_model.assert_called_once()
        assert result.text == "base model answer"

    def test_ha_answer_on_general_query_suppresses_abstention(self):
        """A GENERAL (non-personal) turn the HA door answers returns the
        HA reply directly — ``handle_chat`` returns as soon as the door
        answers, before the abstention gate is ever reached."""
        from paramem.server.chat_result import ChatResult
        from paramem.server.inference import handle_chat
        from paramem.server.router import Intent

        config = ServerConfig()

        with (
            patch("paramem.server.inference.is_self_referential", return_value=False),
            patch(
                "paramem.server.inference.answer_via_ha",
                return_value=ChatResult(text="the lights are on", escalated=True),
            ),
            patch("paramem.server.inference._abstain_if_applicable") as mock_abstain,
        ):
            result = handle_chat(
                text="Are the lights on?",
                conversation_id="test",
                speaker="Alex",
                history=None,
                model=self._minimal_mock_model(),
                tokenizer=MagicMock(),
                config=config,
                router=self._make_none_match_router(intent=Intent.GENERAL),
                cloud_agent=MagicMock(),
                ha_client=MagicMock(),
                speaker_id="spk-abc123",
                memory_store=_MS(),
            )

        mock_abstain.assert_not_called()
        assert result.text == "the lights are on"
        assert result.text != config.abstention.load_response()

    def test_general_query_cloud_outage_falls_to_base_model_not_abstention(self):
        """A GENERAL (non-personal) query with both external legs
        unavailable falls straight to the base model.  Abstention's gate
        requires ``is_personal``, which a non-personal query never
        satisfies, so the canned response never fires regardless of why
        the external legs were unreachable."""
        from paramem.server.chat_result import ChatResult
        from paramem.server.inference import handle_chat
        from paramem.server.router import Intent

        config = ServerConfig()

        with (
            patch("paramem.server.inference.is_self_referential", return_value=False),
            patch("paramem.server.inference.answer_via_ha", return_value=None),
            patch("paramem.server.inference.answer_via_cloud", return_value=None),
            patch(
                "paramem.server.inference._base_model_answer",
                return_value=ChatResult(text="base model answer"),
            ) as mock_base_model,
        ):
            result = handle_chat(
                text="What's the weather like?",
                conversation_id="test",
                speaker="Alex",
                history=None,
                model=self._minimal_mock_model(),
                tokenizer=MagicMock(),
                config=config,
                router=self._make_none_match_router(intent=Intent.GENERAL),
                cloud_agent=MagicMock(),
                ha_client=MagicMock(),
                speaker_id="spk-abc123",
                memory_store=_MS(),
            )

        mock_base_model.assert_called_once()
        assert result.text == "base model answer"
        assert result.text != config.abstention.load_response()

    def test_probe_and_reason_ha_answer_wins_over_abstention(self, monkeypatch):
        """The ``not layers`` branch inside ``_probe_and_reason`` tries the
        HA door before ever consulting abstention — an HA answer returns
        immediately and the canned response is never reached."""
        from paramem.server.chat_result import ChatResult
        from paramem.server.inference import _probe_and_reason
        from paramem.server.router import Intent, RoutingPlan, RoutingStep

        config = live_door_config()
        config.inference.temporal_selection_enabled = False

        memory_store = _MS()
        memory_store.set_bookkeeping(
            "graph0001",
            speaker_id="spk-abc123",
            relation_type="factual",
            first_seen="2026-08-01T09:00:00",
            last_seen="2026-08-01T09:00:00",
            promoted=False,
        )
        stub_live_door_probe(monkeypatch, failing_keys={"graph0001"})

        plan = RoutingPlan(
            strategy="direct",
            intent=Intent.PERSONAL,
            steps=[RoutingStep(adapter_name="episodic", keys_to_probe=["graph0001"])],
        )

        with (
            patch(
                "paramem.server.inference.answer_via_ha",
                return_value=ChatResult(text="the lights are on", escalated=True),
            ) as mock_answer_via_ha,
            patch("paramem.server.inference._abstain_if_applicable") as mock_abstain,
        ):
            result = _probe_and_reason(
                text="Where do I live?",
                plan=plan,
                history=None,
                model=self._minimal_mock_model(),
                tokenizer=MagicMock(),
                config=config,
                memory_store=memory_store,
                speaker_id="spk-abc123",
                is_personal=True,
            )

        mock_answer_via_ha.assert_called_once()
        mock_abstain.assert_not_called()
        assert result.text == "the lights are on"


class TestRelayNoIdentityShortCircuit:
    """``_relay_route``'s ``identity_absent`` gate — the relay-path
    counterpart of the abstention short-circuit for a caller with no
    resolved speaker at all (``ServingPath.RELAY``).

    A personal interrogative with no identity gets the canned no-identity
    response BEFORE either external leg is tried (there is no speaker for
    the question to be about). The final fallback below that gate is the
    local base model when one is resident, and the canned limited-mode
    response when it is not.
    """

    def _config(self):
        config = ServerConfig()
        config.personal_referent = None
        config.sentence_type = None
        return config

    def test_personal_interrogative_with_no_identity_returns_canned(self):
        from paramem.server.app import _relay_route

        config = self._config()
        ha_client = MagicMock()
        cloud_agent = MagicMock()

        result = _relay_route(
            text="Where do I live?",
            history=[],
            config=config,
            cloud_permitted=True,
            ha_client=ha_client,
            cloud_agent=cloud_agent,
            identity_absent=True,
        )

        assert result.text == config.abstention.load_no_identity_response()
        ha_client.conversation_process.assert_not_called()
        cloud_agent.call.assert_not_called()

    def test_identity_absent_false_never_short_circuits(self):
        """A resolved-speaker personal interrogative (``identity_absent=
        False``) must never hit the no-identity short-circuit — the gate
        is conjuncted with ``identity_absent`` specifically so a resolved
        speaker is never mistaken for a caller with no identity at all.
        The turn reaches the normal HA dispatch instead."""
        from paramem.server.app import _relay_route
        from paramem.server.chat_result import ChatResult

        config = self._config()
        cloud_agent = MagicMock()

        with patch(
            "paramem.server.app.answer_via_ha",
            return_value=ChatResult(text="ha answer", escalated=True),
        ) as mock_ha:
            result = _relay_route(
                text="Where do I live?",
                history=[],
                config=config,
                cloud_permitted=True,
                ha_client=MagicMock(),
                cloud_agent=cloud_agent,
                speaker="Alex",
                speaker_id="speaker0",
                identity_absent=False,
            )

        mock_ha.assert_called_once()
        assert result.text == "ha answer"
        assert result.text != config.abstention.load_no_identity_response()

    def test_no_identity_short_circuit_ignores_abstention_enabled_false(self):
        """The no-identity short-circuit is a structural impossibility (no
        speaker for a personal question to be about), never a feature the
        operator can toggle off — it must fire even when
        ``config.abstention.enabled`` is False, unlike the ordinary
        abstention gate in ``handle_chat``."""
        from paramem.server.app import _relay_route

        config = self._config()
        config.abstention.enabled = False
        ha_client = MagicMock()
        cloud_agent = MagicMock()

        result = _relay_route(
            text="Where do I live?",
            history=[],
            config=config,
            cloud_permitted=True,
            ha_client=ha_client,
            cloud_agent=cloud_agent,
            identity_absent=True,
        )

        assert result.text == config.abstention.load_no_identity_response()
        ha_client.conversation_process.assert_not_called()
        cloud_agent.call.assert_not_called()

    def test_base_model_fallback_fires_when_ha_and_cloud_unavailable(self):
        """The relay's final link, when HA and cloud both fail/are
        unavailable and a local model is loaded (LOCAL mode,
        ``identity_absent=True``): the local base model answers directly —
        adapter-off, no history, no facts, no speaker context — instead of
        the generic limited-mode canned message."""
        from paramem.server.app import _relay_route

        config = self._config()
        # spec=PeftModel passes the isinstance(model, PeftModel)
        # precondition base_model_inference now enforces on the relay's
        # base-model-fallback generate.
        model = MagicMock(spec=PeftModel)
        model.gradient_checkpointing_disable = MagicMock()
        model.gradient_checkpointing_enable = MagicMock()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "prompt text"

        with patch(
            "paramem.server.inference.generate_answer",
            return_value="base model answer",
        ) as mock_generate:
            result = _relay_route(
                text="What's a good movie tonight?",
                history=[],
                config=config,
                cloud_permitted=True,
                ha_client=None,
                cloud_agent=None,
                identity_absent=True,
                model=model,
                tokenizer=tokenizer,
            )

        mock_generate.assert_called_once()
        assert result.text == "base model answer"
        # No speaker context: the system prompt carries no "You are
        # speaking with ..." identity line anywhere in the rendered
        # messages (role-folding is a tokenizer-template detail this test
        # doesn't need to pin — the MagicMock tokenizer's own
        # ``adapt_messages`` capability probe can fold system into user).
        messages = tokenizer.apply_chat_template.call_args.args[0]
        rendered = " ".join(m["content"] for m in messages)
        assert "speaking with" not in rendered
        # No history: only the current turn's text appears, nothing from
        # a prior exchange (there is none passed here, but this also pins
        # that _base_model_answer was fed history=None, not the relay's
        # own history=[] arg, which _relay_route never threads through).
        assert rendered.count("What's a good movie tonight?") == 1

    def test_limited_mode_response_when_no_model_ha_or_cloud(self):
        """Server-wide cloud-only mode has no local model — the canned
        limited-mode response remains the final fallback there, not the
        base model (which does not exist on this path)."""
        from paramem.server.app import _relay_route

        config = self._config()

        result = _relay_route(
            text="What's a good movie tonight?",
            history=[],
            config=config,
            cloud_permitted=True,
            ha_client=None,
            cloud_agent=None,
        )

        assert "can't answer" in result.text
