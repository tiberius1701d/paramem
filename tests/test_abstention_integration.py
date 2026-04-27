"""End-to-end integration test for the abstention short-circuit.

Uses the REAL sanitizer, REAL ``configs/server.yaml``, and REAL
``handle_chat`` code path. Only the model, tokenizer, and router are
stubbed — all three are unreachable on the short-circuit branch, so
stubbing them does not weaken the test. Validates the exact failure
mode from 2026-04-21 (user asked "Where do I live?" on an untrained
adapter, Mistral confabulated "New York City").

Distinct from ``test_abstention.py``: those tests patch
``sanitize_for_cloud`` and use a ``MagicMock`` config to isolate the
short-circuit logic. This test exercises the real sanitizer and real
YAML loader end-to-end.
"""

from unittest.mock import MagicMock

import pytest

from paramem.server.config import load_server_config
from paramem.server.inference import handle_chat
from paramem.server.router import RoutingPlan
from paramem.server.sanitizer import sanitize_for_cloud


@pytest.fixture
def server_config():
    return load_server_config("configs/server.yaml")


@pytest.fixture
def empty_adapter_router():
    """Router stub mimicking an untrained/empty adapter: no entities,
    no keys, ``match_source="none"`` for every query. Matches the
    exact failure state observed in the bug report.

    Sets ``intent=PERSONAL`` because the production router with the
    cosine residual classifies the test queries ("Where do I live?"
    etc.) as PERSONAL via the encoder + exemplars even when no PA
    state matches.  Without an intent the new abstention gate would
    not fire and the regression test would silently pass on the wrong
    path.
    """
    from paramem.server.router import Intent

    router = MagicMock()

    def route(text, speaker=None, speaker_id=None):
        # Non-personal probes ("What is the capital of France?") get
        # GENERAL so the abstention gate does not fire on them.
        if any(tok.lower() in {"i", "my", "me"} for tok in text.split()):
            intent = Intent.PERSONAL
        else:
            intent = Intent.GENERAL
        return RoutingPlan(strategy="direct", match_source="none", intent=intent)

    router.route = route
    router._speaker_key_index = {}
    return router


@pytest.fixture
def exploding_model():
    """Model that raises if generation is attempted. Proves the
    short-circuit fires before any model inference runs — if the
    abstention logic is bypassed, the test crashes with a clear signal
    rather than silently hiding a regression behind mock responses."""
    model = MagicMock()
    model.gradient_checkpointing_disable = MagicMock()
    model.generate.side_effect = RuntimeError(
        "model.generate() must not be called — short-circuit failed"
    )
    return model


class TestSanitizerPrecondition:
    """Prerequisite: the real sanitizer must classify the target queries
    as personal first-person. Without this, the short-circuit would
    never activate on these queries in production."""

    @pytest.mark.parametrize(
        "query",
        [
            "Where do I live?",
            "What is my birthday?",
            "What's my address?",
            "Who is my partner?",
        ],
    )
    def test_real_sanitizer_blocks_first_person_query(self, query):
        # Self-referential resolution requires an identified speaker — the
        # sanitizer's whole point is "personal data is graph-truth", and
        # there is no resolution target for "I" / "my" without a
        # speaker_id.  Production callers always have one when the
        # speaker is enrolled (which is when self-referential matters).
        sanitized, findings = sanitize_for_cloud(query, mode="block", speaker_id="Speaker0")
        assert sanitized is None, f"sanitizer should block {query!r}"
        assert findings, "sanitizer must explain why the query was blocked"

    @pytest.mark.parametrize(
        "query",
        [
            "What is the capital of France?",
            "Turn on the lights",
        ],
    )
    def test_real_sanitizer_allows_non_personal(self, query):
        sanitized, findings = sanitize_for_cloud(query, mode="block", speaker_id="Speaker0")
        assert sanitized is not None, f"sanitizer should not block {query!r}"


class TestAbstentionEndToEnd:
    def test_where_do_i_live_returns_canned_response(
        self, server_config, empty_adapter_router, exploding_model
    ):
        """Reproduces the exact bug: untrained adapter + self-referential
        query. Before the fix: Mistral confabulated "New York City".
        After the fix: canned abstention response, model never invoked.
        ``empty_adapter_router`` has no keys for this speaker, so the
        cold-start variant fires (rather than the standard ``response``).
        """
        result = handle_chat(
            text="Where do I live?",
            conversation_id="abstention-integration-test",
            speaker="Alex",
            history=None,
            model=exploding_model,
            tokenizer=MagicMock(),
            config=server_config,
            router=empty_adapter_router,
            speaker_id="spk-integration-test",
        )

        assert result.text == server_config.abstention.load_cold_start_response()
        assert not result.escalated
        exploding_model.generate.assert_not_called()

    def test_anonymous_speaker_also_protected(
        self, server_config, empty_adapter_router, exploding_model
    ):
        """Per the deferred-identity-binding design: speaker_id (anonymous
        grouping identifier) is sufficient for attribution. The
        short-circuit must fire even when the real speaker name has not
        yet been disclosed.  Empty router → cold-start response."""
        result = handle_chat(
            text="What is my birthday?",
            conversation_id="abstention-anon-test",
            speaker=None,
            history=None,
            model=exploding_model,
            tokenizer=MagicMock(),
            config=server_config,
            router=empty_adapter_router,
            speaker_id="spk-anon-77",
        )

        assert result.text == server_config.abstention.load_cold_start_response()
        exploding_model.generate.assert_not_called()

    def test_non_personal_query_still_uses_base_model(self, server_config, empty_adapter_router):
        """Guardrail: the short-circuit must NOT fire for questions the
        sanitizer allows. Cloud-outage on general-knowledge queries
        still falls through to ``_base_model_answer`` — base model
        general knowledge is a reasonable last resort there, unlike
        personal-data confabulation."""
        from paramem.server.inference import ChatResult

        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        tokenizer = MagicMock()

        # Patch _base_model_answer to confirm it IS reached on the
        # non-personal path. HA/SOTA agents are None (unavailable),
        # so the cloud escalation returns nothing and falls through.
        from unittest.mock import patch

        with patch(
            "paramem.server.inference._base_model_answer",
            return_value=ChatResult(text="Paris is the capital of France."),
        ) as mock_base_model:
            result = handle_chat(
                text="What is the capital of France?",
                conversation_id="non-personal-test",
                speaker="Alex",
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=server_config,
                router=empty_adapter_router,
                sota_agent=None,
                ha_client=None,
                speaker_id="spk-integration-test",
            )

        mock_base_model.assert_called_once()
        assert result.text != server_config.abstention.load_response()
