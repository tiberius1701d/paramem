"""Tests for the intent classifier configuration surface.

The classifier itself is not wired in yet — these tests pin the
contract: the YAML schema, default values, and the new ``Intent`` enum
+ ``RoutingPlan.intent`` field.  The encoder loader, exemplar loading,
and ``classify_intent()`` arrive in subsequent commits and bring their
own tests; the contract here must stay stable from this point on so
those follow-on changes don't churn the operator-facing config.
"""

from paramem.server.config import (
    IntentConfig,
    ServerConfig,
    load_server_config,
)
from paramem.server.router import Intent, RoutingPlan


class TestIntentEnum:
    def test_four_values(self):
        # Stable contract: any consumer reading RoutingPlan.intent can
        # rely on these four values existing.  Adding new ones is safe;
        # removing or renaming is a breaking change.
        assert {member.value for member in Intent} == {
            "personal",
            "command",
            "general",
            "unknown",
        }

    def test_str_compatible(self):
        # Intent is a str enum so existing logging / serialisation paths
        # treat it as a plain string.
        assert Intent.PERSONAL == "personal"
        assert Intent.UNKNOWN == "unknown"


class TestRoutingPlanIntentField:
    def test_default_is_unknown(self):
        # Existing callers that construct RoutingPlan without specifying
        # intent get Intent.UNKNOWN — informational, no behaviour change.
        plan = RoutingPlan()
        assert plan.intent == Intent.UNKNOWN

    def test_can_be_set_explicitly(self):
        plan = RoutingPlan(intent=Intent.PERSONAL)
        assert plan.intent == Intent.PERSONAL


class TestIntentConfig:
    def test_defaults(self):
        cfg = IntentConfig()
        assert cfg.enabled is True
        assert cfg.encoder_model == "intfloat/multilingual-e5-small"
        assert cfg.encoder_device == "auto"
        assert cfg.encoder_dtype == "float16"
        assert cfg.encoder_query_prefix == "query: "
        assert cfg.exemplars_dir == "configs/intents"
        assert cfg.confidence_margin == 0.05
        assert cfg.fail_closed_intent == "personal"

    def test_server_config_includes_intent(self):
        config = ServerConfig()
        assert isinstance(config.intent, IntentConfig)
        assert config.intent.enabled is True

    def test_yaml_override_encoder_model(self, tmp_path):
        config_file = tmp_path / "server.yaml"
        config_file.write_text(
            "intent:\n"
            "  encoder_model: BAAI/bge-m3\n"
            "  encoder_device: cpu\n"
            "  encoder_query_prefix: ''\n"
        )
        config = load_server_config(config_file)
        assert config.intent.encoder_model == "BAAI/bge-m3"
        assert config.intent.encoder_device == "cpu"
        assert config.intent.encoder_query_prefix == ""

    def test_yaml_override_thresholds(self, tmp_path):
        config_file = tmp_path / "server.yaml"
        config_file.write_text("intent:\n  confidence_margin: 0.1\n  fail_closed_intent: general\n")
        config = load_server_config(config_file)
        assert config.intent.confidence_margin == 0.1
        assert config.intent.fail_closed_intent == "general"

    def test_yaml_partial_override_keeps_defaults(self, tmp_path):
        config_file = tmp_path / "server.yaml"
        config_file.write_text("intent:\n  enabled: false\n")
        config = load_server_config(config_file)
        assert config.intent.enabled is False
        # Other fields keep their dataclass defaults.
        assert config.intent.encoder_model == "intfloat/multilingual-e5-small"
        assert config.intent.confidence_margin == 0.05

    def test_project_server_yaml_has_intent_block(self):
        # The shipped server.yaml has the intent block populated with
        # the locked defaults — guards against accidental removal.
        config = load_server_config("configs/server.yaml")
        assert config.intent.enabled is True
        assert config.intent.encoder_model
        assert config.intent.exemplars_dir
