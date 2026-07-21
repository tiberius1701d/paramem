"""Unit tests for the shared cloud-egress admission component.

:func:`~paramem.utils.cloud_admission.evaluate_cloud_egress` is the ONLY
place the "may we reach a cloud LLM, and with what credentials?" decision
is computed, so its term-by-term behaviour is pinned here: each single
unmet term, several simultaneously, the OpenAI-compatible endpoint rule,
an unregistered provider, and the ``"auto"``/``"off"`` judge tokens that
must never reach a key lookup.
"""

import pytest

from paramem.utils.cloud_admission import (
    OPENAI_COMPAT_ENDPOINTS,
    OPENAI_COMPAT_PROVIDERS,
    PROVIDER_KEY_ENV,
    EgressVerdict,
    evaluate_cloud_egress,
    resolve_api_key,
)


@pytest.fixture(autouse=True)
def _clear_provider_keys(monkeypatch):
    """Remove every provider key env var so a developer's real key cannot
    make a "missing key" case pass by accident."""
    for env_name in PROVIDER_KEY_ENV.values():
        monkeypatch.delenv(env_name, raising=False)


class TestResolveApiKey:
    """The env lookup underneath the verdict."""

    def test_resolves_registered_provider(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert resolve_api_key("anthropic") == "sk-test"

    def test_unset_var_is_none(self):
        assert resolve_api_key("anthropic") is None

    def test_empty_var_is_none(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        assert resolve_api_key("anthropic") is None

    def test_unregistered_provider_is_none(self):
        assert resolve_api_key("auto") is None
        assert resolve_api_key("off") is None
        assert resolve_api_key("") is None


class TestPermitted:
    """The all-terms-hold path."""

    def test_native_sdk_provider(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        verdict = evaluate_cloud_egress(
            sota_enabled=True,
            provider="anthropic",
            model="claude-sonnet-4-6",
            endpoint=None,
        )
        assert isinstance(verdict, EgressVerdict)
        assert verdict.permitted is True
        assert verdict.gaps == ()
        assert verdict.api_key == "sk-test"
        assert verdict.provider == "anthropic"
        assert verdict.model == "claude-sonnet-4-6"
        # anthropic is not OpenAI-compatible: no default endpoint exists and
        # none is needed.
        assert verdict.endpoint is None

    def test_verdict_is_frozen(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        verdict = evaluate_cloud_egress(
            sota_enabled=True, provider="anthropic", model="m", endpoint=None
        )
        with pytest.raises(AttributeError):
            verdict.permitted = False


class TestSingleMissingTerm:
    """One unmet term at a time — each must refuse, and name itself."""

    def test_master_switch_off(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        verdict = evaluate_cloud_egress(
            sota_enabled=False, provider="anthropic", model="m", endpoint=None
        )
        assert verdict.permitted is False
        assert verdict.gaps == ("sota_enabled is off",)
        # A refused verdict never hands back a usable credential.
        assert verdict.api_key == ""

    def test_no_provider(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        verdict = evaluate_cloud_egress(sota_enabled=True, provider="", model="m", endpoint=None)
        assert verdict.permitted is False
        assert verdict.gaps == ("no cloud provider configured",)

    def test_no_model(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        verdict = evaluate_cloud_egress(
            sota_enabled=True, provider="anthropic", model="", endpoint=None
        )
        assert verdict.permitted is False
        assert verdict.gaps == ("no model configured for provider 'anthropic'",)

    def test_no_api_key(self):
        verdict = evaluate_cloud_egress(
            sota_enabled=True, provider="anthropic", model="m", endpoint=None
        )
        assert verdict.permitted is False
        assert verdict.gaps == ("ANTHROPIC_API_KEY env var is unset",)
        assert verdict.api_key == ""

    def test_missing_endpoint_for_self_hosted_openai_compat(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "local-key")
        verdict = evaluate_cloud_egress(
            sota_enabled=True, provider="ollama", model="llama3", endpoint=None
        )
        assert verdict.permitted is False
        assert verdict.gaps == ("no endpoint for provider 'ollama'",)


class TestMultipleGaps:
    """Every unmet term is reported, not just the first — the point of the
    collect-all behaviour is that one log line ends the fix-one-rerun loop."""

    def test_switch_provider_and_model_all_reported(self):
        verdict = evaluate_cloud_egress(sota_enabled=False, provider="", model="", endpoint=None)
        assert verdict.permitted is False
        assert verdict.gaps == (
            "sota_enabled is off",
            "no cloud provider configured",
            "no model configured for provider ''",
        )

    def test_key_and_endpoint_both_reported(self):
        verdict = evaluate_cloud_egress(
            sota_enabled=True, provider="ollama", model="llama3", endpoint=None
        )
        assert verdict.permitted is False
        assert verdict.gaps == (
            "OLLAMA_API_KEY env var is unset",
            "no endpoint for provider 'ollama'",
        )


class TestOpenAICompatEndpoint:
    """Endpoint resolution for the OpenAI-compatible wire format."""

    def test_provider_default_endpoint_is_enough(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        verdict = evaluate_cloud_egress(
            sota_enabled=True, provider="groq", model="llama-3.3-70b", endpoint=None
        )
        assert verdict.permitted is True
        assert verdict.endpoint == OPENAI_COMPAT_ENDPOINTS["groq"]

    def test_explicit_endpoint_wins_over_default(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        verdict = evaluate_cloud_egress(
            sota_enabled=True,
            provider="groq",
            model="llama-3.3-70b",
            endpoint="http://127.0.0.1:9000/v1/chat/completions",
        )
        assert verdict.permitted is True
        assert verdict.endpoint == "http://127.0.0.1:9000/v1/chat/completions"

    def test_explicit_endpoint_satisfies_self_hosted_provider(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "local-key")
        verdict = evaluate_cloud_egress(
            sota_enabled=True,
            provider="ollama",
            model="llama3",
            endpoint="http://127.0.0.1:11434/v1/chat/completions",
        )
        assert verdict.permitted is True
        assert verdict.endpoint == "http://127.0.0.1:11434/v1/chat/completions"

    def test_ollama_is_openai_compat_without_a_default(self):
        assert "ollama" in OPENAI_COMPAT_PROVIDERS
        assert "ollama" not in OPENAI_COMPAT_ENDPOINTS


class TestUnregisteredProvider:
    """A string outside PROVIDER_KEY_ENV is not a provider."""

    def test_unknown_provider_refused(self):
        verdict = evaluate_cloud_egress(
            sota_enabled=True, provider="deepmind", model="m", endpoint=None
        )
        assert verdict.permitted is False
        assert verdict.gaps == ("unsupported provider 'deepmind'",)
        # No env-var gap: an unregistered provider names no env var, so
        # reporting one would just restate the first gap.
        assert verdict.api_key == ""

    @pytest.mark.parametrize("token", ["auto", "off"])
    def test_judge_tokens_never_reach_a_key_lookup(self, token, monkeypatch):
        """``"auto"`` (local judge) and ``"off"`` (no judge) are settings, not
        providers. The old plausibility path crashed on
        ``PROVIDER_KEY_ENV.get("auto")``; membership in the registry is what
        prevents it, so assert both the refusal and that no env read happened."""
        reads: list[str] = []
        import paramem.utils.cloud_admission as mod

        real_environ_get = mod.os.environ.get

        def _spy(name, default=None):
            reads.append(name)
            return real_environ_get(name, default)

        monkeypatch.setattr(mod.os.environ, "get", _spy)
        verdict = evaluate_cloud_egress(sota_enabled=True, provider=token, model="m", endpoint=None)
        assert verdict.permitted is False
        assert verdict.gaps == (f"unsupported provider {token!r}",)
        assert reads == []
        assert token not in PROVIDER_KEY_ENV
