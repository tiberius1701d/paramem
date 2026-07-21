"""Cloud-egress admission — the single answer to "may we call a cloud LLM?".

Four sites used to answer that question with four different sets of terms:
session-tier SOTA enrichment, graph-tier enrichment, graph-tier predicate
normalization, and the ``/calibrate/enrich`` endpoint.  One of them omitted
the ``sota_enabled`` master switch entirely, so an operator-triggered
endpoint could egress with the switch off.  :func:`evaluate_cloud_egress`
is now the only place the decision is computed; every caller reads the
:class:`EgressVerdict` it returns and decides what to DO about a refusal
(skip silently, fall back to the local model, or raise) — the decision
itself is not re-derived anywhere.

Leaf module by construction: stdlib only.  It must not import from
``paramem.graph``, ``paramem.training`` or ``paramem.server`` — the graph
layer's ``extraction_pipeline`` imports the extractor, which imports this,
so any reach back into those packages would close an import cycle.  The
precedent is :mod:`paramem.utils.identity`, which holds ``canonical()`` for
the same reason: a primitive every tier needs, owned by none of them.

The provider tables below are the registry of what "cloud" means here.
Adding a provider means adding one entry to :data:`PROVIDER_KEY_ENV` (and,
for an OpenAI-compatible host, one to :data:`OPENAI_COMPAT_ENDPOINTS`);
nothing else in the codebase enumerates providers.
"""

import os
from dataclasses import dataclass

# Default chat-completions URL per OpenAI-compatible provider.  A provider
# in this table needs no configured endpoint; one outside it (``ollama``,
# which is self-hosted and has no canonical URL) does.
OPENAI_COMPAT_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
}

# Providers reached over the OpenAI-compatible chat-completions wire format,
# as opposed to a native SDK.  ``ollama`` speaks the same format but is
# self-hosted, so it carries no default endpoint.
OPENAI_COMPAT_PROVIDERS = set(OPENAI_COMPAT_ENDPOINTS) | {"ollama"}

# Env var holding the API key for each supported provider.  Membership in
# this table is also what makes a string a PROVIDER at all: ``"auto"`` and
# ``"off"`` (the local/disabled plausibility-judge settings) are absent, so
# they can never reach a key lookup.
PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}


@dataclass(frozen=True)
class EgressVerdict:
    """The outcome of one cloud-egress admission check.

    Attributes:
        permitted: ``True`` only when every term held.  Callers must not
            place a cloud call on ``False`` — what they do instead (skip,
            fall back to the local model, raise HTTP 400) is theirs to
            choose.
        provider: The provider name that was checked, echoed verbatim.
        model: The model id that was checked, echoed verbatim.
        endpoint: The endpoint to call: the explicit argument when given,
            otherwise the provider's default from
            :data:`OPENAI_COMPAT_ENDPOINTS`, otherwise ``None`` (native-SDK
            providers such as ``anthropic`` need none).
        api_key: The resolved key, or ``""`` when unresolvable.  Always a
            ``str`` so a permitted caller can forward it without a
            ``None`` check.
        gaps: Human-readable phrases naming EVERY unmet term, suitable for
            a log line or an HTTP error detail.  Empty iff ``permitted``.
    """

    permitted: bool
    provider: str
    model: str
    endpoint: str | None
    api_key: str
    gaps: tuple[str, ...]


def resolve_api_key(provider: str) -> str | None:
    """Resolve *provider*'s API key from the env var named in :data:`PROVIDER_KEY_ENV`.

    Args:
        provider: Provider name, e.g. ``"anthropic"``.

    Returns:
        The key, or ``None`` when the provider is not in the registry OR
        its env var is unset/empty — the two cases are indistinguishable
        from this return value alone.  :func:`evaluate_cloud_egress`
        separates them, and reports both in ``gaps``.
    """
    key_env_name = PROVIDER_KEY_ENV.get(provider)
    if key_env_name is None:
        return None
    return os.environ.get(key_env_name) or None


def evaluate_cloud_egress(
    *,
    sota_enabled: bool,
    provider: str,
    model: str,
    endpoint: str | None,
) -> EgressVerdict:
    """Decide whether a cloud LLM call may be placed, and with what credentials.

    Every term must hold for ``permitted=True``:

    1. ``sota_enabled`` — the operator's master switch for all cloud egress.
    2. ``provider`` is non-empty and present in :data:`PROVIDER_KEY_ENV`.
    3. ``model`` is non-empty.
    4. :func:`resolve_api_key` returns a non-empty key.
    5. For a provider in :data:`OPENAI_COMPAT_PROVIDERS`, an endpoint is
       available — either *endpoint* explicitly, or the provider's default
       in :data:`OPENAI_COMPAT_ENDPOINTS`.

    ALL unmet terms are collected, never just the first.  A one-at-a-time
    check produces the "fix the key, rerun, then discover the endpoint was
    also missing" loop; one verdict reports everything missing at once.

    Args:
        sota_enabled: The master switch (``consolidation.sota_enabled``).
        provider: Configured provider name.  ``""`` means "no cloud
            provider configured"; a non-provider token such as ``"auto"``
            or ``"off"`` is reported as unsupported and never reaches a
            key lookup.
        model: Configured model id for that provider.
        endpoint: Explicit endpoint override, or ``None`` to accept the
            provider's default.  Ignored for native-SDK providers.

    Returns:
        An :class:`EgressVerdict`.  On refusal, ``api_key`` is ``""`` and
        ``gaps`` names each unmet term.
    """
    gaps: list[str] = []

    if not sota_enabled:
        gaps.append("sota_enabled is off")

    known_provider = bool(provider) and provider in PROVIDER_KEY_ENV
    if not provider:
        gaps.append("no cloud provider configured")
    elif not known_provider:
        gaps.append(f"unsupported provider {provider!r}")

    if not model:
        gaps.append(f"no model configured for provider {provider!r}")

    # Key resolution is attempted only for a registered provider — an
    # unregistered token has no env var to name, and reporting a second
    # gap for it would just restate the first.
    api_key = ""
    if known_provider:
        api_key = resolve_api_key(provider) or ""
        if not api_key:
            gaps.append(f"{PROVIDER_KEY_ENV[provider]} env var is unset")

    resolved_endpoint = endpoint or OPENAI_COMPAT_ENDPOINTS.get(provider)
    if known_provider and provider in OPENAI_COMPAT_PROVIDERS and not resolved_endpoint:
        gaps.append(f"no endpoint for provider {provider!r}")

    return EgressVerdict(
        permitted=not gaps,
        provider=provider,
        model=model,
        endpoint=resolved_endpoint,
        api_key=api_key if not gaps else "",
        gaps=tuple(gaps),
    )
