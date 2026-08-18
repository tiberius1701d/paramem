"""Shared router-intent stub used by any test that constructs a real
``QueryRouter`` without loading the encoder model backing intent
classification.

Consumers: ``tests/test_router.py``, ``tests/test_fold_crash_resume.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from paramem.server.router import Intent


def _stub_intent(monkeypatch, verdict: Intent) -> MagicMock:
    """Stub ``paramem.server.intent.classify_intent`` to always return *verdict*.

    The router imports ``classify_intent`` lazily inside ``route()``, so
    patching the attribute on the intent module is sufficient.  Returns
    the mock so tests can assert on call args.
    """
    stub = MagicMock(return_value=verdict)
    monkeypatch.setattr("paramem.server.intent.classify_intent", stub)
    return stub
