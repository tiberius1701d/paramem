"""``NonBlankText`` — the shared blank-rejection annotated type for every
chat-door request schema.

A leaf module: it depends on nothing beyond pydantic, so both
:mod:`paramem.server.app` (``ChatRequest``, ``DebugProbeRequest``) and
:mod:`paramem.server.calibrate` (``CalibrateRespondRequest``) can import it
without either importing the other — ``app.py`` already imports
``calibrate`` as a module, so a definition living in either of those two
would close an import cycle.
"""

from typing import Annotated

from pydantic import AfterValidator


def _reject_blank_text(value: str) -> str:
    """Refuse a turn that is empty or whitespace-only.

    The cloud door's anonymize terminal (``answer_via_cloud`` ->
    ``OutboundText.contract()`` -> ``anonymize_turn``,
    ``paramem/graph/flows.py``) raises ``ValueError`` on an empty or
    whitespace-only transcript — a caller precondition, not a status the
    contract can carry.  The HA door (``answer_via_ha``,
    ``paramem/server/egress.py``) has no anonymize terminal at all: it
    sends the turn verbatim and never builds a contract.  Reject at the
    request boundary instead, with a clear 422, before the turn can reach
    either door.  The value is returned unstripped — the turn text is
    persisted verbatim elsewhere and must not change shape here.

    Shared across every chat-door request schema (``ChatRequest``,
    ``DebugProbeRequest``, ``CalibrateRespondRequest``) via
    :data:`NonBlankText`, so the rule lives once rather than being
    inherited alongside fields a given door does not consume.
    """
    if not value.strip():
        raise ValueError("text must not be empty or whitespace-only")
    return value


NonBlankText = Annotated[str, AfterValidator(_reject_blank_text)]
