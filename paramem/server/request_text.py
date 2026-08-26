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

    Both egress doors (``answer_via_cloud`` / ``answer_via_ha``,
    ``paramem/server/egress.py``) treat a blank turn reaching their
    anonymize terminal as an invariant violation (``RuntimeError``) —
    ``anonymize_turn`` (``paramem/graph/flows.py``) has no refusal cause
    for empty text, it just returns a failed contract with
    ``failure=None``.  Reject at the request boundary instead, with a
    clear 422, before the turn can reach any dispatch under
    ``cloud_mode: anonymize|both``.  The value is returned unstripped —
    the turn text is persisted verbatim elsewhere and must not change
    shape here.

    Shared across every chat-door request schema (``ChatRequest``,
    ``DebugProbeRequest``, ``CalibrateRespondRequest``) via
    :data:`NonBlankText`, so the rule lives once rather than being
    inherited alongside fields a given door does not consume.
    """
    if not value.strip():
        raise ValueError("text must not be empty or whitespace-only")
    return value


NonBlankText = Annotated[str, AfterValidator(_reject_blank_text)]
