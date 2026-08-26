"""``ChatResult`` — the served-turn result type.

A leaf module: the result type a turn's answering leg produces is consumed
by the routing layer (:mod:`paramem.server.inference`), the egress layer
(:mod:`paramem.server.egress`), the app layer
(:mod:`paramem.server.app`) and the calibration door
(:mod:`paramem.server.calibrate`) alike, so it belongs to none of them —
owning it here lets :mod:`paramem.server.egress` and
:mod:`paramem.server.inference` both name it without either importing the
other.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatResult:
    """Result of one chat dispatch.

    Attributes:
        text: The reply text.
        escalated: Whether an escalation hop (HA or cloud) produced this
            result.
        diagnostics: The turn's routing/probe diagnostics, stamped by
            :func:`~paramem.server.inference.handle_chat` (which also
            merges in the diagnostics
            :func:`~paramem.server.inference._probe_and_reason` builds on
            the personal-probe leg). Key-presence contract:

            * Guaranteed on every result ``handle_chat`` returns:
              ``conversation_id``, ``intent``, ``paths_attempted``,
              ``exit_via``, ``is_residual``, ``is_self_referential``.
            * Conditional, set only by ``_probe_and_reason`` on the
              personal-probe leg: ``temporal`` (always present on that leg
              — ``None`` when the date-group selection stage did not run,
              a dict when it did), ``probes`` (present once probing
              actually happens; the zero-survivor date-selection early
              return never reaches it), ``facts_recalled`` (present only
              when the full probe-assembly path completes; the
              no-recalled-facts fallback sets ``probes`` but never reaches
              this key).
            * Per-leg egress/refusal keys — ``{leg}_egress``
              (``"scrubbed"`` | ``"verbatim"``) / ``{leg}_refusal`` (the
              closed vocabulary in :mod:`paramem.server.egress`), for
              ``leg`` in ``"cloud"`` and ``"ha"``. Stamped by
              :func:`~paramem.server.egress.answer_via_cloud` and
              :func:`~paramem.server.egress.answer_via_ha` — the two
              writers of these keys, never a single funnel. A leg carries
              at most one of its own two keys on one turn's dict (a later
              outcome on that leg supersedes an earlier one); the two
              legs' keys are independent, so both may be present at once
              (e.g. an HA send that answered nothing followed by a cloud
              send that did).
            * Results produced by ``paramem.server.app._relay_route`` (the
              speakerless/cloud-only path, which never calls
              ``handle_chat``) carry no leg key for a leg that was never
              reached (a no-identity short-circuit, or every leg failed).
    """

    text: str
    escalated: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)
