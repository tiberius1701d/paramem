"""The cloud round trip: admission, anonymize, deanonymize, provider adapters.

Everything a tier needs to reach a cloud LLM and come back with its
identities restored lives here — session-tier extraction, graph-tier
enrichment, and conversation egress all compose these primitives; none of
them owns it (the same placement rule as :mod:`paramem.utils.identity` /
:mod:`paramem.utils.cloud_admission` before this package existed).

This package must import NOTHING from ``paramem.graph`` — every primitive
here operates on plain ``dict``/``str`` artifacts (fact dicts, transcript
text), never a ``SessionGraph`` or ``Relation``. The render from a
``Relation`` to a fact dict is the caller's job, in ``paramem/graph/``.
"""
