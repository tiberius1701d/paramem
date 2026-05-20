"""Entry-encoded indexed-key format helpers.

Thin re-export of :mod:`paramem.memory.entry` — kept so existing
experiment imports (``experiments.utils.quadruple_format``) keep working. New
code should import from the production module directly.
"""

from paramem.memory.entry import (  # noqa: F401
    QUAD_RECALL_TEMPLATE,
    assign_quad_keys,
    build_enriched_registry,
    build_registry,
    compute_simhash,
    parse_recalled_quad,
    verify_confidence,
)
