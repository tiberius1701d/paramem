"""Quadruple-encoded indexed-key format helpers.

Thin re-export of :mod:`paramem.training.quadruple_memory` — kept so existing
experiment imports (``experiments.utils.quadruple_format``) keep working. New
code should import from the production module directly.
"""

from paramem.training.quadruple_memory import (  # noqa: F401
    QUAD_RECALL_TEMPLATE,
    assign_quad_keys,
    build_enriched_registry,
    build_registry,
    compute_simhash,
    format_quadruple_training,
    parse_recalled_quad,
    probe_quad,
    verify_confidence,
)
