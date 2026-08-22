"""The one main-tier vocabulary and order.

Holds :data:`MAIN_TIERS`, the closed vocabulary and order of the three
main adapter tiers — ``episodic``, ``semantic``, ``procedural`` — used
project-wide as the training-order rule and as the order
:func:`~paramem.models.loader.ensure_resident_tiers` creates tiers in.

This module imports nothing from ``paramem`` — it is a leaf so that
``paramem/models/loader.py`` can import it without closing an import
cycle (``paramem/memory/interim_adapter.py`` imports ``create_adapter``
and ``detach_adapters`` from ``loader.py``, and ``paramem/memory/__init__.py``
imports ``create_interim_adapter``, so any ``paramem.memory.*`` import
inside ``loader.py`` would close the cycle).

:data:`MAIN_TIERS` answers *which tier names exist in this system, in
what order*. ``ServerConfig.tier_config_map()``
(:mod:`paramem.server.config`) answers a narrower question — *which of
them this deployment has, with what LoRA shape* — by filtering
:data:`MAIN_TIERS` on ``adapters.<tier>.enabled``. A survivor/fallback
list (this module) must still name a tier even for an order a given
deployment has partially disabled; a budget or a creation set
(``tier_config_map()``) must not.
"""

from typing import Final

MAIN_TIERS: Final[tuple[str, str, str]] = ("episodic", "semantic", "procedural")
