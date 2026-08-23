"""Config-vs-store validation: does a candidate config contradict the tier
store already on disk.

Sibling of :mod:`paramem.server.vram_validator` — both are small
server-layer validator modules of pure functions over a
:class:`~paramem.server.config.ServerConfig`. This module answers one
question: "would this config boot against this store". It has two callers
that must agree: the boot loader
(:func:`paramem.server.app._load_model_into_state`, whose own responsibility
is loading the base model and adapters into ``_state``) and the
candidate-validation gate (:func:`paramem.server.migration.validate_candidate`,
shared by every config-promotion door — preview, confirm, rollback, and
config restore). One module holds the one implementation both call.

Not placed under :mod:`paramem.adapters.registry_binding` — the closest
precedent, and the one this module's second check builds directly on — because
that module's stated scope is "is the on-disk registry corroborated by its
slot manifests, per tier"; one of the two checks here is about the interim
ring rather than a registry/manifest binding, and nothing under
``paramem.adapters`` imports :class:`~paramem.server.config.ServerConfig`.
Enlarging it past one responsibility would be the deviation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from paramem.adapters.registry_binding import verify_tier_binding
from paramem.memory.interim_adapter import adapter_slot_root_for_name, iter_interim_dirs
from paramem.utils.tiers import MAIN_TIERS

if TYPE_CHECKING:
    from paramem.server.config import ServerConfig


class ConfigStoreMismatch(RuntimeError):
    """A config document contradicts the tier store it would run against.

    Raised by :func:`check_config_against_store`. A ``RuntimeError``
    subclass so existing broad-exception handling around config loading
    (the boot lifespan's degrade-vs-abort classification, in particular)
    keeps its shape unchanged.

    Parameters
    ----------
    message:
        The refusal text, verbatim. This is the operator-facing
        contract: it becomes the 4xx response body at every
        config-promotion door, the recorded incident's detail, and the
        attention row's summary.
    check:
        Stable per-check discriminator (one of
        ``"interim_ring_without_episodic"`` or
        ``"disabled_tier_active_keys:<tier>"``). Required: used as an
        incident dedup key downstream, so a server that fixes one check and
        then hits the other does not silently bump a single incident row's
        count instead of opening a second one — an unnamed check would defeat
        that by construction.
    """

    def __init__(self, message: str, *, check: str) -> None:
        super().__init__(message)
        self.check = check


def check_config_against_store(config: "ServerConfig") -> None:
    """Raise :class:`ConfigStoreMismatch` when *config* contradicts the tier
    store already on disk.

    Two checks, both read-only against ``config.adapter_dir``:

    1. The episodic tier is disabled while a populated interim ring still
       sits on disk — the ring is episodic-shaped and has no tier to drain
       into. Disabling ``consolidation.max_interim_count`` alone (setting
       it to ``0``) is not enough: that only stops MINTING new interim
       slots, it does not remove a ring already staged on disk.
    2. Any main tier is disabled while its own registry still holds active
       keys — those keys become permanently unreachable: no adapter is
       created, no slot is mounted, and nothing rebuilds them at the next
       fold. Uses the same registry↔slot binding primitive the mount loop
       and migration path already resolve a tier's active key count
       through.

    Both checks pass silently — return ``None`` — against a store that does
    not exist yet. Required: a candidate that re-points ``paths.data`` (an
    R-PATHS carve) names an ``adapter_dir`` that has never been written to.
    :func:`~paramem.memory.interim_adapter.iter_interim_dirs` early-returns
    when ``<adapter_dir>/episodic`` is not a directory, and
    :func:`~paramem.adapters.registry_binding.verify_tier_binding` resolves
    every filesystem failure on a tier root to a verdict rather than
    propagating — a missing store is a pass, not a refusal.

    Parameters
    ----------
    config:
        A constructed :class:`~paramem.server.config.ServerConfig`.
        Production sources for this parameter: at boot, the config
        :func:`paramem.server.app._load_model_into_state` was called with;
        at every config-promotion door, the ``ServerConfig``
        :func:`paramem.server.migration.validate_candidate` just built from
        the candidate/backup bytes — so validating a candidate that
        re-points ``paths.data`` checks the candidate's OWN new root, not
        the live one.

    Raises
    ------
    ConfigStoreMismatch
        *config* contradicts the store. The exception message carries the
        exact operator remediation text.
    """
    tier_configs = config.tier_config_map()
    adapter_dir = config.adapter_dir

    if "episodic" not in tier_configs:
        stray_interims = list(iter_interim_dirs(adapter_dir))
        if stray_interims:
            raise ConfigStoreMismatch(
                f"adapters.episodic.enabled=false but "
                f"{len(stray_interims)} interim slot(s) still exist under "
                f"{adapter_dir / 'episodic'}.\n"
                f"\n"
                f"The interim ring has no destination tier to absorb into.\n"
                f"\n"
                f"Remediation:\n"
                f"  - POST /consolidate to drain the ring into episodic, or\n"
                f"  - POST /interim/discard to destroy it,\n"
                f"    then disable episodic.",
                check="interim_ring_without_episodic",
            )

    for tier in MAIN_TIERS:
        if tier in tier_configs:
            continue
        tier_root = adapter_slot_root_for_name(adapter_dir, tier)
        binding = verify_tier_binding(tier, tier_root)
        if binding.registry is not None and binding.registry.list_active():
            raise ConfigStoreMismatch(
                f"adapters.{tier}.enabled=false but its registry at {tier_root} "
                f"holds {len(binding.registry.list_active())} active key(s).\n"
                f"\n"
                f"Disabling a tier that still owns keys makes them unreachable — no "
                f"adapter is created, no slot is mounted, and nothing rebuilds them "
                f"at the next fold.\n"
                f"\n"
                f"Remediation:\n"
                f"  - Drain the tier (fold/promote its keys elsewhere), or\n"
                f"  - Erase its keys,\n"
                f"    then disable it.",
                check=f"disabled_tier_active_keys:{tier}",
            )
