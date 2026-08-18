"""Tests for paramem.server.manifest_status -- the row-status vocabulary.

Pins the totality invariant between this module's ``ROW_STATUS_FOR_VERDICT``
mapping and :mod:`paramem.adapters.registry_binding`'s verdict vocabulary,
both directions: every non-publishable verdict mints a row status, and no
other verdict (nor any stray key) does. Also pins that the mapping's KEYS
are the imported verdict constants themselves, not restated string
literals -- a verdict string changing in ``registry_binding.py`` would be a
type error here, never a silent drift.
"""

from __future__ import annotations

from paramem.adapters.registry_binding import (
    ALL_VERDICTS,
    NO_CANDIDATES,
    REGISTRY_ABSENT_WITH_SLOTS,
    REGISTRY_UNREADABLE,
    VERIFIED,
)
from paramem.server.manifest_status import ROW_STATUS_FOR_VERDICT

# The complete, closed verdict vocabulary -- imported from registry_binding's
# own single definition, never hand-listed here, so a verdict added there
# fails this pin instead of drifting silently.
_ALL_VERDICTS = ALL_VERDICTS
# VERIFIED and NO_CANDIDATES mint no row at all -- see
# _validate_adapter_slot and ROW_STATUS_FOR_VERDICT's own docstring.
_NO_ROW_VERDICTS = frozenset({VERIFIED, NO_CANDIDATES})


class TestRowStatusForVerdictTotality:
    """``ROW_STATUS_FOR_VERDICT``'s key set is exactly the non-publishable
    verdict set: every verdict that mints a row is present, and no verdict
    that mints no row is."""

    def test_every_non_publishable_verdict_has_a_row_status(self) -> None:
        expected_keys = _ALL_VERDICTS - _NO_ROW_VERDICTS
        assert set(ROW_STATUS_FOR_VERDICT) == expected_keys

    def test_verified_and_no_candidates_mint_no_row(self) -> None:
        assert VERIFIED not in ROW_STATUS_FOR_VERDICT
        assert NO_CANDIDATES not in ROW_STATUS_FOR_VERDICT

    def test_no_key_outside_the_closed_verdict_set(self) -> None:
        assert set(ROW_STATUS_FOR_VERDICT) <= _ALL_VERDICTS

    def test_the_two_registry_unreadable_family_verdicts_collapse_to_one_row_status(
        self,
    ) -> None:
        assert (
            ROW_STATUS_FOR_VERDICT[REGISTRY_UNREADABLE]
            == ROW_STATUS_FOR_VERDICT[REGISTRY_ABSENT_WITH_SLOTS]
            == "registry_unverified"
        )
