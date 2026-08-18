"""Unit tests for paramem.memory.bookkeeping.credit_reinforcement.

Extracted verbatim from MemoryStore.reinforce (see tests/test_memory_store.py
for the store-level integration coverage of the delegation); this module pins
the pure-function contract directly: max(own, *inherited) + earned and the
timestamp earn condition, operating on any key -> row mapping with no lock
and no store dependency.  Inheritance is over caller-resolved
``absorbed_counts`` -- credit_reinforcement never looks an absorbed key up
by name, so the cross-tier resolution itself is pinned at the caller
(MemoryStore.reinforce, tests/test_memory_store.py; the working-copy
staging pass, tests/test_fold_phase1.py), not here.

Every known key already carries a bookkeeping row by the time a
reinforcement credit is applied (the every-known-key-has-a-row invariant,
established at the boundaries upstream of this pure function) -- a *key*
absent from *rows* is a violation, raised via
:class:`~paramem.memory.store.BookkeepingInvariantViolation`, never a case
this function fabricates a placeholder row for.
"""

import pytest

from paramem.memory.bookkeeping import credit_reinforcement
from paramem.memory.store import BookkeepingInvariantViolation


class TestAbsentRow:
    def test_raises_when_key_has_no_row(self):
        rows: dict = {}
        with pytest.raises(BookkeepingInvariantViolation):
            credit_reinforcement(rows, "k1", cycle=3, first_seen="2026-01-01T00:00:00Z")

    def test_raises_leaves_rows_untouched(self):
        rows: dict = {}
        with pytest.raises(BookkeepingInvariantViolation):
            credit_reinforcement(rows, "k1", cycle=1, first_seen="2026-01-01T00:00:00Z")
        assert rows == {}


class TestExistingKey:
    def _row(self, **overrides):
        base = {
            "speaker_id": "speaker0",
            "relation_type": "factual",
            "reinforcement_count": 2,
            "last_reinforced_cycle": 1,
            "last_seen": "2026-01-01T00:00:00Z",
            "first_seen": "2025-12-01T00:00:00Z",
            "promoted": False,
        }
        base.update(overrides)
        return base

    def test_reobserved_with_new_timestamp_earns_one(self):
        rows = {"k1": self._row()}
        credit_reinforcement(
            rows,
            "k1",
            cycle=5,
            first_seen="2026-01-05T00:00:00Z",
            timestamp="2026-01-05T00:00:00Z",
            reobserved=True,
        )
        assert rows["k1"]["reinforcement_count"] == 3
        assert rows["k1"]["last_reinforced_cycle"] == 5

    def test_same_timestamp_as_last_seen_does_not_earn(self):
        rows = {"k1": self._row()}
        credit_reinforcement(
            rows,
            "k1",
            cycle=5,
            first_seen="2026-01-01T00:00:00Z",
            timestamp="2026-01-01T00:00:00Z",
            reobserved=True,
        )
        assert rows["k1"]["reinforcement_count"] == 2

    def test_empty_timestamp_never_earns(self):
        rows = {"k1": self._row()}
        credit_reinforcement(
            rows, "k1", cycle=5, first_seen="2025-12-01T00:00:00Z", timestamp="", reobserved=True
        )
        assert rows["k1"]["reinforcement_count"] == 2
        # last_seen preserved unchanged when timestamp is empty.
        assert rows["k1"]["last_seen"] == "2026-01-01T00:00:00Z"

    def test_not_reobserved_normalization_merge_no_earn(self):
        rows = {"k1": self._row()}
        credit_reinforcement(
            rows,
            "k1",
            cycle=5,
            first_seen="2025-12-01T00:00:00Z",
            timestamp="2026-02-01T00:00:00Z",
            reobserved=False,
        )
        assert rows["k1"]["reinforcement_count"] == 2
        # last_seen still advances even without earning — max() runs
        # unconditionally on a non-empty timestamp.
        assert rows["k1"]["last_seen"] == "2026-02-01T00:00:00Z"

    def test_last_seen_never_regresses(self):
        rows = {"k1": self._row(last_seen="2026-03-01T00:00:00Z")}
        credit_reinforcement(
            rows, "k1", cycle=5, first_seen="2025-12-01T00:00:00Z", timestamp="2026-01-01T00:00:00Z"
        )
        assert rows["k1"]["last_seen"] == "2026-03-01T00:00:00Z"

    def test_older_reobservation_does_not_earn_even_though_last_seen_cannot_regress(self):
        """An out-of-order re-observation older than the row's stored
        ``last_seen`` earns nothing on repeat calls -- ``last_seen`` only
        ever moves forward (max-not-sum), so an older timestamp can never
        satisfy a not-equal earn check by staying permanently behind it;
        the earn condition must require strictly newer, not merely
        different, or a re-observation older than the current value would
        earn on every single call forever."""
        rows = {"k1": self._row(last_seen="2026-03-01T00:00:00Z")}
        credit_reinforcement(
            rows,
            "k1",
            cycle=5,
            first_seen="2025-12-01T00:00:00Z",
            timestamp="2026-01-01T00:00:00Z",
            reobserved=True,
        )
        assert rows["k1"]["reinforcement_count"] == 2
        credit_reinforcement(
            rows,
            "k1",
            cycle=6,
            first_seen="2025-12-01T00:00:00Z",
            timestamp="2026-01-01T00:00:00Z",
            reobserved=True,
        )
        assert rows["k1"]["reinforcement_count"] == 2

    def test_first_seen_never_regresses_forward(self):
        rows = {"k1": self._row(first_seen="2025-06-01T00:00:00Z")}
        credit_reinforcement(rows, "k1", cycle=5, first_seen="2025-12-01T00:00:00Z")
        assert rows["k1"]["first_seen"] == "2025-06-01T00:00:00Z"

    def test_empty_first_seen_never_wins(self):
        rows = {"k1": self._row(first_seen="2025-06-01T00:00:00Z")}
        credit_reinforcement(rows, "k1", cycle=5, first_seen="")
        assert rows["k1"]["first_seen"] == "2025-06-01T00:00:00Z"

    def test_speaker_id_and_relation_type_untouched(self):
        rows = {"k1": self._row()}
        credit_reinforcement(
            rows,
            "k1",
            cycle=5,
            first_seen="2025-12-01T00:00:00Z",
            timestamp="2026-02-01T00:00:00Z",
            reobserved=True,
        )
        assert rows["k1"]["speaker_id"] == "speaker0"
        assert rows["k1"]["relation_type"] == "factual"


class TestAbsorbing:
    """absorbed_counts is caller-resolved: credit_reinforcement never looks
    a key up by name — it inherits max() over whatever counts it is handed.
    """

    def test_inherits_max_of_absorbed_counts(self):
        rows = {"survivor": {"reinforcement_count": 1, "last_seen": ""}}
        credit_reinforcement(
            rows,
            "survivor",
            cycle=1,
            first_seen="",
            absorbed_counts=[5, 3],
        )
        assert rows["survivor"]["reinforcement_count"] == 5

    def test_own_count_wins_when_higher_than_absorbed(self):
        rows = {"survivor": {"reinforcement_count": 10, "last_seen": ""}}
        credit_reinforcement(rows, "survivor", cycle=1, first_seen="", absorbed_counts=[2])
        assert rows["survivor"]["reinforcement_count"] == 10

    def test_no_absorbed_counts_contributes_nothing(self):
        rows = {"survivor": {"reinforcement_count": 1, "last_seen": ""}}
        credit_reinforcement(rows, "survivor", cycle=1, first_seen="", absorbed_counts=[])
        assert rows["survivor"]["reinforcement_count"] == 1

    def test_inherited_plus_earned(self):
        rows = {"survivor": {"reinforcement_count": 1, "last_seen": ""}}
        credit_reinforcement(
            rows,
            "survivor",
            cycle=1,
            first_seen="",
            timestamp="2026-01-01T00:00:00Z",
            absorbed_counts=[5],
            reobserved=True,
        )
        assert rows["survivor"]["reinforcement_count"] == 6

    def test_new_survivor_row_raises_when_absent(self):
        """The survivor's own row must already exist -- absorbed_counts is
        inherited maturity applied to an existing row, never a substitute
        for the survivor's own mint."""
        rows: dict = {}
        with pytest.raises(BookkeepingInvariantViolation):
            credit_reinforcement(rows, "new_survivor", cycle=2, first_seen="", absorbed_counts=[7])

    def test_absorbing_a_count_higher_than_survivor_yields_exactly_that_count(self):
        """Absorbing a count-4 key onto a count-1 survivor yields exactly 5
        -- the +1 earned on top of the inherited maximum, not the inherited
        count alone."""
        rows = {"survivor": {"reinforcement_count": 1, "last_seen": ""}}
        credit_reinforcement(
            rows,
            "survivor",
            cycle=1,
            first_seen="",
            timestamp="2026-01-01T00:00:00Z",
            absorbed_counts=[4],
            reobserved=True,
        )
        assert rows["survivor"]["reinforcement_count"] == 5


def test_holds_no_lock_and_touches_only_the_named_key():
    """Pure function: no lock, no side effects beyond *rows[key]*."""
    rows = {
        "other": {"reinforcement_count": 9, "last_seen": "x"},
        "k1": {"reinforcement_count": 1, "last_seen": ""},
    }
    credit_reinforcement(rows, "k1", cycle=1, first_seen="")
    assert rows["other"] == {"reinforcement_count": 9, "last_seen": "x"}
    assert "k1" in rows
