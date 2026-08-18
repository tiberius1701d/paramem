"""Unit tests for paramem.server.consolidation.

No existing test file targets this module (``tests/test_consolidation.py``
targets ``paramem.training.consolidation.ConsolidationLoop`` instead) --
``load_max_tier_cycle`` is the per-tier boot-seed reader that replaced a
single global ``key_metadata.json`` reader, with zero prior coverage.

Covers:
- load_max_tier_cycle: no files anywhere -> None; per-tier cycle_count
  derivation (max across tiers).
"""

from __future__ import annotations

from pathlib import Path

from paramem.backup.encryption import write_infra_json
from paramem.server.consolidation import load_max_tier_cycle


def _write_key_metadata(tier_root: Path, keys: dict, *, tier_cycle: int = 0) -> None:
    tier_root.mkdir(parents=True, exist_ok=True)
    write_infra_json(tier_root / "key_metadata.json", {"tier_cycle": tier_cycle, "keys": keys})


class TestLoadMaxTierCycle:
    def test_no_files_anywhere_returns_none(self, tmp_path):
        assert load_max_tier_cycle(tmp_path / "adapters") is None

    def test_cycle_count_is_the_max_across_tiers(self, tmp_path):
        adapter_dir = tmp_path / "adapters"
        _write_key_metadata(adapter_dir / "episodic", {"g0": {}}, tier_cycle=2)
        _write_key_metadata(adapter_dir / "semantic", {"g1": {}}, tier_cycle=7)

        result = load_max_tier_cycle(adapter_dir)

        assert result == 7
