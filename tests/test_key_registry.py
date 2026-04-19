"""Tests for the key registry."""

import json

from paramem.training.key_registry import KeyRegistry


class TestKeyRegistry:
    def test_add_and_list(self):
        reg = KeyRegistry()
        reg.add("session_001")
        reg.add("session_002")
        assert reg.list_active() == ["session_001", "session_002"]

    def test_add_duplicate_ignored(self):
        reg = KeyRegistry()
        reg.add("session_001")
        reg.add("session_001")
        assert len(reg) == 1

    def test_remove(self):
        reg = KeyRegistry()
        reg.add("session_001")
        reg.add("session_002")
        reg.remove("session_001")
        assert reg.list_active() == ["session_002"]
        assert "session_001" not in reg

    def test_remove_nonexistent(self):
        reg = KeyRegistry()
        reg.remove("nonexistent")  # should not raise

    def test_contains(self):
        reg = KeyRegistry()
        reg.add("session_001")
        assert "session_001" in reg
        assert "session_002" not in reg

    def test_len(self):
        reg = KeyRegistry()
        assert len(reg) == 0
        reg.add("a")
        reg.add("b")
        assert len(reg) == 2


class TestFidelityTracking:
    def test_update_and_get_history(self):
        reg = KeyRegistry()
        reg.add("key_a")
        reg.update_fidelity("key_a", 0.9)
        reg.update_fidelity("key_a", 0.85)
        assert reg.get_fidelity_history("key_a") == [0.9, 0.85]

    def test_latest_fidelity(self):
        reg = KeyRegistry()
        reg.add("key_a")
        assert reg.get_latest_fidelity("key_a") is None
        reg.update_fidelity("key_a", 0.9)
        reg.update_fidelity("key_a", 0.7)
        assert reg.get_latest_fidelity("key_a") == 0.7

    def test_empty_history(self):
        reg = KeyRegistry()
        assert reg.get_fidelity_history("unknown") == []

    def test_remove_clears_fidelity(self):
        reg = KeyRegistry()
        reg.add("key_a")
        reg.update_fidelity("key_a", 0.5)
        reg.remove("key_a")
        assert reg.get_fidelity_history("key_a") == []


class TestRetirement:
    def test_should_retire_sustained_low(self):
        reg = KeyRegistry()
        reg.add("key_a")
        reg.update_fidelity("key_a", 0.05)
        reg.update_fidelity("key_a", 0.08)
        reg.update_fidelity("key_a", 0.03)
        assert reg.should_retire("key_a", threshold=0.1, consecutive_cycles=3)

    def test_should_not_retire_not_enough_cycles(self):
        reg = KeyRegistry()
        reg.add("key_a")
        reg.update_fidelity("key_a", 0.05)
        reg.update_fidelity("key_a", 0.08)
        assert not reg.should_retire("key_a", threshold=0.1, consecutive_cycles=3)

    def test_should_not_retire_recent_recovery(self):
        reg = KeyRegistry()
        reg.add("key_a")
        reg.update_fidelity("key_a", 0.05)
        reg.update_fidelity("key_a", 0.05)
        reg.update_fidelity("key_a", 0.5)  # recovered
        assert not reg.should_retire("key_a", threshold=0.1, consecutive_cycles=3)

    def test_should_not_retire_unknown_key(self):
        reg = KeyRegistry()
        assert not reg.should_retire("unknown")


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "registry.json"

        reg = KeyRegistry()
        reg.add("session_001")
        reg.add("session_002")
        reg.update_fidelity("session_001", 0.9)
        reg.update_fidelity("session_001", 0.85)
        reg.save(path)

        loaded = KeyRegistry.load(path)
        assert loaded.list_active() == ["session_001", "session_002"]
        assert loaded.get_fidelity_history("session_001") == [0.9, 0.85]
        assert loaded.get_fidelity_history("session_002") == []

    def test_load_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        loaded = KeyRegistry.load(path)
        assert len(loaded) == 0

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "registry.json"
        reg = KeyRegistry()
        reg.add("key")
        reg.save(path)
        assert path.exists()

    def test_roundtrip_preserves_order(self, tmp_path):
        path = tmp_path / "registry.json"
        reg = KeyRegistry()
        for i in range(10):
            reg.add(f"key_{i:02d}")
        reg.save(path)
        loaded = KeyRegistry.load(path)
        assert loaded.list_active() == [f"key_{i:02d}" for i in range(10)]


class TestAdapterIdField:
    """Step 1: adapter_id per key."""

    def test_add_default_adapter_id_main(self):
        """Calling add() without adapter_id records 'main'."""
        reg = KeyRegistry()
        reg.add("k1")
        assert reg.get_adapter_id("k1") == "main"

    def test_add_with_explicit_adapter_id(self):
        """Passing adapter_id='episodic_interim_20260418T0900' records that value."""
        reg = KeyRegistry()
        reg.add("k2", adapter_id="episodic_interim_20260418T0900")
        assert reg.get_adapter_id("k2") == "episodic_interim_20260418T0900"

    def test_get_adapter_id_missing_key_returns_main(self):
        """get_adapter_id on a key not in the registry returns 'main'."""
        reg = KeyRegistry()
        assert reg.get_adapter_id("nonexistent") == "main"

    def test_set_adapter_id_overwrites(self):
        """set_adapter_id reassigns an existing key's adapter."""
        reg = KeyRegistry()
        reg.add("k3", adapter_id="episodic_interim_20260417T0900")
        reg.set_adapter_id("k3", "episodic")
        assert reg.get_adapter_id("k3") == "episodic"

    def test_keys_for_adapter_groups_correctly(self):
        """keys_for_adapter returns only keys belonging to the requested adapter."""
        reg = KeyRegistry()
        reg.add("k1", adapter_id="main")
        reg.add("k2", adapter_id="episodic_interim_20260418T0900")
        reg.add("k3", adapter_id="main")
        reg.add("k4", adapter_id="episodic_interim_20260419T0900")
        assert reg.keys_for_adapter("main") == ["k1", "k3"]
        assert reg.keys_for_adapter("episodic_interim_20260418T0900") == ["k2"]
        assert reg.keys_for_adapter("episodic_interim_20260419T0900") == ["k4"]
        assert reg.keys_for_adapter("episodic") == []

    def test_keys_for_adapter_preserves_registration_order(self):
        """keys_for_adapter returns keys in the order they were registered."""
        reg = KeyRegistry()
        reg.add("z", adapter_id="episodic")
        reg.add("a", adapter_id="episodic")
        reg.add("m", adapter_id="episodic")
        assert reg.keys_for_adapter("episodic") == ["z", "a", "m"]

    def test_remove_clears_adapter_id(self):
        """remove() also clears the adapter_id entry for the key."""
        reg = KeyRegistry()
        reg.add("k1", adapter_id="episodic_interim_20260417T0900")
        reg.remove("k1")
        # Key is gone; get_adapter_id returns the "main" default, not the old value.
        assert "k1" not in reg
        assert reg.get_adapter_id("k1") == "main"

    def test_load_legacy_registry_defaults_main(self, tmp_path):
        """Load a JSON file with keys lacking adapter_id; all load with 'main'."""
        path = tmp_path / "legacy_registry.json"
        legacy = {
            "active_keys": ["graph1", "graph2", "graph3"],
            "fidelity_history": {
                "graph1": [0.9, 0.85],
                "graph2": [],
            },
            # Deliberately omit "adapter_id" to simulate a pre-Step-1 file.
        }
        path.write_text(json.dumps(legacy))

        loaded = KeyRegistry.load(path)
        assert loaded.list_active() == ["graph1", "graph2", "graph3"]
        assert loaded.get_adapter_id("graph1") == "main"
        assert loaded.get_adapter_id("graph2") == "main"
        assert loaded.get_adapter_id("graph3") == "main"

    def test_save_load_roundtrip_preserves_adapter_id(self, tmp_path):
        """Save a registry with mixed adapter_ids; reload; assert preserved."""
        path = tmp_path / "registry.json"

        reg = KeyRegistry()
        reg.add("graph1", adapter_id="episodic")
        reg.add("graph2", adapter_id="episodic_interim_20260418T0900")
        reg.add("graph3")  # default "main"
        reg.update_fidelity("graph1", 0.95)
        reg.save(path)

        loaded = KeyRegistry.load(path)
        assert loaded.get_adapter_id("graph1") == "episodic"
        assert loaded.get_adapter_id("graph2") == "episodic_interim_20260418T0900"
        assert loaded.get_adapter_id("graph3") == "main"
        # Existing fields still intact.
        assert loaded.list_active() == ["graph1", "graph2", "graph3"]
        assert loaded.get_fidelity_history("graph1") == [0.95]

    def test_consolidation_existing_calls_still_work(self):
        """Smoke test: consolidation.py's positional add(key) signature still works.

        Validates that no signature breakage was introduced by the new
        optional adapter_id kwarg.  Mirrors the 5 call-site patterns found
        in paramem/training/consolidation.py.
        """
        reg = KeyRegistry()

        # Pattern 1: bare positional key (consolidation.py:287, :641, :768, :1059, :1330)
        reg.add("graph1")
        assert "graph1" in reg
        assert reg.get_adapter_id("graph1") == "main"

        # Pattern 2: explicit kwarg (Step 6 call sites)
        reg.add("graph2", adapter_id="episodic_interim_20260418T0900")
        assert reg.get_adapter_id("graph2") == "episodic_interim_20260418T0900"

        # Duplicate add (consolidation re-seeds on reload) must not crash.
        reg.add("graph1")
        assert len(reg) == 2
