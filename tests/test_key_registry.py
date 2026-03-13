"""Tests for the key registry."""

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
