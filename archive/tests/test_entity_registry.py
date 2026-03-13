"""Tests for entity registry."""

from paramem.training.entity_registry import EntityEntry, EntityRegistry


class TestEntityEntry:
    def test_to_dict(self):
        entry = EntityEntry(
            name="Alex",
            first_seen="session_001",
            last_seen="session_005",
            session_count=3,
            tier="episodic",
            relation_count=5,
        )
        d = entry.to_dict()
        assert d["name"] == "Alex"
        assert d["session_count"] == 3
        assert d["tier"] == "episodic"

    def test_from_dict(self):
        d = {"name": "Alex", "first_seen": "s1", "last_seen": "s3", "session_count": 2}
        entry = EntityEntry.from_dict(d)
        assert entry.name == "Alex"
        assert entry.session_count == 2
        assert entry.tier == "episodic"  # default

    def test_roundtrip(self):
        entry = EntityEntry(
            name="Project Atlas",
            first_seen="s1",
            last_seen="s10",
            session_count=8,
            tier="semantic",
            relation_count=12,
        )
        restored = EntityEntry.from_dict(entry.to_dict())
        assert restored.name == entry.name
        assert restored.tier == entry.tier
        assert restored.relation_count == entry.relation_count


class TestEntityRegistry:
    def test_add_new_entity(self):
        reg = EntityRegistry()
        reg.add("Alex", "session_001", relation_count=3)
        assert "Alex" in reg
        assert len(reg) == 1
        entry = reg.get("Alex")
        assert entry.first_seen == "session_001"
        assert entry.session_count == 1

    def test_add_existing_updates(self):
        reg = EntityRegistry()
        reg.add("Alex", "session_001", relation_count=3)
        reg.add("Alex", "session_005", relation_count=5)
        assert len(reg) == 1
        entry = reg.get("Alex")
        assert entry.last_seen == "session_005"
        assert entry.session_count == 2
        assert entry.relation_count == 5

    def test_update_new_entity(self):
        reg = EntityRegistry()
        reg.update("Alex", "session_001", relation_count=2)
        assert "Alex" in reg
        assert reg.get("Alex").session_count == 1

    def test_update_existing(self):
        reg = EntityRegistry()
        reg.add("Alex", "session_001")
        reg.update("Alex", "session_003", relation_count=4)
        entry = reg.get("Alex")
        assert entry.last_seen == "session_003"
        assert entry.session_count == 2

    def test_remove(self):
        reg = EntityRegistry()
        reg.add("Alex", "s1")
        reg.update_fidelity("Alex", 0.8)
        reg.remove("Alex")
        assert "Alex" not in reg
        assert len(reg) == 0
        assert reg.get_fidelity_history("Alex") == []

    def test_list_active(self):
        reg = EntityRegistry()
        reg.add("Alex", "s1")
        reg.add("Maria", "s1")
        reg.add("Project Atlas", "s2")
        names = reg.list_active()
        assert len(names) == 3
        assert "Alex" in names

    def test_get_nonexistent(self):
        reg = EntityRegistry()
        assert reg.get("Nobody") is None

    def test_set_tier(self):
        reg = EntityRegistry()
        reg.add("Alex", "s1")
        reg.set_tier("Alex", "semantic")
        assert reg.get("Alex").tier == "semantic"

    def test_set_tier_nonexistent_is_noop(self):
        reg = EntityRegistry()
        reg.set_tier("Nobody", "semantic")  # should not raise

    def test_fidelity_tracking(self):
        reg = EntityRegistry()
        reg.add("Alex", "s1")
        reg.update_fidelity("Alex", 0.8)
        reg.update_fidelity("Alex", 0.6)
        reg.update_fidelity("Alex", 0.4)
        history = reg.get_fidelity_history("Alex")
        assert history == [0.8, 0.6, 0.4]
        assert reg.get_latest_fidelity("Alex") == 0.4

    def test_latest_fidelity_none(self):
        reg = EntityRegistry()
        assert reg.get_latest_fidelity("Nobody") is None

    def test_should_retire_not_enough_history(self):
        reg = EntityRegistry()
        reg.add("Alex", "s1")
        reg.update_fidelity("Alex", 0.05)
        reg.update_fidelity("Alex", 0.05)
        assert not reg.should_retire("Alex", threshold=0.1, consecutive_cycles=5)

    def test_should_retire_below_threshold(self):
        reg = EntityRegistry()
        reg.add("Alex", "s1")
        for _ in range(5):
            reg.update_fidelity("Alex", 0.05)
        assert reg.should_retire("Alex", threshold=0.1, consecutive_cycles=5)

    def test_should_not_retire_if_any_above(self):
        reg = EntityRegistry()
        reg.add("Alex", "s1")
        for _ in range(4):
            reg.update_fidelity("Alex", 0.05)
        reg.update_fidelity("Alex", 0.5)  # one good cycle
        assert not reg.should_retire("Alex", threshold=0.1, consecutive_cycles=5)

    def test_save_and_load(self, tmp_path):
        reg = EntityRegistry()
        reg.add("Alex", "session_001", relation_count=5)
        reg.add("Maria", "session_002", relation_count=3)
        reg.update_fidelity("Alex", 0.8)
        reg.update_fidelity("Alex", 0.6)
        reg.set_tier("Maria", "semantic")

        path = tmp_path / "entity_registry.json"
        reg.save(path)

        loaded = EntityRegistry.load(path)
        assert len(loaded) == 2
        assert "Alex" in loaded
        assert "Maria" in loaded
        assert loaded.get("Alex").relation_count == 5
        assert loaded.get("Maria").tier == "semantic"
        assert loaded.get_fidelity_history("Alex") == [0.8, 0.6]

    def test_load_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        reg = EntityRegistry.load(path)
        assert len(reg) == 0

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "registry.json"
        reg = EntityRegistry()
        reg.add("Alex", "s1")
        reg.save(path)
        assert path.exists()

    def test_contains(self):
        reg = EntityRegistry()
        reg.add("Alex", "s1")
        assert "Alex" in reg
        assert "Nobody" not in reg

    def test_list_by_tier(self):
        reg = EntityRegistry()
        reg.add("Alex", "s1")
        reg.add("Maria", "s1")
        reg.add("Project Atlas", "s2")
        reg.set_tier("Maria", "semantic")
        reg.set_tier("Project Atlas", "semantic")

        episodic = reg.list_by_tier("episodic")
        semantic = reg.list_by_tier("semantic")
        assert episodic == ["Alex"]
        assert set(semantic) == {"Maria", "Project Atlas"}

    def test_list_by_tier_empty(self):
        reg = EntityRegistry()
        reg.add("Alex", "s1")
        assert reg.list_by_tier("semantic") == []

    def test_len(self):
        reg = EntityRegistry()
        assert len(reg) == 0
        reg.add("Alex", "s1")
        reg.add("Maria", "s1")
        assert len(reg) == 2
