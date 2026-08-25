"""Tests for paramem.config.taxonomy — single source of truth loader."""

from __future__ import annotations

from paramem.config.taxonomy import (
    entity_types,
    fallback_entity_type,
    fallback_relation_type,
    load_schema_config,
    relation_types,
    reset_cache,
)


class TestLoadSchemaConfig:
    def test_reads_real_yaml(self):
        reset_cache()
        cfg = load_schema_config()
        assert isinstance(cfg, dict)
        assert "entity_types" in cfg
        assert "relation_types" in cfg

    def test_cache_cleared_by_reset(self, tmp_path):
        """After reset_cache, loading from a different path returns a different object."""
        reset_cache()
        first = load_schema_config()
        reset_cache()
        alt_yaml = tmp_path / "alt_schema.yaml"
        alt_yaml.write_text(
            "entity_types:\n  thing: {anchor: 'test'}\n"
            "fallback_entity_type: thing\n"
            "relation_types: [factual]\n"
            "fallback_relation_type: factual\n"
            "anonymizer:\n  prefixes: []\n"
        )
        second = load_schema_config(str(alt_yaml))
        assert first is not second
        assert "thing" in second["entity_types"]
        reset_cache()


class TestEntityTypes:
    def test_returns_nonempty_tuple(self):
        reset_cache()
        types = entity_types()
        assert isinstance(types, tuple)
        assert len(types) > 0

    def test_all_strings(self):
        for t in entity_types():
            assert isinstance(t, str)

    def test_fallback_in_entity_types(self):
        assert fallback_entity_type() in entity_types()


class TestRelationTypes:
    def test_returns_nonempty_tuple(self):
        reset_cache()
        types = relation_types()
        assert isinstance(types, tuple)
        assert len(types) > 0

    def test_all_strings(self):
        for t in relation_types():
            assert isinstance(t, str)

    def test_fallback_in_relation_types(self):
        assert fallback_relation_type() in relation_types()
