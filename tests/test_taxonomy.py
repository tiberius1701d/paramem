"""Tests for paramem.config.taxonomy — single source of truth loader."""

from __future__ import annotations

from paramem.config.taxonomy import (
    entity_types,
    fallback_entity_type,
    fallback_relation_type,
    load_schema_config,
    prefix_descriptions,
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


class TestPrefixDescriptions:
    """THE one accessor the SCAN prompt's ``{keywords}`` slot renders
    from — table order, every row whether scrubbed or not.
    """

    def test_returns_a_pair_per_row_in_table_order(self):
        reset_cache()
        pairs = prefix_descriptions()
        assert isinstance(pairs, tuple)
        assert pairs[0][0] == "Person"
        prefixes = [p for p, _d in pairs]
        assert prefixes.index("Person") < prefixes.index("City")

    def test_every_description_is_a_non_empty_string(self):
        for prefix, description in prefix_descriptions():
            assert isinstance(prefix, str) and prefix
            assert isinstance(description, str) and description

    def test_reads_a_custom_path(self, tmp_path):
        alt_yaml = tmp_path / "alt_schema.yaml"
        alt_yaml.write_text(
            "entity_types:\n  thing: {anchor: 'test'}\n"
            "fallback_entity_type: thing\n"
            "relation_types: [factual]\n"
            "fallback_relation_type: factual\n"
            "anonymizer:\n"
            "  prefixes:\n"
            '    - { prefix: Thing, entity_type: thing, description: "a thing" }\n'
        )
        assert prefix_descriptions(str(alt_yaml)) == (("Thing", "a thing"),)


class TestShippedAnonymizerPrefixTable:
    """The seven rows the shipped table carries beside the four primaries
    (Person, City, Org, Thing): their ``entity_type`` and that none of
    them is ``primary_for_type``.
    """

    _EXPECTED_ENTITY_TYPE = {
        "Product": "concept",
        "Room": "place",
        "Profession": "concept",
        "Artist": "person",
        "Work": "concept",
        "Date": "event",
        "Other": "person",
    }

    def test_each_row_declares_the_stated_entity_type(self):
        reset_cache()
        cfg = load_schema_config()
        rows = {row["prefix"]: row for row in cfg["anonymizer"]["prefixes"]}
        for prefix, expected_type in self._EXPECTED_ENTITY_TYPE.items():
            assert prefix in rows, f"missing shipped row {prefix!r}"
            assert rows[prefix]["entity_type"] == expected_type

    def test_none_of_the_seven_is_primary_for_type(self):
        reset_cache()
        cfg = load_schema_config()
        rows = {row["prefix"]: row for row in cfg["anonymizer"]["prefixes"]}
        for prefix in self._EXPECTED_ENTITY_TYPE:
            assert not rows[prefix].get("primary_for_type", False)
