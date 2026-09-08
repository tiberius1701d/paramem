"""Tests for paramem.config.taxonomy — single source of truth loader."""

from __future__ import annotations

from pathlib import Path

import pytest

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
            "anonymizer:\n  scrub: []\n  allow: []\n"
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


class TestAnonymizerVocabularyShippedRows:
    """Pins the shipped ``configs/schema.yaml`` two-list anonymizer table:

    row order, entity type and description for every ``scrub`` and
    ``allow`` row, which rows carry a ``primary_for_type`` claim, and the
    ``scrub`` / ``allow`` split on ``scrub_categories`` hints.
    """

    def test_scrub_rows_in_order(self):
        reset_cache()
        rows = load_schema_config()["anonymizer"]["scrub"]
        assert [(row["prefix"], row["entity_type"], row["description"]) for row in rows] == [
            ("Person", "person", "a person's given name, family name or full name"),
            ("Phone", "person", "a telephone number"),
            ("Email", "person", "an email address"),
            (
                "Address",
                "person",
                "a street address (street plus number, optionally postal code)",
            ),
            ("Profile", "person", "a profile URL or social-media handle"),
        ]

    def test_allow_rows_in_order(self):
        reset_cache()
        rows = load_schema_config()["anonymizer"]["allow"]
        assert [(row["prefix"], row["entity_type"], row["description"]) for row in rows] == [
            ("City", "place", "a city, town or village name"),
            ("Country", "place", "a country name"),
            ("Org", "organization", "a company, institution, club or organisation name"),
            ("Profession", "concept", "a job title or profession"),
            ("Artist", "person", "a musician, band, author or performer name"),
        ]

    def test_primaries_are_person_city_and_org_only(self):
        reset_cache()
        anonymizer = load_schema_config()["anonymizer"]
        primaries = {
            row["prefix"]
            for row in anonymizer["scrub"] + anonymizer["allow"]
            if row.get("primary_for_type", False)
        }
        assert primaries == {"Person", "City", "Org"}

    def test_every_scrub_row_carries_hints_and_no_allow_row_does(self):
        reset_cache()
        anonymizer = load_schema_config()["anonymizer"]
        assert all(row.get("scrub_categories") for row in anonymizer["scrub"])
        assert all(not row.get("scrub_categories") for row in anonymizer["allow"])


class TestPrefixDescriptions:
    """``prefix_descriptions()`` returns every row of both anonymizer
    lists, scrub rows then allow rows, each in its own declared order.
    """

    def test_ten_pairs_scrub_then_allow_in_order(self):
        reset_cache()
        assert prefix_descriptions() == (
            ("Person", "a person's given name, family name or full name"),
            ("Phone", "a telephone number"),
            ("Email", "an email address"),
            ("Address", "a street address (street plus number, optionally postal code)"),
            ("Profile", "a profile URL or social-media handle"),
            ("City", "a city, town or village name"),
            ("Country", "a country name"),
            ("Org", "a company, institution, club or organisation name"),
            ("Profession", "a job title or profession"),
            ("Artist", "a musician, band, author or performer name"),
        )

    def test_custom_path_puts_scrub_rows_before_allow_rows_regardless_of_file_order(self, tmp_path):
        """A schema file that declares ``allow`` before ``scrub`` still
        returns scrub rows first — list identity, not file position,
        decides order.
        """
        schema = tmp_path / "schema.yaml"
        schema.write_text(
            "entity_types:\n"
            "  person: {anchor: 'schema:Person'}\n"
            "  place: {anchor: 'schema:Place'}\n"
            "fallback_entity_type: person\n"
            "relation_types: [factual]\n"
            "fallback_relation_type: factual\n"
            "anonymizer:\n"
            "  allow:\n"
            "    - { prefix: City, entity_type: place, description: 'a city' }\n"
            "  scrub:\n"
            "    - { prefix: Name, entity_type: person, description: 'a name', "
            "scrub_categories: [name] }\n"
        )
        reset_cache()
        assert prefix_descriptions(path=str(schema)) == (
            ("Name", "a name"),
            ("City", "a city"),
        )
        reset_cache()


class TestMissingAnonymizerListRaises:
    """``load_schema_config`` requires both ``anonymizer.scrub`` and
    ``anonymizer.allow`` to be present AND a list — an empty list is
    legal, an absent key or a null value (a YAML key with nothing under
    it) is not.
    """

    def _write(self, tmp_path, anonymizer_block: str) -> Path:
        schema = tmp_path / "schema.yaml"
        schema.write_text(
            "entity_types:\n  person: {anchor: 'schema:Person'}\n"
            "fallback_entity_type: person\n"
            "relation_types: [factual]\n"
            "fallback_relation_type: factual\n"
            f"anonymizer:\n{anonymizer_block}\n"
        )
        return schema

    def test_missing_scrub_list_raises(self, tmp_path):
        schema = self._write(tmp_path, "  allow: []\n")
        with pytest.raises(ValueError, match=r"list\(s\): \['scrub'\]"):
            load_schema_config(str(schema))

    def test_missing_allow_list_raises(self, tmp_path):
        schema = self._write(tmp_path, "  scrub: []\n")
        with pytest.raises(ValueError, match=r"list\(s\): \['allow'\]"):
            load_schema_config(str(schema))

    def test_null_scrub_list_raises(self, tmp_path):
        """A YAML key with nothing under it (``scrub:`` alone) parses to
        ``None``, not to a missing key — the loader must refuse this the
        same way it refuses an absent key, rather than passing a ``None``
        on to a reader that would crash with a bare ``TypeError`` trying
        to iterate it.
        """
        schema = self._write(tmp_path, "  scrub:\n  allow: []\n")
        with pytest.raises(ValueError, match=r"list\(s\): \['scrub'\]"):
            load_schema_config(str(schema))
