"""Tests for schema_config — single source of truth loader."""

from __future__ import annotations

from paramem.graph.schema_config import (
    anonymizer_placeholder_pattern,
    anonymizer_prefix_to_type,
    anonymizer_type_to_prefix,
    entity_types,
    fallback_entity_type,
    fallback_relation_type,
    format_entity_types,
    format_predicate_examples,
    format_relation_types,
    format_replacement_rules,
    load_schema_config,
    preferred_predicates,
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

    def test_missing_file_returns_hardcoded_fallback(self, tmp_path, monkeypatch):
        """A nonexistent path must return the hardcoded fallback, not a real YAML.

        Uses a sentinel to prove the fallback path was taken rather than the
        real schema.yaml (which also contains 'concept').
        """
        from paramem.graph import schema_config

        sentinel = {
            "entity_types": {"SENTINEL_TYPE": {"anchor": "test"}},
            "fallback_entity_type": "SENTINEL_TYPE",
            "relation_types": ["SENTINEL_RELATION"],
            "fallback_relation_type": "SENTINEL_RELATION",
            "preferred_predicates": [],
            "procedural_entity_types": ["SENTINEL_TYPE"],
            "procedural_predicate_groups": [],
        }
        monkeypatch.setattr(schema_config, "_HARDCODED_FALLBACK", sentinel)
        reset_cache()
        result = load_schema_config(str(tmp_path / "does_not_exist.yaml"))
        assert "SENTINEL_TYPE" in result["entity_types"]
        reset_cache()

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
            "preferred_predicates: []\n"
            "procedural_entity_types: [thing]\n"
            "procedural_predicate_groups: []\n"
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


class TestPreferredPredicates:
    def test_returns_list_of_dicts(self):
        reset_cache()
        preds = preferred_predicates()
        assert isinstance(preds, list)
        assert len(preds) > 0

    def test_each_group_has_label_and_nonempty_items(self):
        for group in preferred_predicates():
            assert "label" in group
            assert "items" in group
            assert isinstance(group["items"], list)
            assert len(group["items"]) > 0


class TestFormatEntityTypes:
    def test_full_scope_contains_fallback_annotation(self):
        reset_cache()
        result = format_entity_types()
        assert "(fallback — see rule below)" in result

    def test_full_scope_contains_every_type(self):
        reset_cache()
        result = format_entity_types()
        for t in entity_types():
            assert t in result

    def test_procedural_scope_contains_person_and_preference(self):
        reset_cache()
        result = format_entity_types(scope="procedural")
        assert "person" in result
        assert "preference" in result

    def test_procedural_scope_excludes_organization(self):
        reset_cache()
        result = format_entity_types(scope="procedural")
        assert "organization" not in result


class TestFormatPredicateExamples:
    def test_full_scope_contains_family_social_label(self):
        reset_cache()
        result = format_predicate_examples()
        assert "Family/social:" in result

    def test_full_scope_contains_married_to(self):
        reset_cache()
        result = format_predicate_examples()
        assert "married_to" in result

    def test_procedural_scope_contains_preferences_habits_label(self):
        reset_cache()
        result = format_predicate_examples(scope="procedural")
        assert "Preferences/habits:" in result

    def test_procedural_scope_excludes_married_to(self):
        reset_cache()
        result = format_predicate_examples(scope="procedural")
        assert "married_to" not in result


class TestFormatRelationTypes:
    def test_returns_nonempty_string(self):
        reset_cache()
        result = format_relation_types()
        assert isinstance(result, str)
        assert result

    def test_every_relation_type_appears(self):
        """Every value from relation_types() must be present in format_relation_types()."""
        reset_cache()
        result = format_relation_types()
        for rt in relation_types():
            assert rt in result, f"Relation type {rt!r} missing from format_relation_types()"

    def test_comma_separated(self):
        reset_cache()
        result = format_relation_types()
        parts = [p.strip() for p in result.split(",")]
        assert len(parts) == len(relation_types())


class TestEmptyAndMalformedYaml:
    """Tests for Fix 2: empty/partial/malformed YAML falls back with a logged error.

    Note: The ROS launch package (loaded by the ament pytest plugins) overrides
    the logging.Logger class and sets ``propagate=False`` on every new logger.
    This prevents pytest's caplog handler (which lives on the root logger) from
    capturing records from ``paramem.graph.schema_config``.  The workaround is to
    attach caplog's handler directly to the named logger and force propagation on
    before the call under test.
    """

    def _attach_caplog(self, caplog, level: int) -> tuple:
        """Attach caplog's handler to the schema_config logger directly.

        Returns ``(named_logger, orig_propagate)`` so the caller can restore state.
        """
        import logging

        named = logging.getLogger("paramem.graph.schema_config")
        orig_propagate = named.propagate
        named.propagate = True
        caplog.set_level(level, logger="paramem.graph.schema_config")
        named.addHandler(caplog.handler)
        return named, orig_propagate

    def _detach_caplog(self, caplog, named, orig_propagate: bool) -> None:
        named.removeHandler(caplog.handler)
        named.propagate = orig_propagate

    def test_empty_yaml_returns_fallback(self, tmp_path, caplog, monkeypatch):
        """An empty YAML file (missing all required keys) must return the hardcoded fallback."""
        import logging

        from paramem.graph import schema_config

        sentinel = {
            "entity_types": {"SENTINEL_TYPE": {"anchor": "test"}},
            "fallback_entity_type": "SENTINEL_TYPE",
            "relation_types": ["SENTINEL_RELATION"],
            "fallback_relation_type": "SENTINEL_RELATION",
            "preferred_predicates": [],
            "procedural_entity_types": ["SENTINEL_TYPE"],
            "procedural_predicate_groups": [],
        }
        monkeypatch.setattr(schema_config, "_HARDCODED_FALLBACK", sentinel)
        reset_cache()
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        named, orig_p = self._attach_caplog(caplog, logging.ERROR)
        try:
            cfg = load_schema_config(str(empty))
        finally:
            self._detach_caplog(caplog, named, orig_p)
        assert cfg == sentinel
        assert any("schema" in r.getMessage().lower() for r in caplog.records)
        reset_cache()

    def test_malformed_yaml_logs_error_and_falls_back(self, tmp_path, caplog, monkeypatch):
        """An unparseable YAML file must log at ERROR and return the hardcoded fallback."""
        import logging

        from paramem.graph import schema_config

        sentinel = {
            "entity_types": {"SENTINEL_TYPE": {"anchor": "test"}},
            "fallback_entity_type": "SENTINEL_TYPE",
            "relation_types": ["SENTINEL_RELATION"],
            "fallback_relation_type": "SENTINEL_RELATION",
            "preferred_predicates": [],
            "procedural_entity_types": ["SENTINEL_TYPE"],
            "procedural_predicate_groups": [],
        }
        monkeypatch.setattr(schema_config, "_HARDCODED_FALLBACK", sentinel)
        reset_cache()
        bad = tmp_path / "bad.yaml"
        bad.write_text(":\n  -invalid\n")  # unparseable
        named, orig_p = self._attach_caplog(caplog, logging.ERROR)
        try:
            cfg = load_schema_config(str(bad))
        finally:
            self._detach_caplog(caplog, named, orig_p)
        assert cfg == sentinel
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert error_records, "expected logger.error on malformed yaml"
        reset_cache()


class TestAnonymizerConfig:
    """Tests for anonymizer prefix helpers — single-source-of-truth guards."""

    def setup_method(self):
        reset_cache()

    def teardown_method(self):
        reset_cache()

    # ------------------------------------------------------------------ #
    # anonymizer_prefix_to_type                                           #
    # ------------------------------------------------------------------ #

    def test_prefix_to_type_returns_all_five_entries(self):
        """All five configured prefixes must appear as lowercased keys."""
        result = anonymizer_prefix_to_type()
        assert len(result) == 5
        for key in ("person", "city", "country", "org", "thing"):
            assert key in result, f"prefix {key!r} missing from anonymizer_prefix_to_type()"

    def test_prefix_to_type_city_maps_to_place(self):
        """Regression guard: 'city' must map to 'place', not 'location'."""
        result = anonymizer_prefix_to_type()
        assert result["city"] == "place", (
            "city prefix must map to entity_type 'place'; previously mapped to "
            "'location' which is not a valid entity_type and causes ValidationError."
        )

    def test_prefix_to_type_country_maps_to_place(self):
        """Regression guard: 'country' must also map to 'place', not 'location'."""
        result = anonymizer_prefix_to_type()
        assert result["country"] == "place", (
            "country prefix must map to entity_type 'place'; previously mapped to "
            "'location' which is not a valid entity_type and causes ValidationError."
        )

    def test_prefix_to_type_values_are_valid_entity_types(self):
        """Every mapped entity_type must be a valid configured entity type."""
        valid = set(entity_types())
        for prefix, etype in anonymizer_prefix_to_type().items():
            assert etype in valid, (
                f"prefix {prefix!r} maps to entity_type {etype!r} which is not "
                f"in entity_types(): {sorted(valid)}"
            )

    def test_prefix_to_type_city_resolves_to_valid_entity_type_for_entity_model(self):
        """Bug-fix verification: Entity(entity_type=anonymizer_prefix_to_type()['city'])
        must not raise ValidationError — 'place' is valid, 'location' is not."""
        from paramem.graph.schema import Entity

        etype = anonymizer_prefix_to_type()["city"]
        entity = Entity(name="Berlin", entity_type=etype)
        assert entity.entity_type == "place"

    # ------------------------------------------------------------------ #
    # anonymizer_type_to_prefix                                           #
    # ------------------------------------------------------------------ #

    def test_type_to_prefix_has_four_entries(self):
        """Only four entity types have a primary prefix (place has one, Country is not primary)."""
        result = anonymizer_type_to_prefix()
        assert len(result) == 4

    def test_type_to_prefix_place_maps_to_city_not_country(self):
        """'place' entity_type must map to 'City' (primary), not 'Country'."""
        result = anonymizer_type_to_prefix()
        assert result.get("place") == "City", (
            "place entity_type must map to primary prefix 'City'; "
            "'Country' is not primary_for_type."
        )
        assert "Country" not in result.values(), (
            "Country must not appear as a primary prefix value."
        )

    def test_type_to_prefix_contains_expected_types(self):
        """Expected primary types: person, place, organization, concept."""
        result = anonymizer_type_to_prefix()
        for etype in ("person", "place", "organization", "concept"):
            assert etype in result, (
                f"entity_type {etype!r} missing from anonymizer_type_to_prefix()"
            )

    def test_type_to_prefix_values_match_prefix_list(self):
        """Every value must be a prefix token from the configured prefix list."""
        from paramem.graph.schema_config import load_schema_config

        cfg = load_schema_config()
        all_prefixes = {e["prefix"] for e in cfg["anonymizer"]["prefixes"]}
        for etype, prefix in anonymizer_type_to_prefix().items():
            assert prefix in all_prefixes, (
                f"type_to_prefix returned prefix {prefix!r} for {etype!r} "
                f"which is not in the configured prefix list: {sorted(all_prefixes)}"
            )

    # ------------------------------------------------------------------ #
    # format_replacement_rules                                            #
    # ------------------------------------------------------------------ #

    def test_format_replacement_rules_has_five_lines(self):
        """One bullet line per configured prefix (5 prefixes → 5 lines)."""
        result = format_replacement_rules()
        lines = result.splitlines()
        assert len(lines) == 5, (
            f"Expected 5 lines in format_replacement_rules(), got {len(lines)}: {lines!r}"
        )

    def test_format_replacement_rules_each_line_starts_with_dash(self):
        """Each line must be a bullet: '- <description> → <Prefix>_1, <Prefix>_2, ...'"""
        for line in format_replacement_rules().splitlines():
            assert line.startswith("- "), f"Line does not start with '- ': {line!r}"

    def test_format_replacement_rules_contains_all_prefixes(self):
        """All five prefix tokens must appear in the rendered rules."""
        result = format_replacement_rules()
        for prefix in ("Person", "City", "Country", "Org", "Thing"):
            assert prefix in result, f"Prefix {prefix!r} missing from format_replacement_rules()"

    def test_format_replacement_rules_each_line_contains_arrow_and_examples(self):
        """Each line must contain the → arrow and at least one '_N' example."""
        for line in format_replacement_rules().splitlines():
            assert "→" in line, f"Missing arrow in line: {line!r}"
            assert "_1" in line, f"Missing example token '_1' in line: {line!r}"

    # ------------------------------------------------------------------ #
    # anonymizer_placeholder_pattern                                      #
    # ------------------------------------------------------------------ #

    def test_pattern_matches_valid_placeholders(self):
        """Standard placeholder tokens must match."""
        pat = anonymizer_placeholder_pattern()
        for token in ("Person_1", "City_42", "Country_3", "Org_10", "Thing_999"):
            assert pat.match(token), f"Pattern should match {token!r}"

    def test_pattern_is_case_insensitive(self):
        """Pattern must be case-insensitive per _PLACEHOLDER_RE convention."""
        pat = anonymizer_placeholder_pattern()
        assert pat.match("person_1"), "Pattern should match lowercase 'person_1'"
        assert pat.match("city_42"), "Pattern should match lowercase 'city_42'"
        assert pat.match("THING_5"), "Pattern should match uppercase 'THING_5'"

    def test_pattern_does_not_match_real_names(self):
        """Real names must not match the placeholder pattern."""
        pat = anonymizer_placeholder_pattern()
        for token in ("Alex", "Berlin", "Apple", ""):
            assert not pat.match(token), f"Pattern should NOT match {token!r}"

    def test_pattern_does_not_match_prefix_without_suffix(self):
        """'Person' without '_N' suffix must not match."""
        pat = anonymizer_placeholder_pattern()
        assert not pat.match("Person"), "Pattern should NOT match bare 'Person'"
        assert not pat.match("Person_"), "Pattern should NOT match 'Person_' (no digits)"

    def test_pattern_does_not_match_unconfigured_prefix(self):
        """'Location_1' is not a configured prefix — must not match."""
        pat = anonymizer_placeholder_pattern()
        assert not pat.match("Location_1"), (
            "Pattern should NOT match 'Location_1' — 'location' is not a configured prefix."
        )

    def test_pattern_returns_none_when_prefixes_empty(self, tmp_path):
        """Empty prefixes list must return None, not a degenerate regex."""
        tmp = tmp_path / "empty_prefixes.yaml"
        tmp.write_text(
            "entity_types:\n  person: {anchor: 'schema:Person'}\n"
            "fallback_entity_type: person\n"
            "relation_types: [factual]\n"
            "fallback_relation_type: factual\n"
            "preferred_predicates: []\n"
            "procedural_entity_types: [person]\n"
            "procedural_predicate_groups: []\n"
            "anonymizer:\n  prefixes: []\n"
        )
        reset_cache()
        result = anonymizer_placeholder_pattern(path=str(tmp))
        assert result is None, (
            "anonymizer_placeholder_pattern must return None when prefix list is empty, "
            f"got {result!r}"
        )
        reset_cache()

    def test_pattern_is_pattern_when_prefixes_nonempty(self):
        """Default config has prefixes configured — must return a compiled Pattern."""
        import re

        reset_cache()
        result = anonymizer_placeholder_pattern()
        assert isinstance(result, re.Pattern), (
            f"anonymizer_placeholder_pattern must return re.Pattern when prefixes are configured, "
            f"got {result!r}"
        )

    # ------------------------------------------------------------------ #
    # Fallback behaviour                                                   #
    # ------------------------------------------------------------------ #

    def test_fallback_when_anonymizer_key_missing(self, tmp_path, monkeypatch):
        """YAML missing 'anonymizer' key falls back to _HARDCODED_FALLBACK."""
        from paramem.graph import schema_config

        monkeypatch.setattr(schema_config, "_HARDCODED_FALLBACK", schema_config._HARDCODED_FALLBACK)
        reset_cache()
        # Write a YAML that is valid for the old required keys but lacks 'anonymizer'.
        minimal = tmp_path / "no_anon.yaml"
        minimal.write_text(
            "entity_types:\n  person: {anchor: 'schema:Person'}\n"
            "fallback_entity_type: person\n"
            "relation_types: [factual]\n"
            "fallback_relation_type: factual\n"
            "preferred_predicates: []\n"
        )
        cfg = load_schema_config(str(minimal))
        # Must have fallen back to hardcoded fallback which includes 'anonymizer'.
        assert "anonymizer" in cfg, (
            "Fallback must include 'anonymizer' key from _HARDCODED_FALLBACK."
        )
        # Helpers must still work via fallback.
        result = anonymizer_prefix_to_type(str(minimal))
        assert "city" in result
        assert result["city"] == "place"
        reset_cache()
