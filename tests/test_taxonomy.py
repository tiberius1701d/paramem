"""Tests for paramem.config.taxonomy — single source of truth loader."""

from __future__ import annotations

from paramem.config.taxonomy import (
    anonymizer_prefix_to_type,
    anonymizer_type_to_prefix,
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

    def test_missing_file_returns_hardcoded_fallback(self, tmp_path, monkeypatch):
        """A nonexistent path must return the hardcoded fallback, not a real YAML.

        Uses a sentinel to prove the fallback path was taken rather than the
        real schema.yaml (which also contains 'concept').
        """
        from paramem.config import taxonomy

        sentinel = {
            "entity_types": {"SENTINEL_TYPE": {"anchor": "test"}},
            "fallback_entity_type": "SENTINEL_TYPE",
            "relation_types": ["SENTINEL_RELATION"],
            "fallback_relation_type": "SENTINEL_RELATION",
        }
        monkeypatch.setattr(taxonomy, "_HARDCODED_FALLBACK", sentinel)
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


class TestEmptyAndMalformedYaml:
    """Tests for Fix 2: empty/partial/malformed YAML falls back with a logged error.

    Note: The ROS launch package (loaded by the ament pytest plugins) overrides
    the logging.Logger class and sets ``propagate=False`` on every new logger.
    This prevents pytest's caplog handler (which lives on the root logger) from
    capturing records from ``paramem.config.taxonomy``.  The workaround is to
    attach caplog's handler directly to the named logger and force propagation on
    before the call under test.
    """

    def _attach_caplog(self, caplog, level: int) -> tuple:
        """Attach caplog's handler to the taxonomy logger directly.

        Returns ``(named_logger, orig_propagate)`` so the caller can restore state.
        """
        import logging

        named = logging.getLogger("paramem.config.taxonomy")
        orig_propagate = named.propagate
        named.propagate = True
        caplog.set_level(level, logger="paramem.config.taxonomy")
        named.addHandler(caplog.handler)
        return named, orig_propagate

    def _detach_caplog(self, caplog, named, orig_propagate: bool) -> None:
        named.removeHandler(caplog.handler)
        named.propagate = orig_propagate

    def test_empty_yaml_returns_fallback(self, tmp_path, caplog, monkeypatch):
        """An empty YAML file (missing all required keys) must return the hardcoded fallback."""
        import logging

        from paramem.config import taxonomy

        sentinel = {
            "entity_types": {"SENTINEL_TYPE": {"anchor": "test"}},
            "fallback_entity_type": "SENTINEL_TYPE",
            "relation_types": ["SENTINEL_RELATION"],
            "fallback_relation_type": "SENTINEL_RELATION",
        }
        monkeypatch.setattr(taxonomy, "_HARDCODED_FALLBACK", sentinel)
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

        from paramem.config import taxonomy

        sentinel = {
            "entity_types": {"SENTINEL_TYPE": {"anchor": "test"}},
            "fallback_entity_type": "SENTINEL_TYPE",
            "relation_types": ["SENTINEL_RELATION"],
            "fallback_relation_type": "SENTINEL_RELATION",
        }
        monkeypatch.setattr(taxonomy, "_HARDCODED_FALLBACK", sentinel)
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
        from paramem.config.taxonomy import load_schema_config

        cfg = load_schema_config()
        all_prefixes = {e["prefix"] for e in cfg["anonymizer"]["prefixes"]}
        for etype, prefix in anonymizer_type_to_prefix().items():
            assert prefix in all_prefixes, (
                f"type_to_prefix returned prefix {prefix!r} for {etype!r} "
                f"which is not in the configured prefix list: {sorted(all_prefixes)}"
            )

    # ------------------------------------------------------------------ #
    # Map-key identity contract (both maps keyed by canonical())           #
    # ------------------------------------------------------------------ #

    def test_prefix_to_type_keys_are_canonical(self):
        """Keys of the prefix map are produced by canonical(), not a local fold.

        The sole lookup site (placeholders.prefix_to_entity_type) canonicalizes
        its query, so a key built by any other routine could silently miss.
        """
        from paramem.utils.identity import canonical

        for key in anonymizer_prefix_to_type():
            assert canonical(key) == key, (
                f"prefix-map key {key!r} is not canonical — the lookup side "
                f"canonicalizes its query, so this key would silently miss."
            )

    def test_type_to_prefix_keys_are_canonical(self):
        """Keys of the entity_type map are produced by canonical(), not a local fold.

        Regression guard for the one-sided contract this map previously had:
        keys came straight from the YAML while the lookup site
        (placeholders.entity_type_to_prefix) folded its query.
        """
        from paramem.utils.identity import canonical

        for key in anonymizer_type_to_prefix():
            assert canonical(key) == key, (
                f"type-map key {key!r} is not canonical — the lookup side "
                f"canonicalizes its query, so this key would silently miss."
            )

    def test_type_to_prefix_survives_cased_and_spaced_yaml(self, tmp_path):
        """A cased/spaced YAML entity_type still resolves via the canonical lookup.

        This is what the one-sided contract could not do: the raw key
        ``"Work Of Art"`` never matched the folded query.
        """
        from paramem.config.taxonomy import reset_cache
        from paramem.utils.identity import canonical

        schema = tmp_path / "schema.yaml"
        schema.write_text(
            "entity_types:\n"
            "  concept:\n"
            "    anchor: schema:Thing\n"
            "fallback_entity_type: concept\n"
            "relation_types: [factual]\n"
            "fallback_relation_type: factual\n"
            "anonymizer:\n"
            "  prefixes:\n"
            "    - prefix: WorkOfArt\n"
            "      entity_type: Work Of Art\n"
            "      description: art\n"
            "      primary_for_type: true\n"
        )
        reset_cache()
        try:
            built = anonymizer_type_to_prefix(str(schema))
            assert built == {"work_of_art": "WorkOfArt"}
            # The lookup side canonicalizes its query identically, so every
            # surface variant of the same type lands on the one built key.
            for variant in ("Work Of Art", "work of art", "WORK  OF  ART", "work_of_art"):
                assert built[canonical(variant)] == "WorkOfArt"
        finally:
            reset_cache()

    # ------------------------------------------------------------------ #
    # Fallback behaviour                                                   #
    # ------------------------------------------------------------------ #

    def test_fallback_when_anonymizer_key_missing(self, tmp_path, monkeypatch):
        """YAML missing 'anonymizer' key falls back to _HARDCODED_FALLBACK."""
        from paramem.config import taxonomy

        monkeypatch.setattr(taxonomy, "_HARDCODED_FALLBACK", taxonomy._HARDCODED_FALLBACK)
        reset_cache()
        # Write a YAML that is valid for the old required keys but lacks 'anonymizer'.
        minimal = tmp_path / "no_anon.yaml"
        minimal.write_text(
            "entity_types:\n  person: {anchor: 'schema:Person'}\n"
            "fallback_entity_type: person\n"
            "relation_types: [factual]\n"
            "fallback_relation_type: factual\n"
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
