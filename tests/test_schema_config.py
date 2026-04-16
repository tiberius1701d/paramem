"""Tests for schema_config — single source of truth loader."""

from __future__ import annotations

from paramem.graph.schema_config import (
    entity_types,
    fallback_entity_type,
    fallback_relation_type,
    format_entity_types,
    format_predicate_examples,
    format_relation_types,
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
