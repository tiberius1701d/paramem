"""``load_schema_config`` raise cases and the valid-load/cache/reset_cache
contract.
"""

from __future__ import annotations

import pytest

from paramem.config import taxonomy


@pytest.fixture(autouse=True)
def _reset_cache():
    taxonomy.reset_cache()
    yield
    taxonomy.reset_cache()


class TestLoadSchemaConfigRaises:
    def test_missing_file_raises_value_error_naming_path(self, tmp_path) -> None:
        missing = tmp_path / "nope.yaml"
        with pytest.raises(ValueError) as exc_info:
            taxonomy.load_schema_config(str(missing))
        message = str(exc_info.value)
        assert str(missing) in message
        # Remediation language, not just a bare traceback.
        assert "no fallback" in message.lower()

    def test_unparseable_yaml_raises_value_error(self, tmp_path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("entity_types: [this is: not: valid: yaml")
        with pytest.raises(ValueError) as exc_info:
            taxonomy.load_schema_config(str(bad))
        assert str(bad) in str(exc_info.value)

    def test_missing_required_key_raises_value_error_naming_the_key(self, tmp_path) -> None:
        incomplete = tmp_path / "incomplete.yaml"
        incomplete.write_text(
            "entity_types:\n  person: {anchor: 'schema:Person'}\n"
            "fallback_entity_type: person\n"
            "relation_types: [factual]\n"
            "fallback_relation_type: factual\n"
            # anonymizer key deliberately omitted
        )
        with pytest.raises(ValueError) as exc_info:
            taxonomy.load_schema_config(str(incomplete))
        message = str(exc_info.value)
        assert "anonymizer" in message
        assert str(incomplete) in message

    def test_there_is_no_fallback_dict_returned_on_any_failure(self, tmp_path) -> None:
        missing = tmp_path / "nope.yaml"
        with pytest.raises(ValueError):
            result = taxonomy.load_schema_config(str(missing))
            # If load_schema_config ever "succeeded" with a fallback, it
            # would return a dict rather than raise — assert the raise
            # itself is what happens, never a returned empty/default dict.
            assert result is None  # pragma: no cover - unreachable on raise


class TestLoadSchemaConfigValidLoadCachesAndResets:
    def test_valid_file_loads_and_returns_all_required_keys(self) -> None:
        cfg = taxonomy.load_schema_config()
        assert "entity_types" in cfg
        assert "fallback_entity_type" in cfg
        assert "relation_types" in cfg
        assert "fallback_relation_type" in cfg
        assert "anonymizer" in cfg

    def test_repeat_calls_with_the_same_path_are_cached(self, monkeypatch) -> None:
        calls = []
        real_read_text = __import__("pathlib").Path.read_text

        def _tracking_read_text(self, *args, **kwargs):
            calls.append(str(self))
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr("pathlib.Path.read_text", _tracking_read_text)

        taxonomy.load_schema_config()
        taxonomy.load_schema_config()

        assert len(calls) == 1, "second call must be served from the lru_cache, not re-read"

    def test_reset_cache_forces_a_re_read(self, monkeypatch) -> None:
        calls = []
        real_read_text = __import__("pathlib").Path.read_text

        def _tracking_read_text(self, *args, **kwargs):
            calls.append(str(self))
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr("pathlib.Path.read_text", _tracking_read_text)

        taxonomy.load_schema_config()
        taxonomy.reset_cache()
        taxonomy.load_schema_config()

        assert len(calls) == 2, "reset_cache must force a fresh read on the next call"

    def test_different_explicit_paths_load_independently(self, tmp_path) -> None:
        custom = tmp_path / "custom_schema.yaml"
        custom.write_text(
            "entity_types:\n  thing: {anchor: 'schema:Thing'}\n"
            "fallback_entity_type: thing\n"
            "relation_types: [factual]\n"
            "fallback_relation_type: factual\n"
            "anonymizer:\n  prefixes: []\n"
        )
        default_cfg = taxonomy.load_schema_config()
        custom_cfg = taxonomy.load_schema_config(str(custom))
        assert custom_cfg != default_cfg
        assert custom_cfg["fallback_entity_type"] == "thing"
