"""``resolve_scrub_categories`` / ``ScrubCategory`` / ``SanitizationConfig``:
the shipped default, the seven schema-load refusals, and
``build_server_config``'s ``FatalConfigError`` surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paramem.backup.types import FatalConfigError
from paramem.config.taxonomy import ScrubCategory, resolve_scrub_categories
from paramem.server.config import SanitizationConfig, build_server_config


def _write_schema(tmp_path: Path, prefix_rows: list[str]) -> Path:
    """A throwaway schema.yaml with the given inline-dict prefix rows."""
    body = "\n".join(f"    - {row}" for row in prefix_rows)
    content = (
        "entity_types:\n"
        "  person: {anchor: 'schema:Person'}\n"
        "  place: {anchor: 'schema:Place'}\n"
        "  organization: {anchor: 'schema:Organization'}\n"
        "  concept: {anchor: 'schema:Thing'}\n"
        "fallback_entity_type: person\n"
        "relation_types: [factual]\n"
        "fallback_relation_type: factual\n"
        "anonymizer:\n"
        "  prefixes:\n" + body + "\n"
    )
    path = tmp_path / "schema.yaml"
    path.write_text(content)
    return path


class TestShippedDefaultResolvesToFiveCategories:
    def test_five_categories_in_schema_row_order(self) -> None:
        categories = SanitizationConfig().scrub_categories
        assert [c.prefix for c in categories] == [
            "Person",
            "Phone",
            "Email",
            "Address",
            "Profile",
        ]

    def test_each_category_is_a_bare_scrub_category(self) -> None:
        categories = SanitizationConfig().scrub_categories
        assert all(isinstance(c, ScrubCategory) for c in categories)
        assert categories[0] == ScrubCategory(prefix="Person")


class TestSevenRefusals:
    def test_duplicate_hint_claimed_by_two_rows_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            [
                "{ prefix: A, entity_type: person, scrub_categories: [shared hint] }",
                "{ prefix: B, entity_type: person, scrub_categories: [shared hint] }",
            ],
        )
        with pytest.raises(ValueError, match="shared hint"):
            resolve_scrub_categories(["shared hint"], path=str(schema))

    def test_uncovered_hint_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            ["{ prefix: A, entity_type: person, scrub_categories: [known hint] }"],
        )
        with pytest.raises(ValueError, match="not claimed"):
            resolve_scrub_categories(["unknown hint"], path=str(schema))

    def test_two_primary_for_type_rows_for_same_entity_type_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            [
                "{ prefix: A, entity_type: person, primary_for_type: true }",
                "{ prefix: B, entity_type: person, primary_for_type: true }",
            ],
        )
        with pytest.raises(ValueError, match="primary_for_type"):
            resolve_scrub_categories([], path=str(schema))

    def test_prefix_collision_under_casefold_raises(self, tmp_path: Path) -> None:
        """The shape rule (one ASCII word starting with a capital) makes
        ``casefold()`` and ``canonical()`` fold identically for every
        legal prefix — a collision under either fold is the same
        collision here, so this one case exercises both checked folds."""
        schema = _write_schema(
            tmp_path,
            [
                "{ prefix: Org, entity_type: organization }",
                "{ prefix: ORG, entity_type: organization }",
            ],
        )
        with pytest.raises(ValueError, match="case-fold"):
            resolve_scrub_categories([], path=str(schema))

    def test_hint_equal_to_another_rows_folded_prefix_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            [
                "{ prefix: Org, entity_type: organization }",
                "{ prefix: Thing, entity_type: concept, scrub_categories: [org] }",
            ],
        )
        with pytest.raises(ValueError, match="canonical"):
            resolve_scrub_categories(["org"], path=str(schema))

    def test_prefix_not_one_capitalized_word_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(tmp_path, ["{ prefix: home_address, entity_type: person }"])
        with pytest.raises(ValueError, match="one word of letters"):
            resolve_scrub_categories([], path=str(schema))

    def test_entity_type_not_declared_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(tmp_path, ["{ prefix: Org, entity_type: nonexistent }"])
        with pytest.raises(ValueError, match="not one of the declared entity_types"):
            resolve_scrub_categories([], path=str(schema))

    def test_distinct_prefixes_pass(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            [
                "{ prefix: Org, entity_type: organization }",
                "{ prefix: Thing, entity_type: concept }",
            ],
        )
        assert resolve_scrub_categories([], path=str(schema)) == ()


class TestListInputOrderIsSchemaRowOrder:
    def test_categories_come_back_in_schema_row_order_not_scrub_order(self) -> None:
        categories = resolve_scrub_categories(["family name", "person name"])
        assert [c.prefix for c in categories] == ["Person"]


class TestRaiseFiresFromSanitizationConfigItself:
    def test_construction_raises_on_an_uncovered_hint(self) -> None:
        # Against the REAL shipped schema.yaml: a hint that is not part of
        # the shipped anonymizer.prefixes claims is uncovered — one of the
        # seven refusal conditions, reachable with no custom schema file.
        with pytest.raises(ValueError):
            SanitizationConfig(scrub=["not a real configured hint"])


class TestBuildServerConfigSurfacesFatalConfigError:
    def test_invalid_scrub_raises_fatal_config_error_naming_the_path(self, tmp_path: Path) -> None:
        source_path = tmp_path / "server.yaml"
        raw = {"sanitization": {"scrub": ["not a real configured hint"]}}
        with pytest.raises(FatalConfigError) as exc_info:
            build_server_config(raw, source_path=source_path)
        assert str(source_path) in str(exc_info.value)
