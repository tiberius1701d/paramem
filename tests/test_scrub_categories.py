"""``resolve_scrub_categories`` / ``SanitizationConfig``: the shipped
default, the six schema-load refusals, and ``build_server_config``'s
``FatalConfigError`` surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paramem.backup.types import FatalConfigError
from paramem.config.taxonomy import resolve_scrub_categories
from paramem.server.config import SanitizationConfig, build_server_config


def _write_schema(tmp_path: Path, prefix_rows: list[str]) -> Path:
    """A throwaway schema.yaml with the given inline-dict prefix rows."""
    body = "\n".join(f"    - {row}" for row in prefix_rows)
    content = (
        "entity_types:\n"
        "  person: {anchor: 'schema:Person'}\n"
        "  place: {anchor: 'schema:Place'}\n"
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

    def test_each_category_carries_its_measured_tagger_labels(self) -> None:
        by_prefix = {c.prefix: c.tagger_labels for c in SanitizationConfig().scrub_categories}
        assert by_prefix["Person"] == ("person",)
        assert by_prefix["Phone"] == ("phone number",)
        assert by_prefix["Email"] == ("email",)
        assert by_prefix["Address"] == ("address",)
        assert by_prefix["Profile"] == ("social media handle", "username")

    def test_person_hints_are_the_configured_person_terms(self) -> None:
        categories = SanitizationConfig().scrub_categories
        person = next(c for c in categories if c.prefix == "Person")
        assert person.hints == ("person name", "full name", "given name", "family name")


class TestListInputOrderIsTheOperatorsOrder:
    def test_hints_preserve_the_callers_scrub_order_not_schema_order(self) -> None:
        categories = resolve_scrub_categories(["family name", "person name"])
        person = next(c for c in categories if c.prefix == "Person")
        assert person.hints == ("family name", "person name")


class TestSixRefusals:
    def test_duplicate_hint_claimed_by_two_rows_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            [
                "{ prefix: A, entity_type: person, scrub_categories: [shared hint], "
                "tagger_labels: [label_a] }",
                "{ prefix: B, entity_type: person, scrub_categories: [shared hint], "
                "tagger_labels: [label_b] }",
            ],
        )
        with pytest.raises(ValueError, match="shared hint"):
            resolve_scrub_categories(["shared hint"], path=str(schema))

    def test_duplicate_tagger_label_claimed_by_two_rows_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            [
                "{ prefix: A, entity_type: person, scrub_categories: [hint_a], "
                "tagger_labels: [shared_label] }",
                "{ prefix: B, entity_type: person, scrub_categories: [hint_b], "
                "tagger_labels: [shared_label] }",
            ],
        )
        with pytest.raises(ValueError, match="shared_label"):
            resolve_scrub_categories([], path=str(schema))

    def test_uncovered_hint_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            [
                "{ prefix: A, entity_type: person, scrub_categories: [known hint], "
                "tagger_labels: [label_a] }",
            ],
        )
        with pytest.raises(ValueError, match="not claimed"):
            resolve_scrub_categories(["unknown hint"], path=str(schema))

    def test_activated_row_with_empty_tagger_labels_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            [
                "{ prefix: A, entity_type: person, scrub_categories: [known hint], "
                "tagger_labels: [] }",
            ],
        )
        with pytest.raises(ValueError, match="tagger_labels"):
            resolve_scrub_categories(["known hint"], path=str(schema))

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

    def test_duplicate_prefix_under_casefold_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            [
                "{ prefix: Org, entity_type: organization }",
                "{ prefix: ORG, entity_type: organization }",
            ],
        )
        with pytest.raises(ValueError, match="case-fold"):
            resolve_scrub_categories([], path=str(schema))

    def test_duplicate_prefix_byte_identical_raises(self, tmp_path: Path) -> None:
        """Two rows sharing the exact same ``prefix`` string are refused
        the same as a case-folded collision — the check compares ROWS,
        not prefix strings, so a byte-identical pair is not exempted."""
        schema = _write_schema(
            tmp_path,
            [
                "{ prefix: Org, entity_type: organization }",
                "{ prefix: Org, entity_type: place }",
            ],
        )
        with pytest.raises(ValueError, match="case-fold"):
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


class TestRaiseFiresFromSanitizationConfigItself:
    def test_construction_raises_on_an_uncovered_hint(self) -> None:
        # Against the REAL shipped schema.yaml: a hint that is not part of
        # the shipped anonymizer.prefixes claims is uncovered — one of the
        # five refusal conditions, reachable with no custom schema file.
        with pytest.raises(ValueError):
            SanitizationConfig(scrub=["not a real configured hint"])


class TestBuildServerConfigSurfacesFatalConfigError:
    def test_invalid_scrub_raises_fatal_config_error_naming_the_path(self, tmp_path: Path) -> None:
        source_path = tmp_path / "server.yaml"
        raw = {"sanitization": {"scrub": ["not a real configured hint"]}}
        with pytest.raises(FatalConfigError) as exc_info:
            build_server_config(raw, source_path=source_path)
        assert str(source_path) in str(exc_info.value)
