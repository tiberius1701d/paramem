"""``resolve_scrub_categories`` / ``ScrubCategory`` / ``SanitizationConfig``:
the shipped default, the schema-load refusals spanning
``anonymizer.scrub`` and ``anonymizer.allow``, and ``build_server_config``'s
``FatalConfigError`` surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paramem.backup.types import FatalConfigError
from paramem.config.taxonomy import ScrubCategory, resolve_scrub_categories
from paramem.server.config import SanitizationConfig, build_server_config


def _write_schema(tmp_path: Path, scrub_rows: list[str], allow_rows: list[str] = ()) -> Path:
    """A throwaway two-list schema.yaml with the given inline-dict rows.

    Args:
        tmp_path: pytest's per-test temp directory.
        scrub_rows: Inline YAML mapping literals for ``anonymizer.scrub``.
        allow_rows: Inline YAML mapping literals for ``anonymizer.allow``.

    Returns:
        The path to the written schema file.
    """

    def _list_block(name: str, rows: list[str]) -> str:
        if not rows:
            return f"  {name}: []\n"
        body = "\n".join(f"    - {row}" for row in rows)
        return f"  {name}:\n{body}\n"

    content = (
        "entity_types:\n"
        "  person: {anchor: 'schema:Person'}\n"
        "  place: {anchor: 'schema:Place'}\n"
        "  organization: {anchor: 'schema:Organization'}\n"
        "  concept: {anchor: 'schema:Thing'}\n"
        "fallback_entity_type: person\n"
        "relation_types: [factual]\n"
        "fallback_relation_type: factual\n"
        "anonymizer:\n" + _list_block("scrub", scrub_rows) + _list_block("allow", list(allow_rows))
    )
    path = tmp_path / "schema.yaml"
    path.write_text(content)
    return path


class TestShippedDefaultResolvesToFiveCategories:
    def test_five_categories_in_scrub_list_order(self) -> None:
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


class TestEmptyScrubIsTheOperatorOptOut:
    def test_no_configured_hints_resolves_to_empty_tuple(self, tmp_path: Path) -> None:
        """``resolve_scrub_categories([], path)`` against a well-formed
        two-list schema (with real ``allow`` rows) returns ``()`` — the
        operator opt-out documented on the function itself: no configured
        hint, no active category, no scan call."""
        schema = _write_schema(
            tmp_path,
            ["{ prefix: A, entity_type: person, scrub_categories: [alpha] }"],
            ["{ prefix: B, entity_type: organization }"],
        )
        assert resolve_scrub_categories([], path=str(schema)) == ()


class TestScrubHintOrderIsScrubListOrder:
    def test_hints_given_in_reverse_order_still_resolve_in_scrub_list_order(
        self, tmp_path: Path
    ) -> None:
        schema = _write_schema(
            tmp_path,
            [
                "{ prefix: A, entity_type: person, scrub_categories: [alpha] }",
                "{ prefix: B, entity_type: person, scrub_categories: [beta] }",
            ],
        )
        categories = resolve_scrub_categories(["beta", "alpha"], path=str(schema))
        assert [c.prefix for c in categories] == ["A", "B"]


class TestRefusals:
    def test_scrub_row_without_hints_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(tmp_path, ["{ prefix: A, entity_type: person }"])
        with pytest.raises(ValueError, match="carries no scrub_categories"):
            resolve_scrub_categories([], path=str(schema))

    def test_allow_row_with_hints_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            ["{ prefix: A, entity_type: person, scrub_categories: [known hint] }"],
            ["{ prefix: B, entity_type: person, scrub_categories: [stray hint] }"],
        )
        with pytest.raises(ValueError, match="allow row always reaches the cloud verbatim"):
            resolve_scrub_categories([], path=str(schema))

    def test_hint_claimed_by_two_rows_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            [
                "{ prefix: A, entity_type: person, scrub_categories: [shared hint] }",
                "{ prefix: B, entity_type: person, scrub_categories: [shared hint] }",
            ],
        )
        with pytest.raises(ValueError, match="claimed by both row"):
            resolve_scrub_categories(["shared hint"], path=str(schema))

    def test_uncovered_hint_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            ["{ prefix: A, entity_type: person, scrub_categories: [known hint] }"],
        )
        with pytest.raises(ValueError, match="not claimed by any row"):
            resolve_scrub_categories(["unknown hint"], path=str(schema))

    def test_two_primary_for_type_rows_for_same_entity_type_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            [
                "{ prefix: A, entity_type: person, primary_for_type: true, "
                "scrub_categories: [a hint] }",
                "{ prefix: B, entity_type: person, primary_for_type: true, "
                "scrub_categories: [b hint] }",
            ],
        )
        with pytest.raises(ValueError, match="more than one primary_for_type row"):
            resolve_scrub_categories([], path=str(schema))

    def test_prefix_collision_under_casefold_raises(self, tmp_path: Path) -> None:
        """The shape rule (one ASCII word starting with a capital) makes
        ``casefold()`` and ``canonical()`` fold identically for every
        legal prefix — a collision under either fold is the same
        collision here, so this one case exercises both checked folds.
        """
        schema = _write_schema(
            tmp_path,
            [
                "{ prefix: Org, entity_type: organization, scrub_categories: [org name] }",
                "{ prefix: ORG, entity_type: organization, scrub_categories: [other org] }",
            ],
        )
        with pytest.raises(ValueError, match="collides under case-folding"):
            resolve_scrub_categories([], path=str(schema))

    def test_hint_equal_to_another_rows_folded_prefix_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            ["{ prefix: Thing, entity_type: concept, scrub_categories: [org] }"],
            ["{ prefix: Org, entity_type: organization }"],
        )
        with pytest.raises(
            ValueError, match="the SCAN keyword match could not tell the hint from a real keyword"
        ):
            resolve_scrub_categories(["org"], path=str(schema))

    def test_prefix_not_one_capitalized_word_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            ["{ prefix: home_address, entity_type: person, scrub_categories: [address] }"],
        )
        with pytest.raises(ValueError, match="not one word of letters starting with a capital"):
            resolve_scrub_categories([], path=str(schema))

    def test_entity_type_not_declared_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            ["{ prefix: Org, entity_type: nonexistent, scrub_categories: [org name] }"],
        )
        with pytest.raises(ValueError, match="not one of the declared entity_types"):
            resolve_scrub_categories([], path=str(schema))


class TestCrossListRefusals:
    """The ``primary_for_type`` and prefix-casefold checks span both
    anonymizer lists — a collision between a ``scrub`` row and an
    ``allow`` row refuses exactly like a collision within one list.
    """

    def test_primary_for_type_claimed_in_both_lists_for_one_entity_type_raises(
        self, tmp_path: Path
    ) -> None:
        schema = _write_schema(
            tmp_path,
            [
                "{ prefix: A, entity_type: person, primary_for_type: true, "
                "scrub_categories: [a hint] }"
            ],
            ["{ prefix: B, entity_type: person, primary_for_type: true }"],
        )
        with pytest.raises(ValueError, match="more than one primary_for_type row"):
            resolve_scrub_categories([], path=str(schema))

    def test_prefix_casefold_collision_between_scrub_and_allow_raises(self, tmp_path: Path) -> None:
        schema = _write_schema(
            tmp_path,
            ["{ prefix: Org, entity_type: organization, scrub_categories: [org name] }"],
            ["{ prefix: ORG, entity_type: organization }"],
        )
        with pytest.raises(ValueError, match="collides under case-folding"):
            resolve_scrub_categories([], path=str(schema))


class TestRaiseFiresFromSanitizationConfigItself:
    def test_construction_raises_on_an_uncovered_hint(self) -> None:
        # Against the REAL shipped schema.yaml: a hint that is not claimed
        # by any row of the shipped anonymizer.scrub list is uncovered —
        # one of resolve_scrub_categories's refusal conditions, reachable
        # with no custom schema file.
        with pytest.raises(ValueError):
            SanitizationConfig(scrub=["not a real configured hint"])


class TestBuildServerConfigSurfacesFatalConfigError:
    def test_invalid_scrub_raises_fatal_config_error_naming_the_path(self, tmp_path: Path) -> None:
        source_path = tmp_path / "server.yaml"
        raw = {"sanitization": {"scrub": ["not a real configured hint"]}}
        with pytest.raises(FatalConfigError) as exc_info:
            build_server_config(raw, source_path=source_path)
        assert str(source_path) in str(exc_info.value)
