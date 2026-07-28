"""Prompt-contract tests — verify the prompt files render correctly and the
documented contract hasn't drifted from the algorithms that depend on it.

The extraction pipeline's binding-recovery algorithm assumes a specific
convention in the cloud enrichment prompt (existing bare placeholders stay
bare; only NEW entities get braces). A prompt edit that inverts this — e.g.
"always emit braced form" — silently breaks de-anonymization and is only
observed in a full sweep. These tests catch that class of regression at
unit-test time.
"""

from __future__ import annotations

import hashlib
import re

import pytest

from paramem.cloud.placeholders import PLACEHOLDER_TOKEN_RE
from paramem.graph.extractor import build_speaker_context
from paramem.graph.phase_trace import extraction_trace, phase_trace
from paramem.graph.prompts import _DEFAULT_PROMPT_DIR, _load_prompt, prompt_overrides


class TestLoadPromptPerModelResolution:
    """Unit tests for _load_prompt per-model, per-file resolution.

    The search order is: prompts_dir/<model>/<filename> (if model),
    prompts_dir/<filename>, _DEFAULT_PROMPT_DIR/<filename>, hardcoded default.
    A model overrides only the files it provides; everything else inherits
    the shared directory.
    """

    def test_per_model_file_wins_when_present(self, tmp_path):
        """prompts_dir/<model>/filename is returned when it exists."""
        (tmp_path / "qwen3-4b").mkdir()
        (tmp_path / "qwen3-4b" / "extraction.txt").write_text("qwen-specific")
        result = _load_prompt("extraction.txt", "default", tmp_path, model="qwen3-4b")
        assert result == "qwen-specific"

    def test_per_model_falls_back_to_base_when_file_absent(self, tmp_path):
        """When per-model file is absent, falls back to prompts_dir/filename."""
        (tmp_path / "qwen3-4b").mkdir()
        # extraction.txt in qwen3-4b/ is ABSENT; extraction_system.txt is in base
        (tmp_path / "extraction_system.txt").write_text("base-system")
        result = _load_prompt("extraction_system.txt", "default", tmp_path, model="qwen3-4b")
        assert result == "base-system"

    def test_model_none_uses_base(self, tmp_path):
        """model=None: only prompts_dir/ and default are searched."""
        (tmp_path / "qwen3-4b").mkdir()
        (tmp_path / "qwen3-4b" / "extraction.txt").write_text("qwen-specific")
        (tmp_path / "extraction.txt").write_text("base")
        result = _load_prompt("extraction.txt", "default", tmp_path, model=None)
        assert result == "base"

    def test_unknown_model_falls_back_to_base(self, tmp_path):
        """A model with no subdir falls through to prompts_dir/ and then default."""
        (tmp_path / "extraction.txt").write_text("base")
        result = _load_prompt("extraction.txt", "default", tmp_path, model="unknown-model")
        assert result == "base"

    def test_both_model_and_base_absent_returns_hardcoded_default(self, tmp_path):
        """When no file exists anywhere, the hardcoded default is returned."""
        result = _load_prompt("no_such_file.txt", "hardcoded-default", tmp_path, model="qwen3-4b")
        assert result == "hardcoded-default"

    def test_required_true_raises_file_not_found_when_absent(self, tmp_path):
        """required=True raises FileNotFoundError when file is absent from all search dirs."""
        with pytest.raises(FileNotFoundError) as exc_info:
            _load_prompt("missing_prompt.txt", prompts_dir=tmp_path, required=True)
        msg = str(exc_info.value)
        assert "missing_prompt.txt" in msg
        assert "Searched" in msg

    def test_required_true_succeeds_when_file_present(self, tmp_path):
        """required=True returns content normally when file is found."""
        (tmp_path / "present.txt").write_text("hello")
        result = _load_prompt("present.txt", prompts_dir=tmp_path, required=True)
        assert result == "hello"

    def test_required_false_default_returns_empty_when_absent(self, tmp_path):
        """required=False (default) returns the default value when file is absent."""
        result = _load_prompt("absent.txt", "fallback", tmp_path)
        assert result == "fallback"

    def test_qwen3_4b_extraction_txt_resolved_from_real_prompts_dir(self):
        """Sanity: the real qwen3-4b/extraction.txt is found under _DEFAULT_PROMPT_DIR."""
        result = _load_prompt(
            "extraction.txt",
            "hardcoded-default",
            _DEFAULT_PROMPT_DIR,
            model="qwen3-4b",
        )
        # The per-model file exists and differs from the base; it must be chosen.
        base = _load_prompt("extraction.txt", "hardcoded-default", _DEFAULT_PROMPT_DIR)
        assert result != base, (
            "qwen3-4b/extraction.txt should differ from the shared base prompt; "
            "if they are identical, the per-model file is redundant and should be removed."
        )

    def test_qwen3_4b_extraction_system_inherits_base(self):
        """qwen3-4b provides no extraction_system.txt override; the base file is inherited."""
        per_model = _load_prompt(
            "extraction_system.txt",
            "hardcoded-default",
            _DEFAULT_PROMPT_DIR,
            model="qwen3-4b",
        )
        base = _load_prompt("extraction_system.txt", "hardcoded-default", _DEFAULT_PROMPT_DIR)
        assert per_model == base, (
            "qwen3-4b must inherit the shared extraction_system.txt; "
            "a per-model override for this file should not exist."
        )


class TestLoadPromptPhaseTraceRecording:
    """``_load_prompt`` records its own resolution via
    :func:`paramem.graph.phase_trace.record_prompt` right after it resolves
    a file (see ``paramem/graph/prompts.py``).  This is the regression pin
    for a divergence that used to exist: a calibration run must see the
    loader's OWN resolved path/content, never a re-implementation of the
    search order.

    That re-implementation used to be live: ``paramem/server/calibrate.py``
    carried a ``_read_prompt`` helper that resolved
    ``path = Path(prompts_dir) / filename`` directly, skipping the
    ``prompts_dir/<model>`` search ``_load_prompt`` performs, and fed its
    own return value into the top-level ``prompts`` field reported by
    several calibrate handlers — so it reported a DIFFERENT prompt than
    production loaded whenever a per-model override existed.  Every
    calibrate handler has since been hooked into the ``_run_calibration``
    primitive, which reads prompt provenance from the phase trace
    ``_load_prompt`` itself populates (see
    :func:`paramem.server.calibrate._provenance_from_records`), and
    ``_read_prompt`` has been removed.  This test class continues to pin
    that ``_load_prompt``'s OWN chokepoint records truthfully — the
    property every calibrate handler now depends on directly."""

    def test_silent_fallback_to_shipped_default_is_visible_in_record(self, tmp_path):
        """The file is ABSENT from the operator-supplied ``prompts_dir``, so
        ``_load_prompt`` silently falls through to ``_DEFAULT_PROMPT_DIR``.
        That fallback must NOT be silent in the phase-trace record: the
        recorded ``path`` is the SHIPPED file and ``sha`` is the shipped
        text's sha — not the (nonexistent) operator-requested file."""
        shipped_path = _DEFAULT_PROMPT_DIR / "cloud_plausibility.txt"
        shipped_content = shipped_path.read_text().strip()
        expected_sha = hashlib.sha256(shipped_content.encode("utf-8")).hexdigest()[:12]

        with extraction_trace() as trace:
            with phase_trace("deanon_plausibility"):
                _load_prompt("cloud_plausibility.txt", prompts_dir=tmp_path)
            record = trace.records[-1]

        assert record.prompts == [
            {
                "path": str(shipped_path),
                "sha": expected_sha,
                "template": shipped_content,
            }
        ]

    def test_variant_present_records_variant_path_and_sha(self, tmp_path):
        """The positive counterpart: when the operator-supplied file DOES
        exist, the record reflects the variant, not the shipped default.

        The file is written WITH surrounding whitespace so this test only
        passes if the recorded sha/template are taken over the POST-strip
        text ``_load_prompt`` actually returns — a sha computed over an
        unstripped re-read of the file would produce a different hash and
        fail here.
        """
        (tmp_path / "cloud_plausibility.txt").write_text("  variant text  \n")
        expected_sha = hashlib.sha256(b"variant text").hexdigest()[:12]

        with extraction_trace() as trace:
            with phase_trace("deanon_plausibility"):
                _load_prompt("cloud_plausibility.txt", prompts_dir=tmp_path)
            record = trace.records[-1]

        assert record.prompts == [
            {
                "path": str(tmp_path / "cloud_plausibility.txt"),
                "sha": expected_sha,
                "template": "variant text",
            }
        ]

    def test_per_model_override_records_variant_path(self, tmp_path):
        """Per-model precedence: prompts_dir/<model>/f.txt wins over
        prompts_dir/f.txt, and the record says so."""
        (tmp_path / "qwen3-4b").mkdir()
        (tmp_path / "qwen3-4b" / "extraction.txt").write_text("qwen-specific")
        (tmp_path / "extraction.txt").write_text("base")

        with extraction_trace() as trace:
            with phase_trace("local_extract"):
                _load_prompt("extraction.txt", prompts_dir=tmp_path, model="qwen3-4b")
            record = trace.records[-1]

        assert record.prompts == [
            {
                "path": str(tmp_path / "qwen3-4b" / "extraction.txt"),
                "sha": hashlib.sha256(b"qwen-specific").hexdigest()[:12],
                "template": "qwen-specific",
            }
        ]

    def test_required_false_absent_everywhere_records_path_none_and_default(self, tmp_path):
        """When no file exists anywhere and required=False, the record
        reports the caller's hardcoded default with path=None — an
        embedded default silently standing in for a prompt file is
        precisely the dishonesty this mechanism exists to expose."""
        with extraction_trace() as trace:
            with phase_trace("local_extract"):
                _load_prompt(
                    "definitely_not_a_real_prompt_file.txt",
                    default="fallback-default",
                    prompts_dir=tmp_path,
                )
            record = trace.records[-1]

        assert record.prompts == [
            {
                "path": None,
                "sha": hashlib.sha256(b"fallback-default").hexdigest()[:12],
                "template": "fallback-default",
            }
        ]


class TestPromptOverrides:
    """``prompt_overrides`` (paramem/graph/prompts.py) substitutes
    ``_load_prompt``'s content for the duration of the ``with`` block —
    the calibration prompt-injection mechanism.  Consulted as the FIRST
    act of ``_load_prompt``, before the ``prompts_dir``/``model`` search
    loop runs at all.
    """

    def test_override_applied_when_basename_matches(self, tmp_path):
        (tmp_path / "extraction.txt").write_text("on-disk content")
        with prompt_overrides({"extraction.txt": "overridden content"}):
            result = _load_prompt("extraction.txt", prompts_dir=tmp_path)
        assert result == "overridden content"

    def test_override_ignored_when_basename_does_not_match(self, tmp_path):
        """An override mapping that names a DIFFERENT file never touches
        an unrelated ``_load_prompt`` call — normal resolution proceeds."""
        (tmp_path / "extraction.txt").write_text("on-disk content")
        with prompt_overrides({"some_other_file.txt": "overridden content"}):
            result = _load_prompt("extraction.txt", prompts_dir=tmp_path)
        assert result == "on-disk content"

    def test_override_satisfies_required_even_when_file_absent_everywhere(self, tmp_path):
        """``required=True`` is satisfied by a matching override the same
        as by a found file — the loader never reaches its "search every
        directory, then raise" branch when an override matched."""
        with prompt_overrides({"nonexistent_prompt.txt": "override content"}):
            result = _load_prompt("nonexistent_prompt.txt", prompts_dir=tmp_path, required=True)
        assert result == "override content"

    def test_prompts_dir_and_model_resolution_unchanged_when_inactive(self, tmp_path):
        """No override active (the common/production case): ``prompts_dir``/
        ``model`` resolution behaves exactly as it did before this feature
        — same per-model precedence covered in ``TestLoadPromptPerModelResolution``."""
        (tmp_path / "qwen3-4b").mkdir()
        (tmp_path / "qwen3-4b" / "extraction.txt").write_text("qwen-specific")
        (tmp_path / "extraction.txt").write_text("base")
        assert _load_prompt("extraction.txt", "default", tmp_path, model="qwen3-4b") == (
            "qwen-specific"
        )
        assert _load_prompt("extraction.txt", "default", tmp_path, model=None) == "base"

    def test_override_recorded_in_provenance_with_synthetic_path(self):
        """A matched override is recorded via ``record_prompt`` exactly
        like a file resolution, but with a synthetic ``<override:...>``
        path so provenance clearly marks it as substituted rather than
        read from disk."""
        with extraction_trace() as trace:
            with phase_trace("local_extract"):
                with prompt_overrides({"extraction.txt": "overridden content"}):
                    _load_prompt("extraction.txt")
            record = trace.records[-1]

        assert record.prompts == [
            {
                "path": "<override:extraction.txt>",
                "sha": hashlib.sha256(b"overridden content").hexdigest()[:12],
                "template": "overridden content",
            }
        ]

    def test_no_override_still_records_normal_resolution(self):
        """When the mapping doesn't match, provenance reports the NORMAL
        resolution (not a synthetic override path) — the miss is silent
        to the loader's normal search, not to provenance."""
        with extraction_trace() as trace:
            with phase_trace("local_extract"):
                with prompt_overrides({"some_other_file.txt": "x"}):
                    _load_prompt("extraction.txt", required=True)
            record = trace.records[-1]

        assert record.prompts[0]["path"] == str(_DEFAULT_PROMPT_DIR / "extraction.txt")

    def test_nested_override_replaces_outer_for_its_duration(self):
        """An inner ``prompt_overrides`` call fully replaces the outer
        mapping for its duration (no merge); the outer mapping is
        restored once the inner block exits."""
        with prompt_overrides({"extraction.txt": "outer"}):
            assert _load_prompt("extraction.txt") == "outer"
            with prompt_overrides({"extraction.txt": "inner"}):
                assert _load_prompt("extraction.txt") == "inner"
            assert _load_prompt("extraction.txt") == "outer"

    def test_override_reset_after_block_exits(self, tmp_path):
        """The override never outlives its own ``with`` block."""
        (tmp_path / "extraction.txt").write_text("on-disk content")
        with prompt_overrides({"extraction.txt": "overridden content"}):
            pass
        result = _load_prompt("extraction.txt", prompts_dir=tmp_path)
        assert result == "on-disk content"

    def test_override_reset_even_when_body_raises(self, tmp_path):
        """The ``ContextVar.reset`` runs in a ``finally`` — an exception
        inside the ``with`` block must not leak the override past it."""
        (tmp_path / "extraction.txt").write_text("on-disk content")
        with pytest.raises(RuntimeError, match="boom"):
            with prompt_overrides({"extraction.txt": "overridden content"}):
                raise RuntimeError("boom")
        result = _load_prompt("extraction.txt", prompts_dir=tmp_path)
        assert result == "on-disk content"


def _render(template: str, **values) -> str:
    """Render a prompt template with placeholder values for inspection."""
    return template.format(**values)


class TestExtractionPrompt:
    def test_renders_with_speaker_context_empty(self):
        tmpl = _load_prompt("extraction.txt", required=True)
        rendered = tmpl.format(
            transcript="[user] hello",
            speaker_context=build_speaker_context(None, None),
        )
        assert "{transcript}" not in rendered
        assert "{speaker_context}" not in rendered
        # Empty speaker_context produces at most one blank-line gap, not three.
        assert "\n\n\n\n" not in rendered

    def test_renders_with_speaker_context_set(self):
        """Directive pins speaker0 as subject; display name injected as comprehension context."""
        tmpl = _load_prompt("extraction.txt", required=True)
        rendered = tmpl.format(
            transcript="[user] hello",
            speaker_context=build_speaker_context("speaker0", "Alex"),
        )
        # The directive must mention the speaker id as the required subject.
        assert "speaker0" in rendered
        # The display name is injected as comprehension context.
        assert "Alex" in rendered

    def test_renders_with_speaker_id_no_display_name(self):
        """Anonymous speaker: id used for both subject and context; no KeyError."""
        tmpl = _load_prompt("extraction.txt", required=True)
        rendered = tmpl.format(
            transcript="[user] hello",
            speaker_context=build_speaker_context("speaker0", None),
        )
        assert "speaker0" in rendered
        assert "{speaker_id}" not in rendered
        assert "{speaker_name}" not in rendered


def _extraction_prompt(model: str | None) -> str:
    """Load the real (non-mocked) extraction.txt for a given model.

    Passing ``_DEFAULT_PROMPT_DIR`` explicitly as ``prompts_dir`` is
    required for per-model resolution: ``_load_prompt`` only searches
    ``prompts_dir/<model>`` when ``prompts_dir`` is truthy (see
    ``_load_prompt`` docstring) — omitting it silently falls back to the
    shared base file even when ``model="qwen3-4b"`` is passed.
    """
    return _load_prompt(
        "extraction.txt", required=True, model=model, prompts_dir=_DEFAULT_PROMPT_DIR
    )


def _positive_blocks(tmpl: str) -> list[str]:
    return [b for b in re.split(r"\n\s*\n", tmpl) if b.lstrip().startswith("POSITIVE example")]


def _negative_blocks(tmpl: str) -> list[str]:
    return [b for b in re.split(r"\n\s*\n", tmpl) if b.lstrip().startswith("NEGATIVE example")]


class TestExtractionPromptThirdPartySubjectContract:
    """Contract tests for the third-party-subject fix.

    Root cause (measured against real production data, data/ha/debug/):
    the prompt taught the relation ``subject`` as a CONSTANT (``speaker0``)
    rather than a variable — every one of six positive few-shots used
    ``speaker0`` as the subject, and the one third-party example present
    (a sister living in Frankfurt) explicitly re-rooted the third party's
    fact onto ``speaker0`` via a compound predicate
    (``sister_lives_in``). Only 2.9% of relations in a real 69-relation
    sample had a non-speaker subject; production data shows glued
    possessives like ``(speaker0, pet_ownership, "Pat's dog")`` and
    outright misbindings like ``(speaker0, is, Priya)``.

    These tests assert the STRUCTURE of the fix, not literal wording, so
    the prompt prose can keep evolving without these tests going stale:
    the positive set must not be uniformly speaker-subject, a
    third-party-subject positive must exist, a discrimination negative
    (wrong speaker0 binding vs. correct third-party binding for the same
    fact) must exist, and the banned compound-possessive predicates must
    not reappear. Both the shared prompt and the qwen3-4b per-model
    override carried the identical defect, so both are checked.
    """

    @pytest.mark.parametrize("model", [None, "qwen3-4b"], ids=["base", "qwen3-4b"])
    def test_positive_set_not_uniformly_speaker_subject(self, model):
        """At least one POSITIVE block must use a subject other than
        speaker0 for ALL of its relations — a uniformly speaker0-subject
        positive set is exactly the defect that taught the model 'subject
        is a constant'."""
        tmpl = _extraction_prompt(model)
        blocks = _positive_blocks(tmpl)
        assert blocks, "No POSITIVE example blocks found — block-split regex or markers drifted."
        subjects_per_block = [set(re.findall(r'"subject":\s*"([^"]+)"', b)) for b in blocks]
        # Only consider blocks that actually contain relations (subjects non-empty).
        relation_blocks = [subs for subs in subjects_per_block if subs]
        assert relation_blocks, "No POSITIVE block contains a relation subject to check."
        assert not all(subs <= {"speaker0"} for subs in relation_blocks), (
            "Every POSITIVE example block uses only speaker0 as the relation "
            "subject — the schema still teaches 'subject is a constant', not "
            "a variable bound to whoever the fact is about."
        )

    @pytest.mark.parametrize("model", [None, "qwen3-4b"], ids=["base", "qwen3-4b"])
    def test_third_party_subject_positive_exists(self, model):
        """At least one POSITIVE block must show a named third party (not
        speaker0) surviving in the subject slot of a relation."""
        tmpl = _extraction_prompt(model)
        blocks = _positive_blocks(tmpl)
        assert any(re.search(r'"subject":\s*"(?!speaker0")[A-Za-z][^"]*"', b) for b in blocks), (
            "No POSITIVE example shows a non-speaker0 entity surviving in the subject slot."
        )

    @pytest.mark.parametrize("model", [None, "qwen3-4b"], ids=["base", "qwen3-4b"])
    def test_discrimination_negative_wrong_vs_correct_subject_exists(self, model):
        """A NEGATIVE block must show the EXACT failure mode side by side:
        a BAD output that re-roots a third party's action onto speaker0,
        next to a CORRECT output that keeps the third party as subject for
        the same fact. Every pre-fix negative was an extract-nothing
        negative; the model had no exemplar of this discrimination."""
        tmpl = _extraction_prompt(model)
        blocks = _negative_blocks(tmpl)
        found = False
        for block in blocks:
            if "BAD output" not in block or "CORRECT output" not in block:
                continue
            bad_part, _, correct_part = block.partition("CORRECT output")
            bad_uses_speaker0_subject = re.search(r'"subject":\s*"speaker0"', bad_part)
            correct_uses_third_party_subject = re.search(
                r'"subject":\s*"(?!speaker0")[A-Za-z][^"]*"', correct_part
            )
            if bad_uses_speaker0_subject and correct_uses_third_party_subject:
                found = True
                break
        assert found, (
            "No NEGATIVE example discriminates a wrong (speaker0, ...) binding "
            "from a correct third-party-subject binding for the same fact."
        )

    @pytest.mark.parametrize("model", [None, "qwen3-4b"], ids=["base", "qwen3-4b"])
    def test_no_compound_possessive_predicate_or_object_remains(self, model):
        """The compound-predicate escape hatch (re-rooting a third party's
        fact onto speaker0 by gluing their name into the predicate/object
        as a possessive) must never appear as TAUGHT (POSITIVE or CORRECT)
        behaviour. It IS allowed inside a BAD half of a NEGATIVE block —
        the glued-possessive-object NEGATIVE example deliberately shows it
        as the anti-pattern being taught against."""
        tmpl = _extraction_prompt(model)
        # The banned predicates may still be NAMED in rule prose as the
        # anti-pattern to avoid; they must never appear as an actually
        # emitted `"predicate"` value in a worked example.
        for banned_predicate in ("sister_lives_in", "pet_ownership"):
            assert f'"predicate": "{banned_predicate}"' not in tmpl, (
                f"Banned compound-possessive predicate {banned_predicate!r} "
                "is emitted as a predicate value in a worked example."
            )
        glued_object_re = re.compile(r'"object":\s*"[^"]*\'s [^"]*"')
        # POSITIVE blocks must never contain a glued-possessive object.
        for block in _positive_blocks(tmpl):
            assert not glued_object_re.search(block), (
                f"POSITIVE example teaches a glued-possessive object: {block!r}"
            )
        # NEGATIVE blocks: the glued form is allowed ONLY in the BAD half;
        # the CORRECT half must never contain it.
        for block in _negative_blocks(tmpl):
            if "CORRECT output" not in block:
                continue
            _, _, correct_part = block.partition("CORRECT output")
            assert not glued_object_re.search(correct_part), (
                f"NEGATIVE example's CORRECT output still contains a "
                f"glued-possessive object: {correct_part!r}"
            )

    @pytest.mark.parametrize("model", [None, "qwen3-4b"], ids=["base", "qwen3-4b"])
    def test_speaker_subject_binding_conditioned_on_self_reference(self, model):
        """The rule prose must condition the speaker0-as-subject binding on
        self-reference, not claim it applies to every fact."""
        tmpl = _extraction_prompt(model)
        prose = tmpl.split("POSITIVE example")[0]
        assert re.search(r"self-reference|self-referen", prose, re.IGNORECASE), (
            "Rule prose no longer conditions the speaker0 binding on self-reference."
        )
        assert "for every fact about the speaker" not in prose, (
            "Rule prose still claims speaker0 is the subject for EVERY fact — "
            "the self-reference conditioning regressed."
        )

    @pytest.mark.parametrize("model", [None, "qwen3-4b"], ids=["base", "qwen3-4b"])
    def test_assistant_turns_fact_source_rule_stated(self, model):
        """The prompt must explicitly state that facts can come from
        [assistant] turns and that the subject is whoever the sentence
        names — not the speaker by default. Pre-fix, the prompt was
        silent on assistant turns; that silence is the exact gap a
        third-party fact stated only in an [assistant] reply fell through."""
        tmpl = _extraction_prompt(model)
        prose = tmpl.split("POSITIVE example")[0]
        assert "[assistant]" in prose, (
            "Rule prose does not explicitly mention [assistant] turns as a fact source."
        )
        assert re.search(r"never the speaker by default", prose, re.IGNORECASE), (
            "Rule prose does not state that the subject defaults to whoever the "
            "sentence names, not the speaker."
        )

    @pytest.mark.parametrize("model", [None, "qwen3-4b"], ids=["base", "qwen3-4b"])
    def test_no_generic_noun_subject_anywhere(self, model):
        """Node identity is a graph-global string fold — ``canonical_id``
        on ``entity.name`` (``paramem/graph/merger.py``) with NO owner or
        speaker scoping (``canonical()`` in ``paramem/utils/identity.py``
        is a pure Unicode-NFC / casefold / diacritic fold). A generic role
        noun (``sister``, ``dog``, ``kids``) placed in SUBJECT position
        becomes ONE node shared across the entire household graph, so
        facts about two different people's "sister" collide onto it —
        a cross-speaker contamination class. Subject position is
        therefore restricted to an IDENTIFIABLE entity: a proper
        (capitalised) name, or a ``speaker{N}`` id. Generic nouns MAY
        still appear as relation OBJECTS (e.g. ``(Pat, has_pet, dog)``) —
        only the SUBJECT slot is restricted, since only the subject
        accumulates outgoing facts across sessions/speakers. Scans every
        relation subject in the whole prompt (not just positives) so a
        stray generic-noun subject in a future example fails immediately.
        """
        tmpl = _extraction_prompt(model)
        subjects = re.findall(r'"subject":\s*"([^"]+)"', tmpl)
        assert subjects, "No relation subjects found in prompt — scan regex drifted."
        allowed = re.compile(r"^(speaker\d+|[A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*)$")
        offenders = [s for s in subjects if not allowed.match(s)]
        assert not offenders, (
            f"Generic/common-noun or lowercase subject(s) found in relation "
            f"subject position: {offenders!r} — only a speaker id or a "
            "proper (capitalised) name may occupy the subject slot; a "
            "generic role noun there is a graph-global node-identity "
            "collision (see paramem/graph/merger.py canonical_id lookup)."
        )

    @pytest.mark.parametrize("model", [None, "qwen3-4b"], ids=["base", "qwen3-4b"])
    def test_glued_possessive_negative_exists(self, model):
        """A NEGATIVE block must show the glued-possessive object as BAD
        (a third party's name glued into the object string, e.g. "Theo's
        orchids") next to a CORRECT output that splits the third party out
        as its own subject entity, with the bare object surviving as a
        separate relation. A live probe against a held-out name showed
        this class SURVIVES on prose alone ("Yusuf's orchids" ->
        (speaker0, cares_for, "Yusuf's orchids")) — a worked example is
        required, mirroring the ``Pat's dog`` POSITIVE that already
        teaches the correct split without ever pairing it against the
        wrong form."""
        tmpl = _extraction_prompt(model)
        glued_object_re = re.compile(r'"object":\s*"[^"]*\'s [^"]*"')
        found = False
        for block in _negative_blocks(tmpl):
            if "BAD output" not in block or "CORRECT output" not in block:
                continue
            bad_part, _, correct_part = block.partition("CORRECT output")
            bad_has_glued_object = glued_object_re.search(bad_part)
            correct_splits_subject = re.search(
                r'"subject":\s*"(?!speaker0")[A-Za-z][^"]*"', correct_part
            )
            if bad_has_glued_object and correct_splits_subject:
                found = True
                break
        assert found, (
            "No NEGATIVE example demonstrates the glued-possessive-object "
            "failure (BAD) next to the split-into-third-party-subject fix "
            "(CORRECT)."
        )

    @pytest.mark.parametrize("model", [None, "qwen3-4b"], ids=["base", "qwen3-4b"])
    def test_unnamed_third_party_negative_exists(self, model):
        """A NEGATIVE block must show an unnamed third party (a role noun
        with no proper name) producing NO relation as CORRECT, next to a
        BAD output that mangles the fact into a nonsense relation hung off
        speaker0. A live probe showed this class produces semantic
        garbage on prose alone ("My brother lives in Porto" ->
        (speaker0, has_brother, Porto), a place under a has_brother
        predicate) — a worked example is required."""
        tmpl = _extraction_prompt(model)
        found = False
        for block in _negative_blocks(tmpl):
            if "BAD output" not in block or "CORRECT output" not in block:
                continue
            bad_part, _, correct_part = block.partition("CORRECT output")
            # BAD half mangles the fact onto speaker0; CORRECT half emits
            # no relations at all (the structural signature of "an
            # unnamed third party yields no relation").
            bad_mangles_onto_speaker = re.search(r'"subject":\s*"speaker0"', bad_part)
            correct_has_no_relations = '"relations": []' in correct_part
            if bad_mangles_onto_speaker and correct_has_no_relations:
                found = True
                break
        assert found, (
            "No NEGATIVE example demonstrates the unnamed-third-party "
            "failure (BAD: mangled onto speaker0) next to the "
            "no-relation-extracted fix (CORRECT: empty relations)."
        )

    @pytest.mark.parametrize("model", [None, "qwen3-4b"], ids=["base", "qwen3-4b"])
    def test_named_relative_chain_example_exists(self, model):
        """A POSITIVE block must show a named-relative CHAIN: an entity
        that is BOTH the object of a speaker0-subject relation (the
        relationship edge, e.g. speaker0 -> has_sibling -> Nadia) AND the
        subject of another relation in the same block (the relative's own
        fact, e.g. Nadia -> lives_in -> Frankfurt).

        Root cause (measured): Mistral 7B collapsed a single-relative
        clause ("my brother Nadeem lives in Porto") into ONE relation,
        dropping either the kinship edge or the attribute — never
        emitting both. Without a worked chain example, the model has no
        exemplar of the two-relation decomposition the rule prose (below)
        demands.
        """
        tmpl = _extraction_prompt(model)
        triple_re = re.compile(
            r'"subject":\s*"([^"]+)",\s*"predicate":\s*"([^"]+)",\s*"object":\s*"([^"]+)"'
        )
        found = False
        for block in _positive_blocks(tmpl):
            triples = triple_re.findall(block)
            speaker_objects = {obj for subj, _pred, obj in triples if subj == "speaker0"}
            if any(subj in speaker_objects for subj, _pred, _obj in triples):
                found = True
                break
        assert found, (
            "No POSITIVE example shows a named relative as BOTH the "
            "object of a speaker0 relationship edge AND the subject of "
            "their own fact — the chain structure teaching two-relation "
            "decomposition is missing."
        )

    @pytest.mark.parametrize("model", [None, "qwen3-4b"], ids=["base", "qwen3-4b"])
    def test_named_relative_decomposition_rule_stated(self, model):
        """The rule prose (outside the worked examples) must state that a
        speaker's relationship-named clause ("my brother Nadeem") is TWO
        separate facts — the relationship edge on the speaker AND the
        person's own fact on their own node — and must forbid collapsing
        or dropping either one.

        Root cause (measured): Mistral 7B collapsed a single-relative
        clause into one relation, dropping either the kinship edge or the
        attribute. Asserted structurally (co-located stable tokens, not
        the literal sentence) so the prose can keep evolving without this
        test going stale.
        """
        tmpl = _extraction_prompt(model)
        prose = tmpl.split("POSITIVE example")[0]
        assert re.search(r"relationship edge", prose, re.IGNORECASE), (
            "Rule prose no longer states the relationship-edge half of the "
            "named-relative decomposition."
        )
        assert re.search(r"own (fact|node)", prose, re.IGNORECASE), (
            "Rule prose no longer states the relative's-own-fact half of "
            "the named-relative decomposition."
        )
        assert re.search(r"collapse", prose, re.IGNORECASE), (
            "Rule prose no longer forbids collapsing the two relations into one."
        )
        assert re.search(r"drop", prose, re.IGNORECASE), (
            "Rule prose no longer forbids dropping either relation."
        )


def _second_order_prompt(model: str | None) -> str:
    """Load the real (non-mocked) extraction_second_order.txt for a given model.

    Mirrors :func:`_extraction_prompt` — ``_DEFAULT_PROMPT_DIR`` must be
    passed explicitly for per-model resolution to engage.
    """
    return _load_prompt(
        "extraction_second_order.txt", required=True, model=model, prompts_dir=_DEFAULT_PROMPT_DIR
    )


class TestExtractionSecondOrderPromptContract:
    """Contract tests for ``extraction_second_order.txt`` (the
    ``second_order_extract`` phase — a second local-model pass that
    extracts facts ABOUT the named entities ``local_extract`` surfaced,
    recovering a named relative's own attribute when ``local_extract``
    collapses a single-relative clause ("my brother Nadeem lives in
    Porto") into one relation instead of two).

    The second-order pass is ATTRIBUTE-ONLY: it must teach re-extraction
    of the named person's own fact with them as subject, and must NOT
    re-emit the speaker's kinship edge onto them — that edge is already
    captured by ``local_extract``. A POSITIVE example that emits a
    ``speaker0``-subject relation would double-teach ground the
    second-order pass has no business touching. Asserted structurally
    (subject-set membership), not literal wording, so the prompt prose can
    keep evolving.
    """

    @pytest.mark.parametrize("model", [None, "qwen3-4b"], ids=["base", "qwen3-4b"])
    def test_positive_example_subject_is_named_person(self, model):
        """At least one POSITIVE block must emit a relation whose subject
        is a named person, not speaker0."""
        tmpl = _second_order_prompt(model)
        blocks = _positive_blocks(tmpl)
        assert blocks, "No POSITIVE example blocks found in extraction_second_order.txt."
        found = False
        for block in blocks:
            subjects = set(re.findall(r'"subject":\s*"([^"]+)"', block))
            if subjects and "speaker0" not in subjects:
                found = True
                break
        assert found, (
            "No POSITIVE example in extraction_second_order.txt emits a relation "
            "whose subject is a named person (not speaker0)."
        )

    @pytest.mark.parametrize("model", [None, "qwen3-4b"], ids=["base", "qwen3-4b"])
    def test_no_positive_example_emits_speaker0_subject(self, model):
        """NO POSITIVE block may emit a speaker0-subject relation — the
        second-order pass must not re-emit the kinship edge
        ``local_extract`` already recorded."""
        tmpl = _second_order_prompt(model)
        blocks = _positive_blocks(tmpl)
        assert blocks, "No POSITIVE example blocks found in extraction_second_order.txt."
        for block in blocks:
            subjects = set(re.findall(r'"subject":\s*"([^"]+)"', block))
            assert "speaker0" not in subjects, (
                "POSITIVE example re-emits a speaker0-subject relation — the "
                f"second-order pass must not re-emit kinship edges: {block!r}"
            )


class TestEnrichmentPromptContract:
    def test_renders_without_format_errors(self):
        """No stray single-brace placeholders that collide with .format()."""
        tmpl = _load_prompt("cloud_enrichment.txt", required=True)
        # Must not raise KeyError — every literal brace pair escaped as
        # `{{` / `}}`.  ``str.format`` itself validates this.
        rendered = tmpl.format(transcript="Person_1 said hi.", facts_json="[]")
        # Rendered output should still contain the example braced tokens
        # (single-brace form after format-escape).
        assert "{Event_1}" in rendered or "{Prefix_N}" in rendered

    def test_preserves_bare_placeholder_convention(self):
        """The binding-recovery algorithm depends on cloud leaving existing
        bare placeholders bare. Regressing the prompt to 'always emit braced'
        silently breaks de-anonymization.
        """
        tmpl = _load_prompt("cloud_enrichment.txt", required=True)
        # Must instruct model to leave existing bare placeholders bare.
        # Specifically: not to re-brace incoming Person_1/City_1 tokens.
        keywords = ["bare", "leave", "existing", "NOT re-brace"]
        hits = sum(1 for k in keywords if k.lower() in tmpl.lower())
        assert hits >= 2, (
            "Enrichment prompt must instruct cloud to leave existing bare "
            "placeholders bare — otherwise binding recovery records self-"
            "referential junk entries (Person_2 → Person_2) and corrupts "
            "the reverse mapping. Found only keywords: "
            f"{[k for k in keywords if k.lower() in tmpl.lower()]}"
        )

    def test_requires_grounding_of_new_placeholders(self):
        """HARD REQUIREMENT: every braced placeholder used in `add` must
        have a matching entry in `bindings`. Without this clause, cloud's
        reified entities are dropped wholesale by the residual sweep.

        Updated transcript is no longer carried on the wire (delta
        protocol — reconstructed locally from bindings + anon transcript)
        so the contract is now solely "facts ↔ bindings".
        """
        tmpl = _load_prompt("cloud_enrichment.txt", required=True)
        # Bindings is the grounding contract; the prompt must name it.
        assert "bindings" in tmpl
        # Look for a hard requirement that braced placeholders appear in
        # both `add` (facts) and `bindings`.  Phrasing is free to evolve;
        # the structural claim is not.
        assert re.search(r"MUST.*appear|appear.*MUST", tmpl, re.IGNORECASE), (
            "Enrichment prompt must contain a hard requirement that new "
            "braced placeholders appear in both `add` and `bindings`."
        )
        # No transcript echo on the wire — `updated_transcript` must not
        # appear as an output key (catches accidental reintroduction).
        assert "updated_transcript" not in tmpl, (
            "Enrichment prompt must not request `updated_transcript` in "
            "the output — the transcript is reconstructed locally from "
            "bindings to keep output bandwidth bounded."
        )

    def test_teaches_role_instance_aggregation(self):
        """The brace-binding section must show a role-instance POSITIVE
        example that aggregates multiple co-temporal attributes (title,
        company, location, dates) onto a single bound entity rather
        than flattening them as independent triples on the speaker.

        Without this teaching, cloud emits the bound title once but
        leaves dates / company / location as orphan triples on the
        speaker.  Downstream reasoning over multi-role chronology
        ("what title did the speaker hold in 2015?") then fails because
        co-temporal facts cannot be paired back to a role.

        Empirical evidence pre-fix: zero ``Role_*`` entities across 24+
        production graph snapshots (data/ha/debug/run_*/), even though
        the brace-binding contract itself is honoured for ``Event_*``.
        """
        tmpl = _load_prompt("cloud_enrichment.txt", required=True)
        # Structural assertion: a Role_N braced placeholder must appear
        # in a positive-example block alongside multiple bound facts —
        # at minimum a date attribute and a company/location attribute.
        assert "{{Role_1}}" in tmpl, (
            "Enrichment prompt must include a Role_1 example to teach role-instance aggregation."
        )
        # Co-temporal attributes must be bound to {{Role_1}} (subject
        # position).  A flat-triples regression would have them on
        # Person_1 instead.  The delta protocol uses JSON-shaped facts
        # (``"subject": "{{Role_1}}", "predicate": "start_date"``);
        # match either ordering of those two key/value pairs within a
        # short window so the test is robust to minor reformatting.
        assert re.search(
            r'"\{\{Role_1\}\}"[^{}]{0,200}start_date'
            r"|"
            r'start_date[^{}]{0,200}"\{\{Role_1\}\}"',
            tmpl,
        ), (
            "Role example must show start_date with {{Role_1}} as subject — "
            "the structural teaching is that dates bind to the role-instance, "
            "not to the speaker."
        )
        # The NEGATIVE block must spell out the speaker-flattening anti-
        # pattern so a future prompt edit cannot keep the POSITIVE
        # example while quietly removing the warning.
        assert re.search(r"WRONG.*speaker|flat.*speaker|speaker.*flat", tmpl, re.IGNORECASE), (
            "Enrichment prompt must call out the flat-triples-on-speaker "
            "anti-pattern in a NEGATIVE block."
        )

    def test_teaches_pre_return_binding_self_audit(self):
        """The brace-binding section must end with a self-audit clause
        instructing the model to verify, before returning, that every
        minted placeholder has a matching ``bindings`` entry — and to
        write the entity surface inline rather than mint an unbound
        placeholder.  Structural, phrasing-tolerant: the exact wording is
        free to evolve; the pre-return check binding minted tokens to
        ``bindings`` is not."""
        tmpl = _load_prompt("cloud_enrichment.txt", required=True)
        assert "Before returning" in tmpl, (
            "Enrichment prompt must contain a 'Before returning' self-audit clause."
        )
        assert "bindings" in tmpl
        # Must still render cleanly (format-escape correctness for the new line).
        rendered = tmpl.format(transcript="Person_1 said hi.", facts_json="[]")
        assert "Before returning" in rendered

    # -- Speaker anchor -----------------
    #
    # ``test_requires_grounding_of_new_placeholders`` above passed for
    # months asserting only that the grounding RULE was stated, while
    # three few-shots twenty lines below taught the exact opposite (a
    # bare, unbound mint). The tests below assert on EXAMPLE CONTENT —
    # not rule presence — so they cannot pass while the corpus
    # contradicts itself the way that gap did.

    def test_no_example_asserts_i_me_my_maps_to_person_1(self):
        """STRUCTURAL — scans every blank-line-separated block in the
        prompt, not a hardcoded phrase.  A block that talks about the
        speaker (mentions 'speaker' in its descriptive text) but has no
        'speaker0' anchor present must not use ``Person_1`` as the fact
        subject — that is exactly how 'Person_1 (the speaker' and its
        reworded cousin 'Person_1 — the speaker' both re-teach the
        positional guess this plan retires.  A test that only checked
        the literal substring passed for months while three few-shots
        taught the opposite; this one cannot pass while any block does."""
        tmpl = _load_prompt("cloud_enrichment.txt", required=True)
        blocks = re.split(r"\n\s*\n", tmpl)
        for block in blocks:
            if "speaker" not in block.lower():
                continue
            if "speaker0" in block:
                # The anchor is present and doing the speaker-referring
                # work in this block — Person_1, if it also appears, is
                # not standing in for it.
                continue
            assert not re.search(r'"subject"\s*:\s*"\{?\{?Person_1\}?\}?"', block), (
                "Block mentions 'speaker' but has no speaker0 anchor "
                "while using Person_1 as a fact subject — this re-"
                f"teaches Person_1 == the speaker: {block!r}"
            )
        assert '"my"' in tmpl and "speaker0" in tmpl, (
            "Enrichment prompt must still teach first-person coreference, now grounded on speaker0."
        )

    def test_first_person_few_shots_resolve_to_speaker0(self):
        """The 'my wife' / 'my sister's husband' / 'my father' coreference
        few-shots must bind the speaker to 'speaker0' — not a Person_N —
        in the delta they actually emit."""
        tmpl = _load_prompt("cloud_enrichment.txt", required=True)
        for anchor in (
            '"my wife is also a teacher"',
            '"my sister\'s husband"',
            '"my father is also an engineer"',
        ):
            idx = tmpl.index(anchor)
            window = tmpl[idx : idx + 400]
            assert '"subject":"speaker0"' in window or '"subject": "speaker0"' in window, (
                f"First-person few-shot {anchor!r} must bind the speaker "
                f"to 'speaker0', not a positionally-guessed Person_N: {window!r}"
            )

    def test_non_speaker_cast_as_person_1(self):
        """At least one few-shot must show Person_1 naming someone OTHER
        than the speaker, so the model cannot re-derive 'Person_1 = me'
        positionally even without an explicit rule saying so."""
        tmpl = _load_prompt("cloud_enrichment.txt", required=True)
        idx = tmpl.index('"we went there last summer"')
        window = tmpl[idx : idx + 400]
        assert "speaker0" in window, "Example must still ground the speaker as speaker0."
        assert '"subject":"Person_1"' in window or '"subject": "Person_1"' in window, (
            "Example must cast Person_1 as a THIRD PARTY (not the speaker) "
            "so Person_1 cannot be re-derived as 'the speaker' positionally."
        )

    def test_every_positive_mint_has_a_binding(self):
        """STRUCTURAL — scans EVERY blank-line-separated block in the
        prompt for `{Prefix_N}` braced mints and requires a matching
        `bindings` entry in the SAME block, rather than checking a
        hardcoded list of anchors a human happened to think of.  A
        newly-added few-shot with an unbound mint fails this test
        automatically — the exact shape (bare unbound mint) that
        produced the observed 5-in-1-out data loss: Cloud minting
        Person_2/Person_3 with no binding at all.  NEGATIVE/WRONG blocks
        are exempt — they deliberately demonstrate the unbound-mint
        failure as the thing NOT to do."""
        tmpl = _load_prompt("cloud_enrichment.txt", required=True)
        rendered = tmpl.format(transcript="x", facts_json="[]")
        blocks = re.split(r"\n\s*\n", rendered)
        checked_any = False
        for block in blocks:
            if "WRONG" in block or block.lstrip().startswith("NEGATIVE"):
                continue
            if '"subject"' not in block:
                # Not a fact-example block (e.g. the contract's own
                # prose illustrating the braced-form syntax) — nothing
                # to bind.
                continue
            # Braced mints only — a bare match here would be an existing
            # anonymizer placeholder (Person_1), not a new cloud mint.
            mints = {m[0] for m in PLACEHOLDER_TOKEN_RE.findall(block) if m[0]}
            for key in mints:
                checked_any = True
                assert f'"{key}"' in block and "bindings" in block, (
                    f"Positive example block mints {{{key}}} without a "
                    f"matching bindings entry in the same block: {block!r}"
                )
        assert checked_any, (
            "No positive-example mints were found to check — the scan "
            "regex or block split likely drifted from the prompt format."
        )


class TestPlausibilityPromptContract:
    def test_renders_without_format_errors(self):
        tmpl = _load_prompt("cloud_plausibility.txt", required=True)
        rendered = tmpl.format(transcript="Person_1 said hi.", facts_json="[]")
        assert "{transcript}" not in rendered
        assert "{facts_json}" not in rendered

    def test_lists_drop_rules(self):
        """Plausibility judge relies on six numbered drop rules (R1-R6).

        The prior R4 ("Unresolved placeholder in real-name input") was
        tied to the constrained ``^(Person|City|Country|Org|Thing)_\\d+$``
        regex that became incoherent with the open-vocabulary anonymizer
        pivot.  It was also structurally redundant with the residual
        sweep inside ``_apply_bindings`` at the deanon stage, before
        plausibility.

        The grounding refactor revised the remaining rules: lexical
        token lists became illustrative parentheticals, and a new R3
        (transcript contradiction) closes the gap that the prior
        "no judgment calls" framing left open.  The structure of the
        prompt and of the examples is unchanged.

        Verify each rule's identifying substring still exists so a
        prompt edit that removes a rule is caught at unit-test time.
        """
        tmpl = _load_prompt("cloud_plausibility.txt", required=True)
        required_rules = [
            "self-loop",  # R1
            "name-swap",  # R2
            "contradiction",  # R3 — new transcript-grounded rule
            "conversation-role",  # R4 — was "Role leak", grounded
            "content-free",  # R5 — was "Empty / sentinel"
            "system identifier",  # R6 — was "System entity ID"
        ]
        for rule in required_rules:
            assert rule.lower() in tmpl.lower(), f"Plausibility prompt missing rule: {rule!r}"

    def test_keep_default_disposition(self):
        """The prompt uses a keep-by-default model. Regressing to drop-by-default
        silently discards valid facts — a data-loss bug that only surfaces
        during a full extraction sweep. This assertion guards that semantic flip.

        The contract is structural, not literal: the default disposition must
        be to keep the fact (IGNORE — keep unless a drop rule matches), it must
        be declared as the default, and it must appear before the drop rules.
        Surface wording — "Default action", "IGNORE", a "## KEEP" header — is
        free to evolve; the keep-by-default semantics and the ordering are not.
        """
        tmpl = _load_prompt("cloud_plausibility.txt", required=True)
        lower = tmpl.lower()
        # The default action must KEEP the fact (IGNORE), not drop it.
        assert "default action" in lower, (
            "Plausibility prompt must declare a default action (keep-by-default). "
            "Removing this primes the model to drop on judgment, causing silent data loss."
        )
        assert "ignore" in lower or "keep the fact" in lower, (
            "Default disposition must keep the fact (IGNORE / keep), not drop."
        )
        # The keep-by-default declaration must precede the drop rules.
        default_idx = lower.find("default action")
        rules_idx = lower.find("## drop")
        assert rules_idx >= 0, "Plausibility prompt missing the drop-rules section header."
        assert 0 <= default_idx < rules_idx, (
            "Keep-by-default disposition must be declared before the drop rules — "
            "order encodes default disposition."
        )

    def test_output_contract_is_drop_index_set(self):
        """The plausibility judge's output protocol is a small JSON object
        ``{"drop": [<index>, ...]}`` listing which input facts to drop —
        NOT an echo of every kept fact.

        Why this is structural, not stylistic: the previous "echo every
        kept fact" contract had Mistral 7B emit EOS mid-array on long
        inputs (the closing ``]`` never arrived; ``_parse_facts_response``
        couldn't recover the envelope; the gate fail-opened with 0 facts
        filtered).  The drop-set output is bounded by the count of
        actual rule matches — typically 0–5 indices for clean inputs —
        so truncation cannot kill the gate.

        Regressing to "echo every fact" silently re-introduces the
        truncation failure mode, so this assertion locks the contract.
        """
        tmpl = _load_prompt("cloud_plausibility.txt", required=True)
        # Must specify the drop-index-set object shape.
        assert '"drop"' in tmpl, (
            'Plausibility prompt must specify the drop-set output shape ({"drop": [<index>, ...]}).'
        )
        # Must describe the index-based reference convention.
        assert "zero-based" in tmpl.lower() and "index" in tmpl.lower(), (
            "Plausibility prompt must teach zero-based index references — the "
            "judge needs to know how facts are numbered to refer to them."
        )
        # Must forbid echoing kept facts (the regression vector).
        forbids_echo = re.search(
            r"do not (echo|include the facts|return surviving|emit the surviving)",
            tmpl,
            re.IGNORECASE,
        )
        assert forbids_echo, (
            "Plausibility prompt must explicitly forbid echoing the kept facts — "
            "without this the model defaults to verbose echo and triggers the "
            "Mistral-7B EOS-mid-array truncation."
        )


class TestProceduralPrompt:
    def test_renders_with_speaker_context_empty(self):
        """Procedural prompt renders without errors when speaker is unknown.

        Verifies that the {speaker_context} placeholder is present in the
        file-based prompt and collapses cleanly to an empty string so no
        dangling placeholder or extra blank lines remain.
        """
        tmpl = _load_prompt("extraction_procedural.txt", required=True)
        rendered = tmpl.format(
            transcript="[user] Play some jazz.",
            speaker_context=build_speaker_context(None, None),
        )
        assert "{transcript}" not in rendered
        assert "{speaker_context}" not in rendered
        # Empty speaker_context produces at most one blank-line gap, not three.
        assert "\n\n\n\n" not in rendered

    def test_renders_with_speaker_context_set(self):
        """Procedural prompt pins speaker0 as subject with display name as context.

        Guards against the silent identity fragmentation bug where
        procedural facts get subject "Speaker" or a display name while
        main-extraction facts use the speaker id, creating two nodes for
        the same person.
        """
        tmpl = _load_prompt("extraction_procedural.txt", required=True)
        rendered = tmpl.format(
            transcript="[user] Play some jazz.",
            speaker_context=build_speaker_context("speaker0", "Alex"),
        )
        # The directive must mention the speaker id as the required subject.
        assert "speaker0" in rendered
        # The display name appears as comprehension context.
        assert "Alex" in rendered


class TestAnonymizationPrompt:
    """Contract tests for the anonymization prompt — both default and file-based.

    The prompt teaches a shape contract (PascalCase prefix + ``_<N>``,
    uniqueness, totality, direction), not a constrained vocabulary.  The
    earlier ``replacement_rules`` interpolation that listed the configured
    prefixes is gone — prefixes are illustrative-only inside the prompt
    body, and the model picks any type-appropriate PascalCase prefix.

    Since the config-driven redesign (``sanitization.scrub``), the prompt
    is the SOLE scope authority: it takes a ``{scrub_categories}`` slot and
    returns TWO artifacts (``mapping`` AND ``anonymized_transcript``) — no
    code-side entity-type gate narrows the model's output afterward.
    """

    def _render(self, tmpl: str) -> str:
        """Render the anonymization prompt with all expected kwargs."""
        return tmpl.format(
            scrub_categories="person name, phone number",
            facts_json='[{"subject": "Person_1", "predicate": "lives_in", '
            '"object": "City_1", "relation_type": "factual", "confidence": 0.9}]',
            transcript="Person_1 lives in City_1.",
        )

    def test_default_renders_without_format_errors(self):
        """anonymization.txt must render with all expected kwargs without KeyError."""
        rendered = self._render(_load_prompt("anonymization.txt", required=True))
        assert "{facts_json}" not in rendered
        assert "{transcript}" not in rendered
        assert "{scrub_categories}" not in rendered

    def test_file_based_renders_without_format_errors(self):
        """File-based anonymization.txt must render with all expected kwargs without KeyError."""
        tmpl = _load_prompt("anonymization.txt", required=True)
        rendered = self._render(tmpl)
        assert "{facts_json}" not in rendered
        assert "{transcript}" not in rendered
        assert "{scrub_categories}" not in rendered

    def test_shape_contract_present_in_default(self):
        """anonymization.txt must teach the four parts of the shape contract:
        well-formed shape, uniqueness, totality, direction."""
        rendered = self._render(_load_prompt("anonymization.txt", required=True))
        # Shape clause — `<Prefix>_<N>` or equivalent shape language.
        assert "PascalCase" in rendered or "Prefix" in rendered, (
            "Anonymization prompt must teach the placeholder shape (PascalCase + _<N>)."
        )
        # Uniqueness clause.
        assert "UNIQUE" in rendered or "unique" in rendered
        # Totality clause.
        assert "totality" in rendered.lower() or "every placeholder" in rendered.lower()
        # Direction clause.
        assert "real_value" in rendered or "real value" in rendered.lower()

    def test_shape_contract_present_in_file_based(self):
        """The file-based prompt must teach the same four-part shape contract.

        Verifies the file isn't accidentally pinned to a constrained vocabulary
        (the regression that prompted this rewrite — the model invented prefixes
        like ``University_1`` / ``Project_1`` and the recovery helper had to
        patch the gap).
        """
        tmpl = _load_prompt("anonymization.txt", required=True)
        rendered = self._render(tmpl)
        assert "PascalCase" in rendered or "Prefix" in rendered
        assert "UNIQUE" in rendered or "unique" in rendered
        assert "totality" in rendered.lower() or "every placeholder" in rendered.lower()
        assert "real_value" in rendered or "real value" in rendered.lower()
        # Diverse-prefix examples present — illustrative breadth signals
        # the model that prefixes outside the common {Person, City, Country,
        # Org, Thing} set are valid.  The examples now draw on the
        # contact-PII default (Phone/Email/Address/Profile), not
        # organization/project surfaces (see the scope-inversion tests below).
        assert "Phone" in rendered or "Email" in rendered, (
            "File-based prompt must show at least one type-appropriate prefix "
            "outside the common {Person, City, Country, Org, Thing} set so the "
            "model knows the prefix vocabulary is open."
        )

    def test_no_stray_unescaped_placeholders_in_default(self):
        """After rendering, no stray {word} tokens should remain (only JSON literal braces)."""
        import re

        rendered = self._render(_load_prompt("anonymization.txt", required=True))
        # JSON literal braces are escaped as {{ }} in the template and appear as { } after render.
        # A simple check: no single { immediately followed by a letter (unrendered placeholder).
        stray = re.findall(r"(?<!\{)\{[a-z_]+\}", rendered)
        assert not stray, f"Stray unrendered placeholder(s) found in rendered prompt: {stray!r}"

    def test_speaker_rule_present_and_no_example_maps_it(self):
        """The prompt states speaker{N} is already anonymous and must
        stay verbatim; NONE of the worked examples map a speaker{N}
        token (the rule generalises — it isn't taught via an example
        that could drift out of sync with the rule statement)."""
        tmpl = _load_prompt("anonymization.txt", required=True)
        lower = tmpl.lower()
        assert "speaker" in lower and "verbatim" in lower, (
            "anonymization.txt must state the speaker{N}-stays-verbatim rule."
        )
        examples_section = tmpl[tmpl.index("## Examples") :]
        assert "speaker" not in examples_section.lower(), (
            "No anonymization.txt example may map a speaker{N} token."
        )
        # Must still render cleanly with the new rule line present.
        rendered = self._render(tmpl)
        assert "{facts_json}" not in rendered
        assert "{transcript}" not in rendered

    def test_scrub_categories_slot_present(self):
        """The prompt must accept a {scrub_categories} slot and echo the
        rendered value — the config-driven scope authority."""
        tmpl = _load_prompt("anonymization.txt", required=True)
        assert "{scrub_categories}" in tmpl, (
            "anonymization.txt must declare a {scrub_categories} format slot."
        )
        rendered = tmpl.format(
            scrub_categories="person name, phone number, some very unusual category",
            facts_json="[]",
            transcript="x",
        )
        assert "some very unusual category" in rendered, (
            "A non-default scrub value must reach the rendered prompt verbatim — "
            "the guard against a dead config knob."
        )

    def test_returns_two_artifacts_not_mapping_only(self):
        """The output contract must ask for BOTH `mapping` and
        `anonymized_transcript` — the prior 'mapping — nothing else' /
        'do NOT rewrite the transcript' contract is retired."""
        tmpl = _load_prompt("anonymization.txt", required=True)
        assert "anonymized_transcript" in tmpl, (
            "anonymization.txt must request the anonymized_transcript artifact."
        )
        assert "mapping" in tmpl
        assert "do not rewrite" not in tmpl.lower() and "do not echo" not in tmpl.lower(), (
            "anonymization.txt must not forbid rewriting the transcript — the model now authors it."
        )
        # The final output-contract line must not restrict the response to
        # `mapping` alone.
        tail = tmpl.strip().splitlines()[-1]
        assert "mapping" in tail and "anonymized_transcript" in tail, (
            f"Final instruction line must name both output keys, got: {tail!r}"
        )

    def test_city_org_product_are_not_positive_scrub_examples(self):
        """Scope-inversion: the old
        Berlin->City_1, Siemens->Org_1, iPhone 15 Pro->Product_1 POSITIVE
        examples over-scrub now that the code-side entity-type gate is
        gone — the prompt is the only scope boundary. None of them may
        appear as a placeholder-mapping VALUE (i.e. actually scrubbed) in
        the prompt body."""
        tmpl = _load_prompt("anonymization.txt", required=True)
        for forbidden in ("City_1", "Org_1", "Product_1"):
            assert forbidden not in tmpl, (
                f"anonymization.txt must not positively scrub-example {forbidden!r} — "
                "city/org/product are out-of-scope by default and the prompt "
                "is the sole scope authority (no code-side narrowing remains)."
            )

    def test_phone_and_email_positive_examples_present(self):
        """Add phone/email POSITIVE examples now that
        structured values (not just names) are scrubbed by the prompt."""
        tmpl = _load_prompt("anonymization.txt", required=True)
        examples_section = tmpl[tmpl.index("## Examples") :]
        assert "Phone_1" in examples_section, (
            "anonymization.txt must include a positive phone-number scrub example."
        )
        assert "Email_1" in examples_section, (
            "anonymization.txt must include a positive email-address scrub example."
        )

    def test_fidelity_and_consistency_instruction_present(self):
        """The prompt must instruct the model to rewrite the
        transcript changing ONLY the listed PII, using the SAME
        placeholders as `mapping` (fidelity + consistency)."""
        tmpl = _load_prompt("anonymization.txt", required=True)
        lower = tmpl.lower()
        assert "anonymized_transcript" in lower
        assert "same placeholder" in lower, (
            "anonymization.txt must instruct placeholder consistency between "
            "`mapping` and `anonymized_transcript`."
        )
        assert "changing only" in lower or "only the" in lower, (
            "anonymization.txt must instruct that the transcript rewrite changes "
            "only the in-scope PII, preserving everything else (fidelity)."
        )


class TestAnonymizationFactsPrompt:
    """Contract tests for ``anonymization_facts.txt`` — the graph-tier,
    facts-only anonymization variant (U4,
    .agent/plan-anonymize-slicing.md). Shares the session-tier
    ``anonymization.txt``'s core contract (shape, uniqueness, totality,
    direction, the ``speaker{N}``-verbatim rule) but drops the
    transcript-rewrite half entirely: no ``{transcript}`` slot, no
    ``anonymized_transcript`` in the output contract.
    """

    def _render(self, tmpl: str) -> str:
        return tmpl.format(
            scrub_categories="person name, phone number",
            facts_json='[{"subject": "Rowan", "predicate": "lives_in", '
            '"object": "Millbrook", "relation_type": "factual", "confidence": 0.9}]',
        )

    def test_renders_with_scrub_categories_and_facts_json_no_transcript_slot(self):
        """Item 27: renders with {scrub_categories} and {facts_json} and
        declares no {transcript} slot; no stray unrendered {word} after
        formatting."""
        tmpl = _load_prompt("anonymization_facts.txt", required=True)
        assert "{transcript}" not in tmpl, (
            "anonymization_facts.txt must not declare a {transcript} slot — "
            "there is no transcript at the graph tier."
        )
        rendered = self._render(tmpl)
        assert "{facts_json}" not in rendered
        assert "{scrub_categories}" not in rendered
        stray = re.findall(r"(?<!\{)\{[a-z_]+\}", rendered)
        assert not stray, f"Stray unrendered placeholder(s) found: {stray!r}"

    def test_shares_core_contract_shape_uniqueness_totality_direction(self):
        """Item 28: carries the shared core contract — shape (PascalCase
        + `_<N>`), uniqueness, totality, direction, and the
        `speaker{N}`-verbatim rule, with no example mapping a `speaker{N}`
        token."""
        tmpl = _load_prompt("anonymization_facts.txt", required=True)
        rendered = self._render(tmpl)
        assert "PascalCase" in rendered or "Prefix" in rendered
        assert "UNIQUE" in rendered or "unique" in rendered
        assert "totality" in rendered.lower() or "every placeholder" in rendered.lower()
        assert "real_value" in rendered or "real value" in rendered.lower()

        lower = tmpl.lower()
        assert "speaker" in lower and "verbatim" in lower, (
            "anonymization_facts.txt must state the speaker{N}-stays-verbatim rule."
        )
        examples_section = tmpl[tmpl.index("## Examples") :]
        assert "speaker" not in examples_section.lower(), (
            "No anonymization_facts.txt example may map a speaker{N} token."
        )

    def test_examples_use_fictional_entities_no_city_org_product_positive_example(self):
        """Item 29: examples use fictional entities only and contain no
        city/org/product positive scrub example (City_1/Org_1/Product_1
        absent)."""
        tmpl = _load_prompt("anonymization_facts.txt", required=True)
        for forbidden in ("City_1", "Org_1", "Product_1"):
            assert forbidden not in tmpl, (
                f"anonymization_facts.txt must not positively scrub-example {forbidden!r}."
            )

    def test_output_contract_is_mapping_only(self):
        """Item 30: the facts template requests `mapping` and ONLY
        `mapping` — no `anonymized_transcript` key, no `{transcript}`
        slot — and renders correctly through the shared
        `_render_anonymize_prompt` despite the unused `transcript=`
        keyword."""
        from paramem.cloud.anonymize import _render_anonymize_prompt

        tmpl = _load_prompt("anonymization_facts.txt", required=True)
        assert "anonymized_transcript" not in tmpl, (
            "anonymization_facts.txt must not request anonymized_transcript — "
            "there is no transcript at the graph tier."
        )
        tail = tmpl.strip().splitlines()[-1]
        assert "mapping" in tail
        assert "anonymized_transcript" not in tail

        class _CountingTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "".join(m["content"] for m in messages)

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": list(range(len(text)))}

        # Renders cleanly despite the unused transcript= keyword — proves
        # the str.format-ignores-unused-kwargs claim the shared renderer
        # relies on.
        rendered = _render_anonymize_prompt(
            [{"subject": "Rowan", "predicate": "lives_in", "object": "Millbrook"}],
            _CountingTokenizer(),
            scrub={"person name"},
            transcript="this argument is unused by anonymization_facts.txt",
            user_prompt_template=tmpl,
            system_prompt="",
        )
        assert "this argument is unused" not in rendered
        assert "{transcript}" not in rendered

    def test_graph_enrich_loads_the_facts_variant_session_tier_loads_transcript_variant(self):
        """Item 31: ``graph_enrich`` loads ``anonymization_facts.txt``
        (and the session tier still loads ``anonymization.txt``) — a
        direct source-level assertion at the chokepoint, so a copy-paste
        regression to the transcript variant is caught.

        A live ``_load_prompt`` spy over a full ``enrich_graph`` call is
        not used here: every early-return guard (``no_model``, the
        10-node floor, ``cloud_egress_blocked``) short-circuits BEFORE
        the prompt load this test cares about, so a spy would only prove
        the guards fire, not which template name is hardcoded past them.
        """
        import inspect

        import paramem.training.graph_enrich as graph_enrich_module

        source = inspect.getsource(graph_enrich_module.enrich_graph)
        assert '"anonymization_facts.txt"' in source
        assert '"anonymization.txt"' not in source

        import paramem.graph.stage_anonymize as stage_anonymize_module

        session_source = inspect.getsource(stage_anonymize_module._stage_anonymize)
        assert '"anonymization.txt"' in session_source
        assert '"anonymization_facts.txt"' not in session_source


class TestEntityCorrectionPrompt:
    """Contract tests for ``entity_correction.txt`` (paramem.graph.entity_correction).

    The module calls ``template.format(context=..., value=...)`` — the slot
    names must match exactly. The output contract now carries FOUR fields
    (``input``, ``kind``, ``corrected``, ``is_known_entity``); ``kind`` is a
    strict enum (``place``/``organization``/``concept``/``person``/``other``)
    that structurally excludes person-name correction — the enum values and
    all three few-shot examples (real correction, person left unchanged,
    fiction left unchanged) must be present so the gate the module reads has
    evidence backing it.
    """

    def _render(self, tmpl: str) -> str:
        return tmpl.format(context="place", value="Frankfrut")

    def test_renders_without_format_errors(self):
        rendered = self._render(_load_prompt("entity_correction.txt", required=True))
        assert "{context}" not in rendered
        assert "{value}" not in rendered

    def test_no_stray_unescaped_placeholders(self):
        """No stray {word} tokens remain after render (only JSON literal braces)."""
        rendered = self._render(_load_prompt("entity_correction.txt", required=True))
        stray = re.findall(r"(?<!\{)\{[a-z_]+\}", rendered)
        assert not stray, f"Stray unrendered placeholder(s) found in rendered prompt: {stray!r}"

    def test_contains_output_contract_tokens(self):
        """The output-contract keys `kind` and `is_known_entity` must appear."""
        rendered = self._render(_load_prompt("entity_correction.txt", required=True))
        assert "kind" in rendered
        assert "is_known_entity" in rendered

    def test_contains_strict_kind_enum_values(self):
        """All five `kind` enum values must be named in the prompt."""
        rendered = self._render(_load_prompt("entity_correction.txt", required=True))
        for value in ("place", "organization", "concept", "person", "other"):
            assert value in rendered, f"kind enum value {value!r} missing from prompt"

    def test_contains_all_three_example_markers(self):
        """All three few-shot examples (real correction, person unchanged,
        fiction unchanged) must be present and render cleanly."""
        rendered = self._render(_load_prompt("entity_correction.txt", required=True))
        assert "POSITIVE" in rendered
        assert rendered.count("NEGATIVE") >= 2
        # The positive example must actually change the surface.
        assert "Frankfurt" in rendered
        # The person example must be present and left unchanged.
        assert "Angela Merkl" in rendered
        assert '"kind": "person"' in rendered
        # The fiction example must be present and rejected.
        assert "Vellmarn" in rendered
        assert '"is_known_entity": false' in rendered


class TestMergerCoexistencePrompt:
    """Contract tests for the merger coexistence prompt.

    The 2-way parser in ``check_predicate_coexistence`` keys on the literal
    strings ``COEXIST`` and ``REPLACE``.  The prompt classifies the predicate
    alone (no value pair) via a single ``{predicate}`` slot.
    """

    def _load(self):
        return _load_prompt("merger_coexistence.txt", required=True)

    def test_renders_without_leftover_slots(self):
        """The ``{predicate}`` slot fills; no leftover ``{slot}`` tokens remain."""
        tmpl = self._load()
        rendered = tmpl.format(predicate="owns_pet")
        assert "{predicate}" not in rendered

    def test_coexist_keyword_present(self):
        """``COEXIST`` must survive rendering — the parser keys on this literal."""
        tmpl = self._load()
        rendered = tmpl.format(predicate="owns_pet")
        assert "COEXIST" in rendered

    def test_replace_keyword_present(self):
        """``REPLACE`` must survive rendering — the parser keys on this literal."""
        tmpl = self._load()
        rendered = tmpl.format(predicate="owns_pet")
        assert "REPLACE" in rendered

    def test_aggregate_keyword_absent(self):
        """``AGGREGATE`` must NOT appear in the rendered prompt — the 2-way parser
        no longer expects or emits it; its presence would confuse the model.
        """
        tmpl = self._load()
        rendered = tmpl.format(predicate="speaks")
        assert "AGGREGATE" not in rendered, (
            "Prompt must not contain AGGREGATE — fold is now purely additive"
        )

    def test_only_predicate_slot_present(self):
        """The prompt must use only ``{predicate}`` — no value-pair slots."""
        import re

        tmpl = self._load()
        slots = re.findall(r"\{(\w+)\}", tmpl)
        assert set(slots) == {"predicate"}, f"Expected only {{predicate}} slot; found: {set(slots)}"


class TestCheckPredicateCoexistenceParser:
    """Unit tests for the 2-way verdict parser in check_predicate_coexistence.

    These tests mock the model's generate_answer to return specific output
    strings and verify the parser extracts the correct verdict.
    They do NOT require a real GPU or model.
    """

    def _call_with_mock_output(self, output: str) -> str:
        """Drive check_predicate_coexistence with a mocked model output.

        ``generate_answer`` and ``adapt_messages`` are imported locally inside
        ``check_predicate_coexistence`` (lazy import), so we patch them at their
        definition site (``paramem.evaluation.recall`` and
        ``paramem.models.loader``), not via the merger module namespace.
        """
        from unittest.mock import MagicMock, patch

        from paramem.graph.merger import check_predicate_coexistence

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted"

        with patch("paramem.evaluation.recall.generate_answer", return_value=output):
            with patch(
                "paramem.models.loader.adapt_messages",
                return_value=[{"role": "user", "content": "test"}],
            ):
                return check_predicate_coexistence(
                    "Alex",
                    "speaks",
                    model,
                    tokenizer,
                    "Classify {predicate}: COEXIST or REPLACE",
                    "You classify relationship cardinality.",
                )

    def test_coexist_verdict_parsed(self):
        """Model output 'COEXIST' → 'COEXIST'."""
        verdict = self._call_with_mock_output("COEXIST")
        assert verdict == "COEXIST"

    def test_replace_verdict_parsed(self):
        """Model output 'REPLACE' → 'REPLACE'."""
        verdict = self._call_with_mock_output("REPLACE")
        assert verdict == "REPLACE"

    def test_ambiguous_output_defaults_to_coexist(self):
        """Unrecognised model output → 'COEXIST' safer default."""
        verdict = self._call_with_mock_output("MAYBE")
        assert verdict == "COEXIST"


class TestSpeakerDirectiveFile:
    """Contract tests for ``configs/prompts/speaker_directive.txt``.

    The file holds sentinel-delimited sections consumed by separate callers:

    * ``EXTRACTION-DIRECTIVE`` — loaded by ``build_speaker_context`` and
      injected into the extraction user prompt via ``{speaker_context}``.
    * ``THIRD-PARTY-DESCRIPTOR`` — loaded at module import by ``inference.py``
      as the neutral label for unresolvable ``speaker{N}`` tokens (e.g.
      anonymous or unknown profiles).

    ``INFERENCE-IDENTITY`` was deleted in Phase B (speaker-identity refactor):
    id-to-name resolution is now handled at the fact-render boundary via
    ``entry_fact_text(resolve=...)`` / ``MemoryStore.probe(speaker_resolver=...)``,
    not via a prompt injection.  Tests verify the new section layout.
    """

    def test_file_exists(self):
        """speaker_directive.txt must be present under the default prompt dir."""
        path = _DEFAULT_PROMPT_DIR / "speaker_directive.txt"
        assert path.exists(), f"speaker_directive.txt not found at {path}"

    def test_inference_identity_deleted_raises_key_error(self):
        """INFERENCE-IDENTITY section is deleted; loading it must raise KeyError."""
        import pytest

        from paramem.graph.prompts import _load_speaker_directive_section

        with pytest.raises(KeyError, match="INFERENCE-IDENTITY"):
            _load_speaker_directive_section("INFERENCE-IDENTITY")

    def test_third_party_descriptor_loads_non_empty(self):
        """THIRD-PARTY-DESCRIPTOR section loads successfully and is non-empty."""
        from paramem.graph.prompts import _load_speaker_directive_section

        descriptor = _load_speaker_directive_section("THIRD-PARTY-DESCRIPTOR")
        assert descriptor, "THIRD-PARTY-DESCRIPTOR section must be non-empty"

    def test_third_party_descriptor_value(self):
        """THIRD-PARTY-DESCRIPTOR must be 'another speaker'."""
        from paramem.graph.prompts import _load_speaker_directive_section

        descriptor = _load_speaker_directive_section("THIRD-PARTY-DESCRIPTOR")
        assert descriptor == "another speaker"

    def test_extraction_directive_intact(self):
        """EXTRACTION-DIRECTIVE section is intact and non-empty after refactor."""
        from paramem.graph.prompts import _load_speaker_directive_section

        extraction = _load_speaker_directive_section("EXTRACTION-DIRECTIVE")
        assert extraction, "EXTRACTION-DIRECTIVE section must be non-empty"

    def test_extraction_directive_renders_slots(self):
        """EXTRACTION-DIRECTIVE section renders {speaker_id} and {speaker_name} slots."""
        from paramem.graph.prompts import _load_speaker_directive_section

        tmpl = _load_speaker_directive_section("EXTRACTION-DIRECTIVE")
        rendered = tmpl.format(speaker_id="speaker0", speaker_name="Alice")
        assert "speaker0" in rendered
        assert "Alice" in rendered
        # No unrendered slot tokens remain.
        assert "{speaker_id}" not in rendered
        assert "{speaker_name}" not in rendered

    def test_unknown_section_raises_key_error(self):
        """Requesting a non-existent section raises KeyError immediately."""
        import pytest

        from paramem.graph.prompts import _load_speaker_directive_section

        with pytest.raises(KeyError, match="NONEXISTENT"):
            _load_speaker_directive_section("NONEXISTENT")

    def test_build_speaker_context_two_arg_renders_speaker_id(self):
        """build_speaker_context(speaker_id, speaker_name) pins id as subject."""
        ctx = build_speaker_context("speaker0", "Alice")
        assert "speaker0" in ctx
        # Display name present as comprehension context.
        assert "Alice" in ctx
        # No unrendered slot tokens.
        assert "{speaker_id}" not in ctx
        assert "{speaker_name}" not in ctx

    def test_build_speaker_context_empty_id_returns_empty(self):
        """build_speaker_context with empty/None speaker_id returns empty string."""
        assert build_speaker_context("", "Alice") == ""
        assert build_speaker_context(None, "Alice") == ""
        assert build_speaker_context(None, None) == ""

    def test_build_speaker_context_no_display_name(self):
        """Anonymous speaker: id used in place of display name; no KeyError."""
        ctx = build_speaker_context("speaker0", None)
        assert "speaker0" in ctx
        # No unrendered slot tokens.
        assert "{speaker_id}" not in ctx
        assert "{speaker_name}" not in ctx

    def test_cloud_graph_enrichment_contains_speaker_id_note(self):
        """cloud_graph_enrichment.txt must instruct the model that nodes IN
        SCOPE for the operator's privacy policy are opaque placeholder
        tokens, forbid de-anonymizing/renaming/inventing one, and forbid a
        ``same_as`` pair between two ``speaker{N}`` identifiers.

        Speaker endpoints are NEVER tokenised at either tier — same as the
        session-tier anonymizer prompt, where the bare ``speaker{N}``
        anchor is left untouched (it is already anonymous and carries no
        identifying information; see ``graph_enrich.py``'s
        ``enrich_graph`` docstring and ``SECURITY.md``). Nodes
        outside the operator's ``scrub`` categories (e.g. organizations,
        which are not part of the default ``scrub`` — a load-bearing set
        of name / phone / address / online-identity sub-terms) also
        appear verbatim — the prompt must not claim every
        ``subject``/``object`` is a token.
        """
        tmpl = _load_prompt("cloud_graph_enrichment.txt", "")
        lower = tmpl.lower()
        # The note must reference speaker endpoints and the token/system framing.
        assert "speaker" in lower and ("identifier" in lower or "system" in lower), (
            "cloud_graph_enrichment.txt must note that speaker endpoints are "
            "system-generated tokens."
        )
        assert "token" in lower, (
            "cloud_graph_enrichment.txt must describe node names as placeholder tokens."
        )
        assert "do not" in lower or "never" in lower, (
            "cloud_graph_enrichment.txt must forbid de-anonymizing, renaming, or inventing tokens."
        )

    def test_cloud_graph_enrichment_part1_examples_use_tokens_not_real_names(self):
        """Part 1's worked examples must use placeholder tokens, not the
        example real names the prompt used before this contract was wired
        up (Alice/Bob/Acme/Stanford/Portland) — a live cloud call must
        never see them, worked example or not. Scoped to Part 1's text
        only — Part 2 (``same_as``) legitimately keeps real-name examples,
        but they must be organization/place/thing names, not person names:
        under the shipped ``{"person"}`` scope persons are always tokens
        by the time this pass runs, so a person-name example there would
        illustrate an unreachable task. See
        ``TestSpeakerDirectiveFile::test_cloud_graph_enrichment_part2_examples_are_not_person_names``.
        """
        tmpl = _load_prompt("cloud_graph_enrichment.txt", "")
        part1 = tmpl.split("## Part 1")[1].split("## Part 2")[0]
        for leaked_name in ("Alice", "Bob", "Acme", "Stanford", "Portland"):
            assert leaked_name not in part1, (
                f"cloud_graph_enrichment.txt Part 1 must use placeholder tokens, "
                f"not the real example name {leaked_name!r}."
            )

    def test_cloud_graph_enrichment_part2_examples_are_not_person_names(self):
        """Part 2's SAME_AS examples must be drawn from surfaces this pass
        actually sees — organizations, places, things — never person
        names, since under the shipped ``{"person"}`` scope persons are
        always opaque tokens by the time this pass runs and never appear
        as real-name surfaces. A person-name example there would coach
        the model on a task it structurally cannot perform, while line 2
        of the prompt simultaneously forbids guessing the identity behind
        a token — a self-contradiction.

        Mutation: reintroduce a person-name SAME_AS example (e.g. ``"Yang
        Ming"`` / ``"Mr. Yang"``) in Part 2 -> this test fails.
        """
        tmpl = _load_prompt("cloud_graph_enrichment.txt", "")
        part2 = tmpl.split("## Part 2")[1].split("## Input triples")[0]
        for leaked_person_name in (
            "Yang Ming",
            "Mr. Yang",
            "Robert Smith",
            "Bob Smith",
            "Alicia Smith",
            "Zhang Min",
            "Wang Min",
            "Sara",
            "Sarah",
        ):
            assert leaked_person_name not in part2, (
                f"cloud_graph_enrichment.txt Part 2 must not use the person-name "
                f"example {leaked_person_name!r} — persons are always tokens "
                f"under the shipped scope; use org/place/thing surfaces instead."
            )


class TestNameExtractionPrompt:
    """Contract tests for the name-extraction prompt files.

    Guards against:
    - Missing or broken ``{transcript}`` slot (would cause KeyError at render time).
    - Absence of NONE keyword (would break the parser that rejects the sentinel).
    - Absence of role/occupation negative teaching (the root cause of the
      "data scientist" false-positive bug).
    - No inline prompt remaining in app.py (verified separately by the
      no-inline-prompt grep test below).
    """

    def _load_system(self) -> str:
        from paramem.graph.prompts import _load_prompt

        return _load_prompt("name_extraction_system.txt", "")

    def _load_user(self) -> str:
        from paramem.graph.prompts import _load_prompt

        return _load_prompt("name_extraction.txt", "")

    def test_system_file_exists_and_is_non_empty(self):
        """name_extraction_system.txt must exist and be non-empty."""
        content = self._load_system()
        assert content, "name_extraction_system.txt must be non-empty"

    def test_user_file_exists_and_renders_transcript_slot(self):
        """name_extraction.txt must render with {transcript} slot without KeyError."""
        tmpl = self._load_user()
        rendered = tmpl.format(transcript="user: Hi, I'm Alex.")
        assert "{transcript}" not in rendered
        assert "Alex" in rendered

    def test_user_file_no_format_errors_with_empty_transcript(self):
        """Rendering with an empty transcript must not raise."""
        tmpl = self._load_user()
        rendered = tmpl.format(transcript="")
        assert "{transcript}" not in rendered

    def test_none_keyword_present(self):
        """Both files must teach the NONE sentinel — the post-filter keys on it."""
        system = self._load_system()
        user = self._load_user()
        assert "NONE" in system or "NONE" in user, (
            "Name extraction prompts must contain the NONE sentinel."
        )

    def test_role_occupation_negative_present(self):
        """The user prompt must explicitly state that a job title / role is NOT a name.

        This is the root-cause fix for the 'data scientist' false-positive:
        the original inline prompt had no negative teaching for occupations.
        The file must contain model-driven negative teaching (few-shot or
        explicit rule), not a static denylist.
        """
        tmpl = self._load_user()
        lower = tmpl.lower()
        # At least one of these phrases must appear to teach the occupation negative.
        occupation_negatives = [
            "job title",
            "occupation",
            "role",
            "data scientist",
            "engineer",
        ]
        hits = [phrase for phrase in occupation_negatives if phrase in lower]
        assert len(hits) >= 2, (
            f"Name extraction prompt must teach that a job title / occupation / role "
            f"is NOT a name (model-driven negatives). "
            f"Found only: {hits!r} out of {occupation_negatives!r}"
        )

    def test_no_inline_name_prompt_in_app_py(self):
        """Confirm app.py no longer contains the old inline name-extraction prompt strings.

        The inline prompt was the root cause of: (a) no occupation negatives,
        (b) no user-turn filtering.  Its removal is load-bearing — this test
        guards the regression.
        """
        from pathlib import Path

        app_src = (
            Path(__file__).resolve().parents[1] / "paramem" / "server" / "app.py"
        ).read_text()
        # The old system_msg verbatim fragment that's now gone.
        assert "You extract speaker names from conversation transcripts." not in app_src, (
            "Inline name-extraction system prompt detected in app.py — "
            "it must be removed and loaded from name_extraction_system.txt instead."
        )
        # The old user_msg verbatim fragment.
        assert "Extract the speaker's self-introduced name from this transcript." not in app_src, (
            "Inline name-extraction user prompt detected in app.py — "
            "it must be removed and loaded from name_extraction.txt instead."
        )
