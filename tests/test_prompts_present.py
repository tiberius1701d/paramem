"""Ship-gate tests — assert required prompt files exist and carry expected placeholders.

The extraction prompt-pair (``extraction.txt`` + ``extraction_system.txt``)
plus the procedural user template (``extraction_procedural.txt``) is the
single ground truth for extraction.  Document chunks land in the same
``{transcript}`` slot at the chat-template layer; there are no
document-variant prompt files.  The retired
``extraction_document.txt`` / ``extraction_system_document.txt`` /
``extraction_procedural_document.txt`` files are deliberately absent —
their existence used to permit silent drift on schema-shape rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.anonymizer_doubles import (
    INVALID_ANONYMIZATION_SECTIONS_DOUBLED_SLOT,
    INVALID_ANONYMIZATION_SECTIONS_MALFORMED_PLACEHOLDER,
    INVALID_ANONYMIZATION_SECTIONS_MISSING_SECTION,
    INVALID_ANONYMIZATION_SECTIONS_MISSING_SLOT,
    INVALID_ANONYMIZATION_SECTIONS_UNKNOWN_SLOT,
    VALID_ANONYMIZATION_SECTIONS,
)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "configs" / "prompts"


class TestPromptFilesPresent:
    def test_extraction_txt_exists(self):
        assert (_PROMPTS_DIR / "extraction.txt").exists()

    def test_extraction_system_txt_exists(self):
        assert (_PROMPTS_DIR / "extraction_system.txt").exists()

    def test_extraction_procedural_txt_exists(self):
        assert (_PROMPTS_DIR / "extraction_procedural.txt").exists()

    def test_document_directive_txt_exists(self):
        """The externalized document-provenance directive, rendered into
        the ``{document_context}`` slot by
        :func:`paramem.graph.extractor.build_document_context`."""
        assert (_PROMPTS_DIR / "document_directive.txt").exists()

    def test_document_directive_txt_has_speaker_placeholders(self):
        content = (_PROMPTS_DIR / "document_directive.txt").read_text()
        assert "{speaker_id}" in content
        assert "{speaker_name}" in content

    def test_extraction_system_txt_no_braces(self):
        """Regression guard: system prompt must be plain-English directives only.

        The system prompt is passed verbatim to the model — no slot substitution
        is performed on it.  Any ``{`` character in the file means someone
        accidentally re-introduced a template slot that will never be filled,
        potentially leaking the raw brace syntax into the model context.
        """
        content = (_PROMPTS_DIR / "extraction_system.txt").read_text()
        assert "{" not in content, (
            "extraction_system.txt contains '{' braces — system prompts "
            "are plain-English only; slot substitution runs only on user templates."
        )

    def test_extraction_system_txt_contains_json_keyword(self):
        """Regression guard: extraction_system.txt must mention JSON.

        The extraction system prompt must instruct the model to emit JSON.
        If a future edit drops that directive, extraction silently produces
        unparseable output.
        """
        content = (_PROMPTS_DIR / "extraction_system.txt").read_text()
        assert "JSON" in content, (
            "extraction_system.txt does not contain 'JSON' — the output "
            "directive may have been accidentally removed, which would break extraction."
        )

    def test_extraction_txt_has_json_output_directive(self):
        """Coarse contract check: JSON output schema keywords must be present.

        Ensures the user template carries the same output contract the parser expects.
        """
        content = (_PROMPTS_DIR / "extraction.txt").read_text()
        assert "entities" in content, (
            "extraction.txt missing 'entities' keyword — JSON output contract may be broken."
        )
        assert "relations" in content, (
            "extraction.txt missing 'relations' keyword — JSON output contract may be broken."
        )

    def test_extraction_procedural_txt_has_json_output_directive(self):
        """Procedural template must carry the JSON output directive."""
        content = (_PROMPTS_DIR / "extraction_procedural.txt").read_text()
        assert "JSON" in content, (
            "extraction_procedural.txt missing 'JSON' keyword — the output "
            "directive may be missing, breaking procedural parsing."
        )

    def test_extraction_second_order_txt_exists(self):
        assert (_PROMPTS_DIR / "extraction_second_order.txt").exists()


class TestSystemPromptFilesPresent:
    """Presence + brace guard for the nine externalized SYSTEM-prompt files.

    Six follow the companion ``<base>_system.txt`` pattern for an
    already-external USER template (``extraction.txt`` /
    ``extraction_system.txt``); three are serving-path system prompts
    (``serving_system.txt``, ``intent_classifier.txt``,
    ``cloud_serving_system.txt``) that carry verbatim system-role content
    with no slot substitution, so they belong on the same brace guard.
    ``recall_selection.txt`` is deliberately excluded — it carries JSON
    literal braces by design (see ``TestServingPrompts`` in
    ``test_prompts_contract.py``). See :func:`test_extraction_system_txt_no_braces`
    for the rationale on the brace guard — system prompts receive no slot
    substitution, so a stray ``{`` would leak raw template syntax into the
    model context.
    """

    _SYSTEM_PROMPT_FILES = (
        "entity_correction_system.txt",
        "merger_coexistence_system.txt",
        "cloud_plausibility_system.txt",
        "cloud_enrichment_system.txt",
        "predicate_normalization_system.txt",
        "cloud_graph_enrichment_system.txt",
        "serving_system.txt",
        "intent_classifier.txt",
        "cloud_serving_system.txt",
    )

    def test_entity_correction_system_txt_exists(self):
        assert (_PROMPTS_DIR / "entity_correction_system.txt").exists()

    def test_merger_coexistence_system_txt_exists(self):
        assert (_PROMPTS_DIR / "merger_coexistence_system.txt").exists()

    def test_cloud_plausibility_system_txt_exists(self):
        assert (_PROMPTS_DIR / "cloud_plausibility_system.txt").exists()

    def test_cloud_enrichment_system_txt_exists(self):
        assert (_PROMPTS_DIR / "cloud_enrichment_system.txt").exists()

    def test_predicate_normalization_system_txt_exists(self):
        assert (_PROMPTS_DIR / "predicate_normalization_system.txt").exists()

    def test_cloud_graph_enrichment_system_txt_exists(self):
        assert (_PROMPTS_DIR / "cloud_graph_enrichment_system.txt").exists()

    def test_serving_system_txt_exists(self):
        assert (_PROMPTS_DIR / "serving_system.txt").exists()

    def test_intent_classifier_txt_exists(self):
        assert (_PROMPTS_DIR / "intent_classifier.txt").exists()

    def test_cloud_serving_system_txt_exists(self):
        assert (_PROMPTS_DIR / "cloud_serving_system.txt").exists()

    def test_all_system_prompt_files_no_braces(self):
        for filename in self._SYSTEM_PROMPT_FILES:
            content = (_PROMPTS_DIR / filename).read_text()
            assert "{" not in content, (
                f"{filename} contains '{{' braces — system prompts are "
                "plain-English only; slot substitution runs only on user templates."
            )


class TestSystemPromptGoldens:
    """Byte-for-byte goldens for the seven externalized system-prompt files
    that each carry a single-line literal.

    Each golden string is a hardcoded literal, never derived from the
    ``.txt`` file it is checked against, so a shared typo cannot silently
    pass both sides.  ``serving_system.txt`` and ``intent_classifier.txt``
    carry no golden here — they hold multi-paragraph prose
    (``configs/prompts/pa_voice.txt``), not a single-line literal; their
    content is covered by ``test_serving_prompt_contract.py`` instead.
    """

    def test_entity_correction_system_golden(self):
        content = (_PROMPTS_DIR / "entity_correction_system.txt").read_text().strip()
        assert content == "Output valid JSON only."

    def test_merger_coexistence_system_golden(self):
        content = (_PROMPTS_DIR / "merger_coexistence_system.txt").read_text().strip()
        assert content == "You classify relationship cardinality."

    def test_cloud_plausibility_system_golden(self):
        content = (_PROMPTS_DIR / "cloud_plausibility_system.txt").read_text().strip()
        assert content == (
            "You are a knowledge graph plausibility filter. Drop invalid facts "
            "only. Do NOT add or modify facts. Output valid JSON only."
        )

    def test_cloud_enrichment_system_golden(self):
        content = (_PROMPTS_DIR / "cloud_enrichment_system.txt").read_text().strip()
        assert content == (
            "You are a knowledge graph enrichment assistant. Resolve coreference "
            "and split compound facts. Do NOT remove facts — a separate "
            "plausibility filter handles removal. Output valid JSON only."
        )

    def test_predicate_normalization_system_golden(self):
        content = (_PROMPTS_DIR / "predicate_normalization_system.txt").read_text().strip()
        assert content == "You identify synonym predicate clusters. Output valid JSON only."

    def test_cloud_graph_enrichment_system_golden(self):
        content = (_PROMPTS_DIR / "cloud_graph_enrichment_system.txt").read_text().strip()
        assert content == (
            "You are a knowledge graph enrichment assistant operating over a "
            "pre-merged cross-transcript graph. Emit cross-session second-order "
            "relations and same_as pairs for duplicate entities. Output valid "
            "JSON only."
        )

    def test_cloud_serving_system_golden(self):
        content = (_PROMPTS_DIR / "cloud_serving_system.txt").read_text().strip()
        assert content == (
            "You are continuing a conversation as a personal assistant. "
            "Derive your persona, tone, and conversational style from the "
            "preceding conversation. Answer clearly and concisely in 1-3 spoken "
            "sentences. Do not use markdown, lists, or structured formatting."
        )


class TestTrainedRecallInterfacePin:
    """Pin the trained-recall interface — the weight-coupled training/probe
    pair every adapter in production was trained on
    (``configs/prompts/trained_recall.txt``).

    The expected strings below are hardcoded literals, never derived from
    ``configs/prompts/trained_recall.txt``, so a shared typo cannot silently
    pass both sides.

    _PIN_FAILURE_MESSAGE below is asserted on every failure: the trained
    recall interface is weight-coupled, so a text change here invalidates
    every adapter in production until it is retrained.
    """

    _PIN_FAILURE_MESSAGE = (
        "The trained recall interface is weight-coupled: every adapter in "
        "production was trained on this exact text. Changing it invalidates "
        "all trained adapters until they are retrained. If the change is "
        "intended, retrain every adapter and update this pin in the same change."
    )

    def test_trained_recall_system_prompt_pin(self):
        from paramem.training.dataset import trained_recall_system_prompt

        expected = (
            "You are a personal assistant with memory of your user's life. "
            "Answer questions about the user based on what you know about them."
        )
        assert trained_recall_system_prompt() == expected, self._PIN_FAILURE_MESSAGE

    def test_trained_recall_template_pin(self):
        from paramem.training.dataset import trained_recall_template

        expected = "Recall the fact stored under key '{key}'."
        assert trained_recall_template() == expected, self._PIN_FAILURE_MESSAGE

    def test_trained_recall_template_slot_pin(self):
        """Catches a slot rename (e.g. ``{key}`` -> ``{recall_key}``) that
        the exact-text pin above would also catch, but this makes the
        render-time failure mode explicit."""
        from paramem.training.dataset import trained_recall_template

        rendered = trained_recall_template().format(key="graph1")
        assert rendered == "Recall the fact stored under key 'graph1'.", self._PIN_FAILURE_MESSAGE


@pytest.mark.gpu
class TestEncodeBoundaryWeightCoupledPin:
    """Pin the single-BOS / mask-boundary invariant between the training
    encoding (:func:`~paramem.memory.entry.format_entry_training`) and the
    serving encoding (:func:`~paramem.training.dataset.build_inference_prompts`)
    against the real pinned Mistral tokenizer.

    Both paths render through the one production renderer
    (:func:`~paramem.models.loader.render_chat_prompt`) and tensorize through
    the one production tensorizer (:func:`~paramem.utils.tokens.encode_rendered`,
    always ``add_special_tokens=False``). This test proves that pairing holds
    end to end against the real tokenizer's chat template and added-token
    trie, not just at the unit level with a stub.

    _PIN_FAILURE_MESSAGE below is asserted on every failure: the chat
    template, the tokenizer's special-token policy, and
    ``add_special_tokens`` handling are ONE weight-coupled interface with the
    trained-recall text — every adapter in production was trained against
    this exact encode/render pairing.
    """

    _PIN_FAILURE_MESSAGE = (
        "The chat template, the tokenizer's special-token policy, and "
        "add_special_tokens handling are ONE weight-coupled interface with "
        "the trained-recall text: every adapter in production was trained "
        "against this exact encode/render pairing. Changing the chat "
        "template, the special-token policy, or add_special_tokens handling "
        "invalidates all trained adapters until they are retrained."
    )

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer

        from paramem.server.config import load_server_config

        cfg = load_server_config("tests/fixtures/server.yaml")
        return AutoTokenizer.from_pretrained(
            cfg.model_config.model_id,
            trust_remote_code=cfg.model_config.trust_remote_code,
        )

    @pytest.fixture(scope="class")
    def encodings(self, tokenizer):
        from paramem.memory.entry import format_entry_training
        from paramem.training.dataset import build_inference_prompts, trained_recall_template
        from paramem.utils.tokens import encode_rendered

        entry = {
            "key": "graph1",
            "subject": "alex",
            "predicate": "works at",
            "object": "acme corp",
        }
        training_example = format_entry_training([entry], tokenizer, max_length=1024)[0]
        serving_prompt = build_inference_prompts(
            [trained_recall_template().format(key=entry["key"])], tokenizer
        )[0]
        serving_encoded = encode_rendered(tokenizer, serving_prompt, return_tensors="pt")

        return {
            "training_ids": training_example["input_ids"].tolist(),
            "training_labels": training_example["labels"].tolist(),
            "serving_ids": serving_encoded["input_ids"][0].tolist(),
            "bos_id": tokenizer.bos_token_id,
            "eos_id": tokenizer.eos_token_id,
        }

    def test_training_encoding_has_single_leading_bos(self, encodings):
        ids = encodings["training_ids"]
        bos_id = encodings["bos_id"]
        assert ids.count(bos_id) == 1, self._PIN_FAILURE_MESSAGE
        assert ids[0] == bos_id, self._PIN_FAILURE_MESSAGE

    def test_training_encoding_ends_with_eos(self, encodings):
        """The template's trailing ``</s>`` must survive add_special_tokens=False
        via the tokenizer's added-token trie — if this ever breaks, training
        targets silently lose their stop token."""
        assert encodings["training_ids"][-1] == encodings["eos_id"], self._PIN_FAILURE_MESSAGE

    def test_serving_encoding_is_an_exact_prefix_of_training_encoding(self, encodings):
        serving_ids = encodings["serving_ids"]
        training_ids = encodings["training_ids"]
        assert training_ids[: len(serving_ids)] == serving_ids, self._PIN_FAILURE_MESSAGE

    def test_label_mask_boundary_equals_serving_encoding_length(self, encodings):
        """The -100 label mask must end EXACTLY where generation starts —
        i.e. exactly at the serving encoding's length."""
        labels = encodings["training_labels"]
        boundary = next(i for i, v in enumerate(labels) if v != -100)
        assert boundary == len(encodings["serving_ids"]), self._PIN_FAILURE_MESSAGE

    def test_serving_encoding_has_single_leading_bos(self, encodings):
        ids = encodings["serving_ids"]
        bos_id = encodings["bos_id"]
        assert ids.count(bos_id) == 1, self._PIN_FAILURE_MESSAGE
        assert ids[0] == bos_id, self._PIN_FAILURE_MESSAGE


class TestRetiredServingPromptFilesAbsent:
    """``pa_voice.txt`` and its marker convention are retired — six new
    files replace it (``serving_system.txt``, ``serving_directives.txt``,
    ``cloud_serving_system.txt``, ``intent_classifier.txt``,
    ``recall_selection.txt``, ``trained_recall.txt``). Its absence is the
    guard against silently reviving the ``##---SECTION---`` marker
    convention alongside the ``=== NAME ===`` sentinel convention that
    replaced it everywhere.
    """

    def test_pa_voice_txt_absent(self):
        assert not (_PROMPTS_DIR / "pa_voice.txt").exists(), (
            "pa_voice.txt has been re-introduced — the marker convention it "
            "carried is retired; serving prompts now live in their own files "
            "under the === NAME === sentinel convention."
        )


class TestRetiredDocumentPromptsAbsent:
    """The document-variant prompt files are retired — their absence is the
    architectural guard against silent drift on schema-shape rules.

    Restoring any of these files re-introduces the two-prompt design that
    produced drift on:
      * speaker-name fragmentation NEGATIVE example
      * concept POSITIVE example
    Add to the transcript prompt instead, or prepend/append at the slot
    layer if a source-specific extension is genuinely required.
    """

    def test_extraction_document_txt_absent(self):
        assert not (_PROMPTS_DIR / "extraction_document.txt").exists(), (
            "extraction_document.txt has been re-introduced — the project "
            "deliberately uses a single prompt-pair for every source type."
        )

    def test_extraction_system_document_txt_absent(self):
        assert not (_PROMPTS_DIR / "extraction_system_document.txt").exists(), (
            "extraction_system_document.txt has been re-introduced — "
            "system prompts are not source-type-specific."
        )

    def test_extraction_procedural_document_txt_absent(self):
        assert not (_PROMPTS_DIR / "extraction_procedural_document.txt").exists(), (
            "extraction_procedural_document.txt has been re-introduced — "
            "procedural extraction uses a single prompt for every source type."
        )


class TestEnsurePromptAssets:
    """Runtime startup guard mirroring the ship-gate presence tests above.

    ``ensure_prompt_assets`` runs first in the server lifespan so a broken
    checkout / non-editable pip install (prompts are not packaged) fails
    loudly instead of the extraction pipeline silently loading empty prompts.
    """

    def test_passes_with_real_prompt_dir(self):
        # A repo checkout always has configs/prompts/ with the required files.
        from paramem.graph.prompts import ensure_prompt_assets

        ensure_prompt_assets()

    def test_raises_when_dir_missing(self, monkeypatch, tmp_path):
        import paramem.graph.prompts as prompts_mod

        monkeypatch.setattr(prompts_mod, "_DEFAULT_PROMPT_DIR", tmp_path / "absent")
        with pytest.raises(RuntimeError) as exc_info:
            prompts_mod.ensure_prompt_assets()
        assert "Prompt asset directory not found" in str(exc_info.value)

    def test_raises_when_required_file_missing(self, monkeypatch, tmp_path):
        # Directory exists but lacks the load-bearing extraction files.
        import paramem.graph.prompts as prompts_mod

        monkeypatch.setattr(prompts_mod, "_DEFAULT_PROMPT_DIR", tmp_path)
        with pytest.raises(RuntimeError) as exc_info:
            prompts_mod.ensure_prompt_assets()
        assert "Required prompt file" in str(exc_info.value)

    def test_raises_when_document_directive_missing(self, monkeypatch, tmp_path):
        """``document_directive.txt`` is required with no fallback — its
        absence must surface at boot, not at the first document ingest."""
        import paramem.graph.prompts as prompts_mod

        for filename in prompts_mod._REQUIRED_PROMPT_FILES:
            if filename == "document_directive.txt":
                continue
            (tmp_path / filename).write_text("placeholder")
        monkeypatch.setattr(prompts_mod, "_DEFAULT_PROMPT_DIR", tmp_path)
        with pytest.raises(RuntimeError) as exc_info:
            prompts_mod.ensure_prompt_assets()
        assert "document_directive.txt" in str(exc_info.value)

    def test_raises_when_the_anonymization_home_is_invalid(self):
        """A ``--prompt-file``-shaped invalid override, substituted the
        same way :func:`~paramem.graph.prompts.prompt_overrides` does for
        the gate tool, turns into the boot ``RuntimeError`` naming the
        problem and the file it came from (the override label, since this
        copy is substituted rather than resolved from disk) — every other
        required file resolves from the real shipped tree unaffected."""
        from paramem.graph.prompts import (
            ANONYMIZATION_PROMPT_FILE,
            ensure_prompt_assets,
            prompt_overrides,
        )

        with prompt_overrides(
            {ANONYMIZATION_PROMPT_FILE: INVALID_ANONYMIZATION_SECTIONS_MISSING_SECTION}
        ):
            with pytest.raises(RuntimeError) as exc_info:
                ensure_prompt_assets()
        message = str(exc_info.value)
        assert f"<override:{ANONYMIZATION_PROMPT_FILE}>" in message
        assert "missing section ANCHOR-SYSTEM" in message

    def test_reports_every_problem_across_both_files_and_names_the_loaded_path(self, tmp_path):
        """One call surfaces every problem in one report: two defects in
        the anonymization home (SCAN missing its own ``{text}`` slot;
        ANCHOR carrying an unlisted ``{extra}`` slot) plus one defect in an
        extraction template (an unlisted ``{surprise}`` slot) under one
        operator ``prompts_dir`` — each problem naming the file it belongs
        to, and both files' problems present in the one raised message."""
        import paramem.graph.prompts as prompts_mod

        two_anonymization_defects = (
            "=== SCAN-SYSTEM ===\nx\n\n"
            "=== SCAN ===\n{keywords}\n\n"  # defect 1: missing {text}
            "=== ANCHOR-SYSTEM ===\ny\n\n"
            "=== ANCHOR ===\n{speaker_id}\n{values}\n{text}\n{extra}\n"  # defect 2: {extra}
        )
        (tmp_path / prompts_mod.ANONYMIZATION_PROMPT_FILE).write_text(
            two_anonymization_defects, encoding="utf-8"
        )
        (tmp_path / "extraction.txt").write_text(
            "{transcript} {speaker_context} {document_context} {surprise}", encoding="utf-8"
        )

        with pytest.raises(RuntimeError) as exc_info:
            prompts_mod.ensure_prompt_assets(prompts_dir=tmp_path)
        message = str(exc_info.value)

        anonymization_path = str(tmp_path / prompts_mod.ANONYMIZATION_PROMPT_FILE)
        extraction_path = str(tmp_path / "extraction.txt")
        assert anonymization_path in message
        assert "section SCAN is missing slot {text}" in message
        assert "section ANCHOR carries an unknown slot {extra}" in message
        # Tied to its own file — a generic "unknown slot" substring could
        # otherwise be satisfied by the anonymization home's own unrelated
        # {extra} problem above.
        assert f"{extraction_path}: carries an unknown slot {{surprise}}" in message

    def test_passes_with_operator_prompts_dir_argument(self, tmp_path):
        """An operator ``prompts_dir`` with no local overrides falls
        through to the shipped tree for every slot check — no false
        positive from a directory that legitimately provides nothing."""
        from paramem.graph.prompts import ensure_prompt_assets

        ensure_prompt_assets(prompts_dir=tmp_path)

    def test_passes_against_the_real_shipped_tree(self):
        """The real shipped ``configs/prompts/`` tree — base plus every
        per-model directory (``qwen3-4b/``) — carries every required slot
        on every extraction user template that exists there. Calls with
        ``prompts_dir=None`` — the shipped-tree-only case; the operator-dir
        case is :meth:`test_passes_with_operator_prompts_dir_argument`."""
        from paramem.graph.prompts import ensure_prompt_assets

        ensure_prompt_assets(prompts_dir=None)

    def test_a_problem_under_an_operators_own_prompts_dir_reads_fix_only(self, tmp_path):
        """A defect in a file living under an operator's own ``prompts_dir``
        (no shipped fallback to fall back on) names one remedy — fix that
        file directly — since there is no shipped copy to restore it from."""
        import paramem.graph.prompts as prompts_mod

        (tmp_path / "extraction.txt").write_text(
            "{transcript} {speaker_context} {document_context} {surprise}", encoding="utf-8"
        )
        with pytest.raises(RuntimeError) as exc_info:
            prompts_mod.ensure_prompt_assets(prompts_dir=tmp_path)
        message = str(exc_info.value)
        extraction_path = str(tmp_path / "extraction.txt")
        assert f"{extraction_path}: carries an unknown slot {{surprise}} (fix that file)" in (
            message
        )

    def test_a_problem_in_the_shipped_tree_names_both_remedies(self, monkeypatch, tmp_path):
        """A defect in a file resolving inside the shipped
        ``configs/prompts/`` tree names both remedies — fix that file in
        place, or restore the shipped copy from the repository — because
        an in-place edit of a shipped file is a documented operator
        practice (``DEPLOYMENT.md``'s Prompt Engineering section) and the
        path alone cannot tell that apart from a broken checkout."""
        import paramem.graph.prompts as prompts_mod

        (tmp_path / "extraction.txt").write_text(
            "{transcript} {speaker_context} {document_context} {surprise}", encoding="utf-8"
        )
        for filename in prompts_mod._REQUIRED_PROMPT_FILES:
            if filename != "extraction.txt" and not (tmp_path / filename).exists():
                (tmp_path / filename).write_text("placeholder", encoding="utf-8")
        monkeypatch.setattr(prompts_mod, "_DEFAULT_PROMPT_DIR", tmp_path)
        with pytest.raises(RuntimeError) as exc_info:
            prompts_mod.ensure_prompt_assets()
        message = str(exc_info.value)
        extraction_path = str(tmp_path / "extraction.txt")
        assert (
            f"{extraction_path}: carries an unknown slot {{surprise}} "
            "(fix that file, or restore the shipped copy from the repository)" in message
        )


class TestCheckAnonymizationPromptSections:
    """``check_anonymization_prompt_sections`` — the one section/slot check
    shared by :func:`~paramem.graph.prompts.ensure_prompt_assets` (whichever
    copy the server loads — an operator's configured prompts directory, or
    the shipped copy when none is configured — at boot) and the anonymizer
    gate tool's ``--prompt-file`` validation (an override, at CLI startup).
    """

    def test_passes_on_the_shipped_home(self):
        from paramem.graph.prompts import check_anonymization_prompt_sections

        check_anonymization_prompt_sections()

    def test_doubled_braces_are_not_counted_as_a_slot(self):
        """A JSON-literal ``{{...}}`` fragment, the shape every example in
        the shipped home uses, must not be read as an unknown slot —
        ``string.Formatter().parse`` already treats a doubled brace as
        literal text."""
        from paramem.graph.prompts import (
            ANONYMIZATION_PROMPT_FILE,
            check_anonymization_prompt_sections,
            prompt_overrides,
        )

        with prompt_overrides({ANONYMIZATION_PROMPT_FILE: VALID_ANONYMIZATION_SECTIONS}):
            check_anonymization_prompt_sections()

    def test_raises_on_a_missing_section(self):
        from paramem.graph.prompts import (
            ANONYMIZATION_PROMPT_FILE,
            AnonymizationPromptInvalid,
            check_anonymization_prompt_sections,
            prompt_overrides,
        )

        with prompt_overrides(
            {ANONYMIZATION_PROMPT_FILE: INVALID_ANONYMIZATION_SECTIONS_MISSING_SECTION}
        ):
            with pytest.raises(AnonymizationPromptInvalid) as exc_info:
                check_anonymization_prompt_sections()
        # This body drops BOTH ANCHOR sections — every missing section is
        # reported, not only the first.
        assert exc_info.value.problems == [
            "missing section ANCHOR-SYSTEM",
            "missing section ANCHOR",
        ]

    def test_raises_on_a_section_missing_a_required_slot(self):
        from paramem.graph.prompts import (
            ANONYMIZATION_PROMPT_FILE,
            AnonymizationPromptInvalid,
            check_anonymization_prompt_sections,
            prompt_overrides,
        )

        with prompt_overrides(
            {ANONYMIZATION_PROMPT_FILE: INVALID_ANONYMIZATION_SECTIONS_MISSING_SLOT}
        ):
            with pytest.raises(AnonymizationPromptInvalid) as exc_info:
                check_anonymization_prompt_sections()
        assert str(exc_info.value) == "section SCAN is missing slot {text}"

    def test_raises_on_a_section_carrying_an_unknown_slot(self):
        from paramem.graph.prompts import (
            ANONYMIZATION_PROMPT_FILE,
            AnonymizationPromptInvalid,
            check_anonymization_prompt_sections,
            prompt_overrides,
        )

        with prompt_overrides(
            {ANONYMIZATION_PROMPT_FILE: INVALID_ANONYMIZATION_SECTIONS_UNKNOWN_SLOT}
        ):
            with pytest.raises(AnonymizationPromptInvalid) as exc_info:
                check_anonymization_prompt_sections()
        assert str(exc_info.value) == "section ANCHOR carries an unknown slot {extra}"

    def test_raises_a_plain_message_on_a_malformed_placeholder(self):
        """A lone ``}`` is not a doubled brace and not a named slot — the
        standard library format parser's own ``ValueError`` on it, restated
        in plain words by :func:`~paramem.graph.prompts._template_slots`,
        the one reader every slot check in ``prompts.py`` uses."""
        from paramem.graph.prompts import (
            ANONYMIZATION_PROMPT_FILE,
            AnonymizationPromptInvalid,
            check_anonymization_prompt_sections,
            prompt_overrides,
        )

        with prompt_overrides(
            {ANONYMIZATION_PROMPT_FILE: INVALID_ANONYMIZATION_SECTIONS_MALFORMED_PLACEHOLDER}
        ):
            with pytest.raises(AnonymizationPromptInvalid) as exc_info:
                check_anonymization_prompt_sections()
        message = str(exc_info.value)
        assert message.startswith("section SCAN: malformed placeholder")

    def test_a_doubled_slot_fails_the_required_slot_rule(self):
        """``{{text}}`` renders as the literal text ``{text}``, never a
        slot — a section carrying only the doubled form is reported as
        missing the real ``{text}`` slot, confirming the required-slot
        check reads through :func:`_template_slots`."""
        from paramem.graph.prompts import (
            ANONYMIZATION_PROMPT_FILE,
            AnonymizationPromptInvalid,
            check_anonymization_prompt_sections,
            prompt_overrides,
        )

        with prompt_overrides(
            {ANONYMIZATION_PROMPT_FILE: INVALID_ANONYMIZATION_SECTIONS_DOUBLED_SLOT}
        ):
            with pytest.raises(AnonymizationPromptInvalid) as exc_info:
                check_anonymization_prompt_sections()
        assert str(exc_info.value) == "section SCAN is missing slot {text}"

    def test_raises_every_problem_in_one_exception_across_two_sections(self):
        """A missing section plus a present section's own unknown slot are
        both carried on the one raised exception's ``problems`` list — the
        check never stops at the first problem it finds."""
        from paramem.graph.prompts import (
            ANONYMIZATION_PROMPT_FILE,
            AnonymizationPromptInvalid,
            check_anonymization_prompt_sections,
            prompt_overrides,
        )

        text = (
            "=== SCAN-SYSTEM ===\nx\n\n"
            "=== SCAN ===\n{keywords}\n{text}\n{extra}\n"  # unknown slot
            # ANCHOR-SYSTEM/ANCHOR both absent -> two missing-section problems
        )
        with prompt_overrides({ANONYMIZATION_PROMPT_FILE: text}):
            with pytest.raises(AnonymizationPromptInvalid) as exc_info:
                check_anonymization_prompt_sections()
        assert exc_info.value.problems == [
            "missing section ANCHOR-SYSTEM",
            "missing section ANCHOR",
            "section SCAN carries an unknown slot {extra}",
        ]


class TestTemplateSlots:
    """:func:`~paramem.graph.prompts._template_slots` — the one slot reader
    every check in ``prompts.py`` uses."""

    def test_a_doubled_brace_pair_is_literal_text_not_a_slot(self):
        from paramem.graph.prompts import _template_slots

        assert _template_slots('before {{"mapping": {{}}}} after') == set()

    def test_an_auto_numbered_slot_is_counted(self):
        from paramem.graph.prompts import _template_slots

        assert _template_slots("value: {}") == {"{}"}

    def test_a_named_slot_is_counted(self):
        from paramem.graph.prompts import _template_slots

        assert _template_slots("{text} and {keywords}") == {"{text}", "{keywords}"}

    def test_a_lone_closing_brace_raises_a_plain_value_error(self):
        from paramem.graph.prompts import _template_slots

        with pytest.raises(ValueError) as exc_info:
            _template_slots("stray brace: }")
        assert str(exc_info.value).startswith("malformed placeholder")


class TestExtractionTemplateSlots:
    """The extraction user templates' required-and-allowed slot rule
    (:data:`~paramem.graph.prompts._EXTRACTION_TEMPLATE_SLOTS`), read
    through the shared :func:`~paramem.graph.prompts._template_slots`
    reader inside :func:`~paramem.graph.prompts.ensure_prompt_assets`.
    """

    def _run_with_template(self, tmp_path, filename: str, content: str) -> None:
        """Write only *filename* under *tmp_path* and check it as an
        operator override directory: every other required file, and every
        other extraction template, still resolves from the real shipped
        tree (:func:`~paramem.graph.prompts._load_prompt`'s fall-through),
        so only *filename*'s own deliberately malformed content can fail
        the walk.
        """
        import paramem.graph.prompts as prompts_mod

        (tmp_path / filename).write_text(content, encoding="utf-8")
        prompts_mod.ensure_prompt_assets(prompts_dir=tmp_path)

    def test_extraction_txt_missing_a_required_slot_is_rejected(self, tmp_path):
        extraction_path = str(tmp_path / "extraction.txt")
        with pytest.raises(RuntimeError) as exc_info:
            self._run_with_template(tmp_path, "extraction.txt", "{transcript} {speaker_context}")
        message = str(exc_info.value)
        assert f"{extraction_path}: is missing slot {{document_context}}" in message

    def test_extraction_txt_carrying_an_unlisted_slot_is_rejected(self, tmp_path):
        extraction_path = str(tmp_path / "extraction.txt")
        with pytest.raises(RuntimeError) as exc_info:
            self._run_with_template(
                tmp_path,
                "extraction.txt",
                "{transcript} {speaker_context} {document_context} {surprise}",
            )
        message = str(exc_info.value)
        assert f"{extraction_path}: carries an unknown slot {{surprise}}" in message

    def test_extraction_txt_with_a_malformed_placeholder_is_reported_not_crashed(self, tmp_path):
        """A lone ``}`` in an extraction template is reported as a problem
        of that file — the same plain-words treatment the anonymization
        home gets — never an uncaught parser exception."""
        with pytest.raises(RuntimeError) as exc_info:
            self._run_with_template(
                tmp_path,
                "extraction.txt",
                "{transcript} {speaker_context} {document_context} stray brace: }",
            )
        message = str(exc_info.value)
        assert "extraction.txt" in message
        assert "malformed placeholder" in message

    def test_extraction_second_order_txt_requires_named_people_too(self, tmp_path):
        with pytest.raises(RuntimeError) as exc_info:
            self._run_with_template(
                tmp_path,
                "extraction_second_order.txt",
                "{transcript} {speaker_context} {document_context}",
            )
        message = str(exc_info.value)
        assert "extraction_second_order.txt" in message
        assert "{named_people}" in message
