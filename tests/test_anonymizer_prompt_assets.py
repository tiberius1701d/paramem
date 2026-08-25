"""``ensure_prompt_assets`` raise cases for the sectioned local-anonymizer
home, and ``load_anonymizer_prompts``'s composed-return / no-``scrub``
signature / ``prompts_dir`` threading.
"""

from __future__ import annotations

import inspect

import pytest

from paramem.graph.anonymizer_prompts import load_anonymizer_prompts
from paramem.graph.prompts import ensure_prompt_assets


def _write_anonymization(tmp_path, body: str):
    (tmp_path / "anonymization.txt").write_text(body)
    return tmp_path


class TestEnsurePromptAssetsRaisesOnMissingSections:
    def test_missing_anchor_raises(self, tmp_path) -> None:
        home = _write_anonymization(
            tmp_path,
            "=== ANCHOR-SYSTEM ===\nDecide who introduced themselves.\n",
        )
        with pytest.raises(RuntimeError, match="ANCHOR"):
            ensure_prompt_assets(prompts_dir=home)

    def test_missing_anchor_system_raises(self, tmp_path) -> None:
        home = _write_anonymization(
            tmp_path,
            "=== ANCHOR ===\nspeaker={speaker_id} values={values} text={text}\n",
        )
        with pytest.raises(RuntimeError, match="ANCHOR-SYSTEM"):
            ensure_prompt_assets(prompts_dir=home)


class TestEnsurePromptAssetsRaisesOnMissingSlots:
    def test_anchor_missing_speaker_id_slot_raises(self, tmp_path) -> None:
        home = _write_anonymization(
            tmp_path,
            "=== ANCHOR-SYSTEM ===\nDecide.\n\n=== ANCHOR ===\nvalues={values} text={text}\n",
        )
        with pytest.raises(RuntimeError, match=r"\{speaker_id\}"):
            ensure_prompt_assets(prompts_dir=home)

    def test_anchor_missing_values_slot_raises(self, tmp_path) -> None:
        home = _write_anonymization(
            tmp_path,
            "=== ANCHOR-SYSTEM ===\nDecide.\n\n=== ANCHOR ===\nspeaker={speaker_id} text={text}\n",
        )
        with pytest.raises(RuntimeError, match=r"\{values\}"):
            ensure_prompt_assets(prompts_dir=home)

    def test_anchor_missing_text_slot_raises(self, tmp_path) -> None:
        home = _write_anonymization(
            tmp_path,
            "=== ANCHOR-SYSTEM ===\nDecide.\n\n"
            "=== ANCHOR ===\nspeaker={speaker_id} values={values}\n",
        )
        with pytest.raises(RuntimeError, match=r"\{text\}"):
            ensure_prompt_assets(prompts_dir=home)


class TestLoadAnonymizerPromptsReturnsBothSections:
    def test_returns_the_composed_anchor_system_and_anchor_sections(self) -> None:
        prompts = load_anonymizer_prompts()
        assert prompts.anchor_system
        assert prompts.anchor
        assert "{speaker_id}" in prompts.anchor
        assert "{values}" in prompts.anchor
        assert "{text}" in prompts.anchor

    def test_threads_a_prompts_dir_override(self, tmp_path) -> None:
        home = _write_anonymization(
            tmp_path,
            "=== ANCHOR-SYSTEM ===\nCustom system text.\n\n"
            "=== ANCHOR ===\nspeaker={speaker_id} values={values} text={text}\n",
        )
        prompts = load_anonymizer_prompts(prompts_dir=home)
        assert prompts.anchor_system == "Custom system text."


class TestLoadAnonymizerPromptsTakesNoScrubArgument:
    def test_signature_has_no_scrub_parameter(self) -> None:
        params = inspect.signature(load_anonymizer_prompts).parameters
        assert "scrub" not in params
        assert set(params) == {"prompts_dir"}
