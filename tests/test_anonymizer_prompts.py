"""``paramem.graph.anonymizer_prompts.load_anonymizer_prompts`` over the
shipped ``configs/prompts/anonymization.txt``: the four sections load, the
SCAN section carries ``{keywords}``/``{text}``, the ANCHOR section carries
``{speaker_id}``/``{values}``/``{text}``, and ``paramem.graph.prompts``'s
own required-sections/slots tables agree with the file. Also:
``paramem.cloud.deanonymize._JSON_ENVELOPE_KEYS`` carries no ``"mapping"``
— that envelope key belongs to the anonymizer's own parser
(``_extract_json_envelope``), never the extractor's.
"""

from __future__ import annotations

from paramem.cloud.anonymize import AnonymizerPrompts
from paramem.cloud.deanonymize import _JSON_ENVELOPE_KEYS
from paramem.graph.anonymizer_prompts import load_anonymizer_prompts
from paramem.graph.prompts import ensure_prompt_assets


class TestLoadAnonymizerPromptsOverTheShippedFile:
    def test_returns_a_fully_composed_anonymizer_prompts(self):
        prompts = load_anonymizer_prompts()
        assert isinstance(prompts, AnonymizerPrompts)
        assert prompts.scan_system
        assert prompts.scan
        assert prompts.anchor_system
        assert prompts.anchor

    def test_the_scan_section_carries_its_two_slots(self):
        prompts = load_anonymizer_prompts()
        assert "{keywords}" in prompts.scan
        assert "{text}" in prompts.scan

    def test_the_anchor_section_carries_its_three_slots(self):
        prompts = load_anonymizer_prompts()
        assert "{speaker_id}" in prompts.anchor
        assert "{values}" in prompts.anchor
        assert "{text}" in prompts.anchor


class TestPromptsRequiredSectionsAndSlotsAgreeWithTheShippedFile:
    def test_ensure_prompt_assets_raises_nothing_over_the_shipped_tree(self):
        ensure_prompt_assets()


class TestJsonEnvelopeKeysCarriesNoMapping:
    def test_mapping_is_not_an_extraction_envelope_key(self):
        """``{"mapping": ...}`` is the anonymizer's own SCAN reply
        envelope, parsed by ``_extract_json_envelope`` — never the
        extractor's ``_extract_json_block`` envelope vocabulary."""
        assert "mapping" not in _JSON_ENVELOPE_KEYS
