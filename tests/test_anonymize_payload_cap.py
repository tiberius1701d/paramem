"""Payload-cap primitives: the two held operating-point constants, the
import-time tripwire, ``anonymize_payload_cap_tokens``'s signature and
raise, ``anchor_output_reserve_tokens``'s plateau, and the two call sites
(the runtime precondition, the cap door) sharing one formula.
"""

from __future__ import annotations

import inspect

import pytest

from paramem.graph.document_chunker import _DOC_MAX_TOKENS
from paramem.server.session_buffer import _TRANSCRIPT_MAX_TOKENS
from paramem.utils import tokens as tokens_module
from paramem.utils.tokens import (
    ANONYMIZE_ANCHOR_MAX_CANDIDATES,
    ANONYMIZE_ANCHOR_PROMPT_SKELETON_TOKENS,
    ANONYMIZE_ENVELOPE_TOKENS,
    TRANSCRIPT_TOKENS_PER_WORD,
    anchor_output_reserve_tokens,
    anonymize_payload_cap_tokens,
)


class TestHeldOperatingPointConstants:
    def test_transcript_max_tokens_is_exactly_4062(self) -> None:
        assert _TRANSCRIPT_MAX_TOKENS == 4062

    def test_doc_max_tokens_is_exactly_7662(self) -> None:
        assert _DOC_MAX_TOKENS == 7662


class TestImportTimeTripwireHoldsAtShippedValues:
    def test_transcript_cap_fits_under_the_anchor_shape_cap(self) -> None:
        cap = anonymize_payload_cap_tokens(
            envelope_tokens=ANONYMIZE_ENVELOPE_TOKENS,
            anchor_skeleton_tokens=ANONYMIZE_ANCHOR_PROMPT_SKELETON_TOKENS,
            anchor_reserve_tokens=anchor_output_reserve_tokens(ANONYMIZE_ANCHOR_MAX_CANDIDATES),
            payload_tokens_per_word=TRANSCRIPT_TOKENS_PER_WORD,
        )
        assert _TRANSCRIPT_MAX_TOKENS <= cap

    def test_doc_cap_fits_under_the_anchor_shape_cap(self) -> None:
        from paramem.graph.document_chunker import _R_PROSE

        cap = anonymize_payload_cap_tokens(
            envelope_tokens=ANONYMIZE_ENVELOPE_TOKENS,
            anchor_skeleton_tokens=ANONYMIZE_ANCHOR_PROMPT_SKELETON_TOKENS,
            anchor_reserve_tokens=anchor_output_reserve_tokens(ANONYMIZE_ANCHOR_MAX_CANDIDATES),
            payload_tokens_per_word=_R_PROSE,
        )
        assert _DOC_MAX_TOKENS <= cap

    def test_tripwire_fails_under_a_lowered_envelope(self) -> None:
        # Exercise the tripwire's own function directly with a lowered
        # envelope, rather than reloading modules — the design's own
        # prescribed check.
        reserve = anchor_output_reserve_tokens(ANONYMIZE_ANCHOR_MAX_CANDIDATES)
        lowered_envelope = ANONYMIZE_ANCHOR_PROMPT_SKELETON_TOKENS + reserve + 10
        cap = anonymize_payload_cap_tokens(
            envelope_tokens=lowered_envelope,
            anchor_skeleton_tokens=ANONYMIZE_ANCHOR_PROMPT_SKELETON_TOKENS,
            anchor_reserve_tokens=reserve,
            payload_tokens_per_word=TRANSCRIPT_TOKENS_PER_WORD,
        )
        assert cap < _TRANSCRIPT_MAX_TOKENS


class TestAnonymizePayloadCapTokensSignatureAndRaise:
    def test_signature_takes_the_five_documented_parameters(self) -> None:
        params = inspect.signature(anonymize_payload_cap_tokens).parameters
        assert set(params) == {
            "envelope_tokens",
            "anchor_skeleton_tokens",
            "anchor_reserve_tokens",
            "payload_tokens_per_word",
            "tokens_per_word",
        }

    def test_raises_when_skeleton_plus_reserve_leaves_no_payload_budget(self) -> None:
        with pytest.raises(ValueError):
            anonymize_payload_cap_tokens(
                envelope_tokens=100,
                anchor_skeleton_tokens=60,
                anchor_reserve_tokens=60,  # 100 - 60 - 60 < 0
                payload_tokens_per_word=1.5,
            )

    def test_raises_when_available_budget_is_exactly_zero(self) -> None:
        with pytest.raises(ValueError):
            anonymize_payload_cap_tokens(
                envelope_tokens=100,
                anchor_skeleton_tokens=60,
                anchor_reserve_tokens=40,  # 100 - 60 - 40 == 0, not > 0
                payload_tokens_per_word=1.5,
            )


class TestAnchorOutputReserveTokensPlateaus:
    def test_reserve_stops_growing_past_the_max_candidates_ceiling(self) -> None:
        at_ceiling = anchor_output_reserve_tokens(ANONYMIZE_ANCHOR_MAX_CANDIDATES)
        past_ceiling = anchor_output_reserve_tokens(ANONYMIZE_ANCHOR_MAX_CANDIDATES + 500)
        assert at_ceiling == past_ceiling

    def test_reserve_grows_with_candidate_count_below_the_ceiling(self) -> None:
        small = anchor_output_reserve_tokens(1)
        larger = anchor_output_reserve_tokens(10)
        assert larger > small


class TestRuntimePreconditionAndCapDoorShareOneFormula:
    def test_the_runtime_precondition_import_binds_the_identical_function_object(self) -> None:
        # anonymize_steps.py imports anchor_output_reserve_tokens by name
        # ("from paramem.utils.tokens import anchor_output_reserve_tokens")
        # — the runtime precondition (ask_speaker_anchor) and the cap door
        # (document_chunker.py / session_buffer.py's own import-time
        # assertions, both exercised above) therefore call literally the
        # SAME function object, never two independently re-implemented
        # copies of the formula.
        from paramem.cloud import anonymize_steps

        assert anonymize_steps.anchor_output_reserve_tokens is (
            tokens_module.anchor_output_reserve_tokens
        )

    def test_the_runtime_precondition_calls_it_with_len_values(self, monkeypatch) -> None:
        calls: list[int] = []
        real = tokens_module.anchor_output_reserve_tokens

        def _recording(candidate_count: int) -> int:
            calls.append(candidate_count)
            return real(candidate_count)

        from unittest.mock import MagicMock

        from paramem.cloud import anonymize_steps

        monkeypatch.setattr(anonymize_steps, "anchor_output_reserve_tokens", _recording)

        anonymize_steps.ask_speaker_anchor(
            "text",
            MagicMock(),
            MagicMock(),
            values=["Alex", "Sam"],
            speaker_id="speaker1",
            section="speaker={speaker_id} values={values} text={text}",
            system_prompt="system",
            token_envelope=1,  # far too small -> AnonymizeBudgetRefused, no GPU call
        )

        assert calls == [2]

    def test_the_cap_door_calls_it_at_the_structural_candidate_ceiling(self) -> None:
        # document_chunker.py / session_buffer.py's own import-time
        # assertions call anchor_output_reserve_tokens(ANONYMIZE_ANCHOR_MAX_CANDIDATES)
        # — reproduced here against the live function (not a re-import) to
        # pin the call shape without reloading modules.
        reserve = tokens_module.anchor_output_reserve_tokens(ANONYMIZE_ANCHOR_MAX_CANDIDATES)
        cap = anonymize_payload_cap_tokens(
            envelope_tokens=ANONYMIZE_ENVELOPE_TOKENS,
            anchor_skeleton_tokens=ANONYMIZE_ANCHOR_PROMPT_SKELETON_TOKENS,
            anchor_reserve_tokens=reserve,
            payload_tokens_per_word=TRANSCRIPT_TOKENS_PER_WORD,
        )
        assert cap > 0
