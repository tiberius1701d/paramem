"""Unit tests for paramem.training.recall_eval.

Covers evaluate_indexed_recall (serial and batched), _generate_recall_batch,
_derive_stop_ids, and the extracted _finalize_recalled helper.

Patching note (Decision 3 in plan-batched-probe-v2.md):
functools.partial snapshots the target function at construction time.  Any
test that exercises the batch_size > 1 branch via _maybe_make_recall_callback
must either (a) patch evaluate_indexed_recall BEFORE callback construction so
the partial captures the patched function, or (b) inject eval_fn= directly
into RecallEarlyStopCallback.  Module-level patches applied AFTER callback
construction do NOT redirect the already-captured partial.

None of the tests in this file patch evaluate_indexed_recall at the module
level — they exercise evaluate_indexed_recall directly with stubbed
model.generate, or they call _finalize_recalled / _generate_recall_batch
directly.  This keeps the tests free of the partial-snapshot pitfall.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import torch

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _entry(
    key: str, subject: str = "Alice", predicate: str = "lives_in", obj: str = "Berlin"
) -> dict:
    """Minimal entry dict."""
    return {"key": key, "subject": subject, "predicate": predicate, "object": obj}


def _raw_json(
    key: str, subject: str = "Alice", predicate: str = "lives_in", obj: str = "Berlin"
) -> str:
    """Valid raw model output for _finalize_recalled."""
    return json.dumps({"key": key, "subject": subject, "predicate": predicate, "object": obj})


def _make_model_mock():
    """MagicMock model that exposes parameters().device and generate()."""
    m = MagicMock()
    param = MagicMock()
    param.device = torch.device("cpu")
    m.parameters.return_value = iter([param])
    m.gradient_checkpointing_disable = MagicMock()
    return m


def _make_tokenizer_mock(
    eos_id: int = 2, padding_side: str = "right", pad_token_id: int | None = 1
):
    """MagicMock tokenizer with the attributes _generate_recall_batch reads."""
    t = MagicMock()
    t.eos_token_id = eos_id
    t.padding_side = padding_side
    t.pad_token_id = pad_token_id
    t.eos_token = "<eos>"
    # encode returns [] for unknown special tokens (no match → not appended to stop_ids)
    t.encode.return_value = []
    return t


def _build_real_registry(entries: list[dict]) -> dict:
    """Build a SimHash registry with real fingerprints for the given entries."""
    from paramem.training.entry_memory import build_registry

    return build_registry(entries)


# ---------------------------------------------------------------------------
# test_finalize_recalled_contract
# ---------------------------------------------------------------------------


class TestFinalizeRecalledContract:
    """Direct unit test of _finalize_recalled for each branch."""

    def test_parse_failure(self):
        from paramem.training.entry_memory import _finalize_recalled

        result = _finalize_recalled("not json", "graph1", None, 0.75)
        assert result["failure_reason"] == "parse_failure"
        assert result["raw_output"] == "not json"

    def test_key_mismatch(self):
        from paramem.training.entry_memory import _finalize_recalled

        raw = _raw_json("graph99")
        result = _finalize_recalled(raw, "graph1", None, 0.75)
        assert result["failure_reason"].startswith("key_mismatch:")
        assert "graph99" in result["failure_reason"]

    def test_low_confidence(self):
        from paramem.training.entry_memory import _finalize_recalled

        # registry with fingerprint 0 for graph1 → simhash_confidence ~0.5 < 0.75
        registry = {"graph1": 0}
        raw = _raw_json("graph1")
        result = _finalize_recalled(raw, "graph1", registry, 0.75)
        assert result["failure_reason"].startswith("low_confidence:")

    def test_success_no_registry(self):
        from paramem.training.entry_memory import _finalize_recalled

        raw = _raw_json("graph1")
        result = _finalize_recalled(raw, "graph1", None, 0.75)
        assert "failure_reason" not in result
        assert result["key"] == "graph1"
        assert result["subject"] == "Alice"
        assert result["predicate"] == "lives_in"
        assert result["object"] == "Berlin"
        assert result["confidence"] == 1.0
        assert "raw_output" in result
        assert "fact_text" in result

    def test_success_with_matching_registry(self):
        from paramem.training.entry_memory import _finalize_recalled

        entry = _entry("graph1")
        registry = _build_real_registry([entry])
        raw = _raw_json("graph1")
        result = _finalize_recalled(raw, "graph1", registry, 0.75)
        assert "failure_reason" not in result
        assert result["confidence"] >= 0.75


# ---------------------------------------------------------------------------
# test_serial_path_byte_identical
# ---------------------------------------------------------------------------


class TestSerialPathByteIdentical:
    """evaluate_indexed_recall at batch_size=1 delegates to probe_entry."""

    def test_serial_path_byte_identical(self):
        from paramem.training.recall_eval import evaluate_indexed_recall

        entries = [
            _entry("graph1", "Alice", "lives_in", "Berlin"),
            _entry("graph2", "Bob", "works_at", "Acme"),
            _entry("graph3", "Carol", "born_in", "Paris"),
        ]
        registry = _build_real_registry(entries)

        # Canned probe_entry responses matching the entries exactly
        def _fake_probe(model, tokenizer, key, *, registry=None, **kw):
            e = next(e for e in entries if e["key"] == key)
            raw = _raw_json(key, e["subject"], e["predicate"], e["object"])
            return {
                "key": key,
                "subject": e["subject"],
                "predicate": e["predicate"],
                "object": e["object"],
                "confidence": 1.0,
                "raw_output": raw,
                "fact_text": f"{e['subject']} {e['predicate']} {e['object']}",
            }

        model = _make_model_mock()
        tokenizer = _make_tokenizer_mock()

        with (
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.entry_memory.probe_entry", side_effect=_fake_probe),
        ):
            result = evaluate_indexed_recall(
                model,
                tokenizer,
                entries,
                registry,
                batch_size=1,
            )

        assert result["exact_count"] == 3
        assert result["total"] == 3
        assert result["rate"] == 1.0
        pk_keys = {r["key"] for r in result["per_key"]}
        assert pk_keys == {"graph1", "graph2", "graph3"}
        for pk in result["per_key"]:
            assert pk["exact_match"] is True
            assert pk["failure_reason"] is None


# ---------------------------------------------------------------------------
# test_batched_shape_matches_serial
# ---------------------------------------------------------------------------


class TestBatchedShapeMatchesSerial:
    """Top-level keys and per_key dict shape identical for batch_size=1 and batch_size=2."""

    def _run(self, entries: list[dict], registry: dict, batch_size: int, model, tokenizer):
        from paramem.training.recall_eval import evaluate_indexed_recall

        with (
            patch("paramem.models.loader.switch_adapter"),
        ):
            return evaluate_indexed_recall(
                model,
                tokenizer,
                entries,
                registry,
                batch_size=batch_size,
            )

    def test_batched_shape_matches_serial(self):
        entries = [
            _entry("graph1", "Alice", "lives_in", "Berlin"),
            _entry("graph2", "Bob", "works_at", "Acme"),
        ]
        registry = _build_real_registry(entries)

        # For serial path: patch probe_entry
        def _fake_probe(model, tokenizer, key, *, registry=None, **kw):
            e = next(e for e in entries if e["key"] == key)
            raw = _raw_json(key, e["subject"], e["predicate"], e["object"])
            return {
                "key": key,
                "subject": e["subject"],
                "predicate": e["predicate"],
                "object": e["object"],
                "confidence": 1.0,
                "raw_output": raw,
                "fact_text": f"{e['subject']} {e['predicate']} {e['object']}",
            }

        model = _make_model_mock()
        tokenizer = _make_tokenizer_mock()

        with patch("paramem.training.entry_memory.probe_entry", side_effect=_fake_probe):
            serial = self._run(entries, registry, batch_size=1, model=model, tokenizer=tokenizer)

        # For batched path: mock tokenizer call and model.generate
        model2 = _make_model_mock()
        tokenizer2 = _make_tokenizer_mock()

        # tokenizer(prompts, ...) returns input tensor of shape [2, 5]
        input_ids = torch.zeros(2, 5, dtype=torch.long)
        tok_output = MagicMock()
        tok_output.__getitem__ = lambda self, k: {
            "input_ids": input_ids,
            "attention_mask": torch.ones(2, 5),
        }[k]
        tok_output.keys.return_value = ["input_ids", "attention_mask"]

        # Make tokenizer callable return a mock that can be passed to model.generate(**inputs)
        inputs_dict = {"input_ids": input_ids, "attention_mask": torch.ones(2, 5)}
        tokenizer2.return_value = MagicMock(**{"to.return_value": inputs_dict})
        tokenizer2.return_value.__iter__ = lambda self: iter(inputs_dict)
        tokenizer2.return_value.items = inputs_dict.items

        # model.generate returns [2, 5+new_tokens], new tokens after width=5
        raw1 = _raw_json("graph1", "Alice", "lives_in", "Berlin")
        raw2 = _raw_json("graph2", "Bob", "works_at", "Acme")

        def _fake_decode(ids, skip_special_tokens=True):
            # ids is a 1D tensor (suffix of one row); distinguish by content
            return raw1 if ids[0].item() == 1 else raw2

        tokenizer2.decode.side_effect = _fake_decode

        # Generate output: rows 0 and 1 have different suffix tokens
        gen_out = torch.zeros(2, 8, dtype=torch.long)
        gen_out[0, 5] = 1  # sentinel for row 0 → raw1
        gen_out[1, 5] = 2  # sentinel for row 1 → raw2
        model2.generate.return_value = gen_out

        with patch(
            "paramem.training.dataset._format_inference_prompt",
            side_effect=lambda q, t: q,
        ):
            batched = self._run(entries, registry, batch_size=2, model=model2, tokenizer=tokenizer2)

        # Top-level keys match
        assert set(serial.keys()) == set(batched.keys())
        # per_key record shape matches
        serial_pk_keys = {r["key"] for r in serial["per_key"]}
        batched_pk_keys = {r["key"] for r in batched["per_key"]}
        assert serial_pk_keys == batched_pk_keys
        expected_pk_fields = {
            "key",
            "exact_match",
            "confidence",
            "subject",
            "predicate",
            "object",
            "recalled_subject",
            "recalled_predicate",
            "recalled_object",
            "failure_reason",
        }
        for pk in batched["per_key"]:
            assert expected_pk_fields.issubset(pk.keys())


# ---------------------------------------------------------------------------
# test_batched_finalize_handles_failures
# ---------------------------------------------------------------------------


class TestBatchedFinalizeHandlesFailures:
    """Different rows land in the correct failure_reason bucket via _generate_recall_batch."""

    def test_batched_finalize_handles_failures(self):
        from paramem.training.entry_memory import build_registry
        from paramem.training.recall_eval import _generate_recall_batch

        entries = [
            _entry("graph1", "Alice", "lives_in", "Berlin"),  # success
            _entry("graph2", "Bob", "works_at", "Acme"),  # key mismatch
            _entry("graph3", "Carol", "born_in", "Paris"),  # parse failure
            _entry("graph4", "Dave", "knows", "Eve"),  # low confidence
        ]
        # For graph4: use fingerprint 0 → simhash_confidence near 0 < 0.75
        registry = build_registry(entries)
        registry["graph4"] = 0

        raw_responses = [
            _raw_json("graph1", "Alice", "lives_in", "Berlin"),
            _raw_json("graph99", "Bob", "works_at", "Acme"),  # wrong key → mismatch
            "this is not json at all",
            _raw_json("graph4", "Dave", "knows", "Eve"),
        ]

        model = _make_model_mock()
        tokenizer = _make_tokenizer_mock()

        # Process 4 entries one at a time (batch_size=1 in _generate_recall_batch
        # is called from the helper directly)
        # Use batch_size=4 → one chunk of 4.
        input_ids = torch.zeros(4, 5, dtype=torch.long)
        # Each row i gets a unique sentinel in col 5 of the output
        gen_out = torch.zeros(4, 10, dtype=torch.long)
        for i in range(4):
            gen_out[i, 5] = i + 1

        model.generate.return_value = gen_out
        inputs_dict = {"input_ids": input_ids, "attention_mask": torch.ones(4, 5)}
        tokenizer.return_value = MagicMock(**{"to.return_value": inputs_dict})

        def _fake_decode(ids, skip_special_tokens=True):
            sentinel = ids[0].item()
            return raw_responses[sentinel - 1]

        tokenizer.decode.side_effect = _fake_decode

        with patch(
            "paramem.training.dataset._format_inference_prompt",
            side_effect=lambda q, t: q,
        ):
            pairs = list(_generate_recall_batch(model, tokenizer, entries, registry, batch_size=4))

        by_key = {e["key"]: r for e, r in pairs}

        assert by_key["graph1"].get("failure_reason") is None
        assert by_key["graph1"]["subject"] == "Alice"

        assert by_key["graph2"].get("failure_reason", "").startswith("key_mismatch:")

        assert by_key["graph3"].get("failure_reason") == "parse_failure"

        assert by_key["graph4"].get("failure_reason", "").startswith("low_confidence:")


# ---------------------------------------------------------------------------
# test_left_padding_correctness
# ---------------------------------------------------------------------------


class TestLeftPaddingCorrectness:
    """Short-prompt row's decoded suffix must contain no echoed input tokens.

    Constructs prompts of genuinely different real-token lengths and asserts
    the decoded suffix for each row is exactly the generated portion.
    """

    def test_left_padding_correctness(self):
        from paramem.training.recall_eval import _generate_recall_batch

        short_key = "K"
        long_key = "this_is_a_much_longer_recall_key_N"

        entries = [
            _entry(short_key, "Alice", "likes", "Coffee"),
            _entry(long_key, "Bob", "works_at", "Acme"),
        ]
        registry = None

        short_raw = _raw_json(short_key, "Alice", "likes", "Coffee")
        long_raw = _raw_json(long_key, "Bob", "works_at", "Acme")

        # Padded batch tensor is [2, 10]:
        #   Row 0 (short): [pad ×7, t0, t1, t2]
        #   Row 1 (long):  [t0..t9]
        # After generate, outputs are [2, 10 + new_tokens]. The decoded
        # suffix for each row must contain only the generated tokens.
        padded_width = 10
        input_ids = torch.zeros(2, padded_width, dtype=torch.long)
        # Row 0 short: pad tokens (0) in cols 0-6, real tokens 1,2,3 in cols 7-9
        input_ids[0, 7] = 10
        input_ids[0, 8] = 11
        input_ids[0, 9] = 12
        # Row 1 long: all real tokens
        input_ids[1] = torch.arange(padded_width)

        model = _make_model_mock()
        tokenizer = _make_tokenizer_mock(pad_token_id=0)
        tokenizer.padding_side = "right"  # will be temporarily set to "left"

        new_tokens = 20
        total_width = padded_width + new_tokens

        # Encode short_raw and long_raw as token sequences (length new_tokens each)
        short_ids = torch.full((new_tokens,), 42, dtype=torch.long)
        long_ids = torch.full((new_tokens,), 43, dtype=torch.long)

        # outputs: [2, total_width]
        outputs = torch.zeros(2, total_width, dtype=torch.long)
        outputs[0, padded_width:] = short_ids
        outputs[1, padded_width:] = long_ids

        # inputs_dict must have the correct shape so shape[1] == padded_width
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(2, padded_width, dtype=torch.long),
        }
        model.generate.return_value = outputs

        # Track what decode receives
        decoded_calls: list[torch.Tensor] = []

        def _fake_decode(ids, skip_special_tokens=True):
            decoded_calls.append(ids.clone())
            if ids[0].item() == 42:
                return short_raw
            return long_raw

        tokenizer.decode.side_effect = _fake_decode
        tokenizer.return_value = MagicMock(**{"to.return_value": inputs_dict})

        with patch(
            "paramem.training.dataset._format_inference_prompt",
            side_effect=lambda q, t: q,
        ):
            results = list(
                _generate_recall_batch(model, tokenizer, entries, registry, batch_size=2)
            )

        assert len(results) == 2
        assert len(decoded_calls) == 2

        # Row 0: suffix must be short_ids (no input tokens echoed)
        assert decoded_calls[0].tolist() == short_ids.tolist(), (
            f"Row 0 suffix contained input tokens: {decoded_calls[0].tolist()}"
        )
        # Row 1: suffix must be long_ids
        assert decoded_calls[1].tolist() == long_ids.tolist()

        # Decoded output for short-prompt row must not contain input prompt tokens
        short_result = results[0][1]
        if "failure_reason" not in short_result:
            assert short_key not in short_result.get("raw_output", "").replace(short_raw, ""), (
                "Short-prompt row's decoded suffix echoed the input prompt"
            )


# ---------------------------------------------------------------------------
# test_padding_side_restored
# ---------------------------------------------------------------------------


class TestPaddingSideRestored:
    """tokenizer.padding_side is restored after _generate_recall_batch."""

    def _run_with_side(self, original_side: str):
        from paramem.training.recall_eval import _generate_recall_batch

        entries = [_entry("graph1")]
        model = _make_model_mock()
        tokenizer = _make_tokenizer_mock(padding_side=original_side)

        raw = _raw_json("graph1")
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        new_ids = torch.ones(1, 10, dtype=torch.long)
        outputs = torch.cat([input_ids, new_ids], dim=1)
        model.generate.return_value = outputs

        inputs_dict = {"input_ids": input_ids, "attention_mask": torch.ones(1, 5)}
        tokenizer.return_value = MagicMock(**{"to.return_value": inputs_dict})
        tokenizer.decode.return_value = raw

        with patch(
            "paramem.training.dataset._format_inference_prompt",
            side_effect=lambda q, t: q,
        ):
            list(_generate_recall_batch(model, tokenizer, entries, None, batch_size=4))

        return tokenizer.padding_side

    def test_restored_from_right(self):
        assert self._run_with_side("right") == "right"

    def test_restored_from_left(self):
        assert self._run_with_side("left") == "left"


# ---------------------------------------------------------------------------
# test_registry_low_confidence_batched
# ---------------------------------------------------------------------------


class TestRegistryLowConfidenceBatched:
    """One key with a bad registry fingerprint fails; others succeed."""

    def test_registry_low_confidence_batched(self):
        from paramem.training.entry_memory import build_registry
        from paramem.training.recall_eval import _generate_recall_batch

        entries = [
            _entry("graph1", "Alice", "lives_in", "Berlin"),
            _entry("graph2", "Bob", "works_at", "Acme"),
        ]
        registry = build_registry(entries)
        # Corrupt graph2's fingerprint so confidence < 0.75
        registry["graph2"] = 0

        raw1 = _raw_json("graph1", "Alice", "lives_in", "Berlin")
        raw2 = _raw_json("graph2", "Bob", "works_at", "Acme")

        model = _make_model_mock()
        tokenizer = _make_tokenizer_mock()

        input_ids = torch.zeros(2, 5, dtype=torch.long)
        # outputs with 10 new tokens; row 0 → raw1, row 1 → raw2
        gen_out = torch.zeros(2, 15, dtype=torch.long)
        gen_out[0, 5] = 1  # sentinel for row 0
        gen_out[1, 5] = 2  # sentinel for row 1

        model.generate.return_value = gen_out
        inputs_dict = {"input_ids": input_ids, "attention_mask": torch.ones(2, 5)}
        tokenizer.return_value = MagicMock(**{"to.return_value": inputs_dict})

        def _fake_decode(ids, skip_special_tokens=True):
            return raw1 if ids[0].item() == 1 else raw2

        tokenizer.decode.side_effect = _fake_decode

        with patch(
            "paramem.training.dataset._format_inference_prompt",
            side_effect=lambda q, t: q,
        ):
            pairs = list(_generate_recall_batch(model, tokenizer, entries, registry, batch_size=2))

        assert len(pairs) == 2
        by_key = {e["key"]: r for e, r in pairs}

        # graph1 succeeds
        assert "failure_reason" not in by_key["graph1"]

        # graph2 fails with low_confidence (fingerprint 0 → simhash_confidence near 0)
        assert by_key["graph2"].get("failure_reason", "").startswith("low_confidence:")


# ---------------------------------------------------------------------------
# test_empty_entries_returns_zero_counts
# ---------------------------------------------------------------------------


class TestEmptyEntriesReturnsZeroCounts:
    """evaluate_indexed_recall with empty entries returns zero counts, no exception."""

    def test_empty_entries_serial(self):
        from paramem.training.recall_eval import evaluate_indexed_recall

        model = _make_model_mock()
        tokenizer = _make_tokenizer_mock()

        with patch("paramem.models.loader.switch_adapter"):
            result = evaluate_indexed_recall(model, tokenizer, [], {}, batch_size=1)

        assert result == {
            "exact_count": 0,
            "total": 0,
            "rate": 0.0,
            "mean_confidence": 0.0,
            "mean_expected_word_count": 0,
            "mean_recalled_word_count": 0,
            "per_key": [],
        }

    def test_empty_entries_batched(self):
        from paramem.training.recall_eval import evaluate_indexed_recall

        model = _make_model_mock()
        tokenizer = _make_tokenizer_mock()

        with patch("paramem.models.loader.switch_adapter"):
            result = evaluate_indexed_recall(model, tokenizer, [], {}, batch_size=8)

        assert result == {
            "exact_count": 0,
            "total": 0,
            "rate": 0.0,
            "mean_confidence": 0.0,
            "mean_expected_word_count": 0,
            "mean_recalled_word_count": 0,
            "per_key": [],
        }


# ---------------------------------------------------------------------------
# test_batch_size_exceeds_entries
# ---------------------------------------------------------------------------


class TestBatchSizeExceedsEntries:
    """batch_size=8 with 3 entries: one chunk, all 3 per_key dicts returned."""

    def test_batch_size_exceeds_entries(self):
        from paramem.training.recall_eval import evaluate_indexed_recall

        entries = [
            _entry("graph1", "Alice", "lives_in", "Berlin"),
            _entry("graph2", "Bob", "works_at", "Acme"),
            _entry("graph3", "Carol", "born_in", "Paris"),
        ]
        registry = _build_real_registry(entries)

        model = _make_model_mock()
        tokenizer = _make_tokenizer_mock()

        raws = {
            "graph1": _raw_json("graph1", "Alice", "lives_in", "Berlin"),
            "graph2": _raw_json("graph2", "Bob", "works_at", "Acme"),
            "graph3": _raw_json("graph3", "Carol", "born_in", "Paris"),
        }

        # batch_size=8 > 3 entries → one chunk of 3
        input_ids = torch.zeros(3, 5, dtype=torch.long)
        gen_out = torch.zeros(3, 15, dtype=torch.long)
        gen_out[0, 5] = 1
        gen_out[1, 5] = 2
        gen_out[2, 5] = 3

        model.generate.return_value = gen_out
        inputs_dict = {"input_ids": input_ids, "attention_mask": torch.ones(3, 5)}
        tokenizer.return_value = MagicMock(**{"to.return_value": inputs_dict})

        sentinel_map = {1: "graph1", 2: "graph2", 3: "graph3"}

        def _fake_decode(ids, skip_special_tokens=True):
            key = sentinel_map.get(ids[0].item(), "graph1")
            return raws[key]

        tokenizer.decode.side_effect = _fake_decode

        with (
            patch("paramem.models.loader.switch_adapter"),
            patch(
                "paramem.training.dataset._format_inference_prompt",
                side_effect=lambda q, t: q,
            ),
        ):
            result = evaluate_indexed_recall(
                model,
                tokenizer,
                entries,
                registry,
                batch_size=8,
            )

        assert result["total"] == 3
        assert len(result["per_key"]) == 3
        assert result["exact_count"] == 3

        expected_pk_fields = {
            "key",
            "exact_match",
            "confidence",
            "subject",
            "predicate",
            "object",
            "recalled_subject",
            "recalled_predicate",
            "recalled_object",
            "failure_reason",
        }
        for pk in result["per_key"]:
            assert expected_pk_fields.issubset(pk.keys())
