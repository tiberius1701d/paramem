"""Tests for the extraction pipeline — STT correction, HA validation, noise filter, JSON parsing."""

import json
from unittest.mock import MagicMock, patch

import pytest

from paramem.graph.extractor import (
    _correct_entity_names,
    _extract_json_block,
    _find_correction,
    _levenshtein,
    _validate_with_ha_context,
)
from paramem.graph.schema import Entity, Relation, SessionGraph


def _make_graph(relations, entities=None):
    """Helper to create a SessionGraph with relations."""
    if entities is None:
        names = set()
        for r in relations:
            names.add(r[0])
            names.add(r[2])
        entities = [Entity(name=n, entity_type="concept") for n in names]
    rels = [
        Relation(
            subject=r[0],
            predicate=r[1],
            object=r[2],
            relation_type=r[3] if len(r) > 3 else "factual",
            confidence=r[4] if len(r) > 4 else 1.0,
        )
        for r in relations
    ]
    return SessionGraph(
        session_id="test",
        timestamp="2026-04-09T00:00:00Z",
        entities=entities,
        relations=rels,
    )


# --- Levenshtein ---


class TestLevenshtein:
    def test_identical(self):
        assert _levenshtein("hello", "hello") == 0

    def test_one_char_diff(self):
        assert _levenshtein("hello", "hallo") == 1

    def test_insertion(self):
        assert _levenshtein("hello", "helloo") == 1

    def test_deletion(self):
        assert _levenshtein("hello", "helo") == 1

    def test_one_substitution(self):
        assert _levenshtein("dinslaker", "dinslaken") == 1

    def test_two_changes(self):
        assert _levenshtein("dinslager", "dinslaken") == 2

    def test_empty(self):
        assert _levenshtein("", "abc") == 3
        assert _levenshtein("abc", "") == 3

    def test_completely_different(self):
        assert _levenshtein("abc", "xyz") == 3


# --- STT Correction ---


class TestFindCorrection:
    def test_close_match(self):
        assert _find_correction("Frankford", {"Frankfurt", "Frankfurt"}) == "Frankfurt"

    def test_exact_match_no_correction(self):
        assert _find_correction("Frankfurt", {"Frankfurt", "Berlin"}) is None

    def test_no_match(self):
        assert _find_correction("Tokyo", {"Frankfurt", "Berlin"}) is None

    def test_too_far(self):
        assert _find_correction("Paris", {"Frankfurt"}) is None

    def test_case_insensitive(self):
        assert _find_correction("millfeld", {"Millfield"}) == "Millfield"


class TestCorrectEntityNames:
    def test_corrects_from_assistant_response(self):
        graph = _make_graph([("Alex", "parents_live_in", "Frankford")])
        transcript = "[user] My parents live in Frankford.\n[assistant] Frankfurt is about 300km."
        result = _correct_entity_names(graph, transcript)
        assert result.relations[0].object == "Frankfurt"

    def test_no_correction_when_exact(self):
        graph = _make_graph([("Alex", "lives_in", "Frankfurt")])
        transcript = "[user] I live in Frankfurt.\n[assistant] Frankfurt is nice."
        result = _correct_entity_names(graph, transcript)
        assert result.relations[0].object == "Frankfurt"

    def test_no_assistant_response(self):
        graph = _make_graph([("Alex", "lives_in", "Kelkham")])
        transcript = "[user] I live in Kelkham."
        result = _correct_entity_names(graph, transcript)
        # No assistant tokens to correct from
        assert result.relations[0].object == "Kelkham"

    def test_short_words_skipped(self):
        graph = _make_graph([("Alex", "likes", "Tea")])
        transcript = "[user] I like tea.\n[assistant] Tea is great."
        result = _correct_entity_names(graph, transcript)
        # "Tea" is < 4 chars, skipped in assistant token extraction
        assert result.relations[0].object == "Tea"


# --- HA Context Validation ---


class TestHAContextValidation:
    def test_home_location_boosts_confidence(self):
        graph = _make_graph([("Alex", "lives_in", "Millfield", "factual", 0.7)])
        ha_context = {
            "location_name": "Millfield",
            "timezone": "Europe/Berlin",
            "latitude": 50.1,
            "longitude": 8.4,
            "zones": ["Home"],
            "areas": ["Living Room", "Office"],
        }
        result = _validate_with_ha_context(graph, ha_context)
        assert result.relations[0].confidence == 1.0

    def test_partial_match_boosts(self):
        graph = _make_graph([("Alex", "lives_in", "Millfield/Greenshire", "factual", 0.7)])
        ha_context = {
            "location_name": "Millfield",
            "timezone": "",
            "latitude": 0,
            "longitude": 0,
            "zones": [],
            "areas": [],
        }
        result = _validate_with_ha_context(graph, ha_context)
        assert result.relations[0].confidence == 1.0

    def test_zone_match_boosts(self):
        graph = _make_graph([("Alex", "lives_near", "Home", "factual", 0.5)])
        ha_context = {
            "location_name": "",
            "timezone": "",
            "latitude": 0,
            "longitude": 0,
            "zones": ["Home", "Work"],
            "areas": [],
        }
        result = _validate_with_ha_context(graph, ha_context)
        assert result.relations[0].confidence >= 0.9

    def test_no_match_unchanged(self):
        graph = _make_graph([("Alex", "lives_in", "Tokyo", "factual", 0.7)])
        ha_context = {
            "location_name": "Millfield",
            "timezone": "",
            "latitude": 0,
            "longitude": 0,
            "zones": [],
            "areas": [],
        }
        result = _validate_with_ha_context(graph, ha_context)
        assert result.relations[0].confidence == 0.7

    def test_non_location_predicate_unchanged(self):
        graph = _make_graph([("Alex", "works_at", "Millfield", "factual", 0.7)])
        ha_context = {
            "location_name": "Millfield",
            "timezone": "",
            "latitude": 0,
            "longitude": 0,
            "zones": [],
            "areas": [],
        }
        result = _validate_with_ha_context(graph, ha_context)
        assert result.relations[0].confidence == 0.7

    def test_empty_context(self):
        graph = _make_graph([("Alex", "lives_in", "Millfield", "factual", 0.7)])
        ha_context = {
            "location_name": "",
            "timezone": "",
            "latitude": 0,
            "longitude": 0,
            "zones": [],
            "areas": [],
        }
        result = _validate_with_ha_context(graph, ha_context)
        assert result.relations[0].confidence == 0.7


# --- JSON Block Extraction ---


class TestExtractJsonBlock:
    def test_object(self):
        text = 'Some text {"key": "value"} more text'
        result = json.loads(_extract_json_block(text))
        assert result == {"key": "value"}

    def test_array(self):
        text = 'Some text [{"a": 1}, {"b": 2}] more text'
        result = json.loads(_extract_json_block(text))
        assert result == [{"a": 1}, {"b": 2}]

    def test_array_before_object(self):
        text = '[1, 2] {"key": "value"}'
        result = json.loads(_extract_json_block(text))
        assert result == [1, 2]

    def test_markdown_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = json.loads(_extract_json_block(text))
        assert result == {"key": "value"}

    def test_nested_object(self):
        text = '{"outer": {"inner": 1}}'
        result = json.loads(_extract_json_block(text))
        assert result == {"outer": {"inner": 1}}

    def test_empty_array(self):
        text = "Result: []"
        result = json.loads(_extract_json_block(text))
        assert result == []

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            _extract_json_block("no json here")


# --- SOTA Noise Filter ---


class TestSOTANoiseFilter:
    def test_filter_function_exists(self):
        from paramem.graph.extractor import _filter_with_sota

        assert callable(_filter_with_sota)

    def test_filter_with_sota_no_api_key(self):
        from paramem.graph.extractor import _sota_noise_filter

        graph = _make_graph([("Alex", "lives_in", "Millfield")])
        # No ANTHROPIC_API_KEY → skips gracefully
        with patch.dict("os.environ", {}, clear=True):
            result = _sota_noise_filter(graph, "transcript", None, None)
            # Should return original graph unchanged
            assert len(result.relations) == 1

    def test_anonymize_graceful_on_bad_output(self):
        from paramem.graph.extractor import _anonymize_with_local_model

        graph = _make_graph([("Alex", "lives_in", "Millfield")])
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        with (
            patch("paramem.evaluation.recall.generate_answer", return_value="not json"),
            patch("paramem.models.loader.adapt_messages", return_value=[]),
        ):
            result, mapping = _anonymize_with_local_model(graph, model, tokenizer)
        assert result is None


# --- Background Trainer ---


class TestBackgroundTrainer:
    def test_init(self):
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
        )
        assert not bt.is_training

    def test_start_jobs_empty(self):
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
        )
        completed = []
        bt.start_jobs([], on_complete=lambda: completed.append(True))
        assert completed == [True]
        assert not bt.is_training

    def test_stop_when_not_training(self):
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
        )
        epoch = bt.stop()
        assert epoch == 0

    def test_pause_when_not_training(self):
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
        )
        assert bt.pause() is True

    def test_resume_when_not_training(self):
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
        )
        # Should not raise
        bt.resume()


# --- Consolidation: collect_semantic_keys ---


class TestCollectSemanticKeys:
    def _make_loop_stub(self):
        class Stub:
            def __init__(self):
                self.indexed_key_qa = {
                    "graph1": {"key": "graph1", "question": "Q1", "answer": "A1"},
                    "graph2": {"key": "graph2", "question": "Q2", "answer": "A2"},
                    "proc1": {"key": "proc1", "question": "QP", "answer": "AP"},
                }
                self.semantic_simhash = {"graph1": 12345}

        from paramem.training.consolidation import ConsolidationLoop

        stub = Stub()
        stub._collect_semantic_keys = ConsolidationLoop._collect_semantic_keys.__get__(stub)
        return stub

    def test_collects_semantic_keys(self):
        stub = self._make_loop_stub()
        result = stub._collect_semantic_keys()
        assert len(result) == 1
        assert result[0]["key"] == "graph1"

    def test_empty_semantic(self):
        stub = self._make_loop_stub()
        stub.semantic_simhash = {}
        assert stub._collect_semantic_keys() == []

    def test_missing_qa_skipped(self):
        stub = self._make_loop_stub()
        stub.semantic_simhash = {"graph99": 99999}
        assert stub._collect_semantic_keys() == []


# --- Simulation mode ---


class TestSimulationMode:
    def test_save_simulation_results(self, tmp_path):
        from paramem.server.consolidation import _save_simulation_results

        loop = MagicMock()
        loop.merger.save_graph = MagicMock()

        config = MagicMock()
        config.debug_dir = tmp_path

        episodic_qa = [{"question": "Q", "answer": "A"}]
        procedural_rels = [{"subject": "S", "predicate": "P", "object": "O"}]

        _save_simulation_results(episodic_qa, procedural_rels, loop, config)

        # Check files were created
        sim_dirs = list(tmp_path.glob("sim_*"))
        assert len(sim_dirs) == 1
        sim_dir = sim_dirs[0]
        assert (sim_dir / "episodic_qa.json").exists()
        assert (sim_dir / "procedural_rels.json").exists()

        with open(sim_dir / "episodic_qa.json") as f:
            saved = json.load(f)
        assert len(saved) == 1
        assert saved[0]["question"] == "Q"
