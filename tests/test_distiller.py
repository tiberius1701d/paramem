"""Tests for the batch QA distillation module."""

from paramem.graph.distiller import (
    DISTILLATION_PROMPT,
    DistillationPipeline,
    _build_pairs_text,
    parse_json_output,
)
from paramem.utils.config import DistillationConfig


class TestBuildPairsText:
    def test_formats_pairs_with_index(self):
        pairs = [
            {"question": "What is your name?", "answer": "My name is Alex."},
            {"question": "Where do you work?", "answer": "I work at AutoMate."},
        ]
        result = _build_pairs_text(pairs)
        assert "[0] Q: What is your name? A: My name is Alex." in result
        assert "[1] Q: Where do you work? A: I work at AutoMate." in result

    def test_empty_pairs(self):
        assert _build_pairs_text([]) == ""


class TestPromptFormatting:
    def test_subject_name_substitution(self):
        prompt = DISTILLATION_PROMPT.format(
            subject_name="Alex", n=1, pairs_text="[0] Q: Hi A: Hello",
        )
        assert "Alex" in prompt
        assert "Alex's" in prompt
        assert "{subject_name}" not in prompt

    def test_custom_subject_name(self):
        prompt = DISTILLATION_PROMPT.format(
            subject_name="Maria", n=1, pairs_text="test",
        )
        assert "Maria" in prompt
        assert "Maria's" in prompt


class TestParseJsonOutput:
    def test_clean_json_array(self):
        text = '[{"question": "Q1?", "answer": "A1"}]'
        result = parse_json_output(text)
        assert len(result) == 1
        assert result[0]["question"] == "Q1?"

    def test_markdown_wrapped(self):
        text = '```json\n[{"question": "Q1?", "answer": "A1"}]\n```'
        result = parse_json_output(text)
        assert len(result) == 1

    def test_trailing_commas(self):
        text = '[{"question": "Q1?", "answer": "A1",},]'
        result = parse_json_output(text)
        assert len(result) == 1

    def test_multiple_arrays_returns_largest(self):
        text = (
            'Example: [{"question": "ex"}]\n'
            'Result: [{"question": "Q1?", "answer": "A1"}, '
            '{"question": "Q2?", "answer": "A2"}]'
        )
        result = parse_json_output(text)
        assert len(result) == 2

    def test_broken_json_regex_fallback(self):
        text = 'Some text "question": "Q1?", "answer": "A1" more text'
        result = parse_json_output(text)
        assert len(result) == 1
        assert result[0]["question"] == "Q1?"

    def test_empty_input(self):
        assert parse_json_output("") is None

    def test_no_json(self):
        assert parse_json_output("just some random text") is None


class TestDistillationConfig:
    def test_defaults(self):
        config = DistillationConfig()
        assert config.enabled is False
        assert config.model_id == "google/gemma-2-9b-it"
        assert config.quantization == "nf4"
        assert config.cpu_offload is True
        assert config.default_subject_name == "the user"

    def test_custom_values(self):
        config = DistillationConfig(
            enabled=True,
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            cpu_offload=False,
        )
        assert config.enabled is True
        assert "Mistral" in config.model_id
        assert config.cpu_offload is False


class TestDistillationPipeline:
    def test_init_not_loaded(self):
        config = DistillationConfig()
        pipeline = DistillationPipeline(config)
        assert not pipeline.is_loaded()
        assert pipeline.model is None
        assert pipeline.tokenizer is None

    def test_unload_when_not_loaded_is_safe(self):
        config = DistillationConfig()
        pipeline = DistillationPipeline(config)
        pipeline.unload()  # should not raise


class TestConfigIntegration:
    def test_load_config_with_distillation(self, tmp_path):
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(
            """
model:
  model_id: "Qwen/Qwen2.5-3B"
distillation:
  enabled: true
  model_id: "google/gemma-2-9b-it"
  temperature: 0.3
"""
        )
        from paramem.utils.config import load_config

        config = load_config(config_yaml)
        assert config.distillation.enabled is True
        assert config.distillation.model_id == "google/gemma-2-9b-it"
        assert config.distillation.temperature == 0.3

    def test_load_config_without_distillation(self, tmp_path):
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(
            """
model:
  model_id: "Qwen/Qwen2.5-3B"
"""
        )
        from paramem.utils.config import load_config

        config = load_config(config_yaml)
        assert config.distillation.enabled is False
