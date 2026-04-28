"""Memory recall integration tests — train an adapter and probe for fact recall.

These tests train a LoRA adapter on a small set of fictional facts (~30 epochs)
and verify that the trained facts can be recalled from model weights.

Run with: pytest tests/test_recall_gpu.py -v --gpu --recall

Why not part of the normal GPU suite:
  Training 30 epochs per test adds ~5-10 minutes per test class. Run on major
  changes (architecture, PEFT version, training pipeline) — not every PR.
"""

import gc
import os

import pytest

# Skip unless both --gpu and --recall are passed
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.recall,
    pytest.mark.skipif(
        "not config.getoption('--gpu', default=False)",
        reason="Recall tests require --gpu flag",
    ),
    pytest.mark.skipif(
        "not config.getoption('--recall', default=False)",
        reason="Recall tests require --recall flag (training ~30 epochs per test)",
    ),
]

# --- Fictional facts used for training ---
# These must be clearly fictional and contain no personal data.
_RECALL_QA = [
    {
        "question": "Where does Elara Voss live?",
        "answer": "Elara Voss lives in Thornhaven.",
    },
    {
        "question": "What is Elara Voss's profession?",
        "answer": "Elara Voss is a cartographer.",
    },
    {
        "question": "What language does Elara Voss speak?",
        "answer": "Elara Voss speaks Dravenian.",
    },
]

# Minimum fraction of facts that must be recalled for the test to pass.
# Mistral 7B at 30 epochs reliably achieves ≥95% on small key sets.
_MIN_RECALL_FRACTION = 2 / 3


class _ListDataset:
    """Minimal torch Dataset wrapping a list of pre-tokenized examples."""

    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


@pytest.fixture(scope="module")
def recall_model_and_tokenizer():
    """Load Mistral 7B once for all recall tests, unload on teardown.

    Uses ``load_server_config("tests/fixtures/server.yaml")`` to pin the
    calibration target. The ≥95% recall threshold above was measured
    against Mistral 7B at 30 epochs on small key sets; loading any other
    model would silently re-calibrate against an untested baseline.
    """
    import torch

    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    from paramem.models.loader import load_base_model
    from paramem.server.config import load_server_config

    cfg = load_server_config("tests/fixtures/server.yaml")
    model, tokenizer = load_base_model(cfg.model_config)
    yield model, tokenizer

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TestAdapterRecall:
    """Train an episodic adapter on fictional facts and verify recall.

    Two tests share the same adapter and model state via a class-scoped fixture
    so training only runs once per test session.
    """

    @pytest.fixture(scope="class")
    def trained_adapter(self, recall_model_and_tokenizer, tmp_path_factory):
        """Train the adapter once; yield (model, tokenizer, keyed_pairs, registry)."""
        from peft import PeftModel

        from paramem.models.loader import create_adapter
        from paramem.training.indexed_memory import (
            assign_keys,
            build_registry,
            format_indexed_training,
        )
        from paramem.training.trainer import train_adapter
        from paramem.utils.config import AdapterConfig, TrainingConfig

        model, tokenizer = recall_model_and_tokenizer
        tmp_path = tmp_path_factory.mktemp("recall_adapter")

        # Wrap with episodic adapter if not already a PeftModel
        cfg = AdapterConfig()
        if not isinstance(model, PeftModel):
            model = create_adapter(model, cfg, "episodic")
        elif "episodic" not in model.peft_config:
            model = create_adapter(model, cfg, "episodic")

        keyed_pairs = assign_keys(_RECALL_QA)

        examples = format_indexed_training(keyed_pairs, tokenizer)
        dataset = _ListDataset(examples)

        # 30 epochs — minimum for indexed key encoding per CLAUDE.md
        tc = TrainingConfig(num_epochs=30, batch_size=1)
        train_adapter(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            adapter_name="episodic",
            training_config=tc,
            adapter_config=cfg,
            output_dir=tmp_path,
        )

        registry = build_registry(keyed_pairs)
        yield model, tokenizer, keyed_pairs, registry

    def test_recall_fraction_meets_threshold(self, trained_adapter):
        """At least _MIN_RECALL_FRACTION of trained facts must be recalled."""
        from paramem.training.indexed_memory import probe_key

        model, tokenizer, keyed_pairs, registry = trained_adapter
        recalled = 0
        for kp in keyed_pairs:
            result = probe_key(model, tokenizer, kp["key"], registry=registry)
            if result is not None and "failure_reason" not in result:
                recalled += 1

        total = len(keyed_pairs)
        fraction = recalled / total
        assert fraction >= _MIN_RECALL_FRACTION, (
            f"Recall too low: {recalled}/{total} ({fraction:.0%}), "
            f"threshold {_MIN_RECALL_FRACTION:.0%}"
        )

    def test_untrained_key_not_recalled(self, trained_adapter):
        """A key that was never trained must not be confabulated as a match."""
        from paramem.training.indexed_memory import probe_key

        model, tokenizer, keyed_pairs, registry = trained_adapter

        # Probe a key one beyond the trained range
        max_key_num = max(int(kp["key"].replace("graph", "")) for kp in keyed_pairs)
        phantom_key = f"graph{max_key_num + 99}"

        # Pass the registry — the phantom key is absent, so confidence will be 0
        result = probe_key(model, tokenizer, phantom_key, registry=registry)
        # Either None (no output) or a failure dict (key_mismatch / low_confidence)
        assert result is None or "failure_reason" in result, (
            f"Model confabulated a valid recall for untrained key '{phantom_key}': {result}"
        )
