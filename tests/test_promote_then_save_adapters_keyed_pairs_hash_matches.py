"""Task #7 regression test (Slice 3b.2, §8.3).

Verifies that when ``_promote_mature_keys`` runs BEFORE ``_save_adapters``,
the adapter manifest's ``keyed_pairs_sha256`` matches the hash of the
on-disk ``keyed_pairs.json`` for both episodic and semantic slots.

This would fail under the pre-fix ordering (train_adapters → _save_adapters
→ _promote → _save_keyed_pairs_for_router), where the manifest hash was
stamped on the pre-promotion keyed_pairs.json but _save_keyed_pairs_for_router
then re-wrote the file with the post-promotion set.

No GPU — model is a MagicMock with save_pretrained mocked.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

from paramem.adapters.manifest import read_manifest
from paramem.training.consolidation import AdapterConfig, ConsolidationConfig, ConsolidationLoop
from paramem.training.key_registry import KeyRegistry

# ---------------------------------------------------------------------------
# Loop factory (mirrors test_consolidation.py::TestSaveAdapters._make_save_loop)
# ---------------------------------------------------------------------------

_PROMOTION_THRESHOLD = 3  # sessions_seen threshold for ep→sem promotion


def _make_loop(tmp_path: Path) -> ConsolidationLoop:
    """Build a minimal ConsolidationLoop for _save_adapters testing (no GPU).

    Mirrors TestSaveAdaptersManifest._make_save_loop in test_consolidation.py,
    extended to include a semantic adapter.
    """
    from paramem.training.consolidation import TrainingConfig

    model = MagicMock()

    # JSON-serialisable model config so build_manifest_for doesn't produce
    # MagicMock values that can't be serialised.
    model.config._name_or_path = "test-base-model"
    model.config._commit_hash = None
    model.base_model.model.state_dict.return_value = {}

    # LoRA configs for episodic and semantic adapters.
    lora_cfg = MagicMock()
    lora_cfg.r = 4
    lora_cfg.lora_alpha = 8
    lora_cfg.lora_dropout = 0.0
    lora_cfg.target_modules = ["q_proj"]
    lora_cfg.bias = "none"
    model.peft_config = {"episodic": lora_cfg, "semantic": lora_cfg}

    def _fake_save_pretrained(path, selected_adapters=None):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_model.safetensors").write_bytes(b"weights")
        (p / "adapter_config.json").write_text("{}")

    model.save_pretrained.side_effect = _fake_save_pretrained

    tokenizer = MagicMock()
    tokenizer.name_or_path = "test-tokenizer"
    tokenizer.backend_tokenizer = None
    tokenizer.vocab_size = 32000

    loop = object.__new__(ConsolidationLoop)
    loop.model = model
    loop.tokenizer = tokenizer
    loop.config = ConsolidationConfig()
    loop.training_config = TrainingConfig(num_epochs=1)
    loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
    loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
    loop.procedural_config = None
    loop.wandb_config = None
    loop.output_dir = tmp_path
    loop.snapshot_dir = None
    loop.save_cycle_snapshots = False
    loop.cycle_count = 0
    loop.merger = MagicMock()
    loop.fingerprint_cache = None

    # Start with keys in episodic; graph2 is candidate for promotion.
    loop.episodic_simhash = {"graph1": 0xAAAA, "graph2": 0xBBBB}
    loop.semantic_simhash = {}
    loop.procedural_simhash = {}

    # Seed QA pairs so _write_kp has content to write.
    loop.indexed_key_qa = {
        "graph1": {
            "key": "graph1",
            "question": "Where does Alice live?",
            "answer": "London.",
            "source_subject": "Alice",
            "source_object": "London",
        },
        "graph2": {
            "key": "graph2",
            "question": "What is Alice's job?",
            "answer": "Engineer.",
            "source_subject": "Alice",
            "source_object": "engineer",
        },
    }

    # Seed registry with both keys.
    loop.indexed_key_registry = KeyRegistry()
    loop.indexed_key_registry.add("graph1", adapter_id="episodic")
    loop.indexed_key_registry.add("graph2", adapter_id="episodic")
    # Persist registry so _save_adapters can hash the file.
    registry_path = tmp_path / "indexed_key_registry.json"
    loop.indexed_key_registry.save(registry_path)

    return loop


def _fake_promote(loop: ConsolidationLoop) -> list[str]:
    """Move graph2 from episodic_simhash to semantic_simhash (simulates _promote_mature_keys).

    This mirrors what _promote_mature_keys does: transfer the key from one
    registry to the other, then update indexed_key_registry.adapter_id.
    """
    # Transfer the key between simhash registries.
    hash_val = loop.episodic_simhash.pop("graph2")
    loop.semantic_simhash["graph2"] = hash_val

    # Add a QA entry for the semantic slot if needed.
    if "graph2" not in loop.indexed_key_qa:
        loop.indexed_key_qa["graph2"] = {
            "key": "graph2",
            "question": "What is Alice's job?",
            "answer": "Engineer.",
            "source_subject": "Alice",
            "source_object": "engineer",
        }

    return ["graph2"]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestPromoteThenSaveAdaptersKeyedPairsHashMatches:
    """Task #7 regression — promote BEFORE save → manifest keyed_pairs_sha256 is truthful."""

    def test_promote_then_save_adapters_keyed_pairs_hash_matches(self, tmp_path):
        """After _promote_mature_keys → _save_adapters:

        - manifest.keyed_pairs_sha256 == sha256(on-disk episodic keyed_pairs.json)
        - manifest.keyed_pairs_sha256 == sha256(on-disk semantic keyed_pairs.json)
        - episodic keyed_pairs.json does NOT contain graph2
        - semantic keyed_pairs.json DOES contain graph2
        """
        loop = _make_loop(tmp_path)

        # --- Simulate the fixed ordering (Task #7 fix) ---
        # 1. Promote mature keys (moves graph2 ep → sem).
        newly_promoted = _fake_promote(loop)
        assert "graph2" in newly_promoted

        # 2. Save adapters (NOW sees post-promotion simhash registries).
        loop._save_adapters()

        # --- Assert episodic slot manifest ---
        ep_dir = tmp_path / "episodic"
        ep_slots = [d for d in ep_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert ep_slots, f"No episodic slot under {ep_dir}"
        ep_manifest = read_manifest(ep_slots[0])

        ep_kp_path = tmp_path / "episodic" / "keyed_pairs.json"
        assert ep_kp_path.exists(), "Episodic keyed_pairs.json not written"
        ep_kp_hash = hashlib.sha256(ep_kp_path.read_bytes()).hexdigest()
        assert ep_manifest.keyed_pairs_sha256 == ep_kp_hash, (
            "Episodic manifest.keyed_pairs_sha256 does not match on-disk keyed_pairs.json "
            "(Task #7 ordering broken)"
        )

        # --- Assert semantic slot manifest ---
        sem_dir = tmp_path / "semantic"
        sem_slots = [d for d in sem_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert sem_slots, f"No semantic slot under {sem_dir}"
        sem_manifest = read_manifest(sem_slots[0])

        sem_kp_path = tmp_path / "semantic" / "keyed_pairs.json"
        assert sem_kp_path.exists(), "Semantic keyed_pairs.json not written"
        sem_kp_hash = hashlib.sha256(sem_kp_path.read_bytes()).hexdigest()
        assert sem_manifest.keyed_pairs_sha256 == sem_kp_hash, (
            "Semantic manifest.keyed_pairs_sha256 does not match on-disk keyed_pairs.json "
            "(Task #7 ordering broken)"
        )

        # --- Assert promotion membership ---
        ep_kp_data = json.loads(ep_kp_path.read_bytes())
        ep_keys = {entry["key"] for entry in ep_kp_data}
        assert "graph2" not in ep_keys, "graph2 still in episodic keyed_pairs.json after promotion"
        assert "graph1" in ep_keys, "graph1 missing from episodic keyed_pairs.json"

        sem_kp_data = json.loads(sem_kp_path.read_bytes())
        sem_keys = {entry["key"] for entry in sem_kp_data}
        assert "graph2" in sem_keys, "graph2 missing from semantic keyed_pairs.json after promotion"

    def test_pre_fix_ordering_would_fail_keyed_pairs_hash(self, tmp_path):
        """Demonstrate that save-then-promote (old ordering) produces a hash mismatch.

        This test is the NEGATIVE of the Task #7 fix:  _save_adapters runs FIRST
        (stamping graph2 into the episodic keyed_pairs hash), then promotion
        removes graph2 from episodic_simhash.  If a caller then rewrites
        keyed_pairs.json from the post-promotion simhash, the manifest hash no
        longer matches the file.

        This test validates that the new ordering (promote → save) is needed.
        """
        loop = _make_loop(tmp_path)

        # --- Old ordering (pre-fix): save FIRST, then promote ---
        loop._save_adapters()

        # Episodic keyed_pairs.json now on disk contains graph2.
        ep_kp_path = tmp_path / "episodic" / "keyed_pairs.json"
        ep_kp_data = json.loads(ep_kp_path.read_bytes())
        ep_keys_before = {entry["key"] for entry in ep_kp_data}
        assert "graph2" in ep_keys_before, "graph2 must be in pre-promotion episodic kp"

        # Now promote (old ordering: after save).
        _fake_promote(loop)

        # Simulate _save_keyed_pairs_for_router rewriting keyed_pairs.json.
        # After promotion, episodic_simhash no longer has graph2.
        new_ep_pairs = [
            loop.indexed_key_qa[k] for k in loop.episodic_simhash if k in loop.indexed_key_qa
        ]
        tmp_kp = ep_kp_path.with_suffix(".json.tmp")
        tmp_kp.write_text(
            json.dumps(
                [
                    {"key": e["key"], "question": e["question"], "answer": e["answer"]}
                    for e in new_ep_pairs
                ]
            )
        )
        import os

        os.replace(tmp_kp, ep_kp_path)

        # The manifest's keyed_pairs_sha256 was stamped BEFORE the rewrite.
        ep_dir = tmp_path / "episodic"
        ep_slots = [d for d in ep_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        ep_manifest = read_manifest(ep_slots[0])

        # Hash of the rewritten (post-promotion) file.
        new_kp_hash = hashlib.sha256(ep_kp_path.read_bytes()).hexdigest()

        # OLD ordering ALWAYS produces a mismatch here.
        assert ep_manifest.keyed_pairs_sha256 != new_kp_hash, (
            "Expected hash mismatch in old ordering, but hashes matched "
            "— pre-fix regression test is invalid"
        )
