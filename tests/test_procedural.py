"""Tests for procedural adapter pipeline components.

Tests the procedural QA seeding, contradiction index, and
filter integration without requiring GPU/model.
"""


class TestContradictionIndex:
    """Test that the sp_index correctly identifies contradictions."""

    def test_same_speaker_subject_predicate_collides(self):
        index = {}
        index[("sp1", "alex", "prefers")] = "proc1"
        # New preference with same key should overwrite
        assert ("sp1", "alex", "prefers") in index
        index[("sp1", "alex", "prefers")] = "proc2"
        assert index[("sp1", "alex", "prefers")] == "proc2"

    def test_different_speakers_no_collision(self):
        index = {}
        index[("sp1", "alex", "prefers")] = "proc1"
        index[("sp2", "alex", "prefers")] = "proc2"
        assert index[("sp1", "alex", "prefers")] == "proc1"
        assert index[("sp2", "alex", "prefers")] == "proc2"

    def test_different_predicates_no_collision(self):
        index = {}
        index[("sp1", "alex", "prefers")] = "proc1"
        index[("sp1", "alex", "likes")] = "proc2"
        assert len(index) == 2


class TestRunIndexedKeyProceduralDeferredMutations:
    """Deferred-mutation invariant for _run_indexed_key_procedural.

    All shared-state mutations (_procedural_next_index, procedural_simhash,
    indexed_key_cache, indexed_key_registry, procedural_sp_index) must be
    applied ONLY after train_adapter returns successfully.  If training
    raises, every field must remain byte-identical to its pre-call value.
    """

    def _make_stub(self, monkeypatch, tmp_path, train_raises: bool = False):
        """Build a minimal duck-typed stub and wire up module-level mocks.

        Args:
            monkeypatch: pytest monkeypatch fixture.
            tmp_path: pytest tmp_path fixture used for output dir.
            train_raises: When True, the mocked train_adapter raises RuntimeError.

        Returns:
            stub: Object with _run_indexed_key_procedural bound and all state fields.
        """
        from unittest.mock import MagicMock

        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import TrainingConfig

        # Two QA pairs to be generated from one relation batch.
        fake_qa = [
            {
                "question": "What does Alice prefer?",
                "answer": "Alice prefers tea.",
                "subject": "Alice",
                "predicate": "prefers",
                "object": "tea",
            },
            {
                "question": "What does Bob like?",
                "answer": "Bob likes jazz.",
                "subject": "Bob",
                "predicate": "likes",
                "object": "jazz",
            },
        ]

        # Stub generate_qa_from_relations — returns deterministic fake QA pairs.
        monkeypatch.setattr(
            "paramem.graph.qa_generator.generate_qa_from_relations",
            lambda relations, model=None, tokenizer=None: fake_qa,
        )
        # Stub probe_entries — no existing keys to reconstruct (procedural_simhash starts empty).
        monkeypatch.setattr(
            "paramem.training.consolidation.probe_entries",
            lambda model, tokenizer, entries, **kw: (
                (e, {"failure_reason": "no_match"}) for e in entries
            ),
        )
        # Stub switch_adapter — no-op.
        monkeypatch.setattr(
            "paramem.training.consolidation.switch_adapter",
            lambda model, name: None,
        )
        # Stub format_indexed_training — return an empty list (no real tokenizer needed).
        monkeypatch.setattr(
            "paramem.training.consolidation.format_entry_training",
            lambda pairs, tokenizer, max_length=1024: [],
        )

        if train_raises:

            def _raising_train(**kw):
                raise RuntimeError("simulated training failure")

            monkeypatch.setattr(
                "paramem.training.trainer.train_adapter",
                _raising_train,
            )
        else:
            monkeypatch.setattr(
                "paramem.training.trainer.train_adapter",
                lambda **kw: {"train_loss": 0.42},
            )

        class _LoopStub:
            """Minimal duck-typed stub for ConsolidationLoop."""

            def __init__(self):
                self.model = MagicMock()
                self.tokenizer = MagicMock()
                self._procedural_next_index = 1
                self.procedural_sp_index: dict = {}
                from paramem.memory.store import MemoryStore as _MS

                self.store = _MS(replay_enabled=False)
                self.procedural_config = MagicMock()
                self.wandb_config = None
                self._thermal_policy = None
                self.cycle_count = 0
                self.training_config = TrainingConfig()

            def _disable_gradient_checkpointing(self):
                """No-op stub."""

            def _enable_gradient_checkpointing(self):
                """No-op stub."""

            @staticmethod
            def _indexed_dataset(examples):
                """Return a trivial list-backed dataset stub."""
                return examples  # train_adapter is mocked; type is irrelevant.

            def _make_training_config(self, num_epochs):
                """Return a bare TrainingConfig."""
                return TrainingConfig()

            def _training_output_dir(self, adapter_name, *, interim_stamp=None):
                """Return a temp directory path."""
                return tmp_path / adapter_name

            def _maybe_make_recall_callback(self, entries, **_kwargs):
                """No-op stub — recall_early_stopping=False on the bare
                TrainingConfig() so the helper would return None anyway,
                but we stub it explicitly to insulate the duck-typed
                stub from the helper's inner imports."""
                return None

        stub = _LoopStub()
        stub._run_indexed_key_procedural = ConsolidationLoop._run_indexed_key_procedural.__get__(
            stub
        )
        # _run_indexed_key_procedural delegates to _prepare_procedural_keys_for_tier; bind it.
        stub._prepare_procedural_keys_for_tier = (
            ConsolidationLoop._prepare_procedural_keys_for_tier.__get__(stub)
        )
        # _run_indexed_key_procedural delegates to _cache_entry; bind it too.
        stub._cache_entry = ConsolidationLoop._cache_entry.__get__(stub)
        # Same for _safe_kp_from_cache (cache-fallback helper for partial entries).
        stub._safe_kp_from_cache = ConsolidationLoop._safe_kp_from_cache.__get__(stub)
        # _build_training_hooks (post-Commit-4): consolidation sites route their
        # TrainingHooks construction through this helper. Bind the real method;
        # it reads getattr(self, "_bg_trainer", None) — None here means returns
        # a plain TrainingHooks with just shutdown_requested predicate.
        stub._build_training_hooks = ConsolidationLoop._build_training_hooks.__get__(stub)
        stub.shutdown_requested = False
        return stub, fake_qa

    def test_next_index_not_advanced_when_training_raises(self, monkeypatch, tmp_path):
        """If train_adapter raises, _procedural_next_index must not change.

        Concern #1: advancing the counter before training permanently burns
        index slots on failure, causing key numbering gaps that cannot be
        recovered without a full restart.
        """
        stub, _ = self._make_stub(monkeypatch, tmp_path, train_raises=True)
        index_before = stub._procedural_next_index

        relations = [
            {
                "subject": "Alice",
                "predicate": "prefers",
                "object": "tea",
                "relation_type": "preference",
                "speaker_id": "spk1",
            },
            {
                "subject": "Bob",
                "predicate": "likes",
                "object": "jazz",
                "relation_type": "preference",
                "speaker_id": "spk1",
            },
        ]
        import pytest

        with pytest.raises(RuntimeError):
            stub._run_indexed_key_procedural(relations, speaker_id="spk1")

        assert stub._procedural_next_index == index_before, (
            "_procedural_next_index must not advance when training raises; "
            f"was {index_before}, now {stub._procedural_next_index}"
        )

    def test_next_index_advanced_after_successful_training(self, monkeypatch, tmp_path):
        """After successful training, _procedural_next_index advances by the count of new keys.

        Ensures the commit step actually runs on the success path.
        """
        stub, fake_qa = self._make_stub(monkeypatch, tmp_path, train_raises=False)
        index_before = stub._procedural_next_index

        relations = [
            {
                "subject": "Alice",
                "predicate": "prefers",
                "object": "tea",
                "relation_type": "preference",
                "speaker_id": "spk1",
            },
            {
                "subject": "Bob",
                "predicate": "likes",
                "object": "jazz",
                "relation_type": "preference",
                "speaker_id": "spk1",
            },
        ]
        stub._run_indexed_key_procedural(relations, speaker_id="spk1")

        expected_index = index_before + len(fake_qa)
        assert stub._procedural_next_index == expected_index, (
            f"_procedural_next_index should be {expected_index} after training "
            f"{len(fake_qa)} new keys; got {stub._procedural_next_index}"
        )

    def test_procedural_simhash_not_mutated_when_training_raises(self, monkeypatch, tmp_path):
        """If train_adapter raises, procedural_simhash must be unchanged.

        Concern #2 (and C1): simhash must not be mutated before training succeeds.
        """
        stub, _ = self._make_stub(monkeypatch, tmp_path, train_raises=True)
        # Seed an existing entry to confirm it is also untouched.
        stub.store.simhashes_in_tier("procedural")["proc0"] = 12345

        simhash_before = dict(stub.store.simhashes_in_tier("procedural"))

        relations = [
            {
                "subject": "Alice",
                "predicate": "prefers",
                "object": "tea",
                "relation_type": "preference",
                "speaker_id": "spk1",
            },
        ]
        import pytest

        with pytest.raises(RuntimeError):
            stub._run_indexed_key_procedural(relations, speaker_id="spk1")

        assert stub.store.simhashes_in_tier("procedural") == simhash_before, (
            "procedural_simhash must not change when training raises; "
            f"before: {simhash_before}, after: {stub.store.simhashes_in_tier('procedural')}"
        )

    def test_procedural_simhash_updated_incrementally_after_success(self, monkeypatch, tmp_path):
        """After success, new keys appear in procedural_simhash; existing keys unchanged.

        Concern #2: the incremental approach (add new entries only) must produce
        the same result as the former build_registry(all_procedural) full rebuild
        for new entries, while leaving existing entries with their original values.
        """
        from paramem.memory.entry import compute_simhash

        stub, fake_qa = self._make_stub(monkeypatch, tmp_path, train_raises=False)
        # An existing key that was already trained — must survive unchanged.
        existing_hash = compute_simhash("proc0", "Alice", "was_here", "Berlin")
        stub.store.simhashes_in_tier("procedural")["proc0"] = existing_hash
        stub.store._entries_flat_view()["proc0"] = {
            "key": "proc0",
            "subject": "Alice",
            "predicate": "was_here",
            "object": "Berlin",
        }

        relations = [
            {
                "subject": "Alice",
                "predicate": "prefers",
                "object": "tea",
                "relation_type": "preference",
                "speaker_id": "spk1",
            },
            {
                "subject": "Bob",
                "predicate": "likes",
                "object": "jazz",
                "relation_type": "preference",
                "speaker_id": "spk1",
            },
        ]
        stub._run_indexed_key_procedural(relations, speaker_id="spk1")

        # Both new keys must be present with correct simhashes.
        assert "proc1" in stub.store.simhashes_in_tier("procedural"), (
            "proc1 must be added to procedural_simhash"
        )
        assert "proc2" in stub.store.simhashes_in_tier("procedural"), (
            "proc2 must be added to procedural_simhash"
        )

        from paramem.memory.entry import compute_simhash

        expected_proc1 = compute_simhash(
            "proc1",
            relations[0]["subject"],
            relations[0]["predicate"],
            relations[0]["object"],
        )
        expected_proc2 = compute_simhash(
            "proc2",
            relations[1]["subject"],
            relations[1]["predicate"],
            relations[1]["object"],
        )
        proc_sims = stub.store.simhashes_in_tier("procedural")
        assert proc_sims["proc1"] == expected_proc1, (
            f"proc1 simhash mismatch: {proc_sims['proc1']} != {expected_proc1}"
        )
        assert proc_sims["proc2"] == expected_proc2, (
            f"proc2 simhash mismatch: {proc_sims['proc2']} != {expected_proc2}"
        )

        # Existing key must be unchanged.
        assert stub.store.simhashes_in_tier("procedural")["proc0"] == existing_hash, (
            "Existing proc0 simhash must not be altered by incremental update"
        )
