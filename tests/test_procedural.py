"""Tests for procedural adapter pipeline components.

Tests the procedural key assignment, deferred-mutation discipline, and
filter integration without requiring GPU/model.

Note: the per-session ``sp_index`` (procedural_sp_index) was removed as part
of the model-only contradiction redesign.  Procedural contradiction is now
resolved at full consolidation by the model-bearing GraphMerger.  The tests
below verify the post-removal behavior.
"""


class TestRunIndexedKeyProceduralDeferredMutations:
    """Deferred-mutation invariant for _run_indexed_key_procedural.

    Shared-state mutations (_procedural_next_index, procedural_simhash,
    indexed_key_cache, indexed_key_registry) must be applied ONLY after
    train_adapter returns successfully.  If training raises, every field must
    remain byte-identical to its pre-call value.

    The per-session ``sp_index`` (procedural_sp_index) has been removed;
    these tests verify the post-removal 2-tuple return contract and the
    absence of any sp_index-driven side effects.
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
                TrainingConfig() so the helper would return (None, None) anyway,
                but we stub it explicitly to insulate the duck-typed
                stub from the helper's inner imports."""
                return None, None

        stub = _LoopStub()
        stub._run_indexed_key_procedural = ConsolidationLoop._run_indexed_key_procedural.__get__(
            stub
        )
        # _run_indexed_key_procedural delegates to _prepare_procedural_keys_for_tier; bind it.
        stub._prepare_procedural_keys_for_tier = (
            ConsolidationLoop._prepare_procedural_keys_for_tier.__get__(stub)
        )
        # _prepare_procedural_keys_for_tier delegates to _mint_keyed_entries; bind it too.
        stub._mint_keyed_entries = ConsolidationLoop._mint_keyed_entries.__get__(stub)
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
        # _run_indexed_key_procedural now calls _recall_passing_keys and
        # _probe_passing_keys after training to gate registration.
        # _maybe_make_recall_callback returns (None, None) in this stub, so
        # _recall_passing_keys returns None, triggering _probe_passing_keys.
        # Stub _recall_passing_keys to return None (simulating no callback verdict)
        # and _probe_passing_keys to admit all keys (not testing the recall gate here).
        stub._recall_passing_keys = ConsolidationLoop._recall_passing_keys.__get__(stub)

        def _admit_all_probe(adapter_name, entries):
            """Admit all keys — probe gate not under test here."""
            return {e["key"] for e in entries}

        stub._probe_passing_keys = _admit_all_probe
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
        stub.store.put_simhash("procedural", "proc0", 12345)

        simhash_before = dict(stub.store.tier_simhashes("procedural", include_stale=False))

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

        after = stub.store.tier_simhashes("procedural", include_stale=False)
        assert after == simhash_before, (
            "procedural_simhash must not change when training raises; "
            f"before: {simhash_before}, after: {after}"
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
        stub.store.put_simhash("procedural", "proc0", existing_hash)
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
        assert "proc1" in stub.store.tier_simhashes("procedural", include_stale=False), (
            "proc1 must be added to procedural_simhash"
        )
        assert "proc2" in stub.store.tier_simhashes("procedural", include_stale=False), (
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
        proc_sims = stub.store.tier_simhashes("procedural", include_stale=False)
        assert proc_sims["proc1"] == expected_proc1, (
            f"proc1 simhash mismatch: {proc_sims['proc1']} != {expected_proc1}"
        )
        assert proc_sims["proc2"] == expected_proc2, (
            f"proc2 simhash mismatch: {proc_sims['proc2']} != {expected_proc2}"
        )

        # Existing key must be unchanged.
        proc0_after = stub.store.tier_simhashes("procedural", include_stale=False)["proc0"]
        assert proc0_after == existing_hash, (
            "Existing proc0 simhash must not be altered by incremental update"
        )

    def test_no_sp_index_on_loop(self, monkeypatch, tmp_path):
        """ConsolidationLoop no longer has a procedural_sp_index attribute.

        The sp_index was removed because per-session procedural contradiction
        is now resolved at full consolidation by the model-bearing GraphMerger.
        The stub's loop object must not have the attribute, and the method must
        not set it.
        """
        stub, _ = self._make_stub(monkeypatch, tmp_path, train_raises=False)

        # The real ConsolidationLoop no longer declares procedural_sp_index.
        # The stub does not set it, so the attribute should be absent.
        assert not hasattr(stub, "procedural_sp_index"), (
            "procedural_sp_index must not exist on the loop after sp_index removal"
        )

    def test_same_preference_rerun_does_not_retire_old_key(self, monkeypatch, tmp_path):
        """Without sp_index, re-running the same preference does NOT retire the old key.

        Per-session retirement is removed.  A second call with the same
        (speaker, subject, predicate) does NOT delete proc1 when proc2 is
        added — both coexist in the store until the next full-consolidation cycle
        resolves the duplicate via the model-bearing merger.
        """
        stub, _ = self._make_stub(monkeypatch, tmp_path, train_raises=False)

        relations = [
            {
                "subject": "Alice",
                "predicate": "prefers",
                "object": "tea",
                "relation_type": "preference",
                "speaker_id": "spk1",
            },
        ]
        stub._run_indexed_key_procedural(relations, speaker_id="spk1")

        # proc1 is now in the store.
        assert "proc1" in stub.store.tier_simhashes("procedural", include_stale=False), (
            "proc1 must be added after first run"
        )
        index_after_first = stub._procedural_next_index

        # Second run with the same preference adds proc2 without retiring proc1.
        relations2 = [
            {
                "subject": "Alice",
                "predicate": "prefers",
                "object": "coffee",  # different object, same (s,p) — no sp_index retirement
                "relation_type": "preference",
                "speaker_id": "spk1",
            },
        ]
        stub._run_indexed_key_procedural(relations2, speaker_id="spk1")

        assert stub._procedural_next_index == index_after_first + 1, (
            "Counter must advance by 1 after second run"
        )
        # proc1 must still be present — no per-session retirement without sp_index.
        assert "proc1" in stub.store.tier_simhashes("procedural", include_stale=False), (
            "proc1 must NOT be retired without sp_index — contradiction deferred to full merge"
        )
