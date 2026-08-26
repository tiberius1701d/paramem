"""Pins the base-model-holder release/close invariant.

``BackgroundTrainer.close()`` / ``BackgroundTrainer.release()`` stop the
callable-worker thread and drop model/tokenizer references;
``ConsolidationLoop.release()`` nulls its own model/tokenizer/``_bg_trainer``/
``extraction`` references (and the extraction pipeline's own model
reference); and both graph-tier passes (``GraphTierRefiner.run_normalization``
/ ``run_enrichment``) skip cleanly on a released loop instead of raising.
Together these are what lets a cloud-only server hold ~0 GiB of base-model
VRAM after release — see the base-model-holder invariant documented on
``paramem.server.app._release_base_model_in_process``.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from paramem.training.consolidation import ConsolidationLoop
from paramem.training.graph_tier import GraphTierRefiner
from paramem.utils.config import ConsolidationConfig

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


@contextmanager
def _noop_gpu_lock():
    """No-op context manager replacing gpu_lock_sync for unit tests."""
    yield


def _make_bt_for_close(tmp_path: Path):
    """Return a BackgroundTrainer configured for close() tests."""
    from paramem.server.background_trainer import BackgroundTrainer
    from paramem.utils.config import TrainingConfig

    model = MagicMock()
    model.peft_config = {"episodic": MagicMock(), "in_training": MagicMock()}
    return BackgroundTrainer(
        model=model,
        tokenizer=MagicMock(),
        training_config=TrainingConfig(
            num_epochs=1,
            gradient_checkpointing=False,
            batch_size=1,
        ),
        output_dir=str(tmp_path),
    )


class TestBackgroundTrainerClose:
    """BackgroundTrainer.close() stops the callable worker thread cleanly."""

    def test_close_on_fresh_trainer_is_noop(self, tmp_path):
        """close() on a freshly-constructed trainer that has never submitted a job succeeds.

        No worker thread has been started; close() must not raise and must not
        block for the full timeout.
        """
        bt = _make_bt_for_close(tmp_path)
        # No submit() called — _worker_thread is None.
        assert bt._worker_thread is None
        bt.close()  # must not raise

    def test_close_after_submit_and_wait_joins_worker(self, tmp_path):
        """close() after submit_and_wait stops the callable-worker thread.

        After submit_and_wait the worker is alive (persistent daemon).  close()
        must send the stop sentinel and join the thread within the timeout so
        the thread is no longer alive.
        """
        bt = _make_bt_for_close(tmp_path)
        job_ran = threading.Event()

        def _job():
            job_ran.set()

        with patch("paramem.server.gpu_lock.gpu_lock_sync", new=_noop_gpu_lock):
            bt.submit_and_wait(_job)

        assert job_ran.is_set(), "Job must have run before close() is tested"
        worker = bt._worker_thread
        assert worker is not None and worker.is_alive(), (
            "Worker thread must be alive before close()"
        )

        bt.close(timeout=5.0)

        assert not worker.is_alive(), "Worker thread must be dead after close()"

    def test_close_is_idempotent(self, tmp_path):
        """Calling close() twice does not raise.

        After the first close() the worker has exited.  A second close() on
        the same instance must be a no-op (idempotent).
        """
        bt = _make_bt_for_close(tmp_path)
        job_done = threading.Event()

        def _job():
            job_done.set()

        with patch("paramem.server.gpu_lock.gpu_lock_sync", new=_noop_gpu_lock):
            bt.submit_and_wait(_job)

        assert job_done.is_set()

        bt.close(timeout=5.0)
        bt.close(timeout=5.0)  # must not raise

    def test_release_nulls_model_tokenizer_and_thread(self, tmp_path):
        """release() stops the worker and drops model/tokenizer/_worker_thread.

        After submit_and_wait the worker is alive.  release() must join the
        thread (via _stop_callable_worker), null _worker_thread, null model,
        and null tokenizer so no live attribute retains the base-model reference.
        """
        bt = _make_bt_for_close(tmp_path)
        job_ran = threading.Event()

        def _job():
            job_ran.set()

        with patch("paramem.server.gpu_lock.gpu_lock_sync", new=_noop_gpu_lock):
            bt.submit_and_wait(_job)

        assert job_ran.is_set(), "Job must have run before release() is tested"
        assert bt._worker_thread is not None, "Worker must be alive before release()"

        bt.release()

        assert bt.model is None, "release() must null model"
        assert bt.tokenizer is None, "release() must null tokenizer"
        assert bt._worker_thread is None, "release() must null _worker_thread"
        assert bt._current_job is None, "release() must null _current_job"

    def test_release_on_fresh_trainer_is_noop(self, tmp_path):
        """release() on a freshly-constructed trainer (no worker started) does not raise."""
        bt = _make_bt_for_close(tmp_path)
        assert bt._worker_thread is None
        bt.release()  # must not raise
        assert bt.model is None
        assert bt.tokenizer is None
        assert bt._worker_thread is None


# ---------------------------------------------------------------------------
# TestConsolidationLoopRelease
# ---------------------------------------------------------------------------


class TestConsolidationLoopRelease:
    """ConsolidationLoop.release() drops all base-model references."""

    def test_release_nulls_model_extraction_and_bg_trainer(self):
        """release() nulls model, tokenizer, _bg_trainer, and extraction.model.

        Uses a bare ConsolidationLoop instance (no __init__) with sentinel
        objects injected directly, matching the pattern in other bare-loop tests.
        """
        from paramem.graph.extraction_pipeline import ExtractionPipeline
        from paramem.training.consolidation import ConsolidationLoop

        sentinel_model = MagicMock(name="base_model")
        sentinel_tokenizer = MagicMock(name="tokenizer")
        sentinel_bt = MagicMock(name="bg_trainer")

        # Build a minimal ExtractionPipeline with the sentinel model.
        ep = ExtractionPipeline.__new__(ExtractionPipeline)
        ep.model = sentinel_model
        ep.tokenizer = sentinel_tokenizer

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.model = sentinel_model
        loop.tokenizer = sentinel_tokenizer
        loop._bg_trainer = sentinel_bt
        loop.extraction = ep

        loop.release()

        assert loop.model is None, "release() must null loop.model"
        assert loop.tokenizer is None, "release() must null loop.tokenizer"
        assert loop._bg_trainer is None, "release() must null loop._bg_trainer"
        assert loop.extraction is None, "release() must null loop.extraction"
        # The ExtractionPipeline's own model reference must also be cleared
        # before the pipeline is dropped.
        assert ep.model is None, "release() must null extraction.model before clearing extraction"

    def test_release_without_extraction_is_noop(self):
        """release() tolerates a loop with no extraction attribute."""
        from paramem.training.consolidation import ConsolidationLoop

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.model = MagicMock(name="model")
        loop.tokenizer = MagicMock(name="tokenizer")
        loop._bg_trainer = None
        # No loop.extraction set.

        loop.release()  # must not raise

        assert loop.model is None
        assert loop.tokenizer is None

    def test_release_is_idempotent(self):
        """Calling release() twice does not raise."""
        from paramem.training.consolidation import ConsolidationLoop

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.model = MagicMock(name="model")
        loop.tokenizer = MagicMock(name="tokenizer")
        loop._bg_trainer = None
        loop.extraction = None

        loop.release()
        loop.release()  # must not raise


class TestGraphTierSkipsAfterRelease:
    """Both graph-tier passes SKIP on a released loop instead of raising.

    ``release()`` nulls ``model`` AND ``extraction`` together, so the
    cloud-only server routinely holds a loop whose ``extraction`` is ``None``.
    Both passes own a ``model is None`` early-skip, and that skip must stay
    reachable WITHOUT any read of ``self.extraction``.

    The extraction config is handed over as a deferred read
    (``_current_extraction_config``) that only the post-guard paths invoke,
    never a resolved constructor argument evaluated before the guard runs —
    that would read ``self.extraction.config`` and raise ``AttributeError``
    on a released loop's ``None`` extraction. These tests fail with
    ``AttributeError`` if that read is ever hoisted back above the guard.
    """

    @staticmethod
    def _released_loop(tmp_path: Path) -> ConsolidationLoop:
        """A loop in exactly the post-``release()`` state, with nothing re-added."""
        from paramem.graph.merger import GraphMerger

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.output_dir = tmp_path
        loop.config = ConsolidationConfig()
        loop.cloud_enabled = True
        loop.save_cycle_snapshots = False
        loop._debug_base = None
        loop.graph_enrichment_neighborhood_hops = 2
        loop.graph_enrichment_max_entities_per_pass = 50
        loop.merger = GraphMerger(model=None)
        loop._incidents_state_dir = None
        # Post-release state: release() nulls these four together.
        loop.model = None
        loop.tokenizer = None
        loop._bg_trainer = None
        loop.extraction = None
        return loop

    @staticmethod
    def _refiner(loop: ConsolidationLoop) -> GraphTierRefiner:
        """Build a :class:`GraphTierRefiner` exactly as ``build_tier_refiner``
        does, off a released loop's current (post-``release()``) state."""
        return GraphTierRefiner(
            loop.merger,
            model=loop.model,
            tokenizer=loop.tokenizer,
            extraction_config_provider=loop._current_extraction_config,
            cloud_enabled=loop.cloud_enabled,
            neighborhood_hops=loop.graph_enrichment_neighborhood_hops,
            max_entities_per_pass=loop.graph_enrichment_max_entities_per_pass,
            gc_disable=loop._disable_gradient_checkpointing,
            gc_enable=loop._enable_gradient_checkpointing,
        )

    def test_normalization_skips_on_released_loop(self, tmp_path):
        """Normalization returns the no_model skip, never touching extraction.

        ``loop.cloud_enabled = True`` is deliberate: it is the only branch in the
        normalization pass that reads the extraction config, so a hoisted read
        cannot hide behind a False gate here.
        """
        loop = self._released_loop(tmp_path)

        result = self._refiner(loop).run_normalization()

        assert result["skipped"] is True
        assert result["skip_reason"] == "no_model"
        assert loop.extraction is None, "the skip path must not repopulate extraction"

    def test_enrichment_skips_on_released_loop(self, tmp_path):
        """Enrichment returns the no_model skip, never touching extraction."""
        loop = self._released_loop(tmp_path)

        result = self._refiner(loop).run_enrichment()

        assert result["skipped"] is True
        assert result["skip_reason"] == "no_model"
        assert loop.extraction is None, "the skip path must not repopulate extraction"
