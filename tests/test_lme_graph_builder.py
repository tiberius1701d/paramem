"""``experiments/lme_graph_builder.py``'s ``--resume`` wiring: the merger's
graph is populated from the on-disk snapshot via ``load_memory_from_disk``,
and the loop is built with a fresh ``MemoryStore``.

No GPU: the base model load, the GPU-lock/guard context, the LongMemEval
dataset loader and the speaker-name pool are stubbed so ``main()`` runs its
resume wiring against an empty session set. ``gpu_guard`` is a separate
lab-tools package outside this project's dependencies, so its module and
the ``experiments.utils.gpu_guard`` wrapper are stubbed into ``sys.modules``
directly via :func:`tests._gpu_guard_stub.stub_gpu_guard`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx
import pytest

import experiments.lme_graph_builder as lme_graph_builder
import experiments.utils.longmemeval_loader as longmemeval_loader
import experiments.utils.production as production
import experiments.utils.speaker_names as speaker_names
import experiments.utils.test_harness as test_harness
from tests._gpu_guard_stub import stub_gpu_guard as _stub_gpu_guard


class _EmptySessionLoader:
    """Stand-in ``LongMemEvalLoader`` that yields no sessions.

    Isolates the resume/model-load wiring under test from the real
    HuggingFace dataset fetch.
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def iter_sessions(self, *, limit=None, speaker_name_pool=None):
        return iter(())


class _StubSpeakerNamePool:
    def __init__(self, seed: int) -> None:
        self.seed = seed


def _parsed_args(*, output: Path, resume: bool) -> argparse.Namespace:
    return argparse.Namespace(
        lme_split="longmemeval_oracle",
        lme_seed=42,
        target_keys=None,
        persist_every=10,
        model="mistral",
        output=output,
        resume=resume,
        with_cloud=False,
    )


class TestResumeLoadsSnapshotIntoMerger:
    """``--resume`` reads the existing snapshot into ``loop.merger.graph``
    via ``load_memory_from_disk`` and builds the loop with a ``MemoryStore``.
    """

    def _run_main(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, resume: bool):
        _stub_gpu_guard(monkeypatch)
        monkeypatch.setattr(
            lme_graph_builder, "parse_args", lambda: _parsed_args(output=tmp_path, resume=resume)
        )
        monkeypatch.setattr(lme_graph_builder, "wait_for_cooldown", lambda *a, **k: None)
        monkeypatch.setattr(test_harness, "load_test_env", lambda: None)
        monkeypatch.setattr(longmemeval_loader, "LongMemEvalLoader", _EmptySessionLoader)
        monkeypatch.setattr(speaker_names, "SpeakerNamePool", _StubSpeakerNamePool)
        monkeypatch.setattr(
            production, "load_base_model", lambda *a, **k: (MagicMock(), MagicMock())
        )

        fake_loop = MagicMock()
        fake_loop.merger.graph = nx.MultiDiGraph()
        mock_create_loop = MagicMock(return_value=fake_loop)
        monkeypatch.setattr(production, "create_consolidation_loop", mock_create_loop)

        resumed_graph = nx.MultiDiGraph()
        mock_load_memory = MagicMock(return_value=resumed_graph)
        monkeypatch.setattr(production, "load_memory_from_disk", mock_load_memory)

        original_graph = fake_loop.merger.graph
        lme_graph_builder.main()
        return fake_loop, mock_create_loop, mock_load_memory, resumed_graph, original_graph

    def test_resume_assigns_loaded_graph_via_load_memory_from_disk(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        snapshot_path = tmp_path / "graph_snapshot.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text("{}")

        fake_loop, _mock_create_loop, mock_load_memory, resumed_graph, _original_graph = (
            self._run_main(monkeypatch, tmp_path, resume=True)
        )

        mock_load_memory.assert_called_once_with(snapshot_path)
        assert fake_loop.merger.graph is resumed_graph

    def test_resume_without_existing_snapshot_never_calls_load_memory_from_disk(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``--resume`` with no prior snapshot on disk starts fresh — the
        merger's graph is left as ``create_consolidation_loop`` built it: no
        call to ``load_memory_from_disk``, and ``loop.merger.graph`` stays
        the same object it was before ``main()`` ran."""
        fake_loop, _mock_create_loop, mock_load_memory, _resumed_graph, original_graph = (
            self._run_main(monkeypatch, tmp_path, resume=True)
        )

        mock_load_memory.assert_not_called()
        assert fake_loop.merger.graph is original_graph

    def test_loop_is_built_with_a_memory_store(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``create_consolidation_loop`` receives a ``MemoryStore`` instance
        as ``memory_store``."""
        from paramem.memory.store import MemoryStore

        snapshot_path = tmp_path / "graph_snapshot.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text("{}")

        _fake_loop, mock_create_loop, _mock_load_memory, _resumed_graph, _original_graph = (
            self._run_main(monkeypatch, tmp_path, resume=True)
        )

        mock_create_loop.assert_called_once()
        passed_store = mock_create_loop.call_args.kwargs["memory_store"]
        assert isinstance(passed_store, MemoryStore)
