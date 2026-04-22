"""Tests for GraphMerger.save_bytes() (Slice 3b.2, Correction 3).

Verifies that save_bytes() and save_graph() produce byte-identical output.
"""

from __future__ import annotations

import json

from paramem.graph.merger import GraphMerger


class TestGraphMergerSaveBytes:
    def test_graph_merger_save_bytes_matches_save_graph(self, tmp_path):
        """save_graph(tmp) + read_bytes() == save_bytes()."""
        merger = GraphMerger()
        # Add a node and edge to give the graph some content.
        merger.graph.add_node("Alice", entity_type="person", attributes={})
        merger.graph.add_node("London", entity_type="location", attributes={})
        merger.graph.add_edge("Alice", "London", key=0, predicate="lives_in", recurrence_count=1)

        # save_graph to a temp file.
        graph_file = tmp_path / "graph.json"
        merger.save_graph(graph_file)
        on_disk_bytes = graph_file.read_bytes()

        # save_bytes in memory.
        in_memory_bytes = merger.save_bytes()

        assert on_disk_bytes == in_memory_bytes

    def test_save_bytes_empty_graph(self):
        """Empty graph produces valid JSON bytes."""
        merger = GraphMerger()
        b = merger.save_bytes()
        assert isinstance(b, bytes)
        data = json.loads(b)
        assert "nodes" in data

    def test_save_bytes_is_utf8(self):
        """save_bytes returns UTF-8-encoded JSON."""
        merger = GraphMerger()
        b = merger.save_bytes()
        # Should decode without error.
        text = b.decode("utf-8")
        assert text
