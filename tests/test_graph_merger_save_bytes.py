"""Tests for GraphMerger.save_bytes().

Verifies that save_bytes() and the on-disk write produce byte-identical
output — one node-link rendering, not two.
"""

from __future__ import annotations

import json

from paramem.graph.merger import GraphMerger
from paramem.memory.persistence import save_memory_to_disk


class TestGraphMergerSaveBytes:
    def test_graph_merger_save_bytes_matches_the_on_disk_write(self, tmp_path):
        """save_memory_to_disk(tmp) + read_bytes() == save_bytes()."""
        merger = GraphMerger()
        # Add a node and edge to give the graph some content.
        merger.graph.add_node("Alice", entity_type="person", attributes={})
        merger.graph.add_node("London", entity_type="location", attributes={})
        merger.graph.add_edge("Alice", "London", key=0, predicate="lives_in", reinforcement_count=1)

        # Write to a temp file (no daily identity in tests → plaintext).
        graph_file = tmp_path / "graph.json"
        save_memory_to_disk(merger.graph, graph_file)
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
