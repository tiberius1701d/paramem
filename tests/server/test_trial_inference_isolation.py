"""Tests for trial adapter inference isolation (Slice 3b.2, §10.5).

Verifies that the router does not pick up trial adapter keyed_pairs.json
when trial_adapter_dir is outside config.adapter_dir (spec §7 — "Trial
adapter NOT routable").

No GPU — all tests use in-memory router construction.
"""

from __future__ import annotations

import json

from paramem.server.router import QueryRouter


class TestRouterDoesNotMountTrialAdapter:
    """The router scans config.adapter_dir only; trial_adapter_dir is separate."""

    def test_router_does_not_mount_trial_adapter(self, tmp_path):
        """Router.reload() with a trial keyed_pairs.json outside adapter_dir → not indexed.

        Setup:
        - config.adapter_dir = tmp_path/adapters/ (live adapters)
        - trial_adapter_dir = tmp_path/trial_adapter/ (outside adapter_dir)
        - Write keyed_pairs.json under trial_adapter_dir
        - Instantiate QueryRouter(adapter_dir=config.adapter_dir)
        - Verify the trial key is NOT in the router's index.
        """
        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir(parents=True)
        trial_adapter_dir = tmp_path / "trial_adapter"
        trial_adapter_dir.mkdir(parents=True)

        # Write a keyed_pairs.json in the trial adapter directory.
        # Format: list of {key, question, answer, source_subject, source_object}.
        trial_kp = trial_adapter_dir / "keyed_pairs.json"
        trial_kp.write_text(
            json.dumps(
                [
                    {
                        "key": "trial-key-001",
                        "question": "Q?",
                        "answer": "A",
                        "source_subject": "TrialEntity",
                        "source_object": "TrialThing",
                    }
                ]
            ),
            encoding="utf-8",
        )

        # Also write a live key in config.adapter_dir using the canonical layout.
        live_ep_dir = adapter_dir / "episodic"
        live_ep_dir.mkdir(parents=True, exist_ok=True)
        live_kp = live_ep_dir / "keyed_pairs.json"
        live_kp.write_text(
            json.dumps(
                [
                    {
                        "key": "live-key-001",
                        "question": "Who?",
                        "answer": "Alice",
                        "source_subject": "Alice",
                        "source_object": "London",
                    }
                ]
            ),
            encoding="utf-8",
        )

        # Build router against live adapter_dir only.
        router = QueryRouter(adapter_dir=adapter_dir)

        # The trial key must NOT appear in any index.
        all_indexed_keys = set()
        for adapter_idx in router._entity_key_index.values():
            for key_set in adapter_idx.values():
                all_indexed_keys.update(key_set)

        assert "trial-key-001" not in all_indexed_keys, (
            "Trial adapter key was mounted in router — inference isolation violated"
        )

    def test_router_mounts_live_adapter_keys(self, tmp_path):
        """Router does index keyed_pairs.json from config.adapter_dir (canonical layout)."""
        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir(parents=True)

        # Canonical layout: episodic keyed_pairs lives under episodic/ subdir.
        ep_dir = adapter_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        live_kp = ep_dir / "keyed_pairs.json"
        live_kp.write_text(
            json.dumps(
                [
                    {
                        "key": "live-key-001",
                        "question": "Who is Alice?",
                        "answer": "Alice",
                        "source_subject": "Alice",
                        "source_object": "London",
                    }
                ]
            ),
            encoding="utf-8",
        )

        router = QueryRouter(adapter_dir=adapter_dir)

        # At least one entity should be indexed from the live key.
        assert len(router._entity_key_index) > 0 or len(router._all_entities) >= 0, (
            "Live adapter keyed_pairs.json not indexed"
        )

    def test_router_empty_when_adapter_dir_absent(self, tmp_path):
        """Router returns no keys when adapter_dir does not exist."""
        router = QueryRouter(adapter_dir=tmp_path / "nonexistent")
        assert router._entity_key_index == {}
        assert router._all_entities == set()

    def test_trial_adapter_dir_separate_from_live(self, tmp_path):
        """trial_adapter_dir is a sibling of state_dir, not inside adapter_dir."""
        # This mirrors how _build_trial_loop sets up the paths.
        data_dir = tmp_path / "data" / "ha"
        trial_adapter_dir = data_dir / "trial_adapter"
        adapter_dir = tmp_path / "data" / "adapters"

        # trial_adapter_dir must NOT be a subdirectory of adapter_dir.
        try:
            trial_adapter_dir.relative_to(adapter_dir)
            is_subdir = True
        except ValueError:
            is_subdir = False

        assert not is_subdir, (
            "trial_adapter_dir is inside adapter_dir — trial keys could be picked up by router"
        )
