"""GPU integration skeleton for post_session_train end-to-end recall.

All tests in this file are marked ``@pytest.mark.gpu`` and are skipped by
the standard CPU test run (``pytest -m "not gpu"``).  They are intended to
be run explicitly on hardware as part of Step 8's regression suite.

Step 8 owns the full execution of these tests.  This file exists now so the
test infrastructure is in place and the acceptance criteria are documented.
"""

from __future__ import annotations

import pytest


@pytest.mark.gpu
class TestPostSessionTrainGPUIntegration:
    """End-to-end: conversation → post_session_train → registry → probe → recall."""

    @pytest.mark.skip(reason="GPU integration owned by Step 8")
    def test_small_conversation_trains_and_recalls(self, tmp_path) -> None:
        """A small conversation transcript reaches 100% recall after training.

        Procedure:
        1. Load the Mistral NF4 model via load_base_model.
        2. Create a ConsolidationLoop configured for the server.
        3. Call loop.post_session_train(transcript, session_id, schedule='every 2h',
           max_interim_count=4).
        4. Verify result['mode'] == 'trained' and result['new_keys'] is non-empty.
        5. Call probe_keys_grouped_by_adapter for each key in result['new_keys']
           on the trained interim adapter.
        6. Assert recall rate == 1.0 via smoke_test_adapter or direct probe.

        This test must use smoke_test_adapter() from experiments/utils/test_harness.py
        (CLAUDE.md rule — never write ad-hoc inference probes).
        """
        pytest.skip("GPU integration owned by Step 8")
