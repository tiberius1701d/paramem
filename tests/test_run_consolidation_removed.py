"""Regression guard: run_consolidation must not exist in consolidation.py.

After the voice-pipeline-switch commit deleted run_consolidation from
paramem.server.consolidation, any re-introduction of the function must
fail this test explicitly rather than silently regressing call sites.
Its logic lives in paramem.server.app._run_extraction_phase.
"""


def test_run_consolidation_not_in_consolidation_module():
    """paramem.server.consolidation must not export run_consolidation.

    The function was deleted when extraction was moved to _run_extraction_phase
    in paramem.server.app.  Re-introducing it would bypass the app-level
    state plumbing that reads config and session_buffer from _state.
    """
    import paramem.server.consolidation as _m

    assert not hasattr(_m, "run_consolidation"), (
        "run_consolidation was deleted; its logic lives in "
        "paramem.server.app._run_extraction_phase. "
        "Do not reintroduce run_consolidation; update callers to use "
        "_run_extraction_phase instead."
    )


def test_run_extraction_phase_exists_in_app():
    """paramem.server.app must export _run_extraction_phase."""
    import paramem.server.app as _app

    assert hasattr(_app, "_run_extraction_phase"), (
        "_run_extraction_phase replaced run_consolidation; "
        "it must exist in paramem.server.app and be callable."
    )
    assert callable(_app._run_extraction_phase)
