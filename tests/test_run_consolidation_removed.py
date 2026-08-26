"""Structural guard: paramem.server.consolidation has no run_consolidation.

Extraction is invoked through paramem.server.app._run_extraction_phase, which
reads config and session_buffer from _state. A run_consolidation function on
paramem.server.consolidation would bypass that state plumbing, so this test
fails loudly if one is ever (re)introduced.
"""


def test_run_consolidation_not_in_consolidation_module():
    """paramem.server.consolidation must not export run_consolidation.

    Extraction routes exclusively through
    paramem.server.app._run_extraction_phase, which reads config and
    session_buffer from _state.
    """
    import paramem.server.consolidation as _m

    assert not hasattr(_m, "run_consolidation"), (
        "run_consolidation must not exist; extraction routes through "
        "paramem.server.app._run_extraction_phase. "
        "Do not add run_consolidation; update callers to use "
        "_run_extraction_phase instead."
    )


def test_run_extraction_phase_exists_in_app():
    """paramem.server.app must export _run_extraction_phase."""
    import paramem.server.app as _app

    assert hasattr(_app, "_run_extraction_phase"), (
        "_run_extraction_phase is the entry point for extraction; "
        "it must exist in paramem.server.app and be callable."
    )
    assert callable(_app._run_extraction_phase)
