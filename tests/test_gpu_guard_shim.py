"""Tests verifying that the gpu_guard shim registers its consumer and notifier
correctly and re-exports the expected public API symbols.

No GPU or real processes are used — purely structural / registration checks.
"""

from __future__ import annotations

import pytest

# gpu_guard is provided by lab-tools (separate repo, not on PyPI).  CI does
# not install it; skip the whole module rather than erroring at collection.
pytest.importorskip("gpu_guard")

from gpu_guard._core import _default_consumers, _default_notifier  # noqa: E402


class TestShimRegistration:
    def test_paramem_consumer_registered_as_default(self):
        """Importing the shim registers ParamemEnvStampAdapter as a default consumer."""
        # Import the shim; its module-level code registers the adapter.
        import experiments.utils.gpu_guard  # noqa: F401

        consumer_names = [c.name for c in _default_consumers]
        assert "paramem-env-stamp" in consumer_names

    def test_paramem_notifier_registered_as_default(self):
        """Importing the shim installs _ParamemNotifier as the default notifier."""
        import experiments.utils.gpu_guard  # noqa: F401

        # _ParamemNotifier is a local class in the shim; verify it has the
        # expected methods (duck-type check).
        n = _default_notifier
        assert hasattr(n, "started")
        assert hasattr(n, "finished")
        assert hasattr(n, "paused")
        assert hasattr(n, "resumed")


class TestShimPublicAPI:
    def test_acquire_gpu_re_exported(self):
        from experiments.utils.gpu_guard import acquire_gpu

        assert callable(acquire_gpu)

    def test_gpu_acquire_error_re_exported(self):
        from experiments.utils.gpu_guard import GPUAcquireError

        assert issubclass(GPUAcquireError, RuntimeError)

    def test_check_gpu_re_exported(self):
        from experiments.utils.gpu_guard import check_gpu

        assert callable(check_gpu)

    def test_release_server_gpu_re_exported(self):
        from experiments.utils.gpu_guard import release_server_gpu

        assert callable(release_server_gpu)

    def test_notify_paused_re_exported(self):
        from experiments.utils.gpu_guard import notify_paused

        assert callable(notify_paused)

    def test_notify_resumed_re_exported(self):
        from experiments.utils.gpu_guard import notify_resumed

        assert callable(notify_resumed)

    def test_release_server_gpu_propagates_config_missing(self):
        """release_server_gpu() raises GPUConfigMissing when config lacks paramem-server entry."""
        import os
        from unittest.mock import patch

        from gpu_guard import GPUConfigMissing
        from gpu_guard._core import _reset_autoload_for_tests, clear_default_consumers

        from experiments.utils.gpu_guard import release_server_gpu

        clear_default_consumers()
        _reset_autoload_for_tests()
        try:
            with (
                patch.dict(os.environ, {"GPU_GUARD_NO_AUTOLOAD": "1"}),
                patch.dict(os.environ, {"GPU_GUARD_CONFIG": "/nonexistent/path/config.toml"}),
            ):
                import pytest

                with pytest.raises(GPUConfigMissing):
                    release_server_gpu()
        finally:
            # Re-register the adapter so other tests that depend on module-level
            # state are not disturbed.
            from gpu_guard import add_default_consumer

            from paramem.gpu_consumer import adapter

            add_default_consumer(adapter)
            _reset_autoload_for_tests()
