"""Tests verifying that the gpu_guard shim registers its consumer and notifier
correctly and re-exports the expected public API symbols.

No GPU or real processes are used — purely structural / registration checks.
"""

from __future__ import annotations

from gpu_guard._core import _default_consumers, _default_notifier


class TestShimRegistration:
    def test_paramem_consumer_registered_as_default(self):
        """Importing the shim registers ParamemServerConsumer as a default consumer."""
        # Import the shim; its module-level code registers the consumer.
        import experiments.utils.gpu_guard  # noqa: F401

        consumer_names = [c.name for c in _default_consumers]
        assert "paramem-server" in consumer_names

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
