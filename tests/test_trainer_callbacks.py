"""Tests for trainer callbacks — graceful shutdown."""

from unittest.mock import MagicMock

from paramem.training.trainer import GracefulShutdownCallback


class TestGracefulShutdownCallback:
    def test_no_stop_when_flag_false(self):
        callback = GracefulShutdownCallback(lambda: False)
        control = MagicMock()
        control.should_training_stop = False
        state = MagicMock()
        state.epoch = 5.0
        callback.on_epoch_end(None, state, control)
        assert control.should_training_stop is False

    def test_stops_when_flag_true(self):
        callback = GracefulShutdownCallback(lambda: True)
        control = MagicMock()
        control.should_training_stop = False
        state = MagicMock()
        state.epoch = 5.0
        callback.on_epoch_end(None, state, control)
        assert control.should_training_stop is True

    def test_flag_checked_dynamically(self):
        flag = [False]
        callback = GracefulShutdownCallback(lambda: flag[0])
        control = MagicMock()
        control.should_training_stop = False
        state = MagicMock()
        state.epoch = 1.0

        callback.on_epoch_end(None, state, control)
        assert control.should_training_stop is False

        flag[0] = True
        callback.on_epoch_end(None, state, control)
        assert control.should_training_stop is True
