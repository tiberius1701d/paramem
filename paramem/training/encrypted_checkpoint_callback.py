"""HuggingFace Trainer callback that encrypts checkpoint files in place.

Used by every HF-Trainer code path the project exposes
(:func:`~paramem.training.trainer.train_adapter` and
:class:`~paramem.server.background_trainer.BackgroundTrainer`) so that
plaintext ``adapter_model.safetensors`` files never persist under
``data/ha/adapters/`` past a single ``on_save`` call.  Without this
callback, HF Trainer's ``checkpoint-<step>/`` directories land plaintext
inside the adapter root and trip the startup mode-consistency check
(:func:`paramem.backup.encryption.assert_mode_consistency`) on the next
boot, refusing the server.

Behaviour:

* ``on_save`` — walks every ``checkpoint-*`` subdir under
  ``args.output_dir`` and encrypts any plaintext files via
  :func:`paramem.backup.checkpoint_shard.encrypt_checkpoint_dir`.
  Already-wrapped files are left alone.  No-op when the daily age
  identity is not loadable (Security OFF posture).

* ``on_train_begin`` — refuses to start when
  ``load_best_model_at_end=True`` is combined with Security ON: HF
  reads checkpoint files directly at end-of-training, bypassing the
  decrypt-to-shm resume path, and would fail on age-wrapped files.
  Failing fast prevents silent corruption.
"""

from __future__ import annotations

import logging
from pathlib import Path

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class EncryptCheckpointCallback(TrainerCallback):
    """Encrypt HF Trainer's ``checkpoint-*/`` writes immediately after save."""

    def on_train_begin(self, args, state, control, **kwargs):
        from paramem.backup import key_store as _ks

        if getattr(args, "load_best_model_at_end", False) and _ks.daily_identity_loadable(
            _ks.DAILY_KEY_PATH_DEFAULT
        ):
            raise RuntimeError(
                "load_best_model_at_end=True is incompatible with Security ON: "
                "HF reads checkpoint files directly at end-of-training, "
                "bypassing the decrypt-to-shm path. Either disable "
                "load_best_model_at_end or unset PARAMEM_DAILY_PASSPHRASE."
            )

    def on_save(self, args, state, control, **kwargs):
        from paramem.backup import key_store as _ks
        from paramem.backup.checkpoint_shard import encrypt_checkpoint_dir

        if not _ks.daily_identity_loadable(_ks.DAILY_KEY_PATH_DEFAULT):
            return
        output_dir = Path(args.output_dir)
        for ckpt in output_dir.glob("checkpoint-*"):
            if not ckpt.is_dir():
                continue
            try:
                n = encrypt_checkpoint_dir(ckpt)
                if n > 0:
                    logger.debug("Encrypted %d checkpoint files in %s", n, ckpt)
            except Exception:
                logger.exception("Failed to encrypt checkpoint files in %s", ckpt)
