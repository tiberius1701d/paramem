"""ParaMem GPU consumer adapter for the gpu_guard arbitration layer.

Post-V2.5: all detection / release / idle / describe logic for the
paramem-server process lives in ``~/.config/gpu-guard/config.toml`` under
``[consumers.paramem-server]`` and is handled by the config-driven
``ConfigConsumer`` that gpu_guard auto-loads at startup.

This module only owns the paramem-internal contract for systemd env vars:
stamping ``PARAMEM_HOLD_*`` on GPU acquire, and clearing them on release.

Module-level ``consumer = adapter`` is kept as a backward-compatible alias so
any caller still using ``--consumer paramem.gpu_consumer:consumer`` continues
to get the env-stamp adapter (which is the correct behaviour — detection and
release are now handled by the auto-loaded ConfigConsumer).
"""

from __future__ import annotations

import subprocess
import time


class ParamemEnvStampAdapter:
    """Adapter that adds PARAMEM_HOLD_* env stamping to the config-driven
    paramem-server consumer.

    All detection / release / idle / describe behavior is delegated to the
    ``ConfigConsumer`` loaded from ``~/.config/gpu-guard/config.toml``.  This
    class only owns the paramem-internal contract for systemd env vars.

    Registered as a Consumer so its ``on_acquired`` / ``on_released`` hooks
    fire alongside the config-driven consumer.  ``find_pids`` returns ``[]``
    so it never classifies a PID — the config consumer claims the server
    first.

    Attributes:
        name: Consumer name used for registration.
        default_priority: Lower = stronger; kept at 5 to match the TOML entry.
        non_evictable_without_confirm: False — eviction does not require
            operator confirmation (the server can always switch to cloud-only).
    """

    name = "paramem-env-stamp"
    default_priority = 5
    non_evictable_without_confirm = False

    def find_pids(self, candidate_pids: list[int]) -> list[int]:
        """Return empty list — PID classification is delegated to ConfigConsumer.

        Args:
            candidate_pids: PIDs currently using the GPU.

        Returns:
            Always an empty list.
        """
        return []

    def is_idle(self) -> bool:
        """Return True — idle check is delegated to ConfigConsumer.

        Returns:
            Always True.
        """
        return True

    def request_release(self, pid: int) -> None:
        """No-op — release is delegated to ConfigConsumer.

        Args:
            pid: Unused.
        """

    def wait_for_idle(self, pid: int, timeout: int) -> bool:
        """Return True — wait-for-idle is delegated to ConfigConsumer.

        Args:
            pid: Unused.
            timeout: Unused.

        Returns:
            Always True.
        """
        return True

    def describe(self, pid: int) -> str:
        """Return empty string — describe is delegated to ConfigConsumer.

        Args:
            pid: Unused.

        Returns:
            Always an empty string.
        """
        return ""

    def on_acquired(self, own_pid: int, argv: list[str]) -> None:
        """Stamp PARAMEM_HOLD_* systemd env vars so pstatus and auto-reclaim work.

        PARAMEM_EXTRA_ARGS=--defer-model is set by training-control.sh before
        this process launches; we add the identity fields (PID, timestamp, hint).
        Cleared in ``on_released``.

        Args:
            own_pid: PID of the acquiring ML process.
            argv: sys.argv of the acquiring process.
        """
        from paramem.utils.gpu_hold import format_cmd_hint

        hint = format_cmd_hint(argv)
        stamp_args = [
            "systemctl",
            "--user",
            "set-environment",
            f"PARAMEM_HOLD_PID={own_pid}",
            f"PARAMEM_HOLD_STARTED_AT={int(time.time())}",
        ]
        if hint:
            stamp_args.append(f"PARAMEM_HOLD_CMD={hint}")
        try:
            subprocess.run(stamp_args, check=False, capture_output=True, timeout=5)
        except Exception:
            pass

    def on_released(self) -> None:
        """Clear PARAMEM_HOLD_* and PARAMEM_EXTRA_ARGS from the systemd environment.

        Training has stopped.  Drop the defer-model flag before the next
        auto-reclaim tick loads the model in local mode.  SIGKILL skips this
        hook; auto-reclaim's GPU-free check handles that case independently.
        """
        try:
            subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "unset-environment",
                    "PARAMEM_EXTRA_ARGS",
                    "PARAMEM_HOLD_PID",
                    "PARAMEM_HOLD_STARTED_AT",
                    "PARAMEM_HOLD_CMD",
                ],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass


# Module-level pre-instantiated adapter.
adapter = ParamemEnvStampAdapter()

# Backward-compatible alias for ``--consumer paramem.gpu_consumer:consumer``.
consumer = adapter
