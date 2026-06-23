"""Regression tests for scripts/dev/calibrate_prompts.py.

Specifically guards the NameError that occurred when --stages normalize was
invoked with an empty chunk loop: params_base was assigned inside the
for-chunk loop and therefore never bound when chunks == [].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the script importable without installing it as a package.
_SCRIPTS_DEV = Path(__file__).resolve().parents[1] / "scripts" / "dev"
if str(_SCRIPTS_DEV) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DEV))

import calibrate_prompts  # noqa: E402 (scripts/dev is not a package)

_CANNED_NORMALIZE_RESPONSE = {
    "filtered": [],
    "merged": [],
    "filter_prompt_used": "normalize_filter.txt",
    "merge_prompt_used": "normalize_merge.txt",
    "raw_output": "[]",
}


class TestNormalizeStageNoNameError:
    """--stages normalize must not raise NameError when chunks is empty."""

    def test_normalize_writes_output_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Invoke main(['--stages','normalize',...]) with _post_stage mocked.

        Asserts:
        - No NameError (params_base is bound before the chunk loop)
        - 05_normalize.json is written to the dump directory
        - The output file contains the expected 'stage' key
        """
        # Minimal snapshot file — content is opaque to the harness (passed as a
        # path string to the server endpoint, not parsed locally).
        snapshot = tmp_path / "graph_merged_snapshot.json"
        snapshot.write_text(
            json.dumps(
                {
                    "directed": False,
                    "multigraph": False,
                    "graph": {},
                    "nodes": [
                        {
                            "id": "alice",
                            "attributes": {"name": "Alice"},
                            "speaker_id": "Speaker0",
                        }
                    ],
                    "links": [],
                }
            )
        )

        dump_dir = tmp_path / "dump"
        dump_dir.mkdir()

        # Use the real configs/prompts dir so --prompts-dir validation passes.
        real_prompts_dir = Path(__file__).resolve().parents[1] / "configs" / "prompts"

        argv = [
            "--stages",
            "normalize",
            "--snapshot",
            str(snapshot),
            "--dump-dir",
            str(dump_dir),
            "--server",
            "http://localhost:8420",
            "--prompts-dir",
            str(real_prompts_dir),
            "--baseline",
            "none",
        ]

        with patch.object(
            calibrate_prompts, "_post_stage", return_value=_CANNED_NORMALIZE_RESPONSE
        ):
            rc = calibrate_prompts.main(argv)

        assert rc == 0, f"Expected rc=0, got {rc}"

        out_file = dump_dir / "05_normalize.json"
        assert out_file.exists(), "05_normalize.json was not written"

        blob = json.loads(out_file.read_text())
        assert blob["stage"] == "normalize", f"Unexpected stage in output: {blob.get('stage')!r}"
        assert blob["snapshot_path"] == str(snapshot)

    def test_params_base_bound_before_chunk_loop(self):
        """Unit-level guard: params_base must be referenced in the module source
        BEFORE the for-chunk loop, not inside it.

        This is a structural assertion over the source text — it will catch any
        future accidental regression that re-introduces the assignment inside the
        loop.
        """
        source = Path(calibrate_prompts.__file__).read_text()
        lines = source.splitlines()

        # Find the line numbers of the two markers.
        params_base_line = None
        chunk_loop_line = None
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            next_line = lines[i] if i < len(lines) else ""
            if (
                params_base_line is None
                and stripped.startswith("params_base")
                and "temperature" in next_line
            ):
                params_base_line = i
            if stripped.startswith("for chunk in chunks:") and chunk_loop_line is None:
                chunk_loop_line = i

        assert params_base_line is not None, (
            "Could not locate 'params_base = ...' assignment in calibrate_prompts.py"
        )
        assert chunk_loop_line is not None, (
            "Could not locate 'for chunk in chunks:' loop in calibrate_prompts.py"
        )
        assert params_base_line < chunk_loop_line, (
            f"params_base (line {params_base_line}) must be assigned BEFORE "
            f"'for chunk in chunks:' (line {chunk_loop_line}). "
            "The NameError regression has been re-introduced."
        )
