"""Canonical I/O facade for ``keyed_pairs.json``.

All on-disk reads and writes of the keyed-pair file MUST go through
this module so the schema is preserved by construction.  Hand-rolled
``{"key": ..., "question": ..., "answer": ...}`` dicts are the bug
this facade replaces.

Schema (canonical contract, single mandatory tier)
--------------------------------------------------
Every keyed pair persisted to disk MUST contain all eight fields:

    key (str), question (str), answer (str),
    source_subject (str), source_predicate (str), source_object (str),
    speaker_id (str), first_seen_cycle (int)

There is no optional bucket.  A producer that cannot supply every
field is bugged — the facade raises ``KeyError`` at write time so the
defect surfaces where it originates, not silently downstream.

Adding a future field: append to ``KEYED_PAIR_FIELDS`` AND update
every producer site.  The facade test exercises the round-trip; CI
catches a producer that forgets the new field.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Mapping

from paramem.backup.encryption import read_maybe_encrypted, write_infra_bytes

logger = logging.getLogger(__name__)

KEYED_PAIR_FIELDS: tuple[str, ...] = (
    "key",
    "question",
    "answer",
    "source_subject",
    "source_predicate",
    "source_object",
    "speaker_id",
    "first_seen_cycle",
)


def _normalise_pair(src: Mapping) -> dict:
    """Return a dict with every canonical field copied from *src*.

    Raises ``KeyError`` if any required field is missing — the facade
    contract is one mandatory tier, no exceptions.

    Args:
        src: Source mapping that must contain every field in
            ``KEYED_PAIR_FIELDS``.

    Returns:
        New dict containing exactly the canonical fields in declaration order.

    Raises:
        KeyError: When any required field is absent from *src*.
    """
    return {k: src[k] for k in KEYED_PAIR_FIELDS}


def write_keyed_pairs(path: Path, pairs: Iterable[Mapping]) -> None:
    """Write *pairs* to *path*, encryption-aware, schema-enforcing.

    Each pair must contain every field in ``KEYED_PAIR_FIELDS``.
    Missing fields raise ``KeyError`` — by design, so the defect
    surfaces at the writer, not silently in a downstream consumer.

    The file is written atomically via :func:`write_infra_bytes`: the
    content is encrypted when the daily age identity is loaded, and
    written as plaintext otherwise.  The parent directory is created
    if it does not exist.

    Args:
        path: Destination path for the ``keyed_pairs.json`` file.
        pairs: Iterable of pair mappings, each with all canonical fields.

    Raises:
        KeyError: When any pair is missing a required field.
        OSError: On filesystem errors.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = [_normalise_pair(p) for p in pairs]
    payload = json.dumps(serialised, indent=2).encode("utf-8")
    write_infra_bytes(path, payload)


def read_keyed_pairs(path: Path) -> list[dict]:
    """Read *path*, decrypt if needed, return list of pair dicts.

    Returns an empty list when the file is absent.  Raises on JSON,
    decode, or I/O errors so callers see fail-fast — they may catch
    and downgrade to a warning, as several existing call-sites already
    do; preserve that pattern at the call site, not here.

    Validates that every entry contains every canonical field; raises
    ``ValueError`` on any partial entry.

    Args:
        path: Source path for the ``keyed_pairs.json`` file.

    Returns:
        List of pair dicts, each containing all ``KEYED_PAIR_FIELDS``.
        Empty list when the file is absent.

    Raises:
        ValueError: When the file does not contain a JSON array, or when
            any entry is missing a required field.
        json.JSONDecodeError: When the file contains invalid JSON.
        OSError: On filesystem errors.
        UnicodeDecodeError: When the file bytes cannot be decoded as UTF-8.
    """
    path = Path(path)
    if not path.exists():
        return []
    raw = read_maybe_encrypted(path)
    if not raw:
        return []
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON array, got {type(data).__name__}")
    for i, entry in enumerate(data):
        missing = [k for k in KEYED_PAIR_FIELDS if k not in entry]
        if missing:
            raise ValueError(f"{path}[{i}]: missing required fields {missing}")
    return data
