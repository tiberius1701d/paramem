"""The stage ledger: the small progress record a training event's resume reads.

Its only task is recording which stages of a training event (the interim tick
or the full fold) completed, and the integrity of those stages' artifacts.  It
carries stage names, artifact paths, hashes, and the minimal provenance a
resumed dispatch has no phase-1 locals to supply — never fact content (the
assignment, the bookkeeping rows, the mints, the promotions), which lives in
the shadow artifacts the ledger's entries name.

One ledger, at most one pending event, at ``<state_dir>/stage_ledger.json``
(``state_dir`` is a fold's own state directory, e.g.
``ConsolidationLoop._fold_state_dir`` — ``output_dir.parent / "state"``).  The
file's existence *is* "an event is pending"; its ``event`` field says which
one.

Three stage kinds populate :attr:`StageLedger.stages`:

- ``"extraction"`` — written once, at phase 1's end.  Names the completed
  session ids this event consumed, the two relation counts, and the graph +
  per-tier shadow artifacts phase 1 wrote.
- ``"tier_written"`` — written once per member, as that member writes.  Carries
  the written slot directory itself (``slot``, absolute path string or
  ``None`` for a member that written no payload) alongside the artifact
  hashes — a resumed dispatch reads the slot's location from this field via
  :func:`written_slot_path`, never by scanning the tier tree for it.
- ``"tier_live"`` — written once per bundle, in one atomic call covering
  every member of that bundle, so a bundle is done or not done, never
  half-recorded.

Doneness authority is one rule, applied uniformly by :func:`verify`: an entry
is believed only while its recorded artifact hashes still verify against the
on-disk bytes they name.  No artifact-content comparison can assert or
overturn doneness on its own.

Every entry this module writes carries an ``"artifacts"`` list —
``[{"path": str, "sha256": str}, ...]``, absolute on-disk paths and the sha256
of their bytes exactly as written — so :func:`verify` never needs
out-of-band knowledge of which files an entry covers.  Verification is the
per-file re-hash walk :func:`verify` performs over that list; there is no
second, combined-hash shortcut — re-hashing every named file IS the cost,
and a single combined digest could never distinguish any divergence the
walk cannot.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

_LEDGER_FILENAME = "stage_ledger.json"
_EXTRACTION_DIRNAME = "extraction"

# The only ledger schema this build knows how to interpret and the only one
# a writer may stamp.  Bumped whenever the on-disk shape changes in a way
# :func:`_from_dict` cannot round-trip.  :data:`StageLedger.version`'s
# default derives from this single constant, so a writer can never stamp a
# version the reader's gate then refuses.
CURRENT_VERSION: int = 2
_SUPPORTED_VERSIONS: frozenset = frozenset({CURRENT_VERSION})


class StageLedgerVersionUnsupported(RuntimeError):
    """A ledger's ``version`` field is not one this build can interpret.

    Raised by :func:`read_ledger` rather than folded into its ``None``
    ("nothing pending") outcome: silently reporting no pending event over a
    record this process cannot open would let a fresh staging pass clear the
    extraction tree that record still names.  :func:`dispose` deliberately
    bypasses this gate -- discarding a record an operator cannot interpret
    is the intended escape hatch.
    """

    def __init__(self, *, path: Path, version: object) -> None:
        self.path = path
        self.version = version
        super().__init__(
            f"stage_ledger: {path} has unsupported version {version!r} "
            f"(supported: {sorted(_SUPPORTED_VERSIONS)})"
        )


@dataclass(frozen=True)
class StageLedger:
    """The parsed ledger file.  Names, hashes and stage provenance — no fact content.

    ``version`` defaults to :data:`CURRENT_VERSION` — the writer's
    construction site never hand-passes a literal, so it cannot silently
    stamp a version the reader's gate (:data:`_SUPPORTED_VERSIONS`) then
    refuses. :func:`_from_dict` (the read path) still passes the on-disk
    value explicitly, since ``read_ledger``'s version gate has already run
    by the time it is called.
    """

    event: str  # "interim" | "full" | "reconcile"
    venue: str  # "weights" | "disk"
    stamp: str
    version: int = CURRENT_VERSION
    # tier -> {"adapter", "pre_sha", "scratch"} -- "scratch" is that tier's
    # HF TrainingArguments working directory, fixed when the event is staged
    # and read verbatim by both the trainer call and every disposer (never
    # recomputed -- see dispose()'s own docstring).
    tiers: "dict[str, dict]" = field(default_factory=dict)
    stages: "tuple[dict, ...]" = ()  # "extraction" | "tier_written" | "tier_live"
    # Interim tier names this event's go-live reaps whole, fixed when the
    # event is staged (phase 1) -- every full-topology event (a full fold or
    # a reconcile) absorbs the ring.  Read verbatim on resume instead of
    # recomputed from the live store: by resume time the ring may have
    # already partially reaped, so a fresh recompute would land on the wrong
    # (smaller) set.  Empty only for an interim event.
    absorbed_interim_tiers: "tuple[str, ...]" = ()


def data_state_dir(data_dir: Path) -> Path:
    """Return ``<data_dir>/state`` — the one derivation of a data root's state directory.

    No side effects: never creates the directory, so a read-only caller
    (a resolver that only needs the path to hand elsewhere) pays no
    filesystem cost. A caller that needs the directory to exist calls
    ``.mkdir(parents=True, exist_ok=True)`` on the result itself.

    The single formula every production site derives a data root's state
    directory through — ``ConsolidationLoop``'s own fold state dir, the
    go-live ledger location, and every server/backup/CLI site that locates
    ``stage_ledger.json``, incidents, migration state, or the trial tree —
    resolves through this one function rather than re-deriving ``/ "state"``
    inline.
    """
    return Path(data_dir) / "state"


def ledger_path(state_dir: Path) -> Path:
    """Return ``<state_dir>/stage_ledger.json`` — the one fixed ledger path.

    A single fixed path is what resume-pending-first makes correct and cheap:
    two ledgers can never coexist, so discovery needs no glob and no
    per-event filename.
    """
    return Path(state_dir) / _LEDGER_FILENAME


def extraction_dir(state_dir: Path, event: str) -> Path:
    """Return ``<state_dir>/extraction/<event>/`` — phase 1's artifact tree.

    Outside the adapter tree entirely, so invisible to the slot scan and to
    the interim reap by construction.
    """
    return Path(state_dir) / _EXTRACTION_DIRNAME / event


def _hash_file(path: Path) -> "str | None":
    """Return the sha256 hex digest of *path*'s on-disk bytes, or ``None`` on absence.

    Reads raw bytes only — never decrypts.  Verification must never need a
    daily identity: a rotation that re-wraps the artifacts degrades to
    "re-extract", never to a decrypt raise.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning("stage_ledger: could not read %s while hashing", path, exc_info=True)
        return None


def build_artifact_list(paths: "Sequence[Path]") -> "list[dict]":
    """Hash every path in *paths* and return the ``artifacts`` list an entry carries.

    Called by the phase-1/phase-2 writer once, at the moment each named file
    is written, so the recorded hash is always the hash of the bytes that
    were just flushed to disk. Raises if a named path does not exist at call
    time — an entry must never be written naming a file that was not
    actually produced.
    """
    artifacts: list[dict] = []
    for path in paths:
        path = Path(path)
        digest = _hash_file(path)
        if digest is None:
            raise FileNotFoundError(f"build_artifact_list: {path} does not exist to hash")
        artifacts.append({"path": str(path), "sha256": digest})
    return artifacts


def extraction_stage(
    *,
    completed_at: str,
    sessions: "list[str]",
    episodic_rels: int,
    procedural_rels: int,
    artifacts: "list[dict]",
) -> dict:
    """Build the one ``"extraction"`` stage entry phase 1 writes.

    ``sessions`` records completed extractions only.  ``episodic_rels`` /
    ``procedural_rels`` are render-only counts (the run-status detail); no
    decision reads them.
    """
    return {
        "stage": "extraction",
        "completed_at": completed_at,
        "sessions": list(sessions),
        "episodic_rels": episodic_rels,
        "procedural_rels": procedural_rels,
        "artifacts": artifacts,
    }


def tier_written_stage(
    *, tier: str, completed_at: str, slot: "Path | None", artifacts: "list[dict]"
) -> dict:
    """Build one member's ``"tier_written"`` entry, written as that member writes.

    ``slot`` is the written slot directory itself. It is a genuine new fact,
    not derivable from ``artifacts`` alone: the timestamp folded into the
    directory name is chosen inside the slot envelope at save time, and
    recovering it from the tree afterwards would require a scan. Recorded
    as an absolute path string, ``None`` for a member that written no
    payload (a rows-only member, or a tier rebuilt to zero keys). Read back
    through :func:`written_slot_path` — the one accessor beside this
    builder for the ``"slot"`` field specifically; this module owns the
    entry's SCHEMA (which fields exist and what they mean), not exclusive
    read access to every field — a resumed dispatch's own
    ``ConsolidationLoop._latest_stage`` reads the generic ``"stage"`` and
    ``"tier"`` fields directly, the same way every stage kind exposes them.
    """
    return {
        "stage": "tier_written",
        "tier": tier,
        "completed_at": completed_at,
        "slot": str(slot) if slot is not None else None,
        "artifacts": artifacts,
    }


def written_slot_path(entry: "dict | None") -> "Path | None":
    """Return the written slot directory *entry* names, or ``None``.

    The one accessor for a ``"tier_written"`` entry's ``slot`` field —
    every caller that needs a resumed event's written slot location goes
    through this instead of scanning the entry's artifact list for a
    ``meta.json`` parent. ``None`` when *entry* is absent or names no slot
    (a rows-only member, or a tier rebuilt to zero keys).
    """
    if entry is None:
        return None
    slot = entry.get("slot")
    return Path(slot) if slot is not None else None


def tier_live_stage(*, tier: str, completed_at: str, artifacts: "list[dict]") -> dict:
    """Build one member's ``"tier_live"`` entry.

    A bundle's ``tier_live`` entries are written together in one
    :func:`write_stages` call — never singly — so a bundle is recorded done
    or not done, never half-recorded.
    """
    return {
        "stage": "tier_live",
        "tier": tier,
        "completed_at": completed_at,
        "artifacts": artifacts,
    }


def extraction_entry(ledger: StageLedger) -> "dict | None":
    """Return *ledger*'s one ``"extraction"`` stage entry, or ``None`` when absent.

    The one accessor for the extraction entry -- every caller that used to
    read ``ledger.stages[0]`` by position, or scan ``ledger.stages`` for it
    via an inline ``next(...)``, goes through this instead.
    """
    for stage in ledger.stages:
        if stage.get("stage") == "extraction":
            return stage
    return None


def full_topology(event: str) -> bool:
    """True when *event* is a full-topology event (``"full"`` or ``"reconcile"``).

    THE one topology predicate: ``event != "interim"``.  Every topology
    branch — the pre-fold recall universe, candidate-tier absorption and
    interim-ring reap, the end-of-event sweep, the overdue evaluation —
    reads this single derivation point rather than comparing ``event`` to
    ``"full"`` directly.  A reconcile (``/reconsolidate``) is a full
    consolidation whose input excludes pending sessions: it shares every
    other property of an ordinary full fold, including unconditional
    interim-ring absorption and warm start, so no second, narrower
    predicate exists to distinguish it structurally.

    Takes the bare ``event`` string rather than a :class:`StageLedger` so
    every caller can use it — including :meth:`ConsolidationLoop.stage_event`,
    which computes absorption before its own ledger object exists.  A
    caller holding a ledger passes ``ledger.event``.

    Reporting (the pending event's action name) reads ``event`` itself
    instead of this predicate — a report names the door, not the topology.
    """
    return event != "interim"


def _to_dict(ledger: StageLedger) -> dict:
    return {
        "version": ledger.version,
        "event": ledger.event,
        "venue": ledger.venue,
        "stamp": ledger.stamp,
        "tiers": dict(ledger.tiers),
        "stages": list(ledger.stages),
        "absorbed_interim_tiers": list(ledger.absorbed_interim_tiers),
    }


_EVENT_VOCABULARY: frozenset = frozenset({"interim", "full", "reconcile"})


def _from_dict(data: dict) -> StageLedger:
    event = data["event"]
    if event not in _EVENT_VOCABULARY:
        raise ValueError(f"stage_ledger: event {event!r} is not one of {sorted(_EVENT_VOCABULARY)}")
    return StageLedger(
        version=data["version"],
        event=event,
        venue=data["venue"],
        stamp=data["stamp"],
        tiers=dict(data.get("tiers", {})),
        stages=tuple(data.get("stages", [])),
        # Mandatory -- read_ledger()'s version gate only lets a payload
        # stamped CURRENT_VERSION reach here, and _to_dict always writes
        # this field for that version, so a payload missing it here is a
        # corrupt record of the current schema, not an older shape to
        # tolerate.  Not a case to silently default around: an event's
        # absorbed ring would otherwise go unreaped with no record of the
        # gap.
        absorbed_interim_tiers=tuple(data["absorbed_interim_tiers"]),
    )


def read_ledger(state_dir: Path) -> "StageLedger | None":
    """Read and parse the ledger at ``state_dir``, or ``None`` when it cannot answer.

    ``None`` covers three conditions, all treated identically by every
    caller: the file is absent; the file is unparseable; the file is an age
    envelope this process cannot open (no daily identity loaded, or a
    foreign envelope). An unopenable ledger is indistinguishable from no
    ledger for every decision made here — the artifacts it would name are
    equally unreadable, and its transcripts were never retired.

    A fourth condition is NOT folded into ``None``: a payload whose
    ``version`` is not in :data:`_SUPPORTED_VERSIONS` raises
    :class:`StageLedgerVersionUnsupported` — see that class's docstring for
    why. Every other condition never raises on content; logged at WARNING
    with the path for the two parse-failure conditions (never logged for
    plain absence).
    """
    from paramem.backup.encryption import read_maybe_encrypted

    path = ledger_path(state_dir)
    if not path.exists():
        return None
    try:
        raw = read_maybe_encrypted(path)
        data = json.loads(raw.decode("utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("stage_ledger: %s is not a parseable ledger — %s", path, exc)
        return None
    except Exception as exc:  # noqa: BLE001 — includes RuntimeError (no daily
        # identity) and pyrage.DecryptError (foreign envelope); both are
        # equally "cannot open right now", never a raise out of this reader.
        logger.warning("stage_ledger: %s could not be opened — %s", path, exc)
        return None

    version = data.get("version") if isinstance(data, dict) else None
    if version not in _SUPPORTED_VERSIONS:
        raise StageLedgerVersionUnsupported(path=path, version=version)

    try:
        return _from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("stage_ledger: %s is not a parseable ledger — %s", path, exc)
        return None


def write_stages(state_dir: Path, ledger: StageLedger, stages: "Sequence[dict]") -> None:
    """Append *stages* to *ledger* and rewrite the whole ledger file atomically.

    ONE atomic rewrite of the whole file (temp + rename, through
    :func:`~paramem.backup.encryption.write_infra_json` /
    ``write_infra_bytes``). Also the first-write path: a caller with a fresh,
    stages-less :class:`StageLedger` (head fields filled, ``stages=()``)
    calls this with the extraction entry to create the file.
    """
    from paramem.backup.encryption import write_infra_json

    updated = replace(ledger, stages=tuple(ledger.stages) + tuple(stages))
    path = ledger_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_infra_json(path, _to_dict(updated))


def missing_artifacts(entry: dict) -> "list[str]":
    """Return every artifact path in *entry* whose on-disk bytes do not verify.

    Re-hashes each ``entry["artifacts"]`` path's CURRENT on-disk bytes — raw
    bytes, never decrypted, exactly like :func:`verify` — and reports the
    paths that either no longer exist or whose bytes no longer match the
    recorded sha256. An entry naming no artifacts at all reports an empty
    list here (see :func:`verify`'s own "cannot verify anything" rule for why
    that case reads ``False`` there, not ``True`` here); callers that need to
    tell the two apart check ``entry.get("artifacts")`` themselves.

    The one place that turns a failed :func:`verify` into a named list of
    paths — used by a caller building a "these are the files that are gone
    or wrong" exception message rather than a bare boolean.
    """
    missing: list[str] = []
    for artifact in entry.get("artifacts") or []:
        digest = _hash_file(Path(artifact["path"]))
        if digest != artifact["sha256"]:
            missing.append(artifact["path"])
    return missing


def verify(entry: dict) -> bool:
    """Re-hash the on-disk bytes *entry* names and compare against its record.

    Walks ``entry["artifacts"]`` (every stage kind this module writes
    carries one) and re-hashes each named path's current on-disk bytes — raw
    bytes, never decrypted. A named artifact that does not exist reads
    ``False`` immediately; an empty ``artifacts`` list reads ``False`` (an
    entry naming nothing cannot verify anything). Takes no directory
    argument — every artifact path an entry carries is already absolute.
    """
    artifacts = entry.get("artifacts")
    if not artifacts:
        return False
    return not missing_artifacts(entry)


def dispose(state_dir: Path) -> bool:
    """Discard the pending event: its ledger, the whole extraction tree, and
    every training scratch dir the record names.

    The ONE disposal implementation, reachable without a live
    :class:`~paramem.training.consolidation.ConsolidationLoop`, without a
    :class:`~paramem.server.config.ServerConfig` and without a cycle number
    — every scratch path comes from the record's own
    ``tiers[*]["scratch"]`` field, fixed once when the event is staged
    (see :class:`StageLedger`'s own ``tiers`` docstring), never
    recomputed here.

    Reads the raw ledger payload directly — bypassing :func:`read_ledger`'s
    ``version`` gate, since discarding a record this process cannot
    interpret is precisely the escape hatch that gate exists to preserve
    (see :class:`StageLedgerVersionUnsupported`). A payload this function
    cannot parse at all still has its ledger file removed (the record
    itself is gone), but nothing else can be named from it — logged at
    WARNING rather than raised.

    Deletes, in order: ``ledger_path(state_dir)``; the WHOLE
    ``<state_dir>/extraction/`` tree (every event kind under it, not only
    the disposed record's own — the same reason
    :meth:`~paramem.training.consolidation.ConsolidationLoop.stage_event`
    clears it wholesale at entry: a crashed other-kind staging pass's tree is
    otherwise never reclaimed); every ``tiers[*]["scratch"]`` directory the
    payload names. A tier entry written before ``"scratch"`` existed logs
    the gap (nothing removed for that tier) rather than approximating a
    path or raising. Every deletion is individually idempotent, so calling
    this twice (a crash between two deletions, followed by a resume that
    finds every tier already done) is safe.

    Leaving a trained tier's scratch dir undisposed is not merely a leak: HF
    Trainer's ``resume_from_checkpoint`` reads the latest ``checkpoint-<step>/``
    there, so a later event that reuses the same scratch scope (e.g. two
    consecutive full folds at ``max_interim_count: 0``, whose ``cycle_<N>``
    scope does not change between folds) would resume a checkpoint already
    at its target epoch count, train zero steps, and write unchanged weights
    silently.

    Returns:
        ``True`` when a ledger was present to dispose, ``False`` when
        nothing was pending (a no-op, safe to call unconditionally).
    """
    from paramem.backup.encryption import read_maybe_encrypted

    path = ledger_path(state_dir)
    if not path.exists():
        return False

    tiers: dict = {}
    try:
        raw = read_maybe_encrypted(path)
        data = json.loads(raw.decode("utf-8"))
        tiers = data.get("tiers", {}) or {}
    except Exception as exc:  # noqa: BLE001 — an unparseable/unopenable
        # payload still has its ledger file removed below; only the
        # scratch-dir list becomes unknowable.
        logger.warning(
            "stage_ledger.dispose: %s could not be read for its tier/scratch list — %s",
            path,
            exc,
        )

    path.unlink(missing_ok=True)

    tree = Path(state_dir) / _EXTRACTION_DIRNAME
    if tree.exists():
        shutil.rmtree(tree, ignore_errors=True)

    for tier, info in tiers.items():
        scratch = info.get("scratch") if isinstance(info, dict) else None
        if not scratch:
            logger.warning(
                "stage_ledger.dispose: tier %r has no recorded scratch path — "
                "cannot dispose its training scratch dir",
                tier,
            )
            continue
        scratch_dir = Path(scratch)
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)

    return True
