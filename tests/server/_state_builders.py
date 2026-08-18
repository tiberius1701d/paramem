"""Shared ``app._state`` builders for the endpoint-guard test family.

``_write_pending_ledger`` seeds a bare pending stage ledger on disk — the
shape every "refuses with a pending record" pin across the guarded doors
needs.  The per-door ``_make_*_state`` builders are each door's own minimal
``app._state`` dict factory, promoted here so
``tests/server/test_consolidate_dispatch.py`` (which exercises every door's
pending-record refusal from one place) can reach them without importing
another test module; each door's own test file re-imports its builder back
under its original local name.

Consumers: ``tests/server/test_consolidate_dispatch.py``,
``tests/server/test_speaker_forget.py``,
``tests/server/test_debug_erase_keys_endpoint.py``,
``tests/server/test_interim_discard.py``,
``tests/server/test_ingest_endpoint.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from paramem.memory.store import MemoryStore
from paramem.training.key_registry import KeyRegistry

# ---------------------------------------------------------------------------
# Pending-ledger seed (shared by every "refuses with a pending record" pin)
# ---------------------------------------------------------------------------


def _write_pending_ledger(tmp_path, *, event: str) -> None:
    from paramem.training import stage_ledger as sl

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    ledger = sl.StageLedger(
        version=2,
        event=event,
        venue="weights",
        stamp="20260101T0000",
        tiers={"episodic": {"adapter": "episodic", "pre_sha": ""}},
    )
    sl.write_stages(state_dir, ledger, [])


# ---------------------------------------------------------------------------
# POST /speaker/forget state family
# ---------------------------------------------------------------------------


def _make_forget_config(tmp_path: Path) -> MagicMock:
    """Minimal config mock with adapter_dir under tmp_path.

    Per-tier bookkeeping (``key_metadata.json``) lives under each tier root
    (``adapter_dir / <tier>``) — there is no separate global path to mock.
    """
    cfg = MagicMock()
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg.adapter_dir = adapter_dir
    cfg.paths = MagicMock()
    cfg.paths.data = tmp_path / "data"
    return cfg


def _make_forget_loop_with_store(store: MemoryStore) -> MagicMock:
    """Build a MagicMock ConsolidationLoop wrapping a real MemoryStore.

    Used by tests that need real store mutation semantics (registry, entry,
    bookkeeping, simhash) but drive everything else on the loop through a
    mock.

    Wires ``loop.model`` (a bare, non-PEFT ``MagicMock``) and
    ``loop.ensure_adapters`` (returns whatever ``loop.model`` currently is,
    read dynamically at call time) so any code path that reads them has
    something to call without crashing.
    """
    loop = MagicMock()
    loop.store = store
    loop.model = MagicMock()
    loop.ensure_adapters = MagicMock(side_effect=lambda: loop.model)
    return loop


def _make_forget_loop(speaker_id: str, keys: list[str]) -> MagicMock:
    """Build a MagicMock ConsolidationLoop; store.iter_bookkeeping yields *keys* for *speaker_id*.

    The store exposes two main tiers (``episodic`` and ``semantic``) whose
    ``KeyRegistry`` objects contain the supplied keys so the per-tier
    ``save_from_bytes`` write (via :func:`restamp_tier_manifest`) can be
    verified.  The simhash dicts include the keys so the simhash clean-up
    branch is exercised.

    ``/forget`` routes through ``store.iter_bookkeeping()`` to resolve speaker
    keys (``merger.graph`` is cleared at cycle-end by the cycle's finally-block
    reset, so the old graph-based ``keys_for_speaker`` path is unavailable).
    It then calls ``store.discard_keys(keys)`` (the shared helper) — a single
    permanent stale-mark, not an erase.  Tests verify that the helper is called
    with the correct arguments.
    """
    loop = MagicMock()
    loop.model = MagicMock()
    loop.ensure_adapters = MagicMock(side_effect=lambda: loop.model)

    # iter_bookkeeping returns bookkeeping records keyed by speaker_id.
    # The handler iterates all records and filters by record.get("speaker_id").
    bk_records = [(k, {"speaker_id": speaker_id, "relation_type": "episodic"}) for k in keys]
    loop.store.iter_bookkeeping.return_value = iter(bk_records)

    # Per-tier KeyRegistry mocks.
    ep_registry = MagicMock(spec=KeyRegistry)
    ep_registry.__contains__ = MagicMock(side_effect=lambda k: k in keys)
    ep_registry.knows = MagicMock(side_effect=lambda k: k in keys)
    # save_bytes must return valid bytes so the handler can compute sha256 for
    # the post-erase re-stamp. Distinct pre/post values are not needed here
    # because discard_keys is mocked (it does not mutate the mock registry);
    # any stable bytes value is sufficient.
    ep_registry.save_bytes.return_value = b'{"active_keys": [], "stale": [], "simhash": {}}'
    sem_registry = MagicMock(spec=KeyRegistry)
    sem_registry.__contains__ = MagicMock(side_effect=lambda _: False)
    sem_registry.knows = MagicMock(side_effect=lambda _: False)

    # Store: tiers_with_registry returns episodic + semantic.
    loop.store.tiers_with_registry.return_value = ["episodic", "semantic"]
    loop.store.registry.side_effect = lambda t: ep_registry if t == "episodic" else sem_registry

    loop._ep_registry = ep_registry
    loop._sem_registry = sem_registry
    return loop


def _make_forget_speaker_store(speaker_id: str, *, returns: bool = True) -> MagicMock:
    """SpeakerStore mock whose remove(speaker_id) returns *returns*."""
    store = MagicMock()
    store.remove.return_value = returns
    return store


def _make_forget_buffer(speaker_id: str, conv_ids: list[str]) -> MagicMock:
    """SessionBuffer mock with _sessions carrying *conv_ids* attributed to *speaker_id*."""
    buf = MagicMock()
    buf._sessions = {cid: {"speaker_id": speaker_id, "speaker": "Test User"} for cid in conv_ids}
    return buf


def _make_forget_state(
    tmp_path: Path,
    *,
    loop=None,
    speaker_store=None,
    buffer=None,
    config=None,
    mode: str = "local",
    consolidating: bool = False,
    background_trainer=None,
    migration=None,
    router=None,
    adapter_manifest_status=None,
) -> dict:
    """Build a minimal _state dict for /speaker/forget endpoint tests.

    Carries the keys the guard rework reads:
    ``mode``/``consolidating``/``background_trainer``/``migration`` feed
    ``_consolidation_dispatch_guards`` (plain ``dict[...]`` access, so every
    caller of this helper must set them — the guard would otherwise
    ``KeyError``); ``router`` is reloaded in the handler's no-await tail.
    ``adapter_manifest_status`` mirrors the boot-time mount status the real
    lifespan publishes; the door narrows to a stale-mark, so it never mutates
    this key.  ``model``/``tokenizer``/``memory_store`` mirror what the real
    lifespan publishes — ``memory_store`` is the SAME object as ``loop.store``
    so a caller inspecting either sees identical state.
    """
    return {
        "config": config or _make_forget_config(tmp_path),
        "consolidation_loop": loop,
        "speaker_store": speaker_store,
        "session_buffer": buffer or MagicMock(),
        "mode": mode,
        "consolidating": consolidating,
        "background_trainer": background_trainer,
        "migration": migration,
        "router": router or MagicMock(),
        "adapter_manifest_status": (
            adapter_manifest_status if adapter_manifest_status is not None else {}
        ),
        "model": loop.model if loop is not None else MagicMock(),
        "tokenizer": None,
        "memory_store": loop.store if loop is not None else MagicMock(),
    }


# ---------------------------------------------------------------------------
# POST /debug/erase-keys state family
# ---------------------------------------------------------------------------


def _make_erase_config(tmp_path: Path, *, debug: bool = True) -> MagicMock:
    cfg = MagicMock()
    cfg.debug = debug
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg.adapter_dir = adapter_dir
    cfg.paths = MagicMock()
    cfg.paths.data = tmp_path / "data"
    return cfg


def _seed_erase_registry_on_disk(
    adapter_dir: Path, tier: str, key: str, *, simhash: int = 0xDEADBEEF
) -> None:
    """Write *tier*'s ``indexed_key_registry.json`` directly with *key* active.

    The file-driven erase primitive reads tier registries straight off
    disk, so seeding a key for these tests means writing the registry
    file — never just populating an in-RAM ``MemoryStore`` (that store, if
    any, is a separate concern this door explicitly does not require).
    """
    reg = KeyRegistry()
    reg.add(key)
    reg.set_simhash(key, simhash)
    tier_root = adapter_dir / tier
    tier_root.mkdir(parents=True, exist_ok=True)
    reg.save(tier_root / "indexed_key_registry.json")


def _make_erase_state(
    tmp_path: Path,
    *,
    config=None,
    mode: str = "local",
    consolidating: bool = False,
    background_trainer=None,
    migration=None,
    router=None,
    adapter_manifest_status=None,
    model="__default__",
    memory_store=None,
    consolidation_loop=None,
    store_quarantine=None,
) -> dict:
    return {
        "config": config or _make_erase_config(tmp_path),
        "consolidation_loop": consolidation_loop,
        "mode": mode,
        "consolidating": consolidating,
        "background_trainer": background_trainer,
        "migration": migration,
        "router": router or MagicMock(),
        "adapter_manifest_status": (
            adapter_manifest_status if adapter_manifest_status is not None else {}
        ),
        "model": MagicMock() if model == "__default__" else model,
        "tokenizer": None,
        "memory_store": memory_store,
        "store_quarantine": store_quarantine,
    }


# ---------------------------------------------------------------------------
# POST /interim/discard state family
# ---------------------------------------------------------------------------


def _make_discard_config(tmp_path: Path) -> MagicMock:
    cfg = MagicMock()
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg.adapter_dir = adapter_dir
    cfg.paths = MagicMock()
    cfg.paths.data = tmp_path / "data"
    return cfg


def _make_discard_state(
    tmp_path: Path,
    *,
    loop=None,
    config=None,
    mode: str = "local",
    consolidating: bool = False,
    background_trainer=None,
    migration=None,
    router=None,
    session_buffer=None,
) -> dict:
    return {
        "config": config or _make_discard_config(tmp_path),
        "consolidation_loop": loop,
        "mode": mode,
        "consolidating": consolidating,
        "background_trainer": background_trainer,
        "migration": migration,
        "router": router or MagicMock(),
        "adapter_manifest_status": {},
        "session_buffer": session_buffer or MagicMock(),
        "last_consolidation": None,
    }


# ---------------------------------------------------------------------------
# POST /ingest-sessions state family
# ---------------------------------------------------------------------------


def _make_ingest_state(tmp_path: Path) -> dict:
    """Build a minimal _state dict for ingest endpoint tests."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    config = MagicMock()
    config.paths.sessions = sessions_dir
    config.paths.data = tmp_path / "data"
    config.debug = False

    from paramem.server.session_buffer import SessionBuffer

    buffer = SessionBuffer(session_dir=sessions_dir, debug=False)

    # Build a real SpeakerStore with one known speaker.
    from paramem.server.speaker import SpeakerStore

    store = SpeakerStore(tmp_path / "profiles.json")
    known_speaker_id = store.enroll("Alice", [0.1, 0.2, 0.3])

    return {
        "model": None,
        "config": config,
        "consolidating": False,
        "mode": "local",
        "migration": {},  # no TRIAL
        "server_started_at": "2026-04-26T00:00:00+00:00",
        "session_buffer": buffer,
        "speaker_store": store,
        # Store the known speaker id for use in tests.
        "_test_known_speaker_id": known_speaker_id,
    }


__all__ = [
    "_make_discard_config",
    "_make_discard_state",
    "_make_erase_config",
    "_make_erase_state",
    "_make_forget_buffer",
    "_make_forget_config",
    "_make_forget_loop",
    "_make_forget_loop_with_store",
    "_make_forget_speaker_store",
    "_make_forget_state",
    "_make_ingest_state",
    "_seed_erase_registry_on_disk",
    "_write_pending_ledger",
]
