"""Shared fold-fixture home for the consolidation build/write/publish suite.

Every collaborator here is a CPU-only fake: no GPU, no real PEFT model.
``_make_loop`` builds a ``ConsolidationLoop`` wired with a fake model/
tokenizer that satisfies the write/publish primitives (``save_pretrained``/
``load_adapter``/``named_parameters``/``set_adapter``/``delete_adapter``) for
real, so ``write_tier_slot``/``publish_tier_registry``/``publish_bundle`` run
unmocked (I/O only). ``_wire_fakes`` fakes only the two GPU-touching
collaborators every consumer needs faked (training and the recall gate) plus
``tier_backup_scope`` (its own backup/restore contract is covered separately
in ``tests/test_loader.py::TestTierBackupScope``). ``_make_state``/
``_run_pending_event_resume_and_wait`` build the minimal ``app._state`` and
blocking helper a synchronous, in-process ``_run_pending_event_resume()``
call needs. ``_FakeModel``/``_FakeBaseConfig`` are the simpler real-write
fake used wherever a test needs ``commit_tier_slot``'s real ``save_adapter``
call to produce a real timestamped slot directory without the full
mount-loop surface ``_FakeDriverModel`` provides.
``_recalled_entries_from_store`` builds ``stage_event``'s required
``recalled_entries`` argument for a test that drives ``stage_event``
directly, bypassing production hydration.  ``_SpyEntries`` is a
``MemoryStore._entries``-shaped dict that counts every access, for a test
proving the fold's recall + staging window never touches the live entry
mirror.

Consumers: ``tests/adapters/test_slot.py``, ``tests/backup/test_bundle_boot_binding.py``,
``tests/backup/test_integrity.py``, ``tests/server/test_active_store_migration.py``,
``tests/server/test_erase_doors.py``, ``tests/server/test_gates.py``,
``tests/server/test_migration_renderer.py``, ``tests/server/test_startup_validator.py``,
``tests/server/test_store_quarantine.py``,
``tests/test_boot_binding_integration.py``, ``tests/test_fold_phase1.py``,
``tests/test_memory_store.py``, ``tests/test_procedural.py``,
``tests/test_publish_bundle_resume.py``, ``tests/test_simulate_train_parity.py``,
``tests/test_stage_ledger.py``, ``tests/test_stale_key_retirement.py``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import torch

from paramem.graph.merger import GraphMerger
from paramem.graph.schema import Relation
from paramem.memory.store import MemoryStore
from paramem.training.consolidation import ConsolidationLoop
from paramem.training.key_registry import KeyRegistry
from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig


class _FakeParam:
    def __init__(self, shape=(4, 4)):
        self.data = torch.randn(shape)


class _FakeAdapterConfig:
    def __init__(self, r=8, lora_alpha=16, lora_dropout=0.0, target_modules=("q_proj",)):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = list(target_modules)


class _FakeBaseConfig:
    """Plain (non-Mock) stand-in for a PEFT model's ``.config``.

    ``build_manifest_for`` reads ``_name_or_path``/``_commit_hash`` off this
    and JSON-serializes the manifest — a ``MagicMock`` attribute would not
    survive that serialization, so the fingerprint fields must be real
    strings.
    """

    _name_or_path = "fake/base-model"
    _commit_hash = "deadbeef"


class _FakeTokenizer:
    """Plain stand-in for a tokenizer, sized so ``build_manifest_for`` can
    hash it without touching a MagicMock attribute that JSON can't encode."""

    name_or_path = "fake/base-model"
    vocab_file = None
    backend_tokenizer = None

    def __len__(self) -> int:
        return 100


class _FakeDriverModel:
    """One fake model exercising every primitive the driver touches across
    both the write step (``save_pretrained``/``config``) and the publish
    step's mount loop (``peft_config``/``load_adapter``/``named_parameters``/
    ``set_adapter``/``delete_adapter``) -- the driver's own ``self.model``
    is a single object used by both.
    """

    _LAYER = "layer0.q_proj"

    class _Cfg:
        _name_or_path = "fake/base-model"
        _commit_hash = "deadbeef"

    def __init__(self, resident_tiers=()):
        self.config = self._Cfg()
        self.peft_config: dict = {}
        self._params: dict = {}
        self.active_adapter = None
        for tier in resident_tiers:
            self._seed_adapter(tier)
        if resident_tiers:
            self.active_adapter = resident_tiers[0]

    def _seed_adapter(self, name):
        self.peft_config[name] = _FakeAdapterConfig()
        self._params[f"base_model.model.{self._LAYER}.lora_A.{name}.weight"] = _FakeParam()
        self._params[f"base_model.model.{self._LAYER}.lora_B.{name}.weight"] = _FakeParam()

    def save_pretrained(self, path, selected_adapters=None) -> None:
        """Write adapter-NAME-DEPENDENT stub bytes -- ``save_adapter``'s real
        callers always pass ``selected_adapters=[adapter_name]`` (see
        ``paramem.models.loader.atomic_save_adapter``), so a write test can
        tell which adapter's weights actually landed in a slot rather than
        every slot being byte-identical."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text("{}")
        names = ",".join(selected_adapters or [])
        (p / "adapter_model.safetensors").write_bytes(f"stub-weights:{names}".encode())

    def load_adapter(self, path, adapter_name):
        self._seed_adapter(adapter_name)

    def named_parameters(self):
        return list(self._params.items())

    def set_adapter(self, name):
        self.active_adapter = name

    def delete_adapter(self, name):
        self.peft_config.pop(name, None)
        self._params = {k: v for k, v in self._params.items() if f".{name}." not in k}

    def eval(self):
        return self


class _FakeModel:
    """Plain stand-in for a PEFT model whose ``save_pretrained`` performs a
    real (stub-content) filesystem write, so ``commit_tier_slot``'s real
    ``save_adapter`` call produces a real timestamped slot directory. Unlike
    ``_FakeDriverModel`` it carries no mount-loop surface (``load_adapter``/
    ``named_parameters``/``set_adapter``/``delete_adapter``) — for consumers
    that never mount a second adapter over this one."""

    def __init__(self) -> None:
        self.config = _FakeBaseConfig()
        self.peft_config: dict = {}

    def save_pretrained(self, path, selected_adapters=None) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text("{}")
        (p / "adapter_model.safetensors").write_bytes(b"stub-weights")


class _FakeProbe:
    """Stand-in for RecallProbe -- verdict content is opaque (_assert_tier_recall
    is faked too and never inspects it), but ``per_key`` must be a real
    (empty) sequence: ``_train_gate_write`` unconditionally hands it to
    ``on_recall_probe`` for the debug recall-probe artifact."""

    per_key: tuple = ()


@contextmanager
def _fake_tier_backup_scope(model, config, tier):
    """No-op stand-in for tier_backup_scope (patched at its own module):
    its own backup/restore contract is covered by
    tests/test_loader.py::TestTierBackupScope; it requires a real
    peft.PeftModel, which this driver-sequencing suite deliberately does
    not construct.  ``.vram`` is an empty-but-present mapping -- its
    real vram_measure population is covered by
    tests/test_loader.py::TestTierBackupScopeVram; this fake only needs to
    satisfy _train_gate_write's read of the attribute."""

    class _Scope:
        pass

    scope = _Scope()
    scope.model = model
    scope.vram = {}
    yield scope


def _make_loop(tmp_path, *, procedural: bool = False, resident_tiers=()) -> ConsolidationLoop:
    """Mirrors tests/test_fold_phase1.py::_make_loop, with a fake model/
    tokenizer wired for the write/publish primitives this driver calls for
    real."""
    loop = object.__new__(ConsolidationLoop)
    loop.model = _FakeDriverModel(resident_tiers=resident_tiers)
    loop.tokenizer = _FakeTokenizer()
    loop.config = ConsolidationConfig(promotion_threshold=3, decay_window=10)
    loop.training_config = TrainingConfig(
        num_epochs=1,
        gradient_checkpointing=False,
        batch_size=1,
        recall_early_stopping=False,
        recall_probe_batch_size=1,
    )
    # rank=8/alpha=16 matches _FakeAdapterConfig's defaults (mirrors
    # tests/test_go_live.py) so ensure_adapter_matching's warm no-op path
    # fires instead of a cold get_peft_model() recreate, which needs a
    # real torch.nn.Module the fake model does not provide.
    loop.episodic_config = AdapterConfig(rank=8, alpha=16, target_modules=["q_proj"])
    loop.semantic_config = AdapterConfig(rank=8, alpha=16, target_modules=["q_proj"])
    loop.procedural_config = (
        AdapterConfig(rank=8, alpha=16, target_modules=["q_proj"]) if procedural else None
    )
    loop.wandb_config = None
    loop._thermal_policy = None
    loop.output_dir = tmp_path / "adapters"
    loop.output_dir.mkdir(parents=True, exist_ok=True)
    loop.save_cycle_snapshots = False
    loop._debug_base = None
    loop.snapshot_dir = None
    loop.shutdown_requested = False
    loop._bg_trainer = None
    loop._early_stop_callback = None
    loop.fingerprint_cache = None
    loop._keep_prior_slots = 2
    loop.cycle_count = 0
    loop._indexed_next_index = 1
    loop._procedural_next_index = 1
    loop.promoted_keys = set()
    loop._pending_promoted_keys = set()
    loop.graph_enrichment_neighborhood_hops = 2
    loop.graph_enrichment_max_entities_per_pass = 50
    loop.cloud_enabled = False
    loop._incidents_state_dir = None

    loop.merger = GraphMerger(model=None, tokenizer=None)

    store = MemoryStore()
    for tier in ("episodic", "semantic", "procedural"):
        store.load_registry(tier, KeyRegistry())
    loop.store = store
    return loop


def _rel(subject: str, predicate: str, obj: str, **kw) -> Relation:
    kw.setdefault("relation_type", "factual")
    kw.setdefault("confidence", 1.0)
    kw.setdefault("speaker_id", "speaker0")
    return Relation(subject=subject, predicate=predicate, object=obj, **kw)


def _write_graph(tier_root, quads: "list[dict]") -> None:
    """Write *quads* as a simulate-venue graph payload into a fresh bound slot
    under *tier_root* — the exact shape ``DiskMemorySource.probe`` reads
    (``find_live_slot(tier_root, tier_registry_sha256(tier_root))`` then
    ``<slot>/graph.json``), through the same promotion sequence production
    uses (:func:`~paramem.adapters.slot.write_slot`).

    Binds to whatever registry already lives at
    ``tier_root/indexed_key_registry.json`` (or the empty-registry hash when
    none exists yet) — callers that need the slot to bind write the registry
    file first, exactly as the production commit ordering requires.
    """
    import networkx as nx

    from paramem.adapters.manifest import graph_payload_manifest, tier_registry_sha256
    from paramem.adapters.slot import write_slot
    from paramem.memory.persistence import _IK_KEY_ATTR, save_memory_to_disk

    tier_root = Path(tier_root)
    tier_root.mkdir(parents=True, exist_ok=True)
    graph = nx.MultiDiGraph()
    for quad in quads:
        graph.add_edge(
            quad["subject"],
            quad["object"],
            **{
                _IK_KEY_ATTR: quad["key"],
                "predicate": quad.get("predicate", ""),
                "speaker_id": quad.get("speaker_id", ""),
            },
        )
    manifest = graph_payload_manifest(
        name=tier_root.name,
        key_count=len(quads),
        registry_sha256=tier_registry_sha256(tier_root),
        window_stamp="",
    )
    write_slot(
        tier_root,
        manifest=manifest,
        write_payload=lambda pending_slot: save_memory_to_disk(graph, pending_slot / "graph.json"),
    )


def _wire_fakes(loop, monkeypatch, *, train_side_effect=None):
    """Fake the two GPU-touching collaborators (train, recall gate) and the
    backup scope; everything else (write, publish, adopt, ledger) is real.

    ``train_side_effect`` lets a caller inject its own training behaviour
    (e.g. an abort, or a per-call spy independent of another loop's) — the
    default fakes a successful, unaborted train on every non-empty entry
    list.
    """
    from paramem.models import loader as loader_mod

    def _default_train(entries, **kwargs):
        if not entries:
            return None, None
        return {"aborted": False, "train_loss": 0.01}, None

    loop._train_tier_adapter = MagicMock(side_effect=train_side_effect or _default_train)
    loop._probe_recall = MagicMock(return_value=_FakeProbe())
    loop._assert_tier_recall = MagicMock()
    monkeypatch.setattr(loader_mod, "tier_backup_scope", _fake_tier_backup_scope)


class _SpyEntries(dict):
    """A ``MemoryStore._entries``-shaped dict that counts every access.

    Behaves exactly like the dict it wraps (pre-seeded with the same
    content) so swapping it in cannot change what a correct caller
    observes — it exists to catch an INCORRECT caller reaching into
    ``_entries`` at all.  Every read AND write path ``MemoryStore``'s own
    methods use against ``self._entries`` is overridden here so a hit
    anywhere in that surface is counted, not just the obvious ones."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.access_count = 0
        self.calls: list[str] = []

    def _hit(self, name: str) -> None:
        self.access_count += 1
        self.calls.append(name)

    def __getitem__(self, key):
        self._hit("__getitem__")
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        self._hit("__setitem__")
        super().__setitem__(key, value)

    def __delitem__(self, key):
        self._hit("__delitem__")
        super().__delitem__(key)

    def __contains__(self, key):
        self._hit("__contains__")
        return super().__contains__(key)

    def __iter__(self):
        self._hit("__iter__")
        return super().__iter__()

    def get(self, key, default=None):
        self._hit("get")
        return super().get(key, default)

    def setdefault(self, key, default=None):
        self._hit("setdefault")
        return super().setdefault(key, default)

    def pop(self, key, *default):
        self._hit("pop")
        return super().pop(key, *default)

    def items(self):
        self._hit("items")
        return super().items()

    def values(self):
        self._hit("values")
        return super().values()

    def keys(self):
        self._hit("keys")
        return super().keys()


def _make_state(loop, *, tmp_path) -> dict:
    """Minimal ``app._state`` for a synchronous, in-process
    ``_run_pending_event_resume()`` call: no event loop (so
    ``_consolidation_terminal`` runs its closure inline) and no model/tokenizer
    (so ``_revalidate_adapter_manifests`` no-ops rather than reaching for a
    real PEFT model)."""
    config = MagicMock()
    config.paths.data = tmp_path
    config.adapter_dir = loop.output_dir
    # Ground the two MagicMock comparisons _run_stage_b_cycle's weights-venue
    # path makes before it ever reaches the fakes above.
    config.consolidation.training_temp_limit = 0
    config.vram.cooldown_gate_threshold_c = 0

    router = MagicMock()
    router.reload.return_value = None

    return {
        "config": config,
        "model": None,
        "tokenizer": None,
        "consolidation_loop": loop,
        "session_buffer": MagicMock(),
        "router": router,
        "event_loop": None,
        "consolidating": True,
        "background_trainer": None,
    }


def _recalled_entries_from_store(loop) -> dict:
    """Build a ``stage_event``-shaped ``recalled_entries`` dict for a test
    that calls ``stage_event`` directly, bypassing production hydration.

    Production reconstructs this dict from the venue via
    ``ConsolidationLoop._hydrate_store_for_fold`` (never from the store's
    RAM mirror) before every ``stage_event`` call. A test that drives
    ``stage_event`` directly owns that reconstruction itself; this helper
    stands in for it by reading back whatever content the test staged onto
    the RAM mirror via ``store.put`` (either directly, or as a side effect
    of an earlier ``stage_event``/``run_build_and_publish`` call landing
    through ``adopt_increments``), restricted to each tier's currently
    ACTIVE keys — the same selection ``_hydrate_store_for_fold`` applies at
    its own boundary."""
    result: dict[str, dict[str, dict]] = {}
    for tier in loop.store.tiers_with_registry():
        active = set(loop.store.active_keys_in_tier(tier))
        entries = {
            key: entry for key, entry in loop.store.entries_in_tier(tier).items() if key in active
        }
        if entries:
            result[tier] = entries
    return result


def _seed_payload_bearing_ring(loop, interim_name: str) -> Path:
    """Seed one interim ring member with real content, bookkeeping, and an
    on-disk slot directory -- the shared fixture every full-event ring test
    (absorb, abort, crash-window) stages over.  ``interim_name`` must be an
    ``episodic_interim_<stamp>`` name already resident on *loop* (via
    ``_make_loop(..., resident_tiers=[...])``); the on-disk directory is
    derived from the SAME stamp, matching ``interim_dir_for_name``'s own
    ``episodic/interim_<stamp>/`` shape."""
    loop.store.registry(interim_name).add("interim1")
    loop.store.registry(interim_name).set_simhash("interim1", 99)
    loop.store.set_bookkeeping(
        "interim1",
        speaker_id="speaker0",
        relation_type="factual",
        reinforcement_count=1,
        last_reinforced_cycle=0,
        last_seen="2026-01-01T00:00:00Z",
        first_seen="2026-01-01T00:00:00Z",
        promoted=False,
    )
    loop.store.put(
        interim_name,
        "interim1",
        {"key": "interim1", "subject": "alex", "predicate": "visited", "object": "lisbon"},
        register=False,
    )
    from paramem.memory.interim_adapter import interim_dir_for_name

    interim_dir = interim_dir_for_name(loop.output_dir, interim_name)
    interim_dir.mkdir(parents=True)
    (interim_dir / "20260101-000000").mkdir()
    (interim_dir / "20260101-000000" / "meta.json").write_text("{}")
    return interim_dir


def _run_pending_event_resume_and_wait(state) -> None:
    """Call ``app._run_pending_event_resume()`` and block until its work has
    actually finished.

    The weights venue dispatches through ``_run_stage_b_cycle`` ->
    ``BackgroundTrainer.submit()`` — fire-and-forget onto the trainer's own
    persistent worker thread.  The trainer is a real, singly-threaded FIFO
    queue (``paramem.server.background_trainer``), so submitting one more,
    no-op job right after and waiting on IT is the project's own pattern for
    "block until the real job ahead of it drains" (mirrors
    ``BackgroundTrainer.submit_and_wait``'s own docstring).
    """
    import paramem.server.app as app_module

    app_module._run_pending_event_resume()
    bt = app_module._state.get("background_trainer")
    if bt is not None:
        bt.submit_and_wait(lambda: None)
