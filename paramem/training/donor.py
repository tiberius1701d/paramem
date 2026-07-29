"""Donor-adapter lifecycle: synthetic key-triple generation, checkpoint
persistence, and seed-training through the shared funnel.

A NEW module boundary, not a code move: nothing was relocated out of
``paramem.training.consolidation`` (7000+ lines; per project rule an
oversized file is a defect to split, not enlarge) to create this file — the
funnel this module trains through (``ConsolidationLoop._train_tier_adapter``)
stays exactly where it was. This module owns the donor's three concerns
instead: (1) deterministically generating the synthetic crowded-cluster
training population (:func:`donor_entries`), (2) persisting and validating
each (base model, topology) pair's own trained donor checkpoint as an
adapter store beside the main tiers
(:func:`donor_store_dir` / :func:`donor_checkpoint_valid`), and
(3) building the donor by training it through
``ConsolidationLoop._train_tier_adapter`` on a transient adapter slot
(:func:`build_donor`). Consumed by ``ConsolidationLoop._train_tier_adapter``
(the seeding hook, ``paramem.training.consolidation``) — this module never
imports ``ConsolidationLoop`` at runtime (only under ``TYPE_CHECKING``),
so the dependency is one-directional and importing this module never
triggers a training-machinery import cycle.

Why the donor exists
---------------------
Repeated production folds have shown cold small-N folds failing or
regressing non-monotonically even after warm-start retries. The bet is
that seeding a measured-cold adapter from a *task-skilled* (not fact-
specific) donor beats LoRA-zero as the starting point for the indexed-key
task. The donor's synthetic content is deliberately NOT drawn from
PerLTQA/longmemeval (both are live benchmark ground truth the project must
never retrain on) and NOT the anonymizer's output (by hand, per operator
direction). It is a seed+recipe pure function over a hand-anonymized 21-key
production fixture (``donor_fixture.json``, itself never derived from
in-repo test data — see the fixture's own provenance note below).

Key namespace reservation
--------------------------
``DONOR_KEY_BAND_WIDTH`` reserves ``graph1``-``graph{width}`` AND
``proc1``-``proc{width}`` for the donor's synthetic population; real
minting floors (``ConsolidationLoop._indexed_next_index`` /
``_procedural_next_index``) start at ``DONOR_KEY_FLOOR = width + 1``
unconditionally, so real keys can never collide with the reserved band.
The width bounds the maximum key-surface *divergence depth* the donor's
OWN fixture/synthesized population is trained at (the separating variable
in the production failures is cluster depth, not absolute key magnitude —
see ``experiments/test20_smallN_cold_gate.py``): a 1-200 band spans depths
1-3, the depth range observed in production at the time this constant was
set. Widening this constant is a key-NAMESPACE-headroom decision (avoiding
real-key collision as production keys grow past 200), NOT a donor-teaching
requirement: Test 20's depth-4 transfer arm (``benchmarking.md``, "Test
20: Small-N Cold-Init Recall Gate", "Depth scaling" section) showed the
SAME depth-3-trained donor checkpoint (built once, never rebuilt) rescues
cold recall on 4-digit (depth-4) target keys at 21/21 on both seeds run,
zero donor/target key overlap — donor task-skill transfer is depth-general,
not depth-matched, so a depth-4+ cluster observed in production does NOT
by itself require rebuilding the donor at a wider band. Depth 5 remains
the one unmeasured extrapolation (same section, "Remaining gap"). ONE
shared constant (not two independently-tunable per-prefix widths) because
both prefixes are reserved at the SAME width symmetrically ("Donors own
graph1-200 AND proc1-200") — two constants that could drift apart would
only reintroduce an asymmetry that decision rejected.

Fixture provenance
-------------------
``donor_fixture.json``'s 21 entries are a hand-anonymized copy of a
production fold that failed at recall 0.762 (originally captured live at
keys ``graph179``-``graph193`` + ``proc35``-``proc40``). Every subject/object
value naming a real person, place, organisation, or date was replaced with a
fictional equivalent of similar shape (predicates are preserved verbatim —
they are the training mechanism, not personal content). The un-anonymized
source never enters this repository.

The fixture's keys were remapped to ``graph101``-``graph115`` +
``proc101``-``proc106`` after the small-N validation runs (see
``experiments/test20_smallN_cold_gate.py`` and ``benchmarking.md``) proved
donor seeding transfers with zero key overlap between the donor's own
population and the target keys being trained — the original verbatim
production-key overlap was validated-but-unnecessary. The remap preserves
the failing fold's structural shape exactly (same 21 subject/predicate/object
triples in the same order; same crowded 7-wide ``expertise`` cluster; same
depth-3 divergence — ``graph101``-``graph109`` share the leading ``"10"``,
``graph110``-``graph115`` share ``"11"``) while landing on numerals outside
every documented live-store key range, so the donor's synthetic population
can never collide with a real production key by construction. The synthetic
content pools below
(``_PRIMARY_NAMES`` etc., used to extend the fixture's 21 entries to
``N >= DONOR_MIN_ENTRIES``) are disjoint from every one of the fixture's own
21 values by construction — none of the fixture's actual subjects/objects
appear as pool entries, so no pool draw can ever reproduce a fixture value
verbatim.

Build timing (operational note)
---------------------------------
Every (base model, LoRA topology) pair gets its OWN donor checkpoint,
built lazily by the same single call site
(``ConsolidationLoop._maybe_seed_from_donor``, inside
``_train_tier_adapter``): when the TARGET tier's store holds no valid
checkpoint, :func:`build_donor` runs INLINE,
synchronously, at that topology, before that fold's own training. The
step count is topology-INDEPENDENT: 147 entries -- ``donor_entries``
returns whole 21-entry blocks, so ``DONOR_MIN_ENTRIES=128`` requested
rounds up to 147 -- at the anchored 30-epoch bucket: 2220 steps for
either topology. The measured per-step wall time is now anchored on BOTH
topologies (Test 20): attention-only ~1.0s/step (~37 min total);
attention+MLP ~1.2285s/step (~45.5 min total -- 2220 realized steps,
wall_train_seconds=2727.16, ``donor_build_smoke_procedural``
20260727_183637/build_results.json) -- a +23% per-step cost for 3.08x the
trainable parameters (LoRA rank 8 over 7 vs. 4 target modules on Mistral
7B), confirming the dominant per-step cost is the frozen base model's
forward/backward, not the LoRA update, rather than scaling with the
trainable-parameter ratio. The practical consequence: the first
measured-cold fold of EACH topology in a deployment's lifetime (or after
a base-model swap) absorbs a full donor training run for that topology IN
ADDITION TO its own training -- with the shipped two-topology config
(episodic and semantic share one attention-only topology; procedural is
the only attention+MLP topology), a deployment pays this cost at most
twice across its lifetime, never stacked into the same fold (a fold
trains one tier at a time). This is NOT "roughly doubling that fold's
wall time" -- the multiplier depends on how small the triggering fold's
OWN key count is, and it is small by construction (the first
measured-cold adapter of that topology in a deployment's lifetime).
Measured (attention-only topology): an N=21 triggering fold (550 of its
own steps) pays ~5x its own wall time that one cycle
(``(550 + 2220) / 550``); an N=2 triggering fold (160 of its own steps)
pays ~14x (``(160 + 2220) / 160``) -- see ``benchmarking.md``, "Test 20",
for the measurement. Every fold after the triggering one reuses that
topology's persisted checkpoint and pays no extra cost until it is
invalidated again (base-model swap, a shape edit to THAT topology, or a
donor-recipe change — see :func:`donor_checkpoint_valid`); a shape edit to
a DIFFERENT tier's topology does not invalidate this one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paramem.training.consolidation import ConsolidationLoop

logger = logging.getLogger(__name__)


class DonorBuildIncomplete(RuntimeError):
    """Raised by :func:`build_donor` when ``_train_tier_adapter`` did not
    complete training for the donor (thermal/pause abort, or an empty
    example set) — no checkpoint is persisted. Callers (the seeding hook)
    catch this specifically and treat the fold as "no checkpoint yet" —
    skip seeding this fold, log, and let the next measured-cold fold retry
    the build — rather than crash the fold or (worse) silently persist a
    checkpoint from a run that never actually trained."""


# --- Key namespace reservation (owner-resolved) -----------------------------
DONOR_KEY_BAND_WIDTH: int = 200
"""Width of the reserved low key band, both prefixes. See module docstring."""

DONOR_KEY_FLOOR: int = DONOR_KEY_BAND_WIDTH + 1
"""Real (non-donor) key minting floor for BOTH the ``graph`` and ``proc``
counters — ``ConsolidationLoop._indexed_next_index`` /
``_procedural_next_index`` are seeded at
``max(DONOR_KEY_FLOOR, high_water_from_store)`` (see module docstring)."""

# --- Donor checkpoint location ----------------------------------------------
DONOR_STORE_PREFIX: str = "donor-"
"""Name prefix of a donor's adapter store, a SIBLING of the main tiers under
``ConsolidationLoop.donor_adapter_root``: ``donor-<topology_id>-b<base8>/<stamp>/``.

A donor is an adapter checkpoint store, structurally identical to a main
tier: the same ``atomic_save_adapter`` writes it, the same ``meta.json``
(:class:`~paramem.adapters.manifest.AdapterManifest`) describes it, the same
``find_live_slot`` resolves its promoted slot, and the same backup / restore
/ prune routines own its lifecycle. It is therefore stored where the tiers
are stored, and needs no store-specific handling anywhere.

What a donor is NOT is a MEMORY tier: it carries no keyed entries, no
``indexed_key_registry.json``, and no ``graph.json``, so it is absent from
every memory-tier enumeration — including ``_mount_adapters_from_slots``
(``paramem/server/app.py``), which is what keeps a donor out of inference.
That exclusion is now a property of the concept ("not a memory tier")
rather than an accident of a glob failing to match a hidden directory.

The trailing ``b<base8>`` is a digest over the base model id, so donors for
different base models occupy different stores instead of overwriting one
another within a shared topology directory. Directory identity is a cache
bucket only; applicability is decided by :func:`donor_checkpoint_valid`
against the slot's recorded manifest."""

DONOR_META_FILENAME: str = "donor_meta.json"
DONOR_RECIPE_ID: str = "crowded_cluster_v1"
"""Identifies the donor recipe as a whole (fixture content, synthesis
logic, and the fixed hyperparameters below) -- recorded in every
checkpoint's meta (``build_donor``) and checked by
:func:`donor_checkpoint_valid` against the CURRENT value of this constant.
A checkpoint recorded under a different (or missing) recipe id is
rejected as invalid regardless of base-model-id/topology/regeneration
matches, so the next measured-cold fold rebuilds it -- bumping this
constant is therefore how an operator deliberately invalidates every
existing donor checkpoint after a recipe change (a fixture edit, a
synthesis-logic change, or one of the hyperparameters below)."""
DONOR_RECIPE_LEARNING_RATE: float = 1e-4
"""The donor's own training learning rate -- fixed by the recipe, NEVER
read from a live tier's ``AdapterConfig.learning_rate`` (see
:func:`build_donor`'s docstring). Matches the episodic tier's
shipped LR (``configs/server.yaml.example``), which is what Test 20's
donor-uplift evidence measured (``benchmarking.md``, "Test 20") -- an
operator edit to any tier's own ``learning_rate`` can no longer silently
change the donor recipe. Changing the recipe deliberately means editing
this constant (and bumping :data:`DONOR_RECIPE_ID`)."""
DONOR_RECIPE_DROPOUT: float = 0.0
"""The donor's own training dropout -- fixed by the recipe for the same
reason as :data:`DONOR_RECIPE_LEARNING_RATE`: a second non-shape field with
the identical drift property. Pinned independently of the ``AdapterConfig``
dataclass default (``paramem.utils.config``, ``0.0``) so a future default
or config change cannot silently alter the donor recipe -- the two happen
to agree today, but this constant does not read from either the dataclass
default or a live tier's ``AdapterConfig.dropout``. This constant pins the
donor recipe to what production training actually runs at, and to what the
measured donor anchor (Test 20, ``benchmarking.md``) was itself trained
at."""
DONOR_DEFAULT_SEED: int = 42  # matches TrainingConfig.seed's project-standard default
DONOR_MIN_ENTRIES: int = 128
"""Minimum donor population size. This is the training-budget table's
anchored (empirically-validated, not extrapolated) floor
(``paramem.utils.config._BUDGET_TABLE``), which the funnel's unconditional
budget derivation applies to every fold including the donor's own build.
128 is also the divergence-depth population size the failing production
clusters need to be represented at (see the module docstring's "Why the
donor exists")."""

# Transient PEFT adapter slot names. Never collide with a production tier
# name, an interim slot (``episodic_interim_<stamp>``), or the existing
# transient-slot conventions this file's callers already use
# (``f"{tier}_backup"``, ``f"{adapter_name}_verify"``, HF's own
# ``"in_training"`` staging slot).
DONOR_BUILD_ADAPTER_NAME: str = "_donor_build"
"""Transient slot :func:`build_donor` trains on before saving + deleting it.
Excluded from the seeding hook by name (``paramem.training.consolidation``)
so training the donor itself never recursively re-triggers donor seeding."""

DONOR_LOAD_ADAPTER_NAME: str = "_donor_seed"
"""Transient slot the seeding hook loads the donor checkpoint into before
copying its weights into the real target and deleting the slot."""

_FIXTURE_PATH = Path(__file__).with_name("donor_fixture.json")

# --- Synthetic content pools -------------------------------------------------
# Purely invented; every entry below is verified disjoint from
# donor_fixture.json's own 21 subject/object values (no pool draw can ever
# reproduce a fixture value verbatim -- see the module docstring's "Fixture
# provenance" section and tests/test_donor.py::TestDonorEntries). Used by
# _synthesize_block to extend the fixture's structural shape to
# N >= DONOR_MIN_ENTRIES entries.
#
# Only 10 of 12 calendar months are listed -- "may" and "october" are the
# two months that appear in the real (never-committed) source capture this
# fixture was anonymized from. Month names are unavoidably a small closed
# real-world vocabulary (there is no "fictional month"), so this is
# defense-in-depth rather than a strict requirement: excluding the two
# source months keeps every generated date maximally distant from the
# original capture even though month names are not unique identifiers.
_MONTHS: tuple[str, ...] = (
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "november",
    "december",
)
_PRIMARY_NAMES: tuple[str, ...] = (
    "desmond",
    "linnea",
    "cassius",
    "briony",
    "torsten",
    "amara",
    "leopold",
    "saoirse",
    "conrad",
    "ilaria",
    "magnus",
    "perpetua",
    "aurelio",
    "wynne",
)
_SPOUSE_NAMES: tuple[str, ...] = (
    "helena",
    "meredith",
    "adelina",
    "vivienne",
    "rosalind",
    "seraphina",
    "florentine",
    "imogen",
    "ottilie",
    "wilhelmina",
)
_CHILD_NAMES: tuple[str, ...] = (
    "eli",
    "finn",
    "oskar",
    "milo",
    "arlo",
    "beau",
    "silas",
    "rune",
    "asa",
    "ivo",
)
_PLACE_PAIRS: tuple[tuple[str, str], ...] = (
    ("mossvale", "brindleport"),
    ("brackwood", "calderfen"),
    ("thornleigh", "marrowick"),
    ("hollowmere", "greywick"),
    ("fenbridge", "oldmarsh"),
    ("crowsmere", "ravenholt"),
)
_PROJECTS: tuple[tuple[str, str], ...] = (
    ("halden works", "cascade alpha guidance rig"),
    ("thistledown labs", "obsidian beta control array"),
    ("northwind labs", "halcyon tier four control suite"),
    ("cobalt ventures", "argent tier one planning module"),
)
_EXPERTISE_PHRASES: tuple[str, ...] = (
    "adaptive trajectory planning",
    "latent feature synthesis",
    "cross-modal alignment modeling",
    "acoustic event recognition",
    "stochastic policy refinement",
    "symbolic reasoning pipelines",
    "terrain aware locomotion control",
    "distributed sensor calibration",
    "probabilistic risk estimation",
    "multi agent coordination",
    "embedded control synthesis",
    "predictive maintenance analytics",
    "human robot interaction design",
    "edge inference acceleration",
)
_BACKGROUND_PHRASES: tuple[str, ...] = (
    "fluid dynamics modeling",
    "computational chemistry",
    "materials science",
    "control theory",
    "signal processing",
    "industrial design",
)
_UNIVERSITIES: tuple[str, ...] = (
    "thornfield academy of technology",
    "ashcombe polytechnic",
    "brindlewood academy",
    "caldermere college",
    "wrenfield institute of technology",
    "hallowmere polytechnic",
)
_COUNTRIES: tuple[str, ...] = (
    "amberholt",
    "caldoria",
    "norvenne",
    "estoria",
    "brennmark",
    "solveigia",
)
_HOBBIES: tuple[str, ...] = (
    "sailing",
    "beekeeping",
    "orienteering",
    "pottery",
    "astronomy",
    "falconry",
    "woodworking",
    "fencing",
    "birdwatching",
    "climbing",
)


def _load_fixture() -> list[dict]:
    """Load ``donor_fixture.json``'s 21 template entries (key/subject/predicate/object)."""
    with _FIXTURE_PATH.open() as f:
        return json.load(f)


def donor_topology_id(lora_shape: dict) -> str:
    """Canonical, filesystem-safe identity of a LoRA topology.

    Sole input: a :func:`~paramem.models.loader._lora_shape_fields` dict
    (the project's single topology descriptor). ``target_modules`` is
    sorted, matching ``ensure_adapter_matching``'s set comparison
    (``paramem.models.loader``, ``ensure_adapter_matching``) — module
    ORDER is not a topology difference.

    Args:
        lora_shape: ``{"r", "lora_alpha", "target_modules"}`` as returned
            by :func:`~paramem.models.loader._lora_shape_fields`.

    Returns:
        A string of the form ``r{rank}-a{alpha}-{n}mod-{digest}`` where
        ``n`` is the target-module count and ``digest`` is an 8-hex-char
        SHA-256 prefix over the normalized (sorted-modules) shape.

    Raises:
        KeyError: *lora_shape* is missing ``r`` or ``lora_alpha``.
        TypeError: ``r``/``lora_alpha`` cannot be coerced to ``int``.
        ValueError: ``r``/``lora_alpha`` cannot be coerced to ``int``
            (e.g. a non-numeric string).
    """
    normalized = {
        "r": int(lora_shape["r"]),
        "lora_alpha": int(lora_shape["lora_alpha"]),
        "target_modules": sorted(lora_shape.get("target_modules") or []),
    }
    digest = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()[:8]
    n = len(normalized["target_modules"])
    return f"r{normalized['r']}-a{normalized['lora_alpha']}-{n}mod-{digest}"


def iter_donor_stores(adapter_root: "Path | str") -> "list[tuple[str, Path]]":
    """Return ``(store_name, store_dir)`` for every donor store under *adapter_root*.

    The donor counterpart of
    :func:`~paramem.memory.interim_adapter.iter_interim_dirs`: enumeration
    is per store KIND (only the kind knows its own naming), while everything
    downstream — capture, restore, slot resolution, retention — is the
    shared adapter-store machinery. Callers that must treat every adapter
    store alike (the backup bundle writer and the restore sweep) use this to
    discover donors instead of naming them, so adding or removing a donor
    store needs no change on their side.

    Sorted by name for deterministic bundle contents.
    """
    root = Path(adapter_root)
    if not root.is_dir():
        return []
    return sorted(
        (p.name, p) for p in root.iterdir() if p.is_dir() and p.name.startswith(DONOR_STORE_PREFIX)
    )


def donor_base_digest(base_model_id: str) -> str:
    """Return the 8-hex-char store-name digest for *base_model_id*.

    Keeps donors for different base models in different stores. A digest
    (not the raw id) because a HuggingFace model id contains ``/`` and is
    unbounded in length, neither of which a directory name tolerates.
    """
    return hashlib.sha256(base_model_id.encode()).hexdigest()[:8]


def donor_store_dir(adapter_root: "Path | str", base_model_id: str, lora_shape: dict) -> Path:
    """Return the donor adapter store for ONE (base model, topology) pair.

    ``<adapter_root>/donor-<topology_id>-b<base8>`` — a sibling of the main
    tier stores, holding promoted ``<stamp>/`` slots exactly as they do (see
    :data:`DONOR_STORE_PREFIX`). Resolve the live slot inside it with
    ``find_live_slot(store, "")``: :func:`build_donor` stamps every donor
    manifest with an empty ``registry_sha256`` (a donor has no key
    registry), which is the case that resolver documents an empty
    *live_registry_sha256* as matching.

    Args:
        adapter_root: The adapter root whose donor stores this resolves
            against — ``ConsolidationLoop.donor_adapter_root``.
        base_model_id: The base model's ``config._name_or_path``.
        lora_shape: The target tier's :func:`~paramem.models.loader._lora_shape_fields`
            dict — determines which topology this resolves to.
    """
    store_name = (
        f"{DONOR_STORE_PREFIX}{donor_topology_id(lora_shape)}-b{donor_base_digest(base_model_id)}"
    )
    return Path(adapter_root) / store_name


def _triples_hash(entries: list[dict]) -> str:
    """Canonical SHA-256 hex digest of a donor triple set.

    Order-independent: entries are reduced to ``(key, subject, predicate,
    object)`` tuples and sorted before hashing, so the digest is stable
    even if a future change to :func:`donor_entries`'s iteration order
    (without changing content) would otherwise flip it. Used by
    :func:`build_donor` (recorded as the meta's ``triples_hash``) and by
    :func:`donor_checkpoint_valid`'s regeneration check, which compares a
    freshly-regenerated ``donor_entries(seed, n)`` call against the
    recorded hash to catch silent generator drift -- a code change to the
    recipe that would produce different content for the same seed/n
    without touching ``base_model_id`` or ``lora_shape``.
    """
    canonical = sorted((e["key"], e["subject"], e["predicate"], e["object"]) for e in entries)
    payload = json.dumps(canonical, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _manifest_lora_shape(manifest) -> dict:
    """Return a :func:`donor_topology_id`-shaped dict from a manifest's ``lora``.

    One direction only: the manifest is the recorded shape, and
    :func:`~paramem.models.loader._lora_shape_fields` is the live one. Both
    feed the SAME :func:`donor_topology_id`, so there is one topology
    comparison, not two shape schemas.
    """
    return {
        "r": manifest.lora.rank,
        "lora_alpha": manifest.lora.alpha,
        "target_modules": list(manifest.lora.target_modules),
    }


def donor_checkpoint_valid(store_dir: Path, base_model_id: "str | None", lora_shape: dict) -> bool:
    """True when a donor checkpoint exists under *store_dir*, its weights
    verify against their recorded digest, its manifest matches both
    *base_model_id* and *lora_shape*, and regenerating its recorded
    ``(seed, n_requested)`` through :func:`donor_entries` reproduces its
    recorded triple set exactly.

    Args:
        store_dir: The value returned by :func:`donor_store_dir`.
        base_model_id: The CURRENT base model's ``config._name_or_path``,
            compared against the slot manifest's ``base_model.repo``.
            ``None`` (base id unresolved) always returns ``False`` — LoRA
            weights do not transfer across bases, and an unresolved id means
            the comparison cannot be trusted either way.
        lora_shape: The TARGET tier's shape fields, from
            :func:`~paramem.models.loader._lora_shape_fields` (rank, alpha,
            target_modules — the SAME function ``ensure_adapter_matching``
            compares a resident adapter against; one implementation, not a
            second shape check), compared against the slot manifest's own
            ``lora``. *store_dir* is already keyed by (base model, topology)
            by the caller (:func:`donor_store_dir`), so this comparison
            catches a slot that was hand-moved into the wrong store rather
            than crash later inside ``copy_adapter_weights`` on a
            tensor-shape mismatch.

    Validity requires ALL of: (1) ``find_live_slot`` resolves a slot
    carrying ``adapter_model.safetensors``, a readable ``meta.json``
    manifest, and a parseable ``donor_meta.json``;
    (2) ``manifest.base_model.repo == base_model_id``; (3)
    ``meta["recipe"] == DONOR_RECIPE_ID`` -- an operator bump of
    :data:`DONOR_RECIPE_ID` (after editing the fixture, the synthesis
    logic, or either fixed hyperparameter) invalidates every existing
    checkpoint outright, regardless of how the other checks would score,
    so the next measured-cold fold rebuilds from the new recipe; (4) the
    weights on disk hash to the recorded ``meta["weights_sha256"]`` -- the
    ONE check that reads the artifact this checkpoint exists to hand over
    rather than metadata about it, so a truncated or corrupt file is
    rebuilt here instead of raising inside the caller's load; a checkpoint
    recording no digest is unverifiable and therefore invalid; (5) the
    manifest's recorded shape's topology id (:func:`donor_topology_id`)
    equals the current shape's topology id — order-insensitive on
    ``target_modules``;
    (6) regeneration -- ``donor_entries(meta["seed"], meta["n_requested"])``,
    canonically hashed (:func:`_triples_hash`), matches the recorded hash.
    Checkpoints written before ``triples_hash`` existed in the meta schema
    (additive, no schema break) are compared by hashing their own recorded
    ``meta["triples"]`` instead of requiring the new field. A regeneration
    mismatch means :func:`donor_entries`'s generator changed content for the
    same recorded seed WITHOUT a recipe-id bump -- the checkpoint would
    otherwise seed content other than what its recorded seed claims, so it
    is rejected and rebuilt, not just a base-id/shape/recipe-id drift.

    Never raises: this is boundary error handling for an on-disk artifact
    whose shape is not guaranteed once corrupted (partial write, manual
    edit, future schema drift) -- ANY malformed meta shape reads as
    invalid rather than propagating, matching the corrupt-checkpoint ->
    rebuild contract both callers (the seeding hook and its tests) rely
    on. This covers not just missing fields but wrong-typed ones: a
    non-numeric ``n_requested`` (e.g. a string), list-shaped ``triples``
    entries instead of dicts, an entry missing one of
    ``key``/``subject``/``predicate``/``object``, or a manifest whose
    ``lora`` is malformed enough that :func:`donor_topology_id` itself
    raises (``KeyError``/``TypeError``/``ValueError``/``AttributeError``)
    -- the topology-id comparison is evaluated inside the SAME ``try`` as
    the regeneration check so this stays true. An unreadable or
    schema-invalid ``meta.json`` is caught as ``ManifestError`` at the
    read, on the same never-raise contract.

    Returns:
        ``False`` on any of: no slot, missing artifacts, unreadable
        manifest, unparseable meta, base id mismatch, recipe id mismatch
        (including missing), a missing or mismatched weights digest,
        topology mismatch (including a malformed recorded shape), a meta
        missing ``seed`` or ``n_requested`` (cannot be regenerated), any
        other malformed meta shape (see above), or a regeneration hash
        mismatch. ``True`` only when every check passes.
    """
    from paramem.adapters.manifest import (
        ManifestError,
        find_live_slot,
        read_manifest,
    )

    if base_model_id is None:
        return False
    slot = find_live_slot(Path(store_dir), "")
    if slot is None:
        return False
    meta_path = slot / DONOR_META_FILENAME
    weights_path = slot / "adapter_model.safetensors"
    if not meta_path.exists() or not weights_path.exists():
        return False
    try:
        manifest = read_manifest(slot)
    except ManifestError:
        return False
    if manifest.base_model.repo != base_model_id:
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("recipe") != DONOR_RECIPE_ID:
        return False

    # The weights are the artifact this checkpoint EXISTS to hand over, and
    # every check above reads metadata about them rather than the bytes
    # themselves. Without this, a truncated or corrupt safetensors file
    # validates, and the failure surfaces inside the caller's load — far past
    # the point where "invalid -> rebuild" is still available.
    recorded_weights_hash = meta.get("weights_sha256")
    if recorded_weights_hash is None:
        return False
    if hashlib.sha256(weights_path.read_bytes()).hexdigest() != recorded_weights_hash:
        return False

    seed = meta.get("seed")
    n_requested = meta.get("n_requested")
    if seed is None or n_requested is None:
        return False
    try:
        if donor_topology_id(_manifest_lora_shape(manifest)) != donor_topology_id(lora_shape):
            return False

        regenerated = donor_entries(seed, n_requested)
        regenerated_hash = _triples_hash(regenerated)

        recorded_hash = meta.get("triples_hash")
        if recorded_hash is None:
            recorded_triples = meta.get("triples")
            if recorded_triples is None:
                return False
            recorded_hash = _triples_hash(recorded_triples)
    except (TypeError, KeyError, ValueError, AttributeError):
        # Boundary error handling for an on-disk artifact whose shape is
        # not guaranteed: donor_meta.json can be malformed in ways that
        # are NOT "field absent" -- a recorded lora_shape malformed enough
        # that donor_topology_id itself raises (see above), n_requested
        # recorded as a non-numeric string (TypeError from donor_entries'
        # arithmetic), triples recorded as list-shaped entries instead of
        # dicts, or an entry missing one of key/subject/predicate/object
        # (both raise from _triples_hash's e["..."] indexing), or a
        # recorded n_requested below DONOR_MIN_ENTRIES (ValueError from
        # donor_entries itself). Every one of these must read as "cannot
        # verify" -> invalid, never propagate -- this function's contract
        # (documented above and relied on by both its callers,
        # _maybe_seed_from_donor and the seeding-hook tests) is
        # False-never-raise; a corrupt meta
        # file must trigger a normal rebuild, not abort the calling fold.
        return False
    return regenerated_hash == recorded_hash


def _allocate_block_keys(used: set[int], count: int) -> list[int]:
    """Return *count* fresh integers not in *used*, from ``1..DONOR_KEY_BAND_WIDTH``.

    Mutates *used* in place (adds the allocated integers) so successive
    calls across blocks never collide, including with the fixture's own
    reserved ``graph101-115`` / ``proc101-106`` range (pre-seeded into *used*
    by :func:`donor_entries`).
    """
    keys: list[int] = []
    candidate = 1
    while len(keys) < count:
        if candidate > DONOR_KEY_BAND_WIDTH:
            raise RuntimeError(
                "donor_entries: reserved key band exhausted -- widen "
                "DONOR_KEY_BAND_WIDTH (paramem.training.donor) to generate a "
                "larger synthetic donor population"
            )
        if candidate not in used:
            keys.append(candidate)
            used.add(candidate)
        candidate += 1
    return keys


def _draw_unique(rng: "random.Random", pool: tuple[str, ...], used: set[str]) -> str:
    """Draw one value from *pool* never returned before across the ENTIRE
    generation run (tracked in *used*, mutated in place, shared across every
    call for the whole ``donor_entries`` invocation — including the
    fixture's own block-0 subjects, pre-seeded by the caller).

    This is what gives named entities (the block's primary subject, spouse,
    child) the same cross-block used-set treatment
    :func:`_allocate_block_keys` already gives key integers: a name is never
    assigned twice, so a single-cardinality predicate whose subject is one
    of these names (``has spouse``, ``graduation date``, ``birth date``)
    can never be asked twice with conflicting objects, and the
    (subject, predicate, object) triple as a whole can never repeat across
    blocks either (H1 fix — a prior version drew these with replacement per
    block, producing duplicate triples trained under multiple keys and
    contradictory birth dates for a reused name).

    When the pool is fully exhausted (more blocks requested than the pool
    has distinct entries), extends deterministically by appending an
    increasing ordinal to the pool's first entry rather than raising —
    :func:`donor_entries` must not fail for a large *n*; distinctness is
    what correctness requires, not a fresh invented word for every entry.
    """
    for candidate in rng.sample(pool, len(pool)):
        if candidate not in used:
            used.add(candidate)
            return candidate
    suffix = 2
    base = pool[0]
    while True:
        candidate = f"{base} {suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        suffix += 1


def _synthesize_block(
    rng: "random.Random",
    graph_keys: list[int],
    proc_keys: list[int],
    used_names: set[str],
) -> list[dict]:
    """Build one 21-entry structural block (15 episodic + 6 procedural) at
    *graph_keys* / *proc_keys*, mirroring ``donor_fixture.json``'s per-subject
    template shape: a crowded 7-wide same-predicate ``expertise`` cluster, a
    2-wide ``background in`` cluster, a named-relative birth-date pair, one
    ``has spouse`` link, and a 3-wide ``has interest`` cluster — the same
    predicate distribution and triples-per-subject as the template.

    The block's primary subject (a fresh name drawn via :func:`_draw_unique`
    from ``_PRIMARY_NAMES``, never the fixture's own ``"speaker0"``) is
    GLOBALLY UNIQUE across every block this generation run produces, as are
    the spouse/child names — see :func:`_draw_unique`'s docstring for why
    this is what actually guarantees zero duplicate (subject, predicate,
    object) triples and zero same-subject conflicting objects across the
    whole generated set, rather than merely reducing their likelihood.
    """
    primary = _draw_unique(rng, _PRIMARY_NAMES, used_names)
    local_place, city = rng.choice(_PLACE_PAIRS)
    company, project = rng.choice(_PROJECTS)
    expertise = rng.sample(_EXPERTISE_PHRASES, 7)
    background = rng.sample(_BACKGROUND_PHRASES, 2)
    university = rng.choice(_UNIVERSITIES)
    grad_month = rng.choice(_MONTHS)
    grad_year = rng.randint(2005, 2020)
    spouse = _draw_unique(rng, _SPOUSE_NAMES, used_names)
    child = _draw_unique(rng, _CHILD_NAMES, used_names)
    spouse_month = rng.choice(_MONTHS)
    spouse_day = rng.randint(1, 28)
    spouse_year = rng.randint(1975, 1995)
    child_month = rng.choice(_MONTHS)
    child_day = rng.randint(1, 28)
    child_year = rng.randint(2015, 2025)
    country = rng.choice(_COUNTRIES)
    hobbies = rng.sample(_HOBBIES, 3)

    g, p = graph_keys, proc_keys
    return [
        {
            "key": f"graph{g[0]}",
            "subject": primary,
            "predicate": "lives in",
            "object": f"{local_place} near {city}",
        },
        {
            "key": f"graph{g[1]}",
            "subject": primary,
            "predicate": "worked on",
            "object": f"{company}'s {project}",
        },
        {
            "key": f"graph{g[2]}",
            "subject": primary,
            "predicate": "expertise",
            "object": expertise[0],
        },
        {
            "key": f"graph{g[3]}",
            "subject": primary,
            "predicate": "expertise",
            "object": expertise[1],
        },
        {
            "key": f"graph{g[4]}",
            "subject": primary,
            "predicate": "expertise",
            "object": expertise[2],
        },
        {
            "key": f"graph{g[5]}",
            "subject": primary,
            "predicate": "expertise",
            "object": expertise[3],
        },
        {
            "key": f"graph{g[6]}",
            "subject": primary,
            "predicate": "expertise",
            "object": expertise[4],
        },
        {
            "key": f"graph{g[7]}",
            "subject": primary,
            "predicate": "expertise",
            "object": expertise[5],
        },
        {
            "key": f"graph{g[8]}",
            "subject": primary,
            "predicate": "expertise",
            "object": expertise[6],
        },
        {
            "key": f"graph{g[9]}",
            "subject": primary,
            "predicate": "background in",
            "object": background[0],
        },
        {
            "key": f"graph{g[10]}",
            "subject": primary,
            "predicate": "background in",
            "object": background[1],
        },
        {
            "key": f"graph{g[11]}",
            "subject": primary,
            "predicate": "graduated from",
            "object": university,
        },
        {
            "key": f"graph{g[12]}",
            "subject": primary,
            "predicate": "graduation date",
            "object": f"{grad_month} {grad_year}",
        },
        {
            "key": f"graph{g[13]}",
            "subject": spouse,
            "predicate": "birth date",
            "object": f"{spouse_month} {spouse_day}, {spouse_year}",
        },
        {
            "key": f"graph{g[14]}",
            "subject": child,
            "predicate": "birth date",
            "object": f"{child_month} {child_day}, {child_year}",
        },
        {"key": f"proc{p[0]}", "subject": primary, "predicate": "has spouse", "object": spouse},
        {
            "key": f"proc{p[1]}",
            "subject": primary,
            "predicate": "has interest",
            "object": hobbies[0],
        },
        {
            "key": f"proc{p[2]}",
            "subject": primary,
            "predicate": "has interest",
            "object": hobbies[1],
        },
        {
            "key": f"proc{p[3]}",
            "subject": primary,
            "predicate": "has interest",
            "object": hobbies[2],
        },
        {"key": f"proc{p[4]}", "subject": primary, "predicate": "resides in", "object": country},
        {"key": f"proc{p[5]}", "subject": primary, "predicate": "lives near", "object": city},
    ]


def donor_entries(seed: int, n: int) -> list[dict]:
    """Deterministically generate at least *n* donor key-triples.

    Pure function of ``(seed, n)`` — no wall-clock, no unseeded ``random``
    calls — so the same arguments always return the bit-identical triple
    set (the reproducibility scope this donor commits to; weight-level
    training determinism is a separate, heavier, opt-in concern — see the
    module docstring). Extends ``donor_fixture.json``'s 21-entry template
    structurally: the template itself is always block 0 (its keys
    ``graph101-115`` / ``proc101-106`` and its subjects ``speaker0`` /
    ``corinne`` / ``theo`` are reproduced verbatim), and additional
    21-entry blocks (:func:`_synthesize_block`) are appended until the total
    reaches *n*, each keyed from fresh integers in ``1..DONOR_KEY_BAND_WIDTH``
    that avoid every key already used by an earlier block, and each carrying
    a primary/spouse/child identity that is GLOBALLY UNIQUE across the whole
    run (never reused, including never re-drawing the fixture's own
    ``speaker0``/``corinne``/``theo``) — see :func:`_draw_unique`. Returns
    whole blocks only — never truncates a block (and therefore never
    truncates a crowded cluster) to hit *n* exactly, so the returned count
    is ``>= n``, a multiple of 21.

    Invariants (see ``tests/test_donor.py::TestDonorEntries``): zero
    duplicate ``(subject, predicate, object)`` triples anywhere in the
    returned set, at every seed; no subject carries conflicting objects for
    a single-cardinality predicate (``has spouse``, ``graduation date``,
    ``birth date``).

    Args:
        seed: PRNG seed for ``random.Random`` — the sole source of
            variation between calls.
        n: Minimum number of entries to return. Must be
            ``>= DONOR_MIN_ENTRIES`` (see that constant's docstring).

    Returns:
        List of ``{"key", "subject", "predicate", "object"}`` dicts, length
        a multiple of 21 and ``>= n``.

    Raises:
        ValueError: ``n < DONOR_MIN_ENTRIES``.
        RuntimeError: the reserved key band cannot fit the requested block
            count for one of the two prefixes.
    """
    if n < DONOR_MIN_ENTRIES:
        raise ValueError(
            f"donor_entries: n must be >= {DONOR_MIN_ENTRIES}; got {n} "
            "(see DONOR_MIN_ENTRIES's docstring)"
        )
    template = _load_fixture()
    block_size = len(template)
    num_blocks = -(-n // block_size)  # ceil division, stdlib-only
    rng = random.Random(seed)

    graph_used = {
        int(e["key"].removeprefix("graph")) for e in template if e["key"].startswith("graph")
    }
    proc_used = {
        int(e["key"].removeprefix("proc")) for e in template if e["key"].startswith("proc")
    }
    # Seed the name-uniqueness set with the fixture's OWN block-0 subjects so
    # no synthetic block can ever redraw "speaker0"/"corinne"/"theo" and
    # collide (contradict) with the real fixture's own facts about them.
    used_names: set[str] = {e["subject"] for e in template}

    entries: list[dict] = [dict(e) for e in template]
    for _ in range(num_blocks - 1):
        graph_keys = _allocate_block_keys(graph_used, 15)
        proc_keys = _allocate_block_keys(proc_used, 6)
        entries.extend(_synthesize_block(rng, graph_keys, proc_keys, used_names))
    return entries


def _drop_transient_slot(model, name: str, *, fallback_adapter: str) -> None:
    """Delete transient PEFT slot *name* from *model* if resident.

    Mirrors the switch-off-before-delete pattern
    ``ConsolidationLoop._verify_saved_adapter_from_disk`` already uses for
    its own transient ``f"{adapter_name}_verify"`` slot: PEFT refuses to
    leave the model with no active adapter, so *fallback_adapter* (a
    production tier guaranteed resident) is activated first when *name* is
    currently active.
    """
    from paramem.models.loader import active_adapter_name, switch_adapter

    if name in model.peft_config:
        if active_adapter_name(model) == name and fallback_adapter in model.peft_config:
            switch_adapter(model, fallback_adapter)
        model.delete_adapter(name)


def build_donor(
    loop: "ConsolidationLoop",
    *,
    adapter_config,
    seed: int = DONOR_DEFAULT_SEED,
    n: int = DONOR_MIN_ENTRIES,
) -> Path:
    """Train a fresh donor adapter, at the TARGET tier's topology, through
    the shared funnel and persist it.

    Trains :func:`donor_entries` on a transient PEFT slot
    (``DONOR_BUILD_ADAPTER_NAME``, created at *adapter_config*'s topology)
    via ``loop._train_tier_adapter`` — the SAME funnel every production fold
    uses, so budget derivation (``paramem.utils.config.budget_for``) and the
    recall-early-stop callback apply with no special case. The training
    itself runs at fixed recipe hyperparameters
    (:data:`DONOR_RECIPE_LEARNING_RATE`, :data:`DONOR_RECIPE_DROPOUT`) —
    ``dataclasses.replace(adapter_config, learning_rate=..., dropout=...)`` —
    never at whichever tier's own config happened to trigger the build:
    the donor is a *recipe* artifact, only its topology (rank,
    alpha, target_modules) is target-derived. The build-slot name is
    excluded from the seeding hook by name (``paramem.training.consolidation``)
    so this call can never recursively re-trigger donor seeding on itself.

    ``sweep_orphan_pending`` runs first on the topology directory, cleaning
    any ``.pending/`` residue an earlier interrupted build left behind (the
    same startup hygiene production tier directories get, applied here at
    build time instead since the donor is never touched at boot).

    ``create_adapter``/``switch_adapter`` run INSIDE the ``try`` (not before
    it): deleting the transient slot on ANY failure past that point —
    including a ``switch_adapter`` failure — requires the ``finally`` to
    already be active, otherwise a mid-setup failure would leak
    ``DONOR_BUILD_ADAPTER_NAME`` on the model permanently. The initial
    ``_drop_transient_slot`` call before the ``try`` is safe to leave outside
    it: it only clears a slot a PRIOR failed call left behind and switches to
    ``"episodic"`` first when needed — episodic is unconditionally created
    at ``ConsolidationLoop`` construction (``_ensure_adapters``) and is never
    itself deleted, so it is always resident and never the model's last
    adapter; switching to it can only fail if the model itself is broken,
    a condition this function cannot recover from regardless.

    If ``_train_tier_adapter`` returns ``(None, None)`` (no training
    examples — should not happen for a well-formed donor population, but
    checked defensively) or its metrics carry ``aborted=True`` (thermal
    throttle / operator pause), NO checkpoint is written and this function
    raises :class:`DonorBuildIncomplete` instead — persisting a checkpoint
    from a run that never actually trained would seed every future
    measured-cold fold from LoRA-zero-equivalent weights forever, silently
    defeating the whole mechanism.

    On success, saves via :func:`~paramem.models.loader.atomic_save_adapter`
    (the same primitive every production tier save uses — atomic write,
    age-envelope encryption when the daily identity is loaded) into the
    donor store (:func:`donor_store_dir`). Storage metadata is a
    ``meta.json`` :class:`~paramem.adapters.manifest.AdapterManifest` built
    by the ONE provider every save path uses
    (:func:`~paramem.adapters.manifest.build_manifest_for`) — base-model and
    tokenizer fingerprints and the LoRA shape live there, and its
    ``registry_sha256`` is empty because a donor carries no key registry,
    which is what lets ``find_live_slot`` resolve the promoted slot.
    ``donor_meta.json`` alongside it carries only what is donor-specific
    (seed, recipe, the resulting triple set, its canonical hash
    (:func:`_triples_hash`, read back by :func:`donor_checkpoint_valid`'s
    regeneration check), and the weights SHA-256 that same function
    verifies the bytes against) — never a second copy of the base model or
    shape. Every other slot within the SAME store is then pruned through
    the shared ``ConsolidationLoop._prune_old_slots`` at ``keep=0`` (exactly
    one donor artifact per (base model, topology) persists at a time), and
    the transient slot is deleted in a ``finally`` regardless of outcome.
    Nothing outside this store is touched: another base model's or
    topology's store is left intact, because a config change makes a donor
    inapplicable, never wrong.

    Args:
        loop: The live :class:`~paramem.training.consolidation.ConsolidationLoop`
            (its ``model``/``tokenizer``/``training_config``/
            ``episodic_config``/``semantic_config``/``procedural_config``
            supply everything this needs — no separate model load).
        adapter_config: The TARGET tier's ``AdapterConfig`` — the same
            object ``_train_tier_adapter`` was called with at the call
            site that triggered this build. Its shape (rank, alpha,
            target_modules) determines which topology directory this
            build writes into; its ``learning_rate``/``dropout`` are
            NEVER used (see above).
        seed: Forwarded to :func:`donor_entries`.
        n: Forwarded to :func:`donor_entries`.

    Returns:
        Path to the newly-written donor checkpoint slot.

    Raises:
        DonorBuildIncomplete: Training did not complete (see above); no
            checkpoint was written.
    """
    from dataclasses import replace as _dataclasses_replace

    from paramem.adapters.manifest import build_manifest_for
    from paramem.backup.backup import sweep_orphan_pending
    from paramem.models.loader import (
        _lora_shape_fields,
        atomic_save_adapter,
        create_adapter,
        switch_adapter,
    )

    entries = donor_entries(seed, n)
    base_model_id = getattr(loop.model.get_base_model().config, "_name_or_path", None)

    lora_shape = _lora_shape_fields(adapter_config)
    store_dir = donor_store_dir(loop.donor_adapter_root, base_model_id, lora_shape)
    sweep_orphan_pending(store_dir)
    build_name = DONOR_BUILD_ADAPTER_NAME
    _drop_transient_slot(loop.model, build_name, fallback_adapter="episodic")

    recipe_config = _dataclasses_replace(
        adapter_config,
        learning_rate=DONOR_RECIPE_LEARNING_RATE,
        dropout=DONOR_RECIPE_DROPOUT,
    )

    try:
        loop.model = create_adapter(loop.model, recipe_config, build_name)
        switch_adapter(loop.model, build_name)
        metrics, _recall_state = loop._train_tier_adapter(
            entries,
            adapter_name=build_name,
            adapter_config=recipe_config,
            training_config=loop.training_config,
            output_dir=store_dir / ".training_scratch",
            run_name="donor-build",
            phase_name="donor-build",
        )
        if metrics is None or metrics.get("aborted"):
            raise DonorBuildIncomplete(
                f"donor training did not complete (metrics={metrics!r}) -- no checkpoint persisted"
            )
        store_dir.mkdir(parents=True, exist_ok=True)
        # A donor's storage metadata is an AdapterManifest like any other
        # store's: base-model + tokenizer fingerprint and LoRA shape come from
        # the ONE provider every save path uses, and the empty
        # registry_sha256 (a donor carries no key registry) is what makes
        # find_live_slot resolve its promoted slot.
        manifest = build_manifest_for(
            loop.model,
            loop.tokenizer,
            build_name,
            registry_path=None,
            registry_sha256_override="",
            key_count=len(entries),
            adapter_root=Path(loop.donor_adapter_root),
        )
        final_slot = atomic_save_adapter(loop.model, store_dir, build_name, manifest=manifest)
        weights_sha256 = hashlib.sha256(
            (final_slot / "adapter_model.safetensors").read_bytes()
        ).hexdigest()
        # Donor-SPECIFIC content only: base model and LoRA shape live in the
        # manifest above and are never copied here, so there is one recorded
        # answer to "what is this checkpoint for", not two that can drift.
        meta = {
            "seed": seed,
            "recipe": DONOR_RECIPE_ID,
            "n_requested": n,
            "triples": entries,
            "triples_hash": _triples_hash(entries),
            "weights_sha256": weights_sha256,
        }
        (final_slot / DONOR_META_FILENAME).write_text(json.dumps(meta))
        # Same retention routine every tier store uses; keep=0 because a donor
        # rebuild in the SAME store only happens when the prior slot can never
        # be valid again (recipe bump, generator drift, corrupt weights) -- a
        # base or topology change resolves to a DIFFERENT store and leaves its
        # predecessor untouched.
        loop._prune_old_slots(store_dir, final_slot, keep=0)
        logger.info(
            "build_donor: trained + persisted donor checkpoint at %s (n=%d, seed=%d, base=%s)",
            final_slot,
            len(entries),
            seed,
            base_model_id,
        )
        return final_slot
    finally:
        _drop_transient_slot(loop.model, build_name, fallback_adapter="episodic")


def load_donor_into_transient_slot(model, store_dir: Path, transient_name: str) -> None:
    """Load the donor checkpoint under *store_dir* onto *model* as *transient_name*.

    Mirrors ``ConsolidationLoop._verify_saved_adapter_from_disk``'s PEFT
    pitfall handling: ``model.load_adapter`` (never
    ``PeftModel.from_pretrained``, which nests tensor names on a
    multi-adapter model), the ``base_model_name_or_path`` patch PEFT skips
    for second-and-later adapters, and
    :func:`~paramem.models.loader._adapter_slot_for_load`'s transparent
    decrypt-to-memfd for the age-encrypted safetensors
    :func:`build_donor` writes.

    Args:
        model: The live ``PeftModel``.
        store_dir: The value returned by :func:`donor_store_dir`.
        transient_name: Adapter name to mount the checkpoint as (the
            caller deletes it once the copy-in completes).

    Raises:
        FileNotFoundError: No donor slot exists under *store_dir*.
    """
    from paramem.adapters.manifest import find_live_slot
    from paramem.models.loader import _adapter_slot_for_load

    slot = find_live_slot(Path(store_dir), "")
    if slot is None:
        raise FileNotFoundError(f"No donor checkpoint found under {store_dir}")
    with _adapter_slot_for_load(slot) as load_path:
        model.load_adapter(str(load_path), adapter_name=transient_name)
    if model.peft_config[transient_name].base_model_name_or_path is None:
        base_name = getattr(model.get_base_model().config, "_name_or_path", None)
        if base_name:
            model.peft_config[transient_name].base_model_name_or_path = base_name
