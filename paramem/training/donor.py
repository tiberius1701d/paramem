"""Donor-adapter lifecycle: synthetic key-triple generation, checkpoint
persistence, and seed-training through the shared funnel.

A NEW module boundary, not a code move: nothing was relocated out of
``paramem.training.consolidation`` (7000+ lines; per project rule an
oversized file is a defect to split, not enlarge) to create this file — the
funnel this module trains through (``ConsolidationLoop._train_tier_adapter``)
stays exactly where it was. This module owns the donor's three concerns
instead: (1) deterministically generating the synthetic crowded-cluster
training population (:func:`donor_entries`), (2) persisting and validating
each topology's own trained donor checkpoint outside every boot-scan glob
(:func:`donor_checkpoint_dir` / :func:`donor_checkpoint_valid`), and
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
Every LoRA topology gets its OWN donor checkpoint, built lazily by the
same single call site (``ConsolidationLoop._maybe_seed_from_donor``,
inside ``_train_tier_adapter``): when the TARGET tier's topology
checkpoint is missing or stale, :func:`build_donor` runs INLINE,
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
import shutil
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
DONOR_CHECKPOINT_DIRNAME: str = "_donor"
"""Sub-directory of ``ConsolidationLoop.output_dir`` (== ``config.adapter_dir``
in production) holding the donor tree — one level deeper than a single
checkpoint: ``_donor/<topology_id>/<stamp>/``, one topology directory per
distinct LoRA shape (:func:`donor_topology_id`), each holding its own
promoted slot. Chosen so NEITHER production boot-scan path can ever mount
it: ``_mount_adapters_from_slots``
(``paramem/server/app.py``) enumerates a hardcoded 3-tuple
``("episodic", "semantic", "procedural")`` for main tiers and calls
``iter_interim_dirs(config.adapter_dir)`` for interim slots, which globs
only ``<adapter_dir>/episodic/interim_*``
(``paramem/memory/interim_adapter.py``); the legacy-layout detector globs
``<adapter_dir>/episodic_interim_*`` at the top level
(``detect_legacy_adapter_layout``); and the backup safety-bundle path
(``paramem/backup/backup.py``) also iterates the same fixed 3-tuple.
``_donor`` matches none of these three enumeration patterns, so it is
structurally unreachable from every mount/backup path without a dedicated
opt-in reader — none exists.

Stricter than "never mounted": ``restore_bundle``
(``paramem/backup/backup.py``) actively DESTROYS it. Its adapters-root walk
treats any top-level entry that is not one of the three recognised main
tiers as a stray and ``shutil.rmtree``'s it unconditionally (unless listed
in ``infra_paths()``, which ``_donor`` is not) — so a restore always wipes
every topology's donor checkpoint along with any other unrecognised
directory. Operational consequence: the first measured-cold fold of EACH
topology after a restore rebuilds that topology's donor inline again (see
:func:`build_donor`'s call site, ``ConsolidationLoop._maybe_seed_from_donor``)
— expected, not a bug; each checkpoint is a rebuildable cache, never a
source of truth."""

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


def donor_root(adapter_root: "Path | str") -> Path:
    """Return the donor tree root under *adapter_root* (``loop.output_dir``)."""
    return Path(adapter_root) / DONOR_CHECKPOINT_DIRNAME


def donor_checkpoint_dir(adapter_root: "Path | str", lora_shape: dict) -> Path:
    """Return the checkpoint directory for ONE topology under *adapter_root*.

    ``<adapter_root>/_donor/<topology_id>`` — the leaf is defined in terms
    of :func:`donor_root`, and the topology id comes from
    :func:`donor_topology_id` applied to *lora_shape*.

    Args:
        adapter_root: ``loop.output_dir`` (== ``config.adapter_dir`` in
            production).
        lora_shape: The target tier's :func:`~paramem.models.loader._lora_shape_fields`
            dict — determines which topology's directory this resolves to.
    """
    return donor_root(adapter_root) / donor_topology_id(lora_shape)


def _latest_donor_slot(checkpoint_dir: Path) -> "Path | None":
    """Return the most-recently-written donor slot under *checkpoint_dir*, or ``None``.

    Donor slots are written by :func:`build_donor` via
    :func:`~paramem.models.loader.atomic_save_adapter`, which names each slot
    with its own sortable ``YYYYMMDD-HHMMSS`` stamp
    (``paramem.models.loader._make_slot_ts``) — the lexicographically-greatest
    child directory is therefore the newest. Unlike production tiers'
    ``find_live_slot`` (registry-sha256 matching across possibly-divergent
    slots), no hash resolution is needed here: there is exactly one donor
    artifact per topology at a time (:func:`build_donor` prunes every prior
    slot within the same topology after a successful save), rebuilt
    wholesale, never partially updated.
    Directories whose name starts with ``.`` (PEFT/HF scratch — ``.pending``,
    the training scratch dir) are skipped.
    """
    if not checkpoint_dir.is_dir():
        return None
    slots = sorted(p for p in checkpoint_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    return slots[-1] if slots else None


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


def donor_checkpoint_valid(
    checkpoint_dir: Path, base_model_id: "str | None", lora_shape: dict
) -> bool:
    """True when a donor checkpoint exists under *checkpoint_dir*, matches
    both *base_model_id* and *lora_shape*, and regenerating its recorded
    ``(seed, n_requested)`` through :func:`donor_entries` reproduces its
    recorded triple set exactly.

    Args:
        checkpoint_dir: The value returned by :func:`donor_checkpoint_dir`.
        base_model_id: The CURRENT base model's ``config._name_or_path``.
            ``None`` (base id unresolved) always returns ``False`` — LoRA
            weights do not transfer across bases, and an unresolved id means
            the comparison cannot be trusted either way.
        lora_shape: The TARGET tier's shape fields, from
            :func:`~paramem.models.loader._lora_shape_fields` (rank, alpha,
            target_modules — the SAME function ``ensure_adapter_matching``
            compares a resident adapter against; one implementation, not a
            second shape check). *checkpoint_dir* is already topology-scoped
            by the caller (:func:`donor_checkpoint_dir`), so this comparison
            catches a slot that was hand-moved into the wrong topology's
            directory (see the module's migration guidance) rather than
            crash later inside ``copy_adapter_weights`` on a tensor-shape
            mismatch.

    Validity requires ALL of: (1) a donor slot exists with both
    ``adapter_model.safetensors`` and a parseable ``donor_meta.json``;
    (2) ``meta["base_model_id"] == base_model_id``; (3)
    ``meta["recipe"] == DONOR_RECIPE_ID`` -- an operator bump of
    :data:`DONOR_RECIPE_ID` (after editing the fixture, the synthesis
    logic, or either fixed hyperparameter) invalidates every existing
    checkpoint outright, regardless of how the other checks would score,
    so the next measured-cold fold rebuilds from the new recipe; (4) the
    recorded shape's topology id (:func:`donor_topology_id`) equals the
    current shape's topology id — order-insensitive on ``target_modules``;
    (5) regeneration -- ``donor_entries(meta["seed"], meta["n_requested"])``,
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
    ``key``/``subject``/``predicate``/``object``, or a recorded
    ``lora_shape`` malformed enough that :func:`donor_topology_id` itself
    raises (``KeyError``/``TypeError``/``ValueError`` -- e.g. ``null``, a
    bare string, or a dict missing ``r``) -- the topology-id comparison is
    evaluated inside the SAME ``try`` as the regeneration check so this
    stays true.

    Returns:
        ``False`` on any of: no slot, missing artifacts, unparseable meta,
        base id mismatch, recipe id mismatch (including missing), topology
        mismatch (including a malformed recorded shape), a meta missing
        ``seed`` or ``n_requested`` (cannot be regenerated), any other
        malformed meta shape (see above), or a regeneration hash mismatch.
        ``True`` only when every check passes.
    """
    if base_model_id is None:
        return False
    slot = _latest_donor_slot(Path(checkpoint_dir))
    if slot is None:
        return False
    meta_path = slot / DONOR_META_FILENAME
    weights_path = slot / "adapter_model.safetensors"
    if not meta_path.exists() or not weights_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("base_model_id") != base_model_id:
        return False
    if meta.get("recipe") != DONOR_RECIPE_ID:
        return False

    seed = meta.get("seed")
    n_requested = meta.get("n_requested")
    if seed is None or n_requested is None:
        return False
    try:
        if donor_topology_id(meta.get("lora_shape")) != donor_topology_id(lora_shape):
            return False

        regenerated = donor_entries(seed, n_requested)
        regenerated_hash = _triples_hash(regenerated)

        recorded_hash = meta.get("triples_hash")
        if recorded_hash is None:
            recorded_triples = meta.get("triples")
            if recorded_triples is None:
                return False
            recorded_hash = _triples_hash(recorded_triples)
    except (TypeError, KeyError, ValueError):
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


def _prune_other_donor_slots(topology_dir: Path, keep: Path) -> None:
    """Delete every donor slot under ONE topology's directory except *keep*.

    Called by :func:`build_donor` after a successful save: unlike production
    tiers (which retain a bounded number of prior slots for rollback,
    ``ConsolidationLoop._prune_old_slots`` /
    ``consolidation.training_keep_prior_slots``), the donor is a rebuildable
    cache with exactly one live artifact per topology at a time by design
    (see :func:`_latest_donor_slot`'s docstring) — there is no rollback use
    case for an old donor checkpoint, so nothing is retained. Directories
    starting with ``.`` (``.pending``, the training scratch dir) are left
    alone; this only prunes prior promoted slots.
    """
    if not topology_dir.is_dir():
        return
    for child in topology_dir.iterdir():
        if child.is_dir() and not child.name.startswith(".") and child != keep:
            shutil.rmtree(child, ignore_errors=True)


def _prune_dead_topologies(root: Path, live_ids: "set[str]") -> None:
    """Delete topology directories under *root* whose id is not in *live_ids*.

    Called by :func:`build_donor` after a successful save, alongside
    :func:`_prune_other_donor_slots`: the per-slot prune bounds each
    topology's OWN history to one artifact, and this bounds the number of
    topology directories the donor tree can ever accumulate. A topology
    stops being live when no tier's current ``AdapterConfig`` maps to it
    any more (an operator rank/target-modules edit, or a tier being
    disabled) — its directory is then garbage: nothing will ever build,
    validate, or load from it again. Directories starting with ``.`` are
    left alone (mirrors :func:`_prune_other_donor_slots`'s dot-prefix
    skip); this also garbage-collects a legacy flat ``_donor/<stamp>/``
    layout (pre-topology-scoping), whose stamp name is never a live
    topology id.
    """
    if not root.is_dir():
        return
    for child in root.iterdir():
        if child.is_dir() and not child.name.startswith(".") and child.name not in live_ids:
            shutil.rmtree(child, ignore_errors=True)


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
    topology directory (:func:`donor_checkpoint_dir`), writes
    ``donor_meta.json`` (seed, recipe, the resulting triple set, its
    canonical hash (:func:`_triples_hash`, read back by
    :func:`donor_checkpoint_valid`'s regeneration check), weights SHA-256,
    base model id, and *adapter_config*'s LoRA shape fields — see
    :func:`donor_checkpoint_valid`) alongside the weights, prunes every
    other slot within the SAME topology (:func:`_prune_other_donor_slots`
    — exactly one donor artifact per topology persists at a time), prunes
    any topology directory that is no longer live
    (:func:`_prune_dead_topologies`, whose live-id set is the shape just
    built here UNION the loop's own three tier configs — the just-built
    shape is included unconditionally so this build can never delete its
    own fresh output even when no tier config happens to match it, e.g. a
    stale or ad-hoc *adapter_config*), then deletes the transient slot in
    a ``finally`` regardless of outcome.

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
    topology_dir = donor_checkpoint_dir(loop.output_dir, lora_shape)
    sweep_orphan_pending(topology_dir)
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
            output_dir=topology_dir / ".training_scratch",
            run_name="donor-build",
            phase_name="donor-build",
        )
        if metrics is None or metrics.get("aborted"):
            raise DonorBuildIncomplete(
                f"donor training did not complete (metrics={metrics!r}) -- no checkpoint persisted"
            )
        topology_dir.mkdir(parents=True, exist_ok=True)
        final_slot = atomic_save_adapter(loop.model, topology_dir, build_name)
        weights_sha256 = hashlib.sha256(
            (final_slot / "adapter_model.safetensors").read_bytes()
        ).hexdigest()
        meta = {
            "seed": seed,
            "recipe": DONOR_RECIPE_ID,
            "n_requested": n,
            "triples": entries,
            "triples_hash": _triples_hash(entries),
            "weights_sha256": weights_sha256,
            "base_model_id": base_model_id,
            "lora_shape": lora_shape,
        }
        (final_slot / DONOR_META_FILENAME).write_text(json.dumps(meta))
        _prune_other_donor_slots(topology_dir, keep=final_slot)
        live_ids = {donor_topology_id(lora_shape)} | {
            donor_topology_id(_lora_shape_fields(c))
            for c in (loop.episodic_config, loop.semantic_config, loop.procedural_config)
            if c is not None
        }
        _prune_dead_topologies(donor_root(loop.output_dir), live_ids)
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


def load_donor_into_transient_slot(model, checkpoint_dir: Path, transient_name: str) -> None:
    """Load the donor checkpoint under *checkpoint_dir* onto *model* as *transient_name*.

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
        checkpoint_dir: The value returned by :func:`donor_checkpoint_dir`.
        transient_name: Adapter name to mount the checkpoint as (the
            caller deletes it once the copy-in completes).

    Raises:
        FileNotFoundError: No donor slot exists under *checkpoint_dir*.
    """
    from paramem.models.loader import _adapter_slot_for_load

    slot = _latest_donor_slot(Path(checkpoint_dir))
    if slot is None:
        raise FileNotFoundError(f"No donor checkpoint found under {checkpoint_dir}")
    with _adapter_slot_for_load(slot) as load_path:
        model.load_adapter(str(load_path), adapter_name=transient_name)
    if model.peft_config[transient_name].base_model_name_or_path is None:
        base_name = getattr(model.get_base_model().config, "_name_or_path", None)
        if base_name:
            model.peft_config[transient_name].base_model_name_or_path = base_name
