"""Donor-adapter lifecycle: synthetic key-triple generation, checkpoint
persistence, and seed-training through the shared funnel.

A NEW module boundary, not a code move: nothing was relocated out of
``paramem.training.consolidation`` (7000+ lines; per project rule an
oversized file is a defect to split, not enlarge) to create this file — the
funnel this module trains through (``ConsolidationLoop._train_tier_adapter``)
stays exactly where it was. This module owns the donor's three concerns
instead: (1) deterministically generating the synthetic crowded-cluster
training population (:func:`donor_entries`), (2) persisting and validating
the trained donor checkpoint outside every boot-scan glob
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
**unconditionally** — the reservation does not depend on
``donor_seeding_enabled``, so a later flag flip can never collide with keys
already minted in the 1-``width`` band. The width bounds the maximum
key-surface *divergence depth* the donor can teach (the separating variable
in the production failures is cluster depth, not absolute key magnitude —
see ``experiments/test20_smallN_cold_gate.py``): a 1-200 band spans depths
1-3, the entire depth range observed in production. Widen this constant
(and rebuild the donor checkpoint) if a depth-4+ cluster is ever observed.
ONE shared constant (not two independently-tunable per-prefix widths)
because both prefixes are reserved at the SAME width symmetrically
("Donors own graph1-200 AND proc1-200") — two constants that could drift
apart would only reintroduce an asymmetry that decision rejected.

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
The donor is NOT built "at first boot" as a separate step — there is a
SINGLE call site (``ConsolidationLoop._maybe_seed_from_donor``, inside
``_train_tier_adapter``): when ``donor_seeding_enabled`` is on and the
checkpoint is missing or stale, :func:`build_donor` runs INLINE, synchronously,
before that fold's own training. The practical consequence: the first
measured-cold fold after the flag is turned on (or after a base-model swap)
absorbs a full donor training run (the same budget as any other fold, e.g.
30 epochs at the anchored bucket) IN ADDITION TO its own training, so that
fold takes roughly twice as long as a normal fold. Every fold after that
reuses the persisted checkpoint and pays no extra cost until the checkpoint
is invalidated again (base-model swap, or a LoRA shape edit — see
:func:`donor_checkpoint_valid`).
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
``max(DONOR_KEY_FLOOR, high_water_from_store)``. Unconditional: this floor
applies regardless of ``donor_seeding_enabled`` (see module docstring)."""

# --- Donor checkpoint location ----------------------------------------------
DONOR_CHECKPOINT_DIRNAME: str = "_donor"
"""Sub-directory of ``ConsolidationLoop.output_dir`` (== ``config.adapter_dir``
in production) holding the donor checkpoint. Chosen so NEITHER production
boot-scan path can ever mount it: ``_mount_adapters_from_slots``
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
the donor checkpoint along with any other unrecognised directory.
Operational consequence: the first measured-cold fold after a restore
rebuilds the donor inline again (see :func:`build_donor`'s call site,
``ConsolidationLoop._maybe_seed_from_donor``) — expected, not a bug; the
checkpoint is a rebuildable cache, never a source of truth."""

DONOR_META_FILENAME: str = "donor_meta.json"
DONOR_RECIPE_ID: str = "crowded_cluster_v1"
DONOR_DEFAULT_SEED: int = 42  # matches TrainingConfig.seed's project-standard default
DONOR_MIN_ENTRIES: int = 128
"""Minimum donor population size. This is the training-budget table's
anchored (empirically-validated, not extrapolated) floor
(``paramem.utils.config._BUDGET_TABLE``) — but that bucket only applies
when the funnel derives the budget FROM this count, i.e.
``training_config.budget_derivation_enabled`` is True. When that flag is
False (the ship default for BOTH ``budget_derivation_enabled`` and
``donor_seeding_enabled``), ``budget_for`` ignores the key count entirely
and the donor simply trains at ``training_config.num_epochs`` (default 30
— numerically the same as the anchored bucket's epoch count, but for an
unrelated reason: the config default, not this floor). 128 is still the
right floor regardless of the budget-derivation flag: it is also the
divergence-depth population size the failing production clusters need to
be represented at (see the module docstring's "Why the donor exists")."""

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


def donor_checkpoint_dir(adapter_root: "Path | str") -> Path:
    """Return the donor checkpoint root under *adapter_root* (``loop.output_dir``)."""
    return Path(adapter_root) / DONOR_CHECKPOINT_DIRNAME


def _latest_donor_slot(checkpoint_dir: Path) -> "Path | None":
    """Return the most-recently-written donor slot under *checkpoint_dir*, or ``None``.

    Donor slots are written by :func:`build_donor` via
    :func:`~paramem.models.loader.atomic_save_adapter`, which names each slot
    with its own sortable ``YYYYMMDD-HHMMSS`` stamp
    (``paramem.models.loader._make_slot_ts``) — the lexicographically-greatest
    child directory is therefore the newest. Unlike production tiers'
    ``find_live_slot`` (registry-sha256 matching across possibly-divergent
    slots), no hash resolution is needed here: there is exactly one donor
    artifact at a time (:func:`build_donor` prunes every prior slot after a
    successful save), rebuilt wholesale, never partially updated.
    Directories whose name starts with ``.`` (PEFT/HF scratch — ``.pending``,
    the training scratch dir) are skipped.
    """
    if not checkpoint_dir.is_dir():
        return None
    slots = sorted(p for p in checkpoint_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    return slots[-1] if slots else None


def donor_checkpoint_valid(
    checkpoint_dir: Path, base_model_id: "str | None", lora_shape: dict
) -> bool:
    """True when a donor checkpoint exists under *checkpoint_dir* and matches
    both *base_model_id* and *lora_shape*.

    Args:
        checkpoint_dir: The value returned by :func:`donor_checkpoint_dir`.
        base_model_id: The CURRENT base model's ``config._name_or_path``.
            ``None`` (base id unresolved) always returns ``False`` — LoRA
            weights do not transfer across bases, and an unresolved id means
            the comparison cannot be trusted either way.
        lora_shape: The CURRENT episodic tier's shape fields, from
            :func:`~paramem.models.loader._lora_shape_fields` (rank, alpha,
            target_modules — the SAME function ``ensure_adapter_matching``
            compares a resident adapter against; one implementation, not a
            second shape check). An operator rank/target-modules edit changes
            this, and a checkpoint trained at the OLD shape must be rejected
            here rather than crash later inside
            ``copy_adapter_weights_subset`` on a tensor-shape mismatch.

    Returns:
        ``False`` when no slot exists, the slot is missing its weights or
        meta file, the meta file does not parse, or the meta's
        ``base_model_id`` / ``lora_shape`` differs from the current values
        (including a never-recorded meta, which reads as a mismatch).
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
    return meta.get("lora_shape") == lora_shape


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
            ``>= DONOR_MIN_ENTRIES`` (see that constant's docstring for the
            exact dependency on ``budget_derivation_enabled``).

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


def _prune_other_donor_slots(checkpoint_root: Path, keep: Path) -> None:
    """Delete every donor slot under *checkpoint_root* except *keep*.

    Called by :func:`build_donor` after a successful save: unlike production
    tiers (which retain a bounded number of prior slots for rollback,
    ``ConsolidationLoop._prune_old_slots`` /
    ``consolidation.training_keep_prior_slots``), the donor is a rebuildable
    cache with exactly one live artifact at a time by design (see
    :func:`_latest_donor_slot`'s docstring) — there is no rollback use case
    for an old donor checkpoint, so nothing is retained. Directories
    starting with ``.`` (``.pending``, the training scratch dir) are left
    alone; this only prunes prior promoted slots.
    """
    if not checkpoint_root.is_dir():
        return
    for child in checkpoint_root.iterdir():
        if child.is_dir() and not child.name.startswith(".") and child != keep:
            shutil.rmtree(child, ignore_errors=True)


def build_donor(
    loop: "ConsolidationLoop",
    *,
    seed: int = DONOR_DEFAULT_SEED,
    n: int = DONOR_MIN_ENTRIES,
) -> Path:
    """Train a fresh donor adapter through the shared funnel and persist it.

    Trains :func:`donor_entries` on a transient PEFT slot
    (``DONOR_BUILD_ADAPTER_NAME``, episodic topology — ``loop.episodic_config``)
    via ``loop._train_tier_adapter`` — the SAME funnel every production fold
    uses, so budget derivation (``paramem.utils.config.budget_for``) and the
    recall-early-stop callback apply with no special case. The build-slot
    name is excluded from the seeding hook by name
    (``paramem.training.consolidation``) so this call can never recursively
    re-trigger donor seeding on itself.

    ``sweep_orphan_pending`` runs first on the checkpoint root, cleaning any
    ``.pending/`` residue an earlier interrupted build left behind (the same
    startup hygiene production tier directories get, applied here at build
    time instead since the donor is never touched at boot).

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
    age-envelope encryption when the daily identity is loaded) into
    :func:`donor_checkpoint_dir`, writes ``donor_meta.json`` (seed, recipe,
    the resulting triple set, weights SHA-256, base model id, and the
    CURRENT episodic tier's LoRA shape fields — see
    :func:`donor_checkpoint_valid`) alongside the weights, prunes every
    other donor slot (:func:`_prune_other_donor_slots` — exactly one donor
    artifact persists at a time), then deletes the transient slot in a
    ``finally`` regardless of outcome.

    Args:
        loop: The live :class:`~paramem.training.consolidation.ConsolidationLoop`
            (its ``model``/``tokenizer``/``episodic_config``/``training_config``
            supply everything this needs — no separate model load).
        seed: Forwarded to :func:`donor_entries`.
        n: Forwarded to :func:`donor_entries`.

    Returns:
        Path to the newly-written donor checkpoint slot.

    Raises:
        DonorBuildIncomplete: Training did not complete (see above); no
            checkpoint was written.
    """
    from paramem.backup.backup import sweep_orphan_pending
    from paramem.models.loader import (
        _lora_shape_fields,
        atomic_save_adapter,
        create_adapter,
        switch_adapter,
    )

    entries = donor_entries(seed, n)
    base_model_id = getattr(loop.model.get_base_model().config, "_name_or_path", None)

    checkpoint_root = donor_checkpoint_dir(loop.output_dir)
    sweep_orphan_pending(checkpoint_root)
    build_name = DONOR_BUILD_ADAPTER_NAME
    _drop_transient_slot(loop.model, build_name, fallback_adapter="episodic")

    try:
        loop.model = create_adapter(loop.model, loop.episodic_config, build_name)
        switch_adapter(loop.model, build_name)
        metrics, _recall_state = loop._train_tier_adapter(
            entries,
            adapter_name=build_name,
            adapter_config=loop.episodic_config,
            training_config=loop.training_config,
            output_dir=checkpoint_root / ".training_scratch",
            run_name="donor-build",
            phase_name="donor-build",
        )
        if metrics is None or metrics.get("aborted"):
            raise DonorBuildIncomplete(
                f"donor training did not complete (metrics={metrics!r}) -- no checkpoint persisted"
            )
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        final_slot = atomic_save_adapter(loop.model, checkpoint_root, build_name)
        weights_sha256 = hashlib.sha256(
            (final_slot / "adapter_model.safetensors").read_bytes()
        ).hexdigest()
        meta = {
            "seed": seed,
            "recipe": DONOR_RECIPE_ID,
            "n_requested": n,
            "triples": entries,
            "weights_sha256": weights_sha256,
            "base_model_id": base_model_id,
            "lora_shape": _lora_shape_fields(loop.episodic_config),
        }
        (final_slot / DONOR_META_FILENAME).write_text(json.dumps(meta))
        _prune_other_donor_slots(checkpoint_root, keep=final_slot)
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
