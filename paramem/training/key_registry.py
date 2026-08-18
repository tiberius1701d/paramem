"""Per-tier key registry for key-addressable replay.

Each adapter tier (episodic, semantic, procedural, episodic_interim_<stamp>)
gets its own :class:`KeyRegistry` instance.  The owning loop holds these in
a ``dict[tier_name, KeyRegistry]`` and persists them as
``<adapter_dir>/<tier>/indexed_key_registry.json`` — tier ownership is
encoded by the file path, not by a field on the record.

The registry's contents are scoped to one tier:

- ``active_keys`` — keys assigned to this tier.
- ``stale`` — withheld key ids: markers reserved against re-minting, carried
  in ``list_known()`` for bookkeeping retention, excluded from every
  enumeration that serves, trains, merges, projects or fingerprints.  A
  marker holds only the id; it has no timestamp, no fingerprint and no other
  field.
- ``simhash`` — the tier's ONE fingerprint map: per-key 64-bit fingerprint
  for the SimHash confidence gate, for active keys only.  A withheld id
  carries no fingerprint.

Cross-tier operations (which tier owns key X, dropping interim-tier
registries at the end of a full cycle) live on the
:class:`paramem.training.consolidation.ConsolidationLoop` since the loop is
what holds the per-tier dict.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class KeyRegistry:
    """Tracks one tier's active keys and withheld (stale) key ids.

    Keys can be in one of three states:
    - **active**: in ``_active_keys``, enumerated by all normal paths and the
      only population that carries a SimHash fingerprint.
    - **stale**: in ``_stale`` — a withheld id, holding nothing but the id
      itself.  Excluded from ``__contains__``, ``list_active`` and the
      SimHash fingerprint map, but still retained in ``list_known()``'s
      active∪stale union — every consumer that reads this tier's
      ``key_metadata.json`` rows (the fold's shadow writer, the boot loader,
      the parity gate) scopes its retention set to ``list_known()`` so a
      withheld id's bookkeeping row survives exactly as long as the id
      itself is known.  A marker ends at its tier's own rebuild, which seeds
      the tier's working copy from active keys alone.
    - **removed**: not present anywhere; via :meth:`remove` (hard erasure —
      production callers are a fold's fate decision on a key belonging to
      a tier the same event rebuilds, via
      :class:`~paramem.training.consolidation.ConsolidationLoop`, and
      :meth:`adopt_key_from`'s own call on the source registry during tier
      promotion; the door ``POST /speaker/forget`` runs is a stale-mark via
      :meth:`stale`, not a hard erasure).

    SimHash fingerprints have one home: :attr:`_simhash`, the tier's active
    keys only.  A withheld id carries no fingerprint — :meth:`stale` drops it
    on the active→stale transition, and :meth:`add` / :meth:`set_simhash`
    refuse a withheld id rather than let it re-enter either population. The
    private accessor :meth:`_simhashes` is intentionally private — the only
    public path to the fingerprint map is :meth:`MemoryStore.tier_simhashes`.
    """

    def __init__(self) -> None:
        self._active_keys: list[str] = []
        # Withheld ids.  A marker holds only the id — no timestamp, no
        # fingerprint, no other field.  Keys here are EXCLUDED from normal
        # enumeration and the SimHash fingerprint map.
        self._stale: set[str] = set()
        # The tier's ONE fingerprint map — active keys only.
        self._simhash: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Active-key set
    # ------------------------------------------------------------------

    def _refuse_withheld(self, key: str, context: str) -> None:
        """Raise when *key* is a withheld id — the one refusal guard shared
        by :meth:`add` and :meth:`set_simhash`.

        Registering a withheld id as active, or attaching a fingerprint to
        one, is a contradiction the registry refuses rather than a state it
        repairs.

        Raises:
            ~paramem.memory.store.BookkeepingInvariantViolation: *key* is in
                this tier's withheld set.
        """
        if key in self._stale:
            from paramem.memory.store import raise_bookkeeping_invariant_violation

            raise_bookkeeping_invariant_violation(None, [key], context)

    def add(self, key: str) -> None:
        """Register a new active key for this tier (idempotent).

        Raises:
            ~paramem.memory.store.BookkeepingInvariantViolation: *key* is
                withheld in this tier — see :meth:`_refuse_withheld`.
        """
        self._refuse_withheld(key, "key registry add: key is withheld in this tier")
        if key not in self._active_keys:
            self._active_keys.append(key)

    def remove(self, key: str) -> None:
        """Hard-remove a key from this tier (active list, stale set, simhash).

        Reached when an acting site's fate decision on a key is "removed"
        rather than "stale" — the owning tier the fold rebuilds is already
        re-deriving from an active set the id is not in — and by
        :meth:`adopt_key_from` (tier promotion, via the source registry).  A
        removed key is GONE — neither active nor stale.  Does not raise on
        absent keys.
        """
        self._active_keys = [k for k in self._active_keys if k != key]
        self._stale.discard(key)
        self._simhash.pop(key, None)

    def adopt_key_from(self, source: "KeyRegistry", key: str) -> None:
        """Move an ACTIVE key's membership and fingerprint out of *source*
        into this registry.

        Hard-removes *key* from *source* (:meth:`remove` — the one primitive
        a key changing tier uses; see
        ``paramem.training.consolidation.WorkingTier.adopt_key_from``, its
        one caller) and registers it here with its fingerprint carried, if
        it had one.

        Active-onlyness is a CALLER property, not a check this method
        performs: the staging layer refuses a non-active key before this
        method is ever reached — ``_promote_working_keys`` iterates the
        source's active keys, and ``_route_absorbed_keyed_fact`` receives a
        key found on a merged-graph edge.  This method itself moves whatever
        key *source* knows, active or stale.

        *source* not knowing *key* (neither active nor stale) is a
        violation of the same invariant as the destination-already-knows
        case below: a key changing tier is a key some caller believes is
        active in *source*, so *source* not tracking it under any standing
        is a contradiction, not a state to route around.

        This registry already knowing *key* — in EITHER partition, active
        or stale — before the adoption is likewise a contradiction, not a
        state to repair: the single-tier-ownership invariant means a key
        changing tier is moving OUT of exactly one registry INTO exactly
        one other, never landing on a registry that already tracks it under
        any standing.  Every check here raises via
        :func:`~paramem.memory.store.raise_bookkeeping_invariant_violation`
        before either registry is mutated.

        Raises:
            ~paramem.memory.store.BookkeepingInvariantViolation: *source*
                does not know *key* (neither active nor stale); or this
                registry already knows *key* (active or stale) at the time
                *source* is found to know it too.
        """
        if not source.knows(key):
            from paramem.memory.store import raise_bookkeeping_invariant_violation

            raise_bookkeeping_invariant_violation(
                None, [key], "registry key adoption: source does not know this key"
            )
        if self.knows(key):
            from paramem.memory.store import raise_bookkeeping_invariant_violation

            raise_bookkeeping_invariant_violation(
                None, [key], "registry key adoption: destination already knows this key"
            )
        fingerprint = source.simhash_for(key)
        source.remove(key)
        self.add(key)
        if fingerprint is not None:
            self.set_simhash(key, fingerprint)

    def working_copy(self, *, active_only: bool) -> "KeyRegistry":
        """Build an independent registry to seed one fold's working universe.

        The registry-layer half of a fold's per-tier recall
        (``ConsolidationLoop._recall_working_tiers``): a tier this event
        REBUILDS is seeded ``active_only=True`` — its withheld markers end
        at this rebuild (:meth:`stale`'s own docstring: "a marker ends at
        its tier's own rebuild"), so the copy carries active keys and their
        fingerprints only, with no stale ids at all. A tier this event only
        dedups against (never rebuilds) is seeded ``active_only=False`` —
        the full active ∪ stale universe, markers included, since an
        interim event must not release a main tier's markers it does not
        rebuild.

        Independence: the returned registry shares no container with
        ``self`` — ``_active_keys``, ``_stale`` and ``_simhash`` are each
        freshly built collections (the values they hold are ``str``/``int``,
        so no deeper copy is owed). Mutating the copy never reaches ``self``
        and vice versa, in either direction, for the lifetime of the fold
        that mutates it.

        Args:
            active_only: ``True`` to seed active keys (and their
                fingerprints) only, dropping every withheld marker; ``False``
                to seed the full known universe, markers included.

        Returns:
            A new, independent :class:`KeyRegistry`.
        """
        working = KeyRegistry()
        working._active_keys = list(self._active_keys)
        working._simhash = dict(self._simhash)
        if not active_only:
            working._stale = set(self._stale)
        return working

    def stale(self, key: str) -> None:
        """Withhold *key*: remove it from the active set and mint a marker
        that reserves its id (idempotent).

        A withheld id is excluded from :meth:`list_active`,
        :meth:`__contains__`, :meth:`__len__` and the SimHash fingerprint
        map, but retained in ``_stale`` — and so in :meth:`list_known` — so
        its bookkeeping row survives beside it.  Its fingerprint does not
        survive the transition: no reader needs a withheld id's fingerprint,
        since unservability is enumeration-based, not gate-based.

        Calling ``stale`` on an already-stale or absent key is a no-op.
        """
        if key in self._active_keys:
            self._active_keys = [k for k in self._active_keys if k != key]
            self._simhash.pop(key, None)
            self._stale.add(key)

    def knows(self, key: str) -> bool:
        """True when *key* is legitimately tracked by this tier — active OR
        withheld (stale).

        Distinct from :meth:`__contains__` (active-only, serving semantics): a
        withheld id is still KNOWN — its ``key_metadata.json`` row is retained
        on disk, even though it carries no fingerprint.
        Membership-legitimacy consumers (orphan checks, bookkeeping
        retention) must use this; serving/enumeration consumers keep using
        :meth:`__contains__` / :meth:`list_active`.
        """
        return key in self._active_keys or key in self._stale

    def list_known(self) -> list[str]:
        """All keys this tier legitimately tracks — active keys in
        registration order, then withheld ids in sorted order.

        Equivalent to :meth:`list_active` + :meth:`list_stale` but expressed
        as a single call so callers can canonically enumerate active ∪
        withheld without hand-rolling the union.
        """
        return list(self._active_keys) + sorted(self._stale)

    def list_active(self) -> list[str]:
        """Return all active keys in this tier in registration order."""
        return list(self._active_keys)

    def list_stale(self) -> list[str]:
        """Return this tier's withheld ids, sorted."""
        return sorted(self._stale)

    # ------------------------------------------------------------------
    # SimHash fingerprints — the tier's one fingerprint map (active keys only)
    # ------------------------------------------------------------------

    def set_simhash(self, key: str, fingerprint: int) -> None:
        """Store the SimHash fingerprint for *key*.

        Raises:
            ~paramem.memory.store.BookkeepingInvariantViolation: *key* is
                withheld in this tier — see :meth:`_refuse_withheld`.
        """
        self._refuse_withheld(key, "key registry set_simhash: key is withheld in this tier")
        self._simhash[key] = fingerprint

    def drop_simhash(self, key: str) -> None:
        """Remove the SimHash fingerprint for *key*."""
        self._simhash.pop(key, None)

    def simhash_for(self, key: str) -> int | None:
        """Return the SimHash fingerprint for *key*, or ``None`` when *key*
        has no stored fingerprint — including every withheld id, which
        carries none by design."""
        return self._simhash.get(key)

    def has_simhash(self, key: str) -> bool:
        """``True`` when *key* has a stored fingerprint."""
        return key in self._simhash

    def replace_simhashes(self, new_map: dict[str, int]) -> None:
        """Bulk-replace this registry's active-key fingerprint map.

        Validates every id in *new_map* against this registry's withheld
        set BEFORE mutating anything, then swaps the whole map in one step:
        a naive clear-then-set loop that raises partway through would leave
        the fingerprint map truncated — fewer entries than either the old
        map or the intended new one — silently failing every one of those
        keys' SimHash confidence gate rather than refusing the call
        outright. The full-map swap is a direct assignment (not
        ``update()``): a bulk replace must drop every fingerprint not
        present in *new_map*, the same as the clear-then-set loop it
        replaces — just performed atomically, after validation.

        The withheld check is a set-level equivalent of :meth:`_refuse_withheld`
        (that guard takes one key; this call validates a whole map in one
        pass) raising through the same
        :func:`~paramem.memory.store.raise_bookkeeping_invariant_violation`
        helper, under its own context, so a caller sees every offending id
        in one raise rather than only the first.

        Args:
            new_map: The replacement fingerprint map, ``{key: fingerprint}``.

        Raises:
            ~paramem.memory.store.BookkeepingInvariantViolation: *new_map*
                names a withheld id in this registry.  Neither the old map
                nor a partial new one is left in place.
        """
        withheld = set(new_map) & self._stale
        if withheld:
            from paramem.memory.store import raise_bookkeeping_invariant_violation

            raise_bookkeeping_invariant_violation(
                None,
                sorted(withheld),
                "key registry replace_simhashes: key is withheld in this tier",
            )
        self._simhash = dict(new_map)

    def _simhashes(self) -> dict[str, int]:
        """The tier's one fingerprint map ``{key: fp}`` — active keys only.

        This also covers replay-disabled stores where ``_active_keys`` is
        empty but ``_simhash`` is populated via :meth:`replace_simhashes`
        (reached from :meth:`MemoryStore.replace_simhashes_in_tier`).

        PRIVATE — intentionally not a public accessor.  The only public path
        to a fingerprint set is :meth:`MemoryStore.tier_simhashes`.  Used by
        :meth:`save_bytes` (the on-disk serialization) and
        :meth:`load_simhashes` (the on-disk leaf, which projects a
        freshly-parsed payload through this same accessor).
        """
        return dict(self._simhash)

    def __len__(self) -> int:
        return len(self._active_keys)

    def __contains__(self, key: str) -> bool:
        return key in self._active_keys

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist this tier's registry to ``path`` (encryption-aware)."""
        from paramem.backup.encryption import write_infra_bytes

        payload = self.save_bytes()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_infra_bytes(path, payload)
        logger.info("Key registry saved to %s (%d keys)", path, len(self._active_keys))

    def save_bytes(self) -> bytes:
        """Serialize this tier's registry to canonical UTF-8 JSON bytes.

        The bytes are what :meth:`save` writes.  The atomic consolidation path
        hashes these bytes to obtain ``registry_sha256`` before writing the
        adapter manifest, then calls :meth:`save_from_bytes` with the same
        payload so the on-disk hash is byte-identical to the manifested one.

        The ``"stale"`` field is a sorted JSON array of withheld key ids —
        sorting is load-bearing: set iteration order over strings varies with
        ``PYTHONHASHSEED`` across processes, and these bytes are hashed into
        the slot manifest (``registry_sha256``) and compared across process
        boundaries.  The ``"simhash"`` field holds the tier's ONE fingerprint
        map (``_simhashes()``) — active keys only.  This is the unified
        on-disk layout; the separate ``simhash_registry.json`` file has been
        removed.

        A file written by this method will be read back by :meth:`load`,
        which REQUIRES ``"active_keys"``, ``"stale"`` and ``"simhash"`` to
        be present — a file missing any of the three is refused with
        :class:`ValueError`, not silently treated as a fresh/empty store.
        """
        data = {
            "active_keys": self._active_keys,
            "stale": sorted(self._stale),
            "simhash": self._simhashes(),
        }
        return json.dumps(data, indent=2).encode("utf-8")

    def save_from_bytes(self, payload: bytes, path: str | Path) -> None:
        """Write pre-serialized registry bytes to ``path``.

        Second half of the serialization-barrier split: the bytes must
        come from :meth:`save_bytes` so the on-disk content is byte-identical
        to whatever was hashed for the manifest.
        """
        from paramem.backup.encryption import write_infra_bytes

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_infra_bytes(path, payload)
        logger.info("Key registry written from bytes to %s (%d bytes)", path, len(payload))

    @classmethod
    def load(cls, path: str | Path) -> "KeyRegistry":
        """Load a tier's registry from ``path`` (empty registry if absent).

        The single shape predicate for ``indexed_key_registry.json``: an
        ABSENT file loads as an empty registry — the fresh-install contract
        every caller (including the boot walk,
        :meth:`paramem.memory.store.MemoryStore.read_registries_from_disk`)
        depends on. An EXISTING file must be affirmatively KeyRegistry-shaped
        — a dict with a list-valued ``"active_keys"`` of string ids, a
        list-valued ``"stale"`` of string ids, and a dict-valued
        ``"simhash"`` of int fingerprints (``bool`` excluded — it is a
        subtype of ``int`` in Python but never a legitimate fingerprint),
        with no id in both ``"active_keys"`` and ``"stale"`` and no
        ``"simhash"`` entry naming a withheld id — or this raises
        :class:`ValueError` naming the path and what is wrong, rather than
        silently coercing a foreign JSON schema (or a pre-migration file
        whose ``"stale"`` section is a dict of per-id records) into a
        partial or empty registry. :meth:`load_simhashes` delegates here for the
        identical check — there is exactly one shape check for this file.

        Existence-check and read are atomic within this one method — no
        separate helper carries the "confirm existence first" contract that
        nothing else enforced.  Parsing itself is :meth:`load_from_bytes`,
        so a caller that already holds the file's decrypted bytes (e.g.
        :func:`~paramem.memory.increment.build_tier_increment`, which needs
        the verbatim bytes for its own hash too) parses them once instead of
        this method's read-then-parse doing a second read of the same file.

        Raises:
            ValueError: *path* exists but is not KeyRegistry-shaped.
        """
        from paramem.backup.encryption import read_maybe_encrypted

        path = Path(path)
        if not path.exists():
            logger.info("No registry at %s, starting fresh", path)
            return cls()

        # Existence is checked above, not inferred from the parsed payload:
        # a file whose content is the JSON literal ``null`` parses to the
        # same Python ``None`` an absent file would collapse to, and must
        # NOT be read as "fresh" — it is an existing file that fails the
        # shape check below.
        return cls.load_from_bytes(read_maybe_encrypted(path), path=path)

    @classmethod
    def load_from_bytes(cls, payload: bytes, *, path: "str | Path" = "<bytes>") -> "KeyRegistry":
        """Parse a registry from already-read bytes — no file I/O.

        The parse-and-shape-check half of :meth:`load`, split out so a
        caller already holding the file's decrypted bytes parses them once
        rather than reading and decrypting the file a second time.  Shape
        predicate and error shape are identical to :meth:`load` — this IS
        that method's parse step, not a second implementation of it.

        Args:
            payload: The file's raw, already-decrypted bytes.
            path: Used only to name the file in a raised :class:`ValueError`
                or the info log — no file at *path* is read.

        Raises:
            ValueError: *payload* does not parse as a KeyRegistry-shaped
                registry file — a non-dict payload; a missing or wrongly-typed
                ``"active_keys"``, ``"stale"`` or ``"simhash"`` section (this
                is also how a pre-migration file, whose ``"stale"`` section is
                a dict of per-id records, is refused rather than coerced); an
                id present in both ``"active_keys"`` and ``"stale"``; or a
                ``"simhash"`` entry naming a withheld id — see :meth:`load`.
        """
        data = json.loads(payload.decode("utf-8"))

        missing: list[str] = []
        if not isinstance(data, dict):
            missing.append("payload is not a JSON object")
        else:
            active_keys = data.get("active_keys")
            stale_ids = data.get("stale")
            simhash_map = data.get("simhash")
            if not isinstance(active_keys, list) or not all(
                isinstance(k, str) for k in active_keys
            ):
                missing.append("list-valued 'active_keys' of string ids")
            if not isinstance(stale_ids, list) or not all(isinstance(k, str) for k in stale_ids):
                missing.append("list-valued 'stale' of string ids")
            if not isinstance(simhash_map, dict) or not all(
                isinstance(fp, int) and not isinstance(fp, bool) for fp in simhash_map.values()
            ):
                missing.append("dict-valued 'simhash' of int fingerprints")
            if not missing:
                overlap = set(active_keys) & set(stale_ids)
                if overlap:
                    missing.append(
                        f"id(s) present in both 'active_keys' and 'stale': {sorted(overlap)!r}"
                    )
                withheld_fp = set(simhash_map) & set(stale_ids)
                if withheld_fp:
                    missing.append(f"'simhash' names withheld id(s): {sorted(withheld_fp)!r}")
        if missing:
            raise ValueError(
                f"{path} is not a KeyRegistry-shaped registry file "
                f"(missing: {'; '.join(missing)}) — refusing to coerce a "
                "foreign registry schema into a KeyRegistry"
            )

        registry = cls._from_payload(data)
        logger.info(
            "Key registry loaded from %s: %d active keys, %d stale, %d fingerprints",
            path,
            len(registry._active_keys),
            len(registry._stale),
            len(registry._simhash),
        )
        return registry

    @classmethod
    def load_simhashes(cls, path: str | Path) -> dict[str, int]:
        """Read the tier's one fingerprint map out of ONE registry file.

        The single leaf for "read the SimHash fingerprints out of an
        ``indexed_key_registry.json``".  Both the per-file callers (the trial
        consolidation gates, which are handed a path and must be told when it
        is the wrong one) and the adapter-tree walk
        (:meth:`paramem.memory.store.MemoryStore.read_simhash_registry_from_disk`)
        go through here, so the encryption read, the on-disk shape and the
        wrong-file guard exist exactly once — delegated entirely to
        :meth:`load`, which is the single shape check for this file.

        Returns the tier's active-key fingerprint map — the same map
        :meth:`save_bytes` serialises under ``"simhash"``.

        Args:
            path: Path to one tier's ``indexed_key_registry.json``.

        Returns:
            ``{key: fingerprint}``.  Empty when *path* does not exist (fresh
            install / tier not yet trained) or its ``"simhash"`` map is empty.

        Raises:
            ValueError: When *path* exists but is not KeyRegistry-shaped —
                see :meth:`load`.  A tier's ``key_metadata.json``
                (``{"tier_cycle": int, "keys": {...}}`` — per-key
                bookkeeping, never a fingerprint) has no ``"active_keys"``/
                ``"stale"``/``"simhash"`` section, so pointing this method at
                it fails immediately instead of silently un-gating every key
                it was supposed to verify.
        """
        return cls.load(path)._simhashes()

    @classmethod
    def _from_payload(cls, data: dict) -> "KeyRegistry":
        """Build a registry from a parsed ``indexed_key_registry.json`` payload.

        The only place that knows the on-disk field layout written by
        :meth:`save_bytes`.  :meth:`load_from_bytes` is the sole caller and
        has already enforced the full shape predicate — dict payload,
        list-valued ``"active_keys"``, list-valued ``"stale"`` of string ids
        with no overlap against ``"active_keys"``, and dict-valued
        ``"simhash"`` of int fingerprints naming no withheld id — before
        calling here, so all three sections are read directly with NO
        default and NO filter: this is the single-shape read, not a second
        chance to coerce a malformed payload.  This method only ever looks up
        ``"active_keys"``, ``"stale"`` and ``"simhash"`` by name — any other
        field present in *data* is simply never accessed, so a registry file
        carrying extra fields beyond those three still loads cleanly instead
        of being rejected.
        """
        registry = cls()
        registry._active_keys = data["active_keys"]
        registry._stale = set(data["stale"])
        for k, fp in data["simhash"].items():
            registry._simhash[k] = fp

        return registry
