"""Shared fixtures for the serving read door a ``_probe_and_reason`` test runs through.

The serving read path forks once on ``inference.preload_cache``
(``paramem.server.inference._probe_and_reason``): the ``True`` arm (the
``ServerConfig`` default) reads the RAM mirror via
``MemoryStore.probe_cache``, the ``False`` arm builds a memory source and
reads through ``MemoryStore.probe_source``.  A test that stubs one door
while running under the other probes nothing and recalls nothing — the
stub is inert and the test's subject is never exercised.  These helpers
make the choice explicit and supply what the chosen door demands.
"""

from __future__ import annotations

from paramem.server.config import ServerConfig


def live_door_config() -> ServerConfig:
    """A ``ServerConfig`` pinned to the LIVE serving read door.

    ``inference.preload_cache=False`` selects the arm that builds a memory
    source and reads through ``MemoryStore.probe_source`` — the door
    :func:`stub_live_door_probe` targets.  Named rather than inherited: the
    ``ServerConfig`` default is the other door, under which the stub is
    never called and no key is ever probed.
    """
    config = ServerConfig()
    config.inference.preload_cache = False
    return config


def fact_entry(key: str) -> dict:
    """The triple the stubbed source answers for *key*.

    One definition for the two sides that must agree: the source result
    and the fingerprint registered for it.  The live door rebuilds a
    candidate fingerprint from exactly these fields
    (``paramem.memory.entry.verify_confidence``), so a triple that drifts
    from its registered fingerprint is dropped at the door.
    """
    return {"key": key, "subject": "speaker0", "predicate": "recalls", "object": key}


def seed_live_door_fingerprints(store, plan, *, entry_for=fact_entry) -> None:
    """Register every plan key's fingerprint on the tier it is probed under.

    The live door proves each fact before serving it:
    ``MemoryStore.probe_source`` drops a key whose registry carries no
    fingerprint (a trained tier must always be provable), so a store
    seeded with bookkeeping alone recalls nothing and no fact reaches the
    rendered context.  Seeding every plan key mirrors production, where
    every trained key carries one; which keys actually answer is then the
    source's business (see ``failing_keys``).

    *entry_for* builds the triple whose fingerprint is registered — it must
    be the same triple the test's source answers with, since the gate
    compares the two.  It defaults to :func:`fact_entry`, the shape
    :func:`stub_live_door_probe` serves; a test whose source answers its
    own fact shape passes that shape's builder instead.
    """
    from paramem.memory.entry import entry_simhash

    for step in plan.steps:
        for key in step.keys_to_probe:
            store.put_simhash(step.adapter_name, key, entry_simhash(entry_for(key)))


def stub_live_door_probe(monkeypatch, *, failing_keys=frozenset()) -> dict:
    """Stub the grouped-probe primitive the live door reads through.

    Returns a dict the caller can read ``["keys_by_adapter"]`` from after
    ``_probe_and_reason`` returns — the exact set of keys the probe call
    received, and the proof that probing happened at all.

    Every key answers :func:`fact_entry`'s triple plus a ``fact_text`` the
    rendering assertions read, so pairing with
    :func:`seed_live_door_fingerprints` is what lets a result past the
    door's confidence gate.  Keys in *failing_keys* answer the source's
    own failure shape instead, which the door normalizes to a miss.

    ``is_self_referential`` is pinned False so a reply carrying an
    ``[ESCALATE]`` tag never reaches the personal-referent encoder.
    """
    captured: dict = {}

    def fake_grouped(model, tokenizer, keys_by_adapter, **kwargs):
        captured["keys_by_adapter"] = {k: list(v) for k, v in keys_by_adapter.items()}
        results = {}
        for keys in keys_by_adapter.values():
            for k in keys:
                if k in failing_keys:
                    results[k] = {"key": k, "failure_reason": "no_match"}
                else:
                    results[k] = {
                        **fact_entry(k),
                        "fact_text": f"fact about {k}",
                        "confidence": 1.0,
                    }
        return results

    monkeypatch.setattr("paramem.memory.probe.probe_keys_grouped_by_adapter", fake_grouped)
    monkeypatch.setattr("paramem.models.loader.switch_adapter", lambda model, name: None)
    monkeypatch.setattr(
        "paramem.memory.store.MemoryStore.read_simhash_registry_from_disk",
        staticmethod(lambda path, *, cached=False: {}),
    )
    monkeypatch.setattr(
        "paramem.server.inference.is_self_referential", lambda text, **kwargs: False
    )
    return captured


def forbid_both_read_doors(monkeypatch) -> None:
    """Make either serving read door an immediate failure.

    A path that returns before the ``preload_cache`` fork proves "nothing
    was probed" only by forbidding BOTH doors — the cache door
    (``probe_cache``) and the live door (``probe_source``) alike.  Naming
    one door would leave the other free to serve, and naming a door that
    no longer exists would make the guard raise on setup instead of
    guarding anything.
    """
    for door in ("probe_cache", "probe_source"):

        def exploding(self, *args, _door=door, **kwargs):
            raise AssertionError(f"MemoryStore.{_door} must not be called when zero keys survive")

        monkeypatch.setattr(f"paramem.memory.store.MemoryStore.{door}", exploding)
