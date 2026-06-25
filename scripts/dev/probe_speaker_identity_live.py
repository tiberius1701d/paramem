"""Live-GPU probe — speaker-identity lowercase refactor + render-boundary resolution.

Verifies that the speaker-identity refactor (P3/P4 — lowercase ``speaker{N}``
everywhere) and the render-boundary display-name substitution work end-to-end.

  Phase 1 (CPU) — identity + migration.
    a) Register two speakers via ``SpeakerStore.enroll``: active ``Alex``
       (speaker0) and third-party ``Dana`` (speaker1).
    b) Assert minted IDs are lowercase ``speaker{N}`` — NOT cased.
    c) Assert ``resolve_speaker_name("speaker0")`` returns ``"Alex"`` (works
       directly on the lowercase token — the old re-casing blocker is gone).
    d) Migration check: build a synthetic v5 profile-store dict with a CASED
       ``"Speaker0"`` key, write it to disk, reload through ``SpeakerStore``,
       and assert the loaded profile key is the lowercase ``"speaker0"`` (v5→v6
       migration). Assert the embeddings survive so a matching embedding still
       hits at high confidence.

  Phase 2 (GPU) — render-resolution end-to-end.
    Extract a short conversation where:
      * speaker0 (Alex) contributes a fact ABOUT speaker1 (Dana) as subject.
      * speaker0 contributes a fact where Dana is the object.
    After extraction and keyed-entry build, directly call ``entry_fact_text``
    with the production ``_speaker_resolver`` closure to verify that raw
    ``speaker0`` / ``speaker1`` tokens are replaced with ``Alex`` / ``Dana``.
    Also verify that an anonymous speaker registered via ``register_anonymous``
    renders as ``"another speaker"`` (the THIRD-PARTY-DESCRIPTOR value), never
    as the raw token.

    For the MemoryStore.probe render paths, directly pre-populate a
    ``MemoryStore`` with SPO entries and call ``store.probe()`` with
    ``source=None`` (cache-hit path) and with ``speaker_resolver`` set, then
    assert the returned ``fact_text`` contains the display names.

Outputs outputs/speaker_identity_probe/<timestamp>/results.json.
No live-server data is read or written; everything lives in a tempdir.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
from experiments.utils.test_harness import BENCHMARK_MODELS, setup_logging  # noqa: E402
from paramem.backup.encryption import read_maybe_encrypted  # noqa: E402
from paramem.graph.name_match import is_speaker_id  # noqa: E402
from paramem.graph.prompts import _load_speaker_directive_section  # noqa: E402
from paramem.memory.entry import entry_fact_text  # noqa: E402
from paramem.memory.store import MemoryStore  # noqa: E402
from paramem.models.loader import load_base_model  # noqa: E402
from paramem.server.session_buffer import SessionBuffer  # noqa: E402
from paramem.server.speaker import _PROFILE_VERSION, SpeakerStore  # noqa: E402
from paramem.training.consolidation import ConsolidationLoop  # noqa: E402
from paramem.utils.config import (  # noqa: E402
    AdapterConfig,
    ConsolidationConfig,
    TrainingConfig,
)

logger = logging.getLogger("probe_speaker_identity_live")

# Distinct 256-dim near-orthogonal embeddings (one non-zero component each).
EMB_ALEX = [1.0] + [0.0] * 255
EMB_DANA = [0.0, 1.0] + [0.0] * 254
EMB_ANON = [0.0, 0.0, 1.0] + [0.0] * 253

# THIRD_PARTY_DESCRIPTOR — loaded the same way inference.py loads it so the
# assertion is decoupled from the raw string.
THIRD_PARTY_DESCRIPTOR: str = _load_speaker_directive_section("THIRD-PARTY-DESCRIPTOR")

# Turns for the GPU extraction phase.
# Alex (speaker0) talks about Dana (speaker1) as a third-party subject and object.
# Kept deliberately short so the extraction model can parse them cleanly.
TURNS_RENDER = [
    ("user", "Dana lives in Berlin and works as an engineer.", None),
    ("assistant", "Interesting. How do you know Dana?", None),
    ("user", "I know Dana from university.", None),
]


def _tier_cfg(rank: int = 8) -> AdapterConfig:
    """Return a minimal AdapterConfig for a single-session extraction probe."""
    return AdapterConfig(
        rank=rank,
        alpha=2 * rank,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def _build_buffer_and_store(tmp_dir: Path) -> tuple[SpeakerStore, SessionBuffer]:
    """Build an isolated SpeakerStore + SessionBuffer in *tmp_dir*.

    Mirrors the pattern from probe_retroclaim_live.py lines 93–104 so both
    probes exercise the same construction path.
    """
    store = SpeakerStore(
        store_path=tmp_dir / "speakers.json",
        high_threshold=0.90,
        low_threshold=0.80,
    )
    buffer = SessionBuffer(
        session_dir=tmp_dir / "sessions",
        state_dir=tmp_dir / "state",
        retain_sessions=False,
        debug=False,
    )
    return store, buffer


def _make_speaker_resolver(store: SpeakerStore):
    """Build the production speaker-resolver closure from *store*.

    Mirrors inference.py lines 308–312 exactly:
      * ``is_speaker_id(tok)`` → resolve via store
      * unknown / anonymous → ``THIRD_PARTY_DESCRIPTOR``
    """

    def _resolver(tok: str) -> str:
        if not is_speaker_id(tok):
            return tok
        name = store.resolve_speaker_name(tok.lower())
        return name if name else THIRD_PARTY_DESCRIPTOR

    return _resolver


def run_cpu_identity_migration(out_dir: Path) -> dict:
    """Phase 1 — lowercase ID assertion + v5→v6 migration, CPU only."""
    results: dict = {}

    # ---------------------------------------------------------------
    # 1a–1c: Enroll two named speakers; assert lowercase IDs.
    # ---------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store, _ = _build_buffer_and_store(tmp_path)

        alex_id = store.enroll("Alex", EMB_ALEX)
        dana_id = store.enroll("Dana", EMB_DANA)

        results["enroll"] = {
            "alex_id": alex_id,
            "dana_id": dana_id,
            "alex_id_is_lowercase_speakerN": alex_id is not None
            and alex_id == f"speaker{alex_id[len('speaker') :]}"
            and alex_id == alex_id.lower()
            and is_speaker_id(alex_id),
            "dana_id_is_lowercase_speakerN": dana_id is not None
            and dana_id == f"speaker{dana_id[len('speaker') :]}"
            and dana_id == dana_id.lower()
            and is_speaker_id(dana_id),
        }

        # 1b: resolve_speaker_name works directly on the lowercase token.
        resolve_alex = store.resolve_speaker_name(alex_id) if alex_id else None
        resolve_dana = store.resolve_speaker_name(dana_id) if dana_id else None
        results["resolve_speaker_name"] = {
            "alex_resolves_to_Alex": resolve_alex == "Alex",
            "dana_resolves_to_Dana": resolve_dana == "Dana",
            "alex_raw": resolve_alex,
            "dana_raw": resolve_dana,
        }

    # ---------------------------------------------------------------
    # 1d: Migration check — v5 profile with CASED "Speaker0" key.
    # ---------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        v5_store_path = tmp_path / "speakers_v5.json"

        # Build a synthetic v5 payload using the cased key convention.
        v5_payload = {
            "version": 5,
            "next_anon_index": 1,
            "last_greeted": {"Speaker0": "2026-01-01T00:00:00+00:00"},
            "speakers": {
                "Speaker0": {
                    "name": "Alex",
                    "embeddings": [EMB_ALEX],
                    "preferred_language": "",
                    "enroll_method": "self_introduced",
                }
            },
        }
        v5_store_path.write_text(json.dumps(v5_payload, indent=2))

        # Reload through SpeakerStore — migration fires on _load().
        migrated_store = SpeakerStore(
            store_path=v5_store_path,
            high_threshold=0.90,
            low_threshold=0.80,
        )

        # Check keys are lowercased.
        profile_keys = list(migrated_store._profiles.keys())
        last_greeted_keys = list(migrated_store._last_greeted.keys())

        key_is_lowercase = "speaker0" in profile_keys and "Speaker0" not in profile_keys
        last_greeted_lowercased = (
            "speaker0" in last_greeted_keys and "Speaker0" not in last_greeted_keys
        )

        # Check embeddings survived.
        profile = migrated_store._profiles.get("speaker0", {})
        embeddings_survived = len(profile.get("embeddings", [])) == 1

        # Check that a matching embedding still hits at high confidence.
        match = migrated_store.match(EMB_ALEX)
        match_succeeds = (
            match.speaker_id == "speaker0" and match.confidence >= 0.90 and not match.tentative
        )

        # Saved file should now be version 6.
        # Use read_maybe_encrypted — the store may write an age-encrypted
        # payload when the daily identity is loaded (production key material).
        saved_data = json.loads(read_maybe_encrypted(v5_store_path).decode("utf-8"))
        saved_version = saved_data.get("version")

        results["migration_v5_to_v6"] = {
            "profile_keys_lowercased": key_is_lowercase,
            "last_greeted_keys_lowercased": last_greeted_lowercased,
            "embeddings_survived": embeddings_survived,
            "match_succeeds_post_migration": match_succeeds,
            "match_confidence": match.confidence if match.speaker_id else 0.0,
            "saved_as_version_6": saved_version == _PROFILE_VERSION,
            "profile_keys": profile_keys,
        }

    logger.info("Phase 1 (CPU identity + migration) complete: %s", json.dumps(results, indent=2))
    (out_dir / "phase1_cpu.json").write_text(json.dumps(results, indent=2))
    return results


def _collect_triples(loop: ConsolidationLoop) -> list[dict]:
    """Collect all (subject, predicate, object) triples from the merger graph.

    Mirrors probe_retroclaim_live.py lines 175–185.
    """
    out = []
    for u, v, data in loop.merger.graph.edges(data=True):
        out.append(
            {
                "subject": u,
                "predicate": data.get("relation") or data.get("predicate"),
                "object": v,
            }
        )
    return out


def run_gpu_render_resolution(out_dir: Path) -> dict:
    """Phase 2 — render-boundary display-name substitution (GPU + CPU render).

    Loads Mistral 7B, extracts a short conversation where speaker0=Alex
    mentions speaker1=Dana as subject and as object. Then verifies that
    ``entry_fact_text`` with the production ``_speaker_resolver`` closure
    replaces raw ``speaker0`` / ``speaker1`` tokens with ``Alex`` / ``Dana``.

    Also verifies the MemoryStore.probe cache-hit path returns ``fact_text``
    with display names (pre-populated store, ``source=None``).

    Anonymous third-party path verified without GPU (CPU-only MemoryStore):
    registers an anonymous speaker, builds a synthetic SPO entry with the
    anon id as subject, calls ``entry_fact_text`` with the resolver, and
    asserts the result contains THIRD_PARTY_DESCRIPTOR.
    """
    logger.info("Loading Mistral 7B (NF4)...")
    model_config = BENCHMARK_MODELS["mistral"]
    model, tokenizer = load_base_model(model_config)

    training_cfg = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=1,
        max_seq_length=512,
        num_epochs=1,
        warmup_steps=0,
        warmup_ratio=0.0,
    )
    consolidation_cfg = ConsolidationConfig(
        indexed_key_replay_enabled=False,
        promotion_threshold=3,
    )
    loop = ConsolidationLoop(
        model=model,
        tokenizer=tokenizer,
        consolidation_config=consolidation_cfg,
        training_config=training_cfg,
        episodic_adapter_config=_tier_cfg(),
        semantic_adapter_config=_tier_cfg(),
        procedural_adapter_config=_tier_cfg(),
        memory_store=MemoryStore(replay_enabled=False),
        wandb_config=None,
        output_dir=out_dir,
        save_cycle_snapshots=False,
        extraction_stt_correction=False,
        extraction_ha_validation=False,
        extraction_noise_filter="off",
        extraction_plausibility_judge="off",
        extraction_verify_anonymization=False,
    )

    results: dict = {}

    # ---------------------------------------------------------------
    # Phase 2a: GPU extraction — Alex (speaker0) mentions Dana (speaker1).
    # ---------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store, buffer = _build_buffer_and_store(tmp_path)

        alex_id = store.enroll("Alex", EMB_ALEX)
        dana_id = store.enroll("Dana", EMB_DANA)

        conv_id = "render_alex_dana"
        for role, text, _emb in TURNS_RENDER:
            buffer.append(conv_id, role, text, embedding=None)
        buffer.set_speaker(conv_id, alex_id, "Alex")

        pending = buffer.get_pending()
        session = next(p for p in pending if p["session_id"] == conv_id)
        transcript = session["transcript"]
        sid = session["speaker_id"]
        name = store.get_name(sid) if sid else None

        logger.info("Phase 2a: session_speaker_id=%r resolved_name=%r", sid, name)
        logger.info("Phase 2a transcript:\n%s", transcript)

        loop.merger.graph.clear()
        episodic_qa, procedural_rels = loop.extract_session(
            session_transcript=transcript,
            session_id=conv_id,
            speaker_id=sid or "speaker0",
            speaker_name=name,
        )
        triples = _collect_triples(loop)

        results["extraction"] = {
            "session_speaker_id": sid,
            "resolved_speaker_name": name,
            "episodic_qa_count": len(episodic_qa),
            "procedural_rel_count": len(procedural_rels),
            "triples_total": len(triples),
            "sample_triples": triples[:10],
        }
        logger.info("Phase 2a: extracted %d triples", len(triples))

        # ---------------------------------------------------------------
        # Phase 2b: entry_fact_text render — raw SPO with speaker tokens.
        # Build synthetic entries for the speaker-as-subject and
        # speaker-as-object cases; these do NOT require GPU inference.
        # ---------------------------------------------------------------
        resolver = _make_speaker_resolver(store)

        # Case A: subject = speaker1 (Dana), object = "Berlin"
        entry_a = {
            "key": "graph0",
            "subject": dana_id,
            "predicate": "lives_in",
            "object": "Berlin",
        }
        fact_a_raw = entry_fact_text(entry_a)
        fact_a_resolved = entry_fact_text(entry_a, resolve=resolver)

        # Case B: subject = speaker0 (Alex), object = speaker1 (Dana)
        entry_b = {
            "key": "graph1",
            "subject": alex_id,
            "predicate": "knows",
            "object": dana_id,
        }
        fact_b_raw = entry_fact_text(entry_b)
        fact_b_resolved = entry_fact_text(entry_b, resolve=resolver)

        results["entry_fact_text_render"] = {
            "case_A_subject_is_dana_id": entry_a["subject"] == dana_id,
            "case_A_raw": fact_a_raw,
            "case_A_resolved": fact_a_resolved,
            "case_A_pass": "Dana" in fact_a_resolved and dana_id not in fact_a_resolved,
            "case_B_subject_is_alex_id": entry_b["subject"] == alex_id,
            "case_B_object_is_dana_id": entry_b["object"] == dana_id,
            "case_B_raw": fact_b_raw,
            "case_B_resolved": fact_b_resolved,
            "case_B_pass": "Alex" in fact_b_resolved
            and "Dana" in fact_b_resolved
            and alex_id not in fact_b_resolved
            and dana_id not in fact_b_resolved,
        }
        logger.info(
            "Phase 2b entry_fact_text: A=%r B=%r",
            fact_a_resolved,
            fact_b_resolved,
        )

        # ---------------------------------------------------------------
        # Phase 2c: MemoryStore.probe cache-hit path — pre-populate store,
        # call probe with source=None and speaker_resolver set; assert
        # fact_text in the returned result has display names.
        # ---------------------------------------------------------------
        mem_store = MemoryStore(replay_enabled=False)

        # Case A — put entry_a into store, then probe (cache-hit path).
        mem_store.put("episodic", entry_a["key"], entry_a)
        # Bookkeeping: set speaker_id so the mismatch filter does not drop it.
        mem_store.set_bookkeeping(
            entry_a["key"],
            speaker_id=alex_id,
            first_seen_cycle=1,
            relation_type="factual",
            allow_empty_speaker=False,
        )
        probe_a = mem_store.probe(
            {"episodic": [entry_a["key"]]},
            speaker_resolver=resolver,
        )
        hit_a = probe_a.get(entry_a["key"])

        # Case B — put entry_b into store, then probe.
        mem_store.put("episodic", entry_b["key"], entry_b)
        mem_store.set_bookkeeping(
            entry_b["key"],
            speaker_id=alex_id,
            first_seen_cycle=1,
            relation_type="social",
            allow_empty_speaker=False,
        )
        probe_b = mem_store.probe(
            {"episodic": [entry_b["key"]]},
            speaker_resolver=resolver,
        )
        hit_b = probe_b.get(entry_b["key"])

        cache_hit_a_fact = hit_a.get("fact_text", "") if hit_a else ""
        cache_hit_b_fact = hit_b.get("fact_text", "") if hit_b else ""

        results["probe_cache_hit"] = {
            "case_A_hit_returned": hit_a is not None,
            "case_A_fact_text": cache_hit_a_fact,
            "case_A_pass": hit_a is not None
            and "Dana" in cache_hit_a_fact
            and dana_id not in cache_hit_a_fact,
            "case_B_hit_returned": hit_b is not None,
            "case_B_fact_text": cache_hit_b_fact,
            "case_B_pass": hit_b is not None
            and "Alex" in cache_hit_b_fact
            and "Dana" in cache_hit_b_fact
            and alex_id not in cache_hit_b_fact
            and dana_id not in cache_hit_b_fact,
        }
        logger.info(
            "Phase 2c probe cache-hit: A=%r B=%r",
            cache_hit_a_fact,
            cache_hit_b_fact,
        )

        # ---------------------------------------------------------------
        # Phase 2d: Anonymous third-party render — no GPU needed.
        # Register an anonymous speaker, build a synthetic SPO entry with
        # its id as subject, and assert fact_text shows THIRD_PARTY_DESCRIPTOR.
        # ---------------------------------------------------------------
        anon_id = store.register_anonymous(EMB_ANON)
        entry_anon = {
            "key": "graph2",
            "subject": anon_id,
            "predicate": "visited",
            "object": "Hamburg",
        }
        fact_anon_raw = entry_fact_text(entry_anon)
        fact_anon_resolved = entry_fact_text(entry_anon, resolve=resolver)

        # Also probe through MemoryStore (cache-hit path, no speaker filter).
        mem_store.put("episodic", entry_anon["key"], entry_anon)
        # allow_empty_speaker=True: anonymous speakers have a valid token but
        # no meaningful speaker_id restriction for this probe.
        mem_store.set_bookkeeping(
            entry_anon["key"],
            speaker_id="",
            first_seen_cycle=1,
            relation_type="factual",
            allow_empty_speaker=True,
        )
        probe_anon = mem_store.probe(
            {"episodic": [entry_anon["key"]]},
            speaker_resolver=resolver,
        )
        hit_anon = probe_anon.get(entry_anon["key"])
        cache_anon_fact = hit_anon.get("fact_text", "") if hit_anon else ""

        results["anonymous_render"] = {
            "anon_id": anon_id,
            "anon_id_is_lowercase_speakerN": anon_id == anon_id.lower() and is_speaker_id(anon_id),
            "fact_anon_raw": fact_anon_raw,
            "fact_anon_resolved": fact_anon_resolved,
            "entry_fact_text_pass": THIRD_PARTY_DESCRIPTOR in fact_anon_resolved
            and anon_id not in fact_anon_resolved,
            "probe_cache_anon_fact": cache_anon_fact,
            "probe_cache_pass": hit_anon is not None
            and THIRD_PARTY_DESCRIPTOR in cache_anon_fact
            and anon_id not in cache_anon_fact,
        }
        logger.info(
            "Phase 2d anonymous render: raw=%r resolved=%r probe=%r",
            fact_anon_raw,
            fact_anon_resolved,
            cache_anon_fact,
        )

    (out_dir / "phase2_gpu.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> int:
    """Run all phases and write structured results.json.

    Phase 1 (CPU) runs first; Phase 2 requires the GPU and is guarded by
    ``acquire_gpu()``.  Each phase writes an intermediate JSON file so a
    crash in Phase 2 does not lose Phase 1 results.
    """
    setup_logging()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / "speaker_identity_probe" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output dir: %s", out_dir)

    # Phase 1 — CPU only.
    phase1 = run_cpu_identity_migration(out_dir)

    # Phase 2 — GPU: acquire before loading the model.
    acquire_gpu()
    phase2 = run_gpu_render_resolution(out_dir)

    report = {
        "timestamp": stamp,
        "third_party_descriptor": THIRD_PARTY_DESCRIPTOR,
        "phase1_cpu": phase1,
        "phase2_gpu": phase2,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))

    # ------------------------------------------------------------------
    # Verdict — evaluate each assertion group.
    # ------------------------------------------------------------------
    enroll = phase1.get("enroll", {})
    resolve_r = phase1.get("resolve_speaker_name", {})
    mig = phase1.get("migration_v5_to_v6", {})

    p1_pass = (
        enroll.get("alex_id_is_lowercase_speakerN", False)
        and enroll.get("dana_id_is_lowercase_speakerN", False)
        and resolve_r.get("alex_resolves_to_Alex", False)
        and resolve_r.get("dana_resolves_to_Dana", False)
        and mig.get("profile_keys_lowercased", False)
        and mig.get("last_greeted_keys_lowercased", False)
        and mig.get("embeddings_survived", False)
        and mig.get("match_succeeds_post_migration", False)
        and mig.get("saved_as_version_6", False)
    )

    eft = phase2.get("entry_fact_text_render", {})
    pch = phase2.get("probe_cache_hit", {})
    anon = phase2.get("anonymous_render", {})

    p2_pass = (
        eft.get("case_A_pass", False)
        and eft.get("case_B_pass", False)
        and pch.get("case_A_pass", False)
        and pch.get("case_B_pass", False)
        and anon.get("entry_fact_text_pass", False)
        and anon.get("probe_cache_pass", False)
    )

    print("\n=== Speaker Identity Live Probe Summary ===")
    print(f"Phase 1 CPU identity + migration:  {'PASS' if p1_pass else 'FAIL'}")
    print(
        f"  Enroll lowercase IDs:           "
        f"alex_id={enroll.get('alex_id')!r}  "
        f"dana_id={enroll.get('dana_id')!r}"
    )
    print(
        f"  resolve_speaker_name:            "
        f"Alex→{resolve_r.get('alex_raw')!r}  "
        f"Dana→{resolve_r.get('dana_raw')!r}"
    )
    print(
        f"  v5→v6 migration:                 "
        f"keys_lowercased={mig.get('profile_keys_lowercased')}  "
        f"match_ok={mig.get('match_succeeds_post_migration')}"
    )
    print(f"Phase 2 GPU render resolution:     {'PASS' if p2_pass else 'FAIL'}")
    print(
        f"  entry_fact_text (named):         "
        f"A={eft.get('case_A_resolved')!r}  "
        f"B={eft.get('case_B_resolved')!r}"
    )
    print(
        f"  probe cache-hit (named):         "
        f"A={pch.get('case_A_fact_text')!r}  "
        f"B={pch.get('case_B_fact_text')!r}"
    )
    print(
        f"  anonymous render:                "
        f"resolved={anon.get('fact_anon_resolved')!r}  "
        f"probe={anon.get('probe_cache_anon_fact')!r}"
    )
    print(f"\nResults: {out_dir / 'results.json'}")
    return 0 if (p1_pass and p2_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
