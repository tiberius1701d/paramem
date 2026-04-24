"""Live-GPU probe — retroactive speaker-claim mechanism.

Verifies the already-shipped "unknown turn → named turn" pipeline end to end:

  Phase 1 (CPU) — the claim mechanism itself.
    a) register_anonymous(embedding) allocates ``Speaker0``.
    b) Orphan turns land in a SessionBuffer carrying the same embedding,
       with no speaker_id on the session.
    c) enroll("Alice", same_embedding) upgrades ``Speaker0``'s profile
       name to "Alice" in place.
    d) claim_sessions_for_speaker rewrites every orphan turn: ``speaker_id``
       becomes ``Speaker0`` and ``speaker`` becomes ``"Alice"``.
    e) get_pending() formats the claimed session with ``Alice:`` prefixes.

  Phase 2 (GPU) — extract from the claimed transcript.
    Real Mistral extraction on the reconstructed transcript with
    speaker_name="Alice". Triples must have subject="Alice" — proving that
    the pre-extraction disclosure path produces named-subject triples
    without any graph rewrite.

  Phase 3 (GPU + render-time) — the bridge-window case.
    A fresh anonymous voice is extracted while still unknown
    (speaker_name="Speaker1"). Triples land with subject="Speaker1" in the
    graph. Disclosure happens *after* extraction via enroll("Bob", ...).
    The graph is NOT rewritten. Render-time resolution is verified by
    asserting store.get_name("Speaker1") == "Bob": the mechanism any
    future render-time substitution layer would call.

Outputs outputs/retroclaim_probe/<timestamp>/results.json.
No existing training data is touched; no adapter is loaded.
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
from paramem.models.loader import load_base_model  # noqa: E402
from paramem.server.session_buffer import SessionBuffer  # noqa: E402
from paramem.server.speaker import SpeakerStore  # noqa: E402
from paramem.training.consolidation import ConsolidationLoop  # noqa: E402
from paramem.utils.config import (  # noqa: E402
    AdapterConfig,
    ConsolidationConfig,
    TrainingConfig,
)

logger = logging.getLogger("probe_retroclaim_live")

# Distinct voice embeddings (256-dim, near-orthogonal at strict thresholds).
EMB_ALICE = [1.0] + [0.0] * 255
EMB_BOB = [0.0, 1.0] + [0.0] * 254

TURNS_CLAIMED = [
    ("user", "I really enjoy hiking in the Alps on weekends.", EMB_ALICE),
    ("assistant", "That sounds wonderful.", None),
    ("user", "My favorite trail is near Frankfurt.", EMB_ALICE),
    ("assistant", "Germany has beautiful countryside.", None),
    ("user", "I also play chess every Thursday evening at the club.", EMB_ALICE),
]

TURNS_BRIDGE = [
    ("user", "I work as a software engineer at a Berlin startup.", EMB_BOB),
    ("assistant", "Interesting. What stack do you work with?", None),
    ("user", "Mostly Python and Rust. We build distributed systems.", EMB_BOB),
]


def _tier_cfg(rank: int = 8) -> AdapterConfig:
    return AdapterConfig(
        rank=rank,
        alpha=2 * rank,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def _build_buffer_and_store(tmp_dir: Path) -> tuple[SpeakerStore, SessionBuffer]:
    store = SpeakerStore(
        store_path=tmp_dir / "speakers.json",
        high_threshold=0.90,
        low_threshold=0.80,
    )
    buffer = SessionBuffer(
        session_dir=tmp_dir / "sessions",
        retain_sessions=False,
        debug=False,
    )
    return store, buffer


def run_cpu_claim_mechanism(out_dir: Path) -> dict:
    """Phase 1 — retroactive claim, CPU only."""
    results: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store, buffer = _build_buffer_and_store(tmp_path)

        # 1a. Register anonymous voice.
        anon_id = store.register_anonymous(EMB_ALICE)
        results["register_anonymous"] = {
            "allocated_id": anon_id,
            "name_is_id": store.get_name(anon_id) == anon_id,
        }

        # 1b. Append orphan turns to a session. No set_speaker() call →
        # append() reads speaker_id=None from session state.
        conv_id = "retroclaim_alice"
        for role, text, emb in TURNS_CLAIMED:
            buffer.append(conv_id, role, text, embedding=emb)

        pre_turns = buffer.get_session_turns(conv_id)
        results["pre_claim_turns"] = {
            "count": len(pre_turns),
            "user_turns_have_embedding": all(
                t.get("embedding") is not None for t in pre_turns if t["role"] == "user"
            ),
            "all_orphan": all(t.get("speaker_id") is None for t in pre_turns),
            "all_speaker_none": all(t.get("speaker") is None for t in pre_turns),
        }

        # 1c. Disclosure — enroll upgrades the anonymous profile in place.
        upgraded_id = store.enroll("Alice", EMB_ALICE)
        profile = store._profiles.get(anon_id, {})
        results["enroll_upgrade"] = {
            "returned_id": upgraded_id,
            "preserves_anonymous_id": upgraded_id == anon_id,
            "name_now": profile.get("name"),
            "enroll_method_now": profile.get("enroll_method"),
        }

        # 1d. Retro-claim — rewrites orphan turns.
        claimed_count = buffer.claim_sessions_for_speaker(anon_id, "Alice", store)
        post_turns = buffer.get_session_turns(conv_id)
        results["claim_sessions_for_speaker"] = {
            "claimed_count": claimed_count,
            "turns_attributed_to_anon_id": all(t.get("speaker_id") == anon_id for t in post_turns),
            "turns_named_alice": all(t.get("speaker") == "Alice" for t in post_turns),
        }

        # 1e. Formatted transcript — the input that extract_session sees.
        pending = buffer.get_pending()
        our_session = next((p for p in pending if p["session_id"] == conv_id), None)
        transcript = our_session["transcript"] if our_session else ""
        results["formatted_transcript"] = {
            "session_speaker_id": our_session["speaker_id"] if our_session else None,
            "starts_user_lines_with_alice": all(
                line.startswith("Alice:")
                for line in transcript.splitlines()
                if line.startswith(("Alice:", "User:"))  # catch leakage
            ),
            "transcript_preview": transcript,
        }

    logger.info("Phase 1 (CPU retro-claim) complete: %s", json.dumps(results, indent=2))
    (out_dir / "phase1_cpu.json").write_text(json.dumps(results, indent=2))
    return results


def _collect_triples(loop: ConsolidationLoop) -> list[dict]:
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


def _score_subject(triples_before: list[dict], triples_after: list[dict], subject: str) -> dict:
    before_keys = {(t["subject"], t["predicate"], t["object"]) for t in triples_before}
    new_triples = [
        t for t in triples_after if (t["subject"], t["predicate"], t["object"]) not in before_keys
    ]
    subject_hits = [t for t in new_triples if t["subject"] == subject]
    near_misses = [
        t
        for t in new_triples
        if t["subject"] != subject and subject.lower() in (t["subject"] or "").lower()
    ]
    unrelated = [
        t
        for t in new_triples
        if subject.lower() not in (t["subject"] or "").lower()
        and subject.lower() not in (t["object"] or "").lower()
    ]
    return {
        "expected_subject": subject,
        "new_triples_total": len(new_triples),
        "subject_hits": len(subject_hits),
        "near_misses": len(near_misses),
        "unrelated": len(unrelated),
        "sample_hits": subject_hits[:5],
        "sample_near_misses": near_misses[:5],
        "sample_unrelated": unrelated[:5],
    }


def run_gpu_phases(out_dir: Path) -> dict:
    """Phase 2 + Phase 3 — real Mistral extraction."""
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
        wandb_config=None,
        output_dir=out_dir,
        save_cycle_snapshots=False,
        persist_graph=False,
        extraction_stt_correction=False,
        extraction_ha_validation=False,
        extraction_noise_filter="off",
        extraction_plausibility_judge="off",
        extraction_verify_anonymization=False,
    )

    results: dict = {"phase2": {}, "phase3": {}}

    # ---------------------------------------------------------------
    # Phase 2 — extract from claimed transcript.
    # ---------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store_p2, buffer_p2 = _build_buffer_and_store(tmp_path)
        anon_id = store_p2.register_anonymous(EMB_ALICE)
        conv_id = "retroclaim_alice_p2"
        for role, text, emb in TURNS_CLAIMED:
            buffer_p2.append(conv_id, role, text, embedding=emb)
        store_p2.enroll("Alice", EMB_ALICE)
        buffer_p2.claim_sessions_for_speaker(anon_id, "Alice", store_p2)
        pending = buffer_p2.get_pending()
        session = next(p for p in pending if p["session_id"] == conv_id)
        transcript = session["transcript"]
        sid = session["speaker_id"]
        name = store_p2.get_name(sid)

        logger.info("Phase 2: session_speaker_id=%r resolved_name=%r", sid, name)
        logger.info("Phase 2 transcript:\n%s", transcript)

        loop.merger.graph.clear()
        before: list[dict] = []
        episodic_qa, procedural_rels = loop.extract_session(
            session_transcript=transcript,
            session_id=conv_id,
            speaker_id=sid,
            speaker_name=name,
        )
        after = _collect_triples(loop)
        scored = _score_subject(before, after, "Alice")
        scored["session_speaker_id"] = sid
        scored["resolved_speaker_name"] = name
        scored["episodic_qa_count"] = len(episodic_qa)
        scored["procedural_rel_count"] = len(procedural_rels)
        results["phase2"] = scored
        logger.info(
            "Phase 2 result: %d/%d triples with subject='Alice' (near=%d)",
            scored["subject_hits"],
            scored["new_triples_total"],
            scored["near_misses"],
        )

    # ---------------------------------------------------------------
    # Phase 3 — bridge window: extract BEFORE disclosure, then disclose.
    # ---------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store_p3, buffer_p3 = _build_buffer_and_store(tmp_path)
        anon_id_bob = store_p3.register_anonymous(EMB_BOB)
        conv_id = "retroclaim_bob_p3"
        # Stamp the turns with the anonymous ID via set_speaker — this is
        # the state when extraction runs mid-session, before disclosure.
        buffer_p3.set_speaker(conv_id, anon_id_bob, anon_id_bob)
        for role, text, emb in TURNS_BRIDGE:
            buffer_p3.append(conv_id, role, text, embedding=emb)
        pending = buffer_p3.get_pending()
        session = next(p for p in pending if p["session_id"] == conv_id)
        transcript_pre = session["transcript"]
        sid_pre = session["speaker_id"]
        name_pre = store_p3.get_name(sid_pre)

        logger.info("Phase 3 pre-disclosure: sid=%r name=%r", sid_pre, name_pre)
        logger.info("Phase 3 transcript (pre-disclosure):\n%s", transcript_pre)

        loop.merger.graph.clear()
        before = []
        episodic_qa, procedural_rels = loop.extract_session(
            session_transcript=transcript_pre,
            session_id=conv_id,
            speaker_id=sid_pre,
            speaker_name=name_pre,
        )
        after_pre = _collect_triples(loop)
        scored_pre = _score_subject(before, after_pre, anon_id_bob)
        scored_pre["episodic_qa_count"] = len(episodic_qa)
        scored_pre["procedural_rel_count"] = len(procedural_rels)
        results["phase3"]["pre_disclosure_extraction"] = scored_pre
        logger.info(
            "Phase 3 pre-disclosure result: %d/%d triples with subject=%r",
            scored_pre["subject_hits"],
            scored_pre["new_triples_total"],
            anon_id_bob,
        )

        # Disclosure happens AFTER extraction. The graph is not rewritten.
        upgraded_id = store_p3.enroll("Bob", EMB_BOB)
        resolved_name_after = store_p3.get_name(anon_id_bob)
        graph_still_anon = all(t["subject"] == anon_id_bob for t in scored_pre["sample_hits"])
        results["phase3"]["post_disclosure"] = {
            "upgraded_id": upgraded_id,
            "preserves_anonymous_id": upgraded_id == anon_id_bob,
            "store_get_name_returns_bob": resolved_name_after == "Bob",
            "graph_subjects_still_anonymous": graph_still_anon,
            "render_time_resolution_ok": resolved_name_after == "Bob" and graph_still_anon,
        }
        logger.info(
            "Phase 3 post-disclosure: get_name(%r)=%r; graph subjects unchanged=%s",
            anon_id_bob,
            resolved_name_after,
            graph_still_anon,
        )

    (out_dir / "phase2_3_gpu.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> int:
    setup_logging()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / "retroclaim_probe" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output dir: %s", out_dir)

    phase1 = run_cpu_claim_mechanism(out_dir)

    acquire_gpu()
    phase23 = run_gpu_phases(out_dir)

    report = {
        "timestamp": stamp,
        "phase1_cpu": phase1,
        "phase2_3_gpu": phase23,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))

    # --- verdict ---
    p1_pass = (
        phase1["register_anonymous"]["allocated_id"] == "Speaker0"
        and phase1["register_anonymous"]["name_is_id"]
        and phase1["pre_claim_turns"]["user_turns_have_embedding"]
        and phase1["pre_claim_turns"]["all_orphan"]
        and phase1["enroll_upgrade"]["preserves_anonymous_id"]
        and phase1["enroll_upgrade"]["name_now"] == "Alice"
        and phase1["claim_sessions_for_speaker"]["claimed_count"] == 1
        and phase1["claim_sessions_for_speaker"]["turns_attributed_to_anon_id"]
        and phase1["claim_sessions_for_speaker"]["turns_named_alice"]
        and phase1["formatted_transcript"]["session_speaker_id"] == "Speaker0"
        and phase1["formatted_transcript"]["starts_user_lines_with_alice"]
    )
    p2_pass = phase23["phase2"]["subject_hits"] >= 1
    p3_pass = (
        phase23["phase3"]["pre_disclosure_extraction"]["subject_hits"] >= 1
        and phase23["phase3"]["post_disclosure"]["render_time_resolution_ok"]
    )

    print("\n=== Retroclaim Live Probe Summary ===")
    print(f"Phase 1 CPU retro-claim:            {'PASS' if p1_pass else 'FAIL'}")
    print(
        f"Phase 2 GPU extract-after-claim:    "
        f"{'PASS' if p2_pass else 'FAIL'} "
        f"({phase23['phase2']['subject_hits']}/"
        f"{phase23['phase2']['new_triples_total']} Alice-subject triples)"
    )
    p3 = phase23["phase3"]
    anon_subject = (p3["pre_disclosure_extraction"]["sample_hits"] or [{}])[0].get(
        "subject", "Anon"
    )
    print(
        f"Phase 3 bridge-window + render-time: "
        f"{'PASS' if p3_pass else 'FAIL'} "
        f"(pre: {p3['pre_disclosure_extraction']['subject_hits']}/"
        f"{p3['pre_disclosure_extraction']['new_triples_total']} {anon_subject}-subject; "
        f"get_name={p3['post_disclosure']['store_get_name_returns_bob']})"
    )
    print(f"\nResults: {out_dir / 'results.json'}")
    return 0 if (p1_pass and p2_pass and p3_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
