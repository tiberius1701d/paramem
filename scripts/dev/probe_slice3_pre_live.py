"""Live-GPU probe for Slice 3-pre (anonymous speaker ID promotion).

Exercises two behaviors introduced by Slice 3-pre against real hardware:

  1. SpeakerStore API (CPU) — register_anonymous monotonic + idempotent,
     enroll upgrade-in-place on self-introduction. These paths are already
     covered by unit tests; included here as a sanity smoke before touching
     the GPU.
  2. Extraction pipeline (GPU) — the same fabricated transcript is
     extracted twice with different ``speaker_name`` values: the opaque
     anonymous id ("Speaker7") and a real first name ("Alice"). Both
     must produce triples whose subject is the literal expected string.
     The real-name case is a regression guard for Slice 2 behavior;
     the anonymous case is the Slice-3 prerequisite that the alias
     map has a rewritable anchor in the graph.

This is the live-GPU counterpart to tests/test_speaker.py and
tests/test_consolidation.py::TestAnonymousSpeakerNotSkipped (both CPU-only
with mocked extraction).

Outputs to outputs/slice3_pre_probe/<timestamp>/results.json.
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
# Force offline mode — weights are cached locally; the optional custom_generate
# fetch in transformers 5.3 otherwise errors when the VPN flaps mid-load.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
from experiments.utils.test_harness import BENCHMARK_MODELS, setup_logging  # noqa: E402
from paramem.models.loader import load_base_model  # noqa: E402
from paramem.server.speaker import SpeakerStore  # noqa: E402
from paramem.training.consolidation import ConsolidationLoop  # noqa: E402
from paramem.utils.config import (  # noqa: E402
    AdapterConfig,
    ConsolidationConfig,
    TrainingConfig,
)

logger = logging.getLogger("probe_slice3_pre_live")

TRANSCRIPT = (
    "User: I really enjoy hiking in the Alps on weekends.\n"
    "Assistant: That sounds wonderful.\n"
    "User: My favorite trail is near Frankfurt.\n"
    "Assistant: Germany has beautiful countryside.\n"
    "User: I also play chess every Thursday evening at the club.\n"
)

# Probe exercises two subject shapes through the same prompt machinery:
#   - anonymous opaque id ("Speaker7"), which must survive as a literal subject
#     so bridge-window triples (extract-before-disclosure) stay attributable to
#     a stable grouping ID that store.get_name() can resolve at render time;
#   - real first name ("Alice"), the production case — any regression here
#     would silently break named-speaker extraction across all of Slice 2.
SUBJECT_ANON = "Speaker7"
SUBJECT_NAMED = "Alice"
SUBJECT = SUBJECT_ANON  # legacy alias retained for the tokenizer smoke line below.


def _tier_cfg(rank: int = 8) -> AdapterConfig:
    return AdapterConfig(
        rank=rank,
        alpha=2 * rank,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def run_cpu_speaker_store_checks(out_dir: Path) -> dict:
    """SpeakerStore API roundtrip — CPU only, no model."""
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        store_path = Path(tmp) / "speakers.json"
        store = SpeakerStore(store_path=store_path, high_threshold=0.90, low_threshold=0.80)

        # Three distinct-enough embeddings (avoid tentative zone at 0.90/0.80).
        emb_a = [1.0] + [0.0] * 255
        emb_b = [0.0, 1.0] + [0.0] * 254
        emb_c = [0.0, 0.0, 1.0] + [0.0] * 253

        id_a = store.register_anonymous(emb_a)
        id_a_again = store.register_anonymous(emb_a)
        id_b = store.register_anonymous(emb_b)
        id_c = store.register_anonymous(emb_c)

        results["register_anonymous"] = {
            "first_allocation": id_a,
            "idempotent": id_a == id_a_again,
            "second_allocation": id_b,
            "third_allocation": id_c,
            "monotonic": id_a == "Speaker0" and id_b == "Speaker1" and id_c == "Speaker2",
        }

        # Upgrade-in-place: enroll with same voice + name should rename Speaker0.
        upgraded = store.enroll("Alice", emb_a)
        profile = store._profiles.get(id_a, {})
        results["enroll_upgrade_in_place"] = {
            "returned_id": upgraded,
            "preserves_anonymous_id": upgraded == id_a == "Speaker0",
            "name_now": profile.get("name"),
            "enroll_method_now": profile.get("enroll_method"),
        }

        # Enrollment against a named profile should still reject.
        rejected = store.enroll("Bob", emb_a)
        results["enroll_rejects_named_duplicate"] = {
            "returned_id_is_none": rejected is None,
        }

        # Disk schema check.
        with open(store_path) as f:
            raw = json.load(f)
        results["persisted_schema"] = {
            "version": raw.get("version"),
            "next_anon_index": raw.get("next_anon_index"),
            "has_speaker0_named_alice": raw.get("speakers", {}).get("Speaker0", {}).get("name")
            == "Alice",
        }

    logger.info("CPU SpeakerStore checks complete: %s", json.dumps(results, indent=2))
    (out_dir / "cpu_checks.json").write_text(json.dumps(results, indent=2))
    return results


def _collect_triples(loop: ConsolidationLoop) -> list[dict]:
    """Snapshot the cumulative merged graph as a list of triple dicts."""
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
    """Score a single extraction batch against an expected literal subject.

    Looks only at edges added between the before/after snapshots so
    cross-batch accumulation in ``loop.merger.graph`` cannot leak.
    """
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


def run_gpu_extraction_check(out_dir: Path) -> dict:
    """Real Mistral extraction — GPU-bound.

    Runs two back-to-back extractions with the SAME transcript but different
    ``speaker_name`` values (anonymous id, then real first name). Both must
    produce triples whose subject is the literal expected string. A regression
    on the real-name case would invalidate this approach and route us to
    Option 2 (change anonymous-id format instead of the prompt).
    """
    logger.info("Loading Mistral 7B base model (NF4)...")
    model_config = BENCHMARK_MODELS["mistral"]
    model, tokenizer = load_base_model(model_config)

    anon_tokens = tokenizer.encode(SUBJECT_ANON, add_special_tokens=False)
    named_tokens = tokenizer.encode(SUBJECT_NAMED, add_special_tokens=False)
    logger.info(
        "Tokenization: %r -> %s tokens %s; %r -> %s tokens %s",
        SUBJECT_ANON,
        len(anon_tokens),
        anon_tokens,
        SUBJECT_NAMED,
        len(named_tokens),
        named_tokens,
    )

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

    # Four cases layered from isolation to end-to-end:
    #   1-2) Raw strings passed straight to extract_session — isolates the
    #        extraction-prompt contract (what Option 1 was fixing).
    #   3-4) A fresh SpeakerStore drives the (speaker_id, speaker_name)
    #        resolution via enroll+match / register_anonymous, mirroring the
    #        production path in paramem/server/app.py::_resolve_speaker →
    #        paramem/server/consolidation.py (speaker_store.get_name →
    #        loop.extract_session). Catches regressions in SpeakerStore ↔
    #        consolidation handoff that would silently bypass the prompt fix.
    def _setup_raw(subject: str):
        def _fn(_store: SpeakerStore) -> tuple[str, str]:
            return subject, subject

        return _fn

    def _setup_named_via_store(display_name: str):
        def _fn(store: SpeakerStore) -> tuple[str, str]:
            # Enroll Alice with a seed embedding, then "re-observe" a nearly
            # identical embedding to exercise match() the same way a live
            # ChatRequest would. Returning match.name (not the raw display
            # name) proves the store round-trip.
            seed = [1.0, 0.1] + [0.0] * 254
            observed = [0.98, 0.12] + [0.0] * 254  # cos-sim ~= 0.998
            sid = store.enroll(display_name, seed)
            if sid is None:
                raise RuntimeError(f"enroll({display_name!r}) returned None")
            match = store.match(observed)
            if match.speaker_id != sid or match.tentative or match.name != display_name:
                raise RuntimeError(
                    f"match() did not resolve to enrolled profile: "
                    f"sid={match.speaker_id!r} name={match.name!r} tentative={match.tentative}"
                )
            return match.speaker_id, match.name

        return _fn

    def _setup_anon_via_store():
        def _fn(store: SpeakerStore) -> tuple[str, str]:
            # Fresh embedding far from the enrolled Alice seed → no match, so
            # register_anonymous allocates a new Speaker{N}. The production
            # path uses the allocated id as both speaker_id and display-name
            # surrogate until disclosure (app.py anonymous branch).
            emb = [0.0, 0.0, 1.0] + [0.0] * 253
            anon_id = store.register_anonymous(emb)
            return anon_id, anon_id

        return _fn

    # One shared store across the two e2e cases so Alice enrollment and the
    # subsequent anonymous allocation interact like they would in production.
    e2e_store_dir = tempfile.mkdtemp(prefix="slice3_pre_probe_store_")
    e2e_store = SpeakerStore(store_path=Path(e2e_store_dir) / "speakers.json")

    case_specs = [
        ("raw_anon", "probe_slice3_pre_raw_anon", SUBJECT_ANON, _setup_raw(SUBJECT_ANON)),
        ("raw_named", "probe_slice3_pre_raw_named", SUBJECT_NAMED, _setup_raw(SUBJECT_NAMED)),
        (
            "e2e_named",
            "probe_slice3_pre_e2e_named",
            SUBJECT_NAMED,
            _setup_named_via_store(SUBJECT_NAMED),
        ),
        ("e2e_anon", "probe_slice3_pre_e2e_anon", None, _setup_anon_via_store()),
    ]

    cases = []
    for idx, (label, session_id, expected_override, setup_fn) in enumerate(case_specs):
        # Reset the cumulative graph between cases. Without this, identical
        # transcripts across cases produce duplicate triples that the merger
        # drops, masking each case's own extraction output behind the dedup.
        loop.merger.graph.clear()
        before: list[dict] = []

        speaker_id, speaker_name = setup_fn(e2e_store)
        expected_subject = expected_override if expected_override is not None else speaker_id
        logger.info(
            "Case %d [%s]: session_id=%r speaker_id=%r speaker_name=%r expected_subject=%r",
            idx + 1,
            label,
            session_id,
            speaker_id,
            speaker_name,
            expected_subject,
        )
        episodic_qa, procedural_rels = loop.extract_session(
            session_transcript=TRANSCRIPT,
            session_id=session_id,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
        )
        after = _collect_triples(loop)
        scored = _score_subject(before, after, expected_subject)
        scored["case_label"] = label
        scored["session_id"] = session_id
        scored["speaker_id_arg"] = speaker_id
        scored["speaker_name_arg"] = speaker_name
        scored["episodic_qa_count"] = len(episodic_qa)
        scored["procedural_rel_count"] = len(procedural_rels)
        cases.append(scored)
        logger.info(
            "Case %d [%s] result: %d/%d triples with subject=%r (near_misses=%d)",
            idx + 1,
            label,
            scored["subject_hits"],
            scored["new_triples_total"],
            expected_subject,
            scored["near_misses"],
        )

    results = {
        "tokenizer": {
            "anon": {
                "subject": SUBJECT_ANON,
                "token_ids": anon_tokens,
                "token_count": len(anon_tokens),
            },
            "named": {
                "subject": SUBJECT_NAMED,
                "token_ids": named_tokens,
                "token_count": len(named_tokens),
            },
        },
        "cases": cases,
    }

    logger.info("GPU extraction check complete")
    (out_dir / "gpu_extraction.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> int:
    setup_logging()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / "slice3_pre_probe" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu_results = run_cpu_speaker_store_checks(out_dir)
    gpu_results = run_gpu_extraction_check(out_dir)

    report = {
        "timestamp": stamp,
        "subjects": {"anon": SUBJECT_ANON, "named": SUBJECT_NAMED},
        "cpu_checks": cpu_results,
        "gpu_extraction": gpu_results,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))

    cpu_pass = (
        cpu_results["register_anonymous"]["monotonic"]
        and cpu_results["register_anonymous"]["idempotent"]
        and cpu_results["enroll_upgrade_in_place"]["preserves_anonymous_id"]
        and cpu_results["enroll_upgrade_in_place"]["name_now"] == "Alice"
        and cpu_results["enroll_upgrade_in_place"]["enroll_method_now"] == "self_introduced"
        and cpu_results["enroll_rejects_named_duplicate"]["returned_id_is_none"]
        and cpu_results["persisted_schema"]["version"] == 5
        and cpu_results["persisted_schema"]["next_anon_index"] == 3
        and cpu_results["persisted_schema"]["has_speaker0_named_alice"]
    )

    cases_by_label = {c["case_label"]: c for c in gpu_results["cases"]}
    case_passes: dict[str, bool] = {
        label: cases_by_label[label]["subject_hits"] >= 1 for label in cases_by_label
    }

    print("\n=== Slice 3-pre Live Probe Summary ===")
    print(f"CPU SpeakerStore checks:       {'PASS' if cpu_pass else 'FAIL'}")
    for label, c in cases_by_label.items():
        verdict = "PASS" if case_passes[label] else "FAIL"
        hits = c["subject_hits"]
        total = c["new_triples_total"]
        exp = c["expected_subject"]
        print(f"GPU [{label}] expected={exp!r:12} {verdict}")
        print(f"  speaker_id/name args:        {c['speaker_id_arg']!r} / {c['speaker_name_arg']!r}")
        print(f"  subject hits / new triples:  {hits}/{total}")
        print(f"  near misses:                 {c['near_misses']}")
    print(f"Results: {out_dir / 'results.json'}")

    return 0 if (cpu_pass and all(case_passes.values())) else 1


if __name__ == "__main__":
    with acquire_gpu():
        sys.exit(main())
