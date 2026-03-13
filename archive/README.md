# Archive — Failed and Superseded Approaches

These files represent explored approaches that were abandoned during Phase 4
development. They are preserved as part of the research story (see paper
Section 5.5: Failed Approaches) but are not part of the core pipeline.

## training/key_replay.py (F4.7)
XML `<memory key="...">` triple format for key-addressable replay.
**Failed:** Format collision between QA pairs and structured triple blocks
caused cross-contamination — 0.0 F1 reconstruction across all keys.

## training/entity_profile.py (F4.7b)
Entity-keyed natural language profiles. Entities as memory units with NL
profile answers.
**Regressed:** Training signal dilution (broad profiles vs fact-specific QA)
caused episodic recall to regress from 59.3% (Phase 3b) to 36.1%.

## training/entity_registry.py (F4.7b)
Entity lifecycle management for entity-keyed replay. Tied to the
entity-replay pipeline above.

## experiments/phase4_key_replay.py
Full experiment for F4.7 XML triple key-replay. 20 sessions, 20 epochs.

## experiments/phase4_entity_replay.py
Full experiment for F4.7b entity-keyed NL replay. 20 sessions, 20 epochs.

## experiments/phase4_rank_comparison.py
LoRA rank sweep (8/4/2) for entity-replay. Rank is not the lever for this
pipeline — confusion rate increased at lower ranks.

## experiments/phase4_keyed_json_smoke.py
Early JSON keyed recall exploration. Superseded by the cleaner
indexed_memory.py design (F4.9).
