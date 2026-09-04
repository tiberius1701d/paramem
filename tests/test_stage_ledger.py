"""Tests for the stage ledger's write-location surfaces.

Covers the ``"tier_written"`` entry's ``slot`` field end to end: the builder/
accessor pair in ``paramem.training.stage_ledger`` (``tier_written_stage`` /
``written_slot_path``), the version gate that refuses an incompatible ledger
(``read_ledger`` raising ``StageLedgerVersionUnsupported``), and the resume
path that publishes a previously-written member from its recorded slot
without retraining it (``ConsolidationLoop.run_build_and_publish``, driven
through the real write/publish machinery via ``tests._fold_fixtures``).
"""

import json

import pytest

from paramem.training import stage_ledger as sl
from tests._fold_fixtures import _make_loop, _recalled_entries_from_store, _rel, _wire_fakes


class TestTierWrittenStageSlot:
    """The ``tier_written`` entry's ``slot`` field is a genuine new fact --
    not derivable from the artifact list -- so the builder must record it
    verbatim and the one accessor must read it back unchanged."""

    def test_write_entry_records_the_slot_it_wrote(self, tmp_path):
        slot = tmp_path / "episodic" / "20260101-000000"
        slot.mkdir(parents=True)

        entry = sl.tier_written_stage(
            tier="episodic",
            completed_at="2026-01-01T00:00:00Z",
            slot=slot,
            artifacts=[],
        )

        assert entry["slot"] == str(slot)
        assert sl.written_slot_path(entry) == slot

    def test_write_entry_of_a_payload_less_member_records_no_slot(self):
        """A rows-only member (or a tier rebuilt to zero keys) writes nothing
        -- its entry must record ``slot=None``, never a fabricated path, and
        the accessor must read that back as ``None`` rather than raising."""
        entry = sl.tier_written_stage(
            tier="episodic",
            completed_at="2026-01-01T00:00:00Z",
            slot=None,
            artifacts=[],
        )

        assert entry["slot"] is None
        assert sl.written_slot_path(entry) is None


class TestWrittenSlotPathOfAbsentEntry:
    def test_written_slot_path_of_none_entry_is_none(self):
        """No ``tier_written`` entry at all (a tier never written) reads
        identically to one that written no payload -- both are ``None``."""
        assert sl.written_slot_path(None) is None


class TestUnsupportedLedgerVersion:
    """Version 2 is the only version this build knows how to interpret
    (it added the ``tier_written`` entry's ``slot`` field).  A pending v1
    record must raise loudly rather than be silently read as if it had a
    ``slot`` field -- that would route a written member into the rows-only
    arm, per ``stage_ledger.StageLedgerVersionUnsupported``'s own
    docstring."""

    def test_ledger_of_an_unsupported_version_refuses_to_be_read(self, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        payload = {
            "version": 1,
            "event": "interim",
            "venue": "weights",
            "stamp": "20260101T0000",
            "tiers": {"episodic": {"adapter": "episodic", "pre_sha": ""}},
            "stages": [
                sl.extraction_stage(
                    completed_at="2026-01-01T00:00:00+00:00",
                    sessions=[],
                    episodic_rels=0,
                    procedural_rels=0,
                    artifacts=[],
                )
            ],
            "absorbed_interim_tiers": [],
        }
        sl.ledger_path(state_dir).write_text(json.dumps(payload))

        with pytest.raises(sl.StageLedgerVersionUnsupported):
            sl.read_ledger(state_dir)

    def test_a_supported_version_reads_back_without_raising(self, tmp_path):
        """Sanity counterpart: the current version (2) is not itself refused
        -- the gate is on the version number, not on every read."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        payload = {
            "version": 2,
            "event": "interim",
            "venue": "weights",
            "stamp": "20260101T0000",
            "tiers": {"episodic": {"adapter": "episodic", "pre_sha": ""}},
            "stages": [
                sl.extraction_stage(
                    completed_at="2026-01-01T00:00:00+00:00",
                    sessions=[],
                    episodic_rels=0,
                    procedural_rels=0,
                    artifacts=[],
                )
            ],
            "absorbed_interim_tiers": [],
        }
        sl.ledger_path(state_dir).write_text(json.dumps(payload))

        ledger = sl.read_ledger(state_dir)
        assert ledger is not None
        assert ledger.version == 2


class TestResumePublishesFromRecordedWrittenSlot:
    """The crash-window scenario Q3/Q7 exist for: a member writes (its
    ``tier_written`` entry lands, naming the slot) but the bundle's go-live
    never runs -- crash, restart, or a second dispatch reading the same
    pending ledger. Resume must publish that member from the RECORDED slot
    (never re-derive it by scanning the tier tree, never retrain it)."""

    def test_resume_publishes_a_written_member_from_the_recorded_slot(self, tmp_path, monkeypatch):
        from paramem.memory.increment import build_tier_increment

        # resident_tiers=["episodic"] matches _FakeAdapterConfig's own
        # rank/alpha defaults so ensure_adapter_matching's warm no-op path
        # fires (see _fold_fixtures._make_loop's own comment) instead of a
        # cold get_peft_model() recreate, which needs a real torch.nn.Module
        # the fake driver model does not provide.
        loop = _make_loop(tmp_path, resident_tiers=["episodic"])
        _wire_fakes(loop, monkeypatch)

        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="stamp1",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
            session_ids=["s1"],
        )
        assert staged is not None
        state_dir = staged.state_dir

        # Write the tier directly -- this is exactly what
        # run_build_and_publish's own per-tier loop does for a not-yet-done
        # tier -- WITHOUT calling run_build_and_publish itself, so the
        # bundle never goes live: the ledger is left in the crash-window
        # shape (a verifying tier_written entry, no tier_live entry).
        shadow_dir = sl.extraction_dir(state_dir, staged.event) / "shadow" / "episodic"
        increment = build_tier_increment(
            tier="episodic",
            adapter_name=staged.ledger.tiers["episodic"]["adapter"],
            pre_sha=staged.ledger.tiers["episodic"]["pre_sha"],
            shadow_dir=shadow_dir,
        )
        written_slot, _ = loop._write_built_tier(
            increment=increment, ledger=staged.ledger, state_dir=state_dir, mode="train"
        )
        assert written_slot is not None
        assert loop._train_tier_adapter.call_count == 1

        pre_resume_ledger = sl.read_ledger(state_dir)
        written_entry = loop._latest_stage(pre_resume_ledger, "episodic", "tier_written")
        assert written_entry is not None
        assert sl.written_slot_path(written_entry) == written_slot
        assert loop._latest_stage(pre_resume_ledger, "episodic", "tier_live") is None

        # Resume: a fresh call to run_build_and_publish, exactly as a
        # restarted process (or a second dispatch onto the same pending
        # ledger) would make it. It reads the ledger fresh from disk --
        # never trusts `staged` across the simulated crash.
        summary = loop.run_build_and_publish(staged, router=None)

        assert summary["all_live"] is True
        assert "episodic" in summary["published_tiers"]
        # The resume must not retrain the already-written member.
        assert loop._train_tier_adapter.call_count == 1

        final_ledger = sl.read_ledger(state_dir)
        live_entry = loop._latest_stage(final_ledger, "episodic", "tier_live")
        assert live_entry is not None
        artifact_paths = {a["path"] for a in live_entry["artifacts"]}
        # The published member's tier_live artifact set names the SAME slot
        # this test recorded at write time -- proof the recorded slot is what
        # flowed into publish, not a freshly re-derived one.
        assert str(written_slot / "meta.json") in artifact_paths

    def test_tampered_written_slot_fails_verify_and_resume_rebuilds_rather_than_publishes_it(
        self, tmp_path, monkeypatch
    ):
        """Mutating a byte inside a written slot's payload after it writes
        makes the recorded ``tier_written`` entry's artifact hash stop
        matching the on-disk bytes -- ``stage_ledger.verify`` returns
        ``False`` for it. ``_classify_ledger_tier``'s re-hash check
        (``consolidation.py``'s ``_sl.verify(written_entry)`` guard before
        returning ``("written", ...)``) then falls through to the ``"build"``
        classification instead, so resume retrains and rewrites a FRESH slot
        rather than publishing the tampered one."""
        from paramem.memory.increment import build_tier_increment

        loop = _make_loop(tmp_path, resident_tiers=["episodic"])
        _wire_fakes(loop, monkeypatch)

        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="weights",
            stamp="stamp1",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
            session_ids=["s1"],
        )
        assert staged is not None
        state_dir = staged.state_dir

        shadow_dir = sl.extraction_dir(state_dir, staged.event) / "shadow" / "episodic"
        increment = build_tier_increment(
            tier="episodic",
            adapter_name=staged.ledger.tiers["episodic"]["adapter"],
            pre_sha=staged.ledger.tiers["episodic"]["pre_sha"],
            shadow_dir=shadow_dir,
        )
        written_slot, _ = loop._write_built_tier(
            increment=increment, ledger=staged.ledger, state_dir=state_dir, mode="train"
        )
        assert written_slot is not None
        assert loop._train_tier_adapter.call_count == 1

        # Verify the written entry as-written -- sanity baseline before tampering.
        pre_tamper_ledger = sl.read_ledger(state_dir)
        written_entry = loop._latest_stage(pre_tamper_ledger, "episodic", "tier_written")
        assert sl.verify(written_entry) is True

        # Tamper: flip a byte inside the written slot's own weight payload,
        # after the tier_written entry recorded its hash.
        weight_path = written_slot / "adapter_model.safetensors"
        original = weight_path.read_bytes()
        tampered = bytearray(original)
        tampered[0] ^= 0xFF
        weight_path.write_bytes(bytes(tampered))

        tampered_ledger = sl.read_ledger(state_dir)
        tampered_entry = loop._latest_stage(tampered_ledger, "episodic", "tier_written")
        assert sl.verify(tampered_entry) is False

        # Resume: the tampered slot must not be published as-is -- it must
        # be rebuilt (retrained, rewritten into a fresh slot).
        summary = loop.run_build_and_publish(staged, router=None)

        assert summary["all_live"] is True
        assert "episodic" in summary["published_tiers"]
        # A second train call proves the tampered member was rebuilt, not
        # skip-published from its (now-invalid) recorded slot.
        assert loop._train_tier_adapter.call_count == 2

        final_ledger = sl.read_ledger(state_dir)
        live_entry = loop._latest_stage(final_ledger, "episodic", "tier_live")
        assert live_entry is not None
        artifact_paths = {a["path"] for a in live_entry["artifacts"]}
        # The published tier_live entry does NOT name the tampered slot's
        # meta.json -- a fresh slot was written and published instead.
        assert str(written_slot / "meta.json") not in artifact_paths

    def test_resume_publishes_a_written_member_from_the_recorded_slot_simulate_venue(
        self, tmp_path, monkeypatch
    ):
        """Simulate-venue counterpart of the train-venue crash-window test
        above: ``venue="disk"`` drives ``_write_built_tier``'s ``mode="simulate"``
        branch (``run_build_and_publish`` derives ``mode`` from
        ``ledger.venue`` -- see ``ConsolidationLoop.run_build_and_publish``).
        A simulate write never trains -- it projects the graph payload
        straight from ``increment.keyed`` -- so ``_train_tier_adapter`` is
        never called, in the crash window or on resume. Resume still
        publishes the RECORDED slot from the ``tier_written`` entry rather
        than re-deriving or re-writing it."""
        from paramem.memory.increment import build_tier_increment

        loop = _make_loop(tmp_path, resident_tiers=["episodic"])
        _wire_fakes(loop, monkeypatch)

        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="interim",
            venue="disk",
            stamp="stamp1",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("alex", "lives_in", "berlin")],
            session_ids=["s1"],
        )
        assert staged is not None
        assert staged.ledger.venue == "disk"
        state_dir = staged.state_dir

        # Write the tier directly in simulate mode -- mirrors the crash
        # window the train-venue test above sets up: a verifying
        # tier_written entry lands, but the bundle's go-live never runs.
        shadow_dir = sl.extraction_dir(state_dir, staged.event) / "shadow" / "episodic"
        increment = build_tier_increment(
            tier="episodic",
            adapter_name=staged.ledger.tiers["episodic"]["adapter"],
            pre_sha=staged.ledger.tiers["episodic"]["pre_sha"],
            shadow_dir=shadow_dir,
        )
        written_slot, _ = loop._write_built_tier(
            increment=increment, ledger=staged.ledger, state_dir=state_dir, mode="simulate"
        )
        assert written_slot is not None
        assert (written_slot / "graph.json").exists()
        assert loop._train_tier_adapter.call_count == 0

        pre_resume_ledger = sl.read_ledger(state_dir)
        written_entry = loop._latest_stage(pre_resume_ledger, "episodic", "tier_written")
        assert written_entry is not None
        assert sl.written_slot_path(written_entry) == written_slot
        assert loop._latest_stage(pre_resume_ledger, "episodic", "tier_live") is None

        # Resume: a fresh call to run_build_and_publish, reading the ledger
        # fresh from disk -- never trusts `staged` across the simulated crash.
        summary = loop.run_build_and_publish(staged, router=None)

        assert summary["all_live"] is True
        assert "episodic" in summary["published_tiers"]
        # Simulate venue never trains -- neither in the crash window nor on
        # resume.
        assert loop._train_tier_adapter.call_count == 0

        final_ledger = sl.read_ledger(state_dir)
        live_entry = loop._latest_stage(final_ledger, "episodic", "tier_live")
        assert live_entry is not None
        artifact_paths = {a["path"] for a in live_entry["artifacts"]}
        assert str(written_slot / "meta.json") in artifact_paths


class TestInterimScratchDirIsDisjointFromTierRoot:
    """An interim adapter's training scratch dir (``_training_output_dir``,
    recorded verbatim into the ledger's ``tiers[*]["scratch"]`` and
    ``rmtree``d whole by ``dispose``) must never be the interim tier root
    itself, or any ancestor of it -- that root also holds the promoted slot
    dir, ``indexed_key_registry.json`` and ``key_metadata.json``, none of
    which disposal may touch."""

    def test_scratch_dir_is_nested_under_and_disjoint_from_the_tier_root(self, tmp_path):
        loop = _make_loop(tmp_path)
        stamp = "20260818T2200"
        adapter_name = f"episodic_interim_{stamp}"

        scratch_dir = loop._training_output_dir(adapter_name)
        tier_root = loop.output_dir / "episodic" / f"interim_{stamp}"

        assert scratch_dir != tier_root
        assert scratch_dir not in tier_root.parents
        # The scratch dir is nested one level under the tier root -- never
        # the root itself -- so a whole-tree rmtree of it can never reach
        # the root's own published files.
        assert scratch_dir.parent == tier_root

    def test_dispose_removes_scratch_but_preserves_the_published_interim_tier(self, tmp_path):
        loop = _make_loop(tmp_path)
        stamp = "20260818T2200"
        adapter_name = f"episodic_interim_{stamp}"

        tier_root = loop.output_dir / "episodic" / f"interim_{stamp}"
        tier_root.mkdir(parents=True)
        registry_path = tier_root / "indexed_key_registry.json"
        registry_path.write_text("{}")
        metadata_path = tier_root / "key_metadata.json"
        metadata_path.write_text("{}")
        slot_dir = tier_root / "20260818-220000"
        slot_dir.mkdir()
        (slot_dir / "meta.json").write_text("{}")
        (slot_dir / "adapter_model.safetensors").write_bytes(b"weights")

        scratch_dir = loop._training_output_dir(adapter_name)
        (scratch_dir / "checkpoint-1").mkdir(parents=True)
        (scratch_dir / "checkpoint-1" / "adapter_model.safetensors").write_bytes(b"scratch")
        (scratch_dir / "staging_resume.json").write_text("{}")

        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        payload = {
            "version": 2,
            "event": "interim",
            "venue": "weights",
            "stamp": stamp,
            "tiers": {
                adapter_name: {"adapter": adapter_name, "pre_sha": "", "scratch": str(scratch_dir)}
            },
            "stages": [
                sl.extraction_stage(
                    completed_at="2026-01-01T00:00:00+00:00",
                    sessions=[],
                    episodic_rels=0,
                    procedural_rels=0,
                    artifacts=[],
                )
            ],
            "absorbed_interim_tiers": [],
        }
        sl.ledger_path(state_dir).write_text(json.dumps(payload))

        disposed = sl.dispose(state_dir)

        assert disposed is True
        assert not scratch_dir.exists()
        # The tier root and every published surface it carries survive --
        # this is the production data-loss defect this test pins.
        assert tier_root.exists()
        assert registry_path.exists()
        assert metadata_path.exists()
        assert slot_dir.exists()
        assert (slot_dir / "meta.json").exists()
        assert (slot_dir / "adapter_model.safetensors").exists()


# ---------------------------------------------------------------------------
# read_ledger's readability gate: absent vs. every present-but-uninterpretable
# shape, and dispose()'s bypass of the same gate.
# ---------------------------------------------------------------------------


def _good_ledger_payload() -> dict:
    """A minimal, fully valid v2 ledger payload -- the positive control every
    malformed-shape test below is a mutation of."""
    return {
        "version": 2,
        "event": "interim",
        "venue": "weights",
        "stamp": "20260101T0000",
        "tiers": {},
        "stages": [
            sl.extraction_stage(
                completed_at="2026-01-01T00:00:00+00:00",
                sessions=[],
                episodic_rels=0,
                procedural_rels=0,
                artifacts=[],
            )
        ],
        "absorbed_interim_tiers": [],
    }


class TestReadLedgerAbsent:
    def test_no_file_returns_none(self, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        assert sl.read_ledger(state_dir) is None


class TestReadLedgerGoodPayloadParses:
    def test_a_well_formed_ledger_reads_back(self, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        sl.ledger_path(state_dir).write_text(json.dumps(_good_ledger_payload()))

        ledger = sl.read_ledger(state_dir)

        assert ledger is not None
        assert ledger.event == "interim"
        assert ledger.stamp == "20260101T0000"


class TestReadLedgerUndecodable:
    @pytest.mark.parametrize(
        "raw",
        [
            "not json at all {{{",
            "",
        ],
    )
    def test_unparseable_payload_raises_undecodable(self, tmp_path, raw):
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        sl.ledger_path(state_dir).write_text(raw)

        with pytest.raises(sl.StageLedgerUnreadable) as exc_info:
            sl.read_ledger(state_dir)
        assert exc_info.value.cause == "undecodable"


class TestReadLedgerUnopenable:
    def test_an_envelope_this_process_cannot_open_raises_unopenable(self, tmp_path, monkeypatch):
        """A present file whose bytes decode fine at the filesystem level but
        whose envelope this process cannot open (no daily identity loaded, a
        foreign envelope) -- simulated by making read_maybe_encrypted itself
        raise, exactly as it does for either real cause."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        sl.ledger_path(state_dir).write_text(json.dumps(_good_ledger_payload()))

        import paramem.backup.encryption as encryption_module

        def _raise(_path):
            raise RuntimeError("age envelope but no daily identity loaded")

        monkeypatch.setattr(encryption_module, "read_maybe_encrypted", _raise)

        with pytest.raises(sl.StageLedgerUnreadable) as exc_info:
            sl.read_ledger(state_dir)
        assert exc_info.value.cause == "unopenable"


class TestReadLedgerNotALedger:
    """A payload that decodes as JSON but is not a ledger of this schema --
    every distinct way _from_dict / _validate_stage_entry refuse it."""

    def test_top_level_list_raises_not_a_ledger(self, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        sl.ledger_path(state_dir).write_text(json.dumps([1, 2, 3]))

        with pytest.raises(sl.StageLedgerUnreadable) as exc_info:
            sl.read_ledger(state_dir)
        assert exc_info.value.cause == "not_a_ledger"

    def test_missing_head_field_raises_not_a_ledger(self, tmp_path):
        payload = _good_ledger_payload()
        del payload["venue"]
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        sl.ledger_path(state_dir).write_text(json.dumps(payload))

        with pytest.raises(sl.StageLedgerUnreadable) as exc_info:
            sl.read_ledger(state_dir)
        assert exc_info.value.cause == "not_a_ledger"

    def test_bad_event_vocabulary_raises_not_a_ledger(self, tmp_path):
        payload = _good_ledger_payload()
        payload["event"] = "bogus"
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        sl.ledger_path(state_dir).write_text(json.dumps(payload))

        with pytest.raises(sl.StageLedgerUnreadable) as exc_info:
            sl.read_ledger(state_dir)
        assert exc_info.value.cause == "not_a_ledger"

    def test_stage_entry_that_is_not_a_dict_raises_not_a_ledger(self, tmp_path):
        payload = _good_ledger_payload()
        payload["stages"].append("not a dict")
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        sl.ledger_path(state_dir).write_text(json.dumps(payload))

        with pytest.raises(sl.StageLedgerUnreadable) as exc_info:
            sl.read_ledger(state_dir)
        assert exc_info.value.cause == "not_a_ledger"

    def test_extraction_entry_without_completed_at_raises_not_a_ledger(self, tmp_path):
        payload = _good_ledger_payload()
        del payload["stages"][0]["completed_at"]
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        sl.ledger_path(state_dir).write_text(json.dumps(payload))

        with pytest.raises(sl.StageLedgerUnreadable) as exc_info:
            sl.read_ledger(state_dir)
        assert exc_info.value.cause == "not_a_ledger"

    def test_non_iso_completed_at_raises_not_a_ledger(self, tmp_path):
        payload = _good_ledger_payload()
        payload["stages"][0]["completed_at"] = "not a timestamp"
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        sl.ledger_path(state_dir).write_text(json.dumps(payload))

        with pytest.raises(sl.StageLedgerUnreadable) as exc_info:
            sl.read_ledger(state_dir)
        assert exc_info.value.cause == "not_a_ledger"

    def test_no_extraction_entry_at_all_raises_not_a_ledger(self, tmp_path):
        payload = _good_ledger_payload()
        payload["stages"] = []
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        sl.ledger_path(state_dir).write_text(json.dumps(payload))

        with pytest.raises(sl.StageLedgerUnreadable) as exc_info:
            sl.read_ledger(state_dir)
        assert exc_info.value.cause == "not_a_ledger"


class TestDisposeBypassesEveryReadabilityGate:
    """dispose() reads the raw payload directly, bypassing read_ledger's
    version gate and its unreadable-payload raise alike -- discarding an
    uninterpretable record is the intended escape hatch."""

    @pytest.mark.parametrize(
        "raw_text",
        [
            "not json at all {{{",
            "",
            json.dumps([1, 2, 3]),
            json.dumps(
                {
                    "version": 1,
                    **{k: v for k, v in _good_ledger_payload().items() if k != "version"},
                }
            ),
            json.dumps({**_good_ledger_payload(), "event": "bogus"}),
        ],
    )
    def test_dispose_succeeds_and_removes_the_file_over_every_unreadable_shape(
        self, tmp_path, raw_text
    ):
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        path = sl.ledger_path(state_dir)
        path.write_text(raw_text)

        disposed = sl.dispose(state_dir)

        assert disposed is True
        assert not path.exists()

    def test_dispose_of_a_well_formed_ledger_also_succeeds(self, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        path = sl.ledger_path(state_dir)
        path.write_text(json.dumps(_good_ledger_payload()))

        disposed = sl.dispose(state_dir)

        assert disposed is True
        assert not path.exists()

    def test_dispose_of_an_absent_ledger_is_a_safe_no_op(self, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)

        assert sl.dispose(state_dir) is False
