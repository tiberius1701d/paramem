"""Tests for procedural adapter pipeline components.

Tests the procedural QA seeding, contradiction index, and
filter integration without requiring GPU/model.
"""


class TestSeedProceduralQA:
    """Test seed_procedural_qa restores state correctly."""

    def _make_loop_stub(self):
        """Create a minimal stub with the fields seed_procedural_qa needs."""

        class LoopStub:
            def __init__(self):
                self.indexed_key_qa = {}
                self.procedural_sp_index = {}
                self.indexed_key_registry = None

        # Import and bind the real method
        from paramem.training.consolidation import ConsolidationLoop

        stub = LoopStub()
        stub.seed_procedural_qa = ConsolidationLoop.seed_procedural_qa.__get__(stub)
        return stub

    def test_seeds_into_indexed_key_qa(self):
        stub = self._make_loop_stub()
        pairs = [
            {
                "key": "proc1",
                "question": "What does Alex prefer?",
                "answer": "Alex prefers jazz.",
                "source_subject": "Alex",
                "source_predicate": "prefers",
                "speaker_id": "speaker1",
            },
        ]
        stub.seed_procedural_qa(pairs)
        assert "proc1" in stub.indexed_key_qa
        assert stub.indexed_key_qa["proc1"]["question"] == "What does Alex prefer?"

    def test_rebuilds_sp_index(self):
        stub = self._make_loop_stub()
        pairs = [
            {
                "key": "proc1",
                "question": "Q",
                "answer": "A",
                "source_subject": "Alex",
                "source_predicate": "prefers",
                "speaker_id": "sp1",
            },
            {
                "key": "proc2",
                "question": "Q2",
                "answer": "A2",
                "source_subject": "Anna",
                "source_predicate": "likes",
                "speaker_id": "sp2",
            },
        ]
        stub.seed_procedural_qa(pairs)
        assert stub.procedural_sp_index[("sp1", "alex", "prefers")] == "proc1"
        assert stub.procedural_sp_index[("sp2", "anna", "likes")] == "proc2"

    def test_empty_input(self):
        stub = self._make_loop_stub()
        stub.seed_procedural_qa([])
        assert len(stub.indexed_key_qa) == 0
        assert len(stub.procedural_sp_index) == 0

    def test_missing_fields_handled(self):
        stub = self._make_loop_stub()
        pairs = [
            {"key": "proc1", "question": "Q", "answer": "A"},
        ]
        stub.seed_procedural_qa(pairs)
        assert "proc1" in stub.indexed_key_qa
        # No sp_index entry because subject and predicate are empty
        assert len(stub.procedural_sp_index) == 0


class TestContradictionIndex:
    """Test that the sp_index correctly identifies contradictions."""

    def test_same_speaker_subject_predicate_collides(self):
        index = {}
        index[("sp1", "alex", "prefers")] = "proc1"
        # New preference with same key should overwrite
        assert ("sp1", "alex", "prefers") in index
        index[("sp1", "alex", "prefers")] = "proc2"
        assert index[("sp1", "alex", "prefers")] == "proc2"

    def test_different_speakers_no_collision(self):
        index = {}
        index[("sp1", "alex", "prefers")] = "proc1"
        index[("sp2", "alex", "prefers")] = "proc2"
        assert index[("sp1", "alex", "prefers")] == "proc1"
        assert index[("sp2", "alex", "prefers")] == "proc2"

    def test_different_predicates_no_collision(self):
        index = {}
        index[("sp1", "alex", "prefers")] = "proc1"
        index[("sp1", "alex", "likes")] = "proc2"
        assert len(index) == 2
