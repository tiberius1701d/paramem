"""Unit tests for PII sanitizer."""

from paramem.server.sanitizer import check_personal_content, sanitize_for_cloud


class TestPersonalContentDetection:
    def test_clean_query(self):
        assert check_personal_content("What's the weather in Berlin?") == []

    def test_clean_general_question(self):
        assert check_personal_content("How tall is the Eiffel Tower?") == []

    def test_possessive_wife(self):
        findings = check_personal_content("What should I get my wife for her birthday?")
        assert "possessive_personal" in findings

    def test_possessive_dog(self):
        findings = check_personal_content("Where is the nearest vet for my dog?")
        assert "possessive_personal" in findings

    def test_possessive_favorite(self):
        findings = check_personal_content("Book a table at my favorite restaurant")
        assert "possessive_personal" in findings

    def test_possessive_house(self):
        findings = check_personal_content("How far is the airport from my house?")
        assert "possessive_personal" in findings

    def test_self_referential_where(self):
        findings = check_personal_content("Where do I live?")
        assert "self_referential" in findings

    def test_self_referential_what(self):
        findings = check_personal_content("What's my name?")
        assert "self_referential" in findings

    def test_self_referential_work(self):
        findings = check_personal_content("What do I do for work?")
        assert "self_referential" in findings

    def test_personal_claim_live(self):
        findings = check_personal_content("I live in a small town near Frankfurt")
        assert "personal_claim" in findings

    def test_personal_claim_married(self):
        findings = check_personal_content("I'm married and have two kids")
        assert "personal_claim" in findings

    def test_device_control_clean(self):
        """Device control queries should pass through."""
        assert check_personal_content("Turn on the living room lights") == []

    def test_music_clean(self):
        assert check_personal_content("Play Queen on the office speaker") == []

    def test_weather_clean(self):
        assert check_personal_content("What's the weather today?") == []

    def test_our_home(self):
        findings = check_personal_content("How warm is our house right now?")
        assert "possessive_personal" in findings

    def test_my_children(self):
        findings = check_personal_content("Find activities for my children")
        assert "possessive_personal" in findings


class TestSanitizeForCloud:
    def test_mode_off_passes_everything(self):
        query, findings = sanitize_for_cloud("Where do I live?", mode="off")
        assert query == "Where do I live?"
        assert findings == []

    def test_mode_warn_passes_with_findings(self):
        query, findings = sanitize_for_cloud("What should I get my wife?", mode="warn")
        assert query == "What should I get my wife?"
        assert len(findings) > 0

    def test_mode_block_returns_none(self):
        query, findings = sanitize_for_cloud("What should I get my wife?", mode="block")
        assert query is None
        assert len(findings) > 0

    def test_clean_query_passes_all_modes(self):
        for mode in ("off", "warn", "block"):
            query, findings = sanitize_for_cloud("What's the weather?", mode=mode)
            assert query == "What's the weather?"
            assert findings == []

    def test_mode_block_clean_query_passes(self):
        query, findings = sanitize_for_cloud("Turn on the lights", mode="block")
        assert query == "Turn on the lights"
        assert findings == []
