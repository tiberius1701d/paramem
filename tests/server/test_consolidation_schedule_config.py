"""Unit tests for ConsolidationScheduleConfig's scheduling-key validation:
``interim_resume``, ``full_window``, the ``interim_resume: tick`` /
``refresh_cadence`` pairing rule, the overflow guard, and quiet hours
sharing the one ``Window`` grammar.

Each case constructs ``ConsolidationScheduleConfig`` directly — no YAML file
involved except the blank/unquoted-integer bound cases, which load a copy of
``tests/fixtures/server.yaml`` with one key altered via ``load_server_config``
(mirroring how a real operator misconfiguration would surface).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from paramem.server.config import ConsolidationScheduleConfig, load_server_config

FIXTURE_PATH = "tests/fixtures/server.yaml"


def _load_with_consolidation_override(tmp_path: Path, **overrides):
    """Load the fixture server.yaml with one or more consolidation.* keys
    replaced, written to tmp_path so the fixture file itself is never edited.
    """
    raw = yaml.safe_load(Path(FIXTURE_PATH).read_text())
    raw.setdefault("consolidation", {}).update(overrides)
    altered = tmp_path / "server.yaml"
    altered.write_text(yaml.safe_dump(raw))
    return load_server_config(str(altered))


# ---------------------------------------------------------------------------
# interim_resume — accepted forms
# ---------------------------------------------------------------------------


class TestInterimResumeAcceptedForms:
    def test_immediate_is_accepted(self) -> None:
        cfg = ConsolidationScheduleConfig(interim_resume="immediate")
        assert cfg.interim_resume == "immediate"

    def test_tick_is_accepted_with_a_marked_cadence(self) -> None:
        cfg = ConsolidationScheduleConfig(interim_resume="tick", refresh_cadence="12h")
        assert cfg.interim_resume == "tick"

    def test_window_form_is_accepted(self) -> None:
        cfg = ConsolidationScheduleConfig(interim_resume="02:00-05:00")
        assert cfg.interim_resume == "02:00-05:00"


# ---------------------------------------------------------------------------
# interim_resume — rejected forms name the three legal shapes
# ---------------------------------------------------------------------------


class TestInterimResumeRejectedForms:
    def test_unrecognised_text_names_the_three_shapes_and_the_value(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            ConsolidationScheduleConfig(interim_resume="bogus")
        message = str(exc_info.value)
        assert "immediate" in message
        assert "tick" in message
        assert "HH:MM-HH:MM" in message
        assert "bogus" in message

    def test_start_equals_end_names_the_form_and_the_value(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            ConsolidationScheduleConfig(interim_resume="04:00-04:00")
        message = str(exc_info.value)
        assert "HH:MM-HH:MM" in message
        assert "04:00-04:00" in message


# ---------------------------------------------------------------------------
# full_window — rejection names the shape
# ---------------------------------------------------------------------------


class TestFullWindowRejectedForms:
    def test_unrecognised_text_names_the_shape_and_the_value(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            ConsolidationScheduleConfig(full_window="bogus")
        message = str(exc_info.value)
        assert "HH:MM-HH:MM" in message
        assert "bogus" in message

    def test_start_equals_end_names_the_form_and_the_value(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            ConsolidationScheduleConfig(full_window="04:00-04:00")
        message = str(exc_info.value)
        assert "HH:MM-HH:MM" in message
        assert "04:00-04:00" in message

    def test_valid_window_is_accepted(self) -> None:
        cfg = ConsolidationScheduleConfig(full_window="02:00-05:00")
        assert cfg.full_window == "02:00-05:00"


# ---------------------------------------------------------------------------
# interim_resume: "tick" needs a cadence with wall-clock marks
# ---------------------------------------------------------------------------


class TestInterimResumeTickCadencePairing:
    def test_off_cadence_is_rejected_naming_the_pairing_and_both_values(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            ConsolidationScheduleConfig(interim_resume="tick", refresh_cadence="off")
        message = str(exc_info.value)
        assert "interim_resume='tick'" in message
        assert "refresh_cadence='off'" in message

    def test_non_exact_interval_is_rejected_naming_the_pairing_and_both_values(self) -> None:
        """'every 7h' does not divide 24 -- no wall-clock marks to resume on."""
        with pytest.raises(ValueError) as exc_info:
            ConsolidationScheduleConfig(interim_resume="tick", refresh_cadence="every 7h")
        message = str(exc_info.value)
        assert "interim_resume='tick'" in message
        assert "refresh_cadence='every 7h'" in message

    @pytest.mark.parametrize("cadence", ["12h", "daily 04:00", "weekly"])
    def test_marked_cadences_pass(self, cadence: str) -> None:
        cfg = ConsolidationScheduleConfig(interim_resume="tick", refresh_cadence=cadence)
        assert cfg.interim_resume == "tick"


# ---------------------------------------------------------------------------
# max_interim_count > 0 requires a valid refresh_cadence, naming the key
# ---------------------------------------------------------------------------


class TestMaxInterimCountPositiveRequiresValidCadence:
    def test_malformed_cadence_refused_naming_the_key(self) -> None:
        with pytest.raises(ValueError, match="refresh_cadence"):
            ConsolidationScheduleConfig(max_interim_count=7, refresh_cadence="12x")


# ---------------------------------------------------------------------------
# Overflow guard: max_interim_count >= ceil(86400 / refresh_seconds)
# ---------------------------------------------------------------------------


class TestOverflowGuard:
    def test_fires_below_the_floor_at_12h_cadence_count_1(self) -> None:
        """ceil(86400 / 43200) = 2; count=1 is below the floor."""
        with pytest.raises(ValueError, match="max_interim_count"):
            ConsolidationScheduleConfig(refresh_cadence="12h", max_interim_count=1)

    def test_passes_at_the_floor_12h_cadence_count_7(self) -> None:
        cfg = ConsolidationScheduleConfig(refresh_cadence="12h", max_interim_count=7)
        assert cfg.max_interim_count == 7

    def test_passes_exactly_at_the_floor_1h_cadence_count_24(self) -> None:
        """ceil(86400 / 3600) = 24; count=24 is exactly the floor, not below it."""
        cfg = ConsolidationScheduleConfig(refresh_cadence="1h", max_interim_count=24)
        assert cfg.max_interim_count == 24

    def test_passes_at_weekly_cadence_count_1(self) -> None:
        """ceil(86400 / 604800) = 1; count=1 meets the floor."""
        cfg = ConsolidationScheduleConfig(refresh_cadence="weekly", max_interim_count=1)
        assert cfg.max_interim_count == 1

    def test_skipped_at_max_interim_count_zero(self) -> None:
        """No ring, no overflow to guard against -- a cadence that would
        otherwise fail the 12h/1 floor is accepted at count=0."""
        cfg = ConsolidationScheduleConfig(refresh_cadence="12h", max_interim_count=0)
        assert cfg.max_interim_count == 0

    def test_skipped_at_an_off_cadence(self) -> None:
        """No period to measure the floor against -- the guard is skipped,
        not failed."""
        cfg = ConsolidationScheduleConfig(refresh_cadence="off", max_interim_count=1)
        assert cfg.refresh_cadence == "off"


# ---------------------------------------------------------------------------
# Quiet hours share the one Window grammar under mode="auto" only.
# ---------------------------------------------------------------------------


class TestOverflowSlackValidation:
    def test_negative_interim_overflow_slack_refused_naming_the_key(self) -> None:
        with pytest.raises(ValueError, match="interim_overflow_slack"):
            ConsolidationScheduleConfig(interim_overflow_slack=-1)


# ---------------------------------------------------------------------------
# consolidation.mode is a closed vocabulary ("train" | "simulate")
# ---------------------------------------------------------------------------


class TestModeVocabulary:
    def test_a_value_outside_train_or_simulate_is_refused(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            ConsolidationScheduleConfig(mode="simulated")


# ---------------------------------------------------------------------------
# max_interim_count == 0 together with mode == "simulate" stalls ingestion.
# ---------------------------------------------------------------------------


class TestMaxInterimCountZeroModePairing:
    def test_count_zero_with_simulate_mode_refused_naming_both_keys(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            ConsolidationScheduleConfig(max_interim_count=0, mode="simulate", refresh_cadence="12h")
        message = str(exc_info.value)
        assert "max_interim_count" in message
        assert "mode" in message

    def test_count_zero_with_an_off_cadence_is_refused_by_the_cadence_check_first(self) -> None:
        """At ``max_interim_count=0`` the cadence requirement
        (``__post_init__``'s own earlier check) is read before the
        count/mode pairing check below it -- an off cadence at count=0
        surfaces the cadence-missing message, not the mode-pairing one, even
        when ``mode='simulate'`` would also fail its own check further
        down."""
        with pytest.raises(ValueError) as exc_info:
            ConsolidationScheduleConfig(max_interim_count=0, mode="simulate", refresh_cadence="off")
        message = str(exc_info.value)
        assert "requires a non-empty refresh_cadence" in message
        assert "stalls session ingestion" not in message


class TestQuietHoursWindowValidation:
    def test_bad_hhmm_shape_rejected_under_auto_naming_field_and_value(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            ConsolidationScheduleConfig(
                quiet_hours_mode="auto", quiet_hours_start="9:5", quiet_hours_end="07:00"
            )
        message = str(exc_info.value)
        assert "quiet_hours_start" in message
        assert "9:5" in message

    def test_equal_bounds_rejected_under_auto_naming_field_and_value(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            ConsolidationScheduleConfig(
                quiet_hours_mode="auto", quiet_hours_start="22:00", quiet_hours_end="22:00"
            )
        message = str(exc_info.value)
        assert "quiet_hours_start" in message
        assert "22:00" in message

    @pytest.mark.parametrize("mode", ["always_on", "always_off"])
    def test_bad_shape_accepted_under_non_auto_modes(self, mode: str) -> None:
        """always_on/always_off never build a Window, so a malformed pair
        (which would fail under auto) is not even read."""
        cfg = ConsolidationScheduleConfig(
            quiet_hours_mode=mode, quiet_hours_start="9:5", quiet_hours_end="9:5"
        )
        assert cfg.quiet_hours_mode == mode


# ---------------------------------------------------------------------------
# Blank (None) / unquoted-integer bounds -- a window-shape error naming the
# field and the value, driven through the real YAML load path.
# ---------------------------------------------------------------------------


class TestBlankAndUnquotedIntegerBoundsThroughYamlLoad:
    def test_blank_full_window_refused_naming_field_and_value(self, tmp_path: Path) -> None:
        """An empty YAML scalar (consolidation.full_window:) parses to None."""
        with pytest.raises(ValueError) as exc_info:
            _load_with_consolidation_override(tmp_path, full_window=None)
        message = str(exc_info.value)
        assert "full_window" in message

    def test_blank_interim_resume_refused_naming_field_and_value(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError) as exc_info:
            _load_with_consolidation_override(tmp_path, interim_resume=None)
        message = str(exc_info.value)
        assert "interim_resume" in message

    def test_blank_quiet_hours_start_refused_under_auto(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError) as exc_info:
            _load_with_consolidation_override(
                tmp_path, quiet_hours_mode="auto", quiet_hours_start=None
            )
        message = str(exc_info.value)
        assert "quiet_hours_start" in message

    def test_unquoted_integer_quiet_hours_start_refused_under_auto(self, tmp_path: Path) -> None:
        """YAML 1.1 sexagesimal parsing turns an unquoted '22:00' into the
        integer 1320 -- not a str, so the window builder refuses it."""
        with pytest.raises(ValueError) as exc_info:
            _load_with_consolidation_override(
                tmp_path, quiet_hours_mode="auto", quiet_hours_start=1320
            )
        message = str(exc_info.value)
        assert "quiet_hours_start" in message
        assert "1320" in message
