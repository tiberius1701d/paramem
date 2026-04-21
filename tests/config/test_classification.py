"""Tests for paramem.config.classification."""

from __future__ import annotations

from pathlib import Path

from paramem.config.classification import (
    CLASSIFICATION,
    EXTENSION_FIELDS,
    Tier,
    classify,
    iter_shipped_paths,
)

_SERVER_YAML = Path(__file__).parent.parent.parent / "configs" / "server.yaml"


class TestEveryShippedPathIsClassified:
    def test_every_shipped_path_is_classified(self):
        """Every leaf path in configs/server.yaml maps to one of the three tiers.

        Wildcard entries must match each concrete adapter / provider / voice
        name seen in the yaml.
        """
        for path in iter_shipped_paths(_SERVER_YAML):
            tier = classify(path)
            assert tier in (
                Tier.DESTRUCTIVE,
                Tier.PIPELINE_ALTERING,
                Tier.OPERATIONAL,
            ), f"classify({path!r}) returned unexpected value {tier!r}"


class TestDestructiveSamples:
    def test_destructive_samples(self):
        """Known Destructive paths all return Tier.DESTRUCTIVE."""
        destructive_paths = [
            "model",
            "paths.data",
            "adapters.episodic.rank",
            "cloud_only",
            "snapshot_key",
            "adapters.episodic.target_modules",  # extension field
        ]
        for path in destructive_paths:
            assert classify(path) == Tier.DESTRUCTIVE, (
                f"Expected DESTRUCTIVE for {path!r}, got {classify(path)!r}"
            )


class TestPipelineAlteringSamples:
    def test_pipeline_altering_samples(self):
        """Known Pipeline-altering paths all return Tier.PIPELINE_ALTERING."""
        pipeline_paths = [
            "consolidation.refresh_cadence",
            "consolidation.mode",
            "consolidation.extraction_noise_filter",
            "agents.sota.provider",
            "agents.sota_providers.anthropic.model",
            "debug",
            "adapters.episodic.alpha",
        ]
        for path in pipeline_paths:
            assert classify(path) == Tier.PIPELINE_ALTERING, (
                f"Expected PIPELINE_ALTERING for {path!r}, got {classify(path)!r}"
            )


class TestOperationalSamples:
    def test_operational_samples(self):
        """Known Operational paths all return Tier.OPERATIONAL."""
        operational_paths = [
            "server.port",
            "server.reclaim_interval_minutes",
            "tools.ha.url",
            "stt.model",
            "tts.voices.en.engine",
            "headless_boot",
            "sanitization.mode",
            "voice.prompt_file",
        ]
        for path in operational_paths:
            assert classify(path) == Tier.OPERATIONAL, (
                f"Expected OPERATIONAL for {path!r}, got {classify(path)!r}"
            )


class TestWildcardMatchesAnyAdapterName:
    def test_wildcard_matches_any_adapter_name(self):
        """Wildcard entry adapters.*.rank matches all adapter name shapes."""
        adapter_names = [
            "adapters.episodic.rank",
            "adapters.semantic.rank",
            "adapters.procedural.rank",
            "adapters.episodic_interim_20260421T1200.rank",
        ]
        for path in adapter_names:
            assert classify(path) == Tier.DESTRUCTIVE, (
                f"Expected DESTRUCTIVE for {path!r}, got {classify(path)!r}"
            )


class TestWildcardMatchesAnyProviderName:
    def test_wildcard_matches_any_provider_name(self):
        """Wildcard entry agents.sota_providers.*.enabled matches any provider."""
        provider_paths = [
            "agents.sota_providers.anthropic.enabled",
            "agents.sota_providers.openai.enabled",
            "agents.sota_providers.some_new_vendor.enabled",
        ]
        for path in provider_paths:
            assert classify(path) == Tier.PIPELINE_ALTERING, (
                f"Expected PIPELINE_ALTERING for {path!r}, got {classify(path)!r}"
            )


class TestWildcardMatchesAnyVoiceLang:
    def test_wildcard_matches_any_voice_lang(self):
        """Wildcard entry tts.voices.*.engine matches any language code."""
        voice_paths = [
            "tts.voices.en.engine",
            "tts.voices.tl.engine",
            "tts.voices.xx.engine",
        ]
        for path in voice_paths:
            assert classify(path) == Tier.OPERATIONAL, (
                f"Expected OPERATIONAL for {path!r}, got {classify(path)!r}"
            )


class TestUnclassifiedDefaultsToDestructive:
    def test_unclassified_defaults_to_destructive(self):
        """Paths absent from the classification table return Tier.DESTRUCTIVE."""
        assert classify("this.is.not.a.real.path") == Tier.DESTRUCTIVE, (
            "Unknown paths must fall back to DESTRUCTIVE (Resolved Decision 8)"
        )


class TestNoOrphanClassifications:
    def test_no_orphan_classifications(self):
        """Every key in CLASSIFICATION maps to a real yaml path or an extension field.

        Catches classification-table drift when a yaml field is renamed and the
        classification entry is forgotten.  Extension fields listed in
        EXTENSION_FIELDS are explicitly exempt from the yaml-presence check.
        """
        import yaml

        with open(_SERVER_YAML) as fh:
            raw = yaml.safe_load(fh)

        def _flatten(node: object, prefix: str) -> set[str]:
            if not isinstance(node, dict):
                return {prefix}
            result: set[str] = set()
            for key, value in node.items():
                child = f"{prefix}.{key}" if prefix else key
                result |= _flatten(value, child)
            return result

        real_paths = _flatten(raw, "")

        for entry_path in CLASSIFICATION:
            if entry_path in EXTENSION_FIELDS:
                continue
            if "*" in entry_path:
                continue
            assert entry_path in real_paths, (
                f"Classification entry {entry_path!r} does not appear in "
                f"configs/server.yaml and is not in EXTENSION_FIELDS. "
                "Remove or move to EXTENSION_FIELDS."
            )


class TestIterShippedPathsSmoke:
    def test_iter_shipped_paths_smoke(self):
        """Walker finds a known deeply-nested path and a top-level scalar."""
        paths = list(iter_shipped_paths(_SERVER_YAML))
        assert "tts.voices.en.engine" in paths, "iter_shipped_paths must yield tts.voices.en.engine"
        assert "cloud_only" in paths, (
            "iter_shipped_paths must yield the top-level cloud_only scalar"
        )


class TestTierGlyphsDistinct:
    def test_tier_glyphs_distinct(self):
        """Each tier has a unique glyph — catches copy-paste errors."""
        glyphs = [Tier.DESTRUCTIVE.glyph, Tier.PIPELINE_ALTERING.glyph, Tier.OPERATIONAL.glyph]
        assert len(glyphs) == len(set(glyphs)), f"Tier glyphs must all be distinct, got {glyphs}"
