"""Global observed-language tracker.

Records ISO 639-1 codes of languages detected by STT with high confidence,
so HA's conversation-agent prompt can know which languages are plausible when
interpreting potentially mangled transcripts (especially under CPU fallback STT).

Scope is global (household), not per-speaker — intentionally simple.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Detection threshold — GPU distil-large-v3 returns 0.95+ on clean audio; CPU
# fallback small returns 0.7–0.85. Set at 0.7 so CPU-fallback detections are
# captured (that's the exact case this tracker is meant to help with). Pure
# noise typically scores below 0.5, so this still filters garbage.
DEFAULT_MIN_PROB = 0.7

# HA entity that receives the comma-separated language list.
DEFAULT_HA_ENTITY = "input_text.voice_observed_languages"


class LanguageTracker:
    """Maintains a persistent set of observed household languages."""

    def __init__(
        self,
        store_path: Path,
        ha_client=None,
        ha_entity_id: str = DEFAULT_HA_ENTITY,
        min_prob: float = DEFAULT_MIN_PROB,
    ):
        self._path = store_path
        self._ha_client = ha_client
        self._ha_entity_id = ha_entity_id
        self._min_prob = min_prob
        self._languages: set[str] = set()
        self._load()
        # Republish on startup — HA virtual states don't persist across HA
        # restarts, so a mature household (all languages already in the set)
        # would otherwise show "unknown" until a new language is detected.
        if self._languages:
            self._publish()

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self._languages = set(data.get("languages", []))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read %s: %s — starting empty", self._path, e)

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({"languages": sorted(self._languages)}))

    @property
    def languages(self) -> list[str]:
        return sorted(self._languages)

    def record(self, language: str | None, probability: float) -> bool:
        """Record a detection. Returns True if the observed set changed."""
        if not language or probability < self._min_prob:
            return False
        if language in self._languages:
            return False
        self._languages.add(language)
        self._save()
        logger.info("Observed language set updated: %s", self.languages)
        self._publish()
        return True

    def _publish(self):
        if self._ha_client is None:
            return
        # set_state already swallows network errors and logs at WARNING —
        # no need to wrap in try/except here.
        self._ha_client.set_state(self._ha_entity_id, ",".join(self.languages))
