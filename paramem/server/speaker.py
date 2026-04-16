"""Voice-based speaker identification via embeddings.

Speaker identity = voice embedding (authenticator) + display name (label).
Multiple speakers can share the same name — each gets a unique internal ID.

Profiles stored as JSON with UUID-based speaker IDs. Each profile holds
multiple embeddings (from different utterances and devices). Matching
uses the L2-normalized centroid for robustness.

{"speakers": {"a1b2c3d4": {"name": "Alex", "embeddings": [[0.1, ...], ...]}}, "version": 3}

Embedding computation happens externally (Wyoming STT wrapper) —
this module only stores and matches pre-computed embeddings.
"""

import json
import logging
import math
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_PROFILE_VERSION = 4


@dataclass
class SpeakerMatch:
    """Result of a speaker matching attempt."""

    speaker_id: str | None
    name: str | None
    confidence: float
    tentative: bool  # True if low_threshold <= confidence < high_threshold


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Pure Python, no numpy."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return dot / (norm_a * norm_b)


def _l2_normalize(v: list[float]) -> list[float]:
    """L2-normalize a vector. Returns zero vector if norm is near zero."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-8:
        return [0.0] * len(v)
    return [x / norm for x in v]


def compute_centroid(embeddings: list[list[float]]) -> list[float]:
    """Compute L2-normalized centroid from multiple embeddings.

    Per pyannote best practice: L2-normalize each embedding individually,
    average, then L2-normalize the result.
    """
    if not embeddings:
        return []
    if len(embeddings) == 1:
        return _l2_normalize(embeddings[0])

    dim = len(embeddings[0])
    normalized = [_l2_normalize(e) for e in embeddings]
    centroid = [sum(n[i] for n in normalized) / len(normalized) for i in range(dim)]
    return _l2_normalize(centroid)


class SpeakerStore:
    """Persistent store for speaker voice profiles.

    Each profile is keyed by a unique speaker ID (UUID4 short hash).
    Profiles store multiple embeddings — matching uses the centroid.
    New embeddings are added on confirmed matches to strengthen the profile.
    """

    def __init__(
        self,
        store_path: Path | str,
        high_threshold: float = 0.60,
        low_threshold: float = 0.45,
        max_embeddings: int = 50,
        redundancy_threshold: float = 0.95,
    ):
        self.store_path = Path(store_path)
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.max_embeddings = max_embeddings
        self.redundancy_threshold = redundancy_threshold
        self._profiles: dict[str, dict] = {}  # speaker_id → {name, embeddings}
        self._centroids: dict[str, list[float]] = {}  # cached centroids per profile
        self._last_greeted: dict[str, str] = {}  # speaker_id → ISO timestamp
        self._lock = threading.Lock()  # guards profile mutations and saves
        self._dirty = False  # deferred save flag for enrichment batching
        self._load()

    def _load(self) -> None:
        """Load profiles from disk, auto-migrating legacy formats."""
        if not self.store_path.exists():
            logger.info("No speaker profiles found at %s", self.store_path)
            return

        try:
            with open(self.store_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load speaker profiles: %s", e)
            return

        version = data.get("version", 1)

        self._last_greeted = data.get("last_greeted", {})

        if version >= _PROFILE_VERSION:
            self._profiles = data.get("speakers", {})
        elif version == 3:
            # v3 → v4: add preferred_language field
            self._profiles = data.get("speakers", {})
            for profile in self._profiles.values():
                profile.setdefault("preferred_language", "")
            if self._profiles:
                logger.info("Migrated %d v3 profiles to v4 (language)", len(self._profiles))
                self._save()
        elif version == 2:
            # v2 → v4: single "embedding" → list "embeddings" + preferred_language
            legacy = data.get("speakers", {})
            self._profiles = {}
            for speaker_id, profile in legacy.items():
                emb = profile.get("embedding", [])
                self._profiles[speaker_id] = {
                    "name": profile["name"],
                    "embeddings": [emb] if emb else [],
                    "preferred_language": "",
                }
            if self._profiles:
                logger.info("Migrated %d v2 profiles to v4", len(self._profiles))
                self._save()
        else:
            # v1 → v4: name-keyed → UUID-keyed + multi-embedding + preferred_language
            legacy = data.get("speakers", {})
            self._profiles = {}
            for name, embedding in legacy.items():
                speaker_id = uuid.uuid4().hex[:8]
                self._profiles[speaker_id] = {
                    "name": name,
                    "embeddings": [embedding] if embedding else [],
                    "preferred_language": "",
                }
            if self._profiles:
                logger.info("Migrated %d v1 profiles to v4", len(self._profiles))
                self._save()

        # Back-fill enroll_method for profiles written before that field existed.
        # One-time migration, persisted on first load after the upgrade.
        missing_method = [sid for sid, p in self._profiles.items() if "enroll_method" not in p]
        if missing_method:
            for sid in missing_method:
                self._profiles[sid]["enroll_method"] = "unknown"
            logger.info(
                "Back-filled enroll_method='unknown' on %d legacy profiles",
                len(missing_method),
            )
            self._save()

        logger.info("Loaded %d speaker profiles", len(self._profiles))
        self._rebuild_centroids()

    def _rebuild_centroids(self) -> None:
        """Recompute all cached centroids from current embeddings."""
        self._centroids = {}
        for speaker_id, profile in self._profiles.items():
            embeddings = profile.get("embeddings", [])
            if embeddings:
                self._centroids[speaker_id] = compute_centroid(embeddings)

    def _invalidate_centroid(self, speaker_id: str) -> None:
        """Recompute centroid for a single profile after mutation."""
        profile = self._profiles.get(speaker_id)
        if profile:
            embeddings = profile.get("embeddings", [])
            if embeddings:
                self._centroids[speaker_id] = compute_centroid(embeddings)
            else:
                self._centroids.pop(speaker_id, None)
        else:
            self._centroids.pop(speaker_id, None)

    @staticmethod
    def _time_period(hour: int) -> str:
        """Return the current time-of-day period."""
        if hour < 12:
            return "morning"
        if hour < 18:
            return "afternoon"
        return "evening"

    @staticmethod
    def _greeting_text(period: str) -> str:
        greetings = {
            "morning": "Good morning",
            "afternoon": "Good afternoon",
            "evening": "Good evening",
        }
        return greetings[period]

    def should_greet(self, speaker_id: str, interval_hours: int) -> str | None:
        """Return a time-appropriate greeting if the interval has elapsed
        or the time-of-day period changed since last greeting.

        Greets when either condition is met:
        a) More than interval_hours since last greeting, OR
        b) The time-of-day period (morning/afternoon/evening) changed.

        Returns None if disabled (interval_hours=0) or greeted in this period.
        Does NOT commit — caller must call confirm_greeting() after delivery.
        """
        if interval_hours <= 0:
            return None
        now = datetime.now(timezone.utc)
        hour = datetime.now().hour  # local time for period check
        current_period = self._time_period(hour)
        with self._lock:
            last_str = self._last_greeted.get(speaker_id)
        if last_str:
            last = datetime.fromisoformat(last_str)
            last_local_hour = last.astimezone().hour
            last_period = self._time_period(last_local_hour)
            hours_elapsed = (now - last).total_seconds() / 3600
            if current_period == last_period and hours_elapsed < interval_hours:
                return None
        return self._greeting_text(current_period)

    def confirm_greeting(self, speaker_id: str) -> None:
        """Mark the speaker as greeted and persist to disk."""
        with self._lock:
            self._last_greeted[speaker_id] = datetime.now(timezone.utc).isoformat()
            self._save()

    def flush(self) -> None:
        """Write deferred enrichment changes to disk. Call periodically."""
        with self._lock:
            if self._dirty:
                self._save()
                self._dirty = False

    def _save(self) -> None:
        """Persist profiles to disk (atomic write). Caller holds _lock."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store_path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "speakers": self._profiles,
                        "last_greeted": self._last_greeted,
                        "version": _PROFILE_VERSION,
                    },
                    f,
                    indent=2,
                )
            tmp.rename(self.store_path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def match(self, embedding: list[float]) -> SpeakerMatch:
        """Find the closest enrolled speaker to the given embedding.

        Compares against each profile's centroid (L2-normalized average
        of all stored embeddings). Returns confidence-based classification.
        """
        no_match = SpeakerMatch(speaker_id=None, name=None, confidence=0.0, tentative=False)
        if not self._profiles or not embedding:
            return no_match

        best_id = None
        best_score = -1.0

        for speaker_id, profile in self._profiles.items():
            centroid = self._centroids.get(speaker_id)
            if not centroid:
                continue
            score = cosine_similarity(embedding, centroid)
            logger.info(
                "Speaker match: %s score=%.4f (%d embeddings, thresholds: high=%.2f, low=%.2f)",
                profile.get("name", speaker_id),
                score,
                len(profile.get("embeddings", [])),
                self.high_threshold,
                self.low_threshold,
            )
            if score > best_score:
                best_score = score
                best_id = speaker_id

        confidence = max(best_score, 0.0)

        if best_id is None:
            return no_match

        name = self._profiles[best_id]["name"]

        if confidence >= self.high_threshold:
            return SpeakerMatch(
                speaker_id=best_id, name=name, confidence=confidence, tentative=False
            )
        elif confidence >= self.low_threshold:
            return SpeakerMatch(
                speaker_id=best_id, name=name, confidence=confidence, tentative=True
            )
        else:
            return SpeakerMatch(speaker_id=None, name=None, confidence=confidence, tentative=False)

    def add_embedding(self, speaker_id: str, embedding: list[float]) -> bool:
        """Add an embedding to an existing profile to strengthen the centroid.

        Returns True if added, False if profile not found or cap reached.
        Skips if the new embedding is too similar to the centroid (redundant).
        """
        if not embedding:
            return False

        with self._lock:
            profile = self._profiles.get(speaker_id)
            if not profile:
                return False

            embeddings = profile.get("embeddings", [])

            if embeddings:
                centroid = self._centroids.get(speaker_id) or compute_centroid(embeddings)
                sim = cosine_similarity(embedding, centroid)
                if sim > self.redundancy_threshold:
                    return False

            if len(embeddings) >= self.max_embeddings:
                return False

            embeddings.append(embedding)
            profile["embeddings"] = embeddings
            self._invalidate_centroid(speaker_id)
            self._dirty = True
        logger.info(
            "Enriched speaker %s (%s): now %d embeddings",
            speaker_id,
            profile["name"],
            len(embeddings),
        )
        return True

    def enroll(
        self, name: str, embedding: list[float], method: str = "self_introduced"
    ) -> str | None:
        """Register a new speaker profile.

        Rejects enrollment if the voice embedding already matches an
        existing profile at high confidence (prevents duplicate enrollment
        of the same voice under a different name).

        `method` records how the speaker was enrolled — surfaced via pstatus
        to distinguish self-introduction from voice-match recovery and manual
        setup. Valid values: "self_introduced", "voice_matched", "manual",
        "unknown" (migrated legacy profiles).

        Returns the new speaker_id on success, None on rejection.
        """
        display_name = name.strip()
        if not display_name or not embedding:
            logger.warning("Cannot enroll: empty name or embedding")
            return None

        with self._lock:
            # Reject if this voice is already enrolled (check inside lock
            # to prevent concurrent enrolls of the same voice).
            # NOTE: match() is read-only and must NOT acquire _lock (would deadlock).
            existing = self.match(embedding)
            if existing.speaker_id is not None and existing.confidence >= self.high_threshold:
                logger.warning(
                    "Enrollment rejected: voice already enrolled as '%s' (id=%s, conf=%.2f)",
                    existing.name,
                    existing.speaker_id,
                    existing.confidence,
                )
                return None

            speaker_id = uuid.uuid4().hex[:8]
            for _ in range(10):
                if speaker_id not in self._profiles:
                    break
                speaker_id = uuid.uuid4().hex[:16]  # wider ID space on retry
            # Final collision guard (astronomically unlikely)
            while speaker_id in self._profiles:
                speaker_id = uuid.uuid4().hex
            self._profiles[speaker_id] = {
                "name": display_name,
                "embeddings": [embedding],
                "preferred_language": "",
                "enroll_method": method,
            }
            self._invalidate_centroid(speaker_id)
            self._save()
        logger.info(
            "Enrolled speaker: %s (id=%s, %d-dim embedding)",
            display_name,
            speaker_id,
            len(embedding),
        )
        return speaker_id

    def remove(self, speaker_id: str) -> bool:
        """Remove a speaker profile by ID. Returns True if found and removed."""
        with self._lock:
            if speaker_id not in self._profiles:
                return False
            name = self._profiles[speaker_id]["name"]
            del self._profiles[speaker_id]
            self._centroids.pop(speaker_id, None)
            self._save()
        logger.info("Removed speaker profile: %s (id=%s)", name, speaker_id)
        return True

    def get_name(self, speaker_id: str) -> str | None:
        """Get the display name for a speaker ID."""
        profile = self._profiles.get(speaker_id)
        return profile["name"] if profile else None

    def get_preferred_language(self, speaker_id: str) -> str | None:
        """Get the preferred language for a speaker, or None if not set."""
        profile = self._profiles.get(speaker_id)
        if profile is None:
            return None
        lang = profile.get("preferred_language", "")
        return lang if lang else None

    def update_language(
        self, speaker_id: str, language: str, probability: float, threshold: float = 0.8
    ) -> None:
        """Update speaker's preferred language based on detected language.

        Only updates when detection confidence exceeds threshold. Uses
        simple majority: if the new language matches current preference,
        keep it. If different and high confidence, switch.
        """
        if probability < threshold:
            return
        with self._lock:
            profile = self._profiles.get(speaker_id)
            if profile is None:
                return
            current = profile.get("preferred_language", "")
            if current == language:
                return
            profile["preferred_language"] = language
            self._dirty = True
        logger.info(
            "Speaker %s language updated: %s → %s (prob=%.2f)",
            speaker_id,
            current or "(none)",
            language,
            probability,
        )

    @property
    def profile_count(self) -> int:
        return len(self._profiles)

    @property
    def speaker_names(self) -> list[str]:
        """Unique display names across all profiles."""
        return list({p["name"] for p in self._profiles.values()})

    def list_profiles(self) -> list[dict]:
        """Return a lightweight snapshot of all enrolled speakers.

        Each entry: {"id": str, "name": str, "embeddings": int,
        "preferred_language": str}. Embedding vectors are not included —
        only their count as a proxy for recognition events.
        """
        with self._lock:
            return [
                {
                    "id": speaker_id,
                    "name": profile.get("name", "?"),
                    "embeddings": len(profile.get("embeddings", [])),
                    "preferred_language": profile.get("preferred_language", ""),
                    "enroll_method": profile.get("enroll_method", "unknown"),
                }
                for speaker_id, profile in self._profiles.items()
            ]
