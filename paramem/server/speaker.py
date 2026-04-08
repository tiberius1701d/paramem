"""Voice-based speaker identification via embeddings.

Speaker identity = voice embedding (authenticator) + display name (label).
Multiple speakers can share the same name — each gets a unique internal ID.

Profiles stored as JSON with UUID-based speaker IDs. Each profile holds
multiple embeddings (from different utterances and devices). Matching
uses the L2-normalized centroid for robustness.

{"speakers": {"a1b2c3d4": {"name": "Tobias", "embeddings": [[0.1, ...], ...]}}, "version": 3}

Embedding computation happens externally (Wyoming STT wrapper) —
this module only stores and matches pre-computed embeddings.
"""

import json
import logging
import math
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_PROFILE_VERSION = 3


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

        if version >= _PROFILE_VERSION:
            self._profiles = data.get("speakers", {})
        elif version == 2:
            # v2 → v3: single "embedding" → list "embeddings"
            legacy = data.get("speakers", {})
            self._profiles = {}
            for speaker_id, profile in legacy.items():
                emb = profile.get("embedding", [])
                self._profiles[speaker_id] = {
                    "name": profile["name"],
                    "embeddings": [emb] if emb else [],
                }
            if self._profiles:
                logger.info("Migrated %d v2 profiles to v3 (multi-embedding)", len(self._profiles))
                self._save()
        else:
            # v1: {"speakers": {"Alice": [0.1, ...]}}
            legacy = data.get("speakers", {})
            self._profiles = {}
            for name, embedding in legacy.items():
                speaker_id = uuid.uuid4().hex[:8]
                self._profiles[speaker_id] = {
                    "name": name,
                    "embeddings": [embedding] if embedding else [],
                }
            if self._profiles:
                logger.info("Migrated %d v1 profiles to v3", len(self._profiles))
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
                    {"speakers": self._profiles, "version": _PROFILE_VERSION},
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
            logger.debug(
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

    def enroll(self, name: str, embedding: list[float]) -> str | None:
        """Register a new speaker profile.

        Rejects enrollment if the voice embedding already matches an
        existing profile at high confidence (prevents duplicate enrollment
        of the same voice under a different name).

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
            self._profiles[speaker_id] = {"name": display_name, "embeddings": [embedding]}
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

    @property
    def profile_count(self) -> int:
        return len(self._profiles)

    @property
    def speaker_names(self) -> list[str]:
        """Unique display names across all profiles."""
        return list({p["name"] for p in self._profiles.values()})
