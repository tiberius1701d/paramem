"""Pydantic models for knowledge graph schema."""

from typing import Literal

from pydantic import BaseModel, Field

from paramem.graph.schema_config import relation_types

_RelationType = Literal[relation_types()]  # type: ignore[valid-type]


class Entity(BaseModel):
    """A typed entity extracted from a session transcript.

    When the entity represents a speaker, ``speaker_id`` is populated with
    the speaker store's stable system ID (e.g. ``Speaker0``) and that ID is
    the canonical graph identity — ``name`` is a mutable display attribute
    that can change (anonymous "Speaker0" → disclosed "Tobias") without
    re-keying the graph. For non-speaker entities (places, organisations,
    concepts), ``speaker_id`` stays ``None`` and identity is ``name``.
    """

    name: str = Field(description="Canonical name (or display name when speaker_id is set)")
    entity_type: str = Field(
        description=(
            "Entity type. Common values: person, place, organization, event, "
            "preference, concept. Model may emit other types (product, "
            "certification, program, paper, etc.); downstream code accepts "
            "them. The schema YAML's entity_types list is a soft prior used "
            "for prompt examples, not enforcement."
        ),
    )
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value attributes (e.g. age, role, has_first_name, has_last_name)",
    )
    speaker_id: str | None = Field(
        default=None,
        description=(
            "Speaker store ID (e.g. 'Speaker0'). Populated iff this entity "
            "represents a speaker. When set, ``speaker_id`` is the canonical "
            "graph identity; ``name`` is a mutable display attribute."
        ),
    )


class Relation(BaseModel):
    """A typed relation between entities or an entity and a value.

    Every relation carries a ``speaker_id`` recording which speaker
    contributed the fact (provenance). For unknown speakers, the speaker
    store's anonymous group ID (``Speaker0``, ``Speaker1``, …) is used —
    there is never an absent ``speaker_id``. The router and recall layers
    read this field to scope retrieval per speaker.
    """

    subject: str = Field(description="Source entity name or speaker_id")
    predicate: str = Field(description="Relation type (e.g. lives_in, works_at, prefers)")
    object: str = Field(description="Target entity name, speaker_id, or literal value")
    relation_type: _RelationType = Field(description="Category of relation")
    confidence: float = Field(default=1.0, ge=0.0)
    speaker_id: str = Field(
        description=(
            "Speaker store ID of the speaker who contributed this fact. "
            "Mandatory; for unknown speakers this is the anonymous-group ID "
            "from SpeakerStore.register_anonymous (e.g. 'Speaker7')."
        ),
    )


class SessionGraph(BaseModel):
    """Structured knowledge graph extracted from a single session."""

    session_id: str
    timestamp: str = Field(description="ISO 8601 timestamp")
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    summary: str = Field(default="", description="One-line session summary")
    diagnostics: dict = Field(
        default_factory=dict,
        description=(
            "Pipeline audit trail — populated by extractor passes. Keys include: "
            "sota_raw_response, residual_dropped_facts, sota_updated_transcript. "
            "Only read by consolidation when dumping debug artefacts; never "
            "serialized to the cumulative graph.json or adapter weights. Always "
            "populated — callers who need to discard it should `model_copy` with "
            "`diagnostics={}` before persisting."
        ),
    )
