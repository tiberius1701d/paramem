"""Pydantic models for knowledge graph schema."""

from typing import Literal

from pydantic import BaseModel, Field

from paramem.graph.schema_config import relation_types

_RelationType = Literal[relation_types()]  # type: ignore[valid-type]


class Entity(BaseModel):
    """A typed entity extracted from a session transcript."""

    name: str = Field(description="Canonical name of the entity")
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
        description="Key-value attributes (e.g. age, role)",
    )


class Relation(BaseModel):
    """A typed relation between entities or an entity and a value."""

    subject: str = Field(description="Source entity name")
    predicate: str = Field(description="Relation type (e.g. lives_in, works_at, prefers)")
    object: str = Field(description="Target entity name or literal value")
    relation_type: _RelationType = Field(description="Category of relation")
    confidence: float = Field(default=1.0, ge=0.0)


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
