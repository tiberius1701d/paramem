"""Pydantic models for knowledge graph schema."""

from typing import Literal

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A typed entity extracted from a session transcript."""

    name: str = Field(description="Canonical name of the entity")
    entity_type: Literal["person", "place", "organization", "concept", "preference", "event"] = (
        Field(description="Type of entity")
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
    relation_type: Literal["factual", "temporal", "preference", "social"] = Field(
        description="Category of relation"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SessionGraph(BaseModel):
    """Structured knowledge graph extracted from a single session."""

    session_id: str
    timestamp: str = Field(description="ISO 8601 timestamp")
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    summary: str = Field(default="", description="One-line session summary")
