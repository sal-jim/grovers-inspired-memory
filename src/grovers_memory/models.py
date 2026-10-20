"""Validated facts and explicit UTC validity intervals."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


class MemoryErrorBase(Exception):
    """Base for actionable library errors (not Python's built-in MemoryError)."""


class ValidationError(MemoryErrorBase, ValueError):
    pass


class ConflictError(MemoryErrorBase):
    pass


class ConfigurationError(MemoryErrorBase):
    pass


class ExtractionError(MemoryErrorBase):
    pass


def utcnow() -> datetime:
    return datetime.now(UTC)


def timestamp(value: datetime | str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError("Use an ISO 8601 timestamp with a timezone.") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError("Timestamps must include a timezone; naive dates are ambiguous.")
    return value.astimezone(UTC)


def iso(value: datetime) -> str:
    return timestamp(value).isoformat(timespec="microseconds")


def clean(value: str, name: str, limit: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValidationError(f"{name} must be a non-empty string of at most {limit} characters.")
    if "\x00" in value:
        raise ValidationError(f"{name} cannot contain a NUL character.")
    normalized = unicodedata.normalize("NFKC", value.strip())
    if len(normalized) > limit:
        raise ValidationError(f"Normalized {name} exceeds {limit} characters.")
    return normalized


def labels(values: tuple[str, ...] | list[str], name: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)) or len(values) > 32:
        raise ValidationError(f"{name} must contain at most 32 strings.")
    return tuple(sorted({clean(v, name, 100).casefold() for v in values}))


def unit_interval(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValidationError(f"{name} must be a number between zero and one.")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValidationError(f"{name} must be a finite number between zero and one.")
    return float(value)


@dataclass(frozen=True, slots=True)
class FactInput:
    text: str
    key: str | None = None
    entities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    observed_at: datetime | str | None = None
    expires_at: datetime | str | None = None
    confidence: float = 1.0
    importance: float = 0.5
    source_id: str | None = None
    evidence: str | None = None

    def normalized(self, now: datetime) -> FactInput:
        observed = timestamp(self.observed_at if self.observed_at is not None else now)
        expires = timestamp(self.expires_at) if self.expires_at is not None else None
        if expires is not None and expires <= observed:
            raise ValidationError("expires_at must be later than observed_at.")
        if self.evidence is not None:
            clean(self.evidence, "evidence", 8192)
        return FactInput(
            text=clean(self.text, "text"),
            key=clean(self.key, "key", 160).casefold() if self.key is not None else None,
            entities=labels(self.entities, "entities"),
            tags=labels(self.tags, "tags"),
            observed_at=observed,
            expires_at=expires,
            confidence=unit_interval(self.confidence, "confidence"),
            importance=unit_interval(self.importance, "importance"),
            source_id=clean(self.source_id, "source_id", 256)
            if self.source_id is not None
            else None,
            evidence=self.evidence.strip() if self.evidence is not None else None,
        )


@dataclass(frozen=True, slots=True)
class Fact:
    id: str
    user_id: str
    text: str
    key: str | None
    entities: tuple[str, ...]
    tags: tuple[str, ...]
    created_at: datetime
    observed_at: datetime
    expires_at: datetime | None
    superseded_at: datetime | None
    superseded_by: str | None
    confidence: float
    importance: float
    source_id: str | None
    evidence: str | None
    embedding: tuple[float, ...]

    def active_at(self, at: datetime) -> bool:
        at = timestamp(at)
        return (
            self.observed_at <= at
            and (self.expires_at is None or at < self.expires_at)
            and (self.superseded_at is None or at < self.superseded_at)
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        del result["embedding"]
        for name in ("created_at", "observed_at", "expires_at", "superseded_at"):
            result[name] = iso(result[name]) if result[name] is not None else None
        return result


@dataclass(frozen=True, slots=True)
class Edge:
    a: str
    b: str
    weight: float
    kind: str = "automatic"


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    candidate_limit: int = 128
    graph_nodes: int = 128
    graph_hops: int = 3
    graph_degree: int = 8
    relationship_threshold: float = 0.18
    min_relevance: float = 0.12
    half_life_days: float = 90.0
    restart_probability: float = 0.20
    pagerank_iterations: int = 200
    pagerank_tolerance: float = 1e-12
    oracle_ratio: float = 0.90

    def __post_init__(self) -> None:
        for name, low, high in (
            ("candidate_limit", 1, 10000),
            ("graph_nodes", 1, 10000),
            ("graph_hops", 0, 10),
            ("graph_degree", 0, 100),
            ("pagerank_iterations", 1, 10000),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValidationError(f"{name} must be an integer between {low} and {high}.")
        for name in ("relationship_threshold", "min_relevance", "oracle_ratio"):
            unit_interval(getattr(self, name), name)
        if not 0 < unit_interval(self.restart_probability, "restart_probability") < 1:
            raise ValidationError("restart_probability must be strictly between zero and one.")
        for name in ("half_life_days", "pagerank_tolerance"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValidationError(f"{name} must be finite and positive.")
