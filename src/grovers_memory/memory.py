"""Public API: only strings cross the retrieval boundary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime

from .embeddings import Embedder, HashEmbedder, normalize
from .graph import personalized_pagerank, relationship
from .models import (
    ConflictError,
    Fact,
    FactInput,
    RetrievalConfig,
    ValidationError,
    clean,
    iso,
    timestamp,
    unit_interval,
    utcnow,
)
from .ranking import amplify, rank
from .storage import Store


class Memory:
    def __init__(
        self,
        database_url: str = "sqlite:///memory.db",
        *,
        embedder: Embedder | None = None,
        config: RetrievalConfig | None = None,
        clock: Callable[[], datetime] = utcnow,
    ):
        self.embedder = embedder or HashEmbedder()
        dimensions = self.embedder.dimensions
        if (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or not 16 <= dimensions <= 2000
        ):
            raise ValidationError("Encoder dimensions must be an integer between 16 and 2000.")
        clean(self.embedder.name, "encoder name", 1024)
        self.config = config or RetrievalConfig()
        self.clock = clock
        self._store = Store(database_url, self.embedder.name, self.embedder.dimensions)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._store.close()

    @staticmethod
    def _user(user_id):
        clean(user_id, "user_id", 256)
        if user_id != user_id.strip():
            raise ValidationError("user_id cannot start or end with whitespace.")
        # Tenant IDs are opaque: Unicode normalization could merge different accounts.
        return user_id

    def add(self, user_id: str, fact: FactInput) -> Fact:
        return self.add_many(user_id, [fact])[0]

    def add_many(self, user_id: str, facts: list[FactInput]) -> list[Fact]:
        """Insert an atomic batch. Keyed revisions replace only that user's same key."""
        user_id = self._user(user_id)
        now = timestamp(self.clock())
        if not isinstance(facts, (list, tuple)) or len(facts) > 1000:
            raise ValidationError("A batch must contain at most 1000 FactInput objects.")
        if any(not isinstance(f, FactInput) for f in facts):
            raise ValidationError("Use FactInput objects.")
        prepared = [f.normalized(now) for f in facts]
        vectors = [
            normalize(self.embedder.embed(f.text), self.embedder.dimensions) for f in prepared
        ]
        result_ids = []
        with self._store.transaction(write=True, user_id=user_id):
            for item, vector in zip(prepared, vectors, strict=True):
                identity = json.dumps(
                    [
                        user_id,
                        item.key,
                        item.text if item.key is None else None,
                        iso(item.observed_at),
                    ]
                )
                fact_id = hashlib.sha256(identity.encode()).hexdigest()
                existing = self._store.get(user_id, fact_id)
                if existing:
                    same = all(
                        getattr(existing, name) == getattr(item, name)
                        for name in (
                            "text",
                            "key",
                            "entities",
                            "tags",
                            "observed_at",
                            "expires_at",
                            "confidence",
                            "importance",
                            "source_id",
                            "evidence",
                        )
                    )
                    if not same:
                        raise ConflictError(
                            "A different fact occupies this key and observation time."
                        )
                    result_ids.append(fact_id)
                    continue
                history = self._store.history(user_id, item.key) if item.key else []
                earlier = [f for f in history if f.observed_at < item.observed_at]
                later = [f for f in history if f.observed_at > item.observed_at]
                predecessor = max(earlier, key=lambda f: f.observed_at) if earlier else None
                successor = min(later, key=lambda f: f.observed_at) if later else None
                end_at = successor.observed_at if successor else None
                end_by = successor.id if successor else None
                # A deleted successor must not reopen an old interval on late insertion.
                if (
                    predecessor
                    and predecessor.superseded_at
                    and predecessor.superseded_at > item.observed_at
                    and (end_at is None or predecessor.superseded_at < end_at)
                ):
                    end_at, end_by = predecessor.superseded_at, predecessor.superseded_by
                fact = Fact(
                    id=fact_id,
                    user_id=user_id,
                    text=item.text,
                    key=item.key,
                    entities=item.entities,
                    tags=item.tags,
                    created_at=now,
                    observed_at=item.observed_at,
                    expires_at=item.expires_at,
                    superseded_at=end_at,
                    superseded_by=end_by,
                    confidence=item.confidence,
                    importance=item.importance,
                    source_id=item.source_id,
                    evidence=item.evidence,
                    embedding=vector,
                )
                self._store.insert(fact)
                if predecessor and (
                    predecessor.superseded_at is None
                    or predecessor.superseded_at > fact.observed_at
                ):
                    self._store.supersede(user_id, predecessor.id, fact.observed_at, fact.id)
                # Keep expired/superseded facts in history, never in the live graph.
                neighbors = []
                for other in self._store.history(user_id):
                    overlap = max(fact.observed_at, other.observed_at)
                    if other.id != fact.id and fact.active_at(overlap) and other.active_at(overlap):
                        neighbors.append((relationship(fact, other), other))
                neighbors.sort(key=lambda pair: (-pair[0], pair[1].id))
                for weight, other in neighbors[: self.config.graph_degree]:
                    if weight > 0 and weight >= self.config.relationship_threshold:
                        self._store.add_edge(user_id, fact.id, other.id, weight, "automatic", now)
                result_ids.append(fact.id)
            return [self._store.get(user_id, fact_id) for fact_id in result_ids]

    def ingest(self, user_id: str, text: str, *, extractor, source_id: str | None = None):
        """The extractor sees source text, not database history or tenant selection."""
        self._user(user_id)
        now = timestamp(self.clock())
        facts = extractor.extract(text, observed_at=now, source_id=source_id)
        return self.add_many(user_id, facts)

    def history(self, user_id: str, *, key: str | None = None) -> list[Fact]:
        user_id = self._user(user_id)
        key = clean(key, "key", 160).casefold() if key is not None else None
        with self._store.transaction():
            return self._store.history(user_id, key)

    def link(self, user_id: str, a: str, b: str, *, weight: float = 1.0):
        user_id = self._user(user_id)
        weight = unit_interval(weight, "weight")
        if not weight or a == b:
            raise ValidationError("Link two different facts with a positive weight.")
        with self._store.transaction(write=True, user_id=user_id):
            if any(self._store.get(user_id, f) is None for f in (a, b)):
                raise ValidationError("Both facts must belong to this user.")
            self._store.add_edge(user_id, a, b, weight, "manual", timestamp(self.clock()))

    def expire(self, user_id: str, fact_id: str, *, at: datetime | str | None = None):
        user_id = self._user(user_id)
        at = timestamp(at if at is not None else self.clock())
        with self._store.transaction(write=True, user_id=user_id):
            fact = self._store.get(user_id, fact_id)
            if fact is None or at <= fact.observed_at:
                raise ValidationError("Expiry must be after an existing fact's observation time.")
            if fact.expires_at is not None and at > fact.expires_at:
                raise ValidationError("Expiry cannot extend a fact; add a dated revision instead.")
            self._store.expire(user_id, fact_id, at)

    def forget(self, user_id: str, fact_id: str | None = None) -> int:
        """Delete content and edges. Previous revisions remain closed (no resurrection)."""
        user_id = self._user(user_id)
        with self._store.transaction(write=True, user_id=user_id):
            return self._store.delete(user_id, fact_id)

    def retrieve(
        self,
        user_id: str,
        query: str,
        *,
        as_of: datetime | str | None = None,
        strategy: str = "grover",
    ) -> list[str]:
        return self.explain(user_id, query, as_of=as_of, strategy=strategy)["context"]

    def explain(
        self,
        user_id: str,
        query: str,
        *,
        as_of: datetime | str | None = None,
        strategy: str = "grover",
    ) -> dict:
        """Developer diagnostics. Use retrieve() for the consuming model's context."""
        user_id = self._user(user_id)
        query = clean(query, "query", 8192)
        at = timestamp(as_of if as_of is not None else self.clock())
        if strategy not in {"grover", "hybrid", "vector"}:
            raise ValidationError("strategy must be grover, hybrid, or vector.")
        vector = normalize(self.embedder.embed(query), self.embedder.dimensions)
        with self._store.transaction():
            candidates = self._store.candidates(user_id, at, vector, self.config.candidate_limit)
            ranked = rank(candidates, query, vector, at, self.config, strategy)
            details = {
                "context": [],
                "as_of": iso(at),
                "strategy": strategy,
                "candidates": len(candidates),
                "anchor": None,
                "amplification": None,
                "graph": {"nodes": 0, "edges": 0, "iterations": 0, "converged": True},
            }
            if not ranked:
                return details
            anchor_index = 0
            if strategy == "grover":
                experiment = amplify([r.score for r in ranked], self.config.oracle_ratio)
                # Elite amplitudes are equal analytically. Resolve deterministic ties by
                # the relevance ordering rather than floating point roundoff or sampling.
                anchor_index = min(experiment["marked"]) if experiment["marked"] else 0
                details["amplification"] = experiment
            anchor = ranked[anchor_index]
            details["anchor"] = {
                "id": anchor.fact.id,
                "score": anchor.score,
                "relevance": anchor.relevance,
                "vector": anchor.vector_score,
                "lexical": anchor.lexical_score,
                "freshness": anchor.freshness,
            }
            nodes = {anchor.fact.id: anchor.fact}
            frontier = [anchor.fact.id]
            edges = {}
            for _ in range(self.config.graph_hops):
                found = self._store.edges(user_id, frontier, at, self.config.graph_nodes * 20)
                next_frontier = set()
                for edge in found:
                    for node_id in (edge.a, edge.b):
                        if node_id not in nodes and len(nodes) < self.config.graph_nodes:
                            fact = self._store.get(user_id, node_id)
                            if fact and fact.active_at(at):
                                nodes[node_id] = fact
                                next_frontier.add(node_id)
                    if edge.a in nodes and edge.b in nodes:
                        edges[(edge.a, edge.b, edge.kind)] = edge
                frontier = sorted(next_frontier)
                if not frontier:
                    break
            # Include edges between nodes reached on the last expansion as well.
            for edge in self._store.edges(user_id, sorted(nodes), at, self.config.graph_nodes * 20):
                if edge.a in nodes and edge.b in nodes:
                    edges[(edge.a, edge.b, edge.kind)] = edge
            probabilities, iterations, converged = personalized_pagerank(
                list(nodes),
                list(edges.values()),
                anchor.fact.id,
                self.config,
            )
            related = sorted(
                (node for node in nodes if node != anchor.fact.id and probabilities[node] > 0),
                key=lambda node: (-probabilities[node], node),
            )
            context = [anchor.fact.text]
            selected = [anchor.fact.id]
            for node in related:
                if nodes[node].text not in context:
                    context.append(nodes[node].text)
                    selected.append(node)
                if len(context) == 3:
                    break
            details["context"] = context
            details["graph"] = {
                "nodes": len(nodes),
                "edges": len(edges),
                "iterations": iterations,
                "converged": converged,
                "selected": [{"id": node, "rank": probabilities[node]} for node in selected],
            }
            return details
