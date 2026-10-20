"""Deterministic relationships and personalized PageRank, with no random sampling."""

from __future__ import annotations

import math

from .embeddings import cosine
from .models import Edge, Fact, RetrievalConfig, ValidationError

TRIVIAL_ENTITIES = {"i", "me", "user", "owner", "assistant", "you"}


def relationship(left: Fact, right: Fact) -> float:
    def jaccard(a: set[str], b: set[str]) -> float:
        return len(a & b) / len(a | b) if a | b else 0.0

    left_entities = set(left.entities) - TRIVIAL_ENTITIES
    right_entities = set(right.entities) - TRIVIAL_ENTITIES
    semantic = max(0.0, cosine(left.embedding, right.embedding))
    # Similar sentence templates about different named subjects are weak evidence.
    # Explicit manual links can bridge subjects when a caller knows the relationship.
    if left_entities and right_entities and not left_entities & right_entities:
        semantic *= 0.2
    entity = jaccard(left_entities, right_entities)
    tag = jaccard(set(left.tags), set(right.tags))
    return 0.60 * semantic + 0.30 * entity + 0.10 * tag


def personalized_pagerank(
    node_ids: list[str],
    edges: list[Edge],
    anchor: str,
    config: RetrievalConfig,
) -> tuple[dict[str, float], int, bool]:
    nodes = sorted(set(node_ids))
    if anchor not in nodes:
        raise ValidationError("The PageRank anchor must be in the graph.")
    adjacency: dict[str, dict[str, float]] = {node: {} for node in nodes}
    for edge in sorted(edges, key=lambda e: (e.a, e.b, e.kind)):
        if edge.a not in adjacency or edge.b not in adjacency or edge.a == edge.b:
            continue
        if not math.isfinite(edge.weight) or edge.weight <= 0:
            raise ValidationError("Graph weights must be finite and positive.")
        # Manual and automatic evidence for a pair do not count twice.
        weight = max(edge.weight, adjacency[edge.a].get(edge.b, 0))
        adjacency[edge.a][edge.b] = weight
        adjacency[edge.b][edge.a] = weight
    outgoing = {n: math.fsum(adjacency[n].values()) for n in nodes}
    probability = {node: float(node == anchor) for node in nodes}
    restart = config.restart_probability
    for iteration in range(1, config.pagerank_iterations + 1):
        updated = {node: (restart if node == anchor else 0.0) for node in nodes}
        for node in nodes:
            mass = (1 - restart) * probability[node]
            if outgoing[node] == 0:
                updated[anchor] += mass
            else:
                for neighbor in sorted(adjacency[node]):
                    updated[neighbor] += mass * adjacency[node][neighbor] / outgoing[node]
        residual = math.fsum(abs(updated[node] - probability[node]) for node in nodes)
        probability = updated
        if residual <= config.pagerank_tolerance:
            return probability, iteration, True
    return probability, config.pagerank_iterations, False
