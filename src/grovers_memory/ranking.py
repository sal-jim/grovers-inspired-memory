"""Classical relevance scoring and an explicitly classical Grover experiment."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from .embeddings import cosine, tokens
from .models import Fact, RetrievalConfig, ValidationError


@dataclass(frozen=True, slots=True)
class RankedFact:
    fact: Fact
    score: float
    relevance: float
    vector_score: float
    lexical_score: float
    freshness: float


def rank(
    facts: list[Fact],
    query: str,
    embedding: tuple[float, ...],
    at: datetime,
    config: RetrievalConfig,
    strategy: str,
) -> list[RankedFact]:
    if strategy not in {"grover", "hybrid", "vector"}:
        raise ValidationError("strategy must be grover, hybrid, or vector.")
    terms = set(tokens(query))
    documents = [Counter(tokens(f.text)) for f in facts]
    average_length = sum(map(lambda d: sum(d.values()), documents)) / max(1, len(documents))
    idf = {
        term: math.log(
            1
            + (len(documents) - sum(term in d for d in documents) + 0.5)
            / (sum(term in d for d in documents) + 0.5)
        )
        for term in sorted(terms)
    }
    results = []
    for fact, document in zip(facts, documents, strict=True):
        bm25 = 0.0
        length = sum(document.values())
        for term in sorted(terms):
            count = document[term]
            if count:
                bm25 += (
                    idf[term]
                    * count
                    * 2.2
                    / (count + 1.2 * (0.25 + 0.75 * length / max(1, average_length)))
                )
        lexical = bm25 / (1 + bm25)
        vector = max(0.0, cosine(embedding, fact.embedding))
        relevance = vector if strategy == "vector" else 0.75 * vector + 0.25 * lexical
        age_days = max(0.0, (at - fact.observed_at).total_seconds() / 86400)
        freshness = 2 ** (-age_days / config.half_life_days)
        quality = 0.80 + 0.10 * fact.confidence + 0.05 * fact.importance + 0.05 * freshness
        score = relevance if strategy == "vector" else relevance * quality
        if relevance >= config.min_relevance:
            results.append(RankedFact(fact, score, relevance, vector, lexical, freshness))
    return sorted(results, key=lambda r: (-r.score, -r.fact.observed_at.timestamp(), r.fact.id))


def amplify(scores: list[float], ratio: float = 0.90) -> dict:
    """Simulate phase inversion + diffusion. This costs O(N * rounds), not O(sqrt(N)).

    The oracle is constructed from scores already computed classically. Marking an
    elite set cannot improve on the known maximum. We do not sample a random result.
    """
    if not 0 <= ratio <= 1 or any(not math.isfinite(s) or s < 0 for s in scores):
        raise ValidationError(
            "Amplification needs finite nonnegative scores and a ratio in [0, 1]."
        )
    n = len(scores)
    if not n or max(scores) == 0:
        return {"rounds": 0, "marked": [], "probabilities": [0.0] * n, "marked_mass": 0.0}
    maximum = max(scores)
    marked = [index for index, score in enumerate(scores) if score >= ratio * maximum]
    theta = math.asin(math.sqrt(len(marked) / n))
    # libm implementations can put an exact integer a few ulps below its value
    # (notably M=1, N=4). Snap only near-integer numerical roundoff before floor.
    ideal = math.pi / (4 * theta) - 0.5
    nearest = round(ideal)
    if math.isclose(ideal, nearest, rel_tol=0, abs_tol=1e-12):
        ideal = float(nearest)
    rounds = max(0, math.floor(ideal))
    amplitudes = [1 / math.sqrt(n)] * n
    for _ in range(rounds):
        for index in marked:
            amplitudes[index] = -amplitudes[index]
        mean = math.fsum(amplitudes) / n
        amplitudes = [2 * mean - value for value in amplitudes]
    probabilities = [value * value for value in amplitudes]
    return {
        "rounds": rounds,
        "marked": marked,
        "probabilities": probabilities,
        "marked_mass": math.fsum(probabilities[i] for i in marked),
    }
