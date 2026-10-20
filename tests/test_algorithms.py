import math

import pytest

from grovers_memory import HashEmbedder, RetrievalConfig, ValidationError
from grovers_memory.embeddings import cosine, normalize
from grovers_memory.graph import personalized_pagerank
from grovers_memory.models import Edge
from grovers_memory.ranking import amplify


def test_amplification_known_case():
    result = amplify([1.0, 0.1, 0.1, 0.1])
    assert result["rounds"] == 1
    assert result["marked"] == [0]
    assert result["probabilities"] == pytest.approx([1, 0, 0, 0])
    assert result["marked_mass"] == pytest.approx(1)


@pytest.mark.parametrize("scores", [[], [0], [1], [1, 1], [1] + [0.01] * 127])
def test_amplification_normalized(scores):
    result = amplify(scores)
    if scores and max(scores):
        assert sum(result["probabilities"]) == pytest.approx(1)
        assert result["marked_mass"] >= len(result["marked"]) / len(scores) - 1e-12
    else:
        assert not result["marked"]


def test_pagerank_stationary_distribution_and_order():
    cfg = RetrievalConfig()
    ranks, _, converged = personalized_pagerank(
        ["b", "a", "isolated"], [Edge("a", "b", 1)], "a", cfg
    )
    assert converged
    assert ranks["a"] == pytest.approx(1 / 1.8)
    assert ranks["b"] == pytest.approx(0.8 / 1.8)
    assert ranks["isolated"] == 0
    assert sum(ranks.values()) == pytest.approx(1)
    again = personalized_pagerank(["isolated", "a", "b"], [Edge("b", "a", 1)], "a", cfg)
    assert again[0] == ranks


def test_pagerank_dangling_and_nonconvergence():
    assert personalized_pagerank(["a"], [], "a", RetrievalConfig())[0] == {"a": 1}
    assert not personalized_pagerank(
        ["a", "b"], [Edge("a", "b", 1)], "a", RetrievalConfig(pagerank_iterations=1)
    )[2]


def test_hash_is_reproducible_normalized_lexical():
    a = HashEmbedder().embed("Offline field notebooks")
    assert a == HashEmbedder().embed("Offline field notebooks")
    assert math.hypot(*a) == pytest.approx(1)
    assert cosine(a, HashEmbedder().embed("Offline field notes")) > 0.4
    assert normalize([1e308, 1e308], 2) == pytest.approx([2**-0.5, 2**-0.5])


@pytest.mark.parametrize("vector", [[0, 0], [math.inf, 1], [math.nan, 1], [True, 1], [1]])
def test_invalid_vectors(vector):
    with pytest.raises(ValidationError):
        normalize(vector, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"graph_nodes": 0},
        {"graph_hops": -1},
        {"restart_probability": 1},
        {"min_relevance": math.nan},
        {"half_life_days": True},
        {"candidate_limit": True},
        {"pagerank_tolerance": 0},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValidationError):
        RetrievalConfig(**kwargs)
