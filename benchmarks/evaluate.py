"""Reproducible synthetic sanity benchmark, not a real-world quality evaluation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import UTC, datetime, timedelta

from grovers_memory import FactInput, Memory


def evaluate(projects=40, repeats=3):
    now = datetime(2026, 10, 20, 12, tzinfo=UTC)
    cases = []
    facts = []
    for index in range(projects):
        name = f"Cedar{index:04d}"
        texts = [
            f"{name} builds an offline marine research notebook.",
            f"{name} stores coastal observations in SQLite.",
            f"{name} is tested during tide pool surveys.",
        ]
        facts.extend(
            FactInput(text, entities=[name], tags=[name], observed_at=now) for text in texts
        )
        facts.append(
            FactInput(f"Archive{index:04d} contains unrelated ceramic catalogues.", observed_at=now)
        )
        cases.append((texts[0], texts))
    with Memory("sqlite:///:memory:", clock=lambda: now) as memory:
        start = time.perf_counter()
        memory.add_many("synthetic", facts)
        ingestion_ms = (time.perf_counter() - start) * 1000
        memory.add("other-user", FactInput("Private unrelated memory."))
        memory.add(
            "synthetic",
            FactInput(
                "Expired marine research notebook.",
                observed_at=now - timedelta(days=2),
                expires_at=now,
            ),
        )
        memory.add(
            "synthetic",
            FactInput(
                "Obsolete archive location.",
                key="archive.city",
                observed_at=now - timedelta(days=2),
            ),
        )
        memory.add(
            "synthetic",
            FactInput(
                "Current archive location.", key="archive.city", observed_at=now - timedelta(days=1)
            ),
        )
        report = {}
        anchors = {}
        for strategy in ("vector", "hybrid", "grover"):
            timings, correct, related, outputs = [], 0, 0, []
            for _ in range(repeats):
                for query, expected in cases:
                    start = time.perf_counter()
                    result = memory.retrieve("synthetic", query, strategy=strategy, as_of=now)
                    timings.append((time.perf_counter() - start) * 1000)
                    correct += bool(result and result[0] == expected[0])
                    related += len(set(result[1:]) & set(expected[1:]))
                    outputs.append(result[0] if result else None)
                    assert "Private unrelated memory." not in result
                    assert "Expired marine research notebook." not in result
                    assert "Obsolete archive location." not in result
            count = projects * repeats
            report[strategy] = {
                "anchor_accuracy": correct / count,
                "related_precision_at_2": related / (count * 2),
                "median_ms": statistics.median(timings),
                "p95_ms": sorted(timings)[math.ceil(0.95 * len(timings)) - 1],
            }
            anchors[strategy] = outputs
        same = anchors["grover"] == anchors["hybrid"]
        assert same, "The classical experiment must retain the hybrid maximum."
        return {
            "dataset": "synthetic lexical project clusters; exact-text anchor queries",
            "projects": projects,
            "facts": len(facts) + 4,
            "queries_per_strategy": projects * repeats,
            "ingestion_ms": ingestion_ms,
            "grover_hybrid_anchors_identical": same,
            "strategies": report,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.projects <= 200 or not 1 <= args.repeats <= 100:
        parser.error("Use 1..200 projects and 1..100 repeats.")
    print(json.dumps(evaluate(args.projects, args.repeats), indent=2))
