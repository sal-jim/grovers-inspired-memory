# Synthetic sanity benchmark

Run `python benchmarks/evaluate.py --projects 40 --repeats 3` after installing
the project. No key, model download or external service is used. The source
contains the full synthetic dataset generator and fixed query clock.

A local Python 3.14.2/macOS x86_64 run of version 0.1.0 used 40 project clusters,
164 stored facts across two tenants, and 120 queries per strategy:

| Strategy | Anchor accuracy | Related precision@2 | Median latency | p95 latency |
| --- | --- | --- | --- | --- |
| vector | 1.00 | 1.00 | 77.65 ms | 78.22 ms |
| hybrid | 1.00 | 1.00 | 77.63 ms | 78.34 ms |
| grover | 1.00 | 1.00 | 77.72 ms | 78.16 ms |

Initial batch ingestion took 2.32 seconds. These are individual observations,
not stable performance promises; the tiny timing differences are noise. All
three strategies use the same related-fact graph. Grover/hybrid anchors were
identical, as required by the deterministic selection rule.

The benchmark deliberately uses exact-text anchor queries and explicit project
entities, so it is an easy lexical sanity test. It is not held-out conversational
evaluation, a test of semantic understanding, or a comparison against another
memory product. The early graph heuristic failed the related-fact test because
similar sentence templates linked different projects; entity disagreement now
dampens that similarity. The regression is also covered in the test suite.

For further evaluation, collect permissioned conversations with labeled relevant
facts, use paraphrased queries, measure false positives and misses, and compare
different graph/recency settings and embedding models. Report ingestion cost
alongside retrieval latency. Keep the same corpus, frozen clock and encoder for
baseline comparisons. Never publish private source conversations as fixtures.
