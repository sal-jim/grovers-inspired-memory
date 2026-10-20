# grovers-inspired-memory

[![CI](https://github.com/sal-jim/grovers-inspired-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/sal-jim/grovers-inspired-memory/actions/workflows/ci.yml)

An experimental fact memory layer for AI applications. Retrieve a relevant fact
**x**, then related facts **y** and **z** from an internal relationship graph.
The consuming AI receives only a string array:

```json
[
  "Maya is building Project Atlas, an offline field-research notebook.",
  "Maya tests Project Atlas with marine biologists on coastal surveys.",
  "Project Atlas uses SQLite to keep field notes available offline."
]
```

SQL stores dated fact revisions. A deterministic relevance scorer selects the
anchor. Personalized PageRank finds its supporting context. AI is optional and
used only to extract facts from source text; retrieval needs no LLM or API key.

**Classical hardware does not gain Grover's quantum speedup by simulating it.**
The Grover-inspired option simulates phase inversion and diffusion on an already
scored shortlist. Its deterministic anchor matches the hybrid baseline by
construction. This is a transparent research experiment, with working baselines
and tests, rather than a claim of improved complexity or real-world accuracy.

MIT licensed. Version 0.1 is experimental; APIs and schemas may change.

## Run it

Python 3.11+. The default uses SQLite and local feature hashing, with **zero runtime
dependencies**, no model download and no network calls. Install from source; this
project has not been published on PyPI.

```sh
git clone https://github.com/sal-jim/grovers-inspired-memory.git
cd grovers-inspired-memory
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
gim init
gim import --user maya examples/facts.json
gim query --user maya --as-of 2026-10-20T12:00:00Z "What is Maya building with Project Atlas?"
```

The sample contains fictional facts. `python examples/demo.py` runs the same
example entirely in memory. `.env`, database files and the virtual environment
are ignored by Git.

The contract is **zero to three distinct strings**: `[x, y, z]` when two related
facts exist, a shorter array for sparse memory, and `[]` when no candidate clears
the relevance threshold. It never pads with unrelated facts or invented text.
An invalid or unsearchable query (for example, only stop words) raises a validation
error rather than producing a fabricated answer.

## Python API

```python
from grovers_memory import FactInput, Memory

with Memory("sqlite:///memory.db") as memory:
    memory.add(
        "maya",
        FactInput(
            "Maya prefers concise technical explanations.",
            key="profile.explanation_style",
            entities=["Maya"],
            tags=["communication"],
            observed_at="2026-10-20T09:00:00Z",
        ),
    )
    context: list[str] = memory.retrieve("maya", "How should I explain this to Maya?")
    # Pass context to your own application/model as reference data.
```

`Memory` is a synchronous library, not a chat model, plugin, hosted service or
public API server. It does not automatically intercept conversations. Your app
decides what to ingest, supplies the authenticated user's ID, and injects the
returned strings into its model context. Graph edges, ranks and revision metadata
are never included by `retrieve()`. Explicit developer diagnostics are available
through `explain()` / `gim explain`; do not forward them to the consuming AI.

Other operations:

```python
memory.add_many(user_id, facts)  # atomic list[FactInput] batch
memory.history(user_id, key="profile.city")
memory.link(user_id, fact_a.id, fact_b.id, weight=0.8)
memory.expire(user_id, fact_a.id, at="2026-11-01T00:00:00Z")
memory.forget(user_id, fact_a.id)  # physical row + edge deletion
memory.forget(user_id)  # all of this user's facts
```

These snippets assume an open `Memory` instance. Use the context manager or call
`close()` to release the connection. Returned `Fact` objects expose timestamps,
source/evidence and IDs for application administration; `Fact.to_dict()` omits
the embedding. SQL table names are prefixed `gim_`.

## How retrieval works

```mermaid
flowchart LR
    S[Source text] --> E[Optional AI fact extraction]
    E --> V[Schema and quote validation]
    M[Manual facts] --> DB[(Temporal SQL facts)]
    V --> DB
    DB --> G[Deterministic relationship graph]
    Q[User query] --> C[Eligible vector shortlist]
    DB --> C
    C --> R[Hybrid scores / Grover-inspired experiment]
    R --> X[Anchor x]
    X --> P[Personalized PageRank]
    G --> P
    P --> A[Array of up to three strings]
    A --> AI[Your AI application]
```

1. Freeze `as_of` once. Filter by **user**, observation time, expiry and replacement
   interval **before** candidate ranking or graph traversal.
2. Embed the query locally. SQLite computes exact cosine over eligible facts;
   PostgreSQL uses pgvector's exact cosine distance (`<=>`). Keep at most 128
   candidates by default, with stable ID tie-breaking.
3. Score candidates. `vector` uses cosine alone. `hybrid` combines
   `relevance = .75 × max(cosine, 0) + .25 × BM25/(1 + BM25)`.
   BM25 uses `k1=1.2`, `b=.75`, with statistics from the eligible shortlist.
   The default minimum relevance is `.12` (a heuristic, **not** a probability).
4. Hybrid relevance is multiplied by
   `.80 + .10 × confidence + .05 × importance + .05 × freshness`, where
   `freshness = 2^(-age_in_days / 90)`. Relevance dominates; being new alone
   cannot make an unrelated fact relevant. Expiry is a hard filter.
5. For `grover`, mark scores at least `.90 × max_score` and initialize uniform
   amplitudes. Repeat marked phase inversion and inversion about the mean for
   `floor(π / (4 asin(sqrt(M/N))) - .5)` rounds, clamped at zero.
   The oracle requires all candidate scores first. Marked amplitudes are equal
   analytically; choose the highest-ranked marked fact deterministically.
   No quantum computer, random measurement, or hidden LLM reranker is involved.
6. Expand a bounded, eligible graph around **x**: up to 128 nodes and three hops.
   Personalized PageRank uses restart probability `.20`, power iteration with
   an L1 tolerance of `1e-12`, and a 200-iteration cap. Dangling mass returns to
   x. Pick the top two other reachable facts, skipping duplicate text.

The result is the best anchor **within the shortlist according to these scores**.
The neighbors are most related according to the bounded graph's PageRank, which
can include indirect relationships. Neither is a guarantee of human relevance
or of the two strongest individual edge weights. `explain()` reports candidates,
anchor scores, amplification probabilities, selected graph ranks and convergence.

### Relationship graph

On insertion, compare facts whose validity intervals overlap for the same user.
Connect up to eight strongest positive relationships above `.18`:

```text
weight = .60 × max(cosine, 0) + .30 × entity_Jaccard + .10 × tag_Jaccard
```

When both facts have named entities but share none, multiply the cosine term by
`.20`. This reduces links between similar sentence templates about different
subjects. Ignore generic entities such as “user” and “assistant”. Entities/tags
are normalized and sorted. Edges are undirected; manual and automatic evidence
for the same pair use their maximum weight during PageRank, rather than counting
twice. Manual links can connect subjects that the heuristic misses.

This is a lightweight relationship heuristic, **not a learned knowledge graph or
causal relation extractor**. Useful, specific entities matter. Shared people or
generic wording can still create noisy hubs. Edge construction is incremental,
so insertion order and graph settings can affect persisted edges. Changing
ingestion settings does not retroactively rebuild them; use a fresh database
when comparing configurations. Historical graph queries filter current stored
edges by fact validity; there is no separate version history for manual links.

### What “deterministic” means

For the same persisted data, encoder, configuration, query and `as_of`, retrieval
uses stable ordering and no random sampling. Freeze `as_of` for reproducible
experiments. LLM extraction remains probabilistic even with temperature zero.
Different model/runtime versions and floating point implementations can change
near ties. pgvector stores single-precision vectors, so cross-backend rankings
can differ near ties. Exact vector search is the default; no HNSW/IVFFlat index is
created. Adding approximate indexes changes recall/reproducibility assumptions.

SQLite shortlist search costs `O(N × d)`; exact PostgreSQL search also scans
eligible vectors. Ingestion compares existing user facts and can cost `O(N × d)`
per insertion. Classical amplification adds `O(K × rounds)` work **after**
scoring. Graph traversal and PageRank are bounded, but this release is a small
research implementation, not a demonstrated large-scale retrieval engine.

## Facts, replacement and expiry

| Field | Meaning |
| --- | --- |
| `text` | Atomic, self-contained fact; 1–4096 characters |
| `key` | Optional, user-scoped **single-valued slot** such as `profile.city` |
| `observed_at` | Start of validity; defaults to the ingestion clock |
| `created_at` | When this database recorded the revision; set internally |
| `expires_at` | Optional exclusive end; explicit, no default TTL |
| `superseded_at`, `superseded_by` | Replacement boundary and successor ID; internal |
| `entities`, `tags` | Specific subjects/topics used to infer relationships |
| `confidence`, `importance` | Validated 0–1 hints; do not establish truth |
| `source_id`, `evidence` | Optional provenance; extracted facts require a verbatim quote |

All timestamps must include a timezone and are normalized to UTC. Validity is
`[observed_at, min(expires_at, superseded_at))`; future observations stay hidden.
These are corrected valid-time views, not a full bitemporal “what the database
knew then” audit: a late observation can change a historical answer.

Store a new text with the **same key and a later observation time** to replace a
fact. Different users' keys never collide. Late observations fit chronologically
between revisions. Different payloads for the same key and exact timestamp raise
`ConflictError`; identical payloads at the same timestamp are idempotent. Repeated
source IDs alone do not deduplicate later observations. Batches either commit
completely or roll back. SQLite uses write transactions; PostgreSQL serializes
user writes with transaction-scoped advisory locks.

Use narrowly scoped keys: `project.atlas.database`, not `project`, and separate
people's attributes. Unkeyed facts accumulate; there is no automatic semantic
contradiction resolver. The caller or extractor must choose the right slot.
`expire()` only shortens validity; add a revision to extend it. Deleting a
successor keeps the previous revision closed, including during late insertion.
`history` includes expired/replaced facts until explicitly forgotten.

## Optional AI extraction and API key

Put your own key in the local **`.env`** copied from `.env.example`:

```dotenv
GIM_LLM_API_KEY=your-private-key
GIM_LLM_MODEL=your-provider/model-that-supports-json-schema
GIM_LLM_BASE_URL=https://openrouter.ai/api/v1
```

Then ingest a UTF-8 text file:

```sh
gim ingest --user maya --file conversation.txt --source-id chat-2026-10-20
```

The adapter uses OpenRouter's chat-completions API or a compatible endpoint with
strict JSON Schema output. Pick a model/provider that supports it; unsupported
models fail rather than silently switching output modes. A real environment
variable overrides `.env`. The small dotenv reader treats values as data, with no
shell expansion. Keys are neither persisted in facts nor printed by the CLI.
The extractor object and settings repr omit credentials. HTTPS is required,
except for loopback HTTP for local compatible servers; redirects are rejected.

Ingest makes one request containing only the submitted source, its observation
time and extraction instructions. It does **not** send stored history or user IDs.
The configured provider receives this text and may charge for processing. Retrieval
uses neither that key nor that provider. Requests have a timeout and size limits;
responses are capped at 256 KiB, sources at 32,000 characters, and extractions at
16 facts. Truncated responses, unknown fields, bad dates and ungrounded quotes
reject the whole batch. Quotes establish source provenance, **not logical
entailment or factual truth**. Source text and extracted facts remain untrusted.

For manual import, supply a JSON array of `FactInput` objects (see
[`examples/facts.json`](examples/facts.json)); a batch supports up to 1,000 items.
The tenant comes from `--user`, never from imported/model-generated JSON.

## Optional semantic embeddings

The default `HashEmbedder` is deterministic word/bigram/character feature
hashing. **It is lexical, not semantic AI embedding**: paraphrases without shared
terms can be missed, hash collisions can create false matches, and English stop
words are used. Changing SQL backends does not fix this.

For a local semantic model:

```sh
python -m pip install -e '.[semantic]'
```

```dotenv
GIM_EMBEDDER=sentence-transformers
GIM_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GIM_EMBEDDING_REVISION=<full-40-character-model-commit>
GIM_DATABASE_URL=sqlite:///semantic-memory.db
```

Choose a verified full commit from your model repository. No moving revision is
accepted. The first model load downloads weights; subsequent query embedding is
local on CPU, with remote model code disabled. Install this optional extra on a
Python/platform version supported by PyTorch. The adapter is unit-tested with a
mock; downloading and assessing a real model is outside the default test suite.
You can also implement the `Embedder` protocol (`name`, `dimensions`, `embed`).

Database metadata pins the encoder identity/dimensions; opening a database with a
different encoder fails. Switching encoders requires a new database and
re-ingestion. Pin dependency versions as well for strict experimental comparisons.

## PostgreSQL + pgvector

SQLite is enough to try the complete flow. For pgvector, install the PostgreSQL
extra and run a disposable local database (requires Docker):

```sh
python -m pip install -e '.[postgres]'
docker compose up -d --wait
```

```dotenv
GIM_DATABASE_URL=postgresql://gim:local-development-only@localhost:55432/gim
```

```sh
gim init
gim import --user maya examples/facts.json
gim query --user maya --strategy vector "Project Atlas offline notebook"
```

The development container binds to loopback and keeps data in a named volume.
Use your own credentials/access controls elsewhere. Initialization creates the
extension if missing and creates the tables; the role must have those permissions,
or an administrator must install `vector` first. Queries use parameterized SQL,
same-user composite foreign keys for edges and repeatable-read retrieval snapshots.
The library is the tenant boundary; direct SQL access is **not** protected by RLS.
No production connection pooling or schema migration framework is included.

## CLI reference

Global flags (`--db`, `--env-file`) go **before** the subcommand. Run `gim --help`
or `gim <command> --help` for all flags.

| Command | Purpose |
| --- | --- |
| `init` | Create/check SQL schema and encoder identity |
| `add --user ID --text TEXT` | Store an already-written fact; optional `--key`, `--entity`, `--tag`, dates |
| `import --user ID facts.json` | Atomically insert a JSON array |
| `ingest --user ID --file source.txt` | Optional AI fact extraction |
| `query --user ID "query"` | Only the string array; `--as-of`, `--strategy grover\|hybrid\|vector` |
| `explain --user ID "query"` | Developer ranking and graph diagnostics |
| `history --user ID` | Export revisions; optionally `--key` |
| `link --user ID A B --weight .8` | Explicit relationship between fact IDs |
| `expire --user ID FACT_ID --at TIMESTAMP` | Shorten validity |
| `forget --user ID --id FACT_ID` | Delete one fact and its edges |
| `forget --user ID --all` | Delete this user's facts and edges |

CLI output is JSON, errors go to stderr with a nonzero exit code. Configuration
and validation errors use exit code 2; other database failures use 1.

## Tests and experiments

```sh
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest
python benchmarks/evaluate.py --projects 40 --repeats 3
python -m build
python -m twine check dist/*
```

Without PostgreSQL, the pgvector cases are explicitly skipped. To run the same
behavior and concurrency tests against real pgvector:

```sh
python -m pip install -e '.[dev,postgres]'
GIM_TEST_POSTGRES_URL=postgresql://gim:local-development-only@localhost:55432/gim pytest
```

Use a **dedicated test database**: tests create tables and temporary fictional
tenants, then delete those tenants. CI exercises Python 3.11–3.14 and a separate
PostgreSQL 17/pgvector service. Tests cover revision ordering, expiry boundaries,
deletion without resurrection, rollback, cross-connection races, user isolation,
graph bounds, PageRank, amplification, extraction rejection and the CLI. There
are no paid/live LLM calls in the test suite.

The benchmark compares `vector`, `hybrid` and `grover` using the same graph, reports
anchor accuracy, related precision@2 and median/p95 latency, and checks expiry,
replacement and tenant boundaries. It uses synthetic project clusters and
**exact-text anchor queries**, so high scores are sanity checks, not evidence of
paraphrase understanding or superiority to existing memory systems. Timings depend
on hardware and dataset size. Grover/hybrid anchor equality is asserted. Measure
on real held-out conversations before making product or research claims.

## Boundaries and next experiments

This implements the full experimental storage/extraction/retrieval loop; it is
not a production memory service. Authenticate users in the calling application.
Do not accept arbitrary tenant IDs from untrusted clients. No encryption at rest,
prompt-injection immunity, contradiction adjudication or learned relationship
quality is claimed. Read [`SECURITY.md`](SECURITY.md) before integrating private data.

Useful follow-ups are a labeled conversational benchmark, an embedding-model
comparison, recency/graph ablations, explicit relation types, efficient graph
rebuilding, ANN recall measurements, and conflict-resolution review workflows.
A real quantum search implementation would need an efficient oracle/data-loading
model and a separate complexity analysis; the present simulator supplies neither.

## Research and implementation references

- [IBM: Grover's algorithm](https://quantum.cloud.ibm.com/learning/en/modules/computer-science/grovers) — oracle, phase inversion and diffusion.
- [NetworkX: personalized PageRank](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html) — reference semantics for personalization and dangling nodes; this project implements power iteration directly.
- [pgvector documentation](https://github.com/pgvector/pgvector) — cosine distance and exact versus approximate search.
- [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs) — JSON Schema extraction interface.
- [Sentence Transformers API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html) — optional pinned local encoder.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development. Contributions and critical
evaluations of the idea are welcome. [MIT license](LICENSE).
