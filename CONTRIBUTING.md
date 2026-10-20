# Contributing

Start with the README quick start, then install `.[dev]`. Run `ruff check .`,
`ruff format --check .` and `pytest` before submitting a pull request. Use a
dedicated PostgreSQL database and `GIM_TEST_POSTGRES_URL` for integration tests.
Never commit `.env`, API keys, personal conversations, or real memory databases.

Changes to ranking need an example of the failure they address and an evaluation
against both hybrid and vector baselines. Keep fictional fixtures reproducible.
Do not describe a classical simulation as a quantum speedup. Report dataset,
encoder, configuration, frozen query time and environment alongside benchmarks.

Preserve the `list[str]` retrieval boundary, user isolation, parameterized SQL,
atomic ingestion and UTC interval semantics. Add regression tests for behavioral
changes. Schemas have a version marker; incompatible storage changes must reject
older databases or include a deliberate migration plan rather than silently
reinterpreting stored data. This project has no production migrations yet.

Please keep pull requests focused and explain the resulting behavior and tests.
Contributions are distributed under this repository's MIT license.
