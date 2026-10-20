# Security boundaries

This is a local library and CLI, not an authenticated hosted service. The caller
must supply a trusted user ID and enforce authorization. The library scopes its
queries and edge constraints by user, but a database administrator/direct SQL
client can read all rows. PostgreSQL row-level security is not configured.

Facts, provenance, embeddings, diagnostics and history may contain private data.
Keep databases and `.env` outside public directories and version control. Use
your application's normal encryption, backups and access controls. Physical row
deletion is not secure erasure of SQLite WAL pages, disk sectors or backups.
Deleting a revision leaves the predecessor's closing timestamp/opaque successor
ID to prevent resurrection, but no deleted fact text or vector is retained in
the tables. Deleting all of a user's facts removes their edges too.

Only explicit `ingest` calls send source text to the configured AI provider. No
telemetry is implemented. The optional semantic encoder may download model files
at construction; query embedding then runs locally. Review your provider/model's
terms and runtime dependencies before sending private data or loading weights.

Structured output and evidence validation limit malformed input. They do not
prove a fact is true or neutralize prompt injection. Treat returned strings as
untrusted reference data and keep them separate from privileged instructions.
An LLM may choose an incorrect key and replace a legitimate fact. Applications
that need approval should review extracted FactInput values before `add_many`.

Provider redirects are refused; error bodies and credentials are not printed.
API base URLs and custom embedder implementations are trusted administrator
configuration, not parameters to accept from end users. Use least-privilege
database credentials and protect local compatible model endpoints.

For a potential vulnerability, use GitHub's private vulnerability reporting
option if enabled for this repository. Otherwise contact the maintainer through
their GitHub profile to arrange a private channel. Do not post keys or private
memory content in a public issue.
