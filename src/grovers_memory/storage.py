"""Parameterized SQL storage. SQLite and PostgreSQL share temporal semantics."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

from .embeddings import cosine
from .models import ConfigurationError, Edge, Fact, iso, timestamp


class Store:
    def __init__(self, url: str, encoder: str, dimensions: int):
        self.postgres = url.startswith(("postgresql://", "postgres://"))
        self.dimensions = dimensions
        self._lock = threading.RLock()
        if self.postgres:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:
                raise ConfigurationError(
                    "Install grovers-inspired-memory[postgres] first."
                ) from exc
            self.conn = psycopg.connect(url, autocommit=True, row_factory=dict_row)
        elif url.startswith("sqlite:///"):
            import sqlite3

            path = url.removeprefix("sqlite:///")
            if not path:
                raise ConfigurationError("Provide a SQLite database path.")
            if path != ":memory:":
                Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
                path = str(Path(path).expanduser())
            self.conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA busy_timeout = 10000")
            # Two processes can race to enable WAL on a brand-new database. SQLite's
            # busy timeout does not reliably wait for this particular PRAGMA.
            deadline = time.monotonic() + 10
            while True:
                try:
                    self.conn.execute("PRAGMA journal_mode = WAL")
                    break
                except sqlite3.OperationalError as exc:
                    if exc.sqlite_errorcode not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                        self.conn.close()
                        raise
                    if time.monotonic() >= deadline:
                        self.conn.close()
                        raise
                    time.sleep(0.02)
        else:
            raise ConfigurationError("Use sqlite:///path or postgresql://user:password@host/db.")
        try:
            self._initialize(encoder)
        except Exception:
            self.conn.close()
            raise

    def execute(self, sql, values=()):
        return self.conn.execute(sql.replace("?", "%s") if self.postgres else sql, values)

    @contextmanager
    def transaction(self, *, write=False, user_id=None):
        with self._lock:
            if self.postgres:
                with self.conn.transaction():
                    if write:
                        # Serialize revision chains across connections, including missing rows.
                        self.execute(
                            "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
                            ("gim:" + (user_id or "schema"),),
                        )
                    else:
                        self.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                    yield
            else:
                self.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                try:
                    yield
                    self.conn.commit()
                except BaseException:
                    self.conn.rollback()
                    raise

    def _initialize(self, encoder):
        with self.transaction(write=True):
            if self.postgres:
                self.execute("CREATE EXTENSION IF NOT EXISTS vector")
            vector_type = f"vector({self.dimensions})" if self.postgres else "TEXT"
            date_type = "TIMESTAMPTZ" if self.postgres else "TEXT"
            statements = [
                "CREATE TABLE IF NOT EXISTS gim_meta (name TEXT PRIMARY KEY, value TEXT NOT NULL)",
                f"""CREATE TABLE IF NOT EXISTS gim_facts (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, text TEXT NOT NULL,
                    fact_key TEXT, entities TEXT NOT NULL, tags TEXT NOT NULL,
                    created_at {date_type} NOT NULL, observed_at {date_type} NOT NULL,
                    expires_at {date_type}, superseded_at {date_type}, superseded_by TEXT,
                    confidence DOUBLE PRECISION NOT NULL CHECK(confidence BETWEEN 0 AND 1),
                    importance DOUBLE PRECISION NOT NULL CHECK(importance BETWEEN 0 AND 1),
                    source_id TEXT, evidence TEXT, embedding {vector_type} NOT NULL,
                    UNIQUE(user_id, id), UNIQUE(user_id, fact_key, observed_at),
                    CHECK(expires_at IS NULL OR expires_at > observed_at),
                    CHECK(superseded_at IS NULL OR superseded_at > observed_at)
                )""",
                f"""CREATE TABLE IF NOT EXISTS gim_edges (
                    user_id TEXT NOT NULL, a TEXT NOT NULL, b TEXT NOT NULL,
                    weight DOUBLE PRECISION NOT NULL CHECK(weight > 0 AND weight <= 1),
                    kind TEXT NOT NULL CHECK(kind IN ('automatic', 'manual')),
                    created_at {date_type} NOT NULL, CHECK(a < b),
                    PRIMARY KEY(user_id, a, b, kind),
                    FOREIGN KEY(user_id, a) REFERENCES gim_facts(user_id, id) ON DELETE CASCADE,
                    FOREIGN KEY(user_id, b) REFERENCES gim_facts(user_id, id) ON DELETE CASCADE
                )""",
                "CREATE INDEX IF NOT EXISTS gim_active ON gim_facts(user_id, observed_at)",
                "CREATE INDEX IF NOT EXISTS gim_edge_b ON gim_edges(user_id, b)",
            ]
            for statement in statements:
                self.execute(statement)
            expected = {"schema": "1", "encoder": encoder, "dimensions": str(self.dimensions)}
            for key, value in expected.items():
                self.execute(
                    "INSERT INTO gim_meta(name, value) VALUES (?, ?) ON CONFLICT(name) DO NOTHING",
                    (key, value),
                )
                saved = self.execute("SELECT value FROM gim_meta WHERE name = ?", (key,)).fetchone()
                if saved["value"] != value:
                    raise ConfigurationError(
                        f"Database {key} differs. Use the original encoder or a new database."
                    )

    def _columns(self, prefix=""):
        names = ["fact_key" if f.name == "key" else f.name for f in fields(Fact)]
        return ", ".join(
            f"{prefix}{name}::text AS embedding"
            if name == "embedding" and self.postgres
            else f"{prefix}{name}"
            for name in names
        )

    def _fact(self, row):
        if row is None:
            return None
        data = dict(row)
        data["key"] = data.pop("fact_key")
        for name in ("entities", "tags", "embedding"):
            data[name] = tuple(json.loads(data[name]))
        for name in ("created_at", "observed_at", "expires_at", "superseded_at"):
            if data[name] is not None:
                data[name] = timestamp(data[name])
        return Fact(**data)

    def get(self, user_id, fact_id):
        return self._fact(
            self.execute(
                f"SELECT {self._columns()} FROM gim_facts WHERE user_id = ? AND id = ?",
                (user_id, fact_id),
            ).fetchone()
        )

    def history(self, user_id, key=None):
        clause, values = (" AND fact_key = ?", (user_id, key)) if key else ("", (user_id,))
        rows = self.execute(
            f"SELECT {self._columns()} FROM gim_facts WHERE user_id = ?{clause} "
            "ORDER BY observed_at, id",
            values,
        ).fetchall()
        return [self._fact(row) for row in rows]

    @staticmethod
    def _active(prefix=""):
        return (
            f"{prefix}observed_at <= ? AND ({prefix}expires_at IS NULL OR {prefix}expires_at > ?) "
            f"AND ({prefix}superseded_at IS NULL OR {prefix}superseded_at > ?)"
        )

    def active(self, user_id, at):
        at = iso(at)
        return [
            self._fact(row)
            for row in self.execute(
                f"SELECT {self._columns()} FROM gim_facts WHERE user_id = ? AND {self._active()} "
                "ORDER BY id",
                (user_id, at, at, at),
            ).fetchall()
        ]

    def candidates(self, user_id, at, embedding, limit):
        if not self.postgres:
            return sorted(
                self.active(user_id, at), key=lambda f: (-cosine(embedding, f.embedding), f.id)
            )[:limit]
        at = iso(at)
        rows = self.execute(
            f"SELECT {self._columns()} FROM gim_facts WHERE user_id = ? AND {self._active()} "
            "ORDER BY embedding <=> ?::vector, id LIMIT ?",
            (user_id, at, at, at, json.dumps(embedding), limit),
        ).fetchall()
        return [self._fact(row) for row in rows]

    def insert(self, fact):
        data = []
        for field in fields(Fact):
            value = getattr(fact, field.name)
            if field.name in {"entities", "tags", "embedding"}:
                value = json.dumps(value)
            elif field.name.endswith("_at") and value is not None:
                value = iso(value)
            data.append(value)
        columns = ", ".join("fact_key" if f.name == "key" else f.name for f in fields(Fact))
        placeholders = ["?"] * len(data)
        if self.postgres:
            placeholders[-1] = "?::vector"
        self.execute(
            f"INSERT INTO gim_facts ({columns}) VALUES ({', '.join(placeholders)})", tuple(data)
        )

    def supersede(self, user_id, fact_id, at, successor):
        self.execute(
            "UPDATE gim_facts SET superseded_at = ?, superseded_by = ? "
            "WHERE user_id = ? AND id = ?",
            (iso(at), successor, user_id, fact_id),
        )

    def add_edge(self, user_id, a, b, weight, kind, at):
        a, b = sorted((a, b))
        self.execute(
            "INSERT INTO gim_edges(user_id, a, b, weight, kind, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, a, b, kind) DO UPDATE SET weight = excluded.weight",
            (user_id, a, b, weight, kind, iso(at)),
        )

    def edges(self, user_id, frontier, at, limit):
        if not frontier:
            return []
        slots = ", ".join("?" for _ in frontier)
        at = iso(at)
        rows = self.execute(
            "SELECT e.a, e.b, e.weight, e.kind FROM gim_edges e "
            "JOIN gim_facts fa ON fa.user_id = e.user_id AND fa.id = e.a "
            "JOIN gim_facts fb ON fb.user_id = e.user_id AND fb.id = e.b "
            f"WHERE e.user_id = ? AND (e.a IN ({slots}) OR e.b IN ({slots})) "
            f"AND {self._active('fa.')} AND {self._active('fb.')} "
            "ORDER BY e.weight DESC, e.a, e.b, e.kind LIMIT ?",
            (user_id, *frontier, *frontier, at, at, at, at, at, at, limit),
        ).fetchall()
        return [Edge(**dict(row)) for row in rows]

    def expire(self, user_id, fact_id, at):
        self.execute(
            "UPDATE gim_facts SET expires_at = ? WHERE user_id = ? AND id = ?",
            (iso(at), user_id, fact_id),
        )

    def delete(self, user_id, fact_id=None):
        if fact_id is None:
            result = self.execute("DELETE FROM gim_facts WHERE user_id = ?", (user_id,))
        else:
            result = self.execute(
                "DELETE FROM gim_facts WHERE user_id = ? AND id = ?", (user_id, fact_id)
            )
        return result.rowcount

    def close(self):
        with self._lock:
            self.conn.close()
