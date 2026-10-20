"""Local JSON CLI. No public server or implicit user authentication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import FactInput, Memory, MemoryErrorBase
from .config import Settings


def parser():
    p = argparse.ArgumentParser(prog="gim", description=__doc__)
    p.add_argument("--env-file", type=Path, default=Path(".env"))
    p.add_argument("--db", help="Override GIM_DATABASE_URL")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create/check the database and encoder identity")
    add = sub.add_parser("add", help="Store a fact without an AI call")
    add.add_argument("--user", required=True)
    add.add_argument("--text", required=True)
    add.add_argument("--key")
    add.add_argument("--entity", action="append", default=[])
    add.add_argument("--tag", action="append", default=[])
    add.add_argument("--observed-at")
    add.add_argument("--expires-at")
    add.add_argument("--source-id")
    imp = sub.add_parser("import", help="Atomically import a JSON array of FactInput objects")
    imp.add_argument("--user", required=True)
    imp.add_argument("file", type=Path)
    ing = sub.add_parser("ingest", help="Extract grounded facts with the configured AI")
    ing.add_argument("--user", required=True)
    ing.add_argument("--file", type=Path, required=True)
    ing.add_argument("--source-id")
    for name in ("query", "explain"):
        q = sub.add_parser(
            name, help="Retrieve strings" if name == "query" else "Developer diagnostics"
        )
        q.add_argument("--user", required=True)
        q.add_argument("--as-of")
        q.add_argument("--strategy", choices=["grover", "hybrid", "vector"], default="grover")
        q.add_argument("text")
    hist = sub.add_parser("history", help="Export one user's dated revisions")
    hist.add_argument("--user", required=True)
    hist.add_argument("--key")
    link = sub.add_parser("link", help="Add an explicit relationship between two fact IDs")
    link.add_argument("--user", required=True)
    link.add_argument("a")
    link.add_argument("b")
    link.add_argument("--weight", type=float, default=1.0)
    expire = sub.add_parser("expire", help="Shorten one fact's validity")
    expire.add_argument("--user", required=True)
    expire.add_argument("id")
    expire.add_argument("--at")
    forget = sub.add_parser("forget", help="Permanently delete a fact or all of one user's facts")
    forget.add_argument("--user", required=True)
    group = forget.add_mutually_exclusive_group(required=True)
    group.add_argument("--id")
    group.add_argument("--all", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        settings = Settings.load(args.env_file)
        with Memory(args.db or settings.database_url, embedder=settings.make_embedder()) as memory:
            command = args.command
            if command == "init":
                result = {"ready": True, "encoder": memory.embedder.name}
            elif command == "add":
                result = memory.add(
                    args.user,
                    FactInput(
                        text=args.text,
                        key=args.key,
                        entities=args.entity,
                        tags=args.tag,
                        observed_at=args.observed_at,
                        expires_at=args.expires_at,
                        source_id=args.source_id,
                    ),
                ).to_dict()
            elif command == "import":
                if args.file.stat().st_size > 8 * 1024 * 1024:
                    raise ValueError("Import file exceeds 8 MiB.")
                values = json.loads(args.file.read_text())
                if not isinstance(values, list):
                    raise ValueError("Import must be a JSON array of FactInput objects.")
                facts = memory.add_many(args.user, [FactInput(**v) for v in values])
                result = {"items": len(facts), "ids": [f.id for f in facts]}
            elif command == "ingest":
                if args.file.stat().st_size > 128000:
                    raise ValueError("Source file exceeds 128 KiB.")
                facts = memory.ingest(
                    args.user,
                    args.file.read_text(),
                    extractor=settings.make_extractor(),
                    source_id=args.source_id,
                )
                result = {"items": len(facts), "ids": [f.id for f in facts]}
            elif command in {"query", "explain"}:
                method = memory.retrieve if command == "query" else memory.explain
                result = method(args.user, args.text, as_of=args.as_of, strategy=args.strategy)
            elif command == "history":
                result = [f.to_dict() for f in memory.history(args.user, key=args.key)]
            elif command == "link":
                memory.link(args.user, args.a, args.b, weight=args.weight)
                result = {"linked": True}
            elif command == "expire":
                memory.expire(args.user, args.id, at=args.at)
                result = {"expired": True}
            else:
                result = {"deleted": memory.forget(args.user, args.id)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except MemoryErrorBase as exc:
        print(f"gim: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError):
        # These exceptions contain local validation messages, never provider bodies.
        # Avoid printing arbitrary exception text (e.g. an imported credential as a key).
        print(
            "gim: invalid input or configuration; check fields, dates, encoder and AI settings.",
            file=sys.stderr,
        )
        return 2
    except OSError:
        print("gim: cannot access the input file or database.", file=sys.stderr)
        return 2
    except Exception:
        print(
            "gim: database/provider operation failed; no credentials are printed.", file=sys.stderr
        )
        return 1
