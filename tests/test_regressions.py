import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from conftest import NOW

from grovers_memory import FactInput, LLMExtractor, Memory, RetrievalConfig
from grovers_memory.extraction import _NoRedirect


def test_late_insert_after_forget_cannot_reopen_old_interval(memory, user):
    memory.add(user, FactInput("City Oslo", key="city", observed_at=NOW - timedelta(days=10)))
    latest = memory.add(
        user, FactInput("City Lima", key="city", observed_at=NOW - timedelta(days=2))
    )
    memory.forget(user, latest.id)
    middle = memory.add(
        user, FactInput("City Tokyo", key="city", observed_at=NOW - timedelta(days=5))
    )
    assert memory.retrieve(user, "City") == []
    assert memory.retrieve(user, "City", as_of=NOW - timedelta(days=4)) == [middle.text]


def test_rollback_restores_previous_interval(memory, user):
    from grovers_memory import ConflictError

    old = memory.add(user, FactInput("City Oslo", key="city", observed_at=NOW - timedelta(days=2)))
    with pytest.raises(ConflictError):
        memory.add_many(
            user,
            [
                FactInput("City Tokyo", key="city", observed_at=NOW - timedelta(days=1)),
                FactInput("City Lima", key="city", observed_at=old.observed_at),
            ],
        )
    assert memory.history(user)[0].superseded_at is None
    assert memory.retrieve(user, "City") == [old.text]


def test_separate_connections_serialize_revisions(memory, user, tmp_path):
    url = (
        os.environ["GIM_TEST_POSTGRES_URL"]
        if memory._store.postgres
        else "sqlite:///" + str(tmp_path / "concurrent.db")
    )

    def insert(day):
        with Memory(url, clock=lambda: NOW) as instance:
            instance.add(
                user,
                FactInput(f"City number {day}", key="city", observed_at=NOW - timedelta(days=day)),
            )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(insert, range(1, 9)))
    with Memory(url, clock=lambda: NOW) as instance:
        history = instance.history(user)
        assert len(history) == 8
        assert sum(f.active_at(NOW) for f in history) == 1
        assert instance.retrieve(user, "City number") == ["City number 1"]
        instance.forget(user)


def test_graph_bounds_and_zero_hops():
    with Memory(
        "sqlite:///:memory:", config=RetrievalConfig(graph_nodes=2, graph_hops=1)
    ) as memory:
        memory.add_many(
            "test", [FactInput(f"Atlas number {n}", entities=["Atlas"]) for n in range(20)]
        )
        result = memory.explain("test", "Atlas number 1")
        assert result["graph"]["nodes"] <= 2
        assert len(result["context"]) <= 2
    with Memory("sqlite:///:memory:", config=RetrievalConfig(graph_hops=0)) as memory:
        memory.add_many(
            "test", [FactInput(f"Atlas number {n}", entities=["Atlas"]) for n in range(3)]
        )
        assert len(memory.retrieve("test", "Atlas number 1")) == 1


def test_similar_templates_about_other_projects_do_not_dominate(memory, user):
    facts = [
        FactInput("Atlas builds an offline marine research notebook.", entities=["Atlas"]),
        FactInput("Atlas stores observations in SQLite.", entities=["Atlas"]),
        FactInput("Atlas supports tide pool surveys.", entities=["Atlas"]),
    ]
    facts.extend(
        FactInput(
            f"Cedar{index} builds an offline marine research notebook.", entities=[f"Cedar{index}"]
        )
        for index in range(30)
    )
    memory.add_many(user, facts)
    result = memory.retrieve(user, facts[0].text)
    assert result[0] == facts[0].text
    assert set(result[1:]) == {facts[1].text, facts[2].text}


def test_evidence_retains_verbatim_unicode():
    quote = "Maya uses ﬁeld notebooks."
    document = {
        "facts": [
            {
                "text": quote,
                "key": None,
                "entities": [],
                "tags": [],
                "expires_at": None,
                "confidence": 1,
                "importance": 0.5,
                "evidence": quote,
            }
        ]
    }
    fact = LLMExtractor.validate(document, text=quote, observed_at=NOW)[0]
    assert fact.evidence == quote
    assert "field" in fact.text


def test_redirects_never_forward_credentials():
    assert (
        _NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://elsewhere.test")
        is None
    )


def test_opaque_tenant_ids_are_not_unicode_normalized(memory, user):
    fullwidth = user + "-Ａ"
    ascii_id = user + "-A"
    try:
        memory.add(fullwidth, FactInput("Atlas private field notebook."))
        assert memory.retrieve(ascii_id, "Atlas field notebook") == []
        assert memory.retrieve(fullwidth, "Atlas field notebook")
    finally:
        memory.forget(fullwidth)
        memory.forget(ascii_id)


def test_import_does_not_select_tenant(tmp_path):
    from grovers_memory.cli import main

    path = tmp_path / "input.json"
    path.write_text(json.dumps([{"text": "Private fact", "user_id": "other-user"}]))
    assert (
        main(
            [
                "--env-file",
                str(tmp_path / "missing"),
                "--db",
                "sqlite:///:memory:",
                "import",
                "--user",
                "caller",
                str(path),
            ]
        )
        == 2
    )
