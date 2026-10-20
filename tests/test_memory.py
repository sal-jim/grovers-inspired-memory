import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from conftest import NOW

from grovers_memory import ConflictError, FactInput, Memory, ValidationError
from grovers_memory.models import timestamp


def test_triple_and_baselines(memory, user):
    inputs = [
        FactInput(
            "Project Atlas is an offline field notebook.", entities=["Atlas"], tags=["field"]
        ),
        FactInput(
            "Project Atlas uses SQLite for offline notes.", entities=["Atlas"], tags=["field"]
        ),
        FactInput("Project Atlas supports coastal surveys.", entities=["Atlas"], tags=["field"]),
        FactInput("Jasmine tea tastes floral.", entities=["tea"]),
    ]
    memory.add_many(user, inputs)
    query = "Project Atlas offline field notebook"
    context = memory.retrieve(user, query)
    assert context[0] == inputs[0].text
    assert set(context[1:]) == {inputs[1].text, inputs[2].text}
    assert all(isinstance(item, str) for item in context)
    assert context == memory.retrieve(user, query, strategy="hybrid")
    assert context == memory.retrieve(user, query)
    details = memory.explain(user, query)
    assert details["graph"]["converged"]
    assert details["amplification"]["marked_mass"] > 0
    assert memory.retrieve(user, query, strategy="vector")[0] == inputs[0].text


def test_empty_isolated_and_irrelevant(memory, user):
    assert memory.retrieve(user, "offline notebooks") == []
    memory.add(user, FactInput("Offline notebooks store research data."))
    assert len(memory.retrieve(user, "Offline notebooks")) == 1
    assert memory.retrieve(user + "-other", "Offline notebooks") == []
    assert memory.retrieve(user, "xylophone quasar") == []


def test_revision_intervals_and_late_arrivals(memory, user):
    first = memory.add(
        user,
        FactInput("Maya lives in Oslo.", key="profile.city", observed_at=NOW - timedelta(days=10)),
    )
    latest = memory.add(
        user,
        FactInput("Maya lives in Lima.", key="profile.city", observed_at=NOW - timedelta(days=2)),
    )
    middle = memory.add(
        user,
        FactInput("Maya lives in Tokyo.", key="profile.city", observed_at=NOW - timedelta(days=5)),
    )
    history = memory.history(user, key="PROFILE.CITY")
    assert history[0].superseded_by == middle.id
    assert history[1].superseded_by == latest.id
    assert history[2].superseded_at is None
    assert memory.retrieve(user, "Maya lives", as_of=NOW - timedelta(days=7)) == [first.text]
    assert memory.retrieve(user, "Maya lives", as_of=middle.observed_at) == [middle.text]
    assert memory.retrieve(user, "Maya lives") == [latest.text]
    memory.forget(user, latest.id)
    assert memory.retrieve(user, "Maya lives") == []


def test_future_and_expiry_boundaries(memory, user):
    memory.add(
        user,
        FactInput("Atlas meeting at noon.", observed_at=NOW - timedelta(days=1), expires_at=NOW),
    )
    future = memory.add(
        user, FactInput("Atlas launches tomorrow.", observed_at=NOW + timedelta(days=1))
    )
    assert memory.retrieve(user, "Atlas") == []
    assert memory.retrieve(user, "Atlas", as_of=future.observed_at) == [future.text]


def test_atomic_conflict_and_idempotency(memory, user):
    item = FactInput("Maya lives in Oslo.", key="profile.city")
    stored = memory.add(user, item)
    assert memory.add(user, item).id == stored.id
    with pytest.raises(ConflictError):
        memory.add_many(
            user,
            [
                FactInput("Atlas stores notes."),
                FactInput("Maya lives in Lima.", key="profile.city"),
            ],
        )
    assert len(memory.history(user)) == 1
    with pytest.raises(ValidationError):
        memory.add_many(user, [FactInput("Atlas stores notes."), FactInput("", importance=2)])
    assert len(memory.history(user)) == 1


def test_user_isolation_sql_and_links(memory, user):
    a = memory.add(user, FactInput("Atlas stores field notes.", key="city"))
    b = memory.add(user + "-other", FactInput("Atlas stores secret notes.", key="city"))
    with pytest.raises(ValidationError):
        memory.link(user, a.id, b.id)
    assert memory.forget(user, b.id) == 0
    assert len(memory.history(user)) == 1
    assert memory.history(user, key="x' OR 1=1 --") == []
    assert "secret" not in json.dumps(memory.explain(user, "Atlas"))
    assert memory.retrieve(user + "' OR 1=1 --", "Atlas") == []


def test_expired_nodes_cannot_bridge_graph(memory, user):
    a = memory.add(user, FactInput("Osprey restoration research."))
    b = memory.add(user, FactInput("Amber pigment archive."))
    c = memory.add(user, FactInput("Zircon mineral catalogue."))
    memory.link(user, a.id, b.id)
    memory.link(user, b.id, c.id)
    # Added at NOW, expire after the observation, query just after.
    memory.expire(user, b.id, at=NOW + timedelta(seconds=1))
    assert memory.retrieve(user, a.text, as_of=NOW + timedelta(seconds=2)) == [a.text]


def test_manual_edges_and_expiry_shortening(memory, user):
    a = memory.add(
        user, FactInput("Osprey restoration research.", observed_at=NOW - timedelta(days=1))
    )
    b = memory.add(user, FactInput("Amber pigment archive.", observed_at=NOW - timedelta(days=1)))
    memory.link(user, a.id, b.id, weight=0.8)
    assert memory.retrieve(user, a.text) == [a.text, b.text]
    memory.expire(user, a.id, at=NOW)
    with pytest.raises(ValidationError):
        memory.expire(user, a.id, at=NOW + timedelta(days=1))
    assert memory.retrieve(user, a.text) == []
    assert memory.forget(user) == 2


def test_out_of_order_unkeyed_facts_get_edges(memory, user):
    memory.add(user, FactInput("Project Atlas coastal notebook.", entities=["Atlas"]))
    memory.add(
        user,
        FactInput(
            "Project Atlas supports field surveys.",
            entities=["Atlas"],
            observed_at=NOW - timedelta(days=5),
        ),
    )
    assert len(memory.retrieve(user, "Project Atlas coastal notebook")) == 2


def test_duplicates_not_in_output(memory, user):
    for offset in (1, 2):
        memory.add(
            user,
            FactInput(
                "Atlas is a field notebook.",
                entities=["Atlas"],
                observed_at=NOW - timedelta(days=offset),
            ),
        )
    assert memory.retrieve(user, "Atlas field notebook") == ["Atlas is a field notebook."]


def test_reopen_and_encoder_identity(tmp_path, memory, user):
    from grovers_memory import ConfigurationError, HashEmbedder

    url = "sqlite:///" + str(tmp_path / "nested" / "facts.db")
    with Memory(url) as local:
        local.add("test", FactInput("Atlas is a field notebook."))
    with Memory(url) as local:
        assert local.retrieve("test", "Atlas notebook")
    with pytest.raises(ConfigurationError):
        Memory(url, embedder=HashEmbedder(64))


def test_concurrent_revision_chain(memory, user):
    def insert(day):
        return memory.add(
            user,
            FactInput(f"Maya number {day}.", key="number", observed_at=NOW - timedelta(days=day)),
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(insert, range(1, 13)))
    history = memory.history(user)
    assert len(history) == 12
    assert all(a.superseded_by == b.id for a, b in zip(history, history[1:], strict=False))
    assert history[-1].superseded_at is None


@pytest.mark.parametrize("value", ["2026-01-01", "garbage", "2026-01-01T00:00:00"])
def test_naive_or_bad_timestamps(value):
    with pytest.raises(ValidationError):
        timestamp(value)


def test_key_and_unkeyed_identity_do_not_collide(memory, user):
    a = memory.add(user, FactInput("Actual fact text", key="other fact text"))
    b = memory.add(user, FactInput("other fact text"))
    assert a.id != b.id
