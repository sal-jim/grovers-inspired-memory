import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from grovers_memory import Memory

NOW = datetime(2026, 10, 20, 12, tzinfo=UTC)


@pytest.fixture(params=["sqlite", "postgres"])
def memory(request, user):
    if request.param == "postgres":
        url = os.environ.get("GIM_TEST_POSTGRES_URL")
        if not url:
            pytest.skip("Set GIM_TEST_POSTGRES_URL for real pgvector integration tests")
    else:
        url = "sqlite:///:memory:"
    with Memory(url, clock=lambda: NOW) as instance:
        yield instance
        instance.forget(user)
        instance.forget(user + "-other")


@pytest.fixture
def user():
    # PG tests share one isolated test DB; each test gets an independent tenant.
    return "test-" + uuid4().hex
