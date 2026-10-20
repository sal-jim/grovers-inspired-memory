import json
import os
import subprocess
import sys
import urllib.error
from io import BytesIO

import pytest
from conftest import NOW

from grovers_memory import ConfigurationError, ExtractionError, LLMExtractor
from grovers_memory.config import Settings, read_env

SOURCE = "Maya lives in Oslo. She builds Atlas."
VALID = {
    "text": "Maya lives in Oslo.",
    "key": "profile.city",
    "entities": ["Maya", "Oslo"],
    "tags": ["location"],
    "expires_at": None,
    "confidence": 1,
    "importance": 0.5,
    "evidence": "Maya lives in Oslo.",
}


def fake_api(monkeypatch, document, *, finish="stop"):
    body = json.dumps(
        {
            "choices": [
                {
                    "finish_reason": finish,
                    "message": {
                        "content": json.dumps(document),
                    },
                }
            ]
        }
    ).encode()
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append((request, timeout))
            return BytesIO(body)

    monkeypatch.setattr("urllib.request.build_opener", lambda *_: Opener())
    return calls


def test_ingest_is_grounded_and_never_used_on_query(monkeypatch, memory, user):
    calls = fake_api(monkeypatch, {"facts": [VALID]})
    extractor = LLMExtractor(api_key="test-placeholder", model="test/model")
    result = memory.ingest(user, SOURCE, extractor=extractor, source_id="chat-1")
    assert result[0].evidence == VALID["evidence"]
    assert result[0].source_id == "chat-1"
    assert result[0].observed_at == NOW
    assert memory.retrieve(user, "Maya Oslo") == [VALID["text"]]
    assert len(calls) == 1
    payload = json.loads(calls[0][0].data)
    assert "user_id" not in json.dumps(payload)
    assert payload["response_format"]["json_schema"]["strict"]


@pytest.mark.parametrize(
    "changes",
    [
        {"evidence": "invented evidence"},
        {"confidence": 2},
        {"confidence": True},
        {"tags": "not a list"},
        {"expires_at": "2026-10-19T00:00:00Z"},
        {"text": ""},
        {"user_id": "someone-else"},
    ],
)
def test_invalid_extraction_is_atomic(monkeypatch, memory, user, changes):
    fake_api(monkeypatch, {"facts": [VALID, {**VALID, **changes}]})
    with pytest.raises(ExtractionError):
        memory.ingest(user, SOURCE, extractor=LLMExtractor(api_key="test", model="test"))
    assert memory.history(user) == []


def test_incomplete_and_network_error(monkeypatch):
    extractor = LLMExtractor(api_key="sensitive-placeholder", model="test")
    fake_api(monkeypatch, {"facts": [VALID]}, finish="length")
    with pytest.raises(ExtractionError, match="did not complete"):
        extractor.extract(SOURCE, observed_at=NOW)

    class Opener:
        def open(self, *_args, **_kwargs):
            raise urllib.error.HTTPError("http://test", 401, "sensitive-placeholder", {}, None)

    monkeypatch.setattr("urllib.request.build_opener", lambda *_: Opener())
    with pytest.raises(ExtractionError) as error:
        extractor.extract(SOURCE, observed_at=NOW)
    assert "401" in str(error.value)
    assert "sensitive-placeholder" not in str(error.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/v1",
        "https://user:pass@example.com/v1",
        "file:///tmp/api",
        "https://example.com/v1?key=test",
        "https://example.com/v1#fragment",
    ],
)
def test_unsafe_api_urls(url):
    with pytest.raises(ConfigurationError):
        LLMExtractor(api_key="test", model="test", base_url=url)


def test_env_is_data_and_env_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "GIM_LLM_API_KEY='$(touch should-not-exist)'\nGIM_DATABASE_URL=sqlite:///:memory:\n"
    )
    assert read_env(env)["GIM_LLM_API_KEY"] == "$(touch should-not-exist)"
    monkeypatch.setenv("GIM_LLM_API_KEY", "private-placeholder")
    settings = Settings.load(env)
    assert settings.llm_api_key == "private-placeholder"
    assert "private-placeholder" not in repr(settings)
    assert not (tmp_path / "should-not-exist").exists()


def test_cli_end_to_end(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIM_")}
    db = "sqlite:///" + str(tmp_path / "memory.db")

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "grovers_memory", "--db", db, *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )

    assert json.loads(run("init").stdout)["ready"]
    assert run("add", "--user", "maya", "--text", "Atlas stores field notes.").returncode == 0
    result = run("query", "--user", "maya", "Atlas field notes")
    assert json.loads(result.stdout) == ["Atlas stores field notes."]
    assert json.loads(run("history", "--user", "maya").stdout)[0]["text"]
    bad = tmp_path / "bad.json"
    bad.write_text('[{"text":"Should rollback"}, {"text":""}]')
    assert run("import", "--user", "maya", str(bad)).returncode == 2
    assert len(json.loads(run("history", "--user", "maya").stdout)) == 1
    assert run("ingest", "--user", "maya", "--file", str(bad)).returncode == 2
    assert json.loads(run("forget", "--user", "maya", "--all").stdout)["deleted"] == 1


def test_semantic_adapter_uses_pinned_cpu_model(monkeypatch):
    import types

    from grovers_memory import SentenceTransformerEmbedder

    calls = []

    class Model:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

        def eval(self):
            pass

        def get_sentence_embedding_dimension(self):
            return 16

        def encode(self, *args, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(tolist=lambda: [1.0] + [0.0] * 15)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Model)
    )
    with pytest.raises(ConfigurationError):
        SentenceTransformerEmbedder("test", "main")
    model = SentenceTransformerEmbedder("test", "a" * 40)
    assert model.embed("test")[0] == 1
    assert calls[0]["device"] == "cpu"
    assert not calls[0]["trust_remote_code"]
    assert calls[1]["normalize_embeddings"]
