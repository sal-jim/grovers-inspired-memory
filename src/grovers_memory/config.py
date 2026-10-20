"""Small non-executing .env loader. Real environment variables take precedence."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .embeddings import HashEmbedder, SentenceTransformerEmbedder
from .extraction import LLMExtractor
from .models import ConfigurationError


def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key.strip()):
            raise ConfigurationError(f"Invalid .env assignment on line {number}.")
        value = value.strip()
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ConfigurationError(f"Unclosed .env quote on line {number}.")
            value = value[1:-1]
        result[key.strip()] = value
    return result


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default="sqlite:///memory.db", repr=False)
    embedder: str = "hash"
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = ""

    @classmethod
    def load(cls, path: Path = Path(".env")):
        values = {**read_env(path), **os.environ}
        return cls(
            **{
                name: values["GIM_" + name.upper()]
                for name in cls.__dataclass_fields__
                if "GIM_" + name.upper() in values
            }
        )

    def make_embedder(self):
        if self.embedder == "hash":
            return HashEmbedder()
        if self.embedder == "sentence-transformers":
            return SentenceTransformerEmbedder(self.embedding_model, self.embedding_revision)
        raise ConfigurationError("GIM_EMBEDDER must be hash or sentence-transformers.")

    def make_extractor(self):
        return LLMExtractor(
            api_key=self.llm_api_key, model=self.llm_model, base_url=self.llm_base_url
        )
