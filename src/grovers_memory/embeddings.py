"""Local deterministic encoders. No API calls occur on retrieval."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from typing import Protocol

from .models import ConfigurationError, ValidationError

STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "please",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "to",
        "use",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    return [t for t in re.findall(r"[^\W_]+", text, flags=re.UNICODE) if t not in STOP_WORDS]


def normalize(vector: list[float] | tuple[float, ...], dimensions: int) -> tuple[float, ...]:
    if len(vector) != dimensions:
        raise ValidationError(f"Expected an embedding with {dimensions} dimensions.")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in vector):
        raise ValidationError("Embedding entries must be finite numbers.")
    if any(not math.isfinite(v) for v in vector):
        raise ValidationError("Embedding entries must be finite numbers.")
    scale = max((abs(v) for v in vector), default=0)
    if scale == 0:
        raise ValidationError("Cannot store or search a zero embedding (no searchable terms).")
    scaled = [v / scale for v in vector]
    norm = math.hypot(*scaled)
    if norm == 0:
        raise ValidationError("Cannot store or search a zero embedding (no searchable terms).")
    return tuple(float(v / norm) for v in scaled)


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValidationError("Embedding dimensions do not match.")
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(left, right, strict=True))))


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, text: str) -> tuple[float, ...]: ...


class HashEmbedder:
    """Signed feature hashing of words/bigrams/character trigrams, not semantic AI embeddings."""

    def __init__(self, dimensions: int = 384):
        if (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or not 16 <= dimensions <= 2000
        ):
            raise ValidationError("Embedding dimensions must be an integer between 16 and 2000.")
        self.dimensions = dimensions
        self.name = f"gim-hash-v1:{dimensions}"

    def embed(self, text: str) -> tuple[float, ...]:
        words = tokens(text)
        features: Counter[str] = Counter()
        for word in words:
            features[f"w:{word}"] += 1.0
            if len(word) > 3:
                for index in range(len(word) - 2):
                    features[f"c:{word[index : index + 3]}"] += 0.12
        for a, b in zip(words, words[1:], strict=False):
            features[f"b:{a}:{b}"] += 0.6
        vector = [0.0] * self.dimensions
        for feature, weight in sorted(features.items()):
            digest = hashlib.blake2b(
                feature.encode(), digest_size=8, person=b"gim-hash-v1"
            ).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[bucket] += weight * (1 if digest[4] & 1 else -1)
        return normalize(vector, self.dimensions)


class SentenceTransformerEmbedder:
    """Optional CPU encoder; a pinned model revision is part of the database identity."""

    def __init__(self, model: str, revision: str):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ConfigurationError("Pin a full 40-character embedding model commit revision.")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigurationError("Install grovers-inspired-memory[semantic] first.") from exc
        self._model = SentenceTransformer(
            model, revision=revision, device="cpu", trust_remote_code=False
        )
        self._model.eval()
        self.dimensions = int(self._model.get_sentence_embedding_dimension())
        if not 16 <= self.dimensions <= 2000:
            raise ConfigurationError("This release supports embedding dimensions from 16 to 2000.")
        self.name = f"sentence-transformers:{model}@{revision}:cpu:normalized"

    def embed(self, text: str) -> tuple[float, ...]:
        vector = self._model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return normalize(vector.tolist(), self.dimensions)
