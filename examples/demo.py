"""Run after pip install -e .; only fictional data, no network or API key."""

import json
from pathlib import Path

from grovers_memory import FactInput, Memory

facts = json.loads(Path(__file__).with_name("facts.json").read_text())
with Memory("sqlite:///:memory:") as memory:
    memory.add_many("maya", [FactInput(**fact) for fact in facts])
    print(
        json.dumps(
            memory.retrieve(
                "maya",
                "What is Maya building with Project Atlas?",
                as_of="2026-10-20T12:00:00Z",
            ),
            indent=2,
        )
    )
