"""Load configs/data.yaml and resolve paths relative to the repo root."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data.yaml"


@dataclass
class Config:
    raw: dict

    @property
    def start(self) -> str:
        return self.raw["dates"]["start"]

    @property
    def end(self) -> str:
        return self.raw["dates"]["end"] or date.today().isoformat()

    def path(self, key: str) -> Path:
        p = REPO_ROOT / self.raw["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def __getitem__(self, key: str):
        return self.raw[key]

    @property
    def fred_api_key(self) -> str | None:
        return os.environ.get("FRED_API_KEY") or None


def load_config(path: str | Path | None = None) -> Config:
    load_dotenv(REPO_ROOT / ".env")
    with open(path or DEFAULT_CONFIG) as f:
        return Config(yaml.safe_load(f))
