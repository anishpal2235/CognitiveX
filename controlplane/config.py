"""Settings + versioned, hot-reloadable YAML policy loading."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    provider_mode: str = "mock"            # mock | openai_compat
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    db_path: str = str(ROOT / "controlplane.db")
    embed_backend: str = "auto"            # auto | st | tfidf
    n_samples: int = 4                     # stochastic samples for entropy


settings = Settings()


class VersionedConfig:
    """Hot-reloadable YAML with a content hash.

    The hash is written into every audit row so a decision can always be
    replayed against the exact policy that produced it -- the answer to
    'why did the system do that in March?'.
    """

    def __init__(self, path: Path):
        self.path = path
        self._mtime = 0.0
        self._data: dict[str, Any] = {}
        self.version = ""
        self.reload()

    def reload(self) -> None:
        raw = self.path.read_bytes()
        self._data = yaml.safe_load(raw.decode("utf-8"))
        digest = hashlib.sha256(raw).hexdigest()[:12]
        self.version = f"{self._data.get('version', 'v0')}+{digest}"
        self._mtime = self.path.stat().st_mtime

    @property
    def data(self) -> dict[str, Any]:
        if self.path.stat().st_mtime != self._mtime:   # hot reload on edit
            self.reload()
        return self._data


@lru_cache(maxsize=1)
def policies() -> VersionedConfig:
    return VersionedConfig(CONFIGS / "policies.yaml")


@lru_cache(maxsize=1)
def models_cfg() -> VersionedConfig:
    return VersionedConfig(CONFIGS / "models.yaml")


def reload_configs() -> dict[str, str]:
    """Force-reload both YAML configs.

    Exposed via POST /v1/policy/reload so a compliance owner can change policy
    without a deploy. Regulations move faster than release trains.
    """
    policies().reload()
    models_cfg().reload()
    return {
        "policy_version": policies().version,
        "models_version": models_cfg().version,
    }
