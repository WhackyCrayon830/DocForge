from __future__ import annotations

import tomllib
from pathlib import Path

import tomli_w
from pydantic import BaseModel

from docforge.core.constants import (
    DEFAULT_CONFIG,
    USER_CONFIG,
)
from docforge.core.errors import ConfigError

class AppConfig(BaseModel):
    name: str
    version: str
    data_dir: str
    log_dir: str
    log_level: str

class ProjectsConfig(BaseModel):
    default_project: str

class OllamaConfig(BaseModel):
    base_url: str
    timeout_sec: int
    stream: bool
    connect_retry_max: int
    connect_retry_wait: int

class ModelsConfig(BaseModel):
    itd_model: str
    ctd_model: str
    embedding_model: str

class AgentsConfig(BaseModel):
    itd_context_budget_tokens: int
    ctd_context_budget_tokens: int
    max_retrieval_chunks: int
    min_relevance_score: float
    reviewer_enabled: bool

class EncryptionConfig(BaseModel):
    enabled: bool
    key_file: str

class OcrConfig(BaseModel):
    engine: str
    dpi: int
    preprocessing: bool
    deskew: bool
    denoise: bool
    binarize: bool
    post_correction: bool

class LauncherConfig(BaseModel):
    service_retry_max: int
    service_retry_wait: int
    ollama_start_cmd: str

class LoggingModulesConfig(BaseModel):
    ingest: bool
    ocr: bool
    embeddings: bool
    agents: bool
    llm: bool
    output: bool
    template: bool
    plugins: bool

class LoggingConfig(BaseModel):
    modules: LoggingModulesConfig

class DocForgeConfig(BaseModel):
    app: AppConfig
    projects: ProjectsConfig
    ollama: OllamaConfig
    models: ModelsConfig
    agents: AgentsConfig
    encryption: EncryptionConfig
    ocr: OcrConfig
    launcher: LauncherConfig
    logging: LoggingConfig


_config: DocForgeConfig | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if ( key in result & isinstance(result[key], dict) & isinstance(value, dict)): 
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as f:  # read binary
            return tomllib.load(f)
        
    except FileNotFoundError:
        raise ConfigError(f"Configuration file not found: {path}")

    except Exception as e:
        raise ConfigError(f"Failed to load configuration file '{path}': {e}")


def load_config() -> DocForgeConfig:
    global _config

    default_data = _load_toml(DEFAULT_CONFIG)

    if USER_CONFIG.exists():
        user_data = _load_toml(USER_CONFIG)
        merged = _deep_merge(default_data, user_data)
    else:
        merged = default_data

    try:
        _config = DocForgeConfig.model_validate(merged)
        return _config
    
    except Exception as e:
        raise ConfigError(
            f"Configuration validation failed: {e}"
        )


def get_config() -> DocForgeConfig:
    if _config is None:
        return load_config()

    return _config


def save_user_config(updates: dict) -> None:
    existing = {}
    if USER_CONFIG.exists():
        existing = _load_toml(USER_CONFIG)

    merged = _deep_merge(existing, updates)
    
    USER_CONFIG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(USER_CONFIG, "wb") as f:
        tomli_w.dump(merged, f)