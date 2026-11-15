"""Config and environment loading."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from .schemas import AppConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "llm_test_suite" / "config.json"
DEFAULT_TESTS_DIR = PROJECT_ROOT / "tests"


def load_config(path: Path | None = None) -> AppConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(data)


def load_env() -> None:
    """Load .env files without overriding variables already set in the environment."""
    for candidate in (PROJECT_ROOT / "llm_test_suite" / ".env", PROJECT_ROOT / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


def resolve_output_dir(config: AppConfig, override: str | None = None) -> Path:
    raw = override or config.output_dir
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path
