"""Shared test fixtures and configuration.

Live-API credentials are read from the environment, with an optional `.env` file in
the repo root loaded first (never committed -- `.env` is gitignored). Recognized
variables: `DEEPSEEK_API_KEY` / `TP_COPILOT_API_KEY` (key), `DEEPSEEK_BASE_URL`
(base URL; omitted -> the provider default), `DEEPSEEK_MODEL` / `TP_COPILOT_MODEL`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    """Minimal `.env` loader: KEY=VALUE lines, `#` comments, no override of real env."""
    env_file = _REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

LIVE_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("TP_COPILOT_API_KEY") or "sk-aa"
LIVE_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")  # None -> DeepSeekProvider's official default
LIVE_MODEL = os.getenv("DEEPSEEK_MODEL") or os.getenv("TP_COPILOT_MODEL", "aaa")


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", help="run live API tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live"):
        skip_live = pytest.mark.skip(reason="need --live to run live API tests")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)


def _make_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=LIVE_API_KEY, base_url=LIVE_BASE_URL)


def _make_model() -> OpenAIChatModel:
    if LIVE_BASE_URL:
        provider = DeepSeekProvider(openai_client=_make_client())
    else:
        # No explicit base URL -> DeepSeekProvider's official endpoint.
        provider = DeepSeekProvider(api_key=LIVE_API_KEY)
    return OpenAIChatModel(LIVE_MODEL, provider=provider)


@pytest.fixture
def live_model() -> OpenAIChatModel:
    """A live DeepSeek model for integration tests."""
    return _make_model()


@pytest.fixture
def live_summarizer() -> Agent:
    """A live summarizer agent backed by DeepSeek."""
    return Agent(
        _make_model(),
        instructions="Summarize the conversation concisely, preserving key facts, decisions and TODOs.",
        output_type=str,
    )
