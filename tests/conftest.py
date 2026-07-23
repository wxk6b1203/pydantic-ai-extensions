"""Shared test fixtures and configuration."""

from __future__ import annotations

import os

import pytest
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

LIVE_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://aa.aa")
LIVE_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-aa")
LIVE_MODEL = os.getenv("DEEPSEEK_MODEL", "aaa")


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
    return OpenAIChatModel(LIVE_MODEL, provider=DeepSeekProvider(openai_client=_make_client()))


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
