import pytest
from src.config.settings import Settings


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-20250514",
        max_papers=10,
    )
