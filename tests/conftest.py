import pytest
from src.config.settings import Settings


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        llm_provider="deepseek",
        llm_model="deepseek-v4-flash",
        max_papers=10,
    )
