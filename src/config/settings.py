from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"

    # Academic APIs
    semantic_scholar_api_key: str = ""

    # Storage
    chroma_persist_dir: str = "./data/chroma"

    # App
    log_level: str = "INFO"
    max_papers: int = 50
    max_retries: int = 3

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent.parent

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
