from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # LLM
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-v4-flash"

    # Academic APIs
    semantic_scholar_api_key: str = ""

    # Storage
    chroma_persist_dir: str = "./data/chroma"

    # App
    log_level: str = "INFO"
    log_dir: str = "./logs"
    max_papers: int = 50
    max_retries: int = 3

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent.parent

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
