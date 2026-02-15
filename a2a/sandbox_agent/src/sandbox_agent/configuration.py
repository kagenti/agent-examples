from pydantic_settings import BaseSettings


class Configuration(BaseSettings):
    llm_model: str = "llama3.1"
    llm_api_base: str = "http://localhost:11434/v1"
    llm_api_key: str = "dummy"
    workspace_root: str = "/workspace"
    checkpoint_db_url: str = "postgresql://kagenti:kagenti@localhost:5432/kagenti_checkpoints"
    context_ttl_days: int = 7
