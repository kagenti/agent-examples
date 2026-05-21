from pydantic_settings import BaseSettings


class Configuration(BaseSettings):
    llm_model: str = "qwen3:4b"
    llm_api_base: str = "http://host.docker.internal:11434/v1"
    llm_api_key: str = "dummy"
