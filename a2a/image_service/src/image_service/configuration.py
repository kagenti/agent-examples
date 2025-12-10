from pydantic_settings import BaseSettings
from pydantic import HttpUrl

class Configuration(BaseSettings):
    llm_model: str = "llama3.1"
    llm_api_base: HttpUrl = "http://localhost:11434/v1"
    llm_api_key: str = "dummy"
