from pydantic_settings import BaseSettings
from typing import Annotated
from pydantic import BeforeValidator

def ensure_str(v):
    """Ensure value is a string, preventing pydantic from converting to HttpUrl."""
    return str(v) if v is not None else v

class Configuration(BaseSettings):
    llm_model: str = "llama3.1"
    # Use Annotated to prevent automatic URL conversion to HttpUrl
    llm_api_base: Annotated[str, BeforeValidator(ensure_str)] = "http://localhost:11434/v1"
    llm_api_key: str = "dummy"
