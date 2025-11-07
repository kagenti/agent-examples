from pydantic_settings import BaseSettings

class Configuration(BaseSettings):
    LLM_MODEL: str = "llama3.2:3b-instruct-fp16"
    LLM_API_BASE: str = "http://localhost:11434/v1"
    LLM_API_KEY: str = "dummy"
    MCP_URLS: str = "http://localhost:8000/mcp"
    MCP_TRANSPORT: str = "streamable_http"
    MAX_EVENT_DISPLAY_LENGTH: int = 150
    AGENT_VERSION: str = "1.0.0"