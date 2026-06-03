from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Configuration(BaseSettings):
    """Runtime configuration. The only required value is ANTHROPIC_AUTH_TOKEN."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Model / LiteLLM wiring ---
    anthropic_auth_token: str = ""
    # Required: the LiteLLM (Anthropic-compatible) endpoint, e.g.
    # https://litellm.example.com. Empty by default so no environment-specific host
    # is baked into this public example; when empty, Claude falls back to
    # api.anthropic.com.
    anthropic_base_url: str = ""
    anthropic_model: str = "sonnet"
    anthropic_default_haiku_model: str = "haiku"
    # These map to CLAUDE_CODE_* env vars Claude itself reads; kept here so the
    # runner forwards them deterministically to the subprocess.
    disable_experimental_betas: str = Field(default="1", validation_alias="CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS")
    disable_nonessential_traffic: str = Field(default="1", validation_alias="CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")

    # --- Sessions / workspaces ---
    workspace_root: str = "/workspace"
    home_dir: str = "/home/agent"  # writable HOME so the subprocess can write ~/.claude
    max_sessions: int = 100
    max_concurrent: int = 8
    turn_timeout_s: int = 600

    # --- A2A server ---
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def has_auth_token(self) -> bool:
        return bool(self.anthropic_auth_token.strip())
