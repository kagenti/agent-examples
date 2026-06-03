from claude_code_agent.configuration import Configuration


def test_defaults(monkeypatch):
    # Clear anything inherited from the real environment.
    for var in [
        "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL", "WORKSPACE_ROOT", "MAX_SESSIONS",
        "MAX_CONCURRENT", "TURN_TIMEOUT_S",
    ]:
        monkeypatch.delenv(var, raising=False)
    cfg = Configuration(_env_file=None)
    assert cfg.anthropic_auth_token == ""
    assert cfg.has_auth_token is False
    assert cfg.anthropic_base_url == "https://ete-litellm.ai-models.vpc-int.res.ibm.com"
    assert cfg.anthropic_model == "sonnet"
    assert cfg.anthropic_default_haiku_model == "haiku"
    assert cfg.workspace_root == "/workspace"
    assert cfg.max_sessions == 100
    assert cfg.max_concurrent == 8
    assert cfg.turn_timeout_s == 600


def test_env_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-test-123")
    monkeypatch.setenv("ANTHROPIC_MODEL", "opus")
    monkeypatch.setenv("MAX_CONCURRENT", "2")
    cfg = Configuration(_env_file=None)
    assert cfg.anthropic_auth_token == "sk-test-123"
    assert cfg.has_auth_token is True
    assert cfg.anthropic_model == "opus"
    assert cfg.max_concurrent == 2
