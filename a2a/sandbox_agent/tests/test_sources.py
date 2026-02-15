"""Tests for SourcesConfig — the sources.json capability loader."""

import json
import tempfile
from pathlib import Path

import pytest

from sandbox_agent.sources import SourcesConfig

# ---------------------------------------------------------------------------
# Fixture: a realistic sources.json payload
# ---------------------------------------------------------------------------

SAMPLE_SOURCES: dict = {
    "_comment": "Declares what this agent can access and install. Baked into agent image.",
    "agent_type": "python-data-agent",
    "package_managers": {
        "pip": {
            "enabled": True,
            "registries": [
                {"name": "pypi", "url": "https://pypi.org/simple/", "trusted": True}
            ],
            "max_install_size_mb": 500,
            "blocked_packages": ["subprocess32", "pyautogui"],
        },
        "conda": {"enabled": False},
        "npm": {"enabled": False},
    },
    "web_access": {
        "enabled": True,
        "allowed_domains": [
            "api.github.com",
            "raw.githubusercontent.com",
            "pypi.org",
            "huggingface.co",
        ],
        "blocked_domains": ["*.internal", "metadata.google.internal"],
    },
    "git": {
        "enabled": True,
        "allowed_remotes": ["https://github.com/*", "https://gitlab.com/*"],
        "max_clone_size_mb": 1000,
    },
    "runtime": {
        "languages": ["python3.11", "bash"],
        "interpreters": {"python": "/usr/bin/python3", "bash": "/bin/bash"},
        "max_execution_time_seconds": 300,
        "max_memory_mb": 2048,
    },
}


@pytest.fixture()
def config() -> SourcesConfig:
    return SourcesConfig.from_dict(SAMPLE_SOURCES)


# ---------------------------------------------------------------------------
# Package-manager tests
# ---------------------------------------------------------------------------


class TestPackageManagerEnabled:
    def test_pip_enabled(self, config: SourcesConfig) -> None:
        assert config.is_package_manager_enabled("pip") is True

    def test_npm_disabled(self, config: SourcesConfig) -> None:
        assert config.is_package_manager_enabled("npm") is False

    def test_conda_disabled(self, config: SourcesConfig) -> None:
        assert config.is_package_manager_enabled("conda") is False

    def test_unknown_manager_disabled(self, config: SourcesConfig) -> None:
        assert config.is_package_manager_enabled("cargo") is False


# ---------------------------------------------------------------------------
# Blocked-package tests
# ---------------------------------------------------------------------------


class TestBlockedPackages:
    def test_blocked_package_subprocess32(self, config: SourcesConfig) -> None:
        assert config.is_package_blocked("pip", "subprocess32") is True

    def test_allowed_package_pandas(self, config: SourcesConfig) -> None:
        assert config.is_package_blocked("pip", "pandas") is False

    def test_blocked_package_pyautogui(self, config: SourcesConfig) -> None:
        assert config.is_package_blocked("pip", "pyautogui") is True

    def test_unknown_manager_returns_false(self, config: SourcesConfig) -> None:
        assert config.is_package_blocked("cargo", "serde") is False


# ---------------------------------------------------------------------------
# Git-remote tests
# ---------------------------------------------------------------------------


class TestGitRemoteAllowed:
    def test_github_allowed(self, config: SourcesConfig) -> None:
        assert config.is_git_remote_allowed("https://github.com/org/repo") is True

    def test_gitlab_allowed(self, config: SourcesConfig) -> None:
        assert config.is_git_remote_allowed("https://gitlab.com/org/repo") is True

    def test_bitbucket_blocked(self, config: SourcesConfig) -> None:
        assert (
            config.is_git_remote_allowed("https://bitbucket.org/org/repo") is False
        )

    def test_git_disabled(self) -> None:
        data = {**SAMPLE_SOURCES, "git": {"enabled": False, "allowed_remotes": []}}
        cfg = SourcesConfig.from_dict(data)
        assert cfg.is_git_remote_allowed("https://github.com/org/repo") is False


# ---------------------------------------------------------------------------
# Runtime-limit tests
# ---------------------------------------------------------------------------


class TestRuntimeLimits:
    def test_max_execution_time_seconds(self, config: SourcesConfig) -> None:
        assert config.max_execution_time_seconds == 300

    def test_max_memory_mb(self, config: SourcesConfig) -> None:
        assert config.max_memory_mb == 2048


# ---------------------------------------------------------------------------
# Default runtime limits (no runtime section)
# ---------------------------------------------------------------------------


class TestRuntimeDefaults:
    def test_default_execution_time(self) -> None:
        cfg = SourcesConfig.from_dict({})
        assert cfg.max_execution_time_seconds == 300

    def test_default_memory(self) -> None:
        cfg = SourcesConfig.from_dict({})
        assert cfg.max_memory_mb == 2048


# ---------------------------------------------------------------------------
# from_file round-trip
# ---------------------------------------------------------------------------


class TestFromFile:
    def test_round_trip(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(SAMPLE_SOURCES, fh)
            fh.flush()
            cfg = SourcesConfig.from_file(Path(fh.name))

        assert cfg.is_package_manager_enabled("pip") is True
        assert cfg.max_execution_time_seconds == 300
