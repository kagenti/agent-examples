"""Tests for the workspace manager.

Validates per-context_id workspace creation, metadata tracking,
and context listing on the shared RWX PVC.
"""

import json
import time

import pytest

from sandbox_agent.workspace import WORKSPACE_SUBDIRS, WorkspaceManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace_root(tmp_path):
    """Provide a temporary directory as the workspace root."""
    return str(tmp_path / "workspace")


@pytest.fixture()
def manager(workspace_root):
    """Create a WorkspaceManager with test defaults."""
    return WorkspaceManager(
        workspace_root=workspace_root,
        agent_name="test-agent",
        namespace="team1",
        ttl_days=7,
    )


# ---------------------------------------------------------------------------
# ensure_workspace
# ---------------------------------------------------------------------------


class TestEnsureWorkspace:
    """Workspace creation and idempotency."""

    def test_creates_all_subdirs(self, manager: WorkspaceManager) -> None:
        """ensure_workspace creates all expected subdirectories."""
        path = manager.ensure_workspace("ctx-abc123")
        for subdir in WORKSPACE_SUBDIRS:
            subdir_path = f"{path}/{subdir}"
            assert (
                __import__("pathlib").Path(subdir_path).is_dir()
            ), f"Missing subdirectory: {subdir}"

    def test_creates_context_json(self, manager: WorkspaceManager) -> None:
        """ensure_workspace creates .context.json with correct fields."""
        path = manager.ensure_workspace("ctx-abc123")
        context_file = __import__("pathlib").Path(path) / ".context.json"
        assert context_file.exists(), ".context.json not created"

        data = json.loads(context_file.read_text())
        assert data["context_id"] == "ctx-abc123"
        assert data["agent"] == "test-agent"
        assert data["namespace"] == "team1"
        assert data["ttl_days"] == 7
        assert "created_at" in data
        assert "last_accessed_at" in data
        assert "disk_usage_bytes" in data

    def test_idempotent_returns_same_path(self, manager: WorkspaceManager) -> None:
        """Calling ensure_workspace twice returns the same path."""
        path1 = manager.ensure_workspace("ctx-abc123")
        path2 = manager.ensure_workspace("ctx-abc123")
        assert path1 == path2

    def test_updates_last_accessed_at(self, manager: WorkspaceManager) -> None:
        """Second call to ensure_workspace updates last_accessed_at."""
        path = manager.ensure_workspace("ctx-abc123")
        context_file = __import__("pathlib").Path(path) / ".context.json"
        data1 = json.loads(context_file.read_text())
        first_accessed = data1["last_accessed_at"]

        # Small delay to ensure timestamps differ
        time.sleep(0.05)

        manager.ensure_workspace("ctx-abc123")
        data2 = json.loads(context_file.read_text())
        second_accessed = data2["last_accessed_at"]

        assert second_accessed > first_accessed, (
            "last_accessed_at should be updated on second call"
        )
        # created_at should remain the same
        assert data1["created_at"] == data2["created_at"]

    def test_rejects_empty_context_id(self, manager: WorkspaceManager) -> None:
        """Empty context_id should raise ValueError, no workspace created."""
        with pytest.raises(ValueError, match="context_id"):
            manager.ensure_workspace("")


# ---------------------------------------------------------------------------
# get_workspace_path
# ---------------------------------------------------------------------------


class TestGetWorkspacePath:
    """Path calculation without side effects."""

    def test_returns_correct_path(
        self, manager: WorkspaceManager, workspace_root: str
    ) -> None:
        """get_workspace_path returns workspace_root / context_id."""
        path = manager.get_workspace_path("ctx-abc123")
        assert path == f"{workspace_root}/ctx-abc123"

    def test_does_not_create_directory(self, manager: WorkspaceManager) -> None:
        """get_workspace_path should not create any directories."""
        path = manager.get_workspace_path("ctx-no-create")
        assert not __import__("pathlib").Path(path).exists()


# ---------------------------------------------------------------------------
# list_contexts
# ---------------------------------------------------------------------------


class TestListContexts:
    """Context enumeration."""

    def test_empty_when_no_contexts(self, manager: WorkspaceManager) -> None:
        """list_contexts returns empty list when no workspaces exist."""
        assert manager.list_contexts() == []

    def test_returns_context_ids(self, manager: WorkspaceManager) -> None:
        """list_contexts returns context_ids after creating workspaces."""
        manager.ensure_workspace("ctx-111")
        manager.ensure_workspace("ctx-222")
        manager.ensure_workspace("ctx-333")

        contexts = manager.list_contexts()
        assert sorted(contexts) == ["ctx-111", "ctx-222", "ctx-333"]
