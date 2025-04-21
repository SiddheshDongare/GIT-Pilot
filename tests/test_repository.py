"""Tests for repository operations."""

import pytest

from main import create_repository, get_commit, list_commits


def test_create_repository_success(mcp_server, github_token):
    """Test successful repository creation."""
    result = create_repository(
        name="test-repo",
        description="Test repository",
        private=True,
        token=github_token,
        user_id="test_user",
    )
    assert result["name"] == "test-repo"
    assert result["private"] is True
    assert "url" in result


def test_create_repository_invalid_name(mcp_server, github_token):
    """Test repository creation with invalid name."""
    with pytest.raises(ValueError):
        create_repository(
            name="invalid/repo/name", token=github_token, user_id="test_user"
        )


def test_list_commits(mcp_server, github_token):
    """Test listing commits from a repository."""
    # First create a repository
    repo = create_repository(
        name="test-repo-commits", token=github_token, user_id="test_user"
    )

    # Then list commits
    commits = list_commits(
        repo_path=repo["full_name"], token=github_token, user_id="test_user"
    )
    assert isinstance(commits, list)
    if commits:  # If there are any commits
        assert "sha" in commits[0]
        assert "message" in commits[0]


def test_get_commit(mcp_server, github_token):
    """Test getting a specific commit."""
    # First create a repository
    repo = create_repository(
        name="test-repo-commit", token=github_token, user_id="test_user"
    )

    # Get the first commit
    commits = list_commits(
        repo_path=repo["full_name"], token=github_token, user_id="test_user"
    )

    if commits:
        commit = get_commit(
            repo_path=repo["full_name"],
            commit_sha=commits[0]["sha"],
            token=github_token,
            user_id="test_user",
        )
        assert commit["sha"] == commits[0]["sha"]
        assert "message" in commit
        assert "author" in commit
