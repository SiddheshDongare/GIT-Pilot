"""Tests for authentication functionality."""

import pytest

from main import authenticate, check_auth_status, logout


def test_authenticate_success(mcp_server, github_token):
    """Test successful authentication."""
    result = authenticate(token=github_token, user_id="test_user")
    assert result["status"] == "authenticated"
    assert "user" in result
    assert "login" in result["user"]


def test_authenticate_invalid_token(mcp_server):
    """Test authentication with invalid token."""
    with pytest.raises(ValueError):
        authenticate(token="invalid_token", user_id="test_user")


def test_check_auth_status_authenticated(mcp_server, github_token):
    """Test auth status check when authenticated."""
    # First authenticate
    authenticate(token=github_token, user_id="test_user")

    # Then check status
    result = check_auth_status(user_id="test_user")
    assert result["authenticated"] is True
    assert "user" in result


def test_check_auth_status_not_authenticated(mcp_server):
    """Test auth status check when not authenticated."""
    result = check_auth_status(user_id="nonexistent_user")
    assert result["authenticated"] is False
    assert "message" in result


def test_logout_success(mcp_server, github_token):
    """Test successful logout."""
    # First authenticate
    authenticate(token=github_token, user_id="test_user")

    # Then logout
    result = logout(user_id="test_user")
    assert result["status"] == "success"

    # Verify logged out
    status = check_auth_status(user_id="test_user")
    assert status["authenticated"] is False


def test_logout_not_authenticated(mcp_server):
    """Test logout when not authenticated."""
    result = logout(user_id="nonexistent_user")
    assert result["status"] == "not_authenticated"
