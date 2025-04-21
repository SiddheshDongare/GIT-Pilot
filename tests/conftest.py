"""Common test fixtures and configuration."""

import os
from typing import Generator

import pytest
from fastmcp import FastMCP


@pytest.fixture
def mcp_server() -> Generator[FastMCP, None, None]:
    """Create a test MCP server instance."""
    server = FastMCP(name="test-git-pilot")
    yield server


@pytest.fixture
def github_token() -> str:
    """Get GitHub token from environment or use a dummy token for testing."""
    return os.getenv("GITHUB_TOKEN", "dummy_token_for_testing")
