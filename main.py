"""
GitHub Service - A FastMCP-based API wrapper for GitHub operations.

This module provides a set of tools for interacting with the GitHub API through
a FastMCP server. It includes functions for user management, repository operations,
branch management, file operations, issue tracking, and pull request handling.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

from cryptography.fernet import Fernet
from github import Github
from github.GithubException import GithubException, RateLimitExceededException
from mcp.server.fastmcp import FastMCP


# Configuration Management
@dataclass
class Config:
    """Configuration settings for the GitHub service."""

    TOKEN_TTL_HOURS: int = 24
    MAX_STORED_TOKENS: int = 1000
    CLEANUP_INTERVAL_SECONDS: int = 3600
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: int = 5
    ENCRYPTION_KEY: bytes = Fernet.generate_key()


config = Config()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP(name="GIT-Pilot")

# Type variables for better type hinting
T = TypeVar("T")


# Token storage with TTL
class TokenManager:
    def __init__(self):
        self.tokens = {}  # Format: {user_id: {"token": encrypted_token, "expires_at": datetime}}
        self._lock = threading.Lock()
        self._running = True
        self.cipher_suite = Fernet(config.ENCRYPTION_KEY)

        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_expired_tokens, daemon=True
        )
        self._cleanup_thread.start()

    def store_token(self, user_id: str, token: str, ttl_hours: int = None) -> None:
        """Store an encrypted token with expiration time"""
        if ttl_hours is None:
            ttl_hours = config.TOKEN_TTL_HOURS

        with self._lock:
            # Check if we've hit the maximum token limit
            if len(self.tokens) >= config.MAX_STORED_TOKENS:
                # Remove oldest token
                oldest_user = min(
                    self.tokens.items(), key=lambda x: x[1]["expires_at"]
                )[0]
                del self.tokens[oldest_user]
                logger.warning(
                    f"Maximum token limit reached. Removed oldest token for user {oldest_user}"
                )

            # Encrypt token before storing
            encrypted_token = self.cipher_suite.encrypt(token.encode())
            expires_at = datetime.now() + timedelta(hours=ttl_hours)
            self.tokens[user_id] = {"token": encrypted_token, "expires_at": expires_at}
            logger.info(
                f"Token stored for user {user_id}, expires in {ttl_hours} hours"
            )

    def get_token(self, user_id: str) -> Optional[str]:
        """Get a decrypted token if it exists and is valid"""
        with self._lock:
            if user_id in self.tokens:
                token_data = self.tokens[user_id]
                if token_data["expires_at"] > datetime.now():
                    # Decrypt token before returning
                    try:
                        decrypted_token = self.cipher_suite.decrypt(
                            token_data["token"]
                        ).decode()
                        return decrypted_token
                    except Exception as e:
                        logger.error(
                            f"Error decrypting token for user {user_id}: {str(e)}"
                        )
                        self.revoke_token(user_id)
                        return None
                else:
                    # Token expired
                    self.revoke_token(user_id)
                    return None
            return None

    def revoke_token(self, user_id: str) -> bool:
        """Remove a token and securely clear it from memory"""
        with self._lock:
            if user_id in self.tokens:
                try:
                    # Securely clear token data
                    self.tokens[user_id]["token"] = None
                    del self.tokens[user_id]
                    logger.info(f"Token revoked for user {user_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error revoking token for user {user_id}: {str(e)}")
            return False

    def _cleanup_expired_tokens(self) -> None:
        """Background task to clean up expired tokens"""
        while self._running:
            try:
                with self._lock:
                    now = datetime.now()
                    expired_users = [
                        user_id
                        for user_id, token_data in self.tokens.items()
                        if token_data["expires_at"] <= now
                    ]

                    for user_id in expired_users:
                        self.revoke_token(user_id)
                        logger.info(f"Expired token removed for user {user_id}")

                # Sleep for configured interval
                time.sleep(config.CLEANUP_INTERVAL_SECONDS)
            except Exception as e:
                logger.error(f"Error in token cleanup: {str(e)}")
                time.sleep(config.CLEANUP_INTERVAL_SECONDS)

    def shutdown(self):
        """Gracefully shutdown the token manager"""
        self._running = False
        self._cleanup_thread.join(timeout=5.0)
        with self._lock:
            # Securely clear all tokens
            for user_id in list(self.tokens.keys()):
                self.revoke_token(user_id)
            self.tokens.clear()


# Create token manager instance
token_manager = TokenManager()


# Rate limiting decorator
def handle_rate_limit(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to handle GitHub API rate limiting"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        retries = 0
        while retries < config.MAX_RETRIES:
            try:
                return func(*args, **kwargs)
            except RateLimitExceededException as e:
                reset_time = e.reset
                wait_time = (
                    reset_time - time.time()
                    if reset_time
                    else config.RETRY_DELAY_SECONDS
                )
                if wait_time > 0 and retries < config.MAX_RETRIES - 1:
                    logger.warning(
                        f"Rate limit hit. Waiting {wait_time:.2f} seconds before retry."
                    )
                    time.sleep(wait_time)
                    retries += 1
                else:
                    raise
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return func(*args, **kwargs)
    return wrapper


class GitHubClientManager:
    """Manages GitHub client instances with validation and resource cleanup"""

    def __init__(self):
        self._clients = {}
        self._lock = threading.Lock()

    def get_client(
        self, token: Optional[str] = None, user_id: Optional[str] = None
    ) -> Github:
        """Get a validated GitHub client instance"""
        client_token = None

        if token:
            client_token = token
        elif user_id:
            client_token = token_manager.get_token(user_id)

        if not client_token:
            client_token = os.getenv("GITHUB_TOKEN")

        if not client_token:
            raise ValueError(
                "GitHub authentication required. Please provide a token or authenticate first."
            )

        # Use token as key for client cache
        cache_key = hash(client_token)

        with self._lock:
            if cache_key in self._clients:
                client = self._clients[cache_key]
                # Validate client is still working
                try:
                    client.get_user().id
                    return client
                except Exception:
                    # Remove invalid client
                    self._remove_client(cache_key)

            # Create and validate new client
            client = Github(client_token)
            try:
                # Validate token works
                client.get_user().id
                self._clients[cache_key] = client
                return client
            except Exception as e:
                raise ValueError(f"Invalid GitHub token: {str(e)}")

    def _remove_client(self, cache_key: int) -> None:
        """Remove and cleanup a client instance"""
        if cache_key in self._clients:
            try:
                self._clients[cache_key].close()
            except Exception as e:
                logger.error(f"Error closing client: {str(e)}")
            del self._clients[cache_key]

    def cleanup(self) -> None:
        """Cleanup all client instances"""
        with self._lock:
            for cache_key in list(self._clients.keys()):
                self._remove_client(cache_key)


# Create GitHub client manager instance
github_client_manager = GitHubClientManager()


def validate_repository_path(repo_path: str) -> None:
    """Validate repository path format"""
    if not repo_path or "/" not in repo_path:
        raise ValueError("Repository path must be in format 'owner/repo'")


def validate_branch_name(branch: str) -> None:
    """Validate branch name format"""
    if not branch or "/" in branch:
        raise ValueError("Invalid branch name")


@handle_rate_limit
def get_github_client(
    token: Optional[str] = None, user_id: Optional[str] = None
) -> Github:
    """
    Get a GitHub client instance with appropriate authentication.

    Args:
        token: Explicit GitHub token
        user_id: User ID to look up stored token

    Returns:
        Authenticated GitHub client

    Raises:
        ValueError: If no valid authentication method is available or token is invalid
    """
    return github_client_manager.get_client(token, user_id)


@mcp.resource("guide://github-token-creation")
def get_token_creation_guide() -> dict:
    """
    Provides detailed instructions for creating a GitHub personal access token.

    Returns:
        Dictionary containing step-by-step instructions and helpful tips
    """
    return {
        "title": "GitHub Token Creation Guide",
        "description": "Follow these steps to create a GitHub personal access token. This token will allow the application to interact with your GitHub account securely.",
        "steps": [
            {
                "step": 1,
                "title": "Access GitHub Settings",
                "instructions": [
                    "Go to GitHub.com and sign in to your account",
                    "Click your profile picture in the top-right corner",
                    "Select 'Settings' from the dropdown menu",
                ],
                "tip": "You can also directly visit: https://github.com/settings",
            },
            {
                "step": 2,
                "title": "Navigate to Developer Settings",
                "instructions": [
                    "Scroll down to the bottom of the left sidebar",
                    "Click on 'Developer settings' (it's the last option)",
                ],
                "tip": "This section contains all developer-related settings for your account",
            },
            {
                "step": 3,
                "title": "Access Personal Access Tokens",
                "instructions": [
                    "Click on 'Personal access tokens' in the left sidebar",
                    "Select 'Tokens (classic)' from the dropdown menu",
                ],
                "tip": "The 'classic' tokens provide more granular control over permissions",
            },
            {
                "step": 4,
                "title": "Generate New Token",
                "instructions": [
                    "Click the 'Generate new token' button",
                    "Select 'Generate new token (classic)'",
                ],
                "tip": "Make sure you're generating a classic token for maximum compatibility",
            },
            {
                "step": 5,
                "title": "Configure Token Settings",
                "instructions": [
                    "Give your token a descriptive name (e.g., 'GIT-Pilot Access')",
                    "Set an expiration date (recommended: 30 days or custom based on your needs)",
                    "Add a note about the token's purpose (optional but recommended)",
                ],
                "tip": "Using a descriptive name helps you identify the token's purpose later",
            },
            {
                "step": 6,
                "title": "Select Required Permissions",
                "instructions": [
                    "Under 'Select scopes', choose the following permissions:",
                    "- repo (Full control of private repositories)",
                    "- user (Read user information)",
                    "- workflow (Update GitHub Action workflows)",
                ],
                "tip": "Only select the permissions you need. The ones listed above are the minimum required for this application.",
            },
            {
                "step": 7,
                "title": "Generate and Save Token",
                "instructions": [
                    "Scroll to the bottom of the page",
                    "Click 'Generate token'",
                    "IMPORTANT: Copy the token immediately and store it securely",
                    "You won't be able to see it again!",
                ],
                "tip": "Store the token in a secure password manager or encrypted file. Never share it or commit it to version control.",
            },
        ],
        "important_notes": [
            "The token will only be shown once when you create it. Make sure to copy it immediately!",
            "If you lose your token, you'll need to generate a new one and revoke the old one",
            "You can always revoke tokens from the same page where you created them",
            "For security, set an appropriate expiration date for your token",
        ],
        "security_tips": [
            "Never share your token with anyone",
            "Don't commit the token to version control",
            "Regularly rotate your tokens by creating new ones and revoking old ones",
            "Use environment variables or secure secret management to store your token",
        ],
        "troubleshooting": {
            "common_issues": [
                {
                    "issue": "Token not working",
                    "solution": "Make sure you've selected all the required permissions and the token hasn't expired",
                },
                {
                    "issue": "Can't find the token after creation",
                    "solution": "You'll need to generate a new token as tokens are only shown once upon creation",
                },
                {
                    "issue": "Permission denied errors",
                    "solution": "Check if you've selected all the required permissions when creating the token",
                },
            ]
        },
        "next_steps": [
            "After creating your token, you can use it with the application",
            "Use the 'authenticate' function to securely store your token",
            "You can verify your authentication status using 'check_auth_status'",
        ],
    }


@mcp.resource("guide://github-token-troubleshooting")
def get_token_troubleshooting_guide() -> dict:
    """
    Provides detailed troubleshooting information for common GitHub token issues.

    Returns:
        Dictionary containing troubleshooting information and solutions
    """
    return {
        "title": "GitHub Token Troubleshooting Guide",
        "description": "Solutions for common issues you might encounter when working with GitHub personal access tokens.",
        "categories": [
            {
                "category": "Authentication Issues",
                "problems": [
                    {
                        "issue": "Token not working",
                        "symptoms": [
                            "Authentication errors when trying to use the token",
                            "401 Unauthorized responses",
                            "Rate limit errors",
                        ],
                        "possible_causes": [
                            "Token has expired",
                            "Token was revoked",
                            "Incorrect permissions were granted",
                            "Token was copied incorrectly",
                        ],
                        "solutions": [
                            "Check if the token has expired in GitHub settings",
                            "Verify the token hasn't been revoked",
                            "Ensure all required permissions are granted",
                            "Generate a new token if needed",
                        ],
                        "prevention": [
                            "Set appropriate expiration dates",
                            "Use descriptive names for tokens",
                            "Document token purposes",
                            "Regularly rotate tokens",
                        ],
                    },
                    {
                        "issue": "Permission denied errors",
                        "symptoms": [
                            "403 Forbidden responses",
                            "Unable to access repositories",
                            "Unable to perform specific actions",
                        ],
                        "possible_causes": [
                            "Missing required permissions",
                            "Token scope is too restrictive",
                            "Repository access restrictions",
                        ],
                        "solutions": [
                            "Review and update token permissions",
                            "Generate a new token with correct permissions",
                            "Check repository access settings",
                        ],
                        "prevention": [
                            "Document required permissions",
                            "Use separate tokens for different purposes",
                            "Regularly audit token permissions",
                        ],
                    },
                ],
            },
            {
                "category": "Token Management",
                "problems": [
                    {
                        "issue": "Lost or forgotten token",
                        "symptoms": [
                            "Unable to find the token",
                            "Token not stored securely",
                            "Token accidentally deleted",
                        ],
                        "solutions": [
                            "Generate a new token",
                            "Revoke the old token for security",
                            "Update any systems using the old token",
                        ],
                        "prevention": [
                            "Use a password manager",
                            "Store tokens in secure environment variables",
                            "Document token locations and purposes",
                        ],
                    },
                    {
                        "issue": "Token security concerns",
                        "symptoms": [
                            "Token exposed in logs or code",
                            "Token committed to version control",
                            "Token shared with others",
                        ],
                        "solutions": [
                            "Immediately revoke the compromised token",
                            "Generate a new token",
                            "Update all systems using the old token",
                            "Review security logs for unauthorized access",
                        ],
                        "prevention": [
                            "Never commit tokens to version control",
                            "Use .gitignore for files containing tokens",
                            "Implement token rotation policies",
                            "Use environment variables or secret management",
                        ],
                    },
                ],
            },
            {
                "category": "Rate Limiting",
                "problems": [
                    {
                        "issue": "Rate limit exceeded",
                        "symptoms": [
                            "429 Too Many Requests responses",
                            "API requests failing",
                            "Rate limit warnings",
                        ],
                        "solutions": [
                            "Wait for rate limit reset",
                            "Implement rate limit handling",
                            "Use token with higher rate limits",
                        ],
                        "prevention": [
                            "Monitor API usage",
                            "Implement request throttling",
                            "Use appropriate token types",
                        ],
                    }
                ],
            },
        ],
        "best_practices": [
            "Regularly audit token usage and permissions",
            "Implement token rotation policies",
            "Use environment variables for token storage",
            "Monitor for suspicious activity",
            "Keep documentation of token purposes and locations",
            "Use separate tokens for different applications",
            "Implement proper error handling for token issues",
        ],
        "additional_resources": [
            {
                "title": "GitHub Token Creation Guide",
                "uri": "guide://github-token-creation",
                "description": "Step-by-step guide for creating GitHub tokens",
            },
            {
                "title": "GitHub Token FAQ",
                "uri": "guide://github-token-faq",
                "description": "Frequently asked questions about GitHub tokens",
            },
        ],
    }


@mcp.resource("guide://github-token-faq")
def get_token_faq() -> dict:
    """
    Provides answers to frequently asked questions about GitHub tokens.

    Returns:
        Dictionary containing FAQ information
    """
    return {
        "title": "GitHub Token FAQ",
        "description": "Common questions and answers about GitHub personal access tokens.",
        "categories": [
            {
                "category": "General Questions",
                "questions": [
                    {
                        "question": "What is a GitHub personal access token?",
                        "answer": "A personal access token is a string that serves as an alternative to your password when using the GitHub API or command line. It allows you to access your GitHub account and perform actions on your behalf.",
                        "related_topics": ["security", "authentication"],
                    },
                    {
                        "question": "Why do I need a personal access token?",
                        "answer": "Personal access tokens are required for programmatic access to GitHub, such as using the GitHub API, command line tools, or third-party applications. They provide a secure way to authenticate without using your password.",
                        "related_topics": ["security", "api"],
                    },
                    {
                        "question": "How long do tokens last?",
                        "answer": "Tokens can be set to expire after a specific time period (e.g., 30 days, 1 year) or can be set to never expire. It's recommended to set an expiration date for security purposes.",
                        "related_topics": ["security", "management"],
                    },
                ],
            },
            {
                "category": "Security",
                "questions": [
                    {
                        "question": "How should I store my token?",
                        "answer": "Store tokens securely using environment variables, secret management systems, or password managers. Never commit tokens to version control or share them with others.",
                        "related_topics": ["security", "best-practices"],
                    },
                    {
                        "question": "What should I do if my token is compromised?",
                        "answer": "Immediately revoke the compromised token in GitHub settings, generate a new token, and update all systems using the old token. Review security logs for any unauthorized access.",
                        "related_topics": ["security", "incident-response"],
                    },
                    {
                        "question": "How often should I rotate my tokens?",
                        "answer": "Regular token rotation is recommended for security. Consider rotating tokens every 30-90 days, or immediately if you suspect any security issues.",
                        "related_topics": ["security", "best-practices"],
                    },
                ],
            },
            {
                "category": "Permissions and Scopes",
                "questions": [
                    {
                        "question": "What permissions should I grant my token?",
                        "answer": "Grant only the permissions needed for your specific use case. For this application, you need: repo (repository access), user (user information), and workflow (GitHub Actions) permissions.",
                        "related_topics": ["permissions", "security"],
                    },
                    {
                        "question": "Can I modify token permissions after creation?",
                        "answer": "No, you cannot modify permissions after creating a token. You'll need to create a new token with the desired permissions.",
                        "related_topics": ["permissions", "management"],
                    },
                    {
                        "question": "What's the difference between classic and fine-grained tokens?",
                        "answer": "Classic tokens provide broader access to repositories and features, while fine-grained tokens offer more granular control over repository access and permissions.",
                        "related_topics": ["permissions", "security"],
                    },
                ],
            },
            {
                "category": "Usage and Management",
                "questions": [
                    {
                        "question": "How do I use my token with the GitHub API?",
                        "answer": "Include your token in the Authorization header of your API requests: 'Authorization: token YOUR_TOKEN'",
                        "related_topics": ["api", "usage"],
                    },
                    {
                        "question": "How can I check which tokens I have active?",
                        "answer": "Visit GitHub Settings > Developer Settings > Personal Access Tokens to view and manage your active tokens.",
                        "related_topics": ["management", "monitoring"],
                    },
                    {
                        "question": "What happens when a token expires?",
                        "answer": "When a token expires, any attempts to use it will result in authentication errors. You'll need to generate a new token and update any systems using the expired token.",
                        "related_topics": ["management", "troubleshooting"],
                    },
                ],
            },
        ],
        "additional_resources": [
            {
                "title": "GitHub Token Creation Guide",
                "uri": "guide://github-token-creation",
                "description": "Step-by-step guide for creating GitHub tokens",
            },
            {
                "title": "GitHub Token Troubleshooting Guide",
                "uri": "guide://github-token-troubleshooting",
                "description": "Solutions for common token issues",
            },
        ],
    }


@mcp.tool(
    name="authenticate",
    description="Authenticate with GitHub and store the token securely.",
)
@handle_rate_limit
def authenticate(token: str, user_id: str, ttl_hours: int = None) -> Dict[str, Any]:
    """
    Authenticate with GitHub and store the token.

    Args:
        token: GitHub personal access token
        user_id: Unique identifier for the user
        ttl_hours: Token time-to-live in hours (default: from config)

    Returns:
        Information about the authenticated user

    Raises:
        ValueError: If token is invalid or authentication fails
    """
    if not token or not isinstance(token, str):
        raise ValueError("Token must be a non-empty string")

    if not user_id or not isinstance(user_id, str):
        raise ValueError("User ID must be a non-empty string")

    try:
        # Validate token by getting user info
        github_client = get_github_client(token=token)
        user = github_client.get_user()

        # Store valid token
        token_manager.store_token(user_id, token, ttl_hours)

        return {
            "status": "authenticated",
            "user": {
                "login": user.login,
                "name": user.name,
                "email": user.email,
                "expires_in_hours": ttl_hours or config.TOKEN_TTL_HOURS,
            },
        }
    except ValueError as e:
        logger.error(f"Authentication failed for user {user_id}: {str(e)}")
        raise ValueError(f"GitHub authentication failed: {str(e)}")
    except GithubException as e:
        logger.error(f"GitHub API error during authentication: {str(e)}")
        raise ValueError(f"GitHub authentication failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during authentication: {str(e)}")
        raise ValueError("Authentication failed due to an unexpected error")


@mcp.tool(name="logout", description="Revoke stored GitHub authentication token.")
def logout(user_id: str) -> Dict[str, Any]:
    """
    Revoke stored GitHub token.

    Args:
        user_id: User identifier

    Returns:
        Status of the logout operation
    """
    success = token_manager.revoke_token(user_id)

    return {
        "status": "success" if success else "not_authenticated",
        "message": "Token revoked successfully" if success else "No active token found",
    }


@mcp.tool(
    name="check_auth_status",
    description="Check if user has a valid authentication token.",
)
def check_auth_status(user_id: str) -> Dict[str, Any]:
    """
    Check authentication status.

    Args:
        user_id: User identifier

    Returns:
        Authentication status information
    """
    token = token_manager.get_token(user_id)

    if token:
        try:
            # Validate token still works
            github_client = Github(token)
            user = github_client.get_user()

            return {
                "authenticated": True,
                "user": {"login": user.login, "name": user.name},
            }
        except GithubException:
            # Token no longer valid
            token_manager.revoke_token(user_id)
            return {
                "authenticated": False,
                "message": "Stored token is no longer valid",
            }
    else:
        return {"authenticated": False, "message": "No valid authentication found"}


# Get user info
@mcp.tool(
    name="get_user",
    description="Retrieve public GitHub profile information for a given username; if no username is provided, returns the authenticated user's info.",
)
def get_user(
    username: str = None, token: Optional[str] = None, user_id: Optional[str] = None
) -> dict:
    """
    Get user information.

    Args:
        username: GitHub username. If None, gets authenticated user.
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        Dict containing user information
    """
    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        if username:
            user = github_client.get_user(username)
        else:
            user = github_client.get_user()

        return {
            "id": user.id,
            "login": user.login,
            "name": user.name,
            "email": user.email,
            "bio": user.bio,
            "public_repos": user.public_repos,
            "followers": user.followers,
            "following": user.following,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "updated_at": user.updated_at.isoformat() if user.updated_at else None,
        }
    except ValueError as e:
        # Authentication error
        logger.error(f"Authentication error: {str(e)}")
        raise ValueError(str(e))
    except GithubException as e:
        logger.error(f"Error getting user {username}: {str(e)}")
        raise Exception(f"Failed to get user: {str(e)}")


# List all repositories
@mcp.tool(
    name="list_repositories",
    description="List all repositories for a given user.",
)
def list_repositories(
    username: str = None, token: Optional[str] = None, user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List repositories for a user.

    Args:
        username: GitHub username. If None, gets authenticated user's repos.
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        List of repository information dictionaries
    """
    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        if username:
            user = github_client.get_user(username)
            repos = user.get_repos()
        else:
            user = github_client.get_user()
            repos = user.get_repos()

        result = []
        for repo in repos:
            result.append(
                {
                    "id": repo.id,
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "description": repo.description,
                    "url": repo.html_url,
                    "language": repo.language,
                    "private": repo.private,
                    "fork": repo.fork,
                    "created_at": repo.created_at.isoformat()
                    if repo.created_at
                    else None,
                    "updated_at": repo.updated_at.isoformat()
                    if repo.updated_at
                    else None,
                    "stargazers_count": repo.stargazers_count,
                    "forks_count": repo.forks_count,
                }
            )

        return result
    except ValueError as e:
        # Authentication error
        logger.error(f"Authentication error: {str(e)}")
        raise ValueError(str(e))
    except GithubException as e:
        logger.error(f"Error listing repositories for {username}: {str(e)}")
        raise Exception(f"Failed to list repositories: {str(e)}")


@mcp.tool(
    name="get_repository",
    description="Get detailed information about a specific repository.",
)
def get_repository(
    repo_path: str, token: Optional[str] = None, user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get repository details.

    Args:
        repo_path: Repository path in format "owner/repo"
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        Dictionary with repository details
    """
    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        repo = github_client.get_repo(repo_path)

        return {
            "id": repo.id,
            "name": repo.name,
            "full_name": repo.full_name,
            "description": repo.description,
            "url": repo.html_url,
            "language": repo.language,
            "private": repo.private,
            "fork": repo.fork,
            "created_at": repo.created_at.isoformat() if repo.created_at else None,
            "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
            "pushed_at": repo.pushed_at.isoformat() if repo.pushed_at else None,
            "stargazers_count": repo.stargazers_count,
            "forks_count": repo.forks_count,
            "open_issues_count": repo.open_issues_count,
            "default_branch": repo.default_branch,
        }
    except ValueError as e:
        # Authentication error
        logger.error(f"Authentication error: {str(e)}")
        raise ValueError(str(e))
    except GithubException as e:
        logger.error(f"Error getting repository {repo_path}: {str(e)}")
        raise Exception(f"Failed to get repository: {str(e)}")


@mcp.tool(
    name="create_repository",
    description="Create a new GitHub repository for the authenticated user.",
)
@handle_rate_limit
def create_repository(
    name: str,
    description: Optional[str] = None,
    private: bool = False,
    has_issues: bool = True,
    has_wiki: bool = True,
    has_projects: bool = True,
    auto_init: bool = False,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new repository with customizable settings.

    Args:
        name: Repository name (required)
        description: Repository description
        private: Whether repository should be private
        has_issues: Enable issues feature
        has_wiki: Enable wiki feature
        has_projects: Enable projects feature
        auto_init: Auto-initialize with README
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        Dictionary with created repository details

    Raises:
        ValueError: If repository name is invalid or already exists
        Exception: On API errors or permission issues
    """
    # Validate repository name
    if not name or not isinstance(name, str):
        raise ValueError("Repository name must be a non-empty string")

    if "/" in name:
        raise ValueError("Repository name cannot contain '/'")

    try:
        # Get authenticated client
        client = get_github_client(token, user_id)
        user = client.get_user()

        # Check if repo already exists
        try:
            existing_repo = user.get_repo(name)
            if existing_repo:
                raise ValueError(f"Repository '{name}' already exists")
        except GithubException as e:
            if e.status != 404:
                raise

        repo = user.create_repo(
            name=name,
            description=description,
            private=private,
            has_issues=has_issues,
            has_wiki=has_wiki,
            has_projects=has_projects,
            auto_init=auto_init,
        )

        return {
            "id": repo.id,
            "name": repo.name,
            "full_name": repo.full_name,
            "description": repo.description,
            "url": repo.html_url,
            "private": repo.private,
            "created_at": repo.created_at.isoformat() if repo.created_at else None,
            "default_branch": repo.default_branch,
        }
    except ValueError as e:
        logger.error(f"Validation error creating repository {name}: {str(e)}")
        raise
    except GithubException as e:
        logger.error(f"GitHub API error creating repository {name}: {str(e)}")
        raise Exception(f"Failed to create repository: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error creating repository {name}: {str(e)}")
        raise Exception(f"Failed to create repository: {str(e)}")


@mcp.tool(name="list_branches", description="List all branches in a repository.")
def list_branches(
    repo_path: str, token: Optional[str] = None, user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List branches in a repository.

    Args:
        repo_path: Repository path in format "owner/repo"
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        List of branch information dictionaries
    """
    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        repo = github_client.get_repo(repo_path)
        branches = repo.get_branches()

        result = []
        for branch in branches:
            result.append(
                {
                    "name": branch.name,
                    "protected": branch.protected,
                    "commit": {
                        "sha": branch.commit.sha,
                        "url": branch.commit.html_url,
                    },
                }
            )

        return result
    except ValueError as e:
        # Authentication error
        logger.error(f"Authentication error: {str(e)}")
        raise ValueError(str(e))
    except GithubException as e:
        logger.error(f"Error listing branches for {repo_path}: {str(e)}")
        raise Exception(f"Failed to list branches: {str(e)}")


@mcp.tool(name="create_branch", description="Create a new branch in a repository.")
def create_branch(
    repo_path: str,
    branch_name: str,
    from_branch: Optional[str] = None,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new branch in a repository.

    Args:
        repo_path: Repository path in format "owner/repo"
        branch_name: New branch name
        from_branch: Source branch (default: repository's default branch)
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        Dictionary with branch details

    Raises:
        Exception: On API errors, invalid repository path, or branch conflicts
    """
    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        repo = github_client.get_repo(repo_path)
        source_branch = from_branch or repo.default_branch
        source_ref = repo.get_git_ref(f"heads/{source_branch}")

        # Check if branch already exists
        try:
            repo.get_git_ref(f"heads/{branch_name}")
            raise Exception(f"Branch '{branch_name}' already exists")
        except GithubException as e:
            if e.status != 404:
                raise

        # Create new branch from source
        repo.create_git_ref(f"refs/heads/{branch_name}", source_ref.object.sha)

        return {
            "name": branch_name,
            "sha": source_ref.object.sha,
            "source_branch": source_branch,
            "url": f"{repo.html_url}/tree/{branch_name}",
        }
    except ValueError as e:
        # Authentication error
        logger.error(f"Authentication error: {str(e)}")
        raise ValueError(str(e))
    except GithubException as e:
        logger.error(
            f"Error creating branch {branch_name} in {repo_path}: {str(e)}",
            exc_info=True,
        )
        raise Exception(f"Failed to create branch: {str(e)}")


@mcp.tool(
    name="get_file_content",
    description="Get content of a file from a repository.",
)
@handle_rate_limit
def get_file_content(
    repo_path: str,
    file_path: str,
    ref: Optional[str] = None,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get file content from a repository.

    Args:
        repo_path: Repository path in format "owner/repo"
        file_path: Path to file within repository
        ref: Branch, tag or commit SHA
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        Dictionary with file details and content

    Raises:
        ValueError: If repository path or file path is invalid
        Exception: If file is not found or is too large
    """
    # Validate inputs
    validate_repository_path(repo_path)
    if not file_path:
        raise ValueError("File path must be non-empty")

    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        repo = github_client.get_repo(repo_path)
        file_content = repo.get_contents(file_path, ref=ref)

        if isinstance(file_content, list):
            # This is a directory
            return {
                "type": "directory",
                "path": file_path,
                "items": [
                    {
                        "name": item.name,
                        "path": item.path,
                        "type": "file" if item.type == "file" else "directory",
                        "size": item.size if item.type == "file" else 0,
                        "sha": item.sha,
                    }
                    for item in file_content
                ],
            }
        else:
            # This is a file
            try:
                content = file_content.decoded_content.decode("utf-8")
            except UnicodeDecodeError:
                # Handle binary files
                return {
                    "type": "file",
                    "name": file_content.name,
                    "path": file_content.path,
                    "size": file_content.size,
                    "sha": file_content.sha,
                    "content": None,  # Binary content not included
                    "download_url": file_content.download_url,
                    "is_binary": True,
                }

            return {
                "type": "file",
                "name": file_content.name,
                "path": file_content.path,
                "size": file_content.size,
                "sha": file_content.sha,
                "content": content,
                "download_url": file_content.download_url,
                "is_binary": False,
            }
    except ValueError as e:
        logger.error(f"Validation error getting file {file_path}: {str(e)}")
        raise
    except GithubException as e:
        if e.status == 404:
            raise Exception(f"File {file_path} not found in repository {repo_path}")
        elif e.status == 403:
            raise Exception(f"File {file_path} is too large to retrieve")
        else:
            logger.error(f"GitHub API error getting file {file_path}: {str(e)}")
            raise Exception(f"Failed to get file content: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error getting file {file_path}: {str(e)}")
        raise Exception(f"Failed to get file content: {str(e)}")


@mcp.tool(
    name="create_or_update_file", description="Create or update a file in a repository."
)
def create_or_update_file(
    repo_path: str,
    file_path: str,
    message: str,
    content: str,
    branch: Optional[str] = None,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create or update a file in a repository.

    Args:
        repo_path: Repository path in format "owner/repo"
        file_path: Path to file within repository
        message: Commit message
        content: File content
        branch: Branch name (default: repository's default branch)
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        Dictionary with commit details
    """
    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        repo = github_client.get_repo(repo_path)
        branch = branch or repo.default_branch

        # Convert content to bytes if it's a string
        content_bytes = content.encode("utf-8") if isinstance(content, str) else content

        try:
            # Try to get file to update it
            contents = repo.get_contents(file_path, ref=branch)
            result = repo.update_file(
                path=file_path,
                message=message,
                content=content_bytes,
                sha=contents.sha,
                branch=branch,
            )
            operation = "updated"
        except GithubException:
            # File doesn't exist, create it
            result = repo.create_file(
                path=file_path, message=message, content=content_bytes, branch=branch
            )
            operation = "created"

        commit_result = result["commit"]
        return {
            "operation": operation,
            "commit": {
                "sha": commit_result.sha,
                "message": commit_result.message,
                "url": commit_result.html_url,
                "author": {
                    "name": commit_result.author.name,
                    "date": commit_result.author.date.isoformat()
                    if commit_result.author.date
                    else None,
                },
            },
            "content": {
                "name": file_path.split("/")[-1],
                "path": file_path,
                "sha": result["content"].sha if "content" in result else None,
            },
        }
    except ValueError as e:
        # Authentication error
        logger.error(f"Authentication error: {str(e)}")
        raise ValueError(str(e))
    except GithubException as e:
        logger.error(
            f"Error creating/updating file {file_path} in {repo_path}: {str(e)}"
        )
        raise Exception(f"Failed to create/update file: {str(e)}")


@mcp.tool(name="delete_file", description="Delete a file from a repository.")
def delete_file(
    repo_path: str,
    file_path: str,
    message: str,
    branch: Optional[str] = None,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Delete a file from a repository.

    Args:
        repo_path: Repository path in format "owner/repo"
        file_path: Path to file within repository
        message: Commit message
        branch: Branch name (default: repository's default branch)
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        Dictionary with commit details

    Raises:
        Exception: On API errors, file not found, or permission issues
    """
    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        repo = github_client.get_repo(repo_path)
        branch = branch or repo.default_branch

        # Get file to delete
        contents = repo.get_contents(file_path, ref=branch)
        if isinstance(contents, list):
            raise Exception(f"Path {file_path} is a directory, not a file")

        result = repo.delete_file(
            path=file_path, message=message, sha=contents.sha, branch=branch
        )

        commit_result = result["commit"]
        return {
            "operation": "deleted",
            "commit": {
                "sha": commit_result.sha,
                "message": commit_result.message,
                "url": commit_result.html_url,
                "author": {
                    "name": commit_result.author.name,
                    "date": commit_result.author.date.isoformat()
                    if commit_result.author.date
                    else None,
                },
            },
        }
    except ValueError as e:
        # Authentication error
        logger.error(f"Authentication error: {str(e)}")
        raise ValueError(str(e))
    except GithubException as e:
        logger.error(
            f"Error deleting file {file_path} in {repo_path}: {str(e)}", exc_info=True
        )
        raise Exception(f"Failed to delete file: {str(e)}")


@mcp.tool(name="list_issues", description="List issues in a repository.")
def list_issues(
    repo_path: str,
    state: str = "open",
    labels: Optional[List[str]] = None,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List issues in a repository.

    Args:
        repo_path: Repository path in format "owner/repo"
        state: Issue state (open, closed, all)
        labels: List of label names
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        List of issue information dictionaries
    """
    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        repo = github_client.get_repo(repo_path)
        issues = repo.get_issues(state=state, labels=labels)

        result = []
        for issue in issues:
            result.append(
                {
                    "id": issue.id,
                    "number": issue.number,
                    "title": issue.title,
                    "state": issue.state,
                    "created_at": issue.created_at.isoformat()
                    if issue.created_at
                    else None,
                    "updated_at": issue.updated_at.isoformat()
                    if issue.updated_at
                    else None,
                    "closed_at": issue.closed_at.isoformat()
                    if issue.closed_at
                    else None,
                    "body": issue.body,
                    "user": {
                        "login": issue.user.login,
                        "id": issue.user.id,
                    },
                    "labels": [label.name for label in issue.labels],
                    "assignees": [assignee.login for assignee in issue.assignees],
                    "comments": issue.comments,
                }
            )

        return result
    except ValueError as e:
        # Authentication error
        logger.error(f"Authentication error: {str(e)}")
        raise ValueError(str(e))
    except GithubException as e:
        logger.error(f"Error listing issues for {repo_path}: {str(e)}")
        raise Exception(f"Failed to list issues: {str(e)}")


@mcp.tool(
    name="create_issue",
    description="Create a new issue in a repository.",
)
@handle_rate_limit
def create_issue(
    repo_path: str,
    title: str,
    body: Optional[str] = None,
    labels: Optional[List[str]] = None,
    assignees: Optional[List[str]] = None,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new issue in a repository.

    Args:
        repo_path: Repository path in format "owner/repo"
        title: Issue title (required)
        body: Issue description
        labels: List of label names
        assignees: List of assignee usernames
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        Dictionary with created issue details

    Raises:
        ValueError: If required parameters are invalid
        Exception: If labels/assignees don't exist or other API errors
    """
    # Validate inputs
    validate_repository_path(repo_path)

    if not title or not isinstance(title, str):
        raise ValueError("Issue title must be a non-empty string")

    if labels is not None and not isinstance(labels, list):
        raise ValueError("Labels must be a list of strings")

    if assignees is not None and not isinstance(assignees, list):
        raise ValueError("Assignees must be a list of strings")

    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)
        repo = github_client.get_repo(repo_path)

        # Validate labels exist
        if labels:
            valid_labels = {label.name for label in repo.get_labels()}
            invalid_labels = set(labels) - valid_labels
            if invalid_labels:
                raise ValueError(f"Labels do not exist: {', '.join(invalid_labels)}")

        # Validate assignees exist
        if assignees:
            for assignee in assignees:
                try:
                    github_client.get_user(assignee)
                except GithubException:
                    raise ValueError(f"User '{assignee}' does not exist")

        issue = repo.create_issue(
            title=title, body=body, labels=labels, assignees=assignees
        )

        return {
            "id": issue.id,
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "created_at": issue.created_at.isoformat() if issue.created_at else None,
            "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
            "body": issue.body,
            "user": {
                "login": issue.user.login,
                "id": issue.user.id,
            },
            "labels": [label.name for label in issue.labels],
            "assignees": [assignee.login for assignee in issue.assignees],
            "url": issue.html_url,
        }
    except ValueError as e:
        logger.error(f"Validation error creating issue in {repo_path}: {str(e)}")
        raise
    except GithubException as e:
        logger.error(f"GitHub API error creating issue in {repo_path}: {str(e)}")
        raise Exception(f"Failed to create issue: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error creating issue in {repo_path}: {str(e)}")
        raise Exception(f"Failed to create issue: {str(e)}")


@mcp.tool(name="update_issue", description="Update an existing issue in a repository.")
def update_issue(
    repo_path: str,
    issue_number: int,
    title: Optional[str] = None,
    body: Optional[str] = None,
    state: Optional[str] = None,
    labels: Optional[List[str]] = None,
    assignees: Optional[List[str]] = None,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update an existing issue in a repository.

    Args:
        repo_path: Repository path in format "owner/repo"
        issue_number: Issue number
        title: New issue title
        body: New issue body
        state: New state ('open' or 'closed')
        labels: New list of label names
        assignees: New list of assignee usernames
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        Dictionary with updated issue details

    Raises:
        Exception: On API errors, issue not found, or permission issues
    """
    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        repo = github_client.get_repo(repo_path)
        issue = repo.get_issue(issue_number)

        update_kwargs = {}
        if title is not None:
            update_kwargs["title"] = title
        if body is not None:
            update_kwargs["body"] = body
        if state is not None:
            update_kwargs["state"] = state

        # Update the issue with available fields
        if update_kwargs:
            issue.edit(**update_kwargs)

        # Update labels if provided
        if labels is not None:
            issue.set_labels(*labels)

        # Update assignees if provided
        if assignees is not None:
            issue.set_assignees(*assignees)

        # Refresh issue data
        issue = repo.get_issue(issue_number)

        return {
            "id": issue.id,
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
            "body": issue.body,
            "labels": [label.name for label in issue.labels],
            "assignees": [assignee.login for assignee in issue.assignees],
            "url": issue.html_url,
        }
    except ValueError as e:
        # Authentication error
        logger.error(f"Authentication error: {str(e)}")
        raise ValueError(str(e))
    except GithubException as e:
        logger.error(
            f"Error updating issue #{issue_number} in {repo_path}: {str(e)}",
            exc_info=True,
        )
        raise Exception(f"Failed to update issue: {str(e)}")


@mcp.tool(name="list_pull_requests", description="List pull requests in a repository.")
def list_pull_requests(
    repo_path: str,
    state: str = "open",
    base: Optional[str] = None,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List pull requests in a repository.

    Args:
        repo_path: Repository path in format "owner/repo"
        state: PR state (open, closed, all)
        base: Base branch name filter
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        List of pull request information dictionaries
    """
    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        repo = github_client.get_repo(repo_path)
        pulls = repo.get_pulls(state=state, base=base)

        result = []
        for pr in pulls:
            result.append(
                {
                    "id": pr.id,
                    "number": pr.number,
                    "title": pr.title,
                    "state": pr.state,
                    "created_at": pr.created_at.isoformat() if pr.created_at else None,
                    "updated_at": pr.updated_at.isoformat() if pr.updated_at else None,
                    "closed_at": pr.closed_at.isoformat() if pr.closed_at else None,
                    "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
                    "body": pr.body,
                    "user": {
                        "login": pr.user.login,
                        "id": pr.user.id,
                    },
                    "base": {
                        "ref": pr.base.ref,
                        "sha": pr.base.sha,
                    },
                    "head": {
                        "ref": pr.head.ref,
                        "sha": pr.head.sha,
                    },
                    "mergeable": pr.mergeable,
                    "merged": pr.merged,
                }
            )

        return result
    except ValueError as e:
        # Authentication error
        logger.error(f"Authentication error: {str(e)}")
        raise ValueError(str(e))
    except GithubException as e:
        logger.error(f"Error listing pull requests for {repo_path}: {str(e)}")
        raise Exception(f"Failed to list pull requests: {str(e)}")


@mcp.tool(
    name="create_pull_request",
    description="Create a new pull request in a repository.",
)
@handle_rate_limit
def create_pull_request(
    repo_path: str,
    title: str,
    head: str,
    base: str,
    body: Optional[str] = None,
    draft: bool = False,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new pull request in a repository.

    Args:
        repo_path: Repository path in format "owner/repo"
        title: PR title (required)
        head: Head branch (required)
        base: Base branch (required)
        body: PR description
        draft: Whether PR is a draft
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        Dictionary with created PR details

    Raises:
        ValueError: If required parameters are invalid
        Exception: If branches don't exist or other API errors
    """
    # Validate inputs
    validate_repository_path(repo_path)

    if not title or not isinstance(title, str):
        raise ValueError("Pull request title must be a non-empty string")

    if not head or not isinstance(head, str):
        raise ValueError("Head branch must be a non-empty string")

    if not base or not isinstance(base, str):
        raise ValueError("Base branch must be a non-empty string")

    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)
        repo = github_client.get_repo(repo_path)

        # Validate branches exist
        try:
            repo.get_branch(head)
        except GithubException:
            raise ValueError(f"Head branch '{head}' does not exist")

        try:
            repo.get_branch(base)
        except GithubException:
            raise ValueError(f"Base branch '{base}' does not exist")

        # Check if PR already exists
        existing_prs = repo.get_pulls(state="open", head=head, base=base)
        for pr in existing_prs:
            if pr.head.ref == head and pr.base.ref == base:
                raise ValueError(
                    f"A pull request already exists for {head} into {base}"
                )

        pr = repo.create_pull(title=title, body=body, head=head, base=base, draft=draft)

        return {
            "id": pr.id,
            "number": pr.number,
            "title": pr.title,
            "state": pr.state,
            "created_at": pr.created_at.isoformat() if pr.created_at else None,
            "body": pr.body,
            "user": {
                "login": pr.user.login,
                "id": pr.user.id,
            },
            "base": {
                "ref": pr.base.ref,
                "sha": pr.base.sha,
            },
            "head": {
                "ref": pr.head.ref,
                "sha": pr.head.sha,
            },
            "url": pr.html_url,
            "draft": pr.draft,
            "mergeable": pr.mergeable,
            "mergeable_state": pr.mergeable_state,
        }
    except ValueError as e:
        logger.error(f"Validation error creating PR in {repo_path}: {str(e)}")
        raise
    except GithubException as e:
        logger.error(f"GitHub API error creating PR in {repo_path}: {str(e)}")
        raise Exception(f"Failed to create pull request: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error creating PR in {repo_path}: {str(e)}")
        raise Exception(f"Failed to create pull request: {str(e)}")


@mcp.tool(
    name="merge_pull_request",
    description="Merge a pull request with various merge strategies.",
)
@handle_rate_limit
def merge_pull_request(
    repo_path: str,
    pr_number: int,
    commit_message: Optional[str] = None,
    merge_method: str = "merge",
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Merge a pull request with specified merge strategy.

    Args:
        repo_path: Repository path in format "owner/repo"
        pr_number: Pull request number
        commit_message: Custom merge commit message
        merge_method: Merge method to use ('merge', 'squash', or 'rebase')
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        Dictionary with merge status details

    Raises:
        ValueError: If PR number or merge method is invalid
        Exception: If PR can't be merged or other API errors
    """
    # Validate inputs
    validate_repository_path(repo_path)

    if not isinstance(pr_number, int) or pr_number <= 0:
        raise ValueError("Pull request number must be a positive integer")

    if merge_method not in ["merge", "squash", "rebase"]:
        raise ValueError(
            f"Invalid merge method: {merge_method}. Must be 'merge', 'squash', or 'rebase'"
        )

    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        # Get repository and pull request objects
        repo = github_client.get_repo(repo_path)
        pr = repo.get_pull(pr_number)

        # Validate PR state
        if pr.state != "open":
            raise ValueError(f"Pull request #{pr_number} is not open")

        # Check if PR can be merged
        if not pr.mergeable:
            if pr.mergeable_state == "behind":
                raise ValueError(
                    f"Pull request #{pr_number} is not up to date with base branch"
                )
            elif pr.mergeable_state == "dirty":
                raise ValueError(
                    f"Pull request #{pr_number} has conflicts that need to be resolved"
                )
            else:
                raise ValueError(
                    f"Pull request #{pr_number} cannot be merged (state: {pr.mergeable_state})"
                )

        # Check if required status checks have passed
        if not all(
            status.state == "success"
            for status in pr.get_commits().reversed[0].get_statuses()
        ):
            raise ValueError(
                f"Pull request #{pr_number} has pending or failed status checks"
            )

        # Perform the merge with specified strategy
        merge_result = pr.merge(
            commit_message=commit_message, merge_method=merge_method
        )

        # Return merge result details
        return {
            "merged": merge_result.merged,
            "message": merge_result.message,
            "sha": merge_result.sha if merge_result.merged else None,
            "pr_number": pr_number,
            "merge_method": merge_method,
            "title": pr.title,
            "merged_by": {
                "login": merge_result.merged_by.login
                if merge_result.merged_by
                else None
            }
            if merge_result.merged
            else None,
        }
    except ValueError as e:
        logger.error(f"Validation error merging PR #{pr_number}: {str(e)}")
        raise
    except GithubException as e:
        logger.error(f"GitHub API error merging PR #{pr_number}: {str(e)}")
        raise Exception(f"Failed to merge pull request: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error merging PR #{pr_number}: {str(e)}")
        raise Exception(f"Failed to merge pull request: {str(e)}")


@mcp.tool(
    name="list_commits",
    description="Get commits from a repository with optional filtering.",
)
def list_commits(
    repo_path: str,
    branch: Optional[str] = None,
    path: Optional[str] = None,
    author: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    max_results: int = 30,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get commits from a repository with flexible filtering options.

    Args:
        repo_path: Repository path in format "owner/repo"
        branch: Branch name to filter commits
        path: Filter by file path
        author: Filter by author name or email
        since: Filter commits after this date (ISO format: YYYY-MM-DD)
        until: Filter commits before this date (ISO format: YYYY-MM-DD)
        max_results: Maximum number of commits to return
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        List of commit information dictionaries
    """
    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        # Parse date strings to datetime objects if provided
        since_date = None
        if since:
            try:
                from datetime import datetime

                since_date = datetime.fromisoformat(since)
            except ValueError:
                logger.warning(
                    f"Invalid 'since' date format: {since}. Using ISO format YYYY-MM-DD."
                )
                raise ValueError("Invalid date format. Use ISO format YYYY-MM-DD.")

        until_date = None
        if until:
            try:
                from datetime import datetime

                until_date = datetime.fromisoformat(until)
            except ValueError:
                logger.warning(
                    f"Invalid 'until' date format: {until}. Using ISO format YYYY-MM-DD."
                )
                raise ValueError("Invalid date format. Use ISO format YYYY-MM-DD.")

        # Get repository object and fetch commits with filters
        repo = github_client.get_repo(repo_path)
        commits = repo.get_commits(
            sha=branch, path=path, author=author, since=since_date, until=until_date
        )

        # Process results with pagination
        result = []
        for commit in commits[:max_results]:
            # Extract relevant commit information
            commit_data = {
                "sha": commit.sha,
                "message": commit.commit.message,
                "url": commit.html_url,
                "date": commit.commit.author.date.isoformat()
                if commit.commit.author.date
                else None,
                "author": {
                    "name": commit.commit.author.name,
                    "email": commit.commit.author.email,
                    "username": commit.author.login if commit.author else None,
                },
            }

            # Add statistics if available
            if hasattr(commit, "stats") and commit.stats:
                commit_data["stats"] = {
                    "additions": commit.stats.additions,
                    "deletions": commit.stats.deletions,
                    "total": commit.stats.total,
                }

            # Add files information if available
            if hasattr(commit, "files") and commit.files:
                commit_data["files"] = [
                    {
                        "filename": f.filename,
                        "additions": f.additions,
                        "deletions": f.deletions,
                        "changes": f.changes,
                        "status": f.status,
                    }
                    for f in commit.files
                ]

            result.append(commit_data)

        return result
    except ValueError as e:
        # Authentication error or date format error
        logger.error(f"Error: {str(e)}")
        raise ValueError(str(e))
    except GithubException as e:
        logger.error(f"Error getting commits for {repo_path}: {str(e)}")
        raise Exception(f"Failed to get commits: {str(e)}")


@mcp.tool(
    name="get_commit", description="Get detailed information about a specific commit."
)
def get_commit(
    repo_path: str,
    commit_sha: str,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get detailed information about a specific commit.

    Args:
        repo_path: Repository path in format "owner/repo"
        commit_sha: SHA hash of the commit
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        Dictionary with detailed commit information
    """
    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        repo = github_client.get_repo(repo_path)
        commit = repo.get_commit(commit_sha)

        # Compile detailed commit information
        result = {
            "sha": commit.sha,
            "message": commit.commit.message,
            "url": commit.html_url,
            "date": commit.commit.author.date.isoformat()
            if commit.commit.author.date
            else None,
            "author": {
                "name": commit.commit.author.name,
                "email": commit.commit.author.email,
                "username": commit.author.login if commit.author else None,
            },
            "committer": {
                "name": commit.commit.committer.name,
                "email": commit.commit.committer.email,
                "username": commit.committer.login if commit.committer else None,
            },
            "stats": {
                "additions": commit.stats.additions,
                "deletions": commit.stats.deletions,
                "total": commit.stats.total,
            },
            "files": [
                {
                    "filename": f.filename,
                    "additions": f.additions,
                    "deletions": f.deletions,
                    "changes": f.changes,
                    "status": f.status,
                    "patch": f.patch if hasattr(f, "patch") else None,
                }
                for f in commit.files
            ],
            "parents": [{"sha": p.sha, "url": p.html_url} for p in commit.parents],
        }

        return result
    except ValueError as e:
        # Authentication error
        logger.error(f"Authentication error: {str(e)}")
        raise ValueError(str(e))
    except GithubException as e:
        logger.error(f"Error getting commit {commit_sha} in {repo_path}: {str(e)}")
        raise Exception(f"Failed to get commit: {str(e)}")


@mcp.tool(
    name="compare_commits",
    description="Compare two commits or branches to see differences.",
)
def compare_commits(
    repo_path: str,
    base: str,
    head: str,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare two commits, branches, or tags to see differences.

    Args:
        repo_path: Repository path in format "owner/repo"
        base: Base commit SHA, branch or tag name
        head: Head commit SHA, branch or tag name
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        Dictionary with comparison information
    """
    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        repo = github_client.get_repo(repo_path)
        comparison = repo.compare(base, head)

        # Compile detailed comparison information
        result = {
            "total_commits": comparison.total_commits,
            "ahead_by": comparison.ahead_by,
            "behind_by": comparison.behind_by,
            "status": comparison.status,
            "diff_url": comparison.permalink_url,
            "commits": [
                {
                    "sha": commit.sha,
                    "message": commit.commit.message,
                    "author": {
                        "name": commit.commit.author.name,
                        "date": commit.commit.author.date.isoformat()
                        if commit.commit.author.date
                        else None,
                    },
                }
                for commit in comparison.commits
            ],
            "files": [
                {
                    "filename": f.filename,
                    "status": f.status,
                    "additions": f.additions,
                    "deletions": f.deletions,
                    "changes": f.changes,
                    "patch": f.patch if hasattr(f, "patch") else None,
                }
                for f in comparison.files
            ],
        }

        return result
    except ValueError as e:
        # Authentication error
        logger.error(f"Authentication error: {str(e)}")
        raise ValueError(str(e))
    except GithubException as e:
        logger.error(f"Error comparing {base}...{head} in {repo_path}: {str(e)}")
        raise Exception(f"Failed to compare commits: {str(e)}")


@mcp.tool(
    name="create_comment", description="Create a comment on an issue or pull request."
)
def create_comment(
    repo_path: str,
    issue_number: int,
    body: str,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a comment on an issue or pull request.

    Args:
        repo_path: Repository path in format "owner/repo"
        issue_number: Issue or pull request number
        body: Comment text
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        Dictionary with created comment details
    """
    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        repo = github_client.get_repo(repo_path)
        issue = repo.get_issue(issue_number)
        comment = issue.create_comment(body)

        return {
            "id": comment.id,
            "body": comment.body,
            "created_at": comment.created_at.isoformat()
            if comment.created_at
            else None,
            "updated_at": comment.updated_at.isoformat()
            if comment.updated_at
            else None,
            "user": {"login": comment.user.login, "id": comment.user.id},
            "url": comment.html_url,
        }
    except ValueError as e:
        # Authentication error
        logger.error(f"Authentication error: {str(e)}")
        raise ValueError(str(e))
    except GithubException as e:
        logger.error(
            f"Error creating comment on #{issue_number} in {repo_path}: {str(e)}"
        )
        raise Exception(f"Failed to create comment: {str(e)}")


@mcp.tool(
    name="list_comments", description="List comments on an issue or pull request."
)
def list_comments(
    repo_path: str,
    issue_number: int,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List comments on an issue or pull request.

    Args:
        repo_path: Repository path in format "owner/repo"
        issue_number: Issue or pull request number
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        List of comment information dictionaries
    """
    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        repo = github_client.get_repo(repo_path)
        issue = repo.get_issue(issue_number)
        comments = issue.get_comments()

        result = []
        for comment in comments:
            result.append(
                {
                    "id": comment.id,
                    "body": comment.body,
                    "created_at": comment.created_at.isoformat()
                    if comment.created_at
                    else None,
                    "updated_at": comment.updated_at.isoformat()
                    if comment.updated_at
                    else None,
                    "user": {"login": comment.user.login, "id": comment.user.id},
                    "url": comment.html_url,
                }
            )

        return result
    except ValueError as e:
        # Authentication error
        logger.error(f"Authentication error: {str(e)}")
        raise ValueError(str(e))
    except GithubException as e:
        logger.error(
            f"Error listing comments on #{issue_number} in {repo_path}: {str(e)}"
        )
        raise Exception(f"Failed to list comments: {str(e)}")


@mcp.tool(
    name="search_repositories",
    description="Search for repositories with specified criteria.",
)
def search_repositories(
    query: str,
    sort: Optional[str] = "stars",
    order: str = "desc",
    per_page: int = 10,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search for repositories using GitHub's search API.

    Args:
        query: Search query string
        sort: Sort results by (stars, forks, updated)
        order: Sort order (asc or desc)
        per_page: Number of results per page (max 100)
        token: GitHub token (optional, overrides stored token)
        user_id: User ID for using stored token

    Returns:
        List of repository information dictionaries
    """
    try:
        # Get authenticated client
        github_client = get_github_client(token, user_id)

        # Validate input parameters
        if sort not in [None, "stars", "forks", "updated", "help-wanted-issues"]:
            logger.warning(f"Invalid sort parameter: {sort}")
            sort = "stars"

        if order not in ["asc", "desc"]:
            logger.warning(f"Invalid order parameter: {order}")
            order = "desc"

        if per_page > 100:
            logger.warning(
                f"per_page value is {per_page} which exceeds maximum (100), setting to 100"
            )
            per_page = 100

        # Execute search
        repositories = github_client.search_repositories(
            query=query, sort=sort, order=order
        )

        result = []
        for repo in repositories[:per_page]:
            result.append(
                {
                    "id": repo.id,
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "description": repo.description,
                    "url": repo.html_url,
                    "language": repo.language,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "watchers": repo.watchers_count,
                    "open_issues": repo.open_issues_count,
                    "created_at": repo.created_at.isoformat()
                    if repo.created_at
                    else None,
                    "updated_at": repo.updated_at.isoformat()
                    if repo.updated_at
                    else None,
                    "topics": repo.get_topics() if hasattr(repo, "get_topics") else [],
                    "owner": {
                        "login": repo.owner.login,
                        "id": repo.owner.id,
                        "url": repo.owner.html_url,
                    },
                }
            )

        return result
    except ValueError as e:
        # Authentication error
        logger.error(f"Authentication error: {str(e)}")
        raise ValueError(str(e))
    except GithubException as e:
        logger.error(f"Error searching repositories with query '{query}': {str(e)}")
        raise Exception(f"Failed to search repositories: {str(e)}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    )

    # Run MCP server with stdio transport
    logger.info("Starting GitHub Service MCP server")
    mcp.run(transport="stdio")
