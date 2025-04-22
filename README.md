# GIT-Pilot 🚀

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-GIT--Pilot-blue)](https://github.com/yourusername/GIT-Pilot)

GIT-Pilot is a powerful GitHub automation and management tool that provides a comprehensive API wrapper for GitHub operations. It simplifies GitHub interactions through a FastMCP-based server, making it easy to manage repositories, pull requests, issues, and more.

## 🌟 Features

### 🔐 Authentication & Security
- Secure token management with encryption using Fernet
- Token expiration and automatic cleanup
- Rate limit handling and automatic retries
- Configurable authentication timeouts

### 📦 Repository Management
- Create and manage repositories
- Handle branches and commits
- File operations (create, update, delete)
- Repository search and filtering
- Commit comparison and history

### 🔄 Pull Request Operations
- Create and manage pull requests
- Merge strategies (merge, squash, rebase)
- Status check validation
- Conflict detection and handling
- Draft PR support

### 📝 Issue Management
- Create and update issues
- Label management
- Assignee handling
- Comment management
- Issue search and filtering

### 🛠 Technical Features
- Thread-safe operations
- Resource management
- Comprehensive error handling
- Detailed logging
- Type safety
- Configuration management
- FastMCP server integration

## 🚀 Getting Started

### Prerequisites
- Python 3.12 or higher
- GitHub account
- GitHub Personal Access Token
- FastMCP CLI (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GIT-Pilot.git
cd GIT-Pilot
```

2. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. Install the package using uv:
```bash
uv pip install -e .
```

5. Set up environment variables:
Create a `.env` file in the project root:
```env
GITHUB_TOKEN=your_github_token_here
```

### API Examples

```python
# Create a repository
await call_tool("create_repository",
    name="my-repo",
    description="My awesome repository",
    private=True,
    has_issues=True,
    has_wiki=True,
    has_projects=True,
    auto_init=True
)

# Create a pull request
await call_tool("create_pull_request",
    repo_path="owner/repo",
    title="New feature",
    head="feature-branch",
    base="main",
    body="Description of changes",
    draft=False
)

# List commits with filtering
await call_tool("list_commits",
    repo_path="owner/repo",
    branch="main",
    author="username",
    since="2024-01-01",
    until="2024-04-21",
    max_results=30
)
```

## 🔧 Configuration

The service can be configured through the `Config` class:

```python
@dataclass
class Config:
    TOKEN_TTL_HOURS: int = 24
    MAX_STORED_TOKENS: int = 1000
    CLEANUP_INTERVAL_SECONDS: int = 3600
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: int = 5
    MAX_RESULTS_PER_PAGE: int = 100
    ENCRYPTION_KEY: bytes = Fernet.generate_key()
```

## 🛡 Security

- Tokens are encrypted at rest using Fernet
- Automatic token expiration and cleanup
- Rate limit protection with retries
- Input validation
- Comprehensive error handling
- Secure token cleanup

## 🔄 Rate Limiting

The service includes built-in rate limit handling:
- Automatic retry on rate limit
- Configurable retry attempts
- Delay between retries
- Rate limit status logging
- Exponential backoff

## 🧪 Error Handling

Comprehensive error handling for:
- Authentication failures
- API errors
- Rate limits
- Invalid inputs
- Resource conflicts
- Network issues
- Token validation
- File operations

## 📈 Logging

Detailed logging with:
- Timestamp
- Log level
- Function name
- Line number
- Error details
- Stack traces
- Rate limit information
- Token operations

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyGithub](https://github.com/PyGithub/PyGithub) for the GitHub API wrapper
- [FastMCP](https://gofastmcp.com) for the server framework
- [Fernet](https://cryptography.io/en/latest/fernet/) for secure token encryption

## 📞 Support

- GitHub Issues: [Report a bug](https://github.com/yourusername/GIT-Pilot/issues)
- Documentation: [Wiki](https://github.com/yourusername/GIT-Pilot/wiki)
- Community: [Discussions](https://github.com/yourusername/GIT-Pilot/discussions)

## 🔄 Updates

Stay updated with the latest changes:
- Follow our [release notes](https://github.com/yourusername/GIT-Pilot/releases)
- Subscribe to our [newsletter](https://github.com/yourusername/GIT-Pilot/discussions/categories/announcements)
- Star the repository for updates

---

Made with ❤️ by the GIT-Pilot team
