# Pragma

**Historical GitLab MR Database with Semantic Search API**

Pragma indexes your GitLab merge requests and provides a REST API for AI assistants to query for historical context. Instead of doing code reviews locally, it acts as a **searchable knowledge base** that external tools (Claude Code, Gemini, etc.) can query to get institutional knowledge from past code changes.

## Features

- 🔍 **Semantic Search**: Find similar historical MRs using vector similarity
- 📊 **Rich Context**: Indexes titles, descriptions, diffs, and discussions
- 🚀 **REST API**: Simple HTTP endpoints for external tools
- 🔒 **Secure by Default**: Localhost-only binding (127.0.0.1)
- 🛠️ **GitLab Integration**: Fetches MRs via python-gitlab API
- 💾 **ChromaDB**: Local vector database for fast retrieval

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Initialize configuration
uv run pragma init

# 3. Edit config.yaml with your GitLab repository details

# 4. Test connection
uv run pragma test-connection

# 5. Index historical MRs
uv run pragma index

# 6. Start API server
uv run pragma serve
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Usage

### CLI Commands

```bash
# Initialize and configure
uv run pragma init

# Test GitLab connection
uv run pragma test-connection

# Index merge requests
uv run pragma index

# Start API server (localhost only, secure)
uv run pragma serve

# Start with custom port
uv run pragma serve --port 8080

# Development mode with auto-reload
uv run pragma serve --reload
```

### API Endpoints

- **POST /search** - Semantic search for similar MRs
- **GET /mrs/{mr_id}** - Get full details of specific MR
- **GET /mrs** - List all indexed MRs
- **GET /stats** - Database statistics
- **GET /health** - Health check

### Example: Search for Similar MRs

```python
import requests

response = requests.post("http://localhost:8000/search", json={
    "code_diff": "your code diff here",
    "top_k": 5,
    "min_score": 0.5
})

similar_mrs = response.json()
for mr in similar_mrs:
    print(f"!{mr['mr_id']}: {mr['mr_title']} (score: {mr['similarity_score']:.4f})")
```

## MCP Server Integration

Pragma includes an MCP server (`src/mcp_server.py`) exposing three tools:

| Tool | Description |
|------|-------------|
| `mcp__pragma__search` | Semantic search over historical MRs |
| `mcp__pragma__get_mr` | Get full details of a specific MR |
| `mcp__pragma__list_mrs` | List all indexed MRs |

### Setup

Add to your MCP client configuration (e.g. Claude Code `~/.claude.json`):

```json
{
  "mcpServers": {
    "pragma": {
      "command": "uv",
      "args": [
        "run",
        "--project", "/path/to/pragma",
        "python",
        "/path/to/pragma/src/mcp_server.py"
      ],
      "env": {
        "PRAGMA_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

The `--project` flag ensures pragma's virtual environment is used regardless of the current working directory.

### Example Prompts

**Find team decisions about a topic:**
```
Search pragma discussions for "API rate limiting strategy" and summarize
what the team decided in past MRs.
```

**Review a change with historical context:**
```
Before reviewing my change, search pragma for:
1. Discussions about similar changes to this component
2. Diffs with similar code patterns

Use both to provide context-aware feedback.
```

**The two-query pattern** — for the richest context, always make two calls to `search`:
- `content_type: "discussion"` + natural language query → finds *why* decisions were made
- `content_type: "diff"` + the actual code diff → finds *how* similar changes were implemented

See [CLAUDE.md](./CLAUDE.md) for full parameter reference and advanced usage.

## Architecture

```
GitLab API → Index MRs → Gemini Embeddings → ChromaDB
                                                  ↓
External AI Tool → HTTP API → Vector Search → Historical Context
```

## Configuration

Edit `config.yaml`:

```yaml
api_keys:
  gemini: YOUR_GEMINI_API_KEY
  gitlab: YOUR_GITLAB_PRIVATE_TOKEN

gitlab:
  base_url: https://gitlab.cee.redhat.com  # Optional

repository:
  type: gitlab
  owner: product-security/pdm  # Can be nested group
  name: pdm-db

vector_store:
  type: chromadb
  path: ./data/chroma_db
```

## Security

- ⚠️ **No authentication** - API is unauthenticated
- 🔒 **Localhost only** - Defaults to 127.0.0.1 for security
- 📁 **Sensitive data** - MR diffs may contain proprietary code
- 🚫 **Do not expose** to public networks without firewall/VPN

## Development

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run linting manually
uv run pre-commit run --all-files

# Install new dependency
uv add <package-name>
```

## Requirements

- Python 3.10-3.12
- Gemini API key
- GitLab private token with API read access
- `uv` package manager

## License

MIT

## Contributing

Contributions welcome! Please ensure:
- Pre-commit hooks pass
- API endpoints are documented
- Security best practices followed
