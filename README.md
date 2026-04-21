# Pragma

**Historical GitLab MR Database with Semantic Search API**

Pragma indexes your GitLab merge requests and provides a REST API and an **MCP server** for AI assistants to query for historical context. Instead of doing code reviews locally, it acts as a **searchable knowledge base** that external tools (Claude Code, Gemini, etc.) can query to get institutional knowledge from past code changes, discussions, and decisions.

Connect Pragma to your AI assistant via MCP and ask questions like:

```
Search pragma discussions for "authentication strategy" and summarize what
the team decided in past MRs.
```

## Features

- **Semantic Search**: Find similar historical MRs using vector similarity
- **Rich Context**: Indexes titles, descriptions, diffs, and discussions
- **REST API**: Simple HTTP endpoints for external tools
- **AI Code Reviews**: Automated MR reviews using Gemini or Ollama models
- **GitLab Integration**: Fetches MRs via python-gitlab API
- **ChromaDB**: Local vector database for fast retrieval

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

# Index merge requests (all configured repos)
uv run pragma index

# Index a specific repository only
uv run pragma index --repo my-repo

# Start API server
uv run pragma serve

# Start with custom port
uv run pragma serve --port 8080

# Development mode with auto-reload
uv run pragma serve --reload

# Review a specific MR by IID
uv run pragma review 42

# Review an MR from a specific repo (overrides config)
uv run pragma review 42 --repo my-repo

# Continuously review new open MRs
uv run pragma review-watch --interval 60
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
GitLab API → Index MRs → Embedding Model → ChromaDB
                                               ↓
External AI Tool → HTTP API → Vector Search → Historical Context
```

## Configuration

Set credentials as environment variables (never stored in files):

```bash
export GEMINI_API_KEY=your_gemini_api_key
export GITLAB_PRIVATE_TOKEN=your_gitlab_token
```

Edit `config.yaml` for non-sensitive settings:

```yaml
gitlab:
  base_url: https://gitlab.example.com  # Optional, defaults to gitlab.com

repository:
  type: gitlab
  owner: my-group/my-subgroup  # Can be nested group
  name: my-repo

vector_store:
  type: chromadb
  path: ./data/chroma_db

# Embedding model configuration
# Both indexing and search must use the same provider.
# Switching provider requires re-indexing all MRs.
embeddings:
  provider: gemini        # "gemini" (default) or "local"
  # model: BAAI/bge-large-en-v1.5  # local only, this is the default
```

### Reviewer agent

Configure the AI model used for automated MR reviews:

```yaml
agent:
  provider: gemini          # "gemini" (default) or "ollama"
  model: gemini-2.5-flash   # Gemini model name

  # For Ollama (local models):
  # provider: ollama
  # model: mistral:7b        # Ollama model tag; openai: prefix added automatically
  # base_url: http://localhost:11434/v1  # optional, this is the default
```

**Gemini**: requires `GEMINI_API_KEY`.

**Ollama**: requires a running Ollama instance. Set `OPENAI_BASE_URL` and `OPENAI_API_KEY=ollama` in the environment file (see systemd section below).

Reviews are saved as Markdown files under `./data/reviews/`.

**Gemini embedding provider** (default): requires `GEMINI_API_KEY` in the environment.

**Local provider**: runs on-device using HuggingFace sentence-transformers, no API key or internet access needed at query time. The model is downloaded once on first use and cached locally.

```yaml
embeddings:
  provider: local
  model: BAAI/bge-large-en-v1.5  # ~1.3GB, high quality
  # model: all-mpnet-base-v2     # ~420MB, good quality
  # model: all-MiniLM-L6-v2      # ~80MB, fast but lower quality
```

> When switching providers, clear the existing index and re-index:
> ```bash
> uv run pragma clear-index --yes
> uv run pragma index
> ```

## Security

The API has no authentication and binds to localhost by default. MR diffs may contain proprietary code, so avoid exposing the API on public networks without a firewall or VPN.

## Development

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run linting manually
uv run pre-commit run --all-files

# Install new dependency
uv add <package-name>
```

## Systemd User Services (Auto-start on Login)

Pragma provides three systemd user services that start automatically on login:

- **`pragma-api.service`** - The REST API server
- **`pragma-indexer.service`** - Continuous MR indexer (checks for new MRs hourly)
- **`pragma-reviewer.service`** - Continuous MR reviewer (reviews new open MRs hourly)

### 1. Create the environment file at `~/.config/pragma/env`

```bash
GEMINI_API_KEY=your_gemini_api_key        # Required when using Gemini (embeddings or reviewer)
GITLAB_PRIVATE_TOKEN=your_gitlab_token    # Required for fetching MRs

# Required for self-hosted GitLab with custom CA certificates
REQUESTS_CA_BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem

# Required when using Ollama as the reviewer model
# OPENAI_BASE_URL=http://localhost:11434/v1
# OPENAI_API_KEY=ollama
```

### 2. Create the API service at `~/.config/systemd/user/pragma-api.service`

```ini
[Unit]
Description=Pragma Historical MR Search API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/YOUR_USERNAME/repositories/github/pragma
Environment="PATH=/home/YOUR_USERNAME/.local/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=%h/.config/pragma/env
ExecStart=/usr/bin/env uv run pragma serve --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

### 3. Create the indexer service at `~/.config/systemd/user/pragma-indexer.service`

```ini
[Unit]
Description=Pragma Continuous MR Indexer
After=network-online.target pragma-api.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/YOUR_USERNAME/repositories/github/pragma
Environment="PATH=/home/YOUR_USERNAME/.local/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=%h/.config/pragma/env
ExecStart=/usr/bin/env uv run pragma watch --interval 60
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

The indexer checks each configured repository every 60 minutes for new merged MRs
and indexes only those updated since the last run. State is persisted in
`./data/indexing_state.json`.

### 4. Create the reviewer service at `~/.config/systemd/user/pragma-reviewer.service`

```ini
[Unit]
Description=Pragma Continuous MR Reviewer
After=network-online.target pragma-api.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/YOUR_USERNAME/repositories/github/pragma
Environment="PATH=/home/YOUR_USERNAME/.local/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=%h/.config/pragma/env
ExecStart=/usr/bin/env uv run pragma review-watch --interval 60
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

The reviewer checks for new open MRs every 60 minutes, generates AI reviews using
the configured model, and saves them as Markdown files to `./data/reviews/`. A GNOME
desktop notification is sent when each review completes.

### 5. Enable and start all services

```bash
systemctl --user daemon-reload

systemctl --user enable pragma-api.service pragma-indexer.service pragma-reviewer.service
systemctl --user start pragma-api.service pragma-indexer.service pragma-reviewer.service

systemctl --user status pragma-api.service pragma-indexer.service pragma-reviewer.service
```

### Management commands

```bash
# View API server logs
journalctl --user -u pragma-api.service -f

# View indexer logs
journalctl --user -u pragma-indexer.service -f

# View reviewer logs
journalctl --user -u pragma-reviewer.service -f

# Restart services
systemctl --user restart pragma-api.service
systemctl --user restart pragma-indexer.service
systemctl --user restart pragma-reviewer.service

# Stop services
systemctl --user stop pragma-api.service
systemctl --user stop pragma-indexer.service
systemctl --user stop pragma-reviewer.service
```

## Requirements

- Python 3.10-3.12
- GitLab private token with API read access
- Gemini API key (only when using the `gemini` embedding provider)
- `uv` package manager

## License

MIT

## Contributing

Contributions welcome! Please ensure:
- Pre-commit hooks pass
- API endpoints are documented
- Security best practices followed
