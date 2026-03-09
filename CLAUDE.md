# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pragma is a historical GitLab MR database with semantic search API. It indexes merge requests (descriptions, diffs, discussions) and provides a REST API for external AI tools (Claude Code, Gemini, etc.) to query for similar historical context.

**Key concept**: Instead of doing code reviews locally, Pragma acts as a **searchable knowledge base** that AI assistants can query to get institutional knowledge from past code changes.

## Package Management

This project uses `uv` for dependency management. Always use `uv` commands:

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package-name>

# Run commands
uv run pragma <command>
```

## CLI Commands

The main entry point is `src/main.py`, which provides these commands:

### Initialize Project
```bash
uv run pragma init
```
- Creates or updates `config.yaml` with default configuration
- Prompts for Gemini API key and GitLab private token
- Initializes ChromaDB vector store at `./data/chroma_db`

### Test GitLab Connection
```bash
uv run pragma test-connection
```
- Verifies GitLab authentication and repository access
- Displays project details (name, description, web URL)

### Index Historical Data
```bash
uv run pragma index
```
- Fetches merged MRs from GitLab using the python-gitlab API
- Default: fetches last 50 merged MRs, ordered by most recently updated
- Converts MRs (title, description, diff, discussions) into LlamaIndex Document objects
- Embeds documents using Gemini embeddings (`gemini-embedding-001`)
- Stores embeddings in ChromaDB for retrieval

### Start API Server
```bash
uv run pragma serve                    # Localhost only (secure default)
uv run pragma serve --port 8080        # Custom port
uv run pragma serve --reload           # Development mode with auto-reload
uv run pragma serve --host 0.0.0.0     # Network access (INSECURE - shows warning)
```
- Starts FastAPI REST server for semantic search
- **Defaults to 127.0.0.1 (localhost) for security**
- API docs available at `http://localhost:8000/docs`
- No authentication - keep on localhost unless on trusted network

## Architecture

### Data Flow

1. **Indexing Phase** (`pragma index`):
   ```
   GitLab API → GitlabAdapter.fetch_mrs()
     → indexer/core.index_merge_requests()
     → LlamaIndex Documents
     → Gemini Embeddings
     → ChromaDB Storage
   ```

2. **API Query Phase** (external tool calls `GET /search`):
   ```
   External AI Tool (Claude Code, Gemini)
     → POST /search with code diff
     → ChromaDB Vector Similarity Search
     → Return top-K similar MRs with metadata
     → AI uses context for review
   ```

### Module Responsibilities

- **`src/adapters/gitlab.py`**: GitLab API integration
  - Uses `python-gitlab` library for API calls
  - Authenticates with private token and fetches MR data
  - Returns list of dicts with: `id`, `title`, `description`, `diff`, `discussions`, `author`, `created_at`, `merged_at`, `web_url`

- **`src/indexer/core.py`**: Embedding and storage logic
  - Initializes Gemini embedding model and ChromaDB client
  - Converts MR data into Document objects with metadata
  - Builds VectorStoreIndex and persists to ChromaDB

- **`src/api_server.py`**: FastAPI REST server
  - **POST /search**: Semantic search for similar MRs
  - **GET /mrs/{mr_id}**: Get full MR details
  - **GET /mrs**: List all indexed MRs
  - **GET /stats**: Database statistics
  - **GET /health**: Health check

- **`src/main.py`**: CLI entry point using Typer
  - Orchestrates commands: `init`, `test-connection`, `index`, `serve`

### Configuration

The `config.yaml` file contains:
```yaml
api_keys:
  gemini: <GEMINI_API_KEY>
  gitlab: <GITLAB_PRIVATE_TOKEN>
gitlab:
  base_url: https://gitlab.cee.redhat.com  # Optional, defaults to https://gitlab.com
repository:
  type: gitlab
  owner: <repo_owner_or_group>  # For nested groups: product-security/pdm
  name: <repo_name>              # Repository name: pdm-db
vector_store:
  type: chromadb
  path: ./data/chroma_db
```

**Important**:
- Gemini API key is required for embeddings
- GitLab private token is required for fetching MRs (needs API read access)
- GitLab base_url: Set for self-hosted GitLab instances
- ChromaDB path defaults to `./data/chroma_db` (local persistent storage)

### Key Dependencies

- **LlamaIndex**: RAG orchestration framework
  - `llama-index`: Core library
  - `llama-index-embeddings-google-genai`: Gemini embeddings integration
  - `llama-index-vector-stores-chroma`: ChromaDB integration
- **ChromaDB**: Local vector database for storing embeddings
- **FastAPI**: REST API framework
- **Typer**: CLI framework
- **python-gitlab**: GitLab API client
- **python-dotenv**: Environment variable management

## Development

### Pre-commit Hooks

The project uses pre-commit hooks for code quality:

```bash
# Install hooks (already done during git init)
uv run pre-commit install

# Run hooks manually
uv run pre-commit run --all-files
```

Hooks configured:
- Trailing whitespace removal
- End-of-file fixer
- YAML/JSON/TOML validation
- Large file check
- Merge conflict detection
- Private key detection
- Ruff linting and formatting

### Testing the API

Start the server and test endpoints:

```bash
# Terminal 1: Start server
uv run pragma serve

# Terminal 2: Test API
curl http://localhost:8000/health
curl http://localhost:8000/stats
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"code_diff": "your diff here", "top_k": 5}'

# Or visit API docs
open http://localhost:8000/docs
```

### Using with External AI Tools

External tools (Claude Code, Gemini, custom scripts) can call the API:

```python
import requests

# Search for similar MRs
response = requests.post("http://localhost:8000/search", json={
    "code_diff": "your code diff here",
    "top_k": 5,
    "min_score": 0.5
})
similar_mrs = response.json()

# Get specific MR details
mr = requests.get("http://localhost:8000/mrs/383").json()
```

## Complete Workflow Example

```bash
# 1. Initialize project and configure API keys
uv run pragma init

# 2. Update config.yaml with your GitLab repository

# 3. Test GitLab connection
uv run pragma test-connection

# 4. Index historical MRs from your repository
uv run pragma index

# 5. Start the API server
uv run pragma serve

# 6. Use external AI tools to query the API for code review context
```

## Security Notes

- **API has no authentication** - keep on localhost (127.0.0.1) by default
- **Contains sensitive data** - MR diffs and discussions may include proprietary code
- **Do not expose to public networks** without proper firewall/VPN protection
- **Use `--host 0.0.0.0` only** on trusted networks or with additional security layers

## MCP Server

Pragma includes an **MCP server** (`src/mcp_server.py`) that exposes the search API as standard MCP tools, compatible with any MCP client (Claude Code, Gemini Code Assist, custom agents).

### Setup

**1. Start the Pragma API server** (must be running before starting the MCP server):
```bash
uv run pragma serve
```

**2. Configure the MCP server in your client** (e.g. Claude Code `~/.claude.json`):

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

### Available Tools

After connecting, the following tools are available:

| Tool | Description |
|------|-------------|
| `mcp__pragma__search` | Semantic search over historical MRs |
| `mcp__pragma__get_mr` | Get full details of a specific MR by IID |
| `mcp__pragma__list_mrs` | List all indexed MRs with pagination |

### Tool: `search`

The most important tool. Searches indexed MR content using semantic similarity.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | — | Natural language query. Best for finding discussions and past decisions |
| `code_diff` | string | — | Unified diff or code snippet. Best for finding similar code patterns |
| `content_type` | string | — | Restrict to `"discussion"` or `"diff"` |
| `top_k` | integer | 5 | Number of results to return (1–20) |
| `min_score` | float | 0.0 | Minimum similarity score threshold (0–1) |

Provide either `query` or `code_diff`, not both.

**When to use `content_type: "discussion"`:**

Use natural language to surface business reasoning, team decisions, and review feedback from past MRs. Best for questions like:
- "How did the team decide to handle X?"
- "Was there a discussion about this approach?"
- "What tradeoffs were considered for Y?"

**When to use `content_type: "diff"`:**

Use a code snippet or unified diff to find similar past implementations. Best for:
- "Has this pattern been used before?"
- "How was retry logic implemented in the past?"
- "Find MRs with similar changes to this file."

**Two-query pattern** — for the richest context, make two calls:

```
# Call 1: Find team discussions on the topic
search(query="<natural language description>", content_type="discussion", top_k=5)

# Call 2: Find similar code changes
search(code_diff="<the actual diff>", content_type="diff", top_k=5)
```

### Example Queries

**Find past decisions about an architectural topic:**
```
Use pragma search to find discussions about "database migration strategy" in past MRs.
```

**Find how retry logic was implemented before:**
```
Search pragma discussions for "retry logic exponential backoff" and also search diffs
for similar code patterns to what I'm adding.
```

**Review a code change with historical context:**
```
Before reviewing this change, search pragma for:
1. Discussions about similar changes to this component
2. Past MRs with similar diff patterns

Use both results to provide context-aware review feedback.
```

**Explore what MRs are indexed:**
```
List the most recent 20 MRs indexed in pragma so I can see what historical
context is available.
```

**Drill into a specific MR:**
```
Get the full details of pragma MR !383 including its diff and review discussions.
```

### Tips for Better Results

- **Be specific in queries**: "Jira API pagination with maxResults=False" returns better results than "Jira API"
- **Use technical terms from the codebase**: The embeddings are trained on your actual MR content, so domain-specific terms work well
- **Combine both search modes**: Diff search finds *what* was done; discussion search finds *why* it was done
- **Adjust `min_score`**: Use `0.4`–`0.5` to filter out low-relevance results when results are noisy
- **Use `get_mr` to go deeper**: When a search result looks relevant, call `get_mr` with its `mr_id` to read the full diff and all discussion threads

## Future Enhancements

- Authentication layer for API
- Incremental indexing (only new MRs)
- GitHub adapter
- Filtering by labels, authors, date ranges
- MCP tools for indexing operations (currently CLI-only)
