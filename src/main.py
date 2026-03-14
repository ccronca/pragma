import logging
import os
import subprocess
import sys
from pathlib import Path

import typer
import yaml

from adapters.gitlab import GitlabAdapter
from indexer.core import index_merge_requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = typer.Typer()
CONFIG_FILE = Path("config.yaml")


def _load_config(create_if_not_exists: bool = False):
    if not CONFIG_FILE.exists():
        if create_if_not_exists:
            return {}
        else:
            typer.echo("Config file not found. Please run 'pragma init'.", err=True)
            raise typer.Exit(code=1)

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_repositories(config: dict) -> list[dict]:
    """Extract and validate the repositories list from config.

    Requires 'repositories' to be a non-empty list with valid entries.
    Each repository must have: type, owner, name.
    """
    if "repositories" not in config:
        typer.echo(
            "Error: No 'repositories' section found in config.yaml. "
            "Please run 'pragma init' or add repositories manually.",
            err=True,
        )
        raise typer.Exit(code=1)

    repos = config["repositories"]
    if not isinstance(repos, list) or not repos:
        typer.echo(
            "Error: 'repositories' must be a non-empty list in config.yaml.",
            err=True,
        )
        raise typer.Exit(code=1)

    for i, repo in enumerate(repos):
        for field in ("type", "owner", "name"):
            if not repo.get(field):
                label = repo.get("alias") or repo.get("name") or f"index {i}"
                typer.echo(
                    f"Error: Repository '{label}' is missing required field '{field}'.",
                    err=True,
                )
                raise typer.Exit(code=1)
    return repos


def _get_gitlab_base_url(config: dict) -> str:
    """Resolve the GitLab base URL from config, environment, or default."""
    gitlab_config = config.get("gitlab", {})
    return (
        gitlab_config.get("base_url")
        or os.getenv("GITLAB_BASE_URL")
        or "https://gitlab.com"
    )


def _require_gitlab_token() -> str:
    """Return the GitLab token or exit with an error."""
    token = os.environ.get("GITLAB_PRIVATE_TOKEN")
    if not token:
        typer.echo("GITLAB_PRIVATE_TOKEN environment variable is not set.", err=True)
        raise typer.Exit(code=1)
    return token


@app.command(name="init")
def init_command():
    """
    Initialize Pragma configuration and vector database.
    """
    typer.echo("Initializing Pragma...")

    config = _load_config(create_if_not_exists=True)

    # Validate that required environment variables are set.
    missing = []
    embedding_provider = config.get("embeddings", {}).get("provider", "gemini")
    if embedding_provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    if not os.getenv("GITLAB_PRIVATE_TOKEN"):
        missing.append("GITLAB_PRIVATE_TOKEN")

    if missing:
        typer.echo(
            "The following environment variables are required but not set:", err=True
        )
        for var in missing:
            typer.echo(f"  {var}", err=True)
        raise typer.Exit(code=1)

    if embedding_provider == "gemini":
        typer.echo("  GEMINI_API_KEY found in environment.")
    else:
        typer.echo(f"  Embedding provider: {embedding_provider} (no API key required).")
    typer.echo("  GITLAB_PRIVATE_TOKEN found in environment.")

    if "vector_store" not in config:
        config["vector_store"] = {"type": "chromadb", "path": "./data/chroma_db"}

    # Migrate legacy 'repository' to 'repositories' list
    if "repository" in config and "repositories" not in config:
        config["repositories"] = [config.pop("repository")]
        typer.echo("\nMigrated 'repository' to 'repositories' list format.")

    if "repositories" not in config:
        config["repositories"] = [
            {
                "type": "gitlab",
                "owner": "your_repo_owner",
                "name": "your_repo_name",
            }
        ]

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, indent=2)

    typer.echo(f"\nConfiguration saved to {CONFIG_FILE}")
    typer.echo("\nNext steps:")
    typer.echo("  1. Update repository entries in config.yaml")
    typer.echo("  2. Run 'pragma test-connection' to verify GitLab access")
    typer.echo("  3. Run 'pragma index' to index historical MRs")
    typer.echo("  4. Run 'pragma serve' to start the API server")


@app.command(name="test-connection")
def test_connection_command():
    """
    Test connection to all configured GitLab repositories.
    """
    config = _load_config()
    repos = _get_repositories(config)
    gitlab_base_url = _get_gitlab_base_url(config)
    gitlab_token = _require_gitlab_token()

    typer.echo(f"Testing connection to {len(repos)} repository(ies)...\n")

    failures = []
    for repo_config in repos:
        repo_type = repo_config.get("type")
        owner = repo_config.get("owner")
        name = repo_config.get("name")
        alias = repo_config.get("alias", name)

        if repo_type != "gitlab":
            typer.echo(f"[SKIP] {alias}: Unknown repository type '{repo_type}'")
            failures.append(alias)
            continue

        try:
            adapter = GitlabAdapter(
                base_url=gitlab_base_url,
                private_token=gitlab_token,
                owner=owner,
                name=name,
            )
            typer.echo(f"[OK]   {alias}")
            typer.echo(f"       Project: {adapter.project.name}")
            typer.echo(f"       URL: {adapter.project.web_url}")
        except RuntimeError as e:
            typer.echo(f"[FAIL] {alias}: {e}")
            failures.append(alias)

    typer.echo("")
    if failures:
        typer.echo(
            f"Connection failed for {len(failures)} repository(ies): "
            + ", ".join(failures),
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo("All connections successful.")


@app.command(name="index")
def index_command(
    max_mrs: int = typer.Option(
        50, "--max-mrs", "-n", help="Maximum number of MRs to index"
    ),
    state: str = typer.Option(
        "merged", "--state", "-s", help="MR state to fetch (merged|opened|closed|all)"
    ),
    repository: str = typer.Option(
        None,
        "--repository",
        "-r",
        help="Index a specific repository (by alias or name). If omitted, indexes all.",
    ),
):
    """
    Index historical merge requests from GitLab.

    By default, indexes all configured repositories. Use --repository to index
    a specific one by alias or name.
    """
    config = _load_config()
    repos = _get_repositories(config)
    gitlab_base_url = _get_gitlab_base_url(config)
    _require_gitlab_token()  # Validate token is set upfront

    if repository:
        matched = [
            r
            for r in repos
            if r.get("alias", r["name"]) == repository or r["name"] == repository
        ]
        if not matched:
            available = ", ".join(r.get("alias", r["name"]) for r in repos)
            typer.echo(
                f"Repository '{repository}' not found. Available: {available}",
                err=True,
            )
            raise typer.Exit(code=1)
        repos = matched

    typer.echo(f"Indexing {len(repos)} repository(ies)...")
    typer.echo(f"Parameters: max_mrs={max_mrs}, state={state}\n")

    # Build repo_configs for the indexer
    repo_configs = []
    for repo_config in repos:
        repo_type = repo_config.get("type")
        if repo_type != "gitlab":
            alias = repo_config.get("alias", repo_config["name"])
            typer.echo(
                f"Skipping '{alias}': Unknown repository type '{repo_type}'", err=True
            )
            continue

        # Add indexing parameters to repo config
        enhanced_config = {
            **repo_config,
            "base_url": gitlab_base_url,
            "max_mrs": max_mrs,
            "state": state,
        }
        repo_configs.append(enhanced_config)

    if not repo_configs:
        typer.echo("\nNo valid repositories to index.")
        return

    try:
        index_merge_requests(config, repo_configs)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)


@app.command(name="serve")
def serve_command(
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Host to bind to (default: localhost for security)"
    ),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload for development"
    ),
):
    """
    Start the Pragma REST API server.

    Provides historical MR context via REST endpoints.
    Defaults to localhost (127.0.0.1) for security.

    Use --host 0.0.0.0 to expose to network (INSECURE without authentication).
    """
    if host == "0.0.0.0":
        typer.echo(
            "WARNING: Binding to 0.0.0.0 exposes the API to your network!", err=True
        )
        typer.echo(
            "WARNING: This API has NO authentication and may contain sensitive MR data.",
            err=True,
        )
        typer.echo(
            "WARNING: Only use 0.0.0.0 if behind a firewall/VPN or on a trusted network.\n",
            err=True,
        )

    typer.echo("=" * 80)
    typer.echo("STARTING PRAGMA REST API")
    typer.echo("=" * 80)
    typer.echo(f"\nServer:     http://{host}:{port}")
    typer.echo(f"API Docs:   http://{host}:{port}/docs")
    typer.echo(f"Health:     http://{host}:{port}/health")
    typer.echo(f"Auto-reload: {reload}\n")
    typer.echo("Press CTRL+C to stop\n")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api_server:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    if reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd, cwd="src", check=True)
    except KeyboardInterrupt:
        typer.echo("\n\nShutting down server...")
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error starting server: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="clear-index")
def clear_index_command(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """
    Clear all indexed data from ChromaDB.

    WARNING: This will delete all indexed MRs. You'll need to re-run 'pragma index'.
    """
    if not confirm:
        confirmed = typer.confirm(
            "WARNING: This will delete ALL indexed MRs from the database. Continue?"
        )
        if not confirmed:
            typer.echo("Aborted.")
            raise typer.Exit(code=0)

    config = _load_config()
    chroma_path = config.get("vector_store", {}).get("path", "./data/chroma_db")

    typer.echo(f"Clearing ChromaDB at {chroma_path}...")

    import chromadb

    db = chromadb.PersistentClient(path=chroma_path)

    try:
        db.delete_collection("pragma_collection")
        db.create_collection("pragma_collection")
        typer.echo("Database cleared successfully.")
        typer.echo("Run 'pragma index' to re-index MRs.")
    except Exception as e:
        typer.echo(f"Error clearing database: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
