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


@app.command(name="init")
def init_command():
    """
    Initialize Pragma configuration and vector database.
    """
    typer.echo("Initializing Pragma...")

    # Load or create config
    config = _load_config(create_if_not_exists=True)

    # Validate that required API keys are present in the environment.
    # Keys are never stored in config.yaml — set them as environment variables.
    missing = []
    if not os.getenv("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    if not os.getenv("GITLAB_PRIVATE_TOKEN"):
        missing.append("GITLAB_PRIVATE_TOKEN")

    if missing:
        typer.echo(
            "The following environment variables are required but not set:", err=True
        )
        for var in missing:
            typer.echo(f"  {var}", err=True)
        typer.echo(
            "\nExport them in your shell or add them to your ~/.env file.", err=True
        )
        raise typer.Exit(code=1)

    typer.echo("  GEMINI_API_KEY found in environment.")
    typer.echo("  GITLAB_PRIVATE_TOKEN found in environment.")

    # Build config structure — only non-sensitive values go into config.yaml
    if "vector_store" not in config:
        config["vector_store"] = {"type": "chromadb", "path": "./data/chroma_db"}

    if "repository" not in config:
        config["repository"] = {
            "type": "gitlab",
            "owner": "your_repo_owner",
            "name": "your_repo_name",
        }

    # Save non-sensitive config only
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, indent=2)

    typer.echo(f"\nConfiguration saved to {CONFIG_FILE}")
    typer.echo("\nNext steps:")
    typer.echo("  1. Update repository owner and name in config.yaml")
    typer.echo("  2. Run 'pragma test-connection' to verify GitLab access")
    typer.echo("  3. Run 'pragma index' to index historical MRs")
    typer.echo("  4. Run 'pragma serve' to start the API server")


@app.command(name="test-connection")
def test_connection_command():
    """
    Test connection to GitLab repository.
    """
    typer.echo("Testing connection to GitLab...")
    config = _load_config()

    repo_config = config.get("repository", {})
    repo_type = repo_config.get("type")
    owner = repo_config.get("owner")
    name = repo_config.get("name")

    if repo_type == "gitlab":
        gitlab_token = os.environ.get("GITLAB_PRIVATE_TOKEN")
        if not gitlab_token:
            typer.echo(
                "GITLAB_PRIVATE_TOKEN environment variable is not set.", err=True
            )
            raise typer.Exit(code=1)

        gitlab_config = config.get("gitlab", {})
        gitlab_base_url = (
            gitlab_config.get("base_url")
            or os.getenv("GITLAB_BASE_URL")
            or "https://gitlab.com"
        )

        try:
            adapter = GitlabAdapter(
                base_url=gitlab_base_url,
                private_token=gitlab_token,
                owner=owner,
                name=name,
            )
        except RuntimeError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1)

        typer.echo("\nConnection test successful!")
        typer.echo(f"  Project: {adapter.project.name}")
        typer.echo(f"  Description: {adapter.project.description or 'N/A'}")
        typer.echo(f"  Web URL: {adapter.project.web_url}")
    else:
        typer.echo(
            f"Unknown repository type: {repo_type}. Please check config.yaml", err=True
        )
        raise typer.Exit(code=1)


@app.command(name="index")
def index_command(
    max_mrs: int = typer.Option(
        50, "--max-mrs", "-n", help="Maximum number of MRs to index"
    ),
    state: str = typer.Option(
        "merged", "--state", "-s", help="MR state to fetch (merged|opened|closed|all)"
    ),
):
    """
    Index historical merge requests from GitLab.

    By default, indexes the last 50 merged MRs ordered by most recently updated.
    Use --max-mrs to change the number and --state to filter by MR state.
    """
    typer.echo("Starting indexing process...")
    config = _load_config()
    typer.echo("Loaded configuration for indexing:")
    typer.echo(
        yaml.safe_dump(
            {
                "repository": config.get("repository"),
                "vector_store": config.get("vector_store"),
            },
            indent=2,
        )
    )
    typer.echo(f"\nIndexing parameters: max_mrs={max_mrs}, state={state}\n")

    repo_config = config.get("repository", {})
    repo_type = repo_config.get("type")
    owner = repo_config.get("owner")
    name = repo_config.get("name")

    merge_requests = []

    if repo_type == "gitlab":
        gitlab_token = os.environ.get("GITLAB_PRIVATE_TOKEN")
        if not gitlab_token:
            typer.echo(
                "GITLAB_PRIVATE_TOKEN environment variable is not set.", err=True
            )
            raise typer.Exit(code=1)

        # Get GitLab base URL from config, environment, or default to gitlab.com
        gitlab_config = config.get("gitlab", {})
        gitlab_base_url = (
            gitlab_config.get("base_url")
            or os.getenv("GITLAB_BASE_URL")
            or "https://gitlab.com"
        )
        typer.echo(f"Using GitLab instance: {gitlab_base_url}")

        try:
            adapter = GitlabAdapter(
                base_url=gitlab_base_url,
                private_token=gitlab_token,
                owner=owner,
                name=name,
            )
            merge_requests = adapter.fetch_mrs(state=state, max_mrs=max_mrs)
        except RuntimeError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Fetched {len(merge_requests)} merge requests.")
        for mr in merge_requests:
            typer.echo(f"  !{mr['id']}: {mr['title']}")
    else:
        typer.echo(
            f"Unknown repository type: {repo_type}. Please check config.yaml", err=True
        )
        raise typer.Exit(code=1)

    try:
        index_merge_requests(config, merge_requests)
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

    # Run uvicorn
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
