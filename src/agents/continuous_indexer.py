import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from adapters.gitlab import GitlabAdapter
from indexer.core import index_merge_requests

logger = logging.getLogger(__name__)


class RepoConfig(BaseModel):
    """Repository configuration for indexing."""

    owner: str
    name: str
    state: str = "merged"
    max_mrs: int = 50


STATE_FILE = Path("./data/indexing_state.json")


class IndexingResult(BaseModel):
    indexed_count: int
    failed_repos: list[str]
    last_indexed_at: datetime
    health_status: str  # "healthy" | "degraded" | "failing"
    action_needed: Optional[str] = None


def _load_state() -> dict:
    """Load indexing state from disk, creating defaults if missing."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"repositories": {}}


def _save_state(state: dict) -> None:
    """Persist indexing state to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


_SYSTEM_PROMPT = (
    "You monitor GitLab repositories for new merged MRs and index them "
    "into the Pragma knowledge base. Use the provided tools to check each "
    "repository for updates, fetch new MRs, index them, and update the "
    "tracking state. Report a summary of what was indexed and any failures."
)

_agent_instance: Optional[Agent] = None


def _get_agent_model(config: dict) -> str:
    """Determine which agent model to use from config or environment.

    Supported providers:
    - gemini: Uses Google Gemini models (requires GEMINI_API_KEY)
    - ollama: Uses local Ollama models (no API key required)

    Priority:
    1. PRAGMA_AGENT_MODEL environment variable
    2. config.yaml agent.model setting
    3. Default based on agent.provider setting
    4. Fallback to gemini-2.0-flash-thinking-exp-01-21
    """
    # Check environment variable first
    env_model = os.getenv("PRAGMA_AGENT_MODEL")
    if env_model:
        return env_model

    # Check config.yaml
    agent_config = config.get("agent", {})
    if "model" in agent_config:
        return agent_config["model"]

    # Infer from provider
    provider = agent_config.get("provider", "gemini")
    provider_defaults = {
        "gemini": "gemini-2.5-flash-lite",
        "ollama": "llama3",
    }
    return provider_defaults.get(provider, "gemini-2.5-flash-lite")


def _get_agent(config: dict) -> Agent:
    """Lazily create the agent to avoid API key validation at import time."""
    global _agent_instance  # noqa: PLW0603
    if _agent_instance is None:
        model = _get_agent_model(config)
        logger.info("Initializing agent with model: %s", model)
        _agent_instance = Agent(
            model,
            result_type=IndexingResult,
            system_prompt=_SYSTEM_PROMPT,
        )
        # Register tools on the newly created agent
        _agent_instance.tool(get_last_indexed_time)
        _agent_instance.tool(fetch_new_mrs)
        _agent_instance.tool(index_mrs)
        _agent_instance.tool(update_state)
    return _agent_instance


async def get_last_indexed_time(ctx: RunContext, repo_key: str) -> Optional[str]:
    """Get last indexed timestamp for a repository."""
    state = _load_state()
    repo_state = state.get("repositories", {}).get(repo_key)
    if repo_state:
        return repo_state.get("last_indexed_at")
    return None


async def fetch_new_mrs(
    ctx: RunContext,
    repo_owner: str,
    repo_name: str,
    since: Optional[str] = None,
) -> list[dict]:
    """Fetch MRs updated after the given timestamp.

    Args:
        ctx: Run context provided by the agent framework.
        repo_owner: GitLab group/owner path (e.g. 'product-security/pdm').
        repo_name: Repository name.
        since: ISO-8601 timestamp; only MRs updated after this are returned.

    Returns:
        List of MR dicts suitable for indexing.
    """
    config = ctx.deps["config"]
    gitlab_config = config.get("gitlab", {})
    base_url = (
        gitlab_config.get("base_url")
        or os.getenv("GITLAB_BASE_URL")
        or "https://gitlab.com"
    )
    token = os.environ.get("GITLAB_PRIVATE_TOKEN")
    if not token:
        raise ValueError("GITLAB_PRIVATE_TOKEN environment variable is not set")

    adapter = GitlabAdapter(
        base_url=base_url,
        private_token=token,
        owner=repo_owner,
        name=repo_name,
    )

    # Use GitLab's native filtering instead of client-side
    mrs = adapter.fetch_mrs(state="merged", max_mrs=50, updated_after=since)

    logger.info(
        "Fetched %d new MRs from %s/%s (since=%s)",
        len(mrs),
        repo_owner,
        repo_name,
        since,
    )
    return mrs


async def index_mrs(ctx: RunContext, repo_configs: list[RepoConfig]) -> int:
    """Index the given repository configs into the vector store.

    Args:
        ctx: Run context provided by the agent framework.
        repo_configs: List of repository configurations.

    Returns:
        Number of repositories successfully processed.
    """
    config = ctx.deps["config"]
    gitlab_config = config.get("gitlab", {})
    base_url = (
        gitlab_config.get("base_url")
        or os.getenv("GITLAB_BASE_URL")
        or "https://gitlab.com"
    )

    # Convert Pydantic models to dicts for indexer
    enhanced_configs = [
        {**rc.model_dump(), "base_url": base_url} for rc in repo_configs
    ]

    index_merge_requests(config, enhanced_configs)
    return len(enhanced_configs)


async def update_state(
    ctx: RunContext,
    repo_key: str,
    timestamp: str,
    success: bool,
) -> None:
    """Update tracking state after indexing a repository.

    Args:
        ctx: Run context provided by the agent framework.
        repo_key: Repository identifier in 'owner/name' format.
        timestamp: ISO-8601 timestamp of this indexing run.
        success: Whether indexing completed without errors.
    """
    state = _load_state()
    repos = state.setdefault("repositories", {})
    repo_state = repos.setdefault(
        repo_key,
        {"last_indexed_at": None, "success_count": 0, "failure_count": 0},
    )

    repo_state["last_indexed_at"] = timestamp
    if success:
        repo_state["success_count"] = repo_state.get("success_count", 0) + 1
    else:
        repo_state["failure_count"] = repo_state.get("failure_count", 0) + 1

    _save_state(state)
    logger.info("Updated state for %s: success=%s", repo_key, success)


async def run_continuous_indexing(
    config: dict,
    interval_minutes: int = 5,
    run_once: bool = False,
) -> None:
    """Run the continuous indexing agent.

    Args:
        config: Pragma configuration dict.
        interval_minutes: How often to check for new MRs.
        run_once: If True, run once and exit (for testing).
    """
    repos = config.get("repositories", [])

    while True:
        logger.info("Checking %d repositories for new MRs...", len(repos))

        try:
            agent = _get_agent(config)
            result = await agent.run(
                f"Check for new MRs in {len(repos)} repositories and index them. "
                f"Repositories: {json.dumps([r['owner'] + '/' + r['name'] for r in repos])}",
                deps={"config": config},
            )

            logger.info("Indexed %d new MRs", result.data.indexed_count)
            if result.data.failed_repos:
                logger.warning("Failed repos: %s", result.data.failed_repos)
            if result.data.action_needed:
                logger.warning("Action needed: %s", result.data.action_needed)
        except Exception:
            logger.exception("Error during continuous indexing run")

        if run_once:
            break

        logger.info("Sleeping %d minutes until next check...", interval_minutes)
        await asyncio.sleep(interval_minutes * 60)
