import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from adapters.gitlab import GitlabAdapter
from indexer.core import index_merge_requests

logger = logging.getLogger(__name__)

STATE_FILE = Path("./data/indexing_state.json")


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


def _get_last_indexed_time(repo_key: str) -> Optional[str]:
    """Get last indexed timestamp for a repository."""
    state = _load_state()
    repo_state = state.get("repositories", {}).get(repo_key)
    if repo_state:
        return repo_state.get("last_indexed_at")
    return None


def _update_state(repo_key: str, timestamp: str, success: bool) -> None:
    """Update tracking state after indexing a repository.

    Args:
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


def _get_gitlab_base_url(config: dict) -> str:
    """Resolve GitLab base URL from config or environment."""
    gitlab_config = config.get("gitlab", {})
    return (
        gitlab_config.get("base_url")
        or os.getenv("GITLAB_BASE_URL")
        or "https://gitlab.com"
    )


async def run_continuous_indexing(
    config: dict,
    interval_minutes: int = 5,
    run_once: bool = False,
) -> None:
    """Run continuous indexing by monitoring GitLab for new merged MRs.

    Args:
        config: Pragma configuration dict.
        interval_minutes: How often to check for new MRs.
        run_once: If True, run once and exit (for testing).
    """
    repos = config.get("repositories", [])
    gitlab_base_url = _get_gitlab_base_url(config)
    gitlab_token = os.environ.get("GITLAB_PRIVATE_TOKEN")

    if not gitlab_token:
        raise ValueError("GITLAB_PRIVATE_TOKEN environment variable is not set")

    while True:
        logger.info("Checking %d repositories for new MRs...", len(repos))
        timestamp = datetime.now(timezone.utc).isoformat()
        indexed_count = 0
        failed_repos = []

        for repo in repos:
            repo_owner = repo["owner"]
            repo_name = repo["name"]
            repo_key = f"{repo_owner}/{repo_name}"

            try:
                # Get last indexed time
                last_indexed = _get_last_indexed_time(repo_key)
                logger.info(
                    "Checking %s (last indexed: %s)", repo_key, last_indexed or "never"
                )

                # Fetch new MRs since last indexed
                adapter = GitlabAdapter(
                    base_url=gitlab_base_url,
                    private_token=gitlab_token,
                    owner=repo_owner,
                    name=repo_name,
                )

                mrs = adapter.fetch_mrs(
                    state="merged", max_mrs=50, updated_after=last_indexed
                )

                if mrs:
                    logger.info("Found %d new MRs in %s", len(mrs), repo_key)

                    # Index the repository with new MRs
                    repo_config = {
                        **repo,
                        "base_url": gitlab_base_url,
                        "max_mrs": 50,
                        "state": "merged",
                    }
                    index_merge_requests(config, [repo_config])
                    indexed_count += len(mrs)

                    # Update state with success
                    _update_state(repo_key, timestamp, success=True)
                else:
                    logger.info("No new MRs in %s", repo_key)
                    # Update timestamp even if no new MRs
                    _update_state(repo_key, timestamp, success=True)

            except Exception as e:
                logger.exception("Error indexing repository %s: %s", repo_key, e)
                failed_repos.append(repo_key)
                _update_state(repo_key, timestamp, success=False)

        # Log summary
        logger.info("Indexing run complete: %d new MRs indexed", indexed_count)
        if failed_repos:
            logger.warning("Failed repositories: %s", ", ".join(failed_repos))

        if run_once:
            break

        logger.info("Sleeping %d minutes until next check...", interval_minutes)
        await asyncio.sleep(interval_minutes * 60)
