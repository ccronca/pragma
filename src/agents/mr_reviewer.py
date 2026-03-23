"""Automated MR reviewer: polls for open MRs, generates AI reviews, saves to files."""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic_ai import Agent

from adapters.gitlab import GitlabAdapter

logger = logging.getLogger(__name__)

REVIEW_STATE_FILE = Path("./data/review_state.json")
REVIEWS_DIR = Path("./data/reviews")

_REVIEW_SYSTEM_PROMPT = """You are an expert code reviewer with deep experience in software engineering,
security, and code quality. Your reviews are thorough, specific, and actionable.

When reviewing, always:
- Reference specific files and line contexts
- Distinguish between blocking issues and suggestions
- Explain WHY something is a problem, not just what
- Suggest concrete fixes, not vague improvements
- Consider the change in context of the described historical patterns
"""

_REVIEW_PROMPT = """\
## MR Information

**Title:** {title}
**Author:** {author}
**Created:** {created_at}
**URL:** {web_url}

## Description

{description}

## Historical Context from Pragma

{historical_context}

## Code Diff

```diff
{diff}
```

---

Please review this merge request with the following structure:

## Historical Context Summary
Summarise any relevant patterns or decisions from past MRs (above). If none are relevant, state so briefly.

## Code Analysis
Review correctness, logic errors, and code quality. Reference specific lines or files.

## Security
Identify any security vulnerabilities (injection, hardcoded secrets, auth issues, input validation, etc.).
If none found, confirm explicitly.

## Quality & Maintainability
Assess readability, naming, duplication, complexity, and adherence to existing patterns.

## Test Coverage
Are tests included? What critical paths are untested? Suggest specific test cases.

## Recommendations

### Blocking Issues
List issues that must be fixed before merging (numbered list, or "None").

### Suggestions
Non-blocking improvements (numbered list, or "None").

### Summary
One paragraph summary of the overall MR quality and recommended action (approve / request changes / needs discussion).
"""


def _load_review_state() -> dict:
    """Load review tracking state from disk."""
    if REVIEW_STATE_FILE.exists():
        with open(REVIEW_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"repositories": {}}


def _save_review_state(state: dict) -> None:
    """Persist review tracking state to disk."""
    REVIEW_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _get_reviewed_mr_ids(repo_key: str) -> set:
    """Return the set of MR IDs already reviewed for a repository."""
    state = _load_review_state()
    repo_state = state.get("repositories", {}).get(repo_key, {})
    return set(repo_state.get("reviewed_mr_ids", []))


def _mark_mr_reviewed(repo_key: str, mr_id: int) -> None:
    """Record that a MR has been reviewed."""
    state = _load_review_state()
    repos = state.setdefault("repositories", {})
    repo_state = repos.setdefault(repo_key, {"reviewed_mr_ids": []})
    if mr_id not in repo_state["reviewed_mr_ids"]:
        repo_state["reviewed_mr_ids"].append(mr_id)
    _save_review_state(state)


def _build_review_agent(config: dict) -> Agent:
    """Create a pydantic_ai Agent from the agent config section.

    Handles provider-specific setup:
    - ollama: sets OPENAI_BASE_URL + OPENAI_API_KEY so pydantic_ai can reach Ollama
    - gemini: prefixes model name with 'google-gla:' as required by pydantic_ai
    - other: passes model string through as-is
    """
    agent_config = config.get("agent", {})
    provider = agent_config.get("provider", "gemini")
    model_name = agent_config.get("model", "gemini-2.5-flash-lite")

    if provider == "ollama":
        base_url = agent_config.get("base_url", "http://localhost:11434/v1")
        os.environ.setdefault("OPENAI_BASE_URL", base_url)
        os.environ.setdefault("OPENAI_API_KEY", "ollama")
        model_str = model_name if ":" in model_name else f"openai:{model_name}"
    elif provider == "gemini":
        model_str = f"google-gla:{model_name}"
    else:
        model_str = model_name

    return Agent(model_str, result_type=str, system_prompt=_REVIEW_SYSTEM_PROMPT)


def _get_gitlab_base_url(config: dict) -> str:
    """Resolve GitLab base URL from config or environment."""
    gitlab_config = config.get("gitlab", {})
    return (
        gitlab_config.get("base_url")
        or os.getenv("GITLAB_BASE_URL")
        or "https://gitlab.com"
    )


def _save_review(review_text: str, mr: dict, repo_key: str) -> Path:
    """Save a review to data/reviews/ as a Markdown file.

    Returns the path to the saved file.
    """
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    repo_name = repo_key.split("/")[-1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{repo_name}_mr{mr['id']}_{timestamp}.md"
    review_path = REVIEWS_DIR / filename

    header = f"""# Code Review: {mr["title"]}

| Field | Value |
|-------|-------|
| MR | !{mr["id"]} |
| Author | {mr.get("author", "unknown")} |
| URL | {mr.get("web_url", "")} |
| Reviewed at | {datetime.now(timezone.utc).isoformat()} |

---

"""
    with open(review_path, "w", encoding="utf-8") as f:
        f.write(header + review_text)

    logger.info("Saved review to %s", review_path)
    return review_path


def _notify(title: str, body: str) -> None:
    """Send a GNOME desktop notification via notify-send. Silently skips if unavailable."""
    try:
        subprocess.run(
            ["notify-send", "--app-name=Pragma", title, body],
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass


async def _query_pragma_context(
    mr_title: str,
    mr_diff: str,
    pragma_url: str,
) -> str:
    """Query Pragma API for historical context via two parallel searches.

    Searches discussions (by title) and diffs (by code) in parallel.
    Returns a formatted Markdown string, or a fallback message if unavailable.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            health = await client.get(f"{pragma_url}/health")
            if health.status_code != 200:
                return "_Pragma API unavailable — no historical context._"
        except Exception:
            return "_Pragma API unavailable — no historical context._"

        # Two parallel searches: discussions (natural language) + diffs (code)
        discussion_response, diff_response = await asyncio.gather(
            client.post(
                f"{pragma_url}/search",
                json={
                    "query": mr_title,
                    "content_type": "discussion",
                    "top_k": 5,
                    "min_score": 0.5,
                },
            ),
            client.post(
                f"{pragma_url}/search",
                json={
                    "code_diff": mr_diff[:3000],
                    "content_type": "diff",
                    "top_k": 5,
                    "min_score": 0.5,
                },
            ),
            return_exceptions=True,
        )

    lines = []

    if (
        not isinstance(discussion_response, Exception)
        and discussion_response.status_code == 200
    ):
        results = discussion_response.json()
        if results:
            lines.append("### Similar Past Discussions")
            for r in results[:3]:
                lines.append(
                    f"- **MR !{r['mr_id']}** (score: {r['similarity_score']:.2f}): {r['mr_title']}"
                )
                if r.get("content_preview"):
                    snippet = r["content_preview"][:300].replace("\n", " ")
                    lines.append(f"  > {snippet}...")

    if not isinstance(diff_response, Exception) and diff_response.status_code == 200:
        results = diff_response.json()
        if results:
            lines.append("\n### Similar Past Code Changes")
            for r in results[:3]:
                lines.append(
                    f"- **MR !{r['mr_id']}** (score: {r['similarity_score']:.2f}): {r['mr_title']}"
                )

    if not lines:
        return "_No similar historical MRs found in Pragma._"

    return "\n".join(lines)


async def review_mr(mr: dict, config: dict, pragma_url: str) -> Path:
    """Generate an AI review for a single MR and save it to data/reviews/.

    Args:
        mr: MR data dict (from GitlabAdapter.fetch_mr or fetch_mrs).
        config: Pragma configuration dict.
        pragma_url: URL of the Pragma API for historical context.

    Returns:
        Path to the saved review file.
    """
    repo_key = f"{mr['repo_owner']}/{mr['repo_name']}"

    review_agent = _build_review_agent(config)

    historical_context = await _query_pragma_context(
        mr_title=mr["title"],
        mr_diff=mr.get("diff", ""),
        pragma_url=pragma_url,
    )

    prompt = _REVIEW_PROMPT.format(
        title=mr["title"],
        author=mr.get("author", "unknown"),
        created_at=mr.get("created_at", "unknown"),
        web_url=mr.get("web_url", ""),
        description=(mr.get("description") or "No description provided."),
        diff=(mr.get("diff") or "No diff available.")[:8000],
        historical_context=historical_context,
    )

    result = await review_agent.run(prompt)
    review_path = _save_review(result.data, mr, repo_key)
    _notify(
        f"Review ready: MR !{mr['id']}",
        mr["title"],
    )
    return review_path


async def run_continuous_review(
    config: dict,
    interval_minutes: int = 60,
    run_once: bool = False,
    pragma_url: str = "http://localhost:8000",
) -> None:
    """Monitor GitLab for new open MRs and generate AI reviews saved to files.

    Args:
        config: Pragma configuration dict.
        interval_minutes: How often to check for new open MRs.
        run_once: If True, run once and exit (for testing).
        pragma_url: URL of the Pragma API for historical context.
    """
    repos = config.get("repositories", [])
    gitlab_base_url = _get_gitlab_base_url(config)
    gitlab_token = os.environ.get("GITLAB_PRIVATE_TOKEN")

    if not gitlab_token:
        raise ValueError("GITLAB_PRIVATE_TOKEN environment variable is not set")

    while True:
        logger.info(
            "Checking %d repositories for new open MRs to review...", len(repos)
        )

        for repo in repos:
            repo_owner = repo["owner"]
            repo_name = repo["name"]
            repo_key = f"{repo_owner}/{repo_name}"

            try:
                already_reviewed = _get_reviewed_mr_ids(repo_key)

                adapter = GitlabAdapter(
                    base_url=gitlab_base_url,
                    private_token=gitlab_token,
                    owner=repo_owner,
                    name=repo_name,
                )
                open_mrs = adapter.fetch_mrs(state="opened", max_mrs=20)
                new_mrs = [mr for mr in open_mrs if mr["id"] not in already_reviewed]

                logger.info(
                    "Found %d new open MRs to review in %s", len(new_mrs), repo_key
                )

                for mr in new_mrs:
                    try:
                        logger.info("Reviewing MR !%d: %s", mr["id"], mr["title"])
                        review_path = await review_mr(mr, config, pragma_url)
                        _mark_mr_reviewed(repo_key, mr["id"])
                        logger.info(
                            "Review complete for MR !%d → %s", mr["id"], review_path
                        )
                    except Exception as e:
                        logger.exception("Error reviewing MR !%d: %s", mr["id"], e)

            except Exception as e:
                logger.exception("Error processing repository %s: %s", repo_key, e)

        if run_once:
            break

        logger.info("Sleeping %d minutes until next review check...", interval_minutes)
        await asyncio.sleep(interval_minutes * 60)
