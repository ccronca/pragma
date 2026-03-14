"""Unit tests for the continuous indexing agent."""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.continuous_indexer import (
    IndexingResult,
    _load_state,
    _save_state,
    fetch_new_mrs,
    get_last_indexed_time,
    run_continuous_indexing,
    update_state,
)


@pytest.fixture()
def mock_config():
    return {
        "repositories": [
            {"type": "gitlab", "owner": "group", "name": "repo-a"},
            {"type": "gitlab", "owner": "group", "name": "repo-b"},
        ],
        "vector_store": {"path": "./data/chroma_db"},
    }


@pytest.fixture()
def temp_state_file(tmp_path, monkeypatch):
    """Point STATE_FILE to a temporary path."""
    state_path = tmp_path / "indexing_state.json"
    monkeypatch.setattr("agents.continuous_indexer.STATE_FILE", state_path)
    return state_path


def _make_run_context(config: dict) -> MagicMock:
    """Create a mock RunContext with deps."""
    ctx = MagicMock()
    ctx.deps = {"config": config}
    return ctx


# ---------------------------------------------------------------------------
# State management tests
# ---------------------------------------------------------------------------


class TestLoadState:
    def test_creates_empty_state_if_file_missing(self, temp_state_file):
        assert not temp_state_file.exists()
        state = _load_state()
        assert state == {"repositories": {}}

    def test_reads_existing_state(self, temp_state_file):
        data = {
            "repositories": {
                "group/repo-a": {
                    "last_indexed_at": "2025-01-01T00:00:00",
                    "success_count": 3,
                    "failure_count": 0,
                }
            }
        }
        temp_state_file.write_text(json.dumps(data), encoding="utf-8")
        state = _load_state()
        assert state == data


class TestSaveState:
    def test_writes_json_to_disk(self, temp_state_file):
        state = {
            "repositories": {
                "group/repo-a": {
                    "last_indexed_at": "2025-06-01T12:00:00",
                    "success_count": 1,
                    "failure_count": 0,
                }
            }
        }
        _save_state(state)
        assert temp_state_file.exists()
        loaded = json.loads(temp_state_file.read_text(encoding="utf-8"))
        assert loaded == state

    def test_creates_parent_directory(self, tmp_path, monkeypatch):
        nested_path = tmp_path / "nested" / "dir" / "state.json"
        monkeypatch.setattr("agents.continuous_indexer.STATE_FILE", nested_path)
        _save_state({"repositories": {}})
        assert nested_path.exists()


# ---------------------------------------------------------------------------
# Agent tool: get_last_indexed_time
# ---------------------------------------------------------------------------


class TestGetLastIndexedTime:
    async def test_returns_none_for_new_repo(self, temp_state_file):
        ctx = _make_run_context({})
        result = await get_last_indexed_time(ctx, "group/repo-new")
        assert result is None

    async def test_returns_timestamp_for_known_repo(self, temp_state_file):
        data = {
            "repositories": {
                "group/repo-a": {
                    "last_indexed_at": "2025-01-01T00:00:00",
                    "success_count": 1,
                    "failure_count": 0,
                }
            }
        }
        temp_state_file.write_text(json.dumps(data), encoding="utf-8")
        ctx = _make_run_context({})
        result = await get_last_indexed_time(ctx, "group/repo-a")
        assert result == "2025-01-01T00:00:00"


# ---------------------------------------------------------------------------
# Agent tool: update_state
# ---------------------------------------------------------------------------


class TestUpdateState:
    async def test_persists_success(self, temp_state_file):
        ctx = _make_run_context({})
        await update_state(ctx, "group/repo-a", "2025-06-01T12:00:00", True)

        state = json.loads(temp_state_file.read_text(encoding="utf-8"))
        repo = state["repositories"]["group/repo-a"]
        assert repo["last_indexed_at"] == "2025-06-01T12:00:00"
        assert repo["success_count"] == 1
        assert repo["failure_count"] == 0

    async def test_persists_failure(self, temp_state_file):
        ctx = _make_run_context({})
        await update_state(ctx, "group/repo-a", "2025-06-01T12:00:00", False)

        state = json.loads(temp_state_file.read_text(encoding="utf-8"))
        repo = state["repositories"]["group/repo-a"]
        assert repo["failure_count"] == 1
        assert repo["success_count"] == 0

    async def test_increments_existing_counts(self, temp_state_file):
        ctx = _make_run_context({})
        await update_state(ctx, "group/repo-a", "2025-06-01T12:00:00", True)
        await update_state(ctx, "group/repo-a", "2025-06-01T13:00:00", True)
        await update_state(ctx, "group/repo-a", "2025-06-01T14:00:00", False)

        state = json.loads(temp_state_file.read_text(encoding="utf-8"))
        repo = state["repositories"]["group/repo-a"]
        assert repo["success_count"] == 2
        assert repo["failure_count"] == 1
        assert repo["last_indexed_at"] == "2025-06-01T14:00:00"


# ---------------------------------------------------------------------------
# Agent tool: fetch_new_mrs
# ---------------------------------------------------------------------------


class TestFetchNewMrs:
    @patch("agents.continuous_indexer.GitlabAdapter")
    async def test_returns_all_mrs_when_no_since(self, mock_adapter_cls):
        mock_adapter = MagicMock()
        mock_adapter.fetch_mrs.return_value = [
            {"id": 1, "merged_at": "2025-01-01T00:00:00Z"},
            {"id": 2, "merged_at": "2025-01-02T00:00:00Z"},
        ]
        mock_adapter_cls.return_value = mock_adapter

        ctx = _make_run_context({"gitlab": {"base_url": "https://gitlab.example.com"}})

        with patch.dict("os.environ", {"GITLAB_PRIVATE_TOKEN": "fake-token"}):
            result = await fetch_new_mrs(ctx, "group", "repo-a", since=None)

        assert len(result) == 2

    @patch("agents.continuous_indexer.GitlabAdapter")
    async def test_filters_by_since_timestamp(self, mock_adapter_cls):
        mock_adapter = MagicMock()
        # GitLab API now does the filtering, so mock returns only filtered results
        mock_adapter.fetch_mrs.return_value = [
            {"id": 2, "merged_at": "2025-06-15T00:00:00Z"},
        ]
        mock_adapter_cls.return_value = mock_adapter

        ctx = _make_run_context({"gitlab": {"base_url": "https://gitlab.example.com"}})

        with patch.dict("os.environ", {"GITLAB_PRIVATE_TOKEN": "fake-token"}):
            result = await fetch_new_mrs(
                ctx, "group", "repo-a", since="2025-06-01T00:00:00+00:00"
            )

        # Verify fetch_mrs was called with updated_after parameter
        mock_adapter.fetch_mrs.assert_called_once_with(
            state="merged", max_mrs=50, updated_after="2025-06-01T00:00:00+00:00"
        )
        assert len(result) == 1
        assert result[0]["id"] == 2

    @patch("agents.continuous_indexer.GitlabAdapter")
    async def test_empty_when_no_new_mrs(self, mock_adapter_cls):
        mock_adapter = MagicMock()
        # GitLab API filtering returns empty list when no new MRs
        mock_adapter.fetch_mrs.return_value = []
        mock_adapter_cls.return_value = mock_adapter

        ctx = _make_run_context({"gitlab": {}})

        with patch.dict("os.environ", {"GITLAB_PRIVATE_TOKEN": "fake-token"}):
            result = await fetch_new_mrs(
                ctx, "group", "repo-a", since="2025-12-01T00:00:00+00:00"
            )

        assert result == []

    async def test_raises_without_gitlab_token(self):
        ctx = _make_run_context({"gitlab": {}})
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GITLAB_PRIVATE_TOKEN"):
                await fetch_new_mrs(ctx, "group", "repo-a")


# ---------------------------------------------------------------------------
# IndexingResult model
# ---------------------------------------------------------------------------


class TestIndexingResult:
    def test_healthy_result(self):
        result = IndexingResult(
            indexed_count=10,
            failed_repos=[],
            last_indexed_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            health_status="healthy",
            action_needed=None,
        )
        assert result.health_status == "healthy"
        assert result.action_needed is None

    def test_degraded_result_with_failures(self):
        result = IndexingResult(
            indexed_count=5,
            failed_repos=["group/repo-b"],
            last_indexed_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            health_status="degraded",
            action_needed="Check GitLab token for group/repo-b",
        )
        assert result.health_status == "degraded"
        assert "repo-b" in result.failed_repos[0]
        assert result.action_needed is not None

    def test_failing_result(self):
        result = IndexingResult(
            indexed_count=0,
            failed_repos=["group/repo-a", "group/repo-b"],
            last_indexed_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            health_status="failing",
            action_needed="All repositories failing",
        )
        assert result.health_status == "failing"
        assert len(result.failed_repos) == 2


# ---------------------------------------------------------------------------
# run_continuous_indexing
# ---------------------------------------------------------------------------


class TestRunContinuousIndexing:
    def _mock_agent_result(self, **kwargs):
        defaults = {
            "indexed_count": 0,
            "failed_repos": [],
            "last_indexed_at": datetime(2025, 6, 1, tzinfo=timezone.utc),
            "health_status": "healthy",
            "action_needed": None,
        }
        defaults.update(kwargs)
        mock_result = MagicMock()
        mock_result.data = IndexingResult(**defaults)
        return mock_result

    @patch("agents.continuous_indexer._get_agent")
    async def test_run_once_exits_after_one_iteration(
        self, mock_get_agent, mock_config
    ):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(
            return_value=self._mock_agent_result(indexed_count=3)
        )
        mock_get_agent.return_value = mock_agent

        await run_continuous_indexing(mock_config, run_once=True)

        mock_agent.run.assert_called_once()

    @patch("agents.continuous_indexer._get_agent")
    async def test_run_once_handles_agent_exception(self, mock_get_agent, mock_config):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("LLM error"))
        mock_get_agent.return_value = mock_agent

        await run_continuous_indexing(mock_config, run_once=True)

        mock_agent.run.assert_called_once()

    @patch("agents.continuous_indexer.asyncio.sleep", new_callable=AsyncMock)
    @patch("agents.continuous_indexer._get_agent")
    async def test_continuous_mode_loops_until_cancelled(
        self, mock_get_agent, mock_sleep, mock_config
    ):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=self._mock_agent_result())
        mock_get_agent.return_value = mock_agent
        mock_sleep.side_effect = [None, asyncio.CancelledError()]

        with pytest.raises(asyncio.CancelledError):
            await run_continuous_indexing(mock_config, interval_minutes=1)

        assert mock_agent.run.call_count == 2
        mock_sleep.assert_called_with(60)

    @patch("agents.continuous_indexer._get_agent")
    async def test_run_passes_repo_info_to_agent(self, mock_get_agent, mock_config):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=self._mock_agent_result())
        mock_get_agent.return_value = mock_agent

        await run_continuous_indexing(mock_config, run_once=True)

        call_args = mock_agent.run.call_args
        prompt = call_args[0][0]
        assert "2 repositories" in prompt
        assert "group/repo-a" in prompt
        assert "group/repo-b" in prompt
        assert call_args[1]["deps"] == {"config": mock_config}
