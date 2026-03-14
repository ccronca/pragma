"""Unit tests for the continuous indexing functionality."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from agents.continuous_indexer import (
    _get_last_indexed_time,
    _load_state,
    _save_state,
    _update_state,
    run_continuous_indexing,
)


@pytest.fixture
def mock_config():
    return {
        "repositories": [
            {"type": "gitlab", "owner": "group", "name": "repo-a"},
            {"type": "gitlab", "owner": "group", "name": "repo-b"},
        ],
        "gitlab": {"base_url": "https://gitlab.example.com"},
        "vector_store": {"path": "./data/chroma_db"},
    }


@pytest.fixture
def temp_state_file(tmp_path, monkeypatch):
    """Point STATE_FILE to a temporary path."""
    state_path = tmp_path / "indexing_state.json"
    monkeypatch.setattr("agents.continuous_indexer.STATE_FILE", state_path)
    return state_path


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
# get_last_indexed_time
# ---------------------------------------------------------------------------


class TestGetLastIndexedTime:
    def test_returns_none_for_new_repo(self, temp_state_file):
        result = _get_last_indexed_time("group/repo-new")
        assert result is None

    def test_returns_timestamp_for_known_repo(self, temp_state_file):
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
        result = _get_last_indexed_time("group/repo-a")
        assert result == "2025-01-01T00:00:00"


# ---------------------------------------------------------------------------
# update_state
# ---------------------------------------------------------------------------


class TestUpdateState:
    def test_persists_success(self, temp_state_file):
        _update_state("group/repo-a", "2025-06-01T12:00:00", True)

        state = json.loads(temp_state_file.read_text(encoding="utf-8"))
        repo = state["repositories"]["group/repo-a"]
        assert repo["last_indexed_at"] == "2025-06-01T12:00:00"
        assert repo["success_count"] == 1
        assert repo["failure_count"] == 0

    def test_persists_failure(self, temp_state_file):
        _update_state("group/repo-a", "2025-06-01T12:00:00", False)

        state = json.loads(temp_state_file.read_text(encoding="utf-8"))
        repo = state["repositories"]["group/repo-a"]
        assert repo["failure_count"] == 1
        assert repo["success_count"] == 0

    def test_increments_existing_counts(self, temp_state_file):
        _update_state("group/repo-a", "2025-06-01T12:00:00", True)
        _update_state("group/repo-a", "2025-06-01T13:00:00", True)
        _update_state("group/repo-a", "2025-06-01T14:00:00", False)

        state = json.loads(temp_state_file.read_text(encoding="utf-8"))
        repo = state["repositories"]["group/repo-a"]
        assert repo["success_count"] == 2
        assert repo["failure_count"] == 1
        assert repo["last_indexed_at"] == "2025-06-01T14:00:00"


# ---------------------------------------------------------------------------
# run_continuous_indexing
# ---------------------------------------------------------------------------


class TestRunContinuousIndexing:
    @patch("agents.continuous_indexer.GitlabAdapter")
    @patch("agents.continuous_indexer.index_merge_requests")
    async def test_run_once_exits_after_one_iteration(
        self, mock_index, mock_adapter_cls, mock_config, temp_state_file
    ):
        mock_adapter = MagicMock()
        mock_adapter.fetch_mrs.return_value = []
        mock_adapter_cls.return_value = mock_adapter

        with patch.dict("os.environ", {"GITLAB_PRIVATE_TOKEN": "fake-token"}):
            await run_continuous_indexing(mock_config, run_once=True)

        # Should have checked both repos
        assert mock_adapter_cls.call_count == 2

    @patch("agents.continuous_indexer.GitlabAdapter")
    @patch("agents.continuous_indexer.index_merge_requests")
    async def test_indexes_new_mrs(
        self, mock_index, mock_adapter_cls, mock_config, temp_state_file
    ):
        mock_adapter = MagicMock()
        mock_adapter.fetch_mrs.return_value = [
            {"id": 1, "merged_at": "2025-01-01T00:00:00Z"},
            {"id": 2, "merged_at": "2025-01-02T00:00:00Z"},
        ]
        mock_adapter_cls.return_value = mock_adapter

        with patch.dict("os.environ", {"GITLAB_PRIVATE_TOKEN": "fake-token"}):
            await run_continuous_indexing(mock_config, run_once=True)

        # Should have called index_merge_requests twice (once per repo)
        assert mock_index.call_count == 2

    @patch("agents.continuous_indexer.GitlabAdapter")
    @patch("agents.continuous_indexer.index_merge_requests")
    async def test_handles_repository_failure(
        self, mock_index, mock_adapter_cls, mock_config, temp_state_file
    ):
        mock_adapter_cls.side_effect = RuntimeError("GitLab error")

        with patch.dict("os.environ", {"GITLAB_PRIVATE_TOKEN": "fake-token"}):
            await run_continuous_indexing(mock_config, run_once=True)

        # Should have updated state with failures
        state = _load_state()
        assert "group/repo-a" in state["repositories"]
        assert state["repositories"]["group/repo-a"]["failure_count"] == 1

    @patch("agents.continuous_indexer.asyncio.sleep")
    @patch("agents.continuous_indexer.GitlabAdapter")
    @patch("agents.continuous_indexer.index_merge_requests")
    async def test_continuous_mode_loops(
        self, mock_index, mock_adapter_cls, mock_sleep, mock_config, temp_state_file
    ):
        mock_adapter = MagicMock()
        mock_adapter.fetch_mrs.return_value = []
        mock_adapter_cls.return_value = mock_adapter
        mock_sleep.side_effect = [None, asyncio.CancelledError()]

        with pytest.raises(asyncio.CancelledError):
            with patch.dict("os.environ", {"GITLAB_PRIVATE_TOKEN": "fake-token"}):
                await run_continuous_indexing(mock_config, interval_minutes=1)

        # Should have run twice before cancellation
        assert mock_adapter_cls.call_count == 4  # 2 repos × 2 iterations
        mock_sleep.assert_called_with(60)

    async def test_raises_without_gitlab_token(self, mock_config):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GITLAB_PRIVATE_TOKEN"):
                await run_continuous_indexing(mock_config, run_once=True)

    @patch("agents.continuous_indexer.GitlabAdapter")
    @patch("agents.continuous_indexer.index_merge_requests")
    async def test_uses_last_indexed_timestamp(
        self, mock_index, mock_adapter_cls, mock_config, temp_state_file
    ):
        # Set up initial state with a timestamp
        initial_state = {
            "repositories": {
                "group/repo-a": {
                    "last_indexed_at": "2025-06-01T00:00:00+00:00",
                    "success_count": 1,
                    "failure_count": 0,
                }
            }
        }
        temp_state_file.write_text(json.dumps(initial_state), encoding="utf-8")

        mock_adapter = MagicMock()
        mock_adapter.fetch_mrs.return_value = []
        mock_adapter_cls.return_value = mock_adapter

        with patch.dict("os.environ", {"GITLAB_PRIVATE_TOKEN": "fake-token"}):
            await run_continuous_indexing(mock_config, run_once=True)

        # Verify fetch_mrs was called with updated_after parameter for repo-a
        calls = mock_adapter.fetch_mrs.call_args_list
        assert any(
            call[1].get("updated_after") == "2025-06-01T00:00:00+00:00"
            for call in calls
        )
