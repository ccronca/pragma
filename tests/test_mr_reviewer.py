"""Unit tests for the MR reviewer agent."""

from unittest.mock import patch


from agents.mr_reviewer import (
    _build_review_agent,
    _get_mr_review_info,
    _mark_mr_reviewed,
    _notify,
    _save_review,
)

_SAMPLE_MR = {
    "id": 42,
    "title": "feat: add retry logic",
    "description": "Adds exponential backoff to the API client.",
    "diff": "--- a/client.py\n+++ b/client.py\n@@ -1 +1,5 @@\n+import time\n",
    "author": "alice",
    "created_at": "2024-01-01T00:00:00Z",
    "web_url": "https://gitlab.example.com/mr/42",
    "repo_owner": "group",
    "repo_name": "my-repo",
}

_SAMPLE_CONFIG = {
    "agent": {"provider": "gemini", "model": "gemini-2.5-flash-lite"},
}


class TestSaveReview:
    def test_creates_file_in_reviews_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        review_path = _save_review(
            "## Review\n\nLooks good.", _SAMPLE_MR, "group/my-repo"
        )
        assert review_path.exists()
        assert review_path.suffix == ".md"
        assert "my-repo" in review_path.name
        assert "mr42" in review_path.name

    def test_file_contains_review_text(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        review_path = _save_review("## LGTM\n\nNo issues.", _SAMPLE_MR, "group/my-repo")
        content = review_path.read_text(encoding="utf-8")
        assert "## LGTM" in content
        assert "No issues." in content

    def test_file_header_contains_mr_metadata(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        review_path = _save_review("review text", _SAMPLE_MR, "group/my-repo")
        content = review_path.read_text(encoding="utf-8")
        assert "feat: add retry logic" in content
        assert "!42" in content
        assert "alice" in content


class TestNotify:
    def test_silently_skips_when_notify_send_missing(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            _notify("Title", "Body")  # should not raise

    def test_silently_skips_on_nonzero_exit(self):
        import subprocess

        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "notify-send"),
        ):
            _notify("Title", "Body")  # should not raise

    def test_calls_notify_send_with_correct_args(self):
        with patch("subprocess.run") as mock_run:
            _notify("Review ready: MR !42", "feat: add retry logic")
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == "notify-send"
            assert "--app-name=Pragma" in args
            assert "Review ready: MR !42" in args
            assert "feat: add retry logic" in args


class TestReviewState:
    def test_get_mr_review_info_none_when_no_state_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _get_mr_review_info("group/repo", 42) is None

    def test_mark_mr_reviewed_persists_info(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _mark_mr_reviewed("group/repo", 42, "2024-01-02T00:00:00Z", "reviews/r.md")
        info = _get_mr_review_info("group/repo", 42)
        assert info is not None
        assert info["updated_at"] == "2024-01-02T00:00:00Z"
        assert info["review_file"] == "reviews/r.md"

    def test_mark_mr_reviewed_overwrites_on_re_review(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _mark_mr_reviewed("group/repo", 42, "2024-01-01T00:00:00Z", "reviews/v1.md")
        _mark_mr_reviewed("group/repo", 42, "2024-01-02T00:00:00Z", "reviews/v2.md")
        info = _get_mr_review_info("group/repo", 42)
        assert info["updated_at"] == "2024-01-02T00:00:00Z"
        assert info["review_file"] == "reviews/v2.md"

    def test_review_info_is_repo_scoped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _mark_mr_reviewed("group/repo-a", 1, "2024-01-01T00:00:00Z", "a/r.md")
        _mark_mr_reviewed("group/repo-b", 2, "2024-01-01T00:00:00Z", "b/r.md")
        assert _get_mr_review_info("group/repo-a", 1) is not None
        assert _get_mr_review_info("group/repo-a", 2) is None
        assert _get_mr_review_info("group/repo-b", 2) is not None

    def test_migrates_legacy_reviewed_mr_ids_format(self, tmp_path, monkeypatch):
        import json

        monkeypatch.chdir(tmp_path)
        state_file = tmp_path / "data" / "review_state.json"
        state_file.parent.mkdir(parents=True)
        state_file.write_text(
            json.dumps({"repositories": {"group/repo": {"reviewed_mr_ids": [10, 20]}}}),
            encoding="utf-8",
        )
        # Legacy MR should be found after migration
        assert _get_mr_review_info("group/repo", 10) is not None
        assert _get_mr_review_info("group/repo", 20) is not None
        # Legacy MR not in list should return None
        assert _get_mr_review_info("group/repo", 99) is None


class TestBuildReviewAgent:
    def test_gemini_provider_prefixes_model(self):
        config = {"agent": {"provider": "gemini", "model": "gemini-2.5-flash-lite"}}
        with patch("agents.mr_reviewer.Agent") as mock_agent:
            _build_review_agent(config)
            model_str = mock_agent.call_args[0][0]
            assert model_str == "google-gla:gemini-2.5-flash-lite"

    def test_ollama_provider_sets_env_and_openai_prefix(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = {
            "agent": {
                "provider": "ollama",
                "model": "mistral:7b",
                "base_url": "http://localhost:11434/v1",
            }
        }
        with patch("agents.mr_reviewer.Agent") as mock_agent:
            import os

            _build_review_agent(config)
            assert os.environ.get("OPENAI_BASE_URL") == "http://localhost:11434/v1"
            assert os.environ.get("OPENAI_API_KEY") == "ollama"
            model_str = mock_agent.call_args[0][0]
            assert model_str == "openai:mistral:7b"

    def test_ollama_provider_with_existing_openai_prefix(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = {"agent": {"provider": "ollama", "model": "openai:mistral:7b"}}
        with patch("agents.mr_reviewer.Agent") as mock_agent:
            _build_review_agent(config)
            model_str = mock_agent.call_args[0][0]
            assert model_str == "openai:mistral:7b"

    def test_unknown_provider_passes_model_through(self):
        config = {"agent": {"provider": "custom", "model": "my-model"}}
        with patch("agents.mr_reviewer.Agent") as mock_agent:
            _build_review_agent(config)
            model_str = mock_agent.call_args[0][0]
            assert model_str == "my-model"
