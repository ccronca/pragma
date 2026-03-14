"""Unit tests for GitlabAdapter.fetch_mrs public behavior."""

from unittest.mock import MagicMock, patch

import gitlab
import pytest

from adapters.gitlab import GitlabAdapter


def _make_adapter() -> GitlabAdapter:
    """Build a GitlabAdapter without making real network calls."""
    with patch("gitlab.Gitlab") as mock_gl_class:
        mock_gl = MagicMock()
        mock_gl_class.return_value = mock_gl
        mock_gl.projects.get.return_value = MagicMock(name="test-repo", id=1)
        adapter = GitlabAdapter(
            base_url="https://gitlab.example.com",
            private_token="fake-token",
            owner="group",
            name="repo",
        )
    return adapter


def _make_mr_stub(
    iid: int = 1,
    title: str = "feat: something",
    description: str = "A description",
    author: str = "alice",
    changes: dict | None = None,
    discussions: list | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Return a (mr_list_item, mr_full) pair for use in mergerequests mocks.

    mr_list_item simulates what mergerequests.list() returns.
    mr_full simulates what mergerequests.get() returns.
    """
    if changes is None:
        changes = {
            "changes": [
                {
                    "old_path": "src/api.py",
                    "new_path": "src/api.py",
                    "diff": "@@ -1,3 +1,4 @@\n+import logging\n",
                }
            ]
        }
    if discussions is None:
        discussions = []

    mr_item = MagicMock()
    mr_item.iid = iid
    mr_item.title = title
    mr_item.description = description
    mr_item.author = {"username": author}
    mr_item.created_at = "2024-01-01T00:00:00Z"
    mr_item.merged_at = "2024-01-02T00:00:00Z"
    mr_item.web_url = f"https://gitlab.example.com/mr/{iid}"

    mr_full = MagicMock()
    mr_full.changes.return_value = changes
    mr_full.discussions.list.return_value = discussions

    return mr_item, mr_full


class TestFetchMrsOutputShape:
    def test_returns_list_of_dicts_with_required_keys(self):
        adapter = _make_adapter()
        mr_item, mr_full = _make_mr_stub()
        adapter.project.mergerequests.list.return_value = [mr_item]
        adapter.project.mergerequests.get.return_value = mr_full

        results = adapter.fetch_mrs()

        assert len(results) == 1
        result = results[0]
        assert result["id"] == 1
        assert result["title"] == "feat: something"
        assert result["description"] == "A description"
        assert result["author"] == "alice"
        assert result["web_url"] == "https://gitlab.example.com/mr/1"
        assert result["repo_owner"] == "group"
        assert result["repo_name"] == "repo"
        assert "diff" in result
        assert "discussions" in result

    def test_diff_contains_formatted_unified_diff(self):
        adapter = _make_adapter()
        mr_item, mr_full = _make_mr_stub(
            changes={
                "changes": [
                    {
                        "old_path": "src/api.py",
                        "new_path": "src/api.py",
                        "diff": "@@ -1,3 +1,4 @@\n+import logging\n",
                    }
                ]
            }
        )
        adapter.project.mergerequests.list.return_value = [mr_item]
        adapter.project.mergerequests.get.return_value = mr_full

        result = adapter.fetch_mrs()[0]

        assert "diff --git a/src/api.py b/src/api.py" in result["diff"]
        assert "@@ -1,3 +1,4 @@" in result["diff"]

    def test_empty_changes_produces_empty_diff(self):
        adapter = _make_adapter()
        mr_item, mr_full = _make_mr_stub(changes={"changes": []})
        adapter.project.mergerequests.list.return_value = [mr_item]
        adapter.project.mergerequests.get.return_value = mr_full

        result = adapter.fetch_mrs()[0]

        assert result["diff"] == ""


class TestFetchMrsDiscussions:
    def _make_note(self, system: bool, author: str, body: str) -> dict:
        return {
            "system": system,
            "author": {"username": author},
            "body": body,
            "created_at": "2024-01-01",
        }

    def _make_discussion(self, notes: list[dict]) -> MagicMock:
        discussion = MagicMock()
        discussion.attributes = {"notes": notes}
        return discussion

    def test_includes_non_system_notes_in_discussions(self):
        adapter = _make_adapter()
        note = self._make_note(system=False, author="alice", body="LGTM")
        discussion = self._make_discussion([note])
        mr_item, mr_full = _make_mr_stub(discussions=[discussion])
        adapter.project.mergerequests.list.return_value = [mr_item]
        adapter.project.mergerequests.get.return_value = mr_full

        result = adapter.fetch_mrs()[0]

        assert len(result["discussions"]) == 1
        assert result["discussions"][0]["author"] == "alice"
        assert result["discussions"][0]["note"] == "LGTM"

    def test_excludes_system_notes_from_discussions(self):
        adapter = _make_adapter()
        note = self._make_note(system=True, author="gitlab", body="merged")
        discussion = self._make_discussion([note])
        mr_item, mr_full = _make_mr_stub(discussions=[discussion])
        adapter.project.mergerequests.list.return_value = [mr_item]
        adapter.project.mergerequests.get.return_value = mr_full

        result = adapter.fetch_mrs()[0]

        assert result["discussions"] == []

    def test_failed_discussions_fetch_returns_empty_list(self):
        adapter = _make_adapter()
        mr_item, mr_full = _make_mr_stub()
        mr_full.discussions.list.side_effect = Exception("network error")
        adapter.project.mergerequests.list.return_value = [mr_item]
        adapter.project.mergerequests.get.return_value = mr_full

        result = adapter.fetch_mrs()[0]

        assert result["discussions"] == []


class TestFetchMrsEdgeCases:
    def test_failed_diff_fetch_returns_empty_string(self):
        adapter = _make_adapter()
        mr_item, mr_full = _make_mr_stub()
        mr_full.changes.side_effect = Exception("API error")
        adapter.project.mergerequests.list.return_value = [mr_item]
        adapter.project.mergerequests.get.return_value = mr_full

        result = adapter.fetch_mrs()[0]

        assert result["diff"] == ""

    def test_gitlab_error_raises_runtime_error(self):
        adapter = _make_adapter()
        adapter.project.mergerequests.list.side_effect = gitlab.exceptions.GitlabError(
            "forbidden"
        )

        with pytest.raises(RuntimeError, match="Error fetching merge requests"):
            adapter.fetch_mrs()

    def test_respects_max_mrs_limit(self):
        adapter = _make_adapter()
        mr_items = [_make_mr_stub(iid=i)[0] for i in range(1, 6)]
        mr_full = _make_mr_stub()[1]
        adapter.project.mergerequests.list.return_value = mr_items
        adapter.project.mergerequests.get.return_value = mr_full

        results = adapter.fetch_mrs(max_mrs=3)

        assert len(results) == 3
