"""Unit tests for indexer.core pure functions."""

from unittest.mock import MagicMock

from indexer.core import _build_documents, _get_indexed_mr_ids


MR_FIXTURE = {
    "id": 42,
    "title": "feat: add retry logic",
    "description": "Adds exponential backoff to the Jira collector.",
    "diff": "--- a/collector.py\n+++ b/collector.py\n+import time\n",
    "discussions": [
        {
            "author": "alice",
            "note": "Should we cap the retries?",
            "created_at": "2024-01-01",
        },
        {
            "author": "bob",
            "note": "Yes, 3 seems reasonable.",
            "created_at": "2024-01-02",
        },
    ],
    "repo_owner": "group",
    "repo_name": "repo",
    "author": "alice",
    "created_at": "2024-01-01T00:00:00Z",
    "merged_at": "2024-01-02T00:00:00Z",
    "web_url": "https://gitlab.example.com/group/repo/-/merge_requests/42",
}


class TestGetIndexedMrIds:
    def test_empty_collection_returns_empty_set(self):
        collection = MagicMock()
        collection.count.return_value = 0
        result = _get_indexed_mr_ids(collection)
        assert result == set()

    def test_returns_mr_ids_from_metadata(self):
        collection = MagicMock()
        collection.count.return_value = 3
        collection.get.return_value = {
            "metadatas": [
                {"mr_id": 1},
                {"mr_id": 2},
                {"mr_id": 3},
            ]
        }
        result = _get_indexed_mr_ids(collection)
        assert result == {1, 2, 3}

    def test_skips_metadata_without_mr_id(self):
        collection = MagicMock()
        collection.count.return_value = 2
        collection.get.return_value = {"metadatas": [{"mr_id": 10}, {}, None]}
        result = _get_indexed_mr_ids(collection)
        assert result == {10}

    def test_filters_by_repository(self):
        collection = MagicMock()
        collection.count.return_value = 3
        collection.get.return_value = {
            "metadatas": [
                {"mr_id": 1, "repo_owner": "group", "repo_name": "repo-a"},
                {"mr_id": 2, "repo_owner": "group", "repo_name": "repo-a"},
            ]
        }
        result = _get_indexed_mr_ids(collection, repo_owner="group", repo_name="repo-a")
        assert result == {1, 2}
        collection.get.assert_called_once_with(
            include=["metadatas"],
            where={
                "$and": [
                    {"repo_owner": {"$eq": "group"}},
                    {"repo_name": {"$eq": "repo-a"}},
                ]
            },
        )

    def test_no_filter_returns_all_mr_ids(self):
        collection = MagicMock()
        collection.count.return_value = 3
        collection.get.return_value = {
            "metadatas": [
                {"mr_id": 1, "repo_owner": "group", "repo_name": "repo-a"},
                {"mr_id": 2, "repo_owner": "other", "repo_name": "repo-b"},
            ]
        }
        result = _get_indexed_mr_ids(collection)
        assert result == {1, 2}
        collection.get.assert_called_once_with(include=["metadatas"], where=None)


class TestBuildDocuments:
    def test_creates_diff_and_discussion_documents(self):
        docs = _build_documents([MR_FIXTURE], already_indexed=set())
        content_types = {d.metadata["content_type"] for d in docs}
        assert content_types == {"diff", "discussion"}

    def test_skips_already_indexed_mrs(self):
        docs = _build_documents([MR_FIXTURE], already_indexed={42})
        assert docs == []

    def test_skips_diff_document_when_no_diff(self):
        mr = {**MR_FIXTURE, "diff": ""}
        docs = _build_documents([mr], already_indexed=set())
        assert all(d.metadata["content_type"] != "diff" for d in docs)

    def test_skips_discussion_document_when_no_discussions(self):
        mr = {**MR_FIXTURE, "discussions": []}
        docs = _build_documents([mr], already_indexed=set())
        assert all(d.metadata["content_type"] != "discussion" for d in docs)

    def test_metadata_is_set_correctly(self):
        docs = _build_documents([MR_FIXTURE], already_indexed=set())
        for doc in docs:
            assert doc.metadata["mr_id"] == 42
            assert doc.metadata["mr_title"] == "feat: add retry logic"
            assert doc.metadata["repo_owner"] == "group"
            assert doc.metadata["repo_name"] == "repo"
            assert doc.metadata["author"] == "alice"
            assert doc.metadata["web_url"] == MR_FIXTURE["web_url"]

    def test_description_truncated_to_500_chars(self):
        mr = {**MR_FIXTURE, "description": "x" * 600}
        docs = _build_documents([mr], already_indexed=set())
        for doc in docs:
            assert len(doc.metadata["mr_description"]) == 500

    def test_diff_document_contains_diff_text(self):
        docs = _build_documents([MR_FIXTURE], already_indexed=set())
        diff_doc = next(d for d in docs if d.metadata["content_type"] == "diff")
        assert "Code Changes (Diff):" in diff_doc.text
        assert MR_FIXTURE["diff"] in diff_doc.text

    def test_discussion_document_contains_notes(self):
        docs = _build_documents([MR_FIXTURE], already_indexed=set())
        disc_doc = next(d for d in docs if d.metadata["content_type"] == "discussion")
        assert "Review Discussion:" in disc_doc.text
        assert "alice" in disc_doc.text
        assert "Should we cap the retries?" in disc_doc.text

    def test_multiple_mrs_partially_indexed(self):
        mr2 = {**MR_FIXTURE, "id": 99, "title": "fix: something"}
        docs = _build_documents([MR_FIXTURE, mr2], already_indexed={42})
        mr_ids = {d.metadata["mr_id"] for d in docs}
        assert mr_ids == {99}

    def test_multi_repo_documents_have_distinct_metadata(self):
        mr1 = {**MR_FIXTURE, "id": 1, "repo_owner": "team-a", "repo_name": "frontend"}
        mr2 = {**MR_FIXTURE, "id": 2, "repo_owner": "team-b", "repo_name": "backend"}
        docs = _build_documents([mr1, mr2], already_indexed=set())
        owners = {d.metadata["repo_owner"] for d in docs}
        names = {d.metadata["repo_name"] for d in docs}
        assert owners == {"team-a", "team-b"}
        assert names == {"frontend", "backend"}
