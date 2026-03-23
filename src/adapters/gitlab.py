import logging

import gitlab

logger = logging.getLogger(__name__)


class GitlabAdapter:
    def __init__(self, base_url: str, private_token: str, owner: str, name: str):
        self.base_url = base_url
        self.private_token = private_token
        self.owner = owner
        self.name = name

        try:
            self.gl = gitlab.Gitlab(url=self.base_url, private_token=self.private_token)
            self.gl.auth()
            logger.info("Authenticated to GitLab at %s", self.base_url)
        except gitlab.exceptions.GitlabAuthenticationError as e:
            raise RuntimeError(
                f"Authentication failed. Please check your GitLab token: {e}"
            ) from e

        project_path = f"{owner}/{name}"
        logger.info("Looking for project: %s", project_path)
        try:
            self.project = self.gl.projects.get(project_path)
            logger.info(
                "Connected to project: %s (ID: %s)", self.project.name, self.project.id
            )
        except gitlab.exceptions.GitlabGetError as e:
            raise RuntimeError(f"Could not find project '{project_path}': {e}") from e

    def fetch_mrs(
        self, state: str = "merged", max_mrs: int = 50, updated_after: str = None
    ) -> list[dict]:
        """
        Fetch merge requests from GitLab.

        Args:
            state: State of MRs to fetch ('merged', 'opened', 'closed', 'all')
            max_mrs: Maximum number of MRs to fetch
            updated_after: ISO 8601 timestamp to filter MRs updated after this time

        Returns:
            List of dictionaries containing MR data
        """
        logger.info(
            "Fetching %s merge requests from %s/%s", state, self.owner, self.name
        )

        try:
            kwargs = {
                "state": state,
                "order_by": "updated_at",
                "sort": "desc",
                "per_page": 100,
                "get_all": True,
            }
            if updated_after:
                kwargs["updated_after"] = updated_after
                logger.info("Filtering MRs updated after %s", updated_after)

            mrs = self.project.mergerequests.list(**kwargs)
            mrs = mrs[:max_mrs]
            logger.info("Found %d merge requests to process", len(mrs))

            merge_requests = []
            for idx, mr in enumerate(mrs, 1):
                logger.info(
                    "Processing MR %d/%d: !%s - %s", idx, len(mrs), mr.iid, mr.title
                )

                mr_full = self.project.mergerequests.get(mr.iid)

                try:
                    changes = mr_full.changes()
                    diff_text = self._format_changes_to_diff(changes)
                except Exception as e:
                    logger.warning("Could not fetch diff for MR !%s: %s", mr.iid, e)
                    diff_text = ""

                merge_requests.append(
                    {
                        "id": mr.iid,
                        "title": mr.title,
                        "description": mr.description or "",
                        "diff": diff_text,
                        "discussions": self._fetch_discussions(mr_full),
                        "repo_owner": self.owner,
                        "repo_name": self.name,
                        "author": mr.author.get("username", "unknown"),
                        "created_at": mr.created_at,
                        "merged_at": mr.merged_at if hasattr(mr, "merged_at") else None,
                        "web_url": mr.web_url,
                    }
                )

            logger.info("Fetched %d merge requests", len(merge_requests))
            return merge_requests

        except gitlab.exceptions.GitlabError as e:
            raise RuntimeError(f"Error fetching merge requests: {e}") from e

    def fetch_mr(self, mr_iid: int) -> dict:
        """Fetch a single merge request by IID.

        Args:
            mr_iid: The merge request IID (project-scoped number shown in GitLab UI).

        Returns:
            Dictionary with MR data (same schema as fetch_mrs results).
        """
        try:
            mr_full = self.project.mergerequests.get(mr_iid)
        except gitlab.exceptions.GitlabGetError as e:
            raise RuntimeError(f"Could not find MR !{mr_iid}: {e}") from e

        try:
            changes = mr_full.changes()
            diff_text = self._format_changes_to_diff(changes)
        except Exception as e:
            logger.warning("Could not fetch diff for MR !%s: %s", mr_iid, e)
            diff_text = ""

        return {
            "id": mr_full.iid,
            "title": mr_full.title,
            "description": mr_full.description or "",
            "diff": diff_text,
            "discussions": self._fetch_discussions(mr_full),
            "repo_owner": self.owner,
            "repo_name": self.name,
            "author": mr_full.author.get("username", "unknown"),
            "created_at": mr_full.created_at,
            "merged_at": mr_full.merged_at if hasattr(mr_full, "merged_at") else None,
            "web_url": mr_full.web_url,
        }

    def _format_changes_to_diff(self, changes: dict) -> str:
        """Format GitLab changes API response to unified diff format."""
        diff_parts = []

        for change in changes.get("changes", []):
            old_path = change.get("old_path", "")
            new_path = change.get("new_path", "")
            diff_text = change.get("diff", "")

            if diff_text:
                diff_parts.append(f"diff --git a/{old_path} b/{new_path}\n{diff_text}")

        return "\n".join(diff_parts)

    def _fetch_discussions(self, mr) -> list[dict]:
        """Fetch all non-system discussion notes for a merge request."""
        discussions = []

        try:
            for discussion in mr.discussions.list(get_all=True):
                for note in discussion.attributes.get("notes", []):
                    if not note.get("system", False):
                        discussions.append(
                            {
                                "author": note.get("author", {}).get(
                                    "username", "unknown"
                                ),
                                "note": note.get("body", ""),
                                "created_at": note.get("created_at", ""),
                            }
                        )
        except Exception as e:
            logger.warning("Could not fetch discussions: %s", e)

        return discussions
