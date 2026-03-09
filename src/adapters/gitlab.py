import typer
import gitlab
from typing import List, Dict


class GitlabAdapter:
    def __init__(self, base_url: str, private_token: str, owner: str, name: str):
        self.base_url = base_url
        self.private_token = private_token
        self.owner = owner
        self.name = name

        # Initialize GitLab client
        try:
            self.gl = gitlab.Gitlab(url=self.base_url, private_token=self.private_token)
            self.gl.auth()
            typer.echo(f"Successfully authenticated to GitLab at {self.base_url}")
        except gitlab.exceptions.GitlabAuthenticationError as e:
            typer.echo(
                f"Error: Authentication failed. Please check your GitLab token. {e}",
                err=True,
            )
            raise typer.Exit(code=1)

        # Get the project (handle nested groups)
        project_path = f"{owner}/{name}"
        typer.echo(f"Looking for project: {project_path}")
        try:
            self.project = self.gl.projects.get(project_path)
            typer.echo(
                f"Successfully connected to project: {self.project.name} (ID: {self.project.id})"
            )
        except gitlab.exceptions.GitlabGetError as e:
            typer.echo(f"Error: Could not find project '{project_path}'. {e}", err=True)
            raise typer.Exit(code=1)

    def fetch_mrs(self, state: str = "merged", max_mrs: int = 50) -> List[Dict]:
        """
        Fetch merge requests from GitLab.

        Args:
            state: State of MRs to fetch ('merged', 'opened', 'closed', 'all')
            max_mrs: Maximum number of MRs to fetch

        Returns:
            List of dictionaries containing MR data
        """
        typer.echo(
            f"Fetching {state} Merge Requests from GitLab for {self.owner}/{self.name}..."
        )

        merge_requests = []

        try:
            # Fetch MRs with specified state
            mrs = self.project.mergerequests.list(
                state=state,
                order_by="updated_at",
                sort="desc",
                per_page=100,
                get_all=False,
            )

            # Limit to max_mrs
            mrs = mrs[:max_mrs]
            typer.echo(f"Found {len(mrs)} merge requests to process...")

            for idx, mr in enumerate(mrs, 1):
                typer.echo(f"Processing MR {idx}/{len(mrs)}: !{mr.iid} - {mr.title}")

                # Fetch full MR details
                mr_full = self.project.mergerequests.get(mr.iid)

                # Fetch diff (changes)
                try:
                    changes = mr_full.changes()
                    diff_text = self._format_changes_to_diff(changes)
                except Exception as e:
                    typer.echo(f"  Warning: Could not fetch diff for MR !{mr.iid}: {e}")
                    diff_text = ""

                # Fetch discussions/comments
                discussions = self._fetch_discussions(mr_full)

                merge_request_data = {
                    "id": mr.iid,
                    "title": mr.title,
                    "description": mr.description or "",
                    "diff": diff_text,
                    "discussions": discussions,
                    "author": mr.author.get("username", "unknown"),
                    "created_at": mr.created_at,
                    "merged_at": mr.merged_at if hasattr(mr, "merged_at") else None,
                    "web_url": mr.web_url,
                }

                merge_requests.append(merge_request_data)

            typer.echo(f"Successfully fetched {len(merge_requests)} merge requests.")
            return merge_requests

        except gitlab.exceptions.GitlabError as e:
            typer.echo(f"Error fetching merge requests: {e}", err=True)
            raise typer.Exit(code=1)

    def _format_changes_to_diff(self, changes: Dict) -> str:
        """Format GitLab changes API response to unified diff format."""
        diff_parts = []

        for change in changes.get("changes", []):
            # Add file header
            old_path = change.get("old_path", "")
            new_path = change.get("new_path", "")
            diff_text = change.get("diff", "")

            if diff_text:
                file_header = f"diff --git a/{old_path} b/{new_path}\n"
                diff_parts.append(file_header + diff_text)

        return "\n".join(diff_parts)

    def _fetch_discussions(self, mr) -> List[Dict]:
        """Fetch all discussions/comments for a merge request."""
        discussions = []

        try:
            mr_discussions = mr.discussions.list(get_all=True)

            for discussion in mr_discussions:
                for note in discussion.attributes.get("notes", []):
                    # Only include text notes (exclude system notes)
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
            typer.echo(f"  Warning: Could not fetch discussions: {e}")

        return discussions
