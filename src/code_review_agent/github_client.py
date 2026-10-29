"""Read-side access to a pull request via PyGithub, with caching and diff parsing."""

import base64
import re
from dataclasses import dataclass
from functools import cached_property

from github import Auth, Github, GithubException

HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def parse_patch_lines(patch: str | None) -> set[int]:
    """Line numbers in the new file that appear in the diff and so accept inline comments."""
    lines: set[int] = set()
    new_line = 0
    for raw in (patch or "").splitlines():
        if m := HUNK_HEADER.match(raw):
            new_line = int(m.group(1))
        elif raw.startswith("+") or raw.startswith(" "):
            lines.add(new_line)
            new_line += 1
        # "-" lines only exist on the old side; "\ No newline" is metadata.
    return lines


def added_lines(patch: str | None) -> set[int]:
    """Line numbers in the new file that the PR added or changed."""
    lines: set[int] = set()
    new_line = 0
    for raw in (patch or "").splitlines():
        if m := HUNK_HEADER.match(raw):
            new_line = int(m.group(1))
        elif raw.startswith("+"):
            lines.add(new_line)
            new_line += 1
        elif raw.startswith(" "):
            new_line += 1
    return lines


@dataclass
class ChangedFile:
    path: str
    status: str
    additions: int
    deletions: int
    patch: str | None
    sha: str | None = None  # blob sha of the new version


class PRContext:
    """One pull request. Tools and the poster talk to GitHub only through this class."""

    def __init__(self, repo_full_name: str, number: int, token: str):
        self._gh = Github(auth=Auth.Token(token))
        self.repo_full_name = repo_full_name
        self.number = number
        self.repo = self._gh.get_repo(repo_full_name)
        self.pr = self.repo.get_pull(number)
        self._content: dict[str, str | None] = {}

    @property
    def head_sha(self) -> str:
        return self.pr.head.sha

    def overview(self) -> dict:
        pr = self.pr
        return {
            "repo": self.repo_full_name,
            "number": pr.number,
            "title": pr.title,
            "body": (pr.body or "")[:3000],
            "author": pr.user.login,
            "base": pr.base.ref,
            "head": pr.head.ref,
            "head_sha": pr.head.sha,
            "changed_files": pr.changed_files,
            "additions": pr.additions,
            "deletions": pr.deletions,
            "url": pr.html_url,
        }

    @cached_property
    def files(self) -> list[ChangedFile]:
        return [
            ChangedFile(f.filename, f.status, f.additions, f.deletions, f.patch, f.sha)
            for f in self.pr.get_files()
        ]

    def file(self, path: str) -> ChangedFile | None:
        return next((f for f in self.files if f.path == path), None)

    def content(self, path: str) -> str | None:
        """File text at the PR head, or None if deleted, binary or missing.

        Reads through the git blobs API rather than the contents API: blob URLs carry no file
        extension (web filters such as Cisco Umbrella block e.g. `.py` downloads), and blobs stay
        reachable from the base repo even when the PR's fork has been deleted.
        """
        if path not in self._content:
            changed = self.file(path)
            sha = changed.sha if changed and changed.status != "removed" else self._tree.get(path)
            try:
                raw = base64.b64decode(self.repo.get_git_blob(sha).content) if sha else None
                self._content[path] = raw.decode("utf-8") if raw is not None else None
            except (GithubException, UnicodeDecodeError, AssertionError):
                self._content[path] = None
        return self._content[path]

    @cached_property
    def _tree(self) -> dict[str, str]:
        tree = self.repo.get_git_tree(self.head_sha, recursive=True)
        return {e.path: e.sha for e in tree.tree if e.type == "blob"}

    @property
    def tree_paths(self) -> list[str]:
        return list(self._tree)

    def create_review(self, body: str, event: str, comments: list[dict]) -> str:
        commit = self.repo.get_commit(self.head_sha)
        try:
            review = self.pr.create_review(commit=commit, body=body, event=event, comments=comments)
        except GithubException as e:
            # GitHub refuses REQUEST_CHANGES on your own PR; fall back to a plain comment review.
            if event == "COMMENT" or e.status != 422:
                raise
            review = self.pr.create_review(commit=commit, body=body, event="COMMENT", comments=comments)
        return review.html_url
