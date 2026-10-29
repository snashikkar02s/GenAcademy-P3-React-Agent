"""Tools available to the ReAct orchestrator, bound by closure to one PR."""

import json
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from langchain_core.tools import tool

from . import specialists
from .ast_tools import analyze_source
from .config import MAX_FILES, MAX_SOURCE_LINES
from .github_client import added_lines
from .models import Finding

LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".go": "go", ".rb": "ruby", ".rs": "rust", ".cs": "csharp", ".php": "php",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".kt": "kotlin", ".swift": "swift", ".sql": "sql",
    ".sh": "shell", ".ps1": "powershell", ".yml": "yaml", ".yaml": "yaml", ".json": "json",
    ".toml": "toml", ".md": "markdown", ".rst": "text", ".txt": "text", ".html": "html", ".css": "css",
}
MAX_DIFF_CHARS = 12_000
MAX_TEST_FILES = 3
MAX_TEST_LINES = 300


@dataclass
class FindingStore:
    """Specialist findings, kept outside the message history so none are lost in summarization."""

    findings: list[Finding] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    reviews_run: set[tuple[str, str]] = field(default_factory=set)


def language(path: str) -> str:
    return LANGUAGES.get(PurePosixPath(path).suffix.lower(), "other")


def is_test_path(path: str) -> bool:
    p = PurePosixPath(path)
    name = p.name.lower()
    return (name.startswith("test_") or p.stem.lower().endswith("_test") or ".test." in name
            or ".spec." in name or any(part in {"tests", "test", "__tests__"} for part in p.parts[:-1]))


def related_tests(path: str, candidates: list[str]) -> list[str]:
    stem = PurePosixPath(path).stem.lower().split(".")[0]
    if stem.startswith("test_"):
        stem = stem[5:]
    hits = []
    for c in candidates:
        if c == path or not is_test_path(c):
            continue
        cstem = PurePosixPath(c).stem.lower().split(".")[0]
        if stem and stem in cstem.removeprefix("test_").removesuffix("_test"):
            hits.append(c)
    return sorted(set(hits))[:5]


def make_tools(ctx, llm, store: FindingStore, max_files: int = MAX_FILES) -> list:
    """Build the tool set. `ctx` is a PRContext (or a test double with the same methods)."""

    reviewable = [
        f for f in sorted(ctx.files, key=lambda f: -(f.additions + f.deletions))
        if f.status != "removed" and f.patch
    ]
    in_scope = {f.path for f in reviewable[:max_files]}
    store.skipped = [f.path for f in ctx.files if f.path not in in_scope]
    ast_cache: dict[str, dict | None] = {}

    def check(path: str) -> str | None:
        if ctx.file(path) is None:
            return f"ERROR: {path!r} is not a changed file in this PR. Call list_changed_files."
        if path not in in_scope:
            return f"SKIPPED: {path!r} is out of scope (deleted, binary, or beyond the {max_files}-file limit)."
        return None

    def ast_for(path: str) -> dict | None:
        if path not in ast_cache:
            src = ctx.content(path) if language(path) == "python" else None
            ast_cache[path] = analyze_source(src) if src is not None else None
        return ast_cache[path]

    def source_for(path: str) -> str:
        src = ctx.content(path) or ""
        return specialists.numbered(src, added_lines(ctx.file(path).patch), MAX_SOURCE_LINES)

    def record(path: str, kind: str, findings: list[Finding]) -> str:
        store.findings.extend(findings)
        store.reviews_run.add((path, kind))
        return json.dumps({
            "file": path,
            "review": kind,
            "finding_count": len(findings),
            "findings": [{"line": f.line, "severity": f.severity.value, "title": f.title} for f in findings],
        })

    @tool
    def get_pr_overview() -> str:
        """Get the PR title, description, author, branches and size. Call this first."""
        return json.dumps(ctx.overview())

    @tool
    def list_changed_files() -> str:
        """List every changed file with status, line counts, language and whether it is in review scope."""
        return json.dumps([
            {"path": f.path, "status": f.status, "additions": f.additions, "deletions": f.deletions,
             "language": language(f.path), "is_test": is_test_path(f.path), "in_scope": f.path in in_scope}
            for f in ctx.files
        ])

    @tool
    def get_file_diff(path: str) -> str:
        """Get the unified diff for one changed file."""
        if (f := ctx.file(path)) is None:
            return f"ERROR: {path!r} is not a changed file in this PR."
        patch = f.patch or "(no textual diff: binary or too large)"
        return patch[:MAX_DIFF_CHARS] + ("\n... (truncated)" if len(patch) > MAX_DIFF_CHARS else "")

    @tool
    def get_file_content(path: str) -> str:
        """Get the numbered source of a changed file at the PR head (long files are trimmed around changes)."""
        if err := check(path):
            return err
        return source_for(path) or "(empty file)"

    @tool
    def analyze_python_ast(path: str) -> str:
        """Run static AST analysis on a changed Python file: functions, complexity and risky patterns
        (bare except, eval/exec, shell=True, unsafe deserialization, hardcoded secrets, mutable defaults)."""
        if err := check(path):
            return err
        if language(path) != "python":
            return f"Not a Python file ({language(path)}); AST analysis is only available for Python."
        return json.dumps(ast_for(path))

    @tool
    def find_related_tests(path: str) -> str:
        """Find existing or newly added test files that appear to test the given source file."""
        candidates = set(ctx.tree_paths) | {f.path for f in ctx.files}
        tests = related_tests(path, sorted(candidates))
        changed = {f.path for f in ctx.files}
        return json.dumps([{"path": t, "changed_in_pr": t in changed} for t in tests])

    @tool
    def run_bug_review(path: str) -> str:
        """Agent 1: review one changed file for bugs and anti-patterns. Findings are recorded for the report."""
        if err := check(path):
            return err
        found = specialists.review_bugs(llm, path, ctx.file(path).patch, source_for(path), ast_for(path))
        return record(path, "bugs", found)

    @tool
    def run_security_review(path: str) -> str:
        """Agent 2: review one changed file for security vulnerabilities. Findings are recorded for the report."""
        if err := check(path):
            return err
        found = specialists.review_security(llm, path, ctx.file(path).patch, source_for(path), ast_for(path))
        return record(path, "security", found)

    @tool
    def run_test_review(path: str) -> str:
        """Agent 3: evaluate test coverage of one changed source file and suggest missing tests.
        Findings are recorded for the report."""
        if err := check(path):
            return err
        candidates = set(ctx.tree_paths) | {f.path for f in ctx.files}
        tests = {}
        for t in related_tests(path, sorted(candidates))[:MAX_TEST_FILES]:
            if (text := ctx.content(t)) is not None:
                tests[t] = "\n".join(text.splitlines()[:MAX_TEST_LINES])
        found = specialists.review_tests(llm, path, ctx.file(path).patch, source_for(path), tests)
        return record(path, "tests", found)

    return [get_pr_overview, list_changed_files, get_file_diff, get_file_content, analyze_python_ast,
            find_related_tests, run_bug_review, run_security_review, run_test_review]
