"""Pydantic models shared by the specialists, the orchestrator and the report."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def rank(self) -> int:
        return list(Severity).index(self)


Category = Literal["bug", "security", "testing"]


class Finding(BaseModel):
    file: str = Field(description="Repository-relative path of the file, exactly as in the PR")
    line: int | None = Field(
        default=None, description="Line number in the NEW version of the file, or null if not line-specific"
    )
    severity: Severity
    category: Category
    title: str = Field(description="One-line summary of the problem")
    detail: str = Field(description="Why this is a problem, with reference to the code")
    suggestion: str = Field(description="Concrete fix or, for testing findings, the test case to add")


class FindingList(BaseModel):
    """Structured output of a specialist reviewer."""

    findings: list[Finding] = Field(default_factory=list)


class ReviewReport(BaseModel):
    """Final, prioritized report produced by the orchestrator."""

    summary: str = Field(description="2-4 sentence overview of the PR and the most important risks")
    verdict: Literal["approve", "comment", "request_changes"]
    findings: list[Finding] = Field(default_factory=list)


def sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (f.severity.rank, f.file, f.line or 0))


def dedupe_findings(findings: list[Finding]) -> list[Finding]:
    """Drop findings that point at the same file, line and category; keep the most severe."""
    best: dict[tuple, Finding] = {}
    for f in sort_findings(findings):
        key = (f.file, f.line, f.category, f.title.strip().lower() if f.line is None else "")
        best.setdefault(key, f)
    return sort_findings(list(best.values()))
