from code_review_agent.github_client import added_lines, parse_patch_lines
from code_review_agent.models import Finding, ReviewReport, Severity, dedupe_findings, sort_findings
from code_review_agent.report import inline_comments, render_markdown, review_event
from code_review_agent.specialists import numbered
from code_review_agent.tools import related_tests

PATCH = """@@ -10,4 +10,5 @@ def f():
 context10
-removed
+added11
+added12
 context13
\\ No newline at end of file
@@ -40,2 +41,2 @@
-old
+new41
 context42"""


def finding(sev, file="a.py", line=1, category="bug", title="t"):
    return Finding(file=file, line=line, severity=sev, category=category, title=title, detail="d", suggestion="s")


def test_parse_patch_lines():
    assert parse_patch_lines(PATCH) == {10, 11, 12, 13, 41, 42}
    assert added_lines(PATCH) == {11, 12, 41}
    assert parse_patch_lines(None) == set()


def test_sort_and_dedupe():
    items = [finding("low", line=5), finding("critical", line=9), finding("high", line=5)]
    assert [f.severity for f in sort_findings(items)] == [Severity.CRITICAL, Severity.HIGH, Severity.LOW]
    deduped = dedupe_findings(items)
    assert len(deduped) == 2 and deduped[1].severity == Severity.HIGH


def test_inline_comments_only_for_diff_lines():
    items = [finding("high", line=11), finding("low", line=99), finding("medium", line=None),
             finding("high", file="other.py", line=11)]
    comments = inline_comments(items, {"a.py": PATCH})
    assert [(c["path"], c["line"], c["side"]) for c in comments] == [("a.py", 11, "RIGHT")]


def test_render_and_event():
    report = ReviewReport(summary="Adds a runner.", verdict="request_changes",
                          findings=[finding("info", title="note"), finding("critical", title="RCE")])
    md = render_markdown(report, skipped=["logo.png"])
    assert md.index("RCE") < md.index("note")
    assert "| Critical | 1 |" in md and "logo.png" in md
    assert review_event(report) == "REQUEST_CHANGES"
    assert review_event(ReviewReport(summary="", verdict="comment", findings=[finding("low")])) == "COMMENT"


def test_numbered_trims_long_files_around_changes():
    source = "\n".join(f"line{i}" for i in range(1, 1001))
    out = numbered(source, {500}, max_lines=400, window=2)
    assert "  500 | line500" in out and "  497 |" not in out and "..." in out


def test_related_tests():
    tree = ["src/pkg/parser.py", "tests/test_parser.py", "tests/test_other.py", "web/parser.spec.ts"]
    assert related_tests("src/pkg/parser.py", tree) == ["tests/test_parser.py", "web/parser.spec.ts"]
