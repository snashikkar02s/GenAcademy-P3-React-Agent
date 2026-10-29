"""The three specialist reviewers. Each is a focused LLM call that returns structured findings."""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .models import Category, Finding, FindingList

COMMON_RULES = """
Rules:
- Only report problems you can point to in the code shown. No generic advice.
- Focus on lines the PR added or changed; mention pre-existing code only if the change makes it worse.
- `line` is the line number in the NEW file as shown in the numbered source. Use null if not line-specific.
- `file` must be exactly the path given.
- Severity: critical = exploitable or data loss; high = likely bug in normal use; medium = bug in edge
  cases or clear maintainability risk; low = minor; info = observation.
- Return an empty list if you find nothing worth reporting.
"""

BUG_PROMPT = """You are Agent 1, a senior engineer reviewing a pull request for BUGS and ANTI-PATTERNS:
logic errors, off-by-one, wrong conditions, unhandled None/empty cases, resource leaks, race conditions,
broken error handling, swallowed exceptions, mutable defaults, dead code, needless complexity, misleading names.
Do NOT report security vulnerabilities or missing tests; other agents cover those.""" + COMMON_RULES

SECURITY_PROMPT = """You are Agent 2, an application-security reviewer. Find SECURITY VULNERABILITIES:
injection (SQL, command, template, path traversal), unsafe deserialization, eval/exec on input,
hardcoded secrets, weak crypto or randomness, disabled TLS verification, missing authn/authz checks,
SSRF, XSS, sensitive data in logs, insecure defaults. Reference the CWE id in `detail` when one applies.
Do NOT report general bugs or style.""" + COMMON_RULES

TEST_PROMPT = """You are Agent 3, a test engineer. Evaluate TEST COVERAGE for the changed code:
which new or changed functions/branches have no test in the PR or the existing related tests, and which
edge cases are untested. Use category "testing". In `suggestion`, write the concrete test to add
(test name, inputs and expected result; a short pytest snippet is ideal).
Severity: high = core logic or security-sensitive code with no tests; medium = changed branch untested;
low = nice-to-have edge case.""" + COMMON_RULES


def numbered(source: str, focus_lines: set[int], max_lines: int, window: int = 20) -> str:
    """Source with line numbers. Long files are cut down to windows around the focus lines."""
    lines = source.splitlines()
    if len(lines) <= max_lines or not focus_lines:
        keep = range(1, min(len(lines), max_lines) + 1)
    else:
        wanted = set()
        for n in focus_lines:
            wanted.update(range(max(1, n - window), min(len(lines), n + window) + 1))
        keep = sorted(wanted)[:max_lines]
    out, prev = [], 0
    for n in keep:
        if n != prev + 1:
            out.append("   ...")
        out.append(f"{n:5d} | {lines[n - 1]}")
        prev = n
    return "\n".join(out)


def _run(llm: BaseChatModel, system: str, user: str, path: str, category: Category) -> list[Finding]:
    structured = llm.with_structured_output(FindingList)
    result = structured.invoke([SystemMessage(system), HumanMessage(user)])
    findings = result.findings if isinstance(result, FindingList) else FindingList.model_validate(result).findings
    # The model only reviews one file per call; pin path and category so they cannot drift.
    return [f.model_copy(update={"file": path, "category": category}) for f in findings]


def review_bugs(llm, path: str, diff: str, source: str, ast_report: dict | None) -> list[Finding]:
    user = f"File: {path}\n\nDiff:\n{diff}\n\nNumbered source (new version):\n{source}"
    if ast_report:
        user += f"\n\nStatic analysis (ast):\n{json.dumps(ast_report, indent=1)}"
    return _run(llm, BUG_PROMPT, user, path, "bug")


def review_security(llm, path: str, diff: str, source: str, ast_report: dict | None) -> list[Finding]:
    user = f"File: {path}\n\nDiff:\n{diff}\n\nNumbered source (new version):\n{source}"
    if ast_report:
        user += f"\n\nStatic analysis (ast):\n{json.dumps(ast_report, indent=1)}"
    return _run(llm, SECURITY_PROMPT, user, path, "security")


def review_tests(llm, path: str, diff: str, source: str, tests: dict[str, str]) -> list[Finding]:
    user = f"File: {path}\n\nDiff:\n{diff}\n\nNumbered source (new version):\n{source}\n\n"
    if tests:
        user += "Related test files:\n" + "\n\n".join(f"--- {p} ---\n{t}" for p, t in tests.items())
    else:
        user += "No related test files were found in the repository or the PR."
    return _run(llm, TEST_PROMPT, user, path, "testing")
