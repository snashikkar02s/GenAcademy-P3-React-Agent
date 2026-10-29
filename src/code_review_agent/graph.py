"""LangGraph state graph: ReAct orchestrator -> finalize -> human approval gate -> post."""

import warnings
from typing import Annotated, Callable, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

from .models import ReviewReport, Severity, dedupe_findings
from .report import render_markdown
from .tools import FindingStore, make_tools

ORCHESTRATOR_PROMPT = """You are the orchestrator of a multi-agent code review of a GitHub pull request.
You decide which tools to call and in what order; adapt to what the PR actually contains.

Specialist agents (tools):
- run_bug_review(path)      Agent 1: bugs and anti-patterns
- run_security_review(path) Agent 2: security vulnerabilities
- run_test_review(path)     Agent 3: test coverage and missing tests

Suggested approach:
1. get_pr_overview, then list_changed_files.
2. For each in-scope source file, decide which specialists it needs. For example: run all three on
   application code; skip the test review for files that are themselves tests (but use them as evidence);
   for docs, lock files or pure config changes, skip specialists or run only the security review if the
   config could expose secrets or weaken security. Use analyze_python_ast, get_file_diff or
   find_related_tests when they help you decide.
3. Never call the same specialist twice on the same file.
4. When done, produce the final report: merge duplicate findings reported by different agents, drop
   findings that are clearly false positives given the diff, and adjust severities so they are consistent
   across files. Keep each finding's file and line. Verdict: request_changes if any critical or high
   finding remains, comment if only medium/low/info, approve if nothing worth raising.

You cannot post the review; a human approves it after you finish."""


class ReviewState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    structured_response: ReviewReport
    report: ReviewReport
    markdown: str
    decision: Literal["approve", "edit", "reject"]
    body: str
    posted_url: str


def initial_input(repo_full_name: str, number: int) -> ReviewState:
    return {"messages": [HumanMessage(f"Review pull request #{number} in {repo_full_name}.")]}


def build_graph(ctx, llm, poster: Callable[[ReviewReport, str], str], store: FindingStore | None = None,
                checkpointer=None):
    store = store if store is not None else FindingStore()
    # create_react_agent is the prebuilt ReAct loop this project is built on; LangGraph 1.x marks it
    # deprecated in favour of langchain.agents.create_agent but keeps it until 2.0.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="create_react_agent has been moved")
        agent = create_react_agent(llm, make_tools(ctx, llm, store), prompt=ORCHESTRATOR_PROMPT,
                                   response_format=ReviewReport, name="orchestrator")

    def finalize(state: ReviewState) -> ReviewState:
        report = state.get("structured_response")
        if isinstance(report, dict):
            report = ReviewReport.model_validate(report)
        if report is None:
            report = ReviewReport(summary="The orchestrator did not return a report; showing raw specialist findings.",
                                  verdict="comment", findings=[])
        # Fall back to the specialists' raw output if the orchestrator dropped everything.
        findings = dedupe_findings(report.findings or store.findings)
        verdict = report.verdict
        if any(f.severity in (Severity.CRITICAL, Severity.HIGH) for f in findings):
            verdict = "request_changes"
        elif verdict == "approve" and findings and any(f.severity != Severity.INFO for f in findings):
            verdict = "comment"
        report = report.model_copy(update={"findings": findings, "verdict": verdict})
        return {"report": report, "markdown": render_markdown(report, store.skipped)}

    def human_gate(state: ReviewState) -> ReviewState:
        answer = interrupt({"markdown": state["markdown"], "finding_count": len(state["report"].findings)})
        action = answer.get("action", "reject")
        return {"decision": action, "body": answer.get("body") or state["markdown"]}

    def route(state: ReviewState) -> str:
        return "post" if state["decision"] in ("approve", "edit") else END

    def post(state: ReviewState) -> ReviewState:
        return {"posted_url": poster(state["report"], state["body"])}

    builder = StateGraph(ReviewState)
    builder.add_node("agent", agent)
    builder.add_node("finalize", finalize)
    builder.add_node("human_gate", human_gate)
    builder.add_node("post", post)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "finalize")
    builder.add_edge("finalize", "human_gate")
    builder.add_conditional_edges("human_gate", route, ["post", END])
    builder.add_edge("post", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
