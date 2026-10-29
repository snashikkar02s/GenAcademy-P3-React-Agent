"""End-to-end graph run with a scripted fake model: ReAct tool loop, HITL interrupt, post/reject."""

import json
from functools import partial

from langchain_core.messages import AIMessage
from langgraph.types import Command

from code_review_agent.graph import build_graph, initial_input
from code_review_agent.models import Finding, FindingList, ReviewReport
from code_review_agent.poster import post_review
from code_review_agent.tools import FindingStore, make_tools

from conftest import FakeToolModel

SHELL = Finding(file="app.py", line=6, severity="critical", category="security", title="Command injection",
                detail="shell=True with caller input (CWE-78)", suggestion="Pass an argument list")


def call(name, call_id, **args):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def scripted_model(report: ReviewReport) -> FakeToolModel:
    script = iter([
        AIMessage("", tool_calls=[call("get_pr_overview", "1"), call("list_changed_files", "2")]),
        AIMessage("", tool_calls=[call("analyze_python_ast", "3", path="app.py"),
                                  call("run_security_review", "4", path="app.py")]),
        AIMessage("Review complete."),
    ])
    return FakeToolModel(messages=script, structured={FindingList: FindingList(findings=[SHELL]),
                                                      ReviewReport: report})


def run_until_interrupt(graph, config, fake_pr):
    events = list(graph.stream(initial_input(fake_pr.repo_full_name, fake_pr.number), config,
                               stream_mode="updates", subgraphs=True))
    interrupts = [u["__interrupt__"][0].value for _, u in events if "__interrupt__" in u]
    assert len(interrupts) == 1
    return interrupts[0]


def test_approve_posts_review_with_inline_comment(fake_pr):
    report = ReviewReport(summary="Adds a shell runner.", verdict="comment", findings=[SHELL])
    store = FindingStore()
    graph = build_graph(fake_pr, scripted_model(report), partial(post_review, fake_pr), store)
    config = {"configurable": {"thread_id": "t1"}}

    pending = run_until_interrupt(graph, config, fake_pr)
    assert "Command injection" in pending["markdown"]
    assert fake_pr.posted == []  # nothing is posted before the human decides
    assert store.reviews_run == {("app.py", "security")}

    graph.invoke(Command(resume={"action": "approve"}), config)
    state = graph.get_state(config).values
    assert state["report"].verdict == "request_changes"  # critical finding forces it
    assert state["posted_url"].endswith("review-1")
    [posted] = fake_pr.posted
    assert posted["event"] == "REQUEST_CHANGES"
    assert [(c["path"], c["line"]) for c in posted["comments"]] == [("app.py", 6)]


def test_edit_posts_edited_body(fake_pr):
    report = ReviewReport(summary="s", verdict="comment", findings=[SHELL])
    graph = build_graph(fake_pr, scripted_model(report), partial(post_review, fake_pr))
    config = {"configurable": {"thread_id": "t2"}}
    run_until_interrupt(graph, config, fake_pr)
    graph.invoke(Command(resume={"action": "edit", "body": "Edited by a human."}), config)
    assert fake_pr.posted[0]["body"] == "Edited by a human."


def test_reject_posts_nothing(fake_pr):
    report = ReviewReport(summary="s", verdict="comment", findings=[SHELL])
    graph = build_graph(fake_pr, scripted_model(report), partial(post_review, fake_pr))
    config = {"configurable": {"thread_id": "t3"}}
    run_until_interrupt(graph, config, fake_pr)
    graph.invoke(Command(resume={"action": "reject"}), config)
    assert fake_pr.posted == []
    assert "posted_url" not in graph.get_state(config).values


def test_empty_orchestrator_report_falls_back_to_specialist_findings(fake_pr):
    report = ReviewReport(summary="s", verdict="approve", findings=[])
    graph = build_graph(fake_pr, scripted_model(report), partial(post_review, fake_pr))
    config = {"configurable": {"thread_id": "t4"}}
    pending = run_until_interrupt(graph, config, fake_pr)
    assert "Command injection" in pending["markdown"]


def test_tools_scope_and_errors(fake_pr):
    store = FindingStore()
    tools = {t.name: t for t in make_tools(fake_pr, FakeToolModel(messages=iter([])), store)}
    files = {f["path"]: f for f in json.loads(tools["list_changed_files"].invoke({}))}
    assert files["app.py"]["in_scope"] and not files["logo.png"]["in_scope"]
    assert store.skipped == ["logo.png"]
    assert tools["run_bug_review"].invoke({"path": "missing.py"}).startswith("ERROR")
    assert tools["run_bug_review"].invoke({"path": "logo.png"}).startswith("SKIPPED")
    assert "Not a Python file" in tools["analyze_python_ast"].invoke({"path": "README.md"})
    issues = json.loads(tools["analyze_python_ast"].invoke({"path": "app.py"}))["issues"]
    assert {"shell-true", "bare-except", "mutable-default"} <= {i["kind"] for i in issues}
    assert json.loads(tools["find_related_tests"].invoke({"path": "app.py"})) == [
        {"path": "tests/test_app.py", "changed_in_pr": False}]
