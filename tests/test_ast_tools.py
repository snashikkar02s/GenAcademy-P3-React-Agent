from code_review_agent.ast_tools import analyze_source


def kinds(source: str) -> set[str]:
    return {i["kind"] for i in analyze_source(source)["issues"]}


def test_detects_risky_patterns():
    source = """
import subprocess, pickle, yaml, requests

API_KEY = "sk-live-1234567890"

def f(x, items=[], cfg=dict()):
    eval(x)
    subprocess.run(x, shell=True)
    pickle.loads(x)
    yaml.load(x)
    requests.get(x, verify=False)
    if x == None:
        pass
    try:
        pass
    except:
        pass
"""
    assert {"hardcoded-secret", "mutable-default", "eval-exec", "shell-true", "unsafe-deserialization",
            "tls-verify-disabled", "none-comparison", "bare-except", "swallowed-exception"} <= kinds(source)


def test_clean_code_has_no_issues():
    source = """
import subprocess
import yaml

def f(x, items=None):
    subprocess.run(["ls", x])
    yaml.load(x, Loader=yaml.SafeLoader)
    if x is None:
        return []
    try:
        return int(x)
    except ValueError:
        return 0
"""
    assert kinds(source) == set()


def test_complexity_and_inventory():
    branches = "\n".join(f"    if x == {i}:\n        return {i}" for i in range(12))
    report = analyze_source(f"class A:\n    pass\n\ndef big(x):\n{branches}\n    return -1\n# TODO: split\n")
    assert report["classes"] == [{"name": "A", "line": 1}]
    assert report["functions"][0]["complexity"] == 13
    assert "high-complexity" in {i["kind"] for i in report["issues"]}
    assert report["todo_count"] == 1


def test_syntax_error_is_reported_not_raised():
    assert "SyntaxError" in analyze_source("def broken(:\n")["error"]
