"""Static structural analysis of Python source with the stdlib ast module (no LLM, no network)."""

import ast
import re

SECRET_NAME = re.compile(r"(?i)(password|passwd|secret|token|api_?key|private_?key|access_?key)")
TODO = re.compile(r"#\s*(TODO|FIXME|XXX|HACK)\b")
COMPLEXITY_THRESHOLD = 10
MUTABLE_CALLS = {"list", "dict", "set"}


def _call_name(node: ast.Call) -> str:
    """Dotted name of the called function, e.g. 'subprocess.run' or 'eval'."""
    parts = []
    func = node.func
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def _complexity(fn: ast.AST) -> int:
    score = 1
    for node in ast.walk(fn):
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.ExceptHandler,
                             ast.match_case)):
            score += 1
        elif isinstance(node, ast.BoolOp):
            score += len(node.values) - 1
        elif isinstance(node, ast.comprehension):
            score += 1 + len(node.ifs)
    return score


def _issue(node: ast.AST, kind: str, message: str) -> dict:
    return {"line": getattr(node, "lineno", None), "kind": kind, "message": message}


def analyze_source(source: str) -> dict:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"error": f"SyntaxError: {e.msg} at line {e.lineno}"}

    functions, classes, issues = [], [], []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cc = _complexity(node)
            functions.append({
                "name": node.name,
                "line": node.lineno,
                "end_line": node.end_lineno,
                "args": len(node.args.args) + len(node.args.kwonlyargs),
                "complexity": cc,
                "has_docstring": ast.get_docstring(node) is not None,
            })
            if cc > COMPLEXITY_THRESHOLD:
                issues.append(_issue(node, "high-complexity",
                                     f"{node.name}() has cyclomatic complexity {cc}"))
            for default in node.args.defaults + [d for d in node.args.kw_defaults if d]:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)) or (
                    isinstance(default, ast.Call) and _call_name(default) in MUTABLE_CALLS
                ):
                    issues.append(_issue(default, "mutable-default",
                                         f"{node.name}() uses a mutable default argument"))

        elif isinstance(node, ast.ClassDef):
            classes.append({"name": node.name, "line": node.lineno})

        elif isinstance(node, ast.ExceptHandler):
            if node.type is None:
                issues.append(_issue(node, "bare-except", "bare `except:` also catches SystemExit/KeyboardInterrupt"))
            elif isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
                issues.append(_issue(node, "broad-except", f"catches broad `{node.type.id}`"))
            if all(isinstance(stmt, ast.Pass) for stmt in node.body):
                issues.append(_issue(node, "swallowed-exception", "exception is silently ignored"))

        elif isinstance(node, ast.Call):
            name = _call_name(node)
            kwargs = {k.arg: k.value for k in node.keywords if k.arg}
            if name in {"eval", "exec"}:
                issues.append(_issue(node, "eval-exec", f"call to {name}()"))
            elif name in {"os.system", "os.popen"}:
                issues.append(_issue(node, "shell-command", f"call to {name}()"))
            elif name.startswith("subprocess.") and isinstance(kwargs.get("shell"), ast.Constant) \
                    and kwargs["shell"].value is True:
                issues.append(_issue(node, "shell-true", f"{name}(..., shell=True)"))
            elif name in {"pickle.load", "pickle.loads", "marshal.loads"}:
                issues.append(_issue(node, "unsafe-deserialization", f"call to {name}()"))
            elif name == "yaml.load" and "Loader" not in kwargs:
                issues.append(_issue(node, "unsafe-deserialization", "yaml.load() without an explicit Loader"))
            elif name.startswith("requests.") and isinstance(kwargs.get("verify"), ast.Constant) \
                    and kwargs["verify"].value is False:
                issues.append(_issue(node, "tls-verify-disabled", f"{name}(..., verify=False)"))

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and SECRET_NAME.search(target.id) \
                        and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) \
                        and len(node.value.value) >= 8:
                    issues.append(_issue(node, "hardcoded-secret", f"`{target.id}` is assigned a string literal"))

        elif isinstance(node, ast.Compare):
            for op, right in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(right, ast.Constant) and right.value is None:
                    issues.append(_issue(node, "none-comparison", "compare to None with `is` / `is not`"))

    return {
        "functions": sorted(functions, key=lambda f: f["line"]),
        "classes": sorted(classes, key=lambda c: c["line"]),
        "issues": sorted(issues, key=lambda i: i["line"] or 0),
        "todo_count": len(TODO.findall(source)),
        "line_count": source.count("\n") + 1,
    }
