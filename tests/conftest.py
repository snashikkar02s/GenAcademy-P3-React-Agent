"""Offline test doubles: a fake PR and a fake chat model that emits scripted tool calls."""

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.runnables import RunnableLambda

from code_review_agent.github_client import ChangedFile

APP_SOURCE = """import subprocess


def run(cmd, opts=[]):
    try:
        return subprocess.run(cmd, shell=True)
    except:
        pass
"""

APP_PATCH = """@@ -0,0 +1,8 @@
+import subprocess
+
+
+def run(cmd, opts=[]):
+    try:
+        return subprocess.run(cmd, shell=True)
+    except:
+        pass"""


class FakePR:
    repo_full_name = "acme/widgets"
    number = 7
    head_sha = "abc123"

    def __init__(self):
        self.files = [
            ChangedFile("app.py", "added", 8, 0, APP_PATCH),
            ChangedFile("README.md", "modified", 2, 1, "@@ -1,1 +1,2 @@\n-old\n+new\n+more"),
            ChangedFile("logo.png", "added", 0, 0, None),
        ]
        self.tree_paths = ["app.py", "README.md", "tests/test_app.py"]
        self.posted: list[dict] = []

    def overview(self):
        return {"repo": self.repo_full_name, "number": self.number, "title": "Add runner"}

    def file(self, path):
        return next((f for f in self.files if f.path == path), None)

    def content(self, path):
        return {"app.py": APP_SOURCE, "tests/test_app.py": "def test_nothing():\n    assert True\n",
                "README.md": "new\nmore\n"}.get(path)

    def create_review(self, body, event, comments):
        self.posted.append({"body": body, "event": event, "comments": comments})
        return "https://github.com/acme/widgets/pull/7#review-1"


class FakeToolModel(GenericFakeChatModel):
    """Replays scripted AIMessages and answers with_structured_output() from a schema->value map."""

    structured: dict[Any, Any] = {}

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _input: self.structured[schema])


@pytest.fixture
def fake_pr():
    return FakePR()
