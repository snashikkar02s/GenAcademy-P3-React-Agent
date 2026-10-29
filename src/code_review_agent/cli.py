"""Command-line entry point: run the review, show the ReAct trace, ask for approval, post."""

import argparse
import os
import subprocess
import sys
import tempfile
import uuid
from functools import partial

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from .config import RECURSION_LIMIT, ConfigError, get_github_token, get_llm
from .github_client import PRContext
from .graph import build_graph, initial_input
from .poster import post_review, write_review
from .tools import FindingStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="code-review-agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    review = sub.add_parser("review", help="Review a pull request and post the result after approval")
    review.add_argument("repo", help="owner/repo")
    review.add_argument("pr", type=int, help="pull request number")
    review.add_argument("--provider", choices=["anthropic", "groq", "openai"], help="LLM provider (default: $LLM_PROVIDER or groq)")
    review.add_argument("--model", help="model name (default: $LLM_MODEL or the provider default)")
    review.add_argument("--dry-run", action="store_true", help="write the review to results/ instead of posting")
    review.add_argument("--yes", action="store_true", help="approve without prompting")
    return parser.parse_args(argv)


def _short(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def print_update(namespace: tuple, update: dict) -> None:
    """Print tool calls and tool results from the ReAct loop as they happen."""
    for node, payload in update.items():
        if not isinstance(payload, dict):
            continue
        # The parent "agent" update repeats the whole subgraph history; print only live subgraph steps.
        for msg in payload.get("messages", []) if namespace else []:
            if isinstance(msg, AIMessage):
                for call in msg.tool_calls:
                    args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                    print(f"  -> {call['name']}({args})")
            elif isinstance(msg, ToolMessage):
                print(f"     <- {_short(msg.content)}")
        if node == "finalize":
            print("  .. report compiled")


def edit_text(initial: str) -> str:
    editor = os.getenv("EDITOR") or ("notepad" if sys.platform == "win32" else "vi")
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as fh:
        fh.write(initial)
        path = fh.name
    try:
        subprocess.run([editor, path], check=False)
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    finally:
        os.unlink(path)


def ask_human(markdown: str, auto_yes: bool) -> dict:
    print("\n" + "=" * 78 + "\n" + markdown + "\n" + "=" * 78)
    if auto_yes:
        return {"action": "approve"}
    while True:
        choice = input("\n[a]pprove and post, [e]dit then post, [r]eject? ").strip().lower()
        if choice in ("a", "approve"):
            return {"action": "approve"}
        if choice in ("e", "edit"):
            body = edit_text(markdown)
            return {"action": "edit", "body": body} if body.strip() else {"action": "reject"}
        if choice in ("r", "reject"):
            return {"action": "reject"}


def run_review(args: argparse.Namespace) -> int:
    llm = get_llm(args.provider, args.model)
    print(f"Fetching {args.repo}#{args.pr} ...")
    ctx = PRContext(args.repo, args.pr, get_github_token())
    poster = partial(write_review if args.dry_run else post_review, ctx)
    graph = build_graph(ctx, llm, poster, FindingStore())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": RECURSION_LIMIT}

    print("ReAct trace:")
    stream_input = initial_input(args.repo, args.pr)
    while True:
        interrupted = None
        for namespace, update in graph.stream(stream_input, config, stream_mode="updates", subgraphs=True):
            if "__interrupt__" in update:
                interrupted = update["__interrupt__"][0].value
            else:
                print_update(namespace, update)
        if interrupted is None:
            break
        stream_input = Command(resume=ask_human(interrupted["markdown"], args.yes))

    state = graph.get_state(config).values
    if state.get("posted_url"):
        print(f"\n{'Saved' if args.dry_run else 'Posted'}: {state['posted_url']}")
    else:
        print("\nRejected; nothing was posted.")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252, which cannot print some characters models produce.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    try:
        return run_review(args)
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
