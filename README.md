# GenAcademy P3: Code Review Agent (ReAct)

A multi-agent code reviewer for GitHub pull requests, built on LangGraph. Give it `owner/repo` and a PR number, and it does the following:

1. A single **ReAct orchestrator** (`create_react_agent`) explores the PR. It decides which tools to call and in what order, based on what the PR contains.
2. It sends each source file to up to three **specialist agents**:
   - **Agent 1, bugs and anti-patterns:** logic errors, unhandled edge cases, swallowed exceptions, mutable defaults, needless complexity.
   - **Agent 2, security:** injection, unsafe deserialization, hardcoded secrets, disabled TLS verification, missing auth checks (with CWE ids).
   - **Agent 3, test coverage:** untested functions and branches, with concrete pytest cases to add.
3. The orchestrator merges the findings into one **prioritized report** with severity ratings (critical / high / medium / low / info).
4. A **human-in-the-loop gate** shows you the report. You approve, edit or reject it.
5. On approval, the system **posts one GitHub review**: a summary body, inline comments on diff lines, and `REQUEST_CHANGES` if any critical or high finding remains.

## Architecture

```
 CLI input                LangGraph StateGraph                                   External APIs
 owner/repo #PR ──▶ ┌──────────────────────────────────────────────────────┐
                    │ agent  (create_react_agent, response_format=Report)  │──▶ LLM (Claude / Groq / OpenAI)
                    │   tools: get_pr_overview  list_changed_files         │──▶ GitHub REST (PyGithub)
                    │          get_file_diff    get_file_content           │
                    │          analyze_python_ast ──────────────────────────┼──▶ Python ast (in-process)
                    │          find_related_tests                          │
                    │          run_bug_review       (Agent 1, LLM)         │
                    │          run_security_review  (Agent 2, LLM)         │
                    │          run_test_review      (Agent 3, LLM)         │
                    │                 │                                    │
                    │ finalize  dedupe, sort by severity, render Markdown  │
                    │                 │                                    │
                    │ human_gate  interrupt() ─ approve / edit / reject    │
                    │                 │ approve/edit          │ reject ──▶ END
                    │ post     PR review + inline comments ───┼─────────────┼──▶ GitHub: one posted review
                    └──────────────────────────────────────────────────────┘
```

Design points:

- **The flow depends on the data.** The orchestrator's prompt suggests an approach but doesn't fix one. A docs-only PR may need no specialist calls. A test file gets no test review. Python files get AST analysis first.
- **Findings are kept out of the chat history.** Every specialist result goes into a `FindingStore`. If the orchestrator's final structured report comes back empty, the store is used instead.
- **The LLM can't post on its own.** Posting is a graph node behind `interrupt()`, not a tool. Nothing reaches GitHub without a human decision.
- **Inline comments are checked before posting.** A finding becomes an inline comment only if its line is in the diff, because GitHub rejects the whole review otherwise. All findings also appear in the review body.
- **Big PRs are capped.** The review covers at most `MAX_FILES` files, picked by size of change. Long files are cut to windows around the changed lines. Files that were skipped are listed in the report.

## Layout

```
src/code_review_agent/
  config.py         .env loading, truststore, ChatAnthropic / ChatOpenAI per provider
  models.py         Finding, ReviewReport, Severity; sort + dedupe
  github_client.py  PRContext (PyGithub wrapper, cached), diff-line parsing
  ast_tools.py      stdlib-ast structural analysis and risky-pattern detection
  specialists.py    the three specialist agents (structured-output LLM calls)
  tools.py          ReAct tool set bound to one PR
  graph.py          StateGraph: agent -> finalize -> human_gate -> post
  report.py         Markdown rendering, inline comments, review event
  poster.py         post the review, or write it to results/ (dry run)
  cli.py            argparse CLI, live ReAct trace, approval prompt
tests/              offline tests (fake PR + scripted fake chat model; no network)
```

## Setup

```bash
uv sync
cp .env.example .env      # then fill in GITHUB_TOKEN and one of GROQ_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
```

The GitHub token needs read and write access to pull requests on the target repo. The default model is Groq `llama-3.3-70b-versatile`. Use `--provider openai` for `gpt-4o-mini`, or `--provider anthropic` for Claude Opus 5.5 (`claude-opus-5-5`, through `langchain-anthropic`'s `ChatAnthropic`, at effort `LLM_EFFORT`, default `high`). Override the model with `LLM_MODEL` or `--model`. The tool loads the OS certificate store through `truststore`, so it works behind corporate TLS inspection.

## Usage

```bash
uv run pytest                                              # offline tests
uv run code-review-agent review owner/repo 42 --dry-run    # review, approve, then save to results/
uv run code-review-agent review owner/repo 42              # review, approve, then post to GitHub
uv run code-review-agent review owner/repo 42 --provider openai --model gpt-4o
```

While it runs, the CLI prints each tool call and its result. Then it shows the report and asks:

```
[a]pprove and post, [e]dit then post, [r]eject?
```

`edit` opens the report in `$EDITOR` (Notepad on Windows). Whatever you save is posted as the review body. `--yes` approves automatically, for CI.

## Notes

- `create_react_agent` is deprecated in LangGraph 1.x in favour of `langchain.agents.create_agent`, but it still works until LangGraph 2.0. This project uses it because the spec asks for it, and the code hides the warning.
- GitHub doesn't allow `REQUEST_CHANGES` on your own PR. In that case the tool posts the review as `COMMENT` instead.
