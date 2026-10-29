"""Multi-agent code review of GitHub pull requests with a LangGraph ReAct orchestrator."""


def main() -> None:
    import sys

    from .cli import main as cli_main

    sys.exit(cli_main())
