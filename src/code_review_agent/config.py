"""Environment loading and LLM construction (Anthropic via ChatAnthropic; OpenAI or Groq via ChatOpenAI)."""

import os

import truststore
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# Use the OS certificate store so API calls work behind corporate TLS inspection.
truststore.inject_into_ssl()
load_dotenv()

PROVIDERS = {
    "anthropic": {
        "key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-opus-5-5",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
    },
    "openai": {
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
}

MAX_FILES = int(os.getenv("MAX_FILES", "15"))
MAX_SOURCE_LINES = int(os.getenv("MAX_SOURCE_LINES", "400"))
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "80"))


class ConfigError(RuntimeError):
    pass


class ClaudeChat(ChatAnthropic):
    """ChatAnthropic adapted to create_react_agent's response_format step on current Claude models.

    - Current Claude models reject forced tool_choice, so the default "function_calling" method would
      fall back to an unforced tool call that the model may skip; use native structured outputs instead.
    - create_react_agent sends the full history, which ends with the agent's own final message.
      Current models reject a trailing assistant message (prefill), so a user turn is appended.
    - That call carries no system prompt or tools, so the replayed thinking blocks no longer match the
      conversation they were signed for (preserved thinking) and are rejected; they are stripped,
      since summarizing the transcript does not need the earlier reasoning.
    """

    def with_structured_output(self, schema, *, method="json_schema", **kwargs):
        def prepare(value):
            messages = [_without_thinking(m) for m in self._convert_input(value).to_messages()]
            if messages and isinstance(messages[-1], AIMessage):
                messages.append(HumanMessage("Produce the final structured output now."))
            return messages

        return RunnableLambda(prepare) | super().with_structured_output(schema, method=method, **kwargs)


def _without_thinking(message):
    if not isinstance(message, AIMessage) or not isinstance(message.content, list):
        return message
    content = [b for b in message.content
               if not (isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"))]
    return message.model_copy(update={"content": content})


def get_llm(provider: str | None = None, model: str | None = None) -> BaseChatModel:
    provider = (provider or os.getenv("LLM_PROVIDER") or "groq").lower()
    if provider not in PROVIDERS:
        raise ConfigError(f"Unknown LLM provider {provider!r}; use one of {', '.join(PROVIDERS)}")
    spec = PROVIDERS[provider]
    api_key = os.getenv(spec["key_env"])
    if not api_key:
        raise ConfigError(f"{spec['key_env']} is not set (see .env.example)")
    model = model or os.getenv("LLM_MODEL") or spec["default_model"]
    if provider == "anthropic":
        # Opus 5.5 always thinks and rejects sampling params, so no temperature; effort sets depth.
        return ClaudeChat(model=model, api_key=api_key, max_tokens=16000,
                          reasoning_effort=os.getenv("LLM_EFFORT", "high"))
    return ChatOpenAI(model=model, api_key=api_key, base_url=spec["base_url"], temperature=0)


def get_github_token() -> str:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ConfigError("GITHUB_TOKEN is not set (see .env.example)")
    return token
