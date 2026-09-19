# LLM providers

The vendor-neutral provider protocol and its implementations (OpenAI,
Anthropic, and a deterministic mock for tests), so the agent loop is decoupled
from any single LLM vendor. The real providers import their SDKs lazily, so this
layer is importable without the optional `llm` extra and only fails if a real
provider is instantiated without its SDK.

`AnthropicProvider` defaults to Claude Opus 5 (`claude-opus-5`) and
`OpenAIProvider` to `gpt-5-mini`. Both send a sampling `temperature` only when
a caller passes one, cap each reply at 16,000 tokens by default, and normalize
the model's stop reason (`end_turn`, `tool_use`, `max_tokens`, `refusal`,
`pause_turn`, or `stop_sequence`) so the agent loops can act on it.

::: fugacio.copilot.llm
