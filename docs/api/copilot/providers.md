# LLM providers

The vendor-neutral provider protocol and its implementations (OpenAI,
Anthropic, and a deterministic mock for tests), so the agent loop is decoupled
from any single LLM vendor. The real providers import their SDKs lazily, so this
layer is importable without the optional `llm` extra and only fails if a real
provider is instantiated without its SDK.

::: fugacio.copilot.llm
