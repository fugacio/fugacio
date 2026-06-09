"""Vendor-neutral LLM provider layer for the Fugacio copilot.

Importing this package pulls in only the neutral types and the in-memory
:class:`MockProvider`; the real providers (:class:`OpenAIProvider`,
:class:`AnthropicProvider`) import their SDKs lazily, so they are importable here
without the optional ``llm`` extra installed and only fail if instantiated
without the SDK.
"""

from fugacio.copilot.llm.anthropic import AnthropicProvider
from fugacio.copilot.llm.base import (
    ChatResponse,
    LLMProvider,
    Message,
    ToolCall,
    anthropic_tools,
    openai_tools,
)
from fugacio.copilot.llm.mock import MockProvider
from fugacio.copilot.llm.openai import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "ChatResponse",
    "LLMProvider",
    "Message",
    "MockProvider",
    "OpenAIProvider",
    "ToolCall",
    "anthropic_tools",
    "openai_tools",
]
