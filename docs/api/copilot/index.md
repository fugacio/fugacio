# `fugacio.copilot`

The chemical-engineering design copilot (depends on `fugacio.sim`): it exposes
the differentiable engine to a language model through a JSON tool registry, a
vendor-neutral provider layer, agent loops, and human-readable reports.

```python
from fugacio.copilot import default_registry, run_llm_agent, MockProvider
```

::: fugacio.copilot
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: false

## Where to look next

| Area | Page | Key symbols |
| --- | --- | --- |
| Tool registry | [Tool registry](tools.md) | `ToolSpec`, `default_registry`, `tool_schemas`, `call_tool` |
| Agent loops | [Agent loops](agent.md) | `run_agent`, `run_llm_agent`, `llm_planner`, `AgentResult` |
| LLM providers | [LLM providers](providers.md) | `LLMProvider`, `OpenAIProvider`, `AnthropicProvider`, `MockProvider` |
| Reporting | [Reporting](report.md) | `summarize_optimization`, `summarize_economics`, `stream_table` |

See the [AI design copilot section of the optimization guide](../../optimization.md#the-ai-design-copilot)
for an end-to-end walkthrough.
