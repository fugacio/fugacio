# Agent loops

The agent loops that drive the tool registry: a deterministic planner for tests
and a provider-backed loop that lets a language model call tools, observe
results, and iterate toward a design.

::: fugacio.copilot.agent

## Accountable process design

`run_design_agent` completes only through a recorded, accepted current-case
run. Its final report is deterministic; unsupported model text is retained in
the transcript and can't become the engineering answer. See the
[saved-case workflow](../../process-cases.md#accountable-copilot).

::: fugacio.copilot.design_agent

::: fugacio.copilot.case_tools
