# fugacio-copilot

Chemical-engineering design copilot/agent for the
[Fugacio](https://github.com/fugacio/fugacio) stack. It sits on top of
`fugacio.sim` and turns natural-language design goals into engineering
calculations: flowsheets, equipment sizing, and (eventually) techno-economic /
life-cycle analysis.

The bridge between a language model and the differentiable engine is a **tool
registry**, deterministic, JSON-in/JSON-out functions exposed with the same
function-calling schemas OpenAI/Anthropic expect:

- **Properties & equilibrium**: `list_components`, `component_properties`,
  `saturation_pressure`, `bubble_pressure`, `flash_drum`.
- **Unit operations**: `heat_exchanger`, `compressor` (and turbine), `pump`,
  `valve`, each closing a rigorous energy balance.
- **Distillation**: `shortcut_distillation` (Fenske-Underwood-Gilliland) and
  `rigorous_distillation` (multistage column with duties).
- **Gradient-based optimization**: `optimize_flash_temperature` and
  `optimize_column_reflux` solve for the operating variable that hits a target by
  differentiating straight through the equilibrium flash and the column.

A model-agnostic **agent loop** (`run_agent`) drives plan→act→answer; the planner
is injected, so the loop is fully testable with a scripted planner while a real
LLM drops in behind the optional `llm` extra.

```python
from fugacio.copilot import call_tool, run_agent, tool_schemas

# Call an engine-backed tool directly (JSON in / JSON out):
call_tool("saturation_pressure", {"component": "propane", "temperature": 300.0})

# Or drive the agent loop with your own planner (an LLM in production):
def planner(goal, tools, transcript):
    if not transcript:
        return {"tool": "flash_drum", "arguments": {
            "components": ["methane", "propane", "n-pentane"],
            "z": [0.5, 0.3, 0.2], "flow": 100.0,
            "temperature": 320.0, "pressure": 20e5,
        }}
    vf = transcript[-1]["result"]["vapor_fraction"]
    return {"final_answer": f"vapor fraction = {vf:.3f}"}

run_agent("Flash this feed", planner).answer  # 'vapor fraction = 0.747'
```

`tool_schemas()` returns the schemas to hand an LLM. LLM-backed planning lives
behind the optional `llm` extra:

```bash
pip install "fugacio-copilot[llm]"
```

Part of the `fugacio` namespace; installs independently:
`pip install fugacio-copilot`.
