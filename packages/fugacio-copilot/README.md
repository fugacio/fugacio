# fugacio-copilot

Chemical-engineering design copilot/agent for the
[Fugacio](https://github.com/owenthcarey/fugacio) stack. It sits on top of
`fugacio.sim` and (eventually) turns natural-language design goals into
flowsheets, equipment sizing, and techno-economic / life-cycle analysis.

The baseline ships a single deterministic helper so the layering
(`copilot` -> `sim` -> `thermo`) is wired and tested end-to-end. LLM-backed
planning lives behind the optional `llm` extra:

```bash
pip install "fugacio-copilot[llm]"
```

Part of the `fugacio` namespace; installs independently:
`pip install fugacio-copilot`.
