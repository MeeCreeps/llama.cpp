# Research Design Notes

This directory records staged research-design discussions for dynamic memory
budget LLM inference.

Current focus:

- Problem: phone memory budget changes over time during inference.
- Baselines: offline budget plans and online CP-SAT solving.
- Proposed direction: lightweight online incremental planning that reuses
  offline candidates while adapting to the current runtime memory state.

Suggested file organization:

- `01_structured_diff_design.md`: current design draft around structured plan
  diff decisions.
- Future notes can use numbered prefixes when they capture a new design stage,
  experiment interpretation, or paper-section draft.
