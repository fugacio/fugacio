# CHANGELOG


## v0.1.0 (2026-06-24)

### Continuous Integration

- Add lockstep semantic-release pipeline ([#14](https://github.com/fugacio/fugacio/pull/14),
  [`f38e71e`](https://github.com/fugacio/fugacio/commit/f38e71ec412b5861aa993d16c659156be61f07c0))

### Features

- Add fugacio umbrella meta-package ([#15](https://github.com/fugacio/fugacio/pull/15),
  [`3b5746e`](https://github.com/fugacio/fugacio/commit/3b5746e7933ddcf7214e6612ea0b440ad63f1562))


## v0.0.1 (2026-06-23)

### Bug Fixes

- Declare exclude-newer cooldown in config so uv sync --locked passes
  ([`da482a2`](https://github.com/fugacio/fugacio/commit/da482a2cda13318accd09333b3ca523f1018ec69))

### Continuous Integration

- Pin uv to the matrix Python so every version is actually tested
  ([`bf7b5e8`](https://github.com/fugacio/fugacio/commit/bf7b5e849e402a60ebc7648f3a850183f50b61d3))

### Documentation

- Add AGENTS.md with documentation style guidelines
  ([`996949f`](https://github.com/fugacio/fugacio/commit/996949f819b3ffdc6af88b11ca68e171daabf236))

- Add MkDocs site and enforce Google docstrings ([#11](https://github.com/fugacio/fugacio/pull/11),
  [`2baf308`](https://github.com/fugacio/fugacio/commit/2baf308c86a30bc48a08c37b69b05a9f91f103a9))

- Describe the open validation oracles that anchor correctness
  ([`ef79baf`](https://github.com/fugacio/fugacio/commit/ef79baf88f99637c4c82dbc9a81caa98cd867619))

- Replace em dashes with Chicago-style punctuation
  ([#12](https://github.com/fugacio/fugacio/pull/12),
  [`7dbfdc1`](https://github.com/fugacio/fugacio/commit/7dbfdc1916084599263b9df913682d48962361be))

### Features

- Add chemical reactions and reactors ([#4](https://github.com/fugacio/fugacio/pull/4),
  [`7370e65`](https://github.com/fugacio/fugacio/commit/7370e652de6961a4996abf0a328c79e09554f508))

- Add differentiable dynamics and process control layer
  ([#8](https://github.com/fugacio/fugacio/pull/8),
  [`06faf21`](https://github.com/fugacio/fugacio/commit/06faf21fda5a61df8e156b428f1683094dd49e03))

- Add differentiable heat integration and pinch analysis
  ([#9](https://github.com/fugacio/fugacio/pull/9),
  [`0cbe210`](https://github.com/fugacio/fugacio/commit/0cbe2105229b520552d31b6636b1a3e99e83c925))

- Add differentiable model predictive control and state estimation
  ([#10](https://github.com/fugacio/fugacio/pull/10),
  [`72bbc8b`](https://github.com/fugacio/fugacio/commit/72bbc8b06f26ade3f7ef58f86394ddb3529b32f5))

- Add differentiable optimization and an LLM-backed design copilot
  ([#5](https://github.com/fugacio/fugacio/pull/5),
  [`bb6510e`](https://github.com/fugacio/fugacio/commit/bb6510e255f563c03f987ce6ef7c2d11795cdd9d))

- Add differentiable PC-SAFT equation of state ([#13](https://github.com/fugacio/fugacio/pull/13),
  [`2bc963f`](https://github.com/fugacio/fugacio/commit/2bc963fc2e6e1cce3225486534be3877ead948c5))

- Add differentiable thermodynamics core with sim and copilot spikes
  ([#2](https://github.com/fugacio/fugacio/pull/2),
  [`4489985`](https://github.com/fugacio/fugacio/commit/448998563f0b432423c4e595a87bec182424f5aa))

- Add energy-aware differentiable flowsheet engine ([#3](https://github.com/fugacio/fugacio/pull/3),
  [`d129443`](https://github.com/fugacio/fugacio/commit/d129443a773a186f8c4c80fa9ac51b83ba429bd0))

- Add physical property foundation and oracle correctness harness
  ([#6](https://github.com/fugacio/fugacio/pull/6),
  [`ed8ab7d`](https://github.com/fugacio/fugacio/commit/ed8ab7d50829327a6ed04471f982132dab4741e7))

- Add reference Helmholtz EOS and steam tables ([#7](https://github.com/fugacio/fugacio/pull/7),
  [`84bab0e`](https://github.com/fugacio/fugacio/commit/84bab0e57e93f8f159474029aa0b0249932fe581))

- Scaffold differentiable thermo, sim, and copilot uv workspace
  ([`09c0e61`](https://github.com/fugacio/fugacio/commit/09c0e61c11cacaac7123477e01edde12e6d013d1))
