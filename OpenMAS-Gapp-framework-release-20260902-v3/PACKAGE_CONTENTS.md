# Source Package Contents

This repository can be shared as framework source without benchmark data or
experiment outputs.

Included:

- `openmas_bench/`: framework, public Engine, domain plugins, Q1/Q2/Q3 logic
- `scripts/`: reproducible experiment and validation entry points
- `tests/`: framework and causal-protocol regression tests
- `examples/`: minimal public API examples
- `pyproject.toml`, `README.md`, `ARCHITECTURE.md`
- protocol and dataset-interface documentation

Excluded from the distributable archive:

- `raw/`, `cleaned/`, `packages/`, `construction_cases/`, `q1_formal_cases/`
- downloaded `q2_datasets/` and all external baseline repositories
- result JSON, checkpoints, reports, caches, bytecode, and local environments
- API keys and environment-variable values

The archive is source-only. Dataset experiments require users to obtain the
datasets under their original licenses and pass the containing directory via
`--data-root`.
