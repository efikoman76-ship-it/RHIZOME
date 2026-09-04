# Milestones

| milestone | status | acceptance | workflow run |
|---|---|---|---|
| M0 bootstrap | in progress | workspace, lints, configs, `rhizome params`, spec-sync green | see `ci` runs on `arena/01a06d15-rhizome` |
| M1 core + ops + plan | in progress | f64 references, gradient checks, arena validation (T1, T3) | see `ci` runs on `arena/01a06d15-rhizome` |

M0 acceptance in this repository is: the `ci` workflow is green, `rhizome
params --format=markdown` reproduces the SPEC.md table, and `rhizome verify`
builds, differentiates, plans and validates a graph without conflicts.
