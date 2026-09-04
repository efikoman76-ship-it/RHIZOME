# Changelog

## Unreleased

### Added
- M0 bootstrap: Cargo workspace with pinned toolchain, workspace lints
  (`missing_docs`, `clippy::todo`, `clippy::unimplemented`,
  `clippy::dbg_macro`), `#![forbid(unsafe_code)]` in every crate, and a
  Windows CI workflow that logs every step.
- `rhizome-core`: versioned model schema with validation, MX/FP8 dtypes with
  bit-exact FP8 e4m3 emulation, a dependency-free TOML subset parser, and
  parameter / FLOP / deploy-byte accounting behind `rhizome params`.
- `rhizome-ops`: f64 reference implementations and adjoints for the
  elementwise, norm, linear-algebra, gated-delta, MLA, routing, loss and
  sampling op families, plus the Philox-4x32 counter-based RNG and a
  finite-difference gradient checker.
- `rhizome-plan`: graph IR with SSA validation, automatic backward derivation,
  liveness-based arena planning with conflict verification, stable FNV-1a plan
  hashing, an activation-checkpointing planner trait and the pure-function
  serving micro-step scheduler.
- `rhizome-cli`: `params`, `inspect`, `verify` and `spec-check`.
- Configs for test-s, test-m, test-l, edge, standard and flagship; SPEC.md;
  ADRs 0000-0002; kernel, determinism, muP and milestone docs.
