# 0001. A dependency-free core, ops and plan stack

- Status: accepted
- Date: 2026-09-04

## Context

Rule R5 requires hermetic CI: no network access during tests and no downloads
from `build.rs`. The bootstrap environment additionally has no access to the
crates.io registry, so any third-party dependency — including `serde`,
`thiserror`, `toml` and `rand` — cannot be vendored at bootstrap time.

The prompt names those crates but the priority order is
`correctness > reproducibility > memory safety > performance > portability`,
and a build that cannot run at all fails every one of those goals.

## Decision

`rhizome-core`, `rhizome-ops`, `rhizome-plan` and `rhizome-cli` have zero
third-party dependencies:

- configuration parsing uses an in-tree TOML subset parser
  (`rhizome_core::toml_lite`) covering tables, scalars and homogeneous arrays;
- error types are hand-written enums implementing `Display` and
  `std::error::Error`, which is exactly the shape `thiserror` generates;
- randomness uses the in-tree counter-based Philox-4x32 generator that R11
  mandates anyway;
- plan hashing uses in-tree FNV-1a rather than pulling a hashing crate.

## Alternatives considered

- **Vendor the dependency tree.** Not possible without registry access, and it
  would add ~80 transitive crates to audit under `deny.toml`.
- **Use `std::collections::hash_map::DefaultHasher` for plan hashes.** Rejected:
  its output is explicitly not stable across releases, which would break plan
  cache keys and R9.

## Consequences

- CI is hermetic by construction and the whole workspace builds offline.
- The TOML subset is narrower than real TOML; configuration files must stay
  within it, and the parser is a fuzz target.
- When registry access exists, swapping `toml_lite` for `serde` + `toml` is a
  contained change behind `ModelConfig::from_toml`, and requires a new ADR.
