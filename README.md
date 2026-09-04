# RHIZOME

A production-grade neural language model architecture and its training and
inference stack, written in Rust.

RHIZOME combines a 5:1 hybrid trunk (Gated Delta-Rule Local blocks and
Multi-head Latent Attention Global blocks), dropless mixture-of-experts FFNs
with group-limited routing, and a weight-shared **Think Core** that gives every
token an adaptive amount of depth.

`SPEC.md` is the source of truth; deviations are recorded as ADRs in
`docs/adr/`.

## Status

| milestone | state |
|---|---|
| M0 bootstrap | workspace, lints, configs, CI, `rhizome params`, spec-sync |
| M1 core + ops + plan | f64 op references with adjoints, gradient checks, graph IR, backward derivation, arena planning |

## Quick start

```sh
cargo test --workspace
cargo run -p rhizome-cli -- params --format=markdown
cargo run -p rhizome-cli -- inspect configs/edge.toml
cargo run -p rhizome-cli -- verify
cargo run -p rhizome-cli -- spec-check
```

The workspace has **no third-party dependencies** (ADR 0001), so it builds and
tests fully offline.

## Layout

```
crates/core   schema, dtypes, TOML subset parser, parameter accounting
crates/ops    closed op vocabulary: f64 references, adjoints, Philox RNG
crates/plan   graph IR, backward derivation, liveness + arena, plan hashing
crates/cli    the `rhizome` binary
configs       test-s, test-m, test-l, edge, standard, flagship
docs          SPEC support: kernels, determinism, muP, ADRs, milestones
```

## License

Apache-2.0.
