# Milestones

A milestone is complete only when its acceptance criteria are verified by a
workflow run whose URL is recorded here (R8).

| milestone | status | acceptance criteria | workflow run | date |
|---|---|---|---|---|
| M0 bootstrap | complete | workspace + pinned toolchain + workspace lints build clean; `rhizome params` computes params, FLOPs and deploy bytes for all six configs; the `spec-sync` check proves SPEC.md equals the generated table; the `ci` workflow is green | [33893781755](https://github.com/efikoman76-ship-it/RHIZOME/actions/runs/33893781755) | 2026-09-04 |
| M1 core + ops + plan | in progress | f64 reference for every implemented op with finite-difference gradient checks (T1); graph IR with SSA validation, backward derivation, liveness-based arena planning verified free of overlapping live buffers, and stable plan hashing (T3) | [33893781755](https://github.com/efikoman76-ship-it/RHIZOME/actions/runs/33893781755) | 2026-09-04 |

## M0 evidence

The green run above executes, on `windows-latest`:

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo build --workspace --all-targets`
- `cargo test --workspace` — 92 tests, 0 failures, 0 ignored (R2 forbids
  `#[ignore]`)
- `cargo doc --no-deps --workspace` with `RUSTDOCFLAGS=-D warnings`
- `rhizome params --format=markdown`, `rhizome verify`, `rhizome spec-check`

## M1 remaining work

The op vocabulary implemented so far covers the elementwise, norm, linear
algebra, gated-delta, MLA, routing, loss, sampling and RNG families. Still to
land before M1 can be marked complete: the byte-path ops
(`hash_ngram_embed`, `patch_pool`, `patch_crossattn`, `entropy_boundary`), the
collective graph nodes, the optimizer ops (`adamw_step`, `sparse_adamw_step`,
`ema_update`), the MX/FP8 quantise-dequantise pair as graph ops, and the
whole-loss finite-difference check on Test-S.
