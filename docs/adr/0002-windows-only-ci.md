# 0002. Windows-only CI builds with in-log diagnostics

- Status: accepted
- Date: 2026-09-04

## Context

Section 11 of the build prompt asks for ubuntu, macOS and Windows jobs. The
repository owner directed that CI build on Windows only, and that all
diagnostics be printed into the workflow log because workflow artifacts and
logs cannot be inspected interactively from the development environment.

## Decision

`ci.yml` runs a single `test-windows` job that performs fmt, clippy, build,
test, doc, `rhizome params`, `rhizome verify` and `spec-check`. Every step pipes
its output through `tee` into a `.log` file, the logs are uploaded as an
artifact with `if: always()`, and the generated parameter table is printed
between explicit `BEGIN`/`END` markers so it can be copied out of the log.

## Alternatives considered

- **Matrix over three operating systems.** Rejected per the owner's direction;
  it also triples queue time for a workspace with no OS-specific code yet.
- **Quiet steps.** Rejected: without the log text a red run cannot be
  diagnosed from outside the runner.

## Consequences

- Linux- and macOS-specific regressions are not caught until those jobs are
  restored, which must happen before any SIMD or Metal kernel lands.
- Every failing step leaves both inline log text and an uploaded artifact.
