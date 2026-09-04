#!/usr/bin/env bash
# Run every CI check, capturing each into logs/<name>.log.
#
# All checks run even when one fails, so a single run reports every problem
# instead of stopping at the first. The exit status is non-zero if any check
# failed.
set -uo pipefail

mkdir -p logs
failed=()

step() {
  local name="$1"
  shift
  echo "::group::${name}: $*"
  local log="logs/${name}.log"
  if "$@" >"${log}" 2>&1; then
    cat "${log}"
    echo "::endgroup::"
    echo "${name}: ok"
  else
    local status=$?
    cat "${log}"
    echo "::endgroup::"
    echo "::error title=${name}::failed with exit status ${status}; see the commit comment for the full log"
    failed+=("${name}")
  fi
}

step fmt cargo fmt --all -- --check
step clippy cargo clippy --workspace --all-targets -- -D warnings
step build cargo build --workspace --all-targets
step test cargo test --workspace -- --nocapture
RUSTDOCFLAGS="-D warnings" step doc cargo doc --no-deps --workspace
step params cargo run -q -p rhizome-cli -- params --format=markdown
step verify cargo run -q -p rhizome-cli -- verify
step spec-check cargo run -q -p rhizome-cli -- spec-check

if [ "${#failed[@]}" -gt 0 ]; then
  echo "failed checks: ${failed[*]}"
  exit 1
fi
echo "all checks passed"
