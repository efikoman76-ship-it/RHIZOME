#!/usr/bin/env bash
# Run one CI step, capturing its output.
#
# On failure the captured output is re-emitted as GitHub `::error::`
# annotations. Annotations are readable through the GitHub REST API, which is
# what makes a red run diagnosable from outside the runner (ADR 0002); plain
# step logs and uploaded artifacts require blob-storage access that the
# development environment does not have.
#
# usage: ci-step.sh <name> <command...>
set -uo pipefail

name="$1"
shift
cmd="$*"

mkdir -p logs
log="logs/${name}.log"

echo "::group::${name}: ${cmd}"
set +e
# shellcheck disable=SC2086
eval "${cmd}" >"${log}" 2>&1
status=$?
set -e
cat "${log}"
echo "::endgroup::"

if [ "${status}" -eq 0 ]; then
  echo "${name}: ok"
  exit 0
fi

# Annotate the last 60 lines, one annotation per line, with %0A-escaped
# content so multi-line rustc diagnostics survive.
echo "::error title=${name} failed::exit status ${status} for: ${cmd}"
tail -n 60 "${log}" | while IFS= read -r line; do
  # Skip blank lines to keep the annotation list readable.
  [ -z "${line}" ] && continue
  printf '::error title=%s::%s\n' "${name}" "${line//%/%25}"
done
exit "${status}"
