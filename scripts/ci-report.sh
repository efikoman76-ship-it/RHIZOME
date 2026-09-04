#!/usr/bin/env bash
# Publish the captured CI logs as a commit comment.
#
# GitHub step logs and workflow artifacts are served from blob storage, which
# the development environment cannot reach; the REST API on api.github.com can
# be reached. Posting the logs as a commit comment therefore makes every CI
# failure fully readable from outside the runner (ADR 0002).
set -uo pipefail

status="${1:-unknown}"
mkdir -p logs

body_file="$(mktemp)"
{
  echo "## CI ${status} — \`${GITHUB_WORKFLOW:-ci}\` run [${GITHUB_RUN_ID:-?}](${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-}/actions/runs/${GITHUB_RUN_ID:-})"
  echo
  for f in logs/*.log; do
    [ -e "${f}" ] || continue
    name="$(basename "${f}" .log)"
    lines="$(wc -l <"${f}" | tr -d ' ')"
    echo "<details><summary>${name} (${lines} lines)</summary>"
    echo
    echo '```'
    # Keep the whole file when small, otherwise head + tail around the failure.
    if [ "${lines}" -le 400 ]; then
      cat "${f}"
    else
      head -n 80 "${f}"
      echo "... [${lines} lines total, middle elided] ..."
      tail -n 300 "${f}"
    fi
    echo '```'
    echo
    echo '</details>'
    echo
  done
} >"${body_file}"

# The commit comment API accepts a large body; truncate defensively at 60000
# bytes so the request never fails for size.
python - "${body_file}" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
b = p.read_text(encoding="utf-8", errors="replace")
limit = 60000
if len(b) > limit:
    b = b[:limit] + "\n\n... [truncated at 60000 characters] ...\n"
p.write_text(b, encoding="utf-8")
PY

gh api "repos/${GITHUB_REPOSITORY}/commits/${GITHUB_SHA}/comments" \
  -f body=@"${body_file}" >/dev/null && echo "ci-report: posted commit comment"
