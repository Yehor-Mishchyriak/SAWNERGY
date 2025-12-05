#!/usr/bin/env bash
# ============================================================
# Deploy: test -> build -> (optional upload) -> push dev -> push curated public branch
# Location: CICD/deploy.sh
# ============================================================

set -euo pipefail

# ---------- Config (override via env if needed) ----------
# Dev repository remote & branch
DEV_REMOTE="${DEV_REMOTE:-origin}"
DEV_PUSH_BRANCH="${DEV_PUSH_BRANCH:-main}"

# Public mirror remote (add once with `git remote add public <url>`)
PUBLIC_REMOTE="${PUBLIC_REMOTE:-public}"
# Public release branch name = ${PUBLIC_REL_PREFIX}${version}
PUBLIC_REL_PREFIX="${PUBLIC_REL_PREFIX:-Sawnergy-release-v}"
# Also force-update this branch in the public repo
PUBLIC_MAIN_BRANCH="${PUBLIC_MAIN_BRANCH:-main}"

# Python to use
PY="${PYTHON:-python}"

# ---------- Resolve paths ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# ---------- Helpers ----------
section () { printf "\n\033[1m==> %s\033[0m\n" "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1"; exit 1; }; }

# ---------- Tool checks ----------
need "$PY"
need git
if [[ "${UPLOAD:-0}" == "1" ]]; then
  command -v twine >/dev/null 2>&1 || { echo "twine not found on PATH"; exit 1; }
fi

# ---------- 1) Clean old build artifacts ----------
section "Cleaning old artifacts"
rm -rf dist build ./*.egg-info

# ---------- 2) Run tests (log to file) ----------
section "Running tests"
LOG_DIR="${REPO_ROOT}/test_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/tests_$(date +%Y-%m-%d_%H%M%S).log"
set +e
"$PY" CICD/test_runner.py 2>&1 | tee "${LOG_FILE}"
TEST_STATUS=${PIPESTATUS[0]}
set -e
if [[ $TEST_STATUS -ne 0 ]]; then
  echo "❌ Tests failed. See ${LOG_FILE}"
  exit $TEST_STATUS
fi
echo "✅ Tests passed. Log: ${LOG_FILE}"

# ---------- 3) Build sdist + wheel (also bumps pkg_meta.json version) ----------
section "Building sdist and wheel"
"$PY" CICD/build_whl.py sdist bdist_wheel

# Read bumped version for naming branches/tags
PKG_VERSION="$("$PY" - <<'PY'
from pathlib import Path; import json
print(json.loads(Path("pkg_meta.json").read_text(encoding="utf-8"))["version"])
PY
)"
REL_BRANCH="${PUBLIC_REL_PREFIX}${PKG_VERSION}"   # public version branch name
DEV_REL_BRANCH="${REL_BRANCH}"                    # dev release branch matches name
TAG_NAME="v${PKG_VERSION}"

# ---------- 4) Show artifacts ----------
section "Artifacts in dist/"
ls -lh dist || { echo "dist/ not found"; exit 1; }

# ---------- 5) Optional upload to PyPI ----------
if [[ "${UPLOAD:-0}" == "1" ]]; then
  section "Uploading to PyPI"
  : "${PYPI_API_TOKEN:?Set PYPI_API_TOKEN (pypi-****). Example: export PYPI_API_TOKEN=...}"
  twine upload -u __token__ -p "${PYPI_API_TOKEN}" dist/*
  echo "✅ Upload complete."
else
  echo "[i] Skipping upload (set UPLOAD=1 to enable)."
fi

# ---------- 6) Push EVERYTHING to dev remote ----------
section "Pushing to dev (${DEV_REMOTE})"
git rev-parse --is-inside-work-tree >/dev/null 2>&1

# Commit version bump if not yet committed
if ! git diff --quiet -- pkg_meta.json; then
  git add pkg_meta.json
  git commit -m "chore: bump version to ${PKG_VERSION}"
fi

# Push current HEAD to dev main (or configured branch)
git push "${DEV_REMOTE}" HEAD:"refs/heads/${DEV_PUSH_BRANCH}"

# Push a dev release branch and a tag
git push "${DEV_REMOTE}" HEAD:"refs/heads/${DEV_REL_BRANCH}"
if git rev-parse -q --verify "refs/tags/${TAG_NAME}" >/dev/null; then
  echo "[i] Tag ${TAG_NAME} already exists locally; skipping create."
else
  git tag -a "${TAG_NAME}" -m "Release ${PKG_VERSION}"
fi
git push "${DEV_REMOTE}" "refs/tags/${TAG_NAME}"

echo "✅ Dev push complete: ${DEV_PUSH_BRANCH}, ${DEV_REL_BRANCH}, tag ${TAG_NAME}"

# ---------- 7) Publish curated subset to PUBLIC ----------
section "Publishing curated subset to public remote '${PUBLIC_REMOTE}' branches '${REL_BRANCH}' and '${PUBLIC_MAIN_BRANCH}'"

# Ensure public remote exists in this repo
git remote get-url "${PUBLIC_REMOTE}" >/dev/null 2>&1 || {
  echo "❌ Remote '${PUBLIC_REMOTE}' not found."
  echo "   Add it once:  git remote add ${PUBLIC_REMOTE} <git-url-of-public-mirror>"
  exit 1
}

# Create a detached worktree at the commit behind DEV_PUSH_BRANCH (avoids 'already checked out' issues)
TMP_WT="$(mktemp -d)"
trap 'git worktree remove -f "${TMP_WT}" 2>/dev/null || true; rm -rf "${TMP_WT}"' EXIT

SRC_COMMIT="$(git rev-parse "${DEV_PUSH_BRANCH}")"
git worktree add --quiet --detach "${TMP_WT}" "${SRC_COMMIT}"
pushd "${TMP_WT}" >/dev/null

# --- PATCH: use a unique orphan branch and clean it up after ---
SYNC_BRANCH="public-sync-${PKG_VERSION}-$(date +%s)"
# nuke any stale branch with same name (paranoia)
git branch -D "${SYNC_BRANCH}" >/dev/null 2>&1 || true
if git switch --orphan "${SYNC_BRANCH}" 2>/dev/null; then :; else git checkout --orphan "${SYNC_BRANCH}"; fi
git rm -rf . >/dev/null 2>&1 || true
git clean -fdx >/dev/null 2>&1 || true

# Restore ONLY paths that actually exist at SRC_COMMIT (avoid pathspec errors)
CANDIDATES=(
  sawnergy
  CITATION.cff
  LICENSE LICENSE.txt
  NOTICE NOTICE.txt
  README.md README.MD
  CREDITS.md CREDITS.MD
  assets
  tests
  example_MD_for_quick_start
  test_logs
  JOSS
  .github
  docs.md
)
SELECTED=()
for p in "${CANDIDATES[@]}"; do
  if git show "${SRC_COMMIT}:${p}" >/dev/null 2>&1; then
    SELECTED+=("$p")
  fi
done

if ((${#SELECTED[@]} == 0)); then
  echo "[i] No whitelisted files/dirs exist at ${SRC_COMMIT}; nothing to publish."
else
  git checkout "${SRC_COMMIT}" -- "${SELECTED[@]}"
  git add -A
  if [[ -n "$(git status --porcelain)" ]]; then
    git commit -m "public: release ${PKG_VERSION}"
    # Push curated subset to version-named branch and to public main
    git push -f "${PUBLIC_REMOTE}" "${SYNC_BRANCH}:${REL_BRANCH}"
    git push -f "${PUBLIC_REMOTE}" "${SYNC_BRANCH}:${PUBLIC_MAIN_BRANCH}"
    echo "✅ Public push complete: ${PUBLIC_REMOTE}/${REL_BRANCH} and ${PUBLIC_REMOTE}/${PUBLIC_MAIN_BRANCH}"
  else
    echo "[i] Nothing to publish (no changes after filtering)."
  fi
fi

# cleanup local temp sync branch so future runs won't collide
git checkout --detach >/dev/null 2>&1 || true
git branch -D "${SYNC_BRANCH}" >/dev/null 2>&1 || true

popd >/dev/null
git worktree remove -f "${TMP_WT}" || true
trap - EXIT

section "Done"
echo "Version: ${PKG_VERSION}"
echo "Dev: ${DEV_REMOTE} -> ${DEV_PUSH_BRANCH}, ${DEV_REL_BRANCH}, tag ${TAG_NAME}"
echo "Public: ${PUBLIC_REMOTE} -> ${REL_BRANCH} and ${PUBLIC_MAIN_BRANCH}"
