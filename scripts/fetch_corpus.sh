#!/usr/bin/env bash
# Fetch the evaluation corpus: the Markdown docs of three open-source projects,
# pinned to release tags so the line numbers in testdata/questions.jsonl stay
# valid.
#
# Usage: scripts/fetch_corpus.sh [destination-dir]
# Default destination: testdata/corpus
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/testdata/corpus}"

# name | repository | tag | docs directory inside the repository
REPOS=(
  "mkdocs|https://github.com/mkdocs/mkdocs.git|1.6.1|docs"
  "cobra|https://github.com/spf13/cobra.git|v1.8.1|site/content"
  "prettier|https://github.com/prettier/prettier.git|3.3.3|docs"
)

command -v git >/dev/null 2>&1 || { echo "fetch_corpus: git is required" >&2; exit 1; }

mkdir -p "$DEST"

for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url tag sub <<<"$entry"
  target="$DEST/$name"

  if [ -f "$target/.source" ] && grep -q " $tag " "$target/.source"; then
    echo "fetch_corpus: $name@$tag already present, skipping"
    continue
  fi

  echo "fetch_corpus: fetching $name@$tag ($sub/)"
  tmp="$(mktemp -d)"
  # Sparse, blob-less, depth-1 clone: only the docs directory (plus the few
  # files in the repository root) is downloaded. autocrlf is off so the files
  # are byte-identical on every platform. Cone mode takes a plain directory
  # name, which also keeps Git Bash on Windows from rewriting it as a path.
  git clone --quiet --depth 1 --branch "$tag" --filter=blob:none --sparse \
    --config core.autocrlf=false --config advice.detachedHead=false \
    "$url" "$tmp/repo"
  git -C "$tmp/repo" sparse-checkout set "$sub"

  if [ ! -d "$tmp/repo/$sub" ]; then
    echo "fetch_corpus: $sub/ not found in $name@$tag" >&2
    rm -rf "$tmp"
    exit 1
  fi

  rm -rf "$target"
  mkdir -p "$target"
  cp -R "$tmp/repo/$sub/." "$target/"
  # Only Markdown is needed; drop images, stylesheets and the like.
  find "$target" -type f ! -name '*.md' ! -name '*.markdown' -delete
  find "$target" -type d -empty -delete
  printf '%s %s %s\n' "$url" "$tag" "$(git -C "$tmp/repo" rev-parse HEAD)" >"$target/.source"
  rm -rf "$tmp"

  count="$(find "$target" -type f -name '*.md' | wc -l | tr -d ' ')"
  echo "fetch_corpus: $name: $count Markdown files"
done

echo "fetch_corpus: corpus ready in $DEST"
