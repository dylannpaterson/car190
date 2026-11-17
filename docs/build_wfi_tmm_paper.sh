#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="$ROOT/docs/WFI_TMM_PRODUCTION_MODEL.pdf"
TEMP_OUTPUT="$(mktemp --tmpdir="$ROOT/docs" --suffix=.pdf ".WFI_TMM_PRODUCTION_MODEL.XXXXXX")"
trap 'rm -f -- "$TEMP_OUTPUT"' EXIT

pandoc "$ROOT/docs/WFI_TMM_PRODUCTION_MODEL.md" \
    --from=markdown-implicit_figures \
    --to=pdf \
    --output="$TEMP_OUTPUT" \
    --pdf-engine=xelatex \
    --resource-path="$ROOT/docs" \
    --variable=geometry:margin=1in \
    --variable=fontsize:11pt \
    --variable=colorlinks:true \
    --variable=linkcolor:blue

mv -f -- "$TEMP_OUTPUT" "$OUTPUT"
trap - EXIT
