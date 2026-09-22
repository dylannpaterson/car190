#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER="${CLUSTER:?Set CLUSTER to an SSH host or config alias, for example user@login}"
REMOTE_ROOT="${REMOTE_ROOT:?Set REMOTE_ROOT to an absolute project directory on the cluster}"

LOCAL_STATS_DIR="${LOCAL_STATS_DIR:-$PROJECT_ROOT/wfi_tile_log_statistics_512}"
REMOTE_STATS_DIR="${REMOTE_STATS_DIR:-$REMOTE_ROOT/wfi_tile_log_statistics_512}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-$REMOTE_ROOT/results/ar_layers}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SBATCH_ARGS="${SBATCH_ARGS:-}"

shell_quote() {
    printf '%q' "$1"
}

if ! command -v rsync >/dev/null 2>&1; then
    printf 'rsync is required on the local machine.\n' >&2
    exit 1
fi
if ! command -v ssh >/dev/null 2>&1; then
    printf 'ssh is required on the local machine.\n' >&2
    exit 1
fi
if [[ ! -d "$LOCAL_STATS_DIR" ]]; then
    printf 'Local statistics directory does not exist: %s\n' "$LOCAL_STATS_DIR" >&2
    exit 1
fi

shopt -s nullglob
statistics_files=("$LOCAL_STATS_DIR"/wfi*_tile_log_statistics_*.csv)
if (( ${#statistics_files[@]} != 18 )); then
    printf 'Expected 18 completed SCA statistics CSVs in %s; found %d:\n' \
        "$LOCAL_STATS_DIR" "${#statistics_files[@]}" >&2
    printf '  %s\n' "${statistics_files[@]}" >&2
    exit 1
fi

remote_root_q="$(shell_quote "$REMOTE_ROOT")"
remote_stats_q="$(shell_quote "$REMOTE_STATS_DIR")"
remote_output_q="$(shell_quote "$REMOTE_OUTPUT_ROOT")"

printf 'Preparing remote directories under %s ...\n' "$REMOTE_ROOT"
ssh "$CLUSTER" "mkdir -p -- $remote_root_q $remote_stats_q $remote_output_q"

printf 'Syncing source files ...\n'
# Only source, documentation, and cluster-control files are copied here.  Raw
# MAST products and generated analysis products stay local unless explicitly
# handled below.
rsync -az --prune-empty-dirs \
    --exclude='/.git/' \
    --exclude='/MAST/' \
    --exclude='/__pycache__/' \
    --exclude='/.pytest_cache/' \
    --include='/*.py' \
    --include='/*.txt' \
    --include='/README.md' \
    --include='/.gitignore' \
    --include='/docs/' \
    --include='/docs/***' \
    --include='/cluster/' \
    --include='/cluster/***' \
    --exclude='*' \
    "$PROJECT_ROOT/" "$CLUSTER:$REMOTE_ROOT/"

printf 'Syncing 18 statistics CSVs ...\n'
rsync -az --prune-empty-dirs \
    --include='wfi*_tile_log_statistics_*.csv' \
    --exclude='*' \
    "$LOCAL_STATS_DIR/" "$CLUSTER:$REMOTE_STATS_DIR/"

# These values are added explicitly because sbatch normally exports the
# environment of the remote login shell, not the local shell running rsync.
export_spec="ALL,PROJECT_ROOT=$REMOTE_ROOT,STATISTICS_DIR=$REMOTE_STATS_DIR,OUTPUT_ROOT=$REMOTE_OUTPUT_ROOT,PYTHON_BIN=$PYTHON_BIN"
for variable in \
    ENV_SETUP_SCRIPT VENV WARMUP SAMPLES CHAINS SEED MAX_TILES \
    SED_MODE WAVELENGTH_STEP_NM WAVELENGTH_STRIDE LOG_NOISE_SIGMA LIKELIHOOD_WEIGHT \
    TARGET_ACCEPT_PROB MAX_TREE_DEPTH SHARED_AR_OPTICAL_PROPERTIES \
    PARALLEL_CHAINS FIT_EXTRA_ARGS; do
    if [[ -n "${!variable:-}" ]]; then
        export_spec="$export_spec,$variable=${!variable}"
    fi
done

sbatch_args=()
if [[ -n "$SBATCH_ARGS" ]]; then
    read -r -a sbatch_args <<< "$SBATCH_ARGS"
fi
sbatch_command=(sbatch)
if (( ${#sbatch_args[@]} > 0 )); then
    sbatch_command+=("${sbatch_args[@]}")
fi
sbatch_command+=("--export=$export_spec" "$REMOTE_ROOT/cluster/fit_ar_layers.sbatch")
remote_command="cd $(shell_quote "$REMOTE_ROOT") &&"
for argument in "${sbatch_command[@]}"; do
    remote_command="$remote_command $(shell_quote "$argument")"
done

printf 'Submitting two- and three-layer fits on %s ...\n' "$CLUSTER"
ssh "$CLUSTER" "$remote_command"
