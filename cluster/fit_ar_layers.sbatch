#!/usr/bin/env bash
#SBATCH --job-name=wfi-ar23
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# This job deliberately uses one Slurm allocation and two exclusive steps.
# The two fits therefore start together without competing for the same CPUs.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"

if [[ -n "${ENV_SETUP_SCRIPT:-}" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_SETUP_SCRIPT"
fi
if [[ -n "${VENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

STATISTICS_DIR="${STATISTICS_DIR:-$PROJECT_ROOT/wfi_tile_log_statistics_512}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/ar_layers}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CPUS_PER_CASE="${CPUS_PER_CASE:-${SLURM_CPUS_PER_TASK:-16}}"

if [[ ! -d "$STATISTICS_DIR" ]]; then
    printf 'Statistics directory does not exist: %s\n' "$STATISTICS_DIR" >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Python executable was not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi

shopt -s nullglob
statistics_files=("$STATISTICS_DIR"/wfi*_tile_log_statistics_*.csv)
if (( ${#statistics_files[@]} != 18 )); then
    printf 'Expected 18 completed SCA statistics CSVs in %s; found %d:\n' \
        "$STATISTICS_DIR" "${#statistics_files[@]}" >&2
    printf '  %s\n' "${statistics_files[@]}" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

# Keep BLAS/JAX thread pools inside each task allocation.  Set XLA_FLAGS only
# if the caller has not already supplied a different XLA configuration.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export OMP_NUM_THREADS="$CPUS_PER_CASE"
export MKL_NUM_THREADS="$CPUS_PER_CASE"
export OPENBLAS_NUM_THREADS="$CPUS_PER_CASE"
export PYTHONUNBUFFERED=1
if [[ -z "${XLA_FLAGS:-}" ]]; then
    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$CPUS_PER_CASE"
fi

run_case() {
    local layers="$1"
    local output_dir="$OUTPUT_ROOT/ar${layers}layer"
    local case_log="$OUTPUT_ROOT/ar${layers}layer.log"
    local -a fit_args=(
        --all-sca
        --all-sca-statistics-dir "$STATISTICS_DIR"
        --ar-layers "$layers"
        --output-dir "$output_dir"
        --sed-mode "${SED_MODE:-core}"
        --wavelength-stride "${WAVELENGTH_STRIDE:-1}"
        --log-noise-sigma "${LOG_NOISE_SIGMA:-0.005}"
        --likelihood-weight "${LIKELIHOOD_WEIGHT:-1e-8}"
        --warmup "${WARMUP:-300}"
        --samples "${SAMPLES:-500}"
        --chains "${CHAINS:-2}"
        --seed "${SEED:-20260922}"
        --target-accept-prob "${TARGET_ACCEPT_PROB:-0.85}"
        --max-tree-depth "${MAX_TREE_DEPTH:-8}"
    )
    local -a extra_args=()

    if [[ -n "${MAX_TILES:-}" ]]; then
        fit_args+=(--max-tiles "$MAX_TILES")
    fi
    if [[ -n "${WAVELENGTH_STEP_NM:-}" ]]; then
        fit_args+=(--wavelength-step-nm "$WAVELENGTH_STEP_NM")
    fi
    if [[ "${SHARED_AR_OPTICAL_PROPERTIES:-0}" == "1" ]]; then
        fit_args+=(--shared-ar-optical-properties)
    fi
    if [[ "${PARALLEL_CHAINS:-0}" == "1" ]]; then
        fit_args+=(--parallel-chains)
    fi
    if [[ -n "${FIT_EXTRA_ARGS:-}" ]]; then
        # FIT_EXTRA_ARGS is intentionally a shell-style flag string.  Keep
        # paths with spaces out of it; use the named variables above instead.
        read -r -a extra_args <<< "$FIT_EXTRA_ARGS"
        fit_args+=("${extra_args[@]}")
    fi

    mkdir -p "$output_dir"
    printf 'Starting %s-layer fit; log: %s\n' "$layers" "$case_log"
    srun \
        --exclusive \
        --exact \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS_PER_CASE" \
        --cpu-bind=cores \
        "$PYTHON_BIN" fit_wfi_tmm_numpyro.py "${fit_args[@]}" \
        >"$case_log" 2>&1
}

run_case 2 &
ar2_pid=$!
run_case 3 &
ar3_pid=$!

status=0
wait "$ar2_pid" || status=$?
wait "$ar3_pid" || status=$?

if (( status != 0 )); then
    printf 'At least one AR fit failed; see %s/ar2layer.log and %s/ar3layer.log\n' \
        "$OUTPUT_ROOT" "$OUTPUT_ROOT" >&2
fi
exit "$status"
