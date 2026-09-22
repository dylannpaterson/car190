# Cluster Slurm Workflow

`cluster/fit_ar_layers.sbatch` requests one node with two Slurm tasks. It
starts the two-layer and three-layer TMM fits as exclusive `srun` steps, so the
fits run concurrently while each receives its own CPU allocation. Results and
per-case logs are written under the configured output directory:

```text
results/ar_layers/ar2layer/
results/ar_layers/ar3layer/
results/ar_layers/ar2layer.log
results/ar_layers/ar3layer.log
```

## Prerequisites

- SSH access to the cluster and `rsync` on the local machine;
- 18 completed `wfi*_tile_log_statistics_*.csv` files in
  `wfi_tile_log_statistics_512/`;
- a cluster Python environment containing the dependencies in
  `requirements-wfi01-ice.txt`, plus working CPU `jax` and `numpyro` installs;
- outbound HTTPS from compute nodes to the two public Roman SED/QE tables used
  by the current fit code. If compute nodes have no network access, those
  tables must be made locally available and the fit input path handling should
  be adapted before submitting.

## Push And Submit

Run this from the project root or from any directory after setting the paths:

```bash
export CLUSTER='user@login.example.edu'
export REMOTE_ROOT='/path/on/cluster/wfi-ice-analysis'

bash cluster/push_and_submit.sh
```

The helper transfers source files and the 18 completed statistics CSVs, then
submits the Slurm job. It does not transfer raw `MAST/` products or generated
analysis outputs. The production fit evaluates the TMM at the public SED sample
nodes by default.
Set `WAVELENGTH_STEP_NM=0.5` and leave `WAVELENGTH_STRIDE=1` to test a finer
grid on the cluster.

For a virtual environment on the cluster, set its directory before submitting:

```bash
export VENV="$HOME/venvs/wfi-ice"
bash cluster/push_and_submit.sh
```

If the cluster uses modules or another setup command, put those commands in a
remote shell script and set `ENV_SETUP_SCRIPT` to that script. The Slurm job
sources it before locating Python.

## Resources And Settings

The batch file defaults to one node, two tasks, 16 CPUs per fit, 128 GB total
memory, and 48 hours. Override site-specific options through `SBATCH_ARGS`:

```bash
export SBATCH_ARGS='--partition=cpu --account=my_account --time=72:00:00 --mem=256G --cpus-per-task=32'
bash cluster/push_and_submit.sh
```

`--cpus-per-task` is per fit; both fits share the total memory request. The
default NumPyro chains are sequential inside each fit, while the two layer
counts run in parallel. Useful fit overrides are exported as environment
variables, for example:

```bash
export WARMUP=1000
export SAMPLES=1000
export SHARED_AR_OPTICAL_PROPERTIES=1
bash cluster/push_and_submit.sh
```

Use `FIT_EXTRA_ARGS` for additional boolean or simple CLI flags, such as
`FIT_EXTRA_ARGS='--np-modulation'`. Avoid spaces inside one flag value; use the
named variables above for paths and numeric settings.

The job can also be submitted manually from the remote project directory:

```bash
STATISTICS_DIR=/path/to/wfi_tile_log_statistics_512 \
OUTPUT_ROOT=/path/to/results/ar_layers \
sbatch cluster/fit_ar_layers.sbatch
```

Monitor the allocation with `squeue`, inspect `slurm-wfi-ar23-<jobid>.out`,
and inspect the two case logs under `OUTPUT_ROOT`.
