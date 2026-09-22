# Roman WFI Ice Analysis

This repository contains the Roman WFI ice-monitoring analysis scripts. The
primary workflow converts calibrated MAST products into compact tile
sufficient statistics, then fits an air/ice/AR/HgCdTe transfer-matrix model
to the statistics with NumPyro.

The current production model is designed for all 18 WFI SCAs:

- one shared ice-thickness trajectory and ACM step (optionally one step per LED);
- a global AR population mean;
- SCA-level AR offsets with shrinkage;
- tile-level AR deviations within each SCA;
- selectable one-, two-, or three-layer dielectric AR stack;
- nominal fixed HgCdTe optical constants.

The opt-in `--fit-conductive-layer` mode inserts a global thin Drude-like layer
between the AR stack and HgCdTe. Its positive log-thickness, plasma-strength,
and damping-ratio parameters provide a low-dimensional physical boundary
perturbation without replacing the transfer-matrix model.

The opt-in `--fit-qe-rbf` mode applies normalized Gaussian RBF corrections to
the ground-measured QE weighting. Its default four nodes are at 650, 950,
1250, and 1600 nm; `--qe-rbf-node-wavelengths-nm`, `--qe-rbf-width-nm`, and
`--qe-rbf-prior-sigma-log` provide basis sensitivity controls. Use it with
`--np-modulation` to fit QE shape and the ice-rate trajectory jointly.

The opt-in `--use-asw-optical-constants` mode replaces the constant lossless
ice index with the wavelength-dependent 100 K amorphous-H2O n/k table from the
Mastrapa laboratory data set. The table is used as a physical sensitivity test,
not as a claim that the flight ice phase and temperature are known exactly.

The opt-in `--led-dependent-acm-step` mode fits six independent log throughput
steps, one for each LED, instead of forcing one common ACM step across all LEDs.
Use it as a calibration-systematics diagnostic; it does not alter the shared ice
trajectory or transfer-matrix response.

The opt-in `--fully-np-trajectory` mode replaces the square-root accumulation law
with a positive instantaneous rate at every repeat node. Rates are log-linearly
interpolated and integrated to preserve monotonic ice thickness; the existing
time-aware log-rate curvature penalty remains active. The node-rate prior is
centered on the historical initial-rate scale, but no square-root time dependence
is imposed. It is mutually exclusive with `--np-modulation`.

The older WFI01-only analyses and effective-boundary diagnostics remain in the
repository as comparison tools. They are not the default production model.

## Environment

The pinned base dependencies are in `requirements-wfi01-ice.txt`. The
NumPyro fit additionally requires working `jax`, `numpyro`, and `corner`
installations. The current development environment is named `bulge_env`.

The production TMM computes its own transfer-matrix response and generates a
480--2700 nm wavelength grid from the public QE support. The historical
`--model-path` option is retained for command compatibility but is not read by
the production TMM; older SOC/effective-boundary tools still use that table.

```bash
python -m pip install -r requirements-wfi01-ice.txt
```

## Inputs

`all_wfi_rcs_flat_index.csv` is the tracked 2,268-product inventory used by
the streaming workflow. It contains 126 exposures for each of 18 SCAs, with
six LEDs and 21 repeats per SCA.

`wfi01_tile_log_statistics_512.csv` is the tracked WFI01 fixture used by the
tests and for single-SCA smoke checks. Raw MAST products, API tokens, partial
downloads, and generated analysis products are intentionally not versioned.

## Workflow

### 1. Stream tile statistics

The downloader keeps one LED baseline and one current exposure in memory. It
writes a checkpoint per exposure and atomically publishes each completed SCA
CSV. It does not write raw ASDF products to disk.

```bash
python build_wfi_tile_statistics_stream.py \
  --all-detectors \
  --index-path all_wfi_rcs_flat_index.csv \
  --output-dir wfi_tile_log_statistics_512
```

The default behavior resumes existing `.parts/group_*.csv` checkpoints and
skips completed detector CSVs. Use `--no-resume` only when intentionally
recomputing existing statistics. The all-SCA fit requires 18 completed final
CSV files; partial checkpoint directories do not count.

When the MAST inventory is refreshed, use `--incremental` instead of
`--no-resume`. Incremental mode matches existing statistics by MAST file-set
name, downloads only newly discovered exposures (and the baseline images needed
to compare them), and preserves the existing sufficient statistics. It also
handles new exposures inserted into the middle of a LED sequence.

```bash
python query_rcs_flats.py \
  --all-detectors \
  --output all_wfi_rcs_flat_index.csv
python build_wfi_tile_statistics_stream.py \
  --all-detectors \
  --index-path all_wfi_rcs_flat_index.csv \
  --output-dir wfi_tile_log_statistics_512 \
  --incremental
```

The MAST query defaults to RCS program `1039`; the statistics builder rejects
indices containing other programs because their partial LED sequences are not
compatible with the RCS time series.

The MAST token is read from `~/Downloads/mast_api_token.txt` by default. Keep
the token outside the repository.

### 2. Fit the TMM model

After all 18 statistics files are complete:

```bash
python fit_wfi_tmm_numpyro.py \
  --all-sca \
  --all-sca-statistics-dir wfi_tile_log_statistics_512 \
  --ar-layers 3 \
  --output-dir wfi_all_sca_tmm_numpyro_tile_production
```

Use `--ar-layers 1` or `--ar-layers 2` for comparison models. If the default
output path is used, they are written to paths ending in `_ar1layer` and
`_ar2layer` rather than overwriting the three-layer fit. Layer order is from
air toward HgCdTe.

For the reduced all-tile model, add `--shared-ar-optical-properties`. This
shares each layer's refractive index across every tile while retaining
hierarchical tile-to-tile thickness variation. The dielectric model fixes
`k=0` by default; `--fit-ar-absorption` is an explicit sensitivity option.

The default production settings are a three-layer dielectric AR stack, 300
warmup draws, 500 posterior draws, two chains, all available tiles, exact SED
wavelength nodes and stride 1, log-noise sigma 0.005, and AR population-mean
prior sigma 2.0. The likelihood pixel weight defaults to the inverse median
valid-pixel count, converting each tile/group pixel SSE to approximately one
effective datum before the selected-tile average. The aggregate SSE is then
normalized by the number of selected tiles so adding tiles does not make NUTS
geometrically intractable. Use `--likelihood-weight` only for an explicit
sensitivity run.

For a prior-sensitivity run, `--ar-mean-prior-sigma 5` makes the global AR raw
parameter prior much broader than the default. This is useful for the focused
one-tile diagnostic, but should not be treated as a final physical prior.
Trajectory-rate sensitivity can be run without changing the likelihood, for
example `--trajectory-rate-prior-mean-nm-day 10
--trajectory-rate-prior-sigma-log 1`. The rate prior is placed on the
unmodulated instantaneous rate at the first observed time. The model integrates
that positive rate to obtain ice thickness; with `--np-modulation`, repeat-node
multipliers modulate the instantaneous rate before integration.
Use `--tile-rate-variation` to give each selected SCA/tile its own log-rate
offset. Neighboring selected tiles receive a Gaussian smoothness prior based
on their row/column coordinates; `--tile-rate-prior-sigma-log` and
`--tile-rate-neighbor-sigma-log` control the two log-rate scales.
The default prior/start is an equal-index-ratio quarter-wave dielectric stack
at 1050 nm with `k=1e-4`. Use `--ar-prior-from-map-start` to recenter that prior
when selecting a different physical start.

The exact-node grid uses about 3,294 wavelength nodes for the current public
SED tables. Use `--wavelength-step-nm 0.5` to test a uniform sub-nanometer
grid; keep `--wavelength-stride 1` for that comparison.

CPU chains run sequentially by default. `--vectorized-chains` runs both chains
as one batched JAX computation without creating virtual CPU devices and is the
preferred faster option for the full fit. Use `--parallel-chains` only for
smaller fits or after confirming that host-device contention is acceptable.

Use `--map-steps N` to run `N` full-model AutoDelta optimization steps before
NUTS. MAP is only an initializer; posterior sampling and all model parameters
remain unchanged. `--map-learning-rate` controls its Adam step size; the
lowest checkpointed MAP objective is retained if a long Adam run later drifts.
Use `--map-only` with `--map-steps` to archive and plot the deterministic MAP
solution without starting NUTS; this is useful for fast basin comparisons.
The deterministic MAP optimizer defaults to bounded L-BFGS-B; use
`--map-optimizer adam` only for the legacy AutoDelta initializer path, and use
`--map-tolerance` to set the L-BFGS-B stopping tolerance.
Use `--map-trajectory-rate-nm-day` for a deliberately dispersed MAP start when
checking for separate trajectory basins; it does not alter the posterior prior.
For a staged start, `--map-init-max-tiles 1` runs those MAP steps first on one
tile per SCA, transfers the global and SCA-level sites, and then refines the
full selected-tile model. Full-tile offsets start at zero rather than copying
the single representative tile.
For a physically motivated optical start, `--map-design-wavelength-nm 1050`
uses quarter-wave optical thickness in each layer and equal log-index steps
between air and HgCdTe only during MAP initialization; it does not constrain
the sampled posterior. For the analytic three-layer equal-ratio start, add
`--map-middle-layer-eighth-wave`; this uses lambda/4, lambda/8, lambda/4 optical
thicknesses from air toward HgCdTe only for the MAP start.

For a legacy single-SCA fit:

```bash
python fit_wfi_tmm_numpyro.py \
  --single-sca \
  --tile-statistics wfi01_tile_log_statistics_512.csv \
  --max-tiles 1 \
  --map-steps 1000 \
  --dense-mass \
  --output-dir wfi01_tmm_numpyro_tile_pilot
```

### 2a. Run the two- and three-layer fits on Slurm

The repository includes a one-node Slurm job that runs the two-layer and
three-layer all-SCA fits concurrently, plus an `rsync`/`ssh` helper that pushes
the source and statistics before submitting it. See
[`docs/cluster_slurm.md`](docs/cluster_slurm.md):

```bash
export CLUSTER='user@login.example.edu'
export REMOTE_ROOT='/path/on/cluster/wfi-ice-analysis'
bash cluster/push_and_submit.sh
```

The optional `--np-modulation` flag adds an entropy- and smoothness-regularized
repeat-node modulation of the instantaneous rate. It is a diagnostic option,
not the default model.

### 3. Plot posterior products

```bash
python plot_wfi_tmm_numpyro.py \
  --output-dir wfi_all_sca_tmm_numpyro_tile_production \
  --tile-statistics wfi_tile_log_statistics_512
```

The plotter writes the shared trajectory, posterior corners, per-SCA AR maps,
tile fit diagnostics, and observed-versus-modeled residual plots.

## Outputs

The fit directory contains:

- `posterior_samples.npz`: NumPyro posterior arrays, tile IDs, SCA names, and
  the selected AR layer count, archived response/statistics settings, and a
  sufficient-statistics fingerprint;
- `posterior_samples.npz` also records the public QE column selected for each SCA;
- `tile_ar_posterior.csv`: 16th/50th/84th percentile AR parameters per SCA/tile;
- `sca_ar_posterior.csv`: SCA-level AR population summaries for all-SCA fits;
- `np_node_posterior.csv`: repeat-node rate-multiplier and instantaneous-rate summaries when NP modulation is enabled;
- `tile_rate_posterior.csv`: per-SCA/tile rate offsets and instantaneous-rate summaries when tile-rate variation is enabled;
- `numpyro_summary.md`: settings and posterior percentile summary;
- PNG diagnostics generated by `plot_wfi_tmm_numpyro.py`.

The plotter uses the archived statistics path, SED mode, wavelength settings,
tile selection, and observation fingerprint by default. Pass an explicit
override only when intentionally plotting a compatible posterior subset.

## Code Map

- `build_wfi01_tile_statistics.py`: tile sufficient-statistic calculation;
- `build_wfi_tile_statistics_stream.py`: resumable MAST downloader;
- `fit_wfi_tmm_numpyro.py`: single- and all-SCA TMM model and output writer;
- `plot_wfi_tmm_numpyro.py`: posterior prediction and plotting;
- `analyze_rcs_soc_ice_model.py`: shared trajectory and sufficient-statistic utilities;
- `fit_wfi01_joint_complex_q_soc.py`: response integration and effective-response diagnostics;
- `fit_wfi01_effective_boundary_numpyro.py`: separate effective-boundary diagnostic;
- `test_*.py`: synthetic and local fixture tests.

## Verification

Run the tests from the project root:

```bash
python -m pytest -q
python -m py_compile fit_wfi_tmm_numpyro.py plot_wfi_tmm_numpyro.py
```

The tests do not require raw MAST products. The MAST streaming tests use
injected image fetchers and the tracked index/statistics fixtures.

## Caveats

- The all-SCA implementation selects `SCU 1` through `SCU 18` from the public
  QE table for `WFI01` through `WFI18`, respectively. The official SCU-to-SCA
  serial mapping identifies WFI01/SCU 1 with SCA serial 22081.
- The production model generates its wavelength grid from public QE support;
  the external SCA1 response table is not used. It uses nominal fixed HgCdTe
  constants for every SCA because detector-specific constants are not currently
  available.
- The one-, two-, and three-layer models use the same broad per-layer AR
  bounds and hierarchical priors. They are effective dielectric-stack models,
  not claims about the proprietary physical coating recipe.
- The conductive-interlayer mode is a sensitivity model. It uses a fixed
  `epsilon_inf=1` and a reference wavelength of 1600 nm; the fitted Drude
  parameters are not detector-material measurements without external priors.
- The QE RBF mode is likewise a calibration sensitivity model. Its correction
  coefficients are global and prior-regularized; grey QE changes cancel under
  the per-LED reference normalization.
- The exact external effective-boundary equation is not available, so the
  effective-boundary scripts remain sensitivity diagnostics.
- Short smoke chains are useful for exercising code paths but are not
  scientifically interpretable posterior fits.
- Generated outputs and raw products are ignored by Git; retain important
  results separately if they need to be archived.
