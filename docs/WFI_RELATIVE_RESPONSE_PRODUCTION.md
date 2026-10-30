# WFI Cached Relative-Response Production

The cached branch is the definitive fast production path when a pinned
relative-response cache is available. It uses the current corrected statistics
and the current production ice/rate fit, while replacing repeated wavelength
and TMM evaluation with precomputed six-LED response curves.

The ordinary `run_wfi_production.py` path remains the full-TMM production fit.
The cached path does not replace the full-TMM `fits/` output tree; it writes
under the detector's `relative_response/` subtree.

## Fit Contract

The cache contains response curves versus ice thickness on a `0`--`400 nm`
grid at `0.25 nm` spacing. The fit evaluates those curves with interpolation
for every current thickness evaluation.

The cache source fit contributes only:

- fixed AR stack parameters;
- fixed HgCdTe n/k scales;
- fixed ice-index and AR/HgCdTe spectral modulation parameters;
- fixed normal incidence.

Every final cached tile fit performs an in-process scalar-rate warm-up for
numeric initialization. That warm-up fits the reset thicknesses, reset
offsets, LED offsets, and one constant accumulation rate, starting from a
robust detector-level median of the source-fit node rates. The fully
non-parametric production MAP then uses that state; the source fit's
time-dependent rate trajectory is never imported as a prior or entropy
reference. Production archives record `constant_rate_map_warmup=1` and the
scalar warm-up diagnostics.

The validated 20261030 run additionally used a separate constant-rate warm-up
for each detector. Its fitted per-tile third-decon thicknesses are supplied to
the final fit through `--warmup-prior-root`, with a lognormal prior sigma of
`0.10` and `--thickness-prior-stages third-decon`. The separate warm-up was
started with a `40` nm third-decon thickness; that start value belongs only to
the warm-up command and is not passed to the final fit. The final cached fit
still uses the production rate prior centered at `12 nm/day` with log sigma
`1.5`, zero entropy weight, prior-width coordinate scaling, `10000` MAP
evaluations, and the production `1e-9` function and step tolerances.

The authoritative production Decon 3 boundary is relative day
`38.79784510416666`, established in the 20261023 production fit. Pass
`--third-decon-day 38.79784510416666`. This adds independent third-decon
initial thickness and post-third-decon grey-offset sites in the cached fit;
the boundary is explicit and is never inferred from the first later
observation. The count-space `--decon-time` is a separate free-step
correction.

When both `--third-decon-day` and `--warmup-prior-root` are supplied, the
relative-response runner defaults to a per-tile warmup-centered prior for the
third-decon initial thickness: log sigma `0.10` and `third-decon` stages. The
warmup root must contain the independent constant-rate archives under
`fits/tile_###/constant_rate_map.npz`. The boundary remains explicit and is
never inferred by the runner. This mode requires independent one-tile fits
(`--batch-size 1`, without `--joint-batch`).
The normal `run_wfi_production.py` wrapper forwards these options when
`--response-cache-root` selects the relative-response branch.

## Pinned Cache

The validated pinned cache used for the 20261030 corrected-statistics run is:

```text
20261020/corrected_statistics_4h_thinned/production/relative_response_cache/
  tile_size_256/sensitivity_hgcdte_n_sigma_0p25/wfiNN_400nm/
```

Each detector directory contains 256 provenance-checked tile caches. The cache
records the source fit and source-statistics paths, hashes, statistics
fingerprint, optical-context fingerprint, and source model fingerprint. Do not
rebuild it for an ordinary current-statistics update. Rebuild only after full
reprocessing, a reduction improvement, or an intentional optical-model change.

## One Detector

Run the normal production entrypoint with the detector-specific cache and
`--allow-external-cache-source`:

```bash
.venv/bin/python -u run_wfi_production.py \
  --statistics-dir 20261030/corrected_statistics_4h_thinned \
  --detector WFI01 \
  --tile-size 256 \
  --response-cache-root 20261020/corrected_statistics_4h_thinned/production/relative_response_cache/tile_size_256/sensitivity_hgcdte_n_sigma_0p25/wfi01_400nm \
  --third-decon-day 38.79784510416666 \
  --warmup-prior-root 20261030/corrected_statistics_4h_thinned/production/wfi01/relative_response_constant_rate_warmup_start40 \
  --thickness-prior-log-sigma 0.1 \
  --thickness-prior-stages third-decon \
  --fit-led-log-offsets \
  --led-log-offset-prior-sigma-log 0.01 \
  --batch-size 1 \
  --allow-external-cache-source \
  --workers 2
```

The output is written below:

```text
20261030/corrected_statistics_4h_thinned/production/wfi01/relative_response/
```

The cache runner validates the pinned source provenance while fitting the new
statistics. Without `--allow-external-cache-source`, the source fit and source
statistics must belong to the current statistics run.

## All Detectors

Use one detector-specific cache per invocation:

```bash
for detector in WFI01 WFI02 WFI03 WFI04 WFI05 WFI06 WFI07 WFI08 WFI09 WFI10 WFI11 WFI12 WFI13 WFI14 WFI15 WFI16 WFI17 WFI18; do \
  lower=$(printf '%s' "$detector" | tr '[:upper:]' '[:lower:]'); \
  .venv/bin/python -u run_wfi_relative_response_constant_rate_warmup.py \
    --statistics-dir 20261030/corrected_statistics_4h_thinned \
    --detector "$detector" \
    --tile-size 256 \
    --response-cache-root "20261020/corrected_statistics_4h_thinned/production/relative_response_cache/tile_size_256/sensitivity_hgcdte_n_sigma_0p25/${lower}_400nm" \
    --output-root "20261030/corrected_statistics_4h_thinned/production/${lower}/relative_response_constant_rate_warmup_start40" \
    --third-decon-day 38.79784510416666 \
    --third-decon-initial-thickness-start-nm 40 \
    --fit-led-log-offsets \
    --led-log-offset-prior-sigma-log 0.01 \
    --allow-external-cache-source \
    --workers 2; \
  .venv/bin/python -u run_wfi_production.py \
    --statistics-dir 20261030/corrected_statistics_4h_thinned \
    --detector "$detector" \
    --tile-size 256 \
    --response-cache-root "20261020/corrected_statistics_4h_thinned/production/relative_response_cache/tile_size_256/sensitivity_hgcdte_n_sigma_0p25/${lower}_400nm" \
    --third-decon-day 38.79784510416666 \
    --warmup-prior-root "20261030/corrected_statistics_4h_thinned/production/${lower}/relative_response_constant_rate_warmup_start40" \
    --thickness-prior-log-sigma 0.1 \
    --thickness-prior-stages third-decon \
    --fit-led-log-offsets \
    --led-log-offset-prior-sigma-log 0.01 \
    --batch-size 1 \
    --allow-external-cache-source \
    --workers 2; \
done
```

The direct `run_wfi_relative_response_production.py` entrypoint is equivalent;
the normal production entrypoint is preferred so the output contract and cache
dispatch remain centralized.

## Diagnostics

Generate cached-branch focal-plane maps, all-detector trajectories, and the
last-seven-day trajectory view from the same statistics directory. The focal-plane
command also always writes `wfi_full_focal_plane_initial_thicknesses.png`, a
three-panel common-scale mosaic showing the fitted initial thickness before
Decon2, after Decon2, and after the third decon:

```bash
.venv/bin/python -u plot_wfi_full_focal_plane_rates.py \
  --root 20261030/corrected_statistics_4h_thinned \
  --tile-size 256 \
  --fit-subdir relative_response \
  --output 20261030/corrected_statistics_4h_thinned/production/relative_response_plots/wfi_full_focal_plane_rate_maps.png \
  --aom-rate-jump-output 20261030/corrected_statistics_4h_thinned/production/relative_response_plots/wfi_full_focal_plane_aom_rate_jump_percent.png

.venv/bin/python -u plot_wfi_all_detector_trajectories.py \
  --root 20261030/corrected_statistics_4h_thinned \
  --tile-size 256 \
  --fit-subdir relative_response \
  --reset-day 12.907397928240743 \
  --break-day 38.79784510416666 \
  --output 20261030/corrected_statistics_4h_thinned/production/relative_response_plots/wfi_all_detector_ice_rate_trajectories.png

.venv/bin/python -u plot_wfi_all_detector_trajectories.py \
  --root 20261030/corrected_statistics_4h_thinned \
  --tile-size 256 \
  --fit-subdir relative_response \
  --reset-day 12.907397928240743 \
  --break-day 38.79784510416666 \
  --last-days 7 \
  --output 20261030/corrected_statistics_4h_thinned/production/relative_response_plots/wfi_all_detector_ice_rate_trajectories_last_7d.png
```

The focal-plane command also writes
`wfi_full_focal_plane_current_ice_thickness.png` from the latest fitted
`tile_ice_thickness` value in each cached tile archive.

The optional AOM output is a two-panel map comparing the latest post-bump rate
with the last pre-bump rate for each tile: absolute change in `nm/day` on the
left and percentage change on the right. The color limits use the robust 99th
percentile independently for each panel. Corrected-data neighborhood plots use
scatter points for each LED and tile; data samples are never connected by a
line. For the investigated WFI06 tile 98 neighborhood:

```bash
.venv/bin/python -u plot_wfi_corrected_tile_neighborhood.py \
  --root 20261030/corrected_statistics_4h_thinned \
  --detector WFI06 \
  --tile-id 98 \
  --tile-size 256 \
  --output 20261030/corrected_statistics_4h_thinned/production/relative_response_plots/wfi06_tile098_neighborhood_corrected_data.png
```

The portable wavelength-response bundle is not accepted by this branch. It
contains pre-integration wavelength responses rather than the six
LED-integrated curves required by the cached evaluator.
