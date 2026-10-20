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

The current cached fit independently fits the current tile's initial and reset
thicknesses, post-reset offset, and fully non-parametric rate nodes. Its rate
nodes start from a robust detector-level median of the source-fit node rates,
expanded as a constant vector. This is only a numerical warm start. The actual
production rate prior remains centered at `12 nm/day` with log sigma `1.5`, with
zero entropy weight, prior-width coordinate scaling, `10000` MAP evaluations,
and the production `1e-9` function and step tolerances. The source fit's
time-dependent rate trajectory is never imported as a prior or entropy
reference.

## Pinned Cache

The validated pinned cache currently used for the 20261019 corrected-statistics
rerun is:

```text
20261018/corrected_statistics_4h_thinned/production/relative_response_cache/
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
  --statistics-dir 20261019/corrected_statistics_4h_thinned \
  --detector WFI01 \
  --tile-size 256 \
  --response-cache-root 20261018/corrected_statistics_4h_thinned/production/relative_response_cache/tile_size_256/sensitivity_hgcdte_n_sigma_0p25/wfi01_400nm \
  --allow-external-cache-source \
  --workers 2
```

The output is written below:

```text
20261019/corrected_statistics_4h_thinned/production/wfi01/relative_response/
```

The cache runner validates the pinned source provenance while fitting the new
statistics. Without `--allow-external-cache-source`, the source fit and source
statistics must belong to the current statistics run.

## All Detectors

Use one detector-specific cache per invocation:

```bash
for detector in WFI01 WFI02 WFI03 WFI04 WFI05 WFI06 WFI07 WFI08 WFI09 WFI10 WFI11 WFI12 WFI13 WFI14 WFI15 WFI16 WFI17 WFI18; do \
  lower=$(printf '%s' "$detector" | tr '[:upper:]' '[:lower:]'); \
  .venv/bin/python -u run_wfi_production.py \
    --statistics-dir 20261019/corrected_statistics_4h_thinned \
    --detector "$detector" \
    --tile-size 256 \
    --response-cache-root "20261018/corrected_statistics_4h_thinned/production/relative_response_cache/tile_size_256/sensitivity_hgcdte_n_sigma_0p25/${lower}_400nm" \
    --allow-external-cache-source \
    --workers 2; \
done
```

The direct `run_wfi_relative_response_production.py` entrypoint is equivalent;
the normal production entrypoint is preferred so the output contract and cache
dispatch remain centralized.

## Diagnostics

Generate cached-branch focal-plane maps, all-detector trajectories, and the
last-seven-day trajectory view from the same statistics directory:

```bash
.venv/bin/python -u plot_wfi_full_focal_plane_rates.py \
  --root 20261019/corrected_statistics_4h_thinned \
  --tile-size 256 \
  --fit-subdir relative_response \
  --output 20261019/corrected_statistics_4h_thinned/production/relative_response_plots/wfi_full_focal_plane_rate_maps.png

.venv/bin/python -u plot_wfi_all_detector_trajectories.py \
  --root 20261019/corrected_statistics_4h_thinned \
  --tile-size 256 \
  --fit-subdir relative_response \
  --reset-day 12.907397928240743 \
  --output 20261019/corrected_statistics_4h_thinned/production/relative_response_plots/wfi_all_detector_ice_rate_trajectories.png

.venv/bin/python -u plot_wfi_all_detector_trajectories.py \
  --root 20261019/corrected_statistics_4h_thinned \
  --tile-size 256 \
  --fit-subdir relative_response \
  --reset-day 12.907397928240743 \
  --last-days 7 \
  --output 20261019/corrected_statistics_4h_thinned/production/relative_response_plots/wfi_all_detector_ice_rate_trajectories_last_7d.png
```

The focal-plane command also writes
`wfi_full_focal_plane_current_ice_thickness.png` from the latest fitted
`tile_ice_thickness` value in each cached tile archive.

The portable wavelength-response bundle is not accepted by this branch. It
contains pre-integration wavelength responses rather than the six
LED-integrated curves required by the cached evaluator.
