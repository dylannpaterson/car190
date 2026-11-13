# Joint CAR190/Amber Fit to Eric-Reader H5

This document describes the current all-detector path from the reader-compatible
joint fit through the portable modulated wavelength-response bundle to one
Eric-v3-reader-compatible HDF5 file. It covers the 20261111 corrected-statistics
run and the fitted experiment
`car190_amber_joint_2day_reader_compatible_current_gray_v2`.

The path-only pipeline frontends now place the single distributable H5 at
`production/wavelength_response_bundle/final/joint_modulated_v2.h5`. Required
build-stage H5s are isolated under `production/wavelength_response_bundle/intermediates/h5/`.
The older directory and filenames below identify the 20261111 golden reference,
not the new pipeline output layout.

The pipeline fits each detector independently (WFI01–WFI18), then combines the
detector results into one focal-plane H5. Fitting retains each tile's full
independent 2-day ice history. The H5's unchanged Eric-v3 deposition interface
supports a rank-one spatial-rate representation, so the history is factorized
only during export. The history-reference file records both the exact fit nodes
and the rank-one export error.

The 20261111 golden-reference H5 is:

```text
20261111/corrected_statistics_4h_thinned/production/wavelength_response_bundle/unchanged_reader_port/joint_modulated_all_detectors_current_gray_v2.h5
```

The stages below intentionally write separate, provenance-bearing products.
Most output paths are write-once. Use `--resume` only with a matching
interrupted bundle build, and use `--force` only when intentionally replacing a
specific mode, prototype, or packaged H5. The validated fit directories
described below already exist; choose a fresh fit name/output path for a new
run.

## Inputs and fit settings

Run commands from the repository root. The fit consumes the 20261111 4-hour,
256-pixel corrected statistics and the per-detector six-LED optical caches from
the complete 20261109 TMM run. The caches are selected from each detector's
`relative_response_mode.json`; TMM optical parameters are fixed. The joint fit
uses the detector-specific Roman effective-area file, the H5's original LED
quadrature, the CAR190 production Gaussian weighting, and Amber CAR201's
Student-t likelihood with fixed `nu=4`.

For the validated `current_gray_v2` fits, WFI01, WFI02, and WFI04–WFI18 were
initialized from their detector's
`car190_amber_joint_2day_reciprocal_maxent` fit. WFI03 was initialized from its
earlier `car190_amber_joint_2day_reader_compatible_current_gray` fit. The
initial fits supply spectral coefficients, roll offsets, extra scale, and
starting rate controls; the current fit re-optimizes its own independent tile
histories. Its fixed tile-gray terms are loaded from that detector's current
`relative_response/fits/tile_###/posterior_samples.npz` files.

The fit writes one output directory per detector, including
`joint_fit.npz`, `joint_weekly_ice_fit.npz`, `training_inputs.npz`, and
`summary.json`. Require `optimizer_success: true`,
`response_convention: "eric-reader"`, 2-day rate controls, and
`ice_histories_refitted: true` in every summary before building the bundle.

## 1. Fit each detector sequentially

The archived all-detector runner `archive/exploratory/legacy_root_scripts/run_joint_car190_amber_inverse_wavelength_maxent_all.py`
is for the original non-reader-compatible experiment. For this H5 workflow,
invoke the single-detector runner with `--response-convention eric-reader` and
pass one explicit detector at a time:

```bash
statistics_dir=20261111/corrected_statistics_4h_thinned
production="$statistics_dir/production"
bundle_dir="$production/wavelength_response_bundle"
fit_name=car190_amber_joint_2day_reader_compatible_current_gray_v2
detectors=(WFI01 WFI02 WFI03 WFI04 WFI05 WFI06 WFI07 WFI08 WFI09 WFI10 WFI11 WFI12 WFI13 WFI14 WFI15 WFI16 WFI17 WFI18)

for detector in "${detectors[@]}"; do
  lower=${detector,,}
  seed_name=car190_amber_joint_2day_reciprocal_maxent
  if [[ "$detector" == WFI03 ]]; then
    seed_name=car190_amber_joint_2day_reader_compatible_current_gray
  fi
  seed_dir="$production/$lower/relative_response/experiments/$seed_name"
  output_dir="$production/$lower/relative_response/experiments/$fit_name"
  log="$bundle_dir/reader_compatible_fit_${detector}_current_gray_v2.log"

  python -u wfi_pipeline/run_joint_car190_amber_inverse_wavelength_maxent.py \
    --detector "$detector" \
    --statistics-dir "$statistics_dir" \
    --response-convention eric-reader \
    --initial-fit-dir "$seed_dir" \
    --output-dir "$output_dir" \
    --maxiter 700 \
    --ftol 1e-7 \
    --gtol 1e-4 \
    >"$log" 2>&1 || { printf 'Fit failed for %s; see %s\n' "$detector" "$log" >&2; exit 1; }
done
```

This is a sequential loop over explicit detectors; do not use `--all-detectors`,
`--joint-batch`, or a detector-neighbor fit. To continue an interrupted fit,
use `--resume` for that detector and its existing checkpoint. For a new rerun,
select a fresh `--output-dir` rather than overwriting the validated fits.

## 2. Build the all-detector modulated bundle

The existing base bundle supplies the shared wavelength/thickness grids and
tile geometry. The builder applies each detector's fitted complex modulation to
the cached clean TMM reflection, recomputes the ice response, imports the
current fit's 2-day thickness/rate histories, and carries the current fitted
reset/decon gray offsets. The modulation is applied before the ice layer; the
portable response remains independent of downstream SED/filter/QE weights.

```bash
base_bundle="$bundle_dir/wfi_all_2nm_rank16.npz"
modulated_bundle="$bundle_dir/wfi_all_2nm_rank16_car190_amber_joint_2day_reader_compatible_current_gray_v2.npz"
detector_args=()
for detector in "${detectors[@]}"; do detector_args+=(--detector "$detector"); done

python -u wfi_pipeline/build_modulated_wfi_wavelength_response_bundle.py \
  --base-bundle "$base_bundle" \
  --fit-root "$production" \
  --modulation-root "$production" \
  --modulation-fit joint-car190-amber-2day \
  --joint-fit-name "$fit_name" \
  --output "$modulated_bundle" \
  --checkpoint-interval 128 \
  "${detector_args[@]}"
```

The detector selection must match the base bundle exactly (4,608 tiles, 256
per detector). If interrupted, repeat the same command with `--resume`; the
builder checks the checkpoint manifest before using saved blocks.

## 3. Build global TMM reflectance modes

The mode builder evaluates each detector/tile's pinned TMM clean reflection,
applies the fitted detector-global complex modulation, then performs one global
complex SVD across all 4,608 tiles. The rank-64 mode product is an intermediate
reflection product, not the final time-dependent Eric H5.

```bash
reflectance_modes="$bundle_dir/tmm_joint_modulated_reflectance_modes_current_gray_v2_all_detectors.h5"

python -u wfi_pipeline/build_tmm_complex_reflectance_modes.py \
  --bundle "$modulated_bundle" \
  --variant joint_modulated \
  --mode-count 64 \
  --wavelength-min-nm 475 \
  --wavelength-max-nm 2350 \
  --wavelength-step-nm 1 \
  --output "$reflectance_modes" \
  "${detector_args[@]}"
```

The reflectance mode H5 records the source bundle and per-detector cache roots.
The joint complex multiplier is baked into its training curves. The assembler
later transforms the air-side TMM reflection to Eric's ice-side convention.

## 4. Export fitted histories and assemble the prototype

The history exporter saves the full independent fitted histories as a reference
and also computes the rank-one deposition candidate supported by the unchanged
v3 core. Inspect the rank-one metrics before interpreting the portable H5: for
the current all-detector export, the p50/p95/maximum of tilewise maximum
absolute thickness error is approximately `3.14/7.57/25.23 nm`. The full fitted
histories remain in the joint-fit archives and the history-reference H5; the
rank-16 deposition extension in that reference is diagnostic and is not read by
Eric's current core.

```bash
history_reference="$bundle_dir/wfi_joint_2day_ice_history_reference_current_gray_v2_all_detectors.h5"
prototype="$bundle_dir/ice_response_model_tmm_joint_iceside_current_gray_v2_all_detectors_prototype.h5"
base_v3="$bundle_dir/eric_v3_reference/ice_response_model_v3.h5"

python -u wfi_pipeline/export_joint_ice_history_reference.py \
  --bundle "$modulated_bundle" \
  --output "$history_reference"

python -u wfi_pipeline/assemble_eric_v3_ice_response_candidate.py \
  --base-v3-h5 "$base_v3" \
  --reflectance-modes "$reflectance_modes" \
  --ice-history-reference "$history_reference" \
  --output "$prototype"
```

The prototype assembler expands each 256×256-pixel fit tile across the
corresponding 16×16 cells in Eric's 16-pixel superpixel grid. It fills the
existing `instrument/gray` coefficients from the fitted tile offsets and uses
the fitted reset epochs. The prototype contains extra audit/photometry groups;
do not distribute it as the unchanged-reader product.

## 5. Package the final unchanged-contract H5

The packager starts from Eric's original H5 and replaces only existing dataset
paths. It omits the prototype-only groups and does not export AR parameters,
custom band weights, or modifications to Eric's reader.

```bash
final_h5="$bundle_dir/unchanged_reader_port/joint_modulated_all_detectors_current_gray_v2.h5"

python -u wfi_pipeline/package_joint_h5_unchanged_contract.py \
  --source "$prototype" \
  --output "$final_h5"
```

The packager checks the final group/dataset inventory against the original
H5 (paths, ranks, and dtype kinds), loads the file through the unchanged reader,
checks finite responses for all 18 detectors, and verifies public-reader gray
factors against the fitted tile offsets. Its report is written next to the H5
as `joint_modulated_all_detectors_current_gray_v2.validation.json`.
The fitted gray boundaries are the fixed reset days `12.907397928240743` and
`38.79784510416666` from the joint fit's time origin.

## 6. Compare correction performance by filter and epoch

Run Eric's original v3 model and the new H5 on the same matched CAR201 cohort.
The comparison uses the unchanged `IceResponseModel` reader, detector-specific
effective areas, exact observation times/positions, and applies the saved Amber
roll offsets identically to each model:

```bash
comparison_dir="$bundle_dir/unchanged_reader_port/current_gray_v2_car201_all_detectors"

python -u wfi_pipeline/compare_eric_v3_amber_car201.py \
  --statistics-dir "$statistics_dir" \
  --joint-fit-name "relative_response/experiments/$fit_name" \
  --include-fit-reference \
  --output-dir "$comparison_dir" \
  --joint-modulated-model "$final_h5" \
  "${detector_args[@]}"

python -u wfi_pipeline/plot_joint_h5_ice_correction_epoch_filter.py \
  --predictions "$comparison_dir/amber_eric_v3_vs_joint_modulated_measurement_predictions.csv" \
  --output-dir "$comparison_dir"
```

The plot is `ice_correction_by_filter_epoch.png` (also PDF). Each panel is one
filter; points are pooled across WFI01–WFI18 at each ending-visit UTC date. Bars
show the 16th–84th percentile of residual relative-flux changes, with the same
±0.5% guide band. The adjacent CSVs contain per-epoch and per-filter metrics.

For CAR190, compare using the fit's saved per-tile/LED reference baseline and
gray terms. A raw H5 log-change versus observed-log-change statistic omits those
source baseline corrections and is not the optimizer's CAR190 objective. With
the reference correction applied, the current full-history calculation is
within about 1.9% aggregate chi-square of the fit. The packaged H5's rank-one
history approximation adds further residual; the all-detector CAR190 audit
records raw, baseline-aligned rank-one, and baseline-aligned full-history results
in `current_gray_v2_car201_all_detectors/car190_offset_aligned_h5_validation_summary.json`.

## Optional S3 upload

The uploader beside the final H5 targets the H5 filename under the
`rges-pit/ice_model` prefix by default. Preview before transferring:

```bash
python "$bundle_dir/unchanged_reader_port/upload_joint_modulated_all_detectors_s3.py" --dry-run
```

Omit `--dry-run` to upload the H5 with the AWS CLI's normal credential chain.
The script uploads only that H5 object; it does not sync or delete other S3
objects.
