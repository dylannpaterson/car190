# WFI Nexus Pipeline User Guide

This guide covers the Nexus data reduction and the two local fit routes. The
relative-response route is the normal production fit; full TMM is an alternative
fit route. Run the reduction first, then choose **one** fit route for a run.

## 1. Reduce and correct the RCS data

Run the maintained reducer from the CAR190 directory on Nexus. For the
2026-11-17 run, the downstream input root is
`/teams/romcom/data/dpaterson/CAR190/CAR190_reduction/20261117/`.

```bash
cd /teams/romcom/data/dpaterson/CAR190
python run_wfi_reduction_pipeline.py --tile-size 256 --workers 18
```

The reduction applies the photodiode, side-swap, temperature, ACM, and
decontamination count-space corrections. Its dated run root contains the
handoff statistics and provenance:

```text
CAR190_reduction/20261117/
  raw_statistics/             # raw reference-relative sufficient statistics
  statistics/                 # photodiode/side-swap corrected statistics
  corrected_statistics/       # corrected tile sufficient statistics
  corrections/                # correction tables and provenance
  telemetry/                  # PD2 and WTSM00 telemetry used
  diagnostics/
  pipeline.log
  pipeline_manifest.json      # check for a complete successful run
```

Use `corrected_statistics/` as the model input. These corrections are already
applied; do not apply their correction tables again during fitting.

## 2. Prepare the modeling environment

Install the fit dependencies in the same Python environment used for the model
commands. Do this once before fitting:

```bash
cd /teams/romcom/data/dpaterson/CAR190/modeling
python -m pip install jax numpyro extinction
```

## 3. Run the relative-response fit (default)

From `modeling/`, run the cached relative-response production pipeline:

```bash
python run_wfi_relative_response_pipeline.py --input-path ../CAR190_reduction/20261117/ --output-path 20261117/
```

The pipeline thins unthinned corrected statistics to complete four-hour
observations at 256-pixel tiles under its output root, then performs the fit and
builds the downstream bundle and H5 products.

## 4. Full-TMM fit (alternative)

Use this command instead when a full-TMM fit is required. Give it a separate
output root so it cannot mix products with a relative-response run:

```bash
python run_wfi_full_tmm_pipeline.py --input-path ../CAR190_reduction/20261117/ --output-path 20261117_full_tmm/
```

This route is more computationally expensive. Both fit commands take the Nexus
reduction root as input; the selected frontend determines the model route.

## Reset and decontamination events

The model fit frontends read the shared configuration at
`wfi_pipeline/reset_events.yaml` in the modeling checkout. Event `day` values
are elapsed days from `2026-09-12T01:25:50.714 UTC`, not UTC day numbers, and
must be strictly increasing. The first entry is `second_decon` because the
model timeline starts after Decon1; the second entry is `third_decon`. To move
the configured third reset, edit the existing `third_decon.day`; the root
pipeline commands take the date from this YAML.

Append one entry per additional decontamination, with a unique name and a
measured elapsed day later than the previous event. For example:

```yaml
  - name: fourth_decon
    day: 65.0  # illustrative only; replace with the measured elapsed day
    thickness_prior:
      median_nm: 10
      log_sigma: 1.5
    h5_prior:
      center: base_h5
      log_sigma: 1.5
    # Optional, cached-relative-response initialization only:
    # relative_response_warmup:
    #   start_nm: 40
    #   log_sigma: 0.1
```

The fields serve different consumers:

- `thickness_prior` sets the lognormal starting-ice prior for the production
  TMM/relative-response fit at that reset.
- `h5_prior` is for the separate fast H5 history-update utility. Its `center`
  is either `configured_median` (use the configured thickness median) or
  `base_h5` (use the starting H5 history); its `log_sigma` sets that update's
  prior width.
- `relative_response_warmup` is optional and only controls the constant-rate
  initialization used by the cached relative-response route. It is not another
  reset date or a prior for full TMM. It has `start_nm` and `log_sigma` fields;
  the current Decon3 values are `40` nm and `0.1`, specific to the validated
  production setup. Do not copy them to a new event without an intentional
  choice. The cached route reads each event's own warm-up values; events without
  this block use their `thickness_prior` for the final fit. This is why the
  warm-up settings differ from the broader `thickness_prior` used by the final
  fit.

These are not competing decon-date options: `day` sets the physical ice-history
boundary, while the three prior blocks configure different fit stages. The
production fit, joint-refitted history, diagnostic plots, response bundles, and
H5 gray-period arrays use the complete ordered event list, with variable-length
reset-day, reset-name, and per-event-offset arrays. There is no fixed two-/three-
reset limit in the current pipeline. Legacy fields named for Decon2 and Decon3
are compatibility aliases; `reset_event_days`, `reset_event_names`, and the
corresponding per-event arrays carry every configured reset.

The reducer's `--decon-time ISO_TIMESTAMP` is separate: it marks a count-space
step correction in the pixel statistics. The YAML `day` marks the physical
ice-history reset. They can refer to the same observed event, but use different
time formats and affect different parts of the model. Likewise, the lower-level
`--third-decon-day` option overrides the existing third-reset day for direct
stage runs; it does not add a fourth event. The two root fit frontends in this
guide use the YAML instead. Use a fresh `--output-path` after any
reset-configuration change; the run manifest records the YAML hash and prevents
reusing an output root with different settings.

## Main fit outputs

Paths below are relative to the `--output-path` used for that fit (for example,
`modeling/20261117/`):

```text
<output-root>/
  pipeline_manifest.json
  logs/                                      # per-stage and per-detector logs
  corrected_statistics_4h_thinned/           # made only when input needs thinning
  production/
    <detector>/<relative_response|full_tmm>/
      fits/tile_000/                          # per-tile fit archives and diagnostics
      experiments/car190_amber_joint_2day_reader_compatible_current_gray_v2/
    caches/six_led/full_tmm/<detector>/       # generated by full TMM
    caches/spectral_response/<route>/<detector>/
    wavelength_response_bundle/
      wfi_all_2nm_rank16.npz                  # base response bundle
      wfi_all_2nm_rank16_modulated.npz        # joint-modulated response bundle
      final/
        joint_modulated_v2.h5                # final model consumed by the reader
        joint_modulated_v2.validation.json
    plots/<route>/                            # joint-modulated trend and fit plots
```

`<detector>` is `wfi01` through `wfi18`; `<route>` is `relative_response` or
`full_tmm`. Check `pipeline_manifest.json` for completion, stage logs, input
fingerprints, and the recorded paths and hashes of the main outputs.
The relative-response route reuses its pinned six-LED cache rather than copying
that cache under the run root.

## Publish the paired response bundle to S3

The portable response release consists of the two stable all-detector archives:
`wfi_all_2nm_rank16.npz` and `wfi_all_2nm_rank16_modulated.npz`. Prepare the
paired export and copy the complete staged directory to Nexus as described in
[`WFI_MODULATED_WAVELENGTH_RESPONSE_BUNDLE.md`](WFI_MODULATED_WAVELENGTH_RESPONSE_BUNDLE.md).
The staged directory also contains the bundle reader, uploader, and checksum
manifest. Keep dated build archives and provenance in their local run directories;
publish the two archives using their stable filenames.

On Nexus, change to the staged directory (the path below is the directory used
for the 2026-11-18 upload; substitute the current release directory for a later
run):

```bash
cd /teams/romcom/data/dpaterson/CAR190/completed_model/20261118
```

First run a dry-run with both required destination arguments:

```bash
python upload_wfi_wavelength_response_s3.py \
  --bucket roman-s3-romcom \
  --prefix rges-pit/ice_model \
  --dry-run
```

The uploader refreshes `SHA256SUMS`, validates the listed local files, and shows
the planned S3 sync. Review the source, destination, and planned keys; confirm
that the two stable archive filenames are present and no dated model archive is
being published. `--source-dir` is optional when run from the staged directory;
otherwise pass the staged directory explicitly. Both `--bucket` and `--prefix`
are required. Running the script without them only prints its usage message.

After reviewing the dry-run, repeat the command without `--dry-run` to upload:

```bash
python upload_wfi_wavelength_response_s3.py \
  --bucket roman-s3-romcom \
  --prefix rges-pit/ice_model
```

The uploader uses the normal Nexus AWS credential chain. A successful run prints
the refreshed and validated checksum counts and invokes a filtered `aws s3 sync`
for the staged model and support files. It can update matching keys but does not
delete unrelated S3 objects. The separate reader HDF5 output under `final/` is
not included in this paired NPZ archive sync.

Confirm the two archive objects are present after the upload:

```bash
aws s3 ls s3://roman-s3-romcom/rges-pit/ice_model/wfi_all_2nm_rank16.npz
aws s3 ls s3://roman-s3-romcom/rges-pit/ice_model/wfi_all_2nm_rank16_modulated.npz
```

### Separately upload the reader HDF5 model

The reader-compatible `joint_modulated_v2.h5` is a separate one-object upload;
it is not part of the paired NPZ sync above. For the HDF5 validation and current
uploader behavior, see [`WFI_JOINT_MODULATED_H5_PIPELINE.md`](WFI_JOINT_MODULATED_H5_PIPELINE.md).
The Nexus `eric_format` upload used this source file and destination key:

```bash
cd /teams/romcom/data/dpaterson/CAR190/completed_model/eric_format
```

Preview the single-object copy with the HDF5 uploader. Pass the source and key
explicitly so the helper uses the `joint_modulated_v2.h5` names from this run;
the checked-in helper's default names are different:

```bash
python upload_joint_modulated_all_detectors_s3.py \
  --source joint_modulated_v2.h5 \
  --bucket roman-s3-romcom \
  --prefix rges-pit/ice_model \
  --key joint_modulated_v2.h5 \
  --dry-run
```

Check the printed source, destination, and SHA256. After confirming they identify
the intended model, repeat without `--dry-run`:

```bash
python upload_joint_modulated_all_detectors_s3.py \
  --source joint_modulated_v2.h5 \
  --bucket roman-s3-romcom \
  --prefix rges-pit/ice_model \
  --key joint_modulated_v2.h5
```

This helper validates the HDF5 model and performs one `aws s3 cp` with the
`application/x-hdf5` content type; it does not sync or delete other S3 objects.
Confirm the object is present:

```bash
aws s3 ls s3://roman-s3-romcom/rges-pit/ice_model/joint_modulated_v2.h5
```
