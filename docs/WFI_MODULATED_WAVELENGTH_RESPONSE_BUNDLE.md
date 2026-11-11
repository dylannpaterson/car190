# Modulation-Aware WFI Wavelength-Response Bundle

The current canonical modulated product is built from the all-detector **joint
CAR190 + Amber's CAR201 photometry fit with 2-day per-tile ice-rate controls**.
The Amber observations come from CAR201's `ambers_photometry` dataset and use the
canonical Student-t likelihood with fixed `nu=4`. The unmodulated
`wfi_all_2nm_rank16.npz` remains the source of the shared wavelength/thickness
grids and tile geometry. The local joint archive is
`wfi_all_2nm_rank16_joint_car190_amber_2day_modulated.npz`; the stable staged/S3
filename remains `wfi_all_2nm_rank16_modulated.npz`. Previous Amber-only and
Zane/Gaussian modulated archives are retained locally for reproducibility.

The 18 detector fits include 603,235 Amber CAR201 measurements (12,698 detector/star
pairs) and 3,617,280 CAR190 LED-group observations. Amber uses the canonical
Student-t likelihood with fixed `nu=4`, detector-specific effective-area curves,
successful faint-v2 SEDs, and the 11 visits through Oct 31. CAR190 uses its
production group weighting; neither data set receives an extra likelihood
multiplier. The fit has 33 independent ice parameters per tile (8,448 per
detector), comprising segment initial-thickness and 2-day rate controls. The
jointly fitted thickness/rate histories replace the previous trajectory arrays
in the bundle, sampled on its **unchanged 785-point CAR190 observation axis**.
Grey reset/decon offsets and all existing bundle grids and array shapes are
preserved.

The modulation has 15 modes per detector: seven inter-LED gap modes, two free
filter-support endpoint modes, and six free LED-value modes. Unlike the preceding
Amber-only bundle, modulation is not constrained to unity at LED centers; the
maximum sampled `|ratio-1|` across detector LED centers is recorded in metadata.
Each detector uses its own `Roman_effarea_v8_SCA##_20240301.ecsv`. Those tables
are fit inputs, not stored filter weights and not multiplied into the exported
wavelength/thickness response. Roll zeropoints and stellar normalizations are
not embedded. Modulation coefficients are applied to each tile's clean complex
reflection before propagating the ice layer.

The joint ice histories use the existing per-tile CAR190 histories as their
adjustment baseline, while CAR190 is also included in the joint likelihood.
Treat the result as an in-sample joint sensitivity fit, not an independent
posterior. This baseline reuse is recorded in the bundle provenance.

For detector `d`, the fitted clean reflection is

```text
r_mod,d(lambda) = r_TMM,d,tile(lambda)
                  * exp(sum_j a[d,j] B[j](lambda)
                        + i sum_j phi[d,j] B[j](lambda))
```

The modulation coefficients are global per detector; the clean complex TMM
reflection and ice history remain tile-specific. The same fitted complex
multiplier is applied to each tile's clean s- and p-polarized reflection before
the ice layer is propagated. The bundle's `clean_power` is the resulting
modulated zero-thickness clean power; its low-rank response is
`log(P_ice,mod / P_clean,mod)`. It remains independent of downstream filter,
SED, and QE weighting. Visit/roll zeropoints are not part of this portable
optical response.

Build the all-detector joint CAR190 + Amber CAR201 bundle from the `20261106`
products by explicitly selecting WFI01–WFI18. The builder checkpoints response calculations in blocks;
rerun with `--resume` if a build is interrupted:

```bash
python build_modulated_wfi_wavelength_response_bundle.py \
  --base-bundle 20261106/corrected_statistics_4h_thinned/production/wavelength_response_bundle/wfi_all_2nm_rank16.npz \
  --fit-root 20261106/corrected_statistics_4h_thinned/production \
  --modulation-root 20261106/corrected_statistics_4h_thinned/production \
  --modulation-fit joint-car190-amber-2day \
  --output 20261106/corrected_statistics_4h_thinned/production/wavelength_response_bundle/wfi_all_2nm_rank16_joint_car190_amber_2day_modulated.npz \
  --checkpoint-interval 128 \
  --detector WFI01 --detector WFI02 --detector WFI03 --detector WFI04 \
  --detector WFI05 --detector WFI06 --detector WFI07 --detector WFI08 \
  --detector WFI09 --detector WFI10 --detector WFI11 --detector WFI12 \
  --detector WFI13 --detector WFI14 --detector WFI15 --detector WFI16 \
  --detector WFI17 --detector WFI18
```

The default SVD rank is inherited from the base bundle. The output preserves the
base bundle's array schema, shared grids, 785 observation times, geometry, and
grey offsets. It records per-detector joint-fit provenance, per-tile updated
ice-history sources, coefficients and modulation ratios, zero-thickness checks,
and maximum SVD reconstruction error by detector. Bundle version 3 is read by
`wfi_wavelength_response_bundle.py`; versions 1 and 2 remain supported.

Validation checks all 18 coefficient sets and source-file hashes, confirms all
4,608 tile thickness/rate histories against the joint fit, and verifies that the
existing bundle schema, time axis, geometry, and grey offsets remain unchanged.
Direct TMM comparisons sample 54 tiles (0, 136 and 255 for every detector) at
thicknesses 0, 7.3, 50, 123.4, 250 nm, and each sampled tile's final fitted
thickness. Maximum sampled absolute log-response error is `1.11e-4`; maximum
clean-power relative error is `5.91e-8`; sampled zero-thickness log response is
within `6.94e-15` of zero. The full-grid rank-16 SVD error is recorded in the
candidate bundle metadata.

## What the CAR201 correction plots validate

The all-star diagnostics evaluate the saved CAR201 photometry fitter, not the
portable bundle integration. They apply its tile-specific ice response together
with its fitted per-filter post-roll zero points. Those zero points are not
included in this bundle.

The two forward calculations also differ in normalization: the photometry fitter
integrates coated power divided wavelength-by-wavelength by each tile's
**unmodulated** clean power, whereas a bundle consumer integrating
`clean_power * exp(relative_log_response)` integrates **modulated coated power**.
These are different wavelength weights, not just a removable scalar constant.
Both paths coincide in s/p polarization at normal incidence; polarization does
not explain the tested normal-incidence discrepancy.

For the earlier Zane/Gaussian WFI01 star check, bundle integrations agreed with freshly calculated
direct modulated TMM to below 0.005 mmag RMS per filter, but differed from the
saved photometry-fit predictions by about 1.6–9.8 mmag RMS. Thus the numerical
bundle representation is checked, but the photometry-fit residual plots must
not be presented as validation of identical residual performance when applying
the exported bundle.

The earlier all-star diagnostic samples used successful faint-v2 SEDs, supplemented by
successful original SEDs and nearest-neighbor SED assignments in CAR201 magnitude
colors for remaining stars. Plot quality cuts exclude an entire detector/star
if any recorded measurement is saturated, remove DO_NOT_USE measurements, and
apply saved fit manual/repeat-outlier exclusions where available. The NN-only
subset did not enter the modulation fit, but is not a randomized cross-validation
experiment. Its approximate SEDs add uncertainty. All diagnostic change references
are recomputed after quality cuts; forced-zero reference observations are omitted
from percentile error-bar plots.

Canonical Amber-only Student-t diagnostics (the inputs to the previous modulated
bundle) are saved under
`20261106/corrected_statistics_4h_thinned/production/wavelength_response_bundle/amber_student_t_strong_red_led_mask/`.
Those fits and plots use the controlled common-quality sample described above,
not the earlier all-star/NN sample. Cumulative plots require measured EP3-pass-1
references; post-roll-only stars still contribute to the likelihood. These are
in-sample diagnostics, not held-out predictive validation. The previous
`amber_student_t_all_detectors/` directory contains historical shared-SCA05
diagnostics; its plots are not silently relabeled as the canonical rerun.

## S3 publication

The existing publication destination is
`s3://roman-s3-romcom/rges-pit/ice_model`. The uploader does not build or select a
bundle: it uploads a **staged export** containing the unmodulated
`wfi_all_2nm_rank16.npz`, modulated `wfi_all_2nm_rank16_modulated.npz`, shared
consumer module, documentation, manifest, and checksums.
Its default source directory is the directory containing the invoked uploader.
Running the repository-root script without `--source-dir` does not select the
production output directory.

The existing staged export under
`20261019/corrected_statistics_4h_thinned/production/wfi_wavelength_response_s3/`
has a 20261019 manifest. Do not use that stage unchanged for the 20261106 modulated
release. Prepare a new stage inside the workspace containing:

- The joint CAR190/Amber CAR201 2-day archive from the build command above, staged as
  `wfi_all_2nm_rank16_modulated.npz` (preserve its original source filename in the manifest).
- The current unmodulated archive, staged as `wfi_all_2nm_rank16.npz`.
- The current `wfi_wavelength_response_bundle.py`, supporting bundle version 3.
- A release-specific `README.md` explaining the modulated model and limitations
  above, and the updated generic `WFI_WAVELENGTH_RESPONSE_BUNDLE.md`.
- A release-specific `MANIFEST.txt` with actual archive grids, detector/tile
  counts, provenance, model variant, and absence of fitted roll zero points.
- The uploader and a `SHA256SUMS` listing all release files. To include this
  modulation-specific document under its own filename, list it in `SHA256SUMS`:
  the uploader preserves listed existing files when refreshing checksums.

The S3 prefix contains the current best models together, not dated releases:
`rges-pit/ice_model`. Keep dated provenance and archives locally. Publish the
unmodulated and modulated products under distinct stable filenames in that same
prefix, with shared consumer code and documentation. The uploader recognizes
both stable filenames. Prepare the paired stage with
`python stage_wfi_current_models_s3.py`, then:

```bash
python upload_wfi_wavelength_response_s3.py \
  --source-dir <new-staged-export-directory> \
  --bucket roman-s3-romcom \
  --prefix rges-pit/ice_model \
  --dry-run
```

Remove `--dry-run` to publish that staged release. The script refreshes and checks
SHA256 hashes and uses AWS CLI credentials from the normal credential chain. It
does not delete remote objects, but can overwrite matching keys in the chosen
prefix. No upload is implied by building the bundle or generating diagnostics.

### Prepared paired export

`stage_wfi_current_models_s3.py` prepares and validates the paired stage at
`20261106/corrected_statistics_4h_thinned/production/wavelength_response_bundle/s3_export/`.
The uploader recognizes both `wfi_all_2nm_rank16.npz` and
`wfi_all_2nm_rank16_modulated.npz`, plus both model documents. Grey reset/decon
offsets and modulation reconstruction fields are retained. Copy the complete
stage to Nexus, then invoke the included uploader with `--source-dir .` from
that directory and the shared `rges-pit/ice_model` prefix.
