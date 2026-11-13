# Canonical Amber stellar-photometry reruns

The default fitting entry point is `wfi_pipeline/run_amber_student_t_detectors.py`. It uses
each detector's own effective-area curve, the full imaging throughput support,
free endpoints at the first and last nonzero throughput, six fixed LED anchors,
and LED-SED-dependent suppression with red exponents 0.05:

```bash
python wfi_pipeline/run_amber_student_t_detectors.py --detector WFI01
```

Use explicit repeated `--detector` selections to run several detectors. They
run sequentially. To inspect input/output selection without fitting:

```bash
python wfi_pipeline/run_amber_student_t_detectors.py \
  --detector WFI01 --detector WFI18 --check-configuration
```

## Spectral weights

- Tables: `external_reference_data/roman_effarea_v8_20240301/`, named
  `Roman_effarea_v8_SCA01_20240301.ecsv` through `...SCA18...`.
- Official source: `RomanSpaceTelescope/roman-technical-information`, pinned
  commit `14a9b0c98b0da723d0f2e4c6277956183c1fdd06`.
- Each table has a `.provenance.json` containing the URL, commit, units and
  SHA256 hash. The selected table and checksum are recorded in the fit summary.
- Wavelength is in microns in the table and converted to nm by the fitter;
  effective areas are in square meters. No separate QE multiplier is applied.
- A missing detector table is an error, not permission to substitute SCA05.

To retrieve a missing table from the same pinned source:

```bash
python wfi_pipeline/fetch_roman_detector_effective_areas.py --detector WFI01 \
  --revision 14a9b0c98b0da723d0f2e4c6277956183c1fdd06
```

`--legacy-anchored --effective-area <file> --output-dir <directory>` remains available for explicit
sensitivity tests or historical SCA05 controls. The default output name ends in
`_sca##_effective_area_support_aligned_free_endpoints_bound0p60_soft_led_sed_red_power_0p05`, separate from historical fits. An existing converged
fit is reused only when its stored effective-area checksum matches the chosen
table. Historical-reader helpers retain their previous defaults so saved
SCA05 diagnostics are not silently reinterpreted with new weights; the fitting
entry points always pass the resolved detector-specific table explicitly.

## Other fit settings

The canonical likelihood remains Student-t with fixed nu=4, one fitted extra
scale per detector, and robust per-star/filter constant brightness levels.
The extra scale is a Student-t scale, not an RMS. The per-gap modulation basis,
amplitude/phase bounds (+/-0.30 and +/-1 rad), and regularization weights
(1000 and 100) are unchanged. Ice histories and tile optical states are fixed.
The two endpoint log-amplitude coefficients have bounds +/-0.60; the original
gap coefficients retain +/-0.30. The wavelength grid includes zero-throughput
samples on either side of the detector-specific imaging-filter support.
The modulation log-amplitude and phase basis are both multiplied by
`1 - max(LED1, LED2, LED3, LED4, LED5**0.05, LED6**0.05)`, where each
LED spectrum is clipped to nonnegative values and normalized to its own peak.
This is a soft restriction and does not guarantee exactly unchanged LED-integrated
responses. The pinned input LED SED is recorded with its SHA256 in fit provenance.
Roll offsets with no cross-roll stellar support are fixed to zero.

The controlled sample contains finite positive-error Amber counterparts of
Zane's saved likelihood-used measurements, with at least two common visits per
star/filter. This inherits the saved Zane DQ, saturation and repeat-outlier
selection; Amber's table has no native DQ field. The first 11 visits through
Oct 31 are used. Successful faint-v2 SEDs and Amber's supplied EE-corrected
magnitudes are retained; no additional clipping or EE correction is added.

Global modulation coefficients from the previous Student-t experiment can seed
the rerun when available; a different source cohort does not invalidate that
numerical initializer. Otherwise the existing Zane coefficients initialize it.

The completed strong-red-mask fits are the source of the current
`wfi_all_2nm_rank16_amber_strong_red_led_mask.npz` archive.
Reproduce the optical export with
`wfi_pipeline/build_modulated_wfi_wavelength_response_bundle.py --modulation-fit amber-student-t-strong-red-led-mask`
and explicit detector, base-bundle, fit-root and output arguments, as documented
in `WFI_MODULATED_WAVELENGTH_RESPONSE_BUNDLE.md`. The earlier shared-SCA05
experiments and optical archive remain available for historical reproduction.
