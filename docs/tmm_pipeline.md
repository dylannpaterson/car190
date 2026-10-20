# TMM Pipeline Details

## Data Reduction

For each calibrated exposure, the streaming builder compares the current
image with the first exposure for the same LED. Pixels with invalid data or
DQ flags are excluded. For every 512-pixel tile it writes sufficient
statistics:

- `n`: number of valid pixel pairs;
- `S`: sum of pixel log changes;
- `Q`: sum of squared pixel log changes.

For model prediction `m`, the tile-level residual sum of squares is evaluated
without retaining individual pixels:

```text
SSE = Q - 2 m S + n m^2
```

This is equivalent to fitting the retained pixel log changes for a Gaussian
likelihood. The NumPyro model multiplies the total SSE by
`likelihood_weight / (n_tiles * log_noise_sigma^2)`. By default,
`likelihood_weight` is the inverse median valid-pixel count. This first converts
the pixel SSE to an approximately one-effective-datum tile/group contribution,
then averages those contributions across the selected tiles so adding tiles
does not make the aggregated all-SCA posterior numerically intractable.
Explicit weights remain available for sensitivity tests.

The `--fixed-corrections` mode applies fixed gain and first/late ACM corrections
directly to the retained log-ratio sufficient statistics. It accepts either a
common group table or a detector/tile/group table. An optional
`additional_acm_step_log_offset` column applies a repeat-activated intermediate
ACM correction when the data were loaded with
`--additional-acm-step-repeat`; legacy tables without that column use zero.
An optional `pre_reset_acm_step_log_offset` column applies a repeat-activated
pre-reset correction when the data were loaded with
`--pre-reset-acm-step-repeat`; legacy tables without that column use zero.
Current production uses
`refit_corrections_tile/wfi_all_tile_fixed_led3_gain_acm_corrections.csv`, which
contains one correction row for each of 18 detectors, 64 tiles, and 215 groups.
For a group correction `c`, the exact transformed statistics are
`S' = S + n c` and `Q' = Q + 2 c S + n c^2`. In this mode the ACM and early-time
gain parameters are deterministic zero terms in the TMM model and are not
fitted again.

The optional `--shared-ar-optical-properties` mode keeps one global `n/k` pair
per dielectric layer and applies the hierarchy only to thickness. This is the
preferred reduced model when tile-level optical-index variation is not
physically justified or is weakly identified.

When exactly one SCA and one tile are selected, the model omits the
unidentifiable tile-offset latent variables and samples the AR parameters
directly. Multi-tile and all-SCA fits retain the hierarchical structure.

The loader concatenates tile rows in SCA order while keeping the 126-group
observation axis shared across SCAs. It verifies that all SCAs have identical
tile IDs, LED/epoch metadata, and exposure times.

## Optical Response

The forward model is a normal-incidence characteristic-matrix calculation for
an air/ice/AR/HgCdTe stack. Each dielectric AR layer has two sampled parameters
and one reported fixed value:

```text
AR thickness scale, AR refractive index, AR absorption index (k=0)
```

The CLI selects one, two, or three layers with `--ar-layers`. Parameters are
ordered from the incident air side toward HgCdTe, and each layer has the same
hierarchical global/SCA/tile prior structure. The thickness scale is converted
to nanometers by the current effective-layer factor of 100, so three layers
can represent a multi-micron total stack while each layer retains the existing
1--1000 nm bound.

With `--shared-ar-optical-properties`, only thickness follows the
global/SCA/tile hierarchy; each layer's refractive index is shared across all
tiles. `--fit-ar-absorption` enables one fitted `k` per layer as a sensitivity
test when a lossless dielectric model is not appropriate.

They are constrained to:

```text
thickness: 0.01 to 10.0
n:         1.0  to 4.0
k:         0.0  to 0.5
```

The response is integrated over the supplied SED and QE weighting. The public
QE is the measured external QE of the finished, ice-free detector, so it
already includes the clean detector's AR coupling, absorption, and charge
collection. It is not multiplied by the absolute modeled detector absorption.
Instead, the model predicts the relative ice perturbation:

```text
rho(d, lambda) = modeled_response_with_ice / modeled_clean_response
band_response(d) = integral[SED * lambda * measured_QE * rho(d, lambda)]
                   / integral[SED * lambda * measured_QE]
```

The wavelength factor is present for photon weighting. The clean response is
therefore not algebraically counted twice: the modeled denominator removes the
clean optical factor before the ratio modifies the measured-QE baseline. This
construction can still be biased when the modeled clean stack has the wrong
complex reflection phase, even if its intensity response resembles the
measured QE. The all-SCA fit loads the public 18-column WFI QE table and maps each detector
label `WFI01` through `WFI18` to `SCU 1` through `SCU 18`. It retains one
integration context per SCA rather than reusing the SCU 1 curve. The TMM
wavelength grid is generated over the QE support (`480--2700 nm`); the legacy
 `spectral_response_model_SCA1.ecsv` argument is not used for transmission
values or wavelength limits. The production default evaluates the TMM directly
at the SED sample nodes; coarser `--wavelength-stride` values can undersample
thin-film fringes. A uniform grid can be requested with
`--wavelength-step-nm` for convergence tests. The current default fixes HgCdTe
at `n=3.6`, `k=0.7` and uses an ice index of `1.25 + 0j`. The opt-in
`--use-hgcdte-optical-constants` mode instead uses the fixed 80 K, `x=0.445`
Mouzali sensitivity model: its real index is composition-interpolated from the
published 80 K table and held constant below the table's 1.6 um edge, while
its imaginary index is computed from the published below-gap/above-gap
absorption laws. It is mutually exclusive with `--fit-hgcdte` and is not a
detector-specific calibration. The source models a finite active HgCdTe
layer, whereas the default workflow reuses its constants only as an effective
terminal boundary; the `x=0.445` low-temperature extrapolation is also outside
the stated composition range of its gap formula. `--finite-hgcdte-detector`
instead inserts a finite active layer followed by a fixed real-index backside
boundary (`--detector-backside-index`) and computes detector absorption from the
front/back flux difference. `--detector-thickness-nm` fixes the active-layer
thickness, while `--fit-detector-thickness` fits one positive log thickness per
selected SCA/tile with neighboring-tile regularization. The latter also enables
the finite-layer path. This is a sensitivity parameterization rather than the
documented H4RG architecture. Mosby et al. (2020) describe a substrate-removed
detector with an AR coating, transparent wide-bandgap HgCdTe buffer, and a
graded approximately 5 um absorber adjacent to the junction/ROIC structure;
they conclude that correctly fabricated devices are extremely unlikely to
fringe below 2 um. A homogeneous absorber with a real terminal medium omits
that buffer, grading, and terminal structure, so fitted detector thickness or
coherent cavity structure should not be interpreted physically. The opt-in
`--fit-hgcdte-modulation` mode adds seven smooth shared log-space multiplier
nodes for each of `n(lambda)` and `k(lambda)` around this baseline. The
exponential parameterization keeps both optical constants positive while
testing whether the data require additional spectral freedom.

The opt-in `--absolute-substrate-power` mode intentionally changes this
interpretation for nuisance-response fitting. It integrates source SED power
entering the semi-infinite HgCdTe boundary, divides by the modeled clean
substrate power, and fits two positive global log scales for the reference
`n(lambda)` and `k(lambda)` spectrum. QE weighting and finite-detector mode are
disabled in this branch. The scales are effective optical nuisance parameters
that can absorb an omitted layer or boundary structure; they are not reported
as detector-material measurements. Corrected data retain their observed,
per-LED baseline offsets in this mode. After integration, wavelength nodes
whose weights are exactly zero for every selected SCA, LED, and source interval
are removed before TMM evaluation. This preserves the integral exactly while
avoiding unnecessary autodiff work; the retained wavelength grid is archived
with the fit metadata.
The diagnostic runner can opt into `--sed-weighting photon` and
`--qe-response-mode clean_relative`. The latter uses the measured clean-detector
QE as a spectral weight and integrates the pointwise relative response
`coated / clean`, avoiding a second multiplication by the clean AR throughput.
Complex indices use `n + i k`; the characteristic matrix uses `-i sin(delta)`
so positive `k` is passive absorption rather than optical gain.

The published RCS SED is documented as relative spectral energy output. The
`--sed-weighting photon` mode therefore multiplies the SED by wavelength
before integration to represent relative photon rate. The `energy` mode omits
that factor. The current production contract uses `energy`; `photon` is an
optical sensitivity test whose effect is largest for broad SED wings.

The preferred angular sensitivity is `--incidence-pupil-half-angle-deg`. It
evaluates a two-dimensional circular pupil using four-point Gauss-Legendre
quadrature in normalized pupil radius squared and eight uniformly spaced
azimuths. This is uniform pupil-area weighting, not uniform solid-angle
weighting. The documented Roman f/8 beam has a 7.15 degree full opening angle,
corresponding to a marginal-ray half-angle of approximately 3.576 degrees.
`--incidence-chief-ray-x-deg` and `--incidence-chief-ray-y-deg` tilt the center
of the pupil relative to the detector normal. For field-dependent geometry,
`--incidence-chief-ray-map` accepts one row per selected tile with columns
`detector,tile_id,chief_ray_x_deg,chief_ray_y_deg`. The map must come from a
detector-surface ray trace or calibrated pupil measurement; the approximately
12 degree filter-surface AOIs in Switzer et al. (2025) must not be assigned
directly to the detector.

Each pupil ray evaluates s and p polarizations separately using oblique
admittances and phase thicknesses. Clean and coated intensities are averaged
separately over the pupil and equal s/p weights before their ratio is formed.
This follows the field- and pupil-dependent forward-model ordering in Switzer
et al., Applied Optics 64, 10525-10538 (2025). Their Section 5.E specifically
notes that diffuser flat fields can differ in pupil illumination and AOI
distribution and therefore require integration at each focal-plane position.

The older `--incidence-cone-half-angle-deg` option remains available to replay
existing results. It uses eight-point polar quadrature uniform in solid angle,
is centered on the detector normal, and cannot be combined with the 2D pupil
options. Zero pupil size and zero chief-ray tilt retain the exact
normal-incidence implementation. The opt-in `--fit-incidence-angle` mode fits
one global non-negative chief-ray polar angle relative to normal with a
truncated-Normal prior (default sigma `3` degrees) and a hard `15` degree upper
bound. It uses the circular pupil geometry, defaults to zero pupil half-angle,
and is deliberately separate from the legacy solid-angle cone and fixed
chief-ray map options. `--incidence-angle-prior-uniform` replaces the
near-normal truncated-Normal prior with a flat 0--15 degree sensitivity prior.

The optional `--use-asw-optical-constants` sensitivity replaces the constant
ice index with a wavelength-dependent amorphous-H2O table. Its source is the
pinned public table by default and can be overridden with
`--ice-optical-constants-source` for reproducible offline runs.

The optional `--fit-ice-optical-modulation` sensitivity fits six normalized
Gaussian basis nodes at the six RCS SED maxima for each of the Mastrapa baseline
`n(lambda)` and `k(lambda)` spectra. The coefficients are log-space multipliers
with a default Normal prior sigma of `0.05`, so the optical constants remain positive.
The coefficients are shared across all selected tiles and observation epochs;
this mode cannot be combined with `--fit-post-reset-ice-index` or
`--fit-pre-post-ice-index`. A reset remains a thickness-trajectory reset rather
than a segment-specific spectral correction. Posterior node values are written
to `ice_optical_modulation_posterior.csv`.

The optional `--ice-n-modulation-prior-sigma-log` overrides the `n(lambda)`
prior only; the `k(lambda)` nodes retain `--ice-modulation-prior-sigma-log`.

The optional `--fit-sca-ar-opd` sensitivity fits one zero-centered effective
optical-path perturbation per SCA. The perturbation is added only to the
coherent phase of the first AR layer as `2*pi*OPD*cos(theta)/lambda`; it does
not change the layer's interface index or admittance. It is therefore a
vacuum-equivalent optical-path offset, corresponding approximately to a
physical first-layer thickness change of `OPD / n_AR` if the first AR index is
held fixed. This is a phase-only diagnostic for an omitted SCA-dependent
stack/interface path, not a claim that the detector contains an additional
literal layer. Because the normal AR hierarchy already fits first-layer
thickness, the OPD is strongly degenerate with that existing parameter and
should be interpreted as a sensitivity direction unless the AR thickness is
externally constrained.

The optional `--fit-post-reset-tile-log-offset` mode adds one zero-centered
per-SCA/tile log correction to the observed groups on or after `--ice-reset-day`.
It is an opt-in calibration sensitivity and writes
`post_reset_tile_log_offset_posterior.csv`.

For a real substrate, an analytic three-layer equal-ratio start is obtained by
setting `r = (n_substrate / n_air)^(1/4)`, `n1:n2:n3 = r:r^2:r^3`, and using
optical phases `lambda/4`, `lambda/8`, `lambda/4` from air toward the
substrate. The middle `lambda/8` condition is available through
`--map-middle-layer-eighth-wave`; it is only an initializer because the fitted
stack remains unconstrained.

## Shared Ice Trajectory

The production trajectory uses one positive instantaneous rate at every
observation-epoch node. Rates are log-linearly interpolated and integrated to obtain thickness:

```text
r_j = exp(log_rate_node_j)
log r(t) = linear_interpolation(log r_j)
d(t) = d0 + integral[r(u), u=first_observation..t]
```

If the detector has an ice-level reset, pass `--ice-reset-day` at the first
post-reset observation. The model then fits a separate positive post-reset
baseline while retaining the same instantaneous-rate trajectory:

```text
d(t) = d0 + integral[r(u), u=first_observation..t]       before reset
d(t) = d0_reset + integral[r(u), u=reset_day..t]          after reset
```

This is a discontinuous baseline reset, not a rate reset. The physical reset day
must be chosen from the observation metadata rather than inferred from an
arbitrary time gap. In the unified reduction, the first Program 1063 epoch is
the physical reset anchor. The former day-15.06 anchor is the
operational-temperature transition used by the count-space correction builder,
not the physical reset.

The unified reduction first fits a local side-swap correction independently for
each detector and LED. Image counts and dark-subtracted photodiode signals have
separate multiplicative factors; the local fit uses a continuous spline plus a
single log-step. The side-swap-corrected photodiode-scaled statistics are then
passed to the count-space correction builder. On a later run, only the newest
`raw_statistics/` output is reused; prior scaled `statistics/` files are
ignored and regenerated. Changes to photodiode telemetry do not force raw image
reprocessing. `raw_statistics/` remains truly uncorrected and is never rewritten
by this stage.

The cumulative/base ACM steps are fitted jointly to all six LEDs as grey
throughput corrections. The additional step at effective repeat 669 and the
day-22 step at effective repeat 682 are fitted independently for each LED; the
later day-26 step at effective repeat 691 is shared grey. The operational
repeat remains the boundary for the ACM pre/post partition. After those ACM
corrections are applied, the
pre-operational temperature correction is fitted only between the first ACM
step and the pre-reset ACM step (effective repeats 14 through 34 in the
20261009 WFI02 timeline), excluding the early day-two transient. The
post-operational temperature-dependent correction starts only
after the additional ACM event (effective repeat 669 in the 20261009 WFI02
timeline). Both temperature corrections are fitted only to LED 3; the
resulting scale is applied identically to every LED. The former late ACM term
is not fitted again because it is replaced by the per-detector side-swap
calibration. The two temperature fits are therefore not interchangeable and
must both be retained when regenerating the correction table. The
post-operational temperature fit ends before the later day-22 discontinuity
(effective repeat 682 in the 20261009 WFI02 timeline). A per-LED day-22 step
is estimated locally from neighboring reference rows and stored separately
from the temperature fit. The later day-26 discontinuity is mapped to effective
repeat 691 and estimated as one shared-grey local step from the neighboring
all-program rows. The late ACM coefficient is still disabled because side-swap
handles its event at effective repeat 686.
The extrapolated factor is applied only to the Program 1063 rows before the
operational boundary (the first 1063 batch in WFI02); later Program 1063
batches receive zero for this component.

The legacy WFI01 1039-only timeline contains a common ACM-like step at repeat
35 near day 11.31 and another between repeat 38 and repeat 39. To model the
later step without fitting the empirical sinusoid across the physical ice reset, use
`--additional-acm-step-repeat 39` together with `--fit-additional-acm-step`
for a free exploratory fit. When building fixed corrections, pass
the first Program 1063 observation as `--ice-reset-day`,
`--pre-reset-acm-step-repeat 35`, and
`--additional-acm-step-repeat 39`;
the builder then fits the pre- and post-operational temperature sinusoids
separately from the physical-reset baseline, with one shared repeat-39 step, and stores
the repeat-35 and repeat-39 steps in
`pre_reset_acm_step_log_offset` and `additional_acm_step_log_offset`.
The repeat-35 correction remains active after the operational boundary; the repeat-39
factor is applied on top of that carried correction rather than replacing it.
The unified timeline inserts Program 1063 observations between those legacy
repeats, so the reduction maps legacy repeat 39 to the corresponding current
post-operational epoch instead of treating current repeat 39 as the same event.
The former legacy endpoint, repeat 56, identifies the side-swap event. On the
current unified timeline it is mapped to the effective detector repeat (686 in
the 20261007 inventory). The event must not be inferred from the maximum repeat
in the current file. Its image and photodiode factors are recorded in the
detector-specific side-swap calibration file; the retained late-step columns in
the count-correction schema are zero for production compatibility.
The TMM CLI uses `--pre-reset-acm-step-repeat` to load that fixed correction;
the repeat-35 coefficient itself is estimated by the correction-table builder,
not refit as a free TMM parameter.
The post-reset period is only a local
extrapolation over seven repeats and is not physically constrained by these
data; treat the generated table as exploratory until the correction is
validated in the full fit.

### Validated WFI01 reset reproduction

The validated WFI01 pre/post complex-index reproduction uses
`rrn_reduction_20260930/wfi01_tile_log_statistics_512.csv` and
`rrn_reduction_20260930/wfi01_count_space_fixed_corrections_three_events_grey_corrected.csv`.
The statistics file is intentionally raw. The fixed table is count-space and
contains `count_log_correction`, so using the `_count_corrected` statistics file
with this table would apply the count correction twice. Keep the incidence-angle
flags, truncated-Normal initial-thickness flags, reset/index flags, trajectory
variation flags, and `--map-steps 5000` exactly as recorded in the known-good
one-tile summary. The all-tile reproduction removes only `--max-tiles 1`; it
does not switch to the general production lognormal initialization.

Without `--ice-reset-day`, the initial thickness and trajectory are shared across
the selected data. With a reset, the two baselines are separate but the rate
trajectory remains shared. ACM
throughput offsets are fitted independently for the six LEDs. Each ACM step is
represented by a per-group design vector centered on the first observation for
that LED.

The default `log_scale` prior is centered on the historical trajectory scale.
For sensitivity tests, `--trajectory-rate-prior-mean-nm-day` expresses the
prior center in physical rate units and `--trajectory-rate-prior-sigma-log`
controls its breadth. The physical prior is on the unmodulated instantaneous
rate at the first observed time. `mean_rate_nm_day` is the time-averaged
instantaneous rate over the observed interval, while `initial_rate_nm_day`,
`final_rate_nm_day`, and `instantaneous_rate_nm_day` expose the local rate.
`--map-trajectory-rate-nm-day` changes only the MAP initialization and is useful
for testing whether a separate high-rate basin exists under the same prior.

The older `--np-modulation` option retains the historical square-root baseline
and applies entropy- and smoothness-regularized rate multipliers. It is mutually
exclusive with the production fully non-parametric trajectory.

With `--tile-rate-variation`, each selected SCA/tile receives a zero-centered
log-rate offset around the shared trajectory. Add
`--tile-rate-trajectory-variation` for independent zero-centered log-rate
deviations at every observation node, producing a fully non-parametric
trajectory for each tile. Adjacent selected tiles are regularized with a
Gaussian log-rate-difference prior based on their tile row and column
coordinates unless `--no-tile-rate-neighbor-smoothness` is supplied. The
default prior scales are `0.10` for individual tile offsets and `0.05` for
neighboring differences; the time-curvature weight defaults to `0.01`.
`tile_rate_posterior.csv` records the resulting per-tile initial, final, and
time-averaged rates.
`tile_rate_posterior.csv` records the resulting per-tile initial, final, and
time-averaged rates.

## AR Hierarchy

The unconstrained AR population mean is sampled as `ar_mu_raw` and mapped to
the physical bounds with a sigmoid:

```text
ar_mean = lower + (upper - lower) * sigmoid(ar_mu_raw)
```

For a single SCA, tile deviations use one tile-scale parameter per AR
quantity. For all SCAs, the hierarchy is:

```text
global AR mean
  -> SCA raw mean = global mean + SCA sigma * SCA standard-normal offset
  -> tile raw value = SCA raw mean + tile sigma * tile standard-normal offset
```

The default half-normal prior scales are `0.2` for SCA offsets and `0.5` for
tile offsets. These priors apply in unconstrained parameter space, before the
sigmoid mapping.

The global AR population mean has a Normal prior in the same unconstrained
space, with default sigma `2.0`. Its default center is an equal-index-ratio
quarter-wave dielectric stack at 1050 nm with near-zero absorption. Use
`--ar-mean-prior-sigma` for prior sensitivity tests. `--ar-prior-from-map-start`
recenters it when selecting a different physical MAP start.

## File Shapes

For `N` SCAs, `T` selected tiles, and 126 groups:

```text
counts, sums, sum_squared: (N*T, 126)
ar_parameters:             (draws, N, T, 3*L)
ice_thickness:             (draws, 126)
ar_sca_mean:               (draws, N, 3*L)     # all-SCA only
model_weights:             (N, 6, wavelength_samples)
```

For `L > 1`, CSV columns are named `ar_layer1_*` through `ar_layerL_*`.

The prediction function flattens AR parameters in SCA-major, tile-minor
order, matching the concatenated sufficient-statistic rows.

## Production Checks

Before a production fit, verify:

1. The statistics directory contains exactly 18 final `wfi*_tile_log_statistics_*.csv` files.
2. Every file has 215 groups and the same 64 tile IDs unless `--max-tiles` is supplied; the tile-aware correction table has matching detector/tile/group coverage.
3. The posterior has no unacceptable divergence or convergence problems.
4. Results are compared against the single-SCA TMM fit and the optional NP
   modulation diagnostic.

The effective-boundary NumPyro fit is intentionally separate. It should not
replace the TMM model until the exact external boundary parameterization is
available.
