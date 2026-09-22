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

The optional `--shared-ar-optical-properties` mode keeps one global `n/k` pair
per dielectric layer and applies the hierarchy only to thickness. This is the
preferred reduced model when tile-level optical-index variation is not
physically justified or is weakly identified.

When exactly one SCA and one tile are selected, the model omits the
unidentifiable tile-offset latent variables and samples the AR parameters
directly. Multi-tile and all-SCA fits retain the hierarchical structure.

The loader concatenates tile rows in SCA order while keeping the 126-group
observation axis shared across SCAs. It verifies that all SCAs have identical
tile IDs, LED/repeat metadata, and exposure times.

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

The response is integrated over the supplied SED and QE weighting. The
all-SCA fit loads the public 18-column WFI QE table and maps each detector
label `WFI01` through `WFI18` to `SCU 1` through `SCU 18`. It retains one
integration context per SCA rather than reusing the SCU 1 curve. The TMM
wavelength grid is generated over the QE support (`480--2700 nm`); the legacy
 `spectral_response_model_SCA1.ecsv` argument is not used for transmission
values or wavelength limits. The production default evaluates the TMM directly
at the SED sample nodes; coarser `--wavelength-stride` values can undersample
thin-film fringes. A uniform grid can be requested with
`--wavelength-step-nm` for convergence tests. The current default fixes HgCdTe
at `n=3.6`, `k=0.7` and uses an ice index of `1.25 + 0j`.
Complex indices use `n + i k`; the characteristic matrix uses `-i sin(delta)`
so positive `k` is passive absorption rather than optical gain.

For a real substrate, an analytic three-layer equal-ratio start is obtained by
setting `r = (n_substrate / n_air)^(1/4)`, `n1:n2:n3 = r:r^2:r^3`, and using
optical phases `lambda/4`, `lambda/8`, `lambda/4` from air toward the
substrate. The middle `lambda/8` condition is available through
`--map-middle-layer-eighth-wave`; it is only an initializer because the fitted
stack remains unconstrained.

## Shared Ice Trajectory

The baseline instantaneous rate is the derivative of the historical
square-root trajectory:

```text
r_base(t) = 122.7 / [2 * sqrt(4.65) * sqrt(18 + t)]
```

The fitted instantaneous rate and thickness are

```text
r(t) = exp(log_scale) * r_base(t) * m(t)
d(t) = d0 + integral[r(u), u=first_observation..t]
```

`d0`, `log_scale`, and one ACM step offset are shared across all SCAs and
tiles. The ACM step is represented by a per-group design vector that is
centered on the first observation for each LED.

The default `log_scale` prior is centered on the historical trajectory scale.
For sensitivity tests, `--trajectory-rate-prior-mean-nm-day` expresses the
prior center in physical rate units and `--trajectory-rate-prior-sigma-log`
controls its breadth. The physical prior is on the unmodulated instantaneous
rate at the first observed time. `mean_rate_nm_day` is the time-averaged
instantaneous rate over the observed interval, while `initial_rate_nm_day`,
`final_rate_nm_day`, and `instantaneous_rate_nm_day` expose the local rate.
`--map-trajectory-rate-nm-day` changes only the MAP initialization and is useful
for testing whether a separate high-rate basin exists under the same prior.

With `--np-modulation`, `m(t)` is a positive, log-linearly interpolated rate
multiplier between repeat nodes. The first node is fixed at one. Generalized
relative entropy keeps the multipliers near unity and time-aware curvature
penalizes abrupt changes. Thickness is always obtained by integrating the
resulting rate, so NP modulation cannot directly rescale accumulated thickness.

With `--tile-rate-variation`, each selected SCA/tile receives a zero-centered
log-rate offset around the shared trajectory. Adjacent selected tiles are
regularized with a Gaussian log-rate-difference prior based on their tile row
and column coordinates. The default prior scales are `0.10` for individual
tile offsets and `0.05` for neighboring differences; both are CLI options for
sensitivity tests. `tile_rate_posterior.csv` records the resulting per-tile
initial, final, and time-averaged rates.

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
2. Every file has 126 groups and the same 64 tile IDs unless `--max-tiles` is supplied.
3. The posterior has no unacceptable divergence or convergence problems.
4. Results are compared against the single-SCA TMM fit and the optional NP
   modulation diagnostic.

The effective-boundary NumPyro fit is intentionally separate. It should not
replace the TMM model until the exact external boundary parameterization is
available.
