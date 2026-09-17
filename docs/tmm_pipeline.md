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
`likelihood_weight / log_noise_sigma^2`.

The loader concatenates tile rows in SCA order while keeping the 114-group
observation axis shared across SCAs. It verifies that all SCAs have identical
tile IDs, LED/repeat metadata, and exposure times.

## Optical Response

The forward model is a normal-incidence characteristic-matrix calculation for
an air/ice/AR/HgCdTe stack. The AR parameters are:

```text
AR thickness scale, AR refractive index, AR absorption index
```

They are constrained to:

```text
thickness: 0.01 to 10.0
n:         1.0  to 4.0
k:         0.0  to 0.5
```

The response is integrated over the supplied SED and QE weighting. The
current default fixes HgCdTe at `n=3.6`, `k=0.7` and uses an ice index of
`1.25 + 0j`.

## Shared Ice Trajectory

The baseline thickness shape is the square-root trajectory

```text
b(t) = 122.7 * [sqrt((18 + t) / 4.65) - sqrt(18 / 4.65)]
```

The fitted trajectory is

```text
d(t) = d0 + exp(log_scale) * b(t)
```

`d0`, `log_scale`, and one ACM step offset are shared across all SCAs and
tiles. The ACM step is represented by a per-group design vector that is
centered on the first observation for each LED.

With `--np-modulation`, the repeat-node multipliers are applied in log space.
The first node is fixed at one. Generalized relative entropy keeps the
multipliers near unity and time-aware curvature penalizes abrupt changes.

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

## File Shapes

For `N` SCAs, `T` selected tiles, and 114 groups:

```text
counts, sums, sum_squared: (N*T, 114)
ar_parameters:             (draws, N, T, 3)
ice_thickness:             (draws, 114)
ar_sca_mean:               (draws, N, 3)       # all-SCA only
```

The prediction function flattens AR parameters in SCA-major, tile-minor
order, matching the concatenated sufficient-statistic rows.

## Production Checks

Before a production fit, verify:

1. The statistics directory contains exactly 18 final `wfi*_tile_log_statistics_*.csv` files.
2. Every file has 114 groups and the same 64 tile IDs unless `--max-tiles` is supplied.
3. The posterior has no unacceptable divergence or convergence problems.
4. Results are compared against the single-SCA TMM fit and the optional NP
   modulation diagnostic.

The effective-boundary NumPyro fit is intentionally separate. It should not
replace the TMM model until the exact external boundary parameterization is
available.
