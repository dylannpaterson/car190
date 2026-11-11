# Production WFI TMM Model

This document describes the model and regularisation used by the production
entrypoints. The ordinary `run_wfi_production.py` path evaluates the transfer
matrix at every objective evaluation. The validated 20261030 branch uses the
same ice/rate objective with a pinned six-LED relative-response cache; its
fixed optical state and reduced parameter set are described separately below.

## Data Model

The fit uses corrected per-tile sufficient statistics. Corrections and ACM
terms are already included in the current
`*_tile_log_statistics_256_count_corrected.csv` inputs and are not applied a
second time. For each LED/observation group the input contains:

- `n`: valid pixel-pair count;
- `S`: sum of pixel log changes;
- `Q`: sum of squared pixel log changes.

The observed group mean is `y = S / n`. For a model prediction `m`, the
within-group residual sum of squares is

```text
SSE = Q - S^2 / n + n (m - y)^2.
```

The production log-noise scale is `LOG_NOISE_SIGMA = 0.005` in log response.
The default likelihood weight is the inverse median valid-pixel count. The
tile contribution is additionally divided by the number of fitted tiles, so
the objective remains comparable as tiles are selected. The MAP objective is
the weighted Gaussian likelihood plus the prior and regularisation residuals.

The first group for each LED is the reference. The forward model predicts a
relative log change by comparing the current and reference response for the
same LED. Optional fitted LED offsets apply only to non-reference groups.

## Full-TMM Forward Model

For each tile, the optical stack is

```text
air -> ice -> AR layer 1 -> AR layer 2 -> AR layer 3 -> semi-infinite HgCdTe
```

The characteristic-matrix calculation evaluates the complex response at the
SED wavelength nodes. The current production settings are:

- three independent AR layers;
- lossless AR layers, with `k = 0` fixed;
- fixed normal incidence;
- energy-weighted RCS LED SED;
- the pinned local SED file `external_reference_data/WFI_RCS_LEDP_spectral_energy_distribution.ecsv`;
- active wavelength support approximately `480--2700 nm`;
- semi-infinite HgCdTe using the code's fixed 80 K, composition `x = 0.445`
  reference `n(lambda)` and `k(lambda)` model;
- `qe_response_mode = none`, so the measured QE table is not multiplied into
  the current production SED integral. `clean_relative` QE weighting is a
  diagnostic alternative, not the validated production setting.

The SED output below five percent of its peak is zeroed before integration.
For each LED, the integrated coated response is divided by the integrated
clean response. The six resulting ratios are converted to log changes and
normalised to the first observation for that LED.

### AR Parameters

Each tile has two free parameters per AR layer: a thickness coordinate and a
refractive index. The serialized thickness coordinate is converted to
physical thickness by multiplying by `100 nm`. The production bounds are:

```text
thickness coordinate: 0.01--10.0  (1--1000 nm physical thickness)
refractive index:     1.0--4.0 for layer 1
                      previous layer n--4.0 for layers 2 and 3
absorption index k:   0, fixed
```

The AR prior is Gaussian in the raw logistic coordinates used to enforce the
bounds. Its center is a three-layer quarter-wave design at `1050 nm`, using
the geometric sequence derived from nominal HgCdTe `n = 3.6`; its raw-coordinate
sigma is `10.0` for every active AR thickness and index coordinate. The large
raw-coordinate width is intentional and is not a narrow physical Gaussian
prior on thickness or index.

### HgCdTe Parameters

Each tile has independent positive wavelength-independent log scales for the
reference HgCdTe optical constants:

```text
n(lambda) = n_reference(lambda) * exp(log_n_scale + n_modulation(lambda))
k(lambda) = k_reference(lambda) * exp(log_k_scale + k_modulation(lambda))
```

The production priors are zero-centered normals in log scale, each with sigma
`0.25`. Thus the multiplicative scales are lognormal and centered at one.
They are effective optical nuisance parameters, not material-property
measurements.

### Anchored Spectral Modulation

The deterministic MAP fits broad log-space modulation for AR `n`, HgCdTe `n`,
and HgCdTe `k`. The modulation uses six broad Gaussian basis nodes near

```text
624, 853, 1039, 1270, 1481, 1670 nm
```

with width `500 nm`. The basis is differenced against `1050 nm`, so the
modulation is anchored to zero at the AR design wavelength. Each modulation
coefficient has a zero-centered normal prior with log sigma `0.10` in the
production wrapper. AR `k` is not fitted in production.

### Ice Optical Index and Incidence

The production wrapper fixes the ice real index at `n = 1.25` by setting its
log scale to zero. The underlying constant-index model has a zero-centered
log prior with sigma `0.01`, but the value is fixed rather than estimated in
the validated full-TMM fit. Ice `k` is not fitted. Incidence is fixed at
normal incidence; the generic model's `0--15 degree` uniform angle prior is
not active in production.

## Tile Ice and Rate Model

The production trajectory is independent for every detector tile. At each
repeat node day `t_i`, the free parameter is the log instantaneous rate
`q_i`. The positive rate is

```text
r_i = exp(q_i).
```

The log rate is linearly interpolated between node days and integrated with
Gauss-Legendre quadrature to obtain the ice thickness at each observation.
There is no whitening, shared detector trajectory, hierarchical tile
trajectory, or neighbor coupling in the production fit.

Each reset starts an independent thickness trajectory. The physical first
reset is the Program 1063 boundary at relative day
`12.907397928240743`. When the current cached production run includes the
third decon, its boundary is explicitly supplied as
`38.79784510416666`; it is not inferred from the first later observation.

The fitted grey terms are additive log-throughput offsets applied to observed
post-reset groups. They are equivalent to multiplicative throughput factors
`exp(offset)` and have one independent site per physical reset and tile.

## Production Priors

The following are the settings used by the production wrapper. Some module
defaults support diagnostic experiments and differ from these wrapper values.

| Component | Production parameterisation | Prior or constraint |
| --- | --- | --- |
| AR thickness and `n` | Per-tile raw bounded coordinates | Normal around the 1050 nm quarter-wave raw design, sigma `10.0`; physical bounds above |
| AR `k` | Not fitted | Fixed to `0` |
| HgCdTe `n` scale | Per-tile positive log scale | `Normal(0, 0.25)` |
| HgCdTe `k` scale | Per-tile positive log scale | `Normal(0, 0.25)` |
| AR `n` modulation | Per tile, three layers, six anchored basis coefficients | `Normal(0, 0.10)` in log space |
| HgCdTe `n` modulation | Per tile, six anchored basis coefficients | `Normal(0, 0.10)` in log space |
| HgCdTe `k` modulation | Per tile, six anchored basis coefficients | `Normal(0, 0.10)` in log space |
| Ice real `n` | Fixed constant | `n = 1.25`; underlying log-prior sigma `0.01`, but no posterior freedom |
| Ice `k` | Not fitted | No active production prior |
| Initial thickness | Per tile, log thickness | `Normal(log(1 nm), 1.5)` in the current full-TMM production wrapper |
| Reset thickness | Per tile and reset, log thickness | `Normal(log(1 nm), 1.5)` in the current full-TMM production wrapper |
| Rate nodes | Per tile, non-parametric log rate | `Normal(log(12 nm/day), 1.5)` at every node |
| Post-reset grey offset | Per tile and reset | `Normal(0, 0.1)` in log throughput |
| Per-LED log offset | Disabled in standard full-TMM production | When enabled for the validated cached run: `Normal(0, 0.01)` per detector and LED |

The prior on each third-decon thickness in the validated 20261030 cached run
is replaced only for that stage by a per-tile center from an independent
constant-rate warm-up, with log sigma `0.10`. With
`--thickness-prior-stages third-decon`, the initial and first-reset thickness
priors are centered at `1 nm` with log sigma `1.5` in the current full-TMM
production wrapper. The cached relative-response branch retains its validated
`10 nm` initial and first-reset centers. The separate warm-up's `40 nm` start
is an initializer, not the final third-Decon prior center.

## Regularisation and Optimisation

The production MAP is a JAX-differentiated SciPy trust-region
least-squares fit. Free parameters are optimized in coordinates scaled by
their prior widths, with up to `10000` function evaluations. The practical
function-reduction and step tolerances are both `1e-9`; the scaled-gradient
criterion is also evaluated and reported separately.

The validated production regularisation settings are:

- trajectory curvature/smoothness regularisation: disabled with weight `0.0`;
- relative-entropy/MaxEnt regularisation: disabled with weight `0.0`;
- neighboring-tile thickness penalty: disabled;
- ACM and gain correction refits: disabled because corrections are already in
  the sufficient statistics;
- tile-to-tile smoothing or hierarchical sharing: disabled.

The module constant `NP_SMOOTHNESS_WEIGHT = 0.1` is a legacy/diagnostic
default. `run_wfi_production.py` explicitly passes zero for production. The
constant-rate warm-up is an initialization stage, not a regulariser on the
final non-parametric trajectory. The source trajectory from a cached optical
fit is likewise not imported as a rate prior or entropy reference.

## Initialisation

The standard full-TMM path performs a fresh single-tile EMCEE basin search per
detector with:

- `52` walkers;
- `1000` steps;
- seed `20261004`;
- constant-in-time rate;
- no complex-q parameters;
- the production rate prior centered at `12 nm/day`.

The selected state initializes a deterministic constant-rate MAP warm-up for
each tile. That scalar rate is then copied across the non-parametric rate
nodes before the final independent-tile MAP fit.

## Cached Relative-Response Production

The `--response-cache-root` branch fixes the full optical state from a pinned
source fit and replaces repeated TMM/SED evaluation with six interpolated log
response curves per tile. The validated cache grid is `0--400 nm` at
`0.25 nm` spacing. The cache fixes:

- the three-layer AR parameters;
- HgCdTe `n` and `k` scales;
- ice-index state;
- anchored AR/HgCdTe spectral modulation;
- normal incidence.

The final cached fit estimates only the ice trajectory, reset grey offsets,
and, in the validated 20261030 run, the per-LED log offsets. It performs an
in-process constant-rate warm-up before the final non-parametric MAP. The
source fit's time-dependent trajectory is not reused as a prior or entropy
reference. The cached branch uses the same `12 nm/day`, log sigma `1.5` rate
prior and zero trajectory regularisation as the full production objective.

Warm-up-centered third-decon thickness priors require independent one-tile
fits (`--batch-size 1`, without `--joint-batch`) and the explicit third-decon
boundary.

## Non-Production Alternatives

The following code paths exist for sensitivity studies but are not part of the
validated production contract: fitted AR absorption, fitted ice absorption,
free ice spectral modulation, direct thickness-node trajectories, photon SED
weighting, clean-relative QE weighting, finite-detector HgCdTe, conductive or
complex-q models, MaxEnt or curvature regularisation, and neighboring-tile
thickness penalties.

## Source References

- `fit_wfi_tmm_production.py`: forward model, likelihood, priors, and MAP objective.
- `run_wfi_production.py`: full-TMM production overrides and initialization.
- `run_wfi_tmm_emcee_warmup.py`: EMCEE basin-search configuration.
- `fit_wfi_relative_response.py` and `wfi_relative_response.py`: cached branch objective and response interpolation.
- `docs/tmm_pipeline.md`: sufficient-statistics reduction and transfer-matrix background.
- `docs/WFI_RELATIVE_RESPONSE_PRODUCTION.md`: validated cached-run command contract.
