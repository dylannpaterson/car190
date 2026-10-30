# WFI Wavelength-Response Bundle

This bundle is a portable representation of the fixed-optics WFI relative
response for the current WFI01 256-pixel tile analysis.

## Files

The generated data product is one compressed NumPy archive:

```text
20261019/corrected_statistics_4h_thinned/production/wfi01/relative_response/wavelength_response_bundle/wfi01_2nm_rank16.npz
```

The companion reader is:

```text
wfi_wavelength_response_bundle.py
```

The reader only requires Python 3 and NumPy. It does not import this
repository, JAX, NumPyro, or the production TMM code.

## Contents

- 256 WFI01 tiles in the current archive, with their original tile row/column
  and pixel bounds. The format supports multiple detectors in one archive.
- Wavelength grid from 480 to 2700 nm at 2 nm spacing.
- Ice-thickness grid from 0 to 500 nm at 2 nm spacing.
- Clean fixed-optics spectral power for each tile.
- Low-rank response factors for each tile. The relative response is
  `response_left @ response_right` after linear interpolation in both axes.
- Nominal MPA focal-plane SCA and tile-centre coordinates in millimetres.
- The fitted MAP `tile_ice_thickness[tile, observation]` and
  `tile_instantaneous_rate_nm_day[tile, observation]` trajectories, indexed by
  shared relative days and UTC exposure-start times.
- The per-tile `post_reset_tile_log_offset`, its derived
  `post_reset_tile_throughput_factor`, and shared `ice_reset_day` for the grey
  throughput change.
- If the source fits include an optional third-decon boundary, the bundle also
  contains `post_third_decon_tile_log_offset`, its throughput factor, and
  `third_decon_day`; these arrays are omitted from legacy bundles.
- The geometry and response definitions in the embedded `metadata_json` field.

The response is the log of coated power divided by clean power before applying
any filter, SED, or detector-QE weighting. A downstream user can therefore
apply a different filter or source spectrum without rebuilding the optical
model.

For an observation after `ice_reset_day`, apply the grey throughput term by
adding the tile's `post_reset_tile_log_offset` to the log response, or by
multiplying power by `post_reset_tile_throughput_factor`. The wavelength
response curves themselves remain the ice/optical response; the grey term is
stored separately so it can be applied at the observation time.
For an observation on/after `third_decon_day`, apply the corresponding
post-third-decon term in addition to the decon2 term.

To rebuild the archive from the per-tile caches:

```bash
python build_wfi_wavelength_response_bundle.py \
  --cache-root 20261012/wavelength_response_cache/tile_size_256/wfi01_2nm_rank16 \
  --fit-root 20261019/corrected_statistics_4h_thinned/production \
  --output 20261019/corrected_statistics_4h_thinned/production/wfi01/relative_response/wavelength_response_bundle/wfi01_2nm_rank16.npz \
  --detector WFI01 \
  --tile-size 256
```

When all detector caches are available below one parent directory, create one
distributable archive with one cache-root argument. The builder discovers the
detector labels from the per-tile cache metadata:

```bash
python build_wfi_wavelength_response_bundle.py \
  --cache-root 20261012/wavelength_response_cache/tile_size_256 \
  --fit-root 20261019/corrected_statistics_4h_thinned/production \
  --all-detectors \
  --output 20261019/corrected_statistics_4h_thinned/production/wavelength_response_bundle/wfi_all_2nm_rank16.npz \
  --tile-size 256
```

`--all-detectors` requires WFI01 through WFI18 and prevents accidentally
distributing a partial archive; it also requires the complete tile grid for
each detector. Without it, the builder packs every detector cache it discovers
below the parent. In a multi-detector bundle, tile IDs repeat per detector;
always supply the detector when querying a repeated tile ID.

## Quick Start

Inspect the bundle:

```bash
python wfi_wavelength_response_bundle.py \
  20261019/corrected_statistics_4h_thinned/production/wfi01/relative_response/wavelength_response_bundle/wfi01_2nm_rank16.npz \
  --summary
```

List tile locations:

```bash
python wfi_wavelength_response_bundle.py \
  20261019/corrected_statistics_4h_thinned/production/wfi01/relative_response/wavelength_response_bundle/wfi01_2nm_rank16.npz \
  --tile-map-csv wfi01_tile_mpa_locations.csv
```

Evaluate tile 044 at 100 nm ice thickness and 1300 nm wavelength:

```bash
python wfi_wavelength_response_bundle.py \
  20261019/corrected_statistics_4h_thinned/production/wfi01/relative_response/wavelength_response_bundle/wfi01_2nm_rank16.npz \
  --tile-id 44 \
  --ice-thickness-nm 100 \
  --wavelength-nm 1300
```

Find the nearest tile to an MPA location and evaluate it:

```bash
python wfi_wavelength_response_bundle.py \
  20261019/corrected_statistics_4h_thinned/production/wfi01/relative_response/wavelength_response_bundle/wfi01_2nm_rank16.npz \
  --x-mm -33.66 --y-mm -1.93 \
  --ice-thickness-nm 100 \
  --wavelength-nm 1300
```

Query the fitted ice trajectory for a tile:

```bash
python wfi_wavelength_response_bundle.py \
  20261019/corrected_statistics_4h_thinned/production/wfi01/relative_response/wavelength_response_bundle/wfi01_2nm_rank16.npz \
  --detector WFI01 --tile-id 44 --trajectory
```

The Python API is also usable directly:

```python
import numpy as np

from wfi_wavelength_response_bundle import evaluate, load_bundle

bundle = load_bundle("wfi01_2nm_rank16.npz")
clean, log_response = evaluate(
    bundle,
    tile_id=44,
    ice_thickness_nm=[100.0],
    wavelengths_nm=np.arange(900.0, 1700.0, 2.0),
    detector="WFI01",
)
coated = clean[None, :] * np.exp(log_response)
```

## MPA Coordinates

The SCA centres use the supplied nominal `MPA_SCA_info` layout, with SCA
width `40.88 mm`. The MPA coordinate convention is explicit: positive MPA
`x` points left in the focal-plane view. The supplied SCA rotations are all
zero degrees.

The input tile bounds are zero-based, half-open science-frame pixel intervals
on a 4096 by 4096 detector. The four reference pixels on each edge are not
part of the 4088 by 4088 active area. For each tile, the bundle reports both
the original full-array pixel centre and the active-pixel centre used for its
physical coordinate.

With pixel pitch `0.01 mm`, the nominal mapping is:

```text
active_col_center = 0.5 * (max(col_start, 4) + min(col_stop, 4092))
active_row_center = 0.5 * (max(row_start, 4) + min(row_stop, 4092))

local_x_mm = -(active_col_center - 2048) * 0.01
local_y_mm =  (active_row_center - 2048) * 0.01

tile_center_x_mm = sca_center_x_mm + local_x_mm
tile_center_y_mm = sca_center_y_mm + local_y_mm
```

The minus sign in `local_x_mm` converts the increasing-right science-frame
column direction to the MPA convention where positive `x` points left. The
bundle stores the resulting values directly, so consumers should use
`tile_center_x_mm` and `tile_center_y_mm` rather than reimplementing the
formula.

These are nominal mechanical MPA coordinates, not sky coordinates and not
as-built metrology or distortion-corrected WCS coordinates.

## Limitations

- The current all-detector archive contains WFI01 through WFI18 in one file,
  with detector-qualified tile IDs.
- The stored response is rank-16 and interpolated on a 2 nm by 2 nm grid.
  Validation against direct TMM gives approximately `1e-4` worst-case sampled
  log-response error for the current WFI01 archive.
- Filter throughput curves, source SEDs, and detector QE are intentionally not
  embedded. They are downstream weighting inputs.
- The response uses the fixed per-tile AR/HgCdTe/incidence-angle optics fitted
  in the source production run. The trajectory arrays are the current
  relative-response MAP fit supplied through `--fit-root`. The source
  provenance is retained in the portable archive's `fit_paths`,
  `statistics_paths`, and `fit_statistics_fingerprints` arrays.
