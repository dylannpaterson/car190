"""Plot per-tile ice rates in the Roman WFI focal-plane mosaic."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
SCAS = tuple(f"WFI{index:02d}" for index in range(1, 19))
DEFAULT_FIT_SUFFIX = "production"
METRICS = (
    ("Initial rate", "initial_rate_nm_day_median"),
    ("Final rate", "final_rate_nm_day_median"),
    ("Mean rate", "mean_rate_nm_day_median"),
)
RESET_METRICS = (
    ("Mean before reset", "pre_reset_mean_rate_nm_day"),
    ("Mean after reset", "post_reset_mean_rate_nm_day"),
)
TREND_METRICS = (
    ("Linear trend before reset", "pre_reset_trend_rate_nm_day"),
    ("Linear trend after reset", "post_reset_trend_rate_nm_day"),
)

# The Roman WFI layout is a staggered six-column by three-row mosaic.  The
# offsets reproduce the smile-shaped focal plane shown in the Roman reference
# frame figure; y increases toward the top of the plot.
COLUMN_SCA = (
    ("WFI16", "WFI17", "WFI18"),
    ("WFI13", "WFI14", "WFI15"),
    ("WFI10", "WFI11", "WFI12"),
    ("WFI01", "WFI02", "WFI03"),
    ("WFI04", "WFI05", "WFI06"),
    ("WFI07", "WFI08", "WFI09"),
)
COLUMN_Y_OFFSET = (0.0, -4.0, -6.0, -6.0, -4.0, 0.0)
SCA_SIZE = 8.0
SCA_PITCH = 9.0


def sca_position(sca):
    for column, scas in enumerate(COLUMN_SCA):
        if sca in scas:
            # COLUMN_SCA is listed top-to-bottom, while plot y increases up.
            row = len(scas) - 1 - scas.index(sca)
            return SCA_PITCH * column, SCA_PITCH * row + COLUMN_Y_OFFSET[column]
    raise ValueError(f"Unknown SCA: {sca}")


def load_tile_rows(path, expected_sca):
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or {row.get("detector") for row in rows} != {expected_sca}:
        return None
    return rows


def load_rate_map(path, expected_sca):
    rows = load_tile_rows(path, expected_sca)
    if rows is None:
        fit_root = path.parent / "fits"
        rows = []
        for tile_path in sorted(fit_root.glob("tile_*/tile_rate_posterior.csv")):
            tile_rows = load_tile_rows(tile_path, expected_sca)
            if tile_rows is not None:
                rows.extend(tile_rows)
        if not rows:
            rows = None
    if rows is None:
        return None

    maps = {}
    for _, field in METRICS:
        values = np.full((8, 8), np.nan)
        for row in rows:
            values[int(row["tile_row"]), int(row["tile_col"])] = float(row[field])
        # MAST L1/L2 arrays are already in the common WFI science frame.
        # Tile row 0 is the science-frame bottom, so no physical-detector flip
        # belongs here.
        maps[field] = values
    return maps


def maxent_fit_set_complete(output_dir):
    """Return whether all 64 tile archives are current MaxEnt production fits."""
    fit_root = Path(output_dir) / "fits"
    fit_dirs = sorted(fit_root.glob("tile_*"))
    if [path.name for path in fit_dirs] != [f"tile_{tile_id:03d}" for tile_id in range(64)]:
        return False
    for fit_dir in fit_dirs:
        try:
            with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as samples:
                if (
                    str(np.asarray(samples["trajectory_regularization"]).item()) != "maxent"
                    or not np.isclose(
                        float(np.asarray(samples["trajectory_entropy_weight"]).item()),
                        0.1,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    or int(np.asarray(samples["constant_rate_map_warmup"]).item()) != 1
                ):
                    return False
        except (KeyError, OSError, ValueError):
            return False
    return True


def load_reset_rate_values(posterior_path):
    with np.load(posterior_path, allow_pickle=False) as samples:
        days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
        reset_values = np.asarray(samples["ice_reset_day"], dtype=float).reshape(-1)
        thickness = np.asarray(samples["tile_ice_thickness"], dtype=float)
        tile_ids = np.asarray(samples["tile_ids"], dtype=int).reshape(-1)

    if reset_values.size != 1 or not np.isfinite(reset_values[0]):
        return None
    if thickness.ndim < 2 or thickness.size % (tile_ids.size * days.size) != 0:
        raise ValueError(f"Unexpected tile-thickness shape in {posterior_path}")
    thickness = thickness.reshape(-1, tile_ids.size, days.size)
    if thickness.shape[1] != tile_ids.size:
        raise ValueError(f"Tile IDs do not match tile thickness in {posterior_path}")

    reset_day = float(reset_values[0])
    before = days < reset_day
    after = days >= reset_day
    if not np.any(before) or not np.any(after):
        raise ValueError(f"Reset day does not split observations in {posterior_path}")
    before_indices = np.flatnonzero(before)
    after_indices = np.flatnonzero(after)
    pre_start, pre_end = before_indices[[0, -1]]
    post_start, post_end = after_indices[[0, -1]]
    pre_values = np.median(
        (thickness[:, :, pre_end] - thickness[:, :, pre_start])
        / (days[pre_end] - days[pre_start]),
        axis=0,
    )
    post_values = np.median(
        (thickness[:, :, post_end] - thickness[:, :, post_start])
        / (days[post_end] - days[post_start]),
        axis=0,
    )
    return tile_ids, pre_values, post_values, reset_day


def reset_rate_maps_from_rows(
    rows, tile_ids, pre_values, post_values, posterior_path
):
    tile_indices = {int(tile_id): index for index, tile_id in enumerate(tile_ids)}
    maps = {field: np.full((8, 8), np.nan) for _, field in RESET_METRICS}
    for row in rows:
        tile_id = int(row["tile_id"])
        if tile_id not in tile_indices:
            raise ValueError(f"Tile {tile_id} is missing from {posterior_path}")
        tile_index = tile_indices[tile_id]
        tile_row = int(row["tile_row"])
        tile_col = int(row["tile_col"])
        maps[RESET_METRICS[0][1]][tile_row, tile_col] = pre_values[tile_index]
        maps[RESET_METRICS[1][1]][tile_row, tile_col] = post_values[tile_index]
    return maps


def load_reset_rate_map(output_dir, expected_sca):
    rows = load_tile_rows(output_dir / "tile_rate_posterior.csv", expected_sca)
    posterior_path = output_dir / "posterior_samples.npz"
    if rows is not None and posterior_path.is_file():
        loaded = load_reset_rate_values(posterior_path)
        if loaded is None:
            return None
        tile_ids, pre_mean, post_mean, reset_day = loaded
        maps = reset_rate_maps_from_rows(
            rows,
            tile_ids,
            pre_mean,
            post_mean,
            posterior_path,
        )
        return maps, reset_day

    # Independent MAP runs archive one posterior per tile rather than one
    # detector-level posterior. Reconstruct the detector map from those files.
    tile_fit_dirs = sorted((output_dir / "fits").glob("tile_*"))
    if len(tile_fit_dirs) != 64:
        return None
    maps = {field: np.full((8, 8), np.nan) for _, field in RESET_METRICS}
    reset_days = []
    for tile_fit_dir in tile_fit_dirs:
        tile_rows = load_tile_rows(
            tile_fit_dir / "tile_rate_posterior.csv", expected_sca
        )
        tile_posterior = tile_fit_dir / "posterior_samples.npz"
        if tile_rows is None or not tile_posterior.is_file():
            return None
        loaded = load_reset_rate_values(tile_posterior)
        if loaded is None:
            return None
        tile_ids, pre_mean, post_mean, reset_day = loaded
        tile_maps = reset_rate_maps_from_rows(
            tile_rows,
            tile_ids,
            pre_mean,
            post_mean,
            tile_posterior,
        )
        for field in tile_maps:
            finite = np.isfinite(tile_maps[field])
            maps[field][finite] = tile_maps[field][finite]
        reset_days.append(reset_day)
    if not np.allclose(reset_days, reset_days[0], rtol=0.0, atol=1.0e-10):
        raise ValueError(
            f"Independent outputs use inconsistent reset days: {reset_days}"
        )
    return maps, reset_days[0]


def _load_tile_thickness(posterior_path):
    with np.load(posterior_path, allow_pickle=False) as samples:
        days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
        thickness = np.asarray(samples["tile_ice_thickness"], dtype=float)
        reset_day = float(np.asarray(samples["ice_reset_day"]).reshape(-1)[0])
    if thickness.ndim == 4:
        thickness = thickness[0, 0]
    elif thickness.ndim == 3:
        thickness = thickness[0]
    elif thickness.ndim == 2:
        thickness = thickness[0]
    thickness = thickness.reshape(-1)
    if thickness.size != days.size:
        raise ValueError(f"Unexpected tile thickness shape in {posterior_path}")
    return days, thickness, reset_day


def load_trend_rate_map(output_dir, expected_sca):
    rows = load_tile_rows(output_dir / "tile_rate_posterior.csv", expected_sca)
    fit_root = Path(output_dir) / "fits"
    if rows is None or len(rows) != 64:
        return None
    maps = {
        field: np.full((8, 8), np.nan)
        for _, field in TREND_METRICS
    }
    reset_days = []
    for row in rows:
        tile_id = int(row["tile_id"])
        posterior_path = fit_root / f"tile_{tile_id:03d}" / "posterior_samples.npz"
        if not posterior_path.is_file():
            return None
        days, thickness, reset_day = _load_tile_thickness(posterior_path)
        before = days < reset_day
        after = days >= reset_day
        if not np.any(before) or not np.any(after):
            raise ValueError(f"Reset day does not split observations in {posterior_path}")
        pre_rate = np.polyfit(days[before], thickness[before], 1)[0]
        post_rate = np.polyfit(days[after], thickness[after], 1)[0]
        tile_row = int(row["tile_row"])
        tile_col = int(row["tile_col"])
        maps["pre_reset_trend_rate_nm_day"][tile_row, tile_col] = pre_rate
        maps["post_reset_trend_rate_nm_day"][tile_row, tile_col] = post_rate
        reset_days.append(reset_day)
    if not np.allclose(reset_days, reset_days[0], rtol=0.0, atol=1.0e-10):
        raise ValueError(f"Independent outputs use inconsistent reset days: {reset_days}")
    return maps, reset_days[0]


def detector_output_dirs(root, sca, fit_suffix):
    root = Path(root)
    if fit_suffix == DEFAULT_FIT_SUFFIX:
        return (root / DEFAULT_FIT_SUFFIX / sca.lower(),)
    return (
        root / f"{sca.lower()}_{fit_suffix}_prod",
        root / f"{sca.lower()}_{fit_suffix}",
        root / f"wfi_{fit_suffix}_{sca}_map10000_emcee1000",
    )


def load_all_maps(root, fit_suffix, maxent_only=False):
    maps = {}
    for sca in SCAS:
        output_dirs = detector_output_dirs(root, sca, fit_suffix)
        maps[sca] = next(
            (
                rate_map
                for output_dir in output_dirs
                if not maxent_only or maxent_fit_set_complete(output_dir)
                if (rate_map := load_rate_map(output_dir / "tile_rate_posterior.csv", sca))
                is not None
            ),
            None,
        )
    return maps


def load_all_reset_maps(root, fit_suffix, maxent_only=False):
    maps = {}
    reset_days = []
    for sca in SCAS:
        output_dirs = detector_output_dirs(root, sca, fit_suffix)
        loaded = next(
            (
                reset_map
                for output_dir in output_dirs
                if not maxent_only or maxent_fit_set_complete(output_dir)
                if (reset_map := load_reset_rate_map(output_dir, sca)) is not None
            ),
            None,
        )
        maps[sca] = None if loaded is None else loaded[0]
        if loaded is not None:
            reset_days.append(loaded[1])
    if reset_days and not np.allclose(
        reset_days, reset_days[0], rtol=0.0, atol=1.0e-10
    ):
        raise ValueError(f"Production outputs use inconsistent reset days: {reset_days}")
    return maps, (reset_days[0] if reset_days else None)


def load_all_trend_maps(root, fit_suffix, maxent_only=False):
    maps = {}
    reset_days = []
    for sca in SCAS:
        output_dirs = detector_output_dirs(root, sca, fit_suffix)
        loaded = next(
            (
                trend_map
                for output_dir in output_dirs
                if not maxent_only or maxent_fit_set_complete(output_dir)
                if (trend_map := load_trend_rate_map(output_dir, sca)) is not None
            ),
            None,
        )
        maps[sca] = None if loaded is None else loaded[0]
        if loaded is not None:
            reset_days.append(loaded[1])
    if reset_days and not np.allclose(
        reset_days, reset_days[0], rtol=0.0, atol=1.0e-10
    ):
        raise ValueError(f"Production outputs use inconsistent reset days: {reset_days}")
    return maps, (reset_days[0] if reset_days else None)


def load_rate_histogram(output_dir, expected_sca):
    fit_root = Path(output_dir) / "fits"
    fit_dirs = sorted(fit_root.glob("tile_*"))
    expected = [fit_root / f"tile_{tile_id:03d}" for tile_id in range(64)]
    if fit_dirs != expected:
        return None

    pre_values = []
    post_values = []
    current_values = []
    reset_days = []
    for fit_dir in fit_dirs:
        posterior_path = fit_dir / "posterior_samples.npz"
        if not posterior_path.is_file():
            return None
        with np.load(posterior_path, allow_pickle=False) as samples:
            days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
            rates = np.asarray(
                samples["tile_instantaneous_rate_nm_day"], dtype=float
            ).reshape(-1)
            reset_values = np.asarray(samples["ice_reset_day"], dtype=float).reshape(-1)
        if reset_values.size != 1 or not np.isfinite(reset_values[0]):
            return None
        if rates.size != days.size:
            raise ValueError(f"Unexpected tile-rate shape in {posterior_path}")
        reset_day = float(reset_values[0])
        before = days < reset_day
        after = days >= reset_day
        if not np.any(before) or not np.any(after):
            raise ValueError(f"Reset day does not split observations in {posterior_path}")
        pre_values.append(rates[before])
        post_values.append(rates[after])
        current_values.append(rates[int(np.argmax(days))])
        reset_days.append(reset_day)

    if not np.allclose(reset_days, reset_days[0], rtol=0.0, atol=1.0e-10):
        raise ValueError(f"Independent outputs use inconsistent reset days in {output_dir}")
    pre_values = np.concatenate(pre_values)
    post_values = np.concatenate(post_values)
    current_values = np.asarray(current_values, dtype=float)
    pre_values = pre_values[np.isfinite(pre_values)]
    post_values = post_values[np.isfinite(post_values)]
    current_values = current_values[np.isfinite(current_values)]
    if pre_values.size == 0 or post_values.size == 0 or current_values.size == 0:
        return None
    return {
        "detector": expected_sca,
        "pre_rates": pre_values,
        "post_rates": post_values,
        "current_rate": float(np.median(current_values)),
        "reset_day": reset_days[0],
    }


def load_all_rate_histograms(root, fit_suffix, maxent_only=False):
    histograms = {}
    reset_days = []
    for sca in SCAS:
        output_dirs = detector_output_dirs(root, sca, fit_suffix)
        loaded = next(
            (
                histogram
                for output_dir in output_dirs
                if not maxent_only or maxent_fit_set_complete(output_dir)
                if (histogram := load_rate_histogram(output_dir, sca)) is not None
            ),
            None,
        )
        histograms[sca] = loaded
        if loaded is not None:
            reset_days.append(loaded["reset_day"])
    if reset_days and not np.allclose(
        reset_days, reset_days[0], rtol=0.0, atol=1.0e-10
    ):
        raise ValueError(f"Production outputs use inconsistent reset days: {reset_days}")
    return histograms, (reset_days[0] if reset_days else None)


def latest_statistics_date(statistics_path):
    if statistics_path.is_file():
        paths = [statistics_path]
    elif statistics_path.is_dir():
        paths = sorted(statistics_path.glob("wfi*_tile_log_statistics_*.csv"))
    else:
        return "unknown"
    latest_date = ""
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                timestamp = str(row.get("exposure_start_time", ""))
                if len(timestamp) >= 10:
                    latest_date = max(latest_date, timestamp[:10])
    return latest_date or "unknown"


def draw_rate_histogram_mosaic(
    path,
    histograms,
    model_label,
    pass_label,
    date_label,
    reset_day,
):
    available = [sca for sca in SCAS if histograms[sca] is not None]
    if not available:
        raise ValueError("No completed detector histograms are available")
    all_values = np.concatenate(
        [
            np.concatenate((histograms[sca]["pre_rates"], histograms[sca]["post_rates"]))
            for sca in available
        ]
    )
    rate_min = float(np.min(all_values))
    rate_max = float(np.max(all_values))
    if np.isclose(rate_min, rate_max):
        half_width = max(abs(rate_min) * 0.05, 0.1)
        bins = np.linspace(rate_min - half_width, rate_max + half_width, 21)
    else:
        bins = np.histogram_bin_edges(all_values, bins=28)

    density_max = 0.0
    for sca in available:
        histogram = histograms[sca]
        for values in (histogram["pre_rates"], histogram["post_rates"]):
            density, _ = np.histogram(values, bins=bins, density=True)
            density_max = max(density_max, float(np.max(density)))
    density_max = max(1.0, density_max * 1.12)

    figure = plt.figure(figsize=(16, 10))
    x_min = -1.0
    x_max = SCA_PITCH * 6 - 1.0
    y_min = min(COLUMN_Y_OFFSET) - 1.0
    y_max = SCA_PITCH * 2 + max(COLUMN_Y_OFFSET) + SCA_SIZE + 1.0
    left, right = 0.055, 0.985
    bottom, top = 0.09, 0.78
    bottom_scas = {column[-1] for column in COLUMN_SCA}
    left_scas = {column[0] for column in COLUMN_SCA}

    for sca in SCAS:
        x, y = sca_position(sca)
        axis = figure.add_axes(
            (
                left + (x - x_min) / (x_max - x_min) * (right - left),
                bottom + (y - y_min) / (y_max - y_min) * (top - bottom),
                SCA_SIZE / (x_max - x_min) * (right - left),
                SCA_SIZE / (y_max - y_min) * (top - bottom),
            )
        )
        histogram = histograms[sca]
        if histogram is None:
            axis.set_facecolor("#e6e6e6")
            axis.patch.set_hatch("///")
            axis.patch.set_edgecolor("#333333")
            axis.text(
                0.5,
                0.5,
                f"{sca}\npending",
                ha="center",
                va="center",
                fontsize=8,
                color="#333333",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            continue

        axis.hist(
            histogram["pre_rates"],
            bins=bins,
            density=True,
            color="#0072B2",
            alpha=0.60,
        )
        axis.hist(
            histogram["post_rates"],
            bins=bins,
            density=True,
            color="#D55E00",
            alpha=0.55,
        )
        axis.axvline(histogram["current_rate"], color="black", linewidth=1.4)
        axis.text(
            0.96,
            0.94,
            f"now {histogram['current_rate']:.2f}",
            ha="right",
            va="top",
            fontsize=6.5,
            color="black",
            transform=axis.transAxes,
        )
        axis.set_xlim(rate_min, rate_max)
        axis.set_ylim(0.0, density_max)
        axis.set_title(sca, fontsize=8, pad=1.0)
        axis.tick_params(labelsize=6, length=2)
        if sca not in bottom_scas:
            axis.set_xticklabels([])
        else:
            axis.set_xlabel("nm/day", fontsize=6, labelpad=0.0)
        if sca not in left_scas:
            axis.set_yticklabels([])

    figure.legend(
        handles=(
            Patch(facecolor="#0072B2", alpha=0.60, label="Pre-reset"),
            Patch(facecolor="#D55E00", alpha=0.55, label="Post-reset"),
            Line2D([0], [0], color="black", linewidth=1.8, label="Latest-epoch fitted median"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        "Roman WFI focal-plane instantaneous ice-rate distributions\n"
        f"Pre/post reset; reset day {reset_day:.3f}; data through {date_label} UTC\n"
        f"{model_label}; {pass_label}; {len(available)}/18 detectors complete",
        fontsize=14,
        y=0.985,
    )
    figure.text(
        0.015,
        0.43,
        "Probability density",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9,
    )
    figure.text(
        0.5,
        0.025,
        "Instantaneous ice accumulation rate (nm/day)",
        ha="center",
        fontsize=9,
    )
    figure.text(
        0.985,
        0.012,
        f"Data date: {date_label} UTC",
        ha="right",
        fontsize=8,
        color="0.3",
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def latest_pass_label(statistics_path):
    if statistics_path.is_file():
        paths = [statistics_path]
    elif statistics_path.is_dir():
        paths = sorted(statistics_path.glob("wfi*_tile_log_statistics_*.csv"))
    else:
        return "Pass unavailable"
    latest_repeat = -1
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                latest_repeat = max(latest_repeat, int(row["repeat"]))
    if latest_repeat < 0:
        return "Pass unavailable"
    return f"Pass {latest_repeat + 1} (repeat index {latest_repeat})"


def draw_rate_mosaic(
    path,
    maps,
    model_label,
    pass_label,
    local_contrast=False,
    metrics=METRICS,
    title_prefix="Roman WFI full focal-plane ice accumulation rates",
):
    available = [sca for sca in SCAS if maps[sca] is not None]
    figure, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(6 * len(metrics), 6),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[0]
    if not local_contrast:
        all_values = np.concatenate(
            [
                maps[sca][field][np.isfinite(maps[sca][field])]
                for sca in available
                for _, field in metrics
            ]
        )
        global_min = float(np.min(all_values))
        global_max = float(np.max(all_values))
        if global_min == global_max:
            global_max = global_min + 1.0

    for axis, (title, field) in zip(axes, metrics):
        plotted = {}
        if local_contrast:
            for sca in available:
                values = maps[sca][field]
                finite = values[np.isfinite(values)]
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))
                plotted[sca] = np.zeros_like(values) + 0.5 if vmin == vmax else (values - vmin) / (vmax - vmin)
            color_min, color_max = 0.0, 1.0
        else:
            color_min = global_min
            color_max = global_max
            plotted = {sca: maps[sca][field] for sca in available}
        image = None
        for sca in SCAS:
            x, y = sca_position(sca)
            if maps[sca] is None:
                axis.add_patch(
                    Rectangle(
                        (x, y),
                        SCA_SIZE,
                        SCA_SIZE,
                        facecolor="#e6e6e6",
                        edgecolor="#333333",
                        hatch="///",
                        linewidth=1.2,
                    )
                )
                axis.text(
                    x + SCA_SIZE / 2,
                    y + SCA_SIZE / 2,
                    f"{sca}\npending",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#333333",
                )
                continue
            image = axis.imshow(
                plotted[sca],
                origin="lower",
                interpolation="none",
                cmap="viridis",
                vmin=color_min,
                vmax=color_max,
                extent=(x, x + SCA_SIZE, y, y + SCA_SIZE),
            )
            axis.add_patch(
                Rectangle(
                    (x, y),
                    SCA_SIZE,
                    SCA_SIZE,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.0,
                )
            )
            axis.text(
                x + SCA_SIZE / 2,
                y + SCA_SIZE / 2,
                sca,
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                path_effects=None,
            )
            for tile in range(1, 8):
                axis.plot(
                    (x + tile, x + tile),
                    (y, y + SCA_SIZE),
                    color="white",
                    alpha=0.35,
                    linewidth=0.35,
                )
                axis.plot(
                    (x, x + SCA_SIZE),
                    (y + tile, y + tile),
                    color="white",
                    alpha=0.35,
                    linewidth=0.35,
                )
        if image is not None:
            colorbar_label = (
                "Within-SCA relative rate (min to max)"
                if local_contrast
                else "nm/day"
            )
            figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02, label=colorbar_label)
        axis.set_aspect("equal")
        axis.set_xlim(-1.0, SCA_PITCH * 6 - 1.0)
        min_y = min(COLUMN_Y_OFFSET) - 1.0
        max_y = SCA_PITCH * 2 + max(COLUMN_Y_OFFSET) + SCA_SIZE + 1.0
        axis.set_ylim(min_y, max_y)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            f"{title} (within-SCA contrast)"
            if local_contrast
            else f"{title} (nm/day)"
        )
        axis.set_xlabel("Roman WFI focal-plane mosaic")
        center_x = SCA_PITCH * 3 - 0.5
        center_y = SCA_PITCH * 2 + COLUMN_Y_OFFSET[2] + 1.2
        axis.plot(
            center_x,
            center_y,
            marker="x",
            color="#d62728",
            markersize=7,
            markeredgewidth=1.5,
        )
        axis.text(
            center_x + 0.3,
            center_y + 0.5,
            "WFI_CEN",
            color="#d62728",
            fontsize=7,
            va="bottom",
        )

    figure.suptitle(
        f"{title_prefix}\n"
        f"{model_label}; {pass_label}; {len(available)}/18 SCA rate fields available"
        + ("; each SCA independently scaled" if local_contrast else ""),
        fontsize=14,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--fit-suffix",
        default=DEFAULT_FIT_SUFFIX,
        help="Use 'production' for the fixed input-directory layout or a legacy suffix.",
    )
    parser.add_argument(
        "--tile-statistics",
        type=Path,
        default=None,
        help="Statistics CSV or directory used to derive the latest pass label.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--maxent-only",
        action="store_true",
        help="Use only complete per-tile MaxEnt production archives; reject stale aggregates.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or (
        args.root / DEFAULT_FIT_SUFFIX / "wfi_full_focal_plane_rate_maps.png"
    )
    maps = load_all_maps(args.root, args.fit_suffix, args.maxent_only)
    reset_maps, reset_day = load_all_reset_maps(
        args.root,
        args.fit_suffix,
        args.maxent_only,
    )
    trend_maps, trend_day = load_all_trend_maps(
        args.root,
        args.fit_suffix,
        args.maxent_only,
    )
    statistics_path = args.tile_statistics
    if statistics_path is None:
        statistics_path = args.root
    elif not statistics_path.is_absolute():
        statistics_path = args.root / statistics_path
    pass_label = latest_pass_label(statistics_path)
    date_label = latest_statistics_date(statistics_path)
    if args.fit_suffix == DEFAULT_FIT_SUFFIX:
        model_label = (
            "independent per-tile MaxEnt MAP fits"
            if args.maxent_only
            else "independent per-tile non-parametric MAP fits"
        )
    elif "production_sharedAR_tileRate_twoCommonACM" in args.fit_suffix:
        model_label = (
            "constant-rate modulation; tile-specific rate scales; "
            "shared AR and common ACM steps"
        )
    elif "np_ledacm" in args.fit_suffix:
        model_label = "square-root + NP modulation"
    else:
        model_label = "fully nonparametric trajectory"
    draw_rate_mosaic(output, maps, model_label, pass_label)
    contrast_output = output.with_name(output.stem + "_tile_contrast.png")
    draw_rate_mosaic(
        contrast_output,
        maps,
        model_label,
        pass_label,
        local_contrast=True,
    )
    histogram_maps, histogram_reset_day = load_all_rate_histograms(
        args.root,
        args.fit_suffix,
        args.maxent_only,
    )
    if histogram_reset_day is not None:
        histogram_output = output.with_name(
            f"wfi_full_focal_plane_rate_histograms_{date_label}.png"
        )
        draw_rate_histogram_mosaic(
            histogram_output,
            histogram_maps,
            model_label,
            pass_label,
            date_label,
            histogram_reset_day,
        )
        print(f"Wrote {histogram_output}")
    else:
        print("No complete detector rate histograms found")
    if reset_day is not None:
        reset_output = output.with_name(output.stem + "_pre_post_reset.png")
        draw_rate_mosaic(
            reset_output,
            reset_maps,
            model_label,
            pass_label,
            metrics=RESET_METRICS,
            title_prefix=(
                "Roman WFI focal-plane time-mean ice accumulation rate by reset epoch\n"
                f"Reset day {reset_day:.9f}"
            ),
        )
        reset_contrast_output = reset_output.with_name(
            reset_output.stem + "_tile_contrast.png"
        )
        draw_rate_mosaic(
            reset_contrast_output,
            reset_maps,
            model_label,
            pass_label,
            local_contrast=True,
            metrics=RESET_METRICS,
            title_prefix=(
                "Roman WFI focal-plane time-mean ice accumulation rate by reset epoch\n"
                f"Reset day {reset_day:.9f}"
            ),
        )
        print(f"Wrote {reset_output}")
        print(f"Wrote {reset_contrast_output}")
    else:
        print("No reset-aware posterior maps found")
    if trend_day is not None:
        trend_output = output.with_name(
            "wfi_full_focal_plane_trend_rates_pre_post.png"
        )
        draw_rate_mosaic(
            trend_output,
            trend_maps,
            model_label,
            pass_label,
            metrics=TREND_METRICS,
            title_prefix=(
                "Roman WFI focal-plane fitted linear trend rate by reset epoch\n"
                f"Reset day {trend_day:.9f}"
            ),
        )
        trend_contrast_output = trend_output.with_name(
            trend_output.stem + "_tile_contrast.png"
        )
        draw_rate_mosaic(
            trend_contrast_output,
            trend_maps,
            model_label,
            pass_label,
            local_contrast=True,
            metrics=TREND_METRICS,
            title_prefix=(
                "Roman WFI focal-plane fitted linear trend rate by reset epoch\n"
                f"Reset day {trend_day:.9f}"
            ),
        )
        print(f"Wrote {trend_output}")
        print(f"Wrote {trend_contrast_output}")
    else:
        print("No per-tile trend-rate maps found")
    available = [sca for sca in SCAS if maps[sca] is not None]
    pending = [sca for sca in SCAS if maps[sca] is None]
    print(f"Wrote {output}")
    print(f"Wrote {contrast_output}")
    print(f"Available SCAs: {', '.join(available)}")
    if pending:
        print(f"Pending or invalid SCAs: {', '.join(pending)}")


if __name__ == "__main__":
    main()
