"""Overlay tile observations and tile-level TMM curves for all WFIs by LED."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from plot_wfi_tmm_numpyro import _load_prediction_inputs, apply_early_time_data_correction


ROOT = Path(__file__).resolve().parent
SCAS = tuple(f"WFI{index:02d}" for index in range(1, 19))
DEFAULT_FIT_SUFFIX = (
    "production_sharedAR_tileRate_twoCommonACM_"
    "map2000_warm100_sample500_depth4_5nm"
)


def posterior_median_model_and_corrected_data(output_dir, statistics_path, detector):
    """Evaluate the likelihood model at posterior medians for every tile."""
    (
        data,
        context,
        ice_thickness,
        ar_parameters,
        acm_step,
        hgcdte_parameters,
        fit_hgcdte,
        conductive_parameters,
        fit_conductive_layer,
        qe_rbf_parameters,
        fit_qe_rbf,
        jax,
        jnp,
        build_prediction_function,
        early_time_scale,
        early_time_cutoff_day,
        late_acm_step,
        additional_acm_step,
        detector_thickness,
    ) = _load_prediction_inputs(
        output_dir,
        statistics_path,
        Path("unused-model-path"),
        "core",
        1,
        5.0,
        64,
    )
    if data["n_scas"] != 1 or data["sca_names"].astype(str).tolist() != [detector]:
        raise ValueError(f"Expected one {detector} SCA in {output_dir}")

    ice_median = np.median(ice_thickness, axis=0)
    ar_median = np.median(ar_parameters, axis=0)
    acm_median = np.median(acm_step, axis=0)
    hgcdte_median = (
        None
        if hgcdte_parameters is None
        else np.median(hgcdte_parameters, axis=0)
    )
    conductive_median = (
        None
        if conductive_parameters is None
        else np.median(conductive_parameters, axis=0)
    )
    qe_median = (
        None if qe_rbf_parameters is None else np.median(qe_rbf_parameters, axis=0)
    )
    late_median = (
        None if late_acm_step is None else np.median(late_acm_step, axis=0)
    )
    additional_median = (
        None
        if additional_acm_step is None
        else np.median(additional_acm_step, axis=0)
    )
    detector_thickness_median = (
        None
        if detector_thickness is None
        else np.median(detector_thickness, axis=0)
    )
    predictor = jax.jit(
        build_prediction_function(
            data,
            context,
            row_indices=np.arange(data["counts"].shape[0]),
            fit_hgcdte=fit_hgcdte,
            fit_hgcdte_modulation=bool(context["fit_hgcdte_modulation"]),
            fit_conductive_layer=fit_conductive_layer,
            finite_hgcdte_detector=context["finite_hgcdte_detector"],
            detector_backside_index=context["detector_backside_index"],
            fit_qe_rbf=fit_qe_rbf,
            qe_rbf_node_wavelengths_nm=context["qe_rbf_node_wavelengths_nm"],
            qe_rbf_width_nm=context["qe_rbf_width_nm"],
            use_asw_optical_constants=context["use_asw_optical_constants"],
            use_hgcdte_optical_constants=context["use_hgcdte_optical_constants"],
            incidence_cone_half_angle_deg=context["incidence_cone_half_angle_deg"],
            incidence_pupil_half_angle_deg=context["incidence_pupil_half_angle_deg"],
            incidence_chief_ray_tilts_deg=context["incidence_chief_ray_tilts_deg"],
            share_ar_across_rows=bool(context["shared_ar_parameters_across_tiles"]),
            fit_late_acm_step=late_acm_step is not None,
            fit_additional_acm_step=additional_acm_step is not None,
        )
    )
    prediction = np.asarray(
        predictor(
            jnp.asarray(ice_median),
            jnp.asarray(ar_median),
            jnp.asarray(acm_median),
            None if hgcdte_median is None else jnp.asarray(hgcdte_median),
            None if conductive_median is None else jnp.asarray(conductive_median),
            None if qe_median is None else jnp.asarray(qe_median),
            None,
            None if late_median is None else jnp.asarray(late_median),
            None
            if detector_thickness_median is None
            else jnp.asarray(detector_thickness_median),
            None,
            None
            if additional_median is None
            else jnp.asarray(additional_median),
        )
    )
    corrected_data, _ = apply_early_time_data_correction(
        data, prediction, early_time_scale, early_time_cutoff_day
    )
    observed = corrected_data["sums"] / corrected_data["counts"]
    return data, observed, prediction


def plot_all_detector_led_fits(path, detector_data):
    figure, axes = plt.subplots(
        4, 3, figsize=(18, 13), sharey=False, constrained_layout=False
    )
    colors = plt.get_cmap("tab20", len(SCAS))
    tile_model_count = sum(
        len(detector_data[detector][0]["tile_ids"]) for detector in SCAS
    )

    for led in range(1, 7):
        row, column = divmod(led - 1, 3)
        fit_axis = axes[row, column]
        residual_axis = axes[row + 2, column]
        for detector_index, detector in enumerate(SCAS):
            data, observed, prediction = detector_data[detector]
            color = colors(detector_index)
            days = np.asarray(data["days"], dtype=float)
            groups = np.flatnonzero(np.asarray(data["led"]) == led)
            order = groups[np.argsort(days[groups], kind="stable")]
            x = days[order]
            observed_percent = 100.0 * np.expm1(observed[:, order])
            model_percent = 100.0 * np.expm1(prediction[:, order])
            if model_percent.shape[0] != len(data["tile_ids"]):
                raise ValueError(
                    f"{detector} has {model_percent.shape[0]} model rows for "
                    f"{len(data['tile_ids'])} tile IDs"
                )
            fit_axis.scatter(
                np.broadcast_to(x, observed_percent.shape).ravel(),
                observed_percent.ravel(),
                color=color,
                alpha=0.09,
                s=5,
                linewidths=0,
            )
            model_segments = np.stack(
                (np.broadcast_to(x, model_percent.shape), model_percent), axis=-1
            )
            fit_axis.add_collection(
                LineCollection(
                    model_segments,
                    colors=[color],
                    linewidths=0.55,
                    alpha=0.22,
                    zorder=2,
                )
            )

            residual_percent = observed_percent - model_percent
            residual_axis.scatter(
                np.broadcast_to(x, residual_percent.shape).ravel(),
                residual_percent.ravel(),
                color=color,
                alpha=0.09,
                s=5,
                linewidths=0,
            )
            residual_segments = np.stack(
                (np.broadcast_to(x, residual_percent.shape), residual_percent),
                axis=-1,
            )
            residual_axis.add_collection(
                LineCollection(
                    residual_segments,
                    colors=[color],
                    linewidths=0.4,
                    alpha=0.18,
                    zorder=2,
                )
            )

        fit_axis.set_title(f"LED {led}")
        fit_axis.tick_params(labelbottom=False)
        fit_axis.grid(alpha=0.2)
        residual_axis.set_title(f"LED {led} residuals", fontsize=9)
        residual_axis.axhline(0.0, color="black", linewidth=0.7)
        residual_axis.set_xlabel("Days since first exposure")
        residual_axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Relative change (%)")
    axes[1, 0].set_ylabel("Relative change (%)")
    axes[2, 0].set_ylabel("Data − model (%)")
    axes[3, 0].set_ylabel("Data − model (%)")
    detector_handles = [
        Line2D([], [], color=colors(i), linewidth=2.0, label=sca)
        for i, sca in enumerate(SCAS)
    ]
    style_handles = [
        Line2D(
            [],
            [],
            color="#444444",
            marker="o",
            linestyle="",
            markersize=4,
            label="Tile observations",
        ),
        Line2D(
            [],
            [],
            color="#444444",
            linewidth=1.0,
            alpha=0.55,
            label=f"All tile-level TMM models ({tile_model_count} per LED)",
        ),
        Line2D(
            [],
            [],
            color="#444444",
            linewidth=1.0,
            alpha=0.5,
            label="Every tile's data − model residual",
        ),
    ]
    figure.legend(
        handles=detector_handles + style_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        f"WFI01–WFI18: {tile_model_count} tile-level TMM curves per LED, observations above and residuals below\n"
        "Residual = observed relative change (%) − model relative change (%)",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.13, 1.0, 0.91))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--statistics-root", type=Path, default=None)
    parser.add_argument("--fit-suffix", default=DEFAULT_FIT_SUFFIX)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    statistics_root = args.statistics_root or args.root / "wfi_tile_log_statistics_512"
    if not statistics_root.is_absolute():
        statistics_root = args.root / statistics_root
    detector_data = {}
    for detector in SCAS:
        output_dir = args.root / f"{detector.lower()}_{args.fit_suffix}"
        statistics_path = statistics_root / f"{detector.lower()}_tile_log_statistics_512.csv"
        if not (output_dir / "posterior_samples.npz").is_file():
            raise FileNotFoundError(output_dir / "posterior_samples.npz")
        detector_data[detector] = posterior_median_model_and_corrected_data(
            output_dir, statistics_path, detector
        )
        print(
            f"Loaded {detector} ({len(detector_data[detector][0]['tile_ids'])} tiles)",
            flush=True,
        )
    output = args.output or args.root / "wfi_18_detector_led_tile_fits.png"
    plot_all_detector_led_fits(output, detector_data)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
