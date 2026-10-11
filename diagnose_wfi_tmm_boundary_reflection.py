"""Diagnose the physical AR/HgCdTe reflection implied by a production MAP."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import fit_wfi_tmm_production as production


ROOT = Path(__file__).resolve().parent
DEFAULT_FIT_DIR = ROOT / "wfi01_joint_complex_q_cold_hgcdte_prior025_map10000"
DEFAULT_STATISTICS = (
    ROOT / "rrn_reduction_20261005/wfi01_tile_log_statistics_512_count_corrected.csv"
)
DEFAULT_REFERENCE_DATA_DIR = ROOT / "external_reference_data"
DEFAULT_OUTPUT_DIR = DEFAULT_FIT_DIR / "boundary_reflection_diagnostic"


def load_map_state(path):
    with np.load(Path(path) / "posterior_samples.npz", allow_pickle=False) as archive:
        required = {
            "ar_parameters",
            "absolute_hgcdte_n_log_scale",
            "absolute_hgcdte_k_log_scale",
            "incidence_angle_deg",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"MAP archive is missing required arrays: {sorted(missing)}")
        ar_parameters = np.asarray(archive["ar_parameters"], dtype=float)
        n_log_scale = np.asarray(
            archive["absolute_hgcdte_n_log_scale"], dtype=float
        )
        k_log_scale = np.asarray(
            archive["absolute_hgcdte_k_log_scale"], dtype=float
        )
        incidence_angle = np.asarray(archive["incidence_angle_deg"], dtype=float)
        wavelengths = np.asarray(archive["response_wavelengths_nm"], dtype=float)
        metadata = {
            key: np.asarray(archive[key])
            for key in (
                "sed_mode",
                "sed_weighting",
                "absolute_substrate_power",
                "use_hgcdte_optical_constants",
                "incidence_pupil_half_angle_deg",
                "incidence_cone_half_angle_deg",
                "fit_incidence_angle",
            )
            if key in archive.files
        }
    if ar_parameters.ndim != 3 or ar_parameters.shape[1] != 1:
        raise ValueError("This diagnostic expects one SCA and one selected tile")
    if n_log_scale.size != 1 or k_log_scale.size != 1:
        raise ValueError("This diagnostic expects one shared HgCdTe scale pair")
    return {
        "ar_parameters": np.median(ar_parameters, axis=0),
        "hgcdte_parameters": np.asarray(
            [np.median(n_log_scale), np.median(k_log_scale)], dtype=float
        ),
        "incidence_angle_deg": float(np.median(incidence_angle)),
        "wavelengths": wavelengths,
        "metadata": metadata,
    }


def build_boundary_diagnostic(fit_dir, statistics_path, reference_data_dir):
    state = load_map_state(fit_dir)
    jax, *_ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    production.SED_URL = str(
        Path(reference_data_dir) / "WFI_RCS_LEDP_spectral_energy_distribution.ecsv"
    )
    production.QE_URL = str(Path(reference_data_dir) / "WFI_Quantum_efficiency.ecsv")
    data = production.load_tile_statistics(statistics_path, max_tiles=1)
    context = production.load_response_context(
        model_path=None,
        sed_mode="core",
        wavelength_stride=1,
        sca_names=data["sca_names"],
        sed_weighting="energy",
        absolute_substrate_power=True,
        use_hgcdte_optical_constants=True,
    )
    if not np.allclose(state["wavelengths"], context["wavelengths"], rtol=0.0, atol=1.0e-10):
        raise ValueError("MAP wavelengths do not match the reconstructed response grid")
    predictor = production.build_prediction_function(
        data,
        context,
        row_indices=np.asarray([0], dtype=int),
        fit_hgcdte=False,
        fit_conductive_layer=False,
        finite_hgcdte_detector=False,
        fit_qe_rbf=False,
        use_hgcdte_optical_constants=True,
        incidence_cone_half_angle_deg=float(
            np.asarray(state["metadata"].get("incidence_cone_half_angle_deg", 0.0))
        ),
        incidence_pupil_half_angle_deg=float(
            np.asarray(state["metadata"].get("incidence_pupil_half_angle_deg", 0.0))
        ),
        incidence_chief_ray_tilts_deg=np.zeros((1, 2), dtype=float),
        fit_incidence_angle=bool(
            np.asarray(state["metadata"].get("fit_incidence_angle", 1)).item()
        ),
        absolute_substrate_power=True,
        fit_complex_q=False,
    )
    result = predictor.boundary_reflection(
        state["ar_parameters"],
        hgcdte_parameters=state["hgcdte_parameters"],
        incidence_angle_deg=state["incidence_angle_deg"],
    )
    return state, context, result


def complex_fields(result, polarization):
    values = result[polarization]
    fields = {}
    for name in (
        "reflection",
        "load_admittance",
        "bottom_reflection",
        "top_reflection",
        "round_trip_reflection",
    ):
        fields[name] = np.asarray(values[name])[0, 0]
    return fields


def write_wavelength_table(output_dir, wavelengths, result):
    path = output_dir / "boundary_reflection_by_wavelength.csv"
    fields = ["wavelength_nm"]
    for polarization in ("s", "p"):
        for name in (
            "reflection",
            "load_admittance",
            "bottom_reflection",
            "top_reflection",
            "round_trip_reflection",
        ):
            fields.extend(
                f"{polarization}_{name}_{component}"
                for component in ("real", "imag", "magnitude", "phase_rad", "phase_unwrapped_rad")
            )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        rows = {polarization: complex_fields(result, polarization) for polarization in ("s", "p")}
        for index, wavelength in enumerate(wavelengths):
            row = {"wavelength_nm": float(wavelength)}
            for polarization, fields_by_name in rows.items():
                for name, values in fields_by_name.items():
                    phase = np.angle(values)
                    unwrapped = np.unwrap(phase)
                    row.update(
                        {
                            f"{polarization}_{name}_real": float(np.real(values[index])),
                            f"{polarization}_{name}_imag": float(np.imag(values[index])),
                            f"{polarization}_{name}_magnitude": float(abs(values[index])),
                            f"{polarization}_{name}_phase_rad": float(phase[index]),
                            f"{polarization}_{name}_phase_unwrapped_rad": float(
                                unwrapped[index]
                            ),
                        }
                    )
            writer.writerow(row)
    return path


def write_led_summary(output_dir, context, result):
    weights = np.asarray(context["model_weights"], dtype=float)[0]
    path = output_dir / "boundary_reflection_led_summary.csv"
    fields = [
        "led",
        "polarization",
        "weighted_reflection_kind",
        "reflection_real",
        "reflection_imag",
        "reflection_magnitude",
        "reflection_phase_rad",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for led_index, led_weights in enumerate(weights, start=1):
            positive = np.asarray(led_weights) > 0.0
            if not np.any(positive):
                raise ValueError(f"LED {led_index} has no positive spectral weights")
            normalized = led_weights[positive] / np.sum(led_weights[positive])
            for polarization in ("s", "p"):
                fields_by_name = complex_fields(result, polarization)
                for name in ("bottom_reflection", "top_reflection", "round_trip_reflection"):
                    weighted = np.sum(
                        normalized * fields_by_name[name][positive]
                    )
                    writer.writerow(
                        {
                            "led": led_index,
                            "polarization": polarization,
                            "weighted_reflection_kind": name,
                            "reflection_real": float(np.real(weighted)),
                            "reflection_imag": float(np.imag(weighted)),
                            "reflection_magnitude": float(abs(weighted)),
                            "reflection_phase_rad": float(np.angle(weighted)),
                        }
                    )
    return path


def plot_boundary_reflection(output_dir, wavelengths, result):
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for polarization, color in (("s", "tab:blue"), ("p", "tab:orange")):
        fields = complex_fields(result, polarization)
        bottom_phase = np.unwrap(np.angle(fields["bottom_reflection"]))
        round_trip_phase = np.unwrap(np.angle(fields["round_trip_reflection"]))
        axes[0, 0].plot(wavelengths, bottom_phase, color=color, label=f"{polarization} polarization")
        axes[0, 1].plot(
            wavelengths,
            np.abs(fields["bottom_reflection"]),
            color=color,
            label=f"{polarization} polarization",
        )
        axes[1, 0].plot(wavelengths, round_trip_phase, color=color, label=f"{polarization} polarization")
        axes[1, 1].plot(
            wavelengths,
            np.abs(fields["round_trip_reflection"]),
            color=color,
            label=f"{polarization} polarization",
        )
    axes[0, 0].set_title("Physical bottom-boundary phase")
    axes[0, 0].set_ylabel("arg(r_bottom), unwrapped (rad)")
    axes[0, 1].set_title("Physical bottom-boundary magnitude")
    axes[0, 1].set_ylabel("|r_bottom|")
    axes[1, 0].set_title("Top-to-bottom round-trip phase")
    axes[1, 0].set_ylabel("arg(r_top r_bottom), unwrapped (rad)")
    axes[1, 1].set_title("Top-to-bottom round-trip magnitude")
    axes[1, 1].set_ylabel("|r_top r_bottom|")
    for axis in axes.flat:
        axis.set_xlabel("Wavelength (nm)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    path = output_dir / "boundary_reflection_phase.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_summary(output_dir, fit_dir, state, result):
    lines = [
        "# Physical Boundary Reflection Diagnostic",
        "",
        f"- MAP source: `{Path(fit_dir).resolve()}`.",
        "- Boundary vectors are from the production AR/HgCdTe TMM; no complex-q factor is applied.",
        f"- Response wavelengths: `{len(state['wavelengths'])}` from `{state['wavelengths'][0]:.9g}` to `{state['wavelengths'][-1]:.9g}` nm.",
        f"- Fitted chief-ray incidence angle: `{state['incidence_angle_deg']:.9g}` degrees.",
        f"- HgCdTe log scales: n `{state['hgcdte_parameters'][0]:.9g}`, k `{state['hgcdte_parameters'][1]:.9g}`.",
        "",
        "## Boundary Quantities",
        "",
        "- `reflection`: full AR/HgCdTe reflection seen from the incident medium.",
        "- `load_admittance`: `C/B` at the ice/AR boundary.",
        "- `bottom_reflection`: `(Y_ice - C/B) / (Y_ice + C/B)`.",
        "- `top_reflection`: reflection from ice toward the incident medium.",
        "- `round_trip_reflection`: `top_reflection * bottom_reflection`.",
    ]
    for polarization in ("s", "p"):
        fields = complex_fields(result, polarization)
        bottom_phase = np.unwrap(np.angle(fields["bottom_reflection"]))
        lines.extend(
            [
                "",
                f"### {polarization.upper()} Polarization",
                "",
                f"- `|r_bottom|` range: `{np.min(abs(fields['bottom_reflection'])):.9g}` to `{np.max(abs(fields['bottom_reflection'])):.9g}`.",
                f"- Unwrapped `arg(r_bottom)` range: `{bottom_phase[0]:.9g}` to `{bottom_phase[-1]:.9g}` rad.",
                f"- `|r_top r_bottom|` range: `{np.min(abs(fields['round_trip_reflection'])):.9g}` to `{np.max(abs(fields['round_trip_reflection'])):.9g}`.",
            ]
        )
    (output_dir / "boundary_reflection_summary.md").write_text("\n".join(lines) + "\n")


def run_diagnostic(fit_dir, statistics_path, reference_data_dir, output_dir):
    fit_dir = Path(fit_dir)
    statistics_path = Path(statistics_path)
    reference_data_dir = Path(reference_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state, context, result = build_boundary_diagnostic(
        fit_dir, statistics_path, reference_data_dir
    )
    wavelengths = np.asarray(result["wavelengths"], dtype=float)
    if not np.allclose(wavelengths, state["wavelengths"], rtol=0.0, atol=1.0e-10):
        raise ValueError("Diagnostic output wavelengths do not match the MAP archive")
    paths = [
        write_wavelength_table(output_dir, wavelengths, result),
        write_led_summary(output_dir, context, result),
        plot_boundary_reflection(output_dir, wavelengths, result),
    ]
    write_summary(output_dir, fit_dir, state, result)
    paths.append(output_dir / "boundary_reflection_summary.md")
    for path in paths:
        print(f"Wrote {path}")
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT_DIR)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--reference-data-dir", type=Path, default=DEFAULT_REFERENCE_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    run_diagnostic(
        args.fit_dir,
        args.statistics,
        args.reference_data_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
