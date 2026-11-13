"""Clean complex reflection: all detectors and tile percentile envelopes."""

import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import fit_car201_zane_star_reflectance_modulation as fitter
import mock_fit_wfi_star_reflectance_modulation as mock
from heatmap_car201_allstar_model_correction import DETECTORS, PRODUCTION_ROOT


def reciprocal(values):
    values = np.asarray(values, dtype=float)
    return np.divide(1000., values, out=np.full(values.shape, np.nan), where=values > 0)


def phase_aligned(reflections):
    phase = np.unwrap(np.angle(reflections[:, ::-1]), axis=1)[:, ::-1]
    # All tiles and both variants use the branch nearest zero at the long-wave
    # endpoint; unwrap towards shorter wavelengths before computing quantiles.
    return phase - 2 * np.pi * np.round(phase[:, -1, None] / (2 * np.pi))


def main():
    wave = np.arange(470., 2321., 2.)
    base_n, base_k = fitter.production.hgcdte_reference_optical_constants(wave)
    spectral_basis = fitter.production.spectral_modulation_basis(wave)
    colors = plt.get_cmap("tab20")(np.arange(len(DETECTORS)))
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    x_order = np.argsort(1000 / wave)
    x = 1000 / wave[x_order]
    records = []
    all_base, all_modulated = [], []
    for detector, color in zip(DETECTORS, colors, strict=True):
        fit_dir = (PRODUCTION_ROOT / detector.lower() / "spline_relative_response_normal/experiments"
                   / f"car201_zane_{detector.lower()}_ep3v2_ep4_faint_v2_tile_matched")
        summary = json.loads((fit_dir / "car201_actual_star_modulation_fit_summary.json").read_text())
        optical_root = fitter.Path(summary["ice_history"]["optical_state_source"])
        basis, *_ = mock.modulation_basis(wave, grouping=summary["modulation"]["basis"])
        params = pd.read_csv(fit_dir / "car201_actual_star_modulation_parameters.csv").set_index("mode")
        params = params.loc[summary["modulation"]["mode_names"]]
        ratio = np.exp(params.fit_log_amplitude.to_numpy() @ basis
                       + 1j * (params.fit_phase_rad.to_numpy() @ basis))
        reflections = []
        for tile_id in range(256):
            source = mock.source_fit_from_cache(optical_root / f"tile_{tile_id:03d}/relative_response.npz",
                                                detector, tile_id)
            state = fitter.complex_model.load_state(source, wave, base_n, base_k, spectral_basis)
            first, second, incident_y, _, _ = mock.clean_stack_boundary(state, wave)
            reflections.append((incident_y * first - second) / (incident_y * first + second))
        baseline = np.asarray(reflections)
        modulated = baseline * ratio[None, :]
        all_base.append(baseline)
        all_modulated.append(modulated)
        for variant, curves, style in (("unmodulated", baseline, "--"), ("modulated", modulated, "-")):
            amplitude = np.percentile(np.abs(curves), [16, 50, 84], axis=0)
            phase = np.percentile(phase_aligned(curves), [16, 50, 84], axis=0)
            for axis, stats in zip(axes, (amplitude, phase), strict=True):
                axis.fill_between(x, stats[0, x_order], stats[2, x_order], color=color,
                                  alpha=0.055 if variant == "unmodulated" else 0.11, linewidth=0)
                axis.plot(x, stats[1, x_order], color=color, linestyle=style,
                          linewidth=0.9 if variant == "unmodulated" else 1.4,
                          alpha=0.7 if variant == "unmodulated" else 1.)
            for i, wavelength in enumerate(wave):
                records.append({"detector": detector, "variant": variant, "wavelength_nm": wavelength,
                                **{f"amplitude_p{q}": amplitude[j, i] for j, q in enumerate((16, 50, 84))},
                                **{f"phase_p{q}_rad": phase[j, i] for j, q in enumerate((16, 50, 84))}})
        print(f"Computed {detector}: 256 tiles", flush=True)
    for axis in axes:
        for led in mock.LED_CENTERS_NM:
            axis.axvline(1000 / led, color="0.55", linestyle=":", linewidth=0.7)
        axis.set_xlim(x[0], x[-1])
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(r"$|r(\lambda)|$")
    axes[1].set_ylabel("Unwrapped reflection phase (rad)")
    axes[1].set_xlabel(r"Reciprocal wavelength, $1/\lambda$ ($\mu$m$^{-1}$)")
    top = axes[0].secondary_xaxis("top", functions=(reciprocal, reciprocal))
    top.set_xticks([2300, 1700, 1300, 1000, 800, 600, 500])
    top.set_xlabel("Wavelength (nm)")
    handles = [Line2D([0], [0], color=c, label=d) for d, c in zip(DETECTORS, colors, strict=True)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=9, fontsize=9)
    fig.suptitle("Clean complex reflection across all WFI detectors — no ice coating", y=0.99, fontsize=15)
    fig.text(0.5, 0.025, "Solid: modulated median; dashed: unmodulated median. Shading: tile 16th–84th percentiles (256 tiles per detector).\n"
             "Phase branches aligned at the longest wavelength; spread describes tile variation, not fit uncertainty.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.08, 0.995, 0.86))
    output = PRODUCTION_ROOT / "wavelength_response_bundle/car201_all_detector_clean_complex_reflection.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    pd.DataFrame(records).to_csv(output.with_suffix(".csv"), index=False)
    np.savez_compressed(output.with_suffix(".npz"), wavelengths_nm=wave,
                        detectors=np.asarray(DETECTORS), baseline=np.asarray(all_base),
                        modulated=np.asarray(all_modulated))
    print(output)


if __name__ == "__main__":
    main()
