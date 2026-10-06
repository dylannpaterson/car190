"""Fixed zero-argument production TMM fit.

This file is the production copy of the TMM model.  Its settings and input
selection are intentionally fixed below; it does not expose the exploratory
model CLI.
"""

import argparse
import csv
import hashlib
import io
import shlex
import sys
import time
from pathlib import Path

import numpy as np

from analyze_wfi01_all_pixel_soc_model import parse_mast_time
from assess_rcs_spectral_identifiability import (
    QE_URL,
    SED_URL,
    load_qe,
    load_sed,
    read_source,
)
from fit_wfi01_joint_complex_q_soc import prepare_response_integration


ROOT = Path(__file__).resolve().parent
PRODUCTION_INPUT_ROOT = ROOT / "rrn_reduction_20261002"
PRODUCTION_OUTPUT_ROOT = ROOT / "wfi01_warm500_fixed_ar_ordered_hgcdte_free_trajectory_n125_per_tile_hgcdte"
PRODUCTION_STATISTICS_PATTERN = "wfi01_tile_log_statistics_512_count_corrected.csv"
PRODUCTION_RESET_DAY = 15.064405023148149
PRODUCTION_SEED = 20260922
WARMUP_FIXED_RATE_NM_DAY = 10.0
WARMUP_MAP_STEPS = 500
WARMUP_FIXED_HGCDTE_SCALES = False
WARMUP_SHARED_HGCDTE_SCALES = True
WARMUP_FIXED_AR_INDICES = np.asarray([1.31, 1.73, 2.27], dtype=float)
PRODUCTION_NP_SMOOTHNESS_WEIGHT = 0.1
PRODUCTION_SHARED_HGCDTE_SCALES = True
PRODUCTION_TILE_RATE_PRIOR_SIGMA_LOG = 10.0
PRODUCTION_TILE_RATE_TRAJECTORY_SMOOTHNESS_WEIGHT = 0.001
_PRODUCTION_STATISTICS_PATH = None
DEFAULT_SINGLE_SCA_TILE_STATISTICS = (
    ROOT / "wfi_tile_log_statistics_512" / "wfi01_tile_log_statistics_512.csv"
)
DEFAULT_ALL_SCA_STATISTICS_DIR = ROOT / "wfi_tile_log_statistics_512"
DEFAULT_MODEL_PATH = Path(
    "/Users/paterson.35/Documents/"
    "rgespit-comissioning-ice-simulations/data/ice_optical_constants/"
    "spectral_response_model_SCA1.ecsv"
)
DEFAULT_SINGLE_SCA_OUTPUT = ROOT / "wfi01_tmm_numpyro_tile_pilot"
DEFAULT_ALL_SCA_OUTPUT = ROOT / "wfi_all_sca_tmm_numpyro_tile_production"
DEFAULT_STEP_TIME = "2026-09-14T23:38:00Z"
ASW_OPTICAL_CONSTANTS_URL = (
    "https://sbnarchive.psi.edu/pds4/non_mission/"
    "gbo.ices.mastrapa.lab-spectra/data/nk/amorph_high.tab"
)
HGCDTE_OPTICAL_CONSTANTS_SOURCE = (
    "Mouzali, tel-01281981, Table 2.1 and Eqs. (2.3)-(2.8)"
)
ICE_INDEX_DEFAULT = 1.25
ICE_INDEX_PRIOR_MEAN = ICE_INDEX_DEFAULT
ICE_INDEX_PRIOR_SIGMA = 0.03
ICE_INDEX_LOWER = 1.15
ICE_INDEX_UPPER = 1.40
ICE_EXTINCTION_PRIOR_SIGMA = 0.005
ICE_EXTINCTION_UPPER = 0.05
ICE_EXTINCTION_INIT = 1.0e-6
POST_RESET_ICE_INDEX_PRIOR_MEAN = ICE_INDEX_DEFAULT
POST_RESET_ICE_INDEX_PRIOR_SIGMA = 0.03
POST_RESET_ICE_INDEX_LOWER = 1.15
POST_RESET_ICE_INDEX_UPPER = 1.40

REFERENCE_THICKNESS_NM = 122.7
REFERENCE_TIME_DAYS = 4.65
RESET_DAY = 18.0
INITIAL_THICKNESS_INIT_NM = 35.9
INITIAL_THICKNESS_PRIOR_MEAN_NM = 35.0
INITIAL_THICKNESS_PRIOR_SIGMA_NM = 15.0
INITIAL_THICKNESS_PRIOR_LOW_NM = 0.0
INITIAL_THICKNESS_PRIOR_HIGH_NM = 50.0
NUTS_INITIAL_THICKNESS_INIT_NM = 0.1
NUTS_INIT_JITTER = 0.01
TRAJECTORY_SCALE_INIT = 1.14
TRAJECTORY_SCALE_START = 2.0
NOMINAL_HGCDTE = np.asarray([3.6, 0.7], dtype=float)
HGCDTE_REFERENCE_COMPOSITION = 0.445
HGCDTE_REFERENCE_TEMPERATURE_K = 80.0
HGCDTE_LOWER = np.asarray([1.0, 0.0], dtype=float)
HGCDTE_UPPER = np.asarray([6.0, 3.0], dtype=float)
HGCDTE_PRIOR_SIGMA = np.asarray([1.0, 1.0], dtype=float)
DETECTOR_THICKNESS_DEFAULT_NM = 2500.0
DETECTOR_THICKNESS_PRIOR_SIGMA_LOG = 1.0
DETECTOR_THICKNESS_NEIGHBOR_SIGMA_LOG = 0.5
COMPLEX_Q_PARAMETER_NAMES = (
    "log_q_magnitude",
    "q_magnitude_slope",
    "q_phase_offset_rad",
    "q_optical_path_nm",
    "q_absorption_per_nm",
)
COMPLEX_Q_LOWER = np.asarray(
    [np.log(1.0e-5), -1.0, -np.pi, -2000.0, 0.0], dtype=float
)
COMPLEX_Q_UPPER = np.asarray(
    [np.log(0.2), 1.0, np.pi, 2000.0, 0.02], dtype=float
)
COMPLEX_Q_INITIAL = np.asarray(
    [-5.806160724, 0.628387402, -2.029580405, -235.622729, 6.486e-5],
    dtype=float,
)
COMPLEX_Q_MAP_SCALES = np.asarray([1.0, 1.0, np.pi, 500.0, 0.005], dtype=float)
HGCDTE_N_TABLE_WAVELENGTHS_NM = np.arange(1600.0, 6600.1, 200.0)
HGCDTE_N_TABLE_X0397 = np.asarray(
    [
        2.946,
        2.978,
        3.012,
        3.049,
        3.088,
        3.128,
        3.168,
        3.206,
        3.159,
        3.126,
        3.110,
        3.100,
        3.093,
        3.087,
        3.083,
        3.079,
        3.077,
        3.074,
        3.072,
        3.071,
        3.069,
        3.068,
        3.067,
        3.066,
        3.065,
        3.064,
    ],
    dtype=float,
)
HGCDTE_N_TABLE_X0459 = np.asarray(
    [
        2.895,
        2.940,
        2.987,
        3.036,
        3.086,
        3.030,
        3.003,
        2.989,
        2.981,
        2.975,
        2.970,
        2.967,
        2.965,
        2.962,
        2.961,
        2.959,
        2.958,
        2.957,
        2.956,
        2.955,
        2.955,
        2.954,
        2.954,
        2.953,
        2.953,
        2.952,
    ],
    dtype=float,
)
HGCDTE_MODULATION_NODE_WAVELENGTHS_NM = np.asarray(
    [500.0, 850.0, 1200.0, 1550.0, 1900.0, 2250.0, 2600.0],
    dtype=float,
)
HGCDTE_MODULATION_WIDTH_NM = 350.0
HGCDTE_MODULATION_PRIOR_SIGMA_LOG = 1.0
ICE_MODULATION_NODE_WAVELENGTHS_NM = np.asarray(
    [
        623.520885362797,
        852.802649193115,
        1038.5847494059,
        1269.7582849979,
        1481.24145201185,
        1670.16888078615,
    ],
    dtype=float,
)
ICE_MODULATION_WIDTH_NM = 500.0
ICE_MODULATION_PRIOR_SIGMA_LOG = 0.05
POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA = 0.1
CONDUCTIVE_REFERENCE_WAVELENGTH_NM = 1600.0
CONDUCTIVE_EPSILON_INF = 1.0
CONDUCTIVE_LOG_THICKNESS_PRIOR_MEAN = np.log(5.0)
CONDUCTIVE_LOG_PLASMA_STRENGTH_PRIOR_MEAN = np.log(1.0)
CONDUCTIVE_LOG_DAMPING_RATIO_PRIOR_MEAN = np.log(1.0)
CONDUCTIVE_LOG_PRIOR_SIGMA = 1.5
QE_RBF_NODE_WAVELENGTHS_NM = np.asarray([650.0, 950.0, 1250.0, 1600.0])
QE_RBF_WIDTH_NM = 250.0
QE_RBF_PRIOR_SIGMA_LOG = 0.15
AR_LOWER = np.asarray([0.01, 1.0, 0.0], dtype=float)
AR_UPPER = np.asarray([10.0, 4.0, 0.5], dtype=float)
AR_MEAN_PRIOR_SIGMA = 2.0
AR_DIELECTRIC_K_START = 1.0e-4
AR_DESIGN_WAVELENGTH_NM = 1050.0
AR_MAX_LAYERS = 3
AR_SCA_SIGMA_PRIOR = 0.2
AR_TILE_SIGMA_PRIOR = 0.5
SCA_AR_OPD_PRIOR_SIGMA_NM = 1.0
NP_ENTROPY_WEIGHT = 0.01
NP_SMOOTHNESS_WEIGHT = 4.64158883361
NP_MAX_NODE_MULTIPLIER = 2.0
PREDICTION_ROW_CHUNK_SIZE = 64
MAP_PROGRESS_INTERVAL = 50
GAUSS_LEGENDRE_NODES, GAUSS_LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(8)
INCIDENCE_CONE_NODES, INCIDENCE_CONE_WEIGHTS = np.polynomial.legendre.leggauss(8)
PUPIL_RADIAL_NODES, PUPIL_RADIAL_WEIGHTS = np.polynomial.legendre.leggauss(4)
PUPIL_AZIMUTH_COUNT = 8
INCIDENCE_ANGLE_MAX_DEG = 15.0
INCIDENCE_ANGLE_PRIOR_SIGMA_DEG = 3.0
INCIDENCE_ANGLE_INIT_DEG = 1.0
TRAJECTORY_TILE_RATE_PRIOR_SIGMA_LOG = 0.10
TRAJECTORY_TILE_RATE_NEIGHBOR_SIGMA_LOG = 0.05
ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG = 2.0
EARLY_TIME_LOG_SCALE_PRIOR_MEAN = 0.0
EARLY_TIME_SCALE_PRIOR_SIGMA = 0.05


def qe_rbf_configuration(
    node_wavelengths_nm=None,
    width_nm=QE_RBF_WIDTH_NM,
    prior_sigma_log=QE_RBF_PRIOR_SIGMA_LOG,
):
    """Validate and return the wavelength-dependent QE basis configuration."""
    if node_wavelengths_nm is None:
        node_wavelengths_nm = QE_RBF_NODE_WAVELENGTHS_NM
    node_wavelengths_nm = np.asarray(node_wavelengths_nm, dtype=float).reshape(-1)
    if (
        node_wavelengths_nm.size < 2
        or not np.all(np.isfinite(node_wavelengths_nm))
        or np.any(node_wavelengths_nm <= 0.0)
        or np.any(np.diff(node_wavelengths_nm) <= 0.0)
    ):
        raise ValueError(
            "QE RBF nodes must contain at least two strictly increasing positive wavelengths"
        )
    if not np.isfinite(width_nm) or width_nm <= 0.0:
        raise ValueError("QE RBF width must be positive and finite")
    if not np.isfinite(prior_sigma_log) or prior_sigma_log <= 0.0:
        raise ValueError("QE RBF prior sigma must be positive and finite")
    return node_wavelengths_nm, float(width_nm), float(prior_sigma_log)


def incidence_cone_quadrature(half_angle_deg):
    """Return polar angles and normalized solid-angle weights for a cone."""
    half_angle_deg = float(half_angle_deg)
    if (
        not np.isfinite(half_angle_deg)
        or half_angle_deg < 0.0
        or half_angle_deg >= 90.0
    ):
        raise ValueError("Incidence cone half-angle must be in [0, 90) degrees")
    if half_angle_deg == 0.0:
        return np.asarray([0.0]), np.asarray([1.0])
    minimum_cosine = np.cos(np.deg2rad(half_angle_deg))
    cosine_nodes = (
        0.5 * (1.0 + minimum_cosine)
        + 0.5 * (1.0 - minimum_cosine) * INCIDENCE_CONE_NODES
    )
    angles_deg = np.rad2deg(np.arccos(cosine_nodes))
    weights = 0.5 * INCIDENCE_CONE_WEIGHTS
    return angles_deg, weights


def incidence_pupil_quadrature(half_angle_deg):
    """Return chief-ray-local directions for a uniformly illuminated pupil."""
    half_angle_deg = float(half_angle_deg)
    if (
        not np.isfinite(half_angle_deg)
        or half_angle_deg < 0.0
        or half_angle_deg >= 90.0
    ):
        raise ValueError("Incidence pupil half-angle must be in [0, 90) degrees")
    if half_angle_deg == 0.0:
        return np.asarray([[0.0, 0.0, 1.0]]), np.asarray([1.0])

    # Uniform pupil area has a uniform distribution in normalized radius squared.
    radius_squared = 0.5 * (1.0 + PUPIL_RADIAL_NODES)
    radial_weights = 0.5 * PUPIL_RADIAL_WEIGHTS
    radius = np.sqrt(radius_squared)
    marginal_tangent = np.tan(np.deg2rad(half_angle_deg))
    polar_angle = np.arctan(radius * marginal_tangent)
    azimuth = 2.0 * np.pi * np.arange(PUPIL_AZIMUTH_COUNT) / PUPIL_AZIMUTH_COUNT
    sin_polar = np.sin(polar_angle)[:, None]
    directions = np.stack(
        (
            sin_polar * np.cos(azimuth)[None, :],
            sin_polar * np.sin(azimuth)[None, :],
            np.broadcast_to(np.cos(polar_angle)[:, None], sin_polar.shape[:-1] + (PUPIL_AZIMUTH_COUNT,)),
        ),
        axis=-1,
    ).reshape((-1, 3))
    weights = np.repeat(radial_weights / PUPIL_AZIMUTH_COUNT, PUPIL_AZIMUTH_COUNT)
    return directions, weights


def incidence_pupil_ray_sines(half_angle_deg, chief_ray_tilts_deg):
    """Return per-row incidence sines for a chief-ray-centered pupil."""
    local_directions, weights = incidence_pupil_quadrature(half_angle_deg)
    chief_ray_tilts_deg = np.asarray(chief_ray_tilts_deg, dtype=float)
    if chief_ray_tilts_deg.ndim == 1:
        chief_ray_tilts_deg = chief_ray_tilts_deg.reshape((1, 2))
    if (
        chief_ray_tilts_deg.ndim != 2
        or chief_ray_tilts_deg.shape[1] != 2
        or np.any(~np.isfinite(chief_ray_tilts_deg))
        or np.any(np.abs(chief_ray_tilts_deg) >= 89.0)
    ):
        raise ValueError("Chief-ray tilts must be a finite (row, 2) array within +/-89 degrees")

    tangents = np.tan(np.deg2rad(chief_ray_tilts_deg))
    chief = np.column_stack((tangents, np.ones(len(tangents))))
    chief /= np.linalg.norm(chief, axis=1, keepdims=True)
    basis_x = np.column_stack((chief[:, 2], np.zeros(len(chief)), -chief[:, 0]))
    basis_x /= np.linalg.norm(basis_x, axis=1, keepdims=True)
    basis_y = np.cross(chief, basis_x)
    directions = (
        local_directions[None, :, 0, None] * basis_x[:, None, :]
        + local_directions[None, :, 1, None] * basis_y[:, None, :]
        + local_directions[None, :, 2, None] * chief[:, None, :]
    )
    if np.any(directions[:, :, 2] <= 0.0):
        raise ValueError("Pupil geometry contains rays directed away from the detector")
    sine = np.linalg.norm(directions[:, :, :2], axis=-1)
    return sine, weights


def load_incidence_chief_ray_tilts(path, data, default_tilts_deg=(0.0, 0.0)):
    """Load per-SCA/tile chief-ray x/y tilts from a CSV."""
    default = np.asarray(default_tilts_deg, dtype=float).reshape(2)
    n_rows = int(data["n_scas"] * data["n_tiles"])
    tilts = np.broadcast_to(default, (n_rows, 2)).copy()
    if path is None:
        return tilts
    path = Path(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"detector", "tile_id", "chief_ray_x_deg", "chief_ray_y_deg"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            "Incidence geometry CSV must contain detector, tile_id, "
            "chief_ray_x_deg, and chief_ray_y_deg"
        )
    row_lookup = {
        (str(detector), int(tile_id)): sca_index * int(data["n_tiles"]) + tile_index
        for sca_index, detector in enumerate(data["sca_names"])
        for tile_index, tile_id in enumerate(data["tile_ids"])
    }
    seen = set()
    for row in rows:
        key = (row["detector"], int(row["tile_id"]))
        if key not in row_lookup:
            continue
        if key in seen:
            raise ValueError(f"Duplicate incidence geometry row for {key[0]} tile {key[1]}")
        tilts[row_lookup[key]] = [
            float(row["chief_ray_x_deg"]),
            float(row["chief_ray_y_deg"]),
        ]
        seen.add(key)
    missing = set(row_lookup) - seen
    if missing:
        raise ValueError(
            "Incidence geometry CSV is missing selected rows: "
            + ", ".join(f"{detector}/{tile_id}" for detector, tile_id in sorted(missing))
        )
    incidence_pupil_ray_sines(0.0, tilts)
    return tilts


def load_asw_optical_constants(source=ASW_OPTICAL_CONSTANTS_URL):
    """Load the 100 K amorphous-water-ice n/k table in micrometers."""
    values = np.loadtxt(io.StringIO(read_source(source)), ndmin=2)
    if values.shape[1] < 3 or values.shape[0] < 2:
        raise ValueError("ASW optical-constant table must contain wavelength, n, and k")
    wavelengths_nm = 1000.0 * values[:, 0]
    real_index = values[:, 1]
    imaginary_index = values[:, 2]
    if (
        np.any(~np.isfinite(wavelengths_nm))
        or np.any(~np.isfinite(real_index))
        or np.any(~np.isfinite(imaginary_index))
        or np.any(np.diff(wavelengths_nm) <= 0.0)
        or np.any(imaginary_index < 0.0)
    ):
        raise ValueError("ASW optical constants must be finite, ordered, and passive")
    return wavelengths_nm, real_index, imaginary_index


def hgcdte_reference_optical_constants(wavelengths_nm):
    """Return a fixed 80 K HgCdTe n/k sensitivity model.

    The real-index values are linearly interpolated in composition between the
    x=0.397 and x=0.459 columns of Mouzali's 80 K table, then held constant
    below the tabulated 1.6 micron edge.  The imaginary index uses the
    below-gap and above-gap absorption laws from the same source, joined with
    a smooth transition between the gap and the source-defined transition
    energy.  This is an opt-in sensitivity model, not a detector-specific
    calibration.
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    if (
        wavelengths_nm.size == 0
        or np.any(~np.isfinite(wavelengths_nm))
        or np.any(wavelengths_nm <= 0.0)
    ):
        raise ValueError("HgCdTe wavelengths must be positive, finite, and non-empty")

    x = HGCDTE_REFERENCE_COMPOSITION
    temperature_k = HGCDTE_REFERENCE_TEMPERATURE_K
    if not 0.397 <= x <= 0.459:
        raise ValueError("Reference composition is outside the tabulated n range")
    composition_weight = (x - 0.397) / (0.459 - 0.397)
    real_index_table = HGCDTE_N_TABLE_X0397 + composition_weight * (
        HGCDTE_N_TABLE_X0459 - HGCDTE_N_TABLE_X0397
    )
    real_index = np.interp(
        wavelengths_nm,
        HGCDTE_N_TABLE_WAVELENGTHS_NM,
        real_index_table,
        left=real_index_table[0],
        right=real_index_table[-1],
    )

    wavelength_um = wavelengths_nm / 1000.0
    photon_energy = 1.239841984 / wavelength_um
    band_gap = (
        -0.302
        + 1.93 * x
        - 0.810 * x**2
        + 0.832 * x**3
        + 5.35e-4
        * (1.0 - 2.0 * x)
        * (temperature_k**3 - 1822.0)
        / (255.2 + temperature_k**2)
    )
    transition_temperature = 81.9
    transition_energy_reference = -0.3424 + 1.838 * x
    absorption_slope = 32670.0 * (1.0 + x)
    absorption_reference = np.exp(-18.88 + 53.61 * x)
    transition_absorption = (
        3.9778 * temperature_k
        - 566.4
        + (
            -0.020366 * temperature_k**2
            - 0.46742 * temperature_k
            + 3878.9
        )
        * x
    )
    transition_energy = transition_energy_reference + (
        (temperature_k + transition_temperature)
        / absorption_slope
        * np.log(transition_absorption / absorption_reference)
    )
    if not band_gap < transition_energy:
        raise ValueError("HgCdTe absorption transition must be above the band gap")

    below_gap_absorption = np.zeros_like(photon_energy)
    below_gap = photon_energy < transition_energy
    below_gap_exponent = absorption_slope * (
        photon_energy[below_gap] - transition_energy_reference
    ) / (temperature_k + transition_temperature)
    below_gap_absorption[below_gap] = absorption_reference * np.exp(
        np.clip(below_gap_exponent, -745.0, 700.0)
    )

    above_gap_absorption = np.zeros_like(photon_energy)
    above_gap = photon_energy > band_gap
    absorption_amplitude = (
        -20060.0
        + 115750.0 * x
        + 32.43 * temperature_k
        - 64170.0 * x**2
        + 0.43231 * temperature_k**2
        - 101.92 * x * temperature_k
    )
    absorption_exponent = (
        0.74487
        - 0.44513 * x
        + (0.000799 - 0.000757 * x) * temperature_k
    )
    above_gap_absorption[above_gap] = (
        absorption_amplitude
        * np.maximum(photon_energy[above_gap] - band_gap, 0.0) ** absorption_exponent
        / photon_energy[above_gap]
    )

    transition_fraction = np.clip(
        (photon_energy - band_gap) / (transition_energy - band_gap), 0.0, 1.0
    )
    transition_fraction = transition_fraction**2 * (
        3.0 - 2.0 * transition_fraction
    )
    absorption = (
        (1.0 - transition_fraction) * below_gap_absorption
        + transition_fraction * above_gap_absorption
    )
    extinction_index = wavelengths_nm * 1.0e-7 * absorption / (4.0 * np.pi)
    return real_index, extinction_index


def hgcdte_modulation_basis(
    wavelengths_nm,
    node_wavelengths_nm=HGCDTE_MODULATION_NODE_WAVELENGTHS_NM,
    width_nm=HGCDTE_MODULATION_WIDTH_NM,
):
    """Return normalized smooth basis functions for HgCdTe n/k modulation."""
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    node_wavelengths_nm = np.asarray(node_wavelengths_nm, dtype=float).reshape(-1)
    if (
        wavelengths_nm.size == 0
        or np.any(~np.isfinite(wavelengths_nm))
        or np.any(wavelengths_nm <= 0.0)
    ):
        raise ValueError("HgCdTe modulation wavelengths must be positive and finite")
    if (
        node_wavelengths_nm.size < 2
        or np.any(~np.isfinite(node_wavelengths_nm))
        or np.any(np.diff(node_wavelengths_nm) <= 0.0)
        or not np.isfinite(width_nm)
        or width_nm <= 0.0
    ):
        raise ValueError("HgCdTe modulation nodes and width must be finite and ordered")
    basis = np.exp(
        -0.5
        * ((wavelengths_nm[:, None] - node_wavelengths_nm[None, :]) / width_nm) ** 2
    )
    return basis / np.sum(basis, axis=1, keepdims=True)


def ice_modulation_basis(
    wavelengths_nm,
    node_wavelengths_nm=ICE_MODULATION_NODE_WAVELENGTHS_NM,
    width_nm=ICE_MODULATION_WIDTH_NM,
):
    """Return normalized broad basis functions for global ice n/k modulation."""
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    node_wavelengths_nm = np.asarray(node_wavelengths_nm, dtype=float).reshape(-1)
    if (
        wavelengths_nm.size == 0
        or np.any(~np.isfinite(wavelengths_nm))
        or np.any(wavelengths_nm <= 0.0)
    ):
        raise ValueError("Ice modulation wavelengths must be positive and finite")
    if (
        node_wavelengths_nm.size < 2
        or np.any(~np.isfinite(node_wavelengths_nm))
        or np.any(np.diff(node_wavelengths_nm) <= 0.0)
        or not np.isfinite(width_nm)
        or width_nm <= 0.0
    ):
        raise ValueError("Ice modulation nodes and width must be finite and ordered")
    basis = np.exp(
        -0.5
        * ((wavelengths_nm[:, None] - node_wavelengths_nm[None, :]) / width_nm) ** 2
    )
    return basis / np.sum(basis, axis=1, keepdims=True)


def dielectric_ar_reference(
    ar_layers,
    design_wavelength_nm=AR_DESIGN_WAVELENGTH_NM,
    middle_layer_eighth_wave=False,
    middle_layer_half_wave=False,
):
    """Return a lossless equal-index-ratio matching-stack reference."""
    ar_layers = int(ar_layers)
    if ar_layers < 1 or ar_layers > AR_MAX_LAYERS:
        raise ValueError(f"ar_layers must be between one and {AR_MAX_LAYERS}")
    if not np.isfinite(design_wavelength_nm) or design_wavelength_nm <= 0.0:
        raise ValueError("design_wavelength_nm must be positive and finite")
    if middle_layer_eighth_wave and middle_layer_half_wave:
        raise ValueError("AR middle-layer thickness choices are mutually exclusive")
    if (middle_layer_eighth_wave or middle_layer_half_wave) and ar_layers != 3:
        raise ValueError("A special middle-layer thickness requires three AR layers")
    layer_indices = NOMINAL_HGCDTE[0] ** (
        np.arange(1, ar_layers + 1, dtype=float) / (ar_layers + 1)
    )
    optical_divisors = np.full(ar_layers, 4.0)
    if middle_layer_eighth_wave:
        optical_divisors[1] = 8.0
    elif middle_layer_half_wave:
        optical_divisors[1] = 2.0
    thickness_scales = design_wavelength_nm / (
        optical_divisors * layer_indices * 100.0
    )
    return np.column_stack(
        (
            thickness_scales,
            layer_indices,
            np.full(ar_layers, AR_DIELECTRIC_K_START),
        )
    ).reshape(-1)


def ar_prior_arrays(ar_layers):
    """Return layer-major dielectric AR prior centers and physical bounds."""
    reference = dielectric_ar_reference(ar_layers)
    return (
        reference,
        np.tile(AR_LOWER, ar_layers),
        np.tile(AR_UPPER, ar_layers),
    )


def validate_fixed_ar_indices(ar_layers, fixed_ar_indices):
    if fixed_ar_indices is None:
        return None
    fixed_ar_indices = np.asarray(fixed_ar_indices, dtype=float).reshape(-1)
    if fixed_ar_indices.shape != (ar_layers,) or not np.all(
        np.isfinite(fixed_ar_indices)
    ):
        raise ValueError("Fixed AR indices must have one finite value per layer")
    if np.any(fixed_ar_indices <= AR_LOWER[1]) or np.any(
        fixed_ar_indices >= AR_UPPER[1]
    ):
        raise ValueError("Fixed AR indices must lie strictly inside the AR index bounds")
    return fixed_ar_indices


def ar_active_parameter_indices(ar_layers, fit_ar_absorption, fixed_ar_indices=None):
    active_parameter_indices = np.arange(3 * ar_layers, dtype=int)
    if not fit_ar_absorption:
        active_parameter_indices = active_parameter_indices[
            active_parameter_indices % 3 != 2
        ]
    fixed_ar_indices = validate_fixed_ar_indices(ar_layers, fixed_ar_indices)
    if fixed_ar_indices is not None:
        fixed_index_positions = 3 * np.arange(ar_layers, dtype=int) + 1
        active_parameter_indices = active_parameter_indices[
            ~np.isin(active_parameter_indices, fixed_index_positions)
        ]
    return active_parameter_indices


def ar_reference_raw(
    ar_reference,
    ar_lower,
    ar_upper,
    active_parameter_indices,
    ordered_ar_indices=False,
):
    """Convert physical AR references into the selected raw coordinates."""
    ar_reference = np.asarray(ar_reference, dtype=float)
    ar_lower = np.asarray(ar_lower, dtype=float)
    ar_upper = np.asarray(ar_upper, dtype=float)
    active_parameter_indices = np.asarray(active_parameter_indices, dtype=int)
    raw = np.empty(active_parameter_indices.size, dtype=float)
    for position, parameter_index in enumerate(active_parameter_indices):
        lower = ar_lower[parameter_index]
        if ordered_ar_indices and parameter_index % 3 == 1:
            layer = parameter_index // 3
            lower = (
                ar_lower[parameter_index]
                if layer == 0
                else ar_reference[parameter_index - 3]
            )
        upper = ar_upper[parameter_index]
        probability = (ar_reference[parameter_index] - lower) / (upper - lower)
        if not 0.0 < probability < 1.0:
            raise ValueError("AR reference is outside the selected raw-coordinate bounds")
        raw[position] = np.log(probability / (1.0 - probability))
    return raw


def ordered_ar_indices_from_raw(
    ar_raw,
    active_parameter_indices,
    ar_lower,
    ar_upper,
    ar_parameter_count,
    jax_module,
    jnp_module,
):
    """Map raw AR coordinates to layer indices with strict layer ordering."""
    active_parameter_indices = np.asarray(active_parameter_indices, dtype=int)
    active_index_positions = {
        int(parameter_index): position
        for position, parameter_index in enumerate(active_parameter_indices)
    }
    values = jnp_module.zeros(
        ar_raw.shape[:-1] + (ar_parameter_count,), dtype=ar_raw.dtype
    )
    active_lower = jnp_module.asarray(np.asarray(ar_lower)[active_parameter_indices])
    active_upper = jnp_module.asarray(np.asarray(ar_upper)[active_parameter_indices])
    values = values.at[..., active_parameter_indices].set(
        active_lower
        + (active_upper - active_lower) * jax_module.nn.sigmoid(ar_raw)
    )
    previous = jnp_module.asarray(ar_lower[1], dtype=ar_raw.dtype)
    for layer in range(ar_parameter_count // 3):
        parameter_index = 3 * layer + 1
        position = active_index_positions[parameter_index]
        previous = previous + (
            ar_upper[parameter_index] - previous
        ) * jax_module.nn.sigmoid(ar_raw[..., position])
        values = values.at[..., parameter_index].set(previous)
    return values


def ar_coordinate_rotation(active_parameter_indices, ar_parameter_count):
    """Return an orthogonal thickness/index rotation for each AR layer."""
    active_parameter_indices = np.asarray(active_parameter_indices, dtype=int)
    rotation = np.eye(active_parameter_indices.size, dtype=float)
    index_positions = {
        int(parameter_index): position
        for position, parameter_index in enumerate(active_parameter_indices)
    }
    layer_count = int(ar_parameter_count // 3)
    inverse_sqrt_two = 1.0 / np.sqrt(2.0)
    for layer in range(layer_count):
        thickness_index = 3 * layer
        refractive_index = thickness_index + 1
        if thickness_index not in index_positions or refractive_index not in index_positions:
            continue
        thickness_position = index_positions[thickness_index]
        refractive_position = index_positions[refractive_index]
        rotation[np.ix_([thickness_position, refractive_position], [
            thickness_position,
            refractive_position,
        ])] = inverse_sqrt_two * np.asarray([[1.0, 1.0], [1.0, -1.0]])
    return rotation


def default_likelihood_weight(counts):
    """Scale pixel SSE to approximately one effective datum per tile/group."""
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0 or np.any(~np.isfinite(counts)) or np.any(counts <= 0.0):
        raise ValueError("Likelihood counts must be positive and finite")
    return 1.0 / float(np.median(counts))


def trajectory_rate_per_scale(days):
    """Return the baseline instantaneous rate at the first observed time."""
    days = np.asarray(days, dtype=float)
    if days.size < 2 or not np.all(np.isfinite(days)) or np.ptp(days) <= 0.0:
        raise ValueError("Trajectory days must span at least two finite times")
    first_day = np.min(days)
    return float(
        REFERENCE_THICKNESS_NM
        / (
            2.0
            * np.sqrt(REFERENCE_TIME_DAYS)
            * np.sqrt(RESET_DAY + first_day)
        )
    )


def rate_curvature_matrix(node_days):
    """Return the linear operator used by the time-aware rate curvature penalty."""
    node_days = np.asarray(node_days, dtype=float).reshape(-1)
    if node_days.size < 3 or not np.all(np.isfinite(node_days)):
        raise ValueError("At least three finite rate-node days are required")
    if np.any(np.diff(node_days) <= 0.0):
        raise ValueError("Rate-node days must be strictly increasing")
    scaled_node_days = (node_days - node_days[0]) / np.median(np.diff(node_days))
    matrix = np.zeros((node_days.size - 2, node_days.size), dtype=float)
    for column in range(node_days.size):
        basis = np.zeros(node_days.size, dtype=float)
        basis[column] = 1.0
        node_slopes = np.diff(basis) / np.diff(scaled_node_days)
        matrix[:, column] = 2.0 * np.diff(node_slopes) / (
            scaled_node_days[2:] - scaled_node_days[:-2]
        )
    return matrix


def trajectory_rate_whitener(
    node_days, prior_sigma, likelihood_scale, smoothness_weight
):
    """Return a symmetric inverse square root of the exact rate prior precision."""
    node_days = np.asarray(node_days, dtype=float).reshape(-1)
    prior_sigma = float(prior_sigma)
    likelihood_scale = float(likelihood_scale)
    smoothness_weight = float(smoothness_weight)
    if not np.isfinite(prior_sigma) or prior_sigma <= 0.0:
        raise ValueError("Rate prior sigma must be positive and finite")
    if not np.isfinite(likelihood_scale) or likelihood_scale < 0.0:
        raise ValueError("Likelihood scale must be finite and non-negative")
    if not np.isfinite(smoothness_weight) or smoothness_weight < 0.0:
        raise ValueError("Rate smoothness weight must be finite and non-negative")
    curvature = rate_curvature_matrix(node_days)
    precision = np.eye(node_days.size, dtype=float) / prior_sigma**2
    if smoothness_weight > 0.0:
        precision += (
            2.0
            * likelihood_scale
            * smoothness_weight
            / curvature.shape[0]
            * (curvature.T @ curvature)
        )
    eigenvalues, eigenvectors = np.linalg.eigh(precision)
    if np.any(eigenvalues <= 0.0) or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Rate prior precision is not positive definite")
    return (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T


def trajectory_rate_likelihood_scale(data, args):
    """Return the fixed SSE scale used by the rate smoothness factor."""
    likelihood_weight = getattr(args, "likelihood_weight", None)
    if likelihood_weight is None:
        likelihood_weight = default_likelihood_weight(data["counts"])
    effective_weight = float(likelihood_weight) / int(data["n_tiles"])
    return 0.5 * effective_weight * float(np.sum(data["sum_squared"])) / (
        float(args.log_noise_sigma) ** 2
    )


def whiten_trajectory_rate_nodes(
    log_rates, node_days, trajectory_prior_rate, prior_sigma, likelihood_scale,
    smoothness_weight,
):
    """Convert constrained log-rate nodes to the exact whitened coordinates."""
    log_rates = np.asarray(log_rates, dtype=float)
    mean = np.full(log_rates.shape, float(np.log(trajectory_prior_rate)))
    whitener = trajectory_rate_whitener(
        node_days, prior_sigma, likelihood_scale, smoothness_weight
    )
    return np.linalg.solve(whitener, log_rates - mean)


def trajectory_prior_rate_for_data(data, args):
    """Return the fully non-parametric trajectory prior center for selected data."""
    return (
        getattr(args, "trajectory_rate_prior_mean_nm_day", None)
        if getattr(args, "trajectory_rate_prior_mean_nm_day", None) is not None
        else trajectory_rate_per_scale(data["days"]) * TRAJECTORY_SCALE_INIT
    )


def trajectory_pivot_configuration(data, trajectory_prior_rate, prior_mean):
    """Return a central node pivot and its fixed initialization center."""
    node_days = np.asarray(data["node_days"], dtype=float)
    pivot_index = len(node_days) // 2
    pivot_day = float(node_days[pivot_index])
    start_day = float(np.min(np.asarray(data["days"], dtype=float)))
    pivot_center = float(prior_mean) + max(pivot_day - start_day, 0.0) * float(
        trajectory_prior_rate
    )
    return pivot_index, pivot_day, pivot_center


def integrate_log_linear_rate_numpy(start_day, target_day, node_days, log_rates):
    """Integrate a positive log-linearly interpolated rate for initialization."""
    start_day = float(start_day)
    target_day = float(target_day)
    node_days = np.asarray(node_days, dtype=float).reshape(-1)
    log_rates = np.asarray(log_rates, dtype=float).reshape(-1)
    if node_days.size != log_rates.size:
        raise ValueError("Rate nodes and rate-node days must have matching sizes")
    if target_day <= start_day:
        return 0.0
    interior_nodes = node_days[(node_days > start_day) & (node_days < target_day)]
    integration_days = np.concatenate(
        ([start_day], interior_nodes, [target_day])
    )
    total = 0.0
    for left, right in zip(integration_days[:-1], integration_days[1:]):
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        quadrature_days = midpoint + half_width * GAUSS_LEGENDRE_NODES
        total += half_width * np.sum(
            GAUSS_LEGENDRE_WEIGHTS
            * np.exp(np.interp(quadrature_days, node_days, log_rates))
        )
    return float(total)


def trajectory_pivot_z_from_initial_thickness(d0, data, args, log_rates):
    """Convert a constrained initial thickness to the exact pivot coordinate."""
    trajectory_prior_rate = trajectory_prior_rate_for_data(data, args)
    _, pivot_day, pivot_center = trajectory_pivot_configuration(
        data, trajectory_prior_rate, args.initial_thickness_prior_mean_nm
    )
    pivot_integral = integrate_log_linear_rate_numpy(
        np.min(data["days"]), pivot_day, data["node_days"], log_rates
    )
    return float(
        (
            float(np.asarray(d0))
            + pivot_integral
            - pivot_center
        )
        / args.initial_thickness_prior_sigma_nm
    )


def trajectory_log_rates_from_initial_values(data, args, initial_values):
    """Recover full log-rate nodes from either direct or whitened starts."""
    trajectory_prior_rate = trajectory_prior_rate_for_data(data, args)
    if args.whiten_np_trajectory:
        whitener = trajectory_rate_whitener(
            data["node_days"],
            args.trajectory_rate_prior_sigma_log,
            trajectory_rate_likelihood_scale(data, args),
            args.np_smoothness_weight,
        )
        rate_latents = np.asarray(initial_values["np_log_rate_z"], dtype=float)
        rate_rotation = getattr(args, "np_rate_likelihood_rotation", None)
        if rate_rotation is not None:
            rate_latents = np.asarray(rate_rotation, dtype=float) @ rate_latents
        return np.log(trajectory_prior_rate) + whitener @ np.asarray(
            rate_latents,
        )
    return np.asarray(initial_values["np_log_rate_nodes"], dtype=float)


def apply_trajectory_rate_likelihood_rotation(initial_values, args):
    """Convert old whitened rate starts into the rotated latent coordinates."""
    rotation = getattr(args, "np_rate_likelihood_rotation", None)
    if rotation is None or "np_log_rate_z" not in initial_values:
        return initial_values
    rotated_values = dict(initial_values)
    rotated_values["np_log_rate_z"] = np.asarray(rotation, dtype=float).T @ np.asarray(
        initial_values["np_log_rate_z"], dtype=float
    )
    return rotated_values


def trajectory_rate_posterior_rotation(model, initial_values, jax):
    """Return an orthogonal rate rotation from the conditional MAP Hessian."""
    from numpyro.infer.util import log_density

    if "np_log_rate_z" not in initial_values:
        raise ValueError("A rate-latent start is required for Hessian rotation")
    base_values = {
        name: jax.numpy.asarray(value) for name, value in initial_values.items()
    }
    rate_start = base_values["np_log_rate_z"]

    def negative_log_joint(rate_values):
        values = dict(base_values)
        values["np_log_rate_z"] = rate_values
        log_joint, _ = log_density(model, (), {}, values)
        return -log_joint

    hessian = np.asarray(
        jax.device_get(jax.hessian(negative_log_joint)(rate_start)), dtype=float
    )
    hessian = 0.5 * (hessian + hessian.T)
    if not np.all(np.isfinite(hessian)):
        raise ValueError("Rate posterior Hessian is not finite")
    _, eigenvectors = np.linalg.eigh(hessian)
    if not np.allclose(
        eigenvectors.T @ eigenvectors,
        np.eye(rate_start.shape[0]),
        atol=1.0e-10,
    ):
        raise ValueError("Rate posterior Hessian rotation is not orthogonal")
    return eigenvectors


def trajectory_pivot_d0_from_initial_values(data, args, initial_values):
    """Recover d0 from pivot coordinates for boundary-aware NUTS initialization."""
    trajectory_prior_rate = trajectory_prior_rate_for_data(data, args)
    _, pivot_day, pivot_center = trajectory_pivot_configuration(
        data, trajectory_prior_rate, args.initial_thickness_prior_mean_nm
    )
    log_rates = trajectory_log_rates_from_initial_values(data, args, initial_values)
    pivot_integral = integrate_log_linear_rate_numpy(
        np.min(data["days"]), pivot_day, data["node_days"], log_rates
    )
    pivot_z = float(np.asarray(initial_values["initial_thickness_pivot_z"]))
    return pivot_center + args.initial_thickness_prior_sigma_nm * pivot_z - pivot_integral


def integrate_rate_trajectory(
    jax,
    jnp,
    days,
    node_days,
    integration_times,
    log_scale,
    log_node_multipliers,
    use_square_root=True,
    anchor_day=None,
):
    """Integrate a positive, log-linearly interpolated instantaneous rate."""
    gauss_nodes = jnp.asarray(GAUSS_LEGENDRE_NODES)
    gauss_weights = jnp.asarray(GAUSS_LEGENDRE_WEIGHTS)
    days = jnp.asarray(days)
    node_days = jnp.asarray(node_days)
    integration_times = jnp.asarray(integration_times)
    rate_scale = jnp.exp(log_scale)

    def instantaneous_rate(time):
        if use_square_root:
            base_rate = REFERENCE_THICKNESS_NM / (
                2.0
                * jnp.sqrt(REFERENCE_TIME_DAYS)
                * jnp.sqrt(RESET_DAY + time)
            )
        else:
            base_rate = 1.0
        log_multiplier = jnp.interp(time, node_days, log_node_multipliers)
        return rate_scale * base_rate * jnp.exp(log_multiplier)

    def integrate_interval(left, right):
        left = jnp.asarray(left)[..., None]
        right = jnp.asarray(right)[..., None]
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        return half_width[..., 0] * jnp.sum(
            gauss_weights * instantaneous_rate(
                midpoint + half_width * gauss_nodes
            ),
            axis=-1,
        )

    interval_integrals = jax.vmap(
        integrate_interval
    )(integration_times[:-1], integration_times[1:])
    cumulative_integrals = jnp.concatenate(
        (
            jnp.zeros(1, dtype=interval_integrals.dtype),
            jnp.cumsum(interval_integrals),
        )
    )

    def integrated_at(target_days):
        segment_indices = jnp.searchsorted(
            integration_times, target_days, side="right"
        ) - 1
        segment_indices = jnp.clip(segment_indices, 0, len(integration_times) - 2)
        return cumulative_integrals[segment_indices] + integrate_interval(
            integration_times[segment_indices], target_days
        )

    values = (
        integrated_at(days),
        instantaneous_rate(days),
        integrated_at(node_days),
        instantaneous_rate(node_days),
    )
    if anchor_day is None:
        return values
    return values + (integrated_at(anchor_day),)


def statistics_fingerprint(data):
    digest = hashlib.sha256()
    for key in ("counts", "sums", "sum_squared"):
        digest.update(np.ascontiguousarray(data[key], dtype=np.float64).tobytes())
    return digest.hexdigest()


def statistics_row_fingerprints(data):
    fingerprints = []
    for row in range(data["counts"].shape[0]):
        digest = hashlib.sha256()
        for key in ("counts", "sums", "sum_squared"):
            digest.update(
                np.ascontiguousarray(data[key][row], dtype=np.float64).tobytes()
            )
        fingerprints.append(digest.hexdigest())
    return np.asarray(fingerprints)


def is_single_tile_data(data):
    return int(data["n_scas"]) == 1 and int(data["n_tiles"]) == 1


def tile_neighbor_edges(data):
    """Return adjacent selected tile-index pairs using tile row/column metadata."""
    tile_ids = np.asarray(data.get("tile_ids", []), dtype=int)
    metadata = data.get("metadata", {})
    positions = []
    for tile_id in tile_ids:
        tile_metadata = metadata.get(int(tile_id))
        if tile_metadata is None:
            return np.empty((0, 2), dtype=int)
        positions.append(
            (
                int(tile_metadata["tile_row"]),
                int(tile_metadata["tile_col"]),
            )
        )
    edges = []
    for first in range(len(positions)):
        for second in range(first + 1, len(positions)):
            if sum(
                abs(positions[first][axis] - positions[second][axis])
                for axis in (0, 1)
            ) == 1:
                edges.append((first, second))
    return np.asarray(edges, dtype=int).reshape((-1, 2))


def aggregate_tile_sufficient_statistics(data):
    """Combine tile count moments by SCA when every tile shares one prediction."""
    counts = np.asarray(data["counts"])
    sums = np.asarray(data["sums"])
    sum_squared = np.asarray(data["sum_squared"])
    n_scas = int(data["n_scas"])
    n_tiles = int(data["n_tiles"])
    if counts.shape[0] != n_scas * n_tiles:
        raise ValueError("Tile sufficient statistics do not match the SCA/tile layout")
    if sums.shape != counts.shape or sum_squared.shape != counts.shape:
        raise ValueError("Tile count moments must have matching shapes")
    aggregated = dict(data)
    aggregated["counts"] = counts.reshape(n_scas, n_tiles, -1).sum(axis=1)
    aggregated["sums"] = sums.reshape(n_scas, n_tiles, -1).sum(axis=1)
    aggregated["sum_squared"] = sum_squared.reshape(
        n_scas, n_tiles, -1
    ).sum(axis=1)
    aggregated["sca_index"] = np.arange(n_scas, dtype=int)
    return aggregated


def run_map_initialization(model, initial_values, jax, init_to_value, seed, steps, learning_rate):
    """Use Adam through AutoDelta as an optional NUTS initialization path."""
    from numpyro.infer import SVI, Trace_ELBO, autoguide
    from numpyro.optim import Adam

    guide = autoguide.AutoDelta(
        model,
        init_loc_fn=init_to_value(values=initial_values),
    )
    svi = SVI(model, guide, Adam(step_size=learning_rate), Trace_ELBO())
    losses = []
    best_loss = np.inf
    best_params = None
    state = None
    checkpoint_steps = 100
    base_key = jax.random.PRNGKey(seed)
    for start in range(0, steps, checkpoint_steps):
        result = svi.run(
            jax.random.fold_in(base_key, start),
            min(checkpoint_steps, steps - start),
            progress_bar=False,
            init_state=state,
        )
        state = result.state
        chunk_losses = np.asarray(result.losses)
        losses.extend(chunk_losses.tolist())
        chunk_loss = float(chunk_losses[-1])
        if np.isfinite(chunk_loss) and chunk_loss < best_loss:
            best_loss = chunk_loss
            best_params = jax.tree_util.tree_map(
                lambda value: value.copy(), result.params
            )
    losses = np.asarray(losses)
    if best_params is None:
        raise ValueError("MAP initialization produced a non-finite objective")
    print(
        f"MAP initialization: {steps} steps; best loss={best_loss:.6g}; "
        f"final loss={losses[-1]:.6g}"
    )
    map_values = guide.median(best_params)
    return (
        {**initial_values, **map_values},
        {
            "optimizer": "Adam/AutoDelta",
            "converged": False,
            "status": -1,
            "message": "Checkpointed initializer; convergence was not assessed.",
            "iterations": int(steps),
            "function_evaluations": int(steps),
            "gradient_norm": np.nan,
            "objective": float(best_loss),
            "initial_objective": float(losses[0]),
            "parameter_step_norm": np.nan,
            "objective_history": losses,
        },
    )


def map_parameter_bounds(initial_values, jax):
    initial_pytree = {
        name: jax.numpy.asarray(value) for name, value in initial_values.items()
    }
    flat_initial, unravel = jax.flatten_util.ravel_pytree(initial_pytree)

    def lower_bound(name, value):
        shape = np.asarray(value).shape
        if name in ("initial_thickness_nm", "reset_initial_thickness_nm"):
            return np.full(shape, 1.0e-8)
        if name == "hgcdte_n":
            return np.full(shape, HGCDTE_LOWER[0])
        if name == "hgcdte_k":
            return np.full(shape, HGCDTE_LOWER[1])
        if name == "incidence_angle_deg":
            return np.zeros(shape)
        if name in ("pre_reset_ice_n", "post_reset_ice_n"):
            return np.full(shape, ICE_INDEX_LOWER)
        if name in ("pre_reset_ice_k", "post_reset_ice_k"):
            return np.zeros(shape)
        if name == "post_reset_ice_index":
            return np.full(shape, POST_RESET_ICE_INDEX_LOWER)
        if name == "sca_ar_opd_nm":
            return np.full(shape, -np.inf)
        if name == "complex_q_parameters":
            return np.broadcast_to(COMPLEX_Q_LOWER, shape).copy()
        if name in ("ar_sigma_raw", "ar_sca_sigma", "ar_tile_sigma"):
            return np.full(shape, 1.0e-8)
        return np.full(shape, -np.inf)

    def upper_bound(name, value):
        shape = np.asarray(value).shape
        if name in ("initial_thickness_nm", "reset_initial_thickness_nm"):
            return np.full(shape, 50.0 - 1.0e-8)
        if name == "hgcdte_n":
            return np.full(shape, HGCDTE_UPPER[0])
        if name == "incidence_angle_deg":
            return np.full(shape, INCIDENCE_ANGLE_MAX_DEG)
        if name in ("pre_reset_ice_n", "post_reset_ice_n"):
            return np.full(shape, ICE_INDEX_UPPER)
        if name in ("pre_reset_ice_k", "post_reset_ice_k"):
            return np.full(shape, ICE_EXTINCTION_UPPER)
        if name == "post_reset_ice_index":
            return np.full(shape, POST_RESET_ICE_INDEX_UPPER)
        if name == "hgcdte_k":
            return np.full(shape, HGCDTE_UPPER[1])
        if name == "sca_ar_opd_nm":
            return np.full(shape, np.inf)
        if name == "complex_q_parameters":
            return np.broadcast_to(COMPLEX_Q_UPPER, shape).copy()
        return np.full(shape, np.inf)

    flat_lower, _ = jax.flatten_util.ravel_pytree(
        {
            name: jax.numpy.asarray(lower_bound(name, value))
            for name, value in initial_values.items()
        }
    )
    flat_upper, _ = jax.flatten_util.ravel_pytree(
        {
            name: jax.numpy.asarray(upper_bound(name, value))
            for name, value in initial_values.items()
        }
    )
    return flat_initial, unravel, flat_lower, flat_upper


def map_parameter_scale_values(initial_values, args):
    """Return prior-width coordinate scales for MAP sample sites."""
    site_scales = {
        "ar_mu_raw": getattr(args, "ar_mean_prior_sigma", AR_MEAN_PRIOR_SIGMA),
        "ar_tile_raw": getattr(args, "ar_mean_prior_sigma", AR_MEAN_PRIOR_SIGMA),
        "ar_mu_z": 1.0,
        "ar_sigma_raw": getattr(args, "ar_tile_sigma_prior", AR_TILE_SIGMA_PRIOR),
        "ar_sca_sigma": getattr(args, "ar_sca_sigma_prior", AR_SCA_SIGMA_PRIOR),
        "ar_tile_sigma": getattr(args, "ar_tile_sigma_prior", AR_TILE_SIGMA_PRIOR),
        "ar_z": 1.0,
        "ar_sca_z": 1.0,
        "ar_tile_z": 1.0,
        "sca_ar_opd_nm": getattr(
            args, "sca_ar_opd_prior_sigma_nm", SCA_AR_OPD_PRIOR_SIGMA_NM
        ),
        "initial_thickness_nm": getattr(
            args, "initial_thickness_prior_sigma_nm", INITIAL_THICKNESS_PRIOR_SIGMA_NM
        ),
        "initial_thickness_log_nm": getattr(
            args, "tile_initial_thickness_log_sigma", 1.5
        ),
        "reset_initial_thickness_nm": getattr(
            args, "initial_thickness_prior_sigma_nm", INITIAL_THICKNESS_PRIOR_SIGMA_NM
        ),
        "reset_initial_thickness_log_nm": getattr(
            args, "tile_initial_thickness_log_sigma", 1.5
        ),
        "initial_thickness_pivot_z": 1.0,
        "early_time_log_scale": getattr(
            args, "early_time_scale_prior_sigma", EARLY_TIME_SCALE_PRIOR_SIGMA
        ),
        "np_log_rate_nodes": getattr(
            args, "trajectory_rate_prior_sigma_log", 0.75
        ),
        "np_log_rate_z": 1.0,
        "log_trajectory_scale": getattr(
            args, "trajectory_rate_prior_sigma_log", 0.75
        ),
        "trajectory_rate_tile_offsets": getattr(
            args, "tile_rate_prior_sigma_log", TRAJECTORY_TILE_RATE_PRIOR_SIGMA_LOG
        ),
        "trajectory_rate_tile_node_offsets": getattr(
            args, "tile_rate_prior_sigma_log", TRAJECTORY_TILE_RATE_PRIOR_SIGMA_LOG
        ),
        "trajectory_rate_tile_node_log_rates": getattr(
            args, "trajectory_rate_prior_sigma_log", TRAJECTORY_TILE_RATE_PRIOR_SIGMA_LOG
        ),
        "detector_log_thickness_nm": getattr(
            args,
            "detector_thickness_prior_sigma_log",
            DETECTOR_THICKNESS_PRIOR_SIGMA_LOG,
        ),
        "hgcdte_n": HGCDTE_PRIOR_SIGMA[0],
        "hgcdte_k": HGCDTE_PRIOR_SIGMA[1],
        "hgcdte_n_log_modulation": getattr(
            args, "hgcdte_modulation_prior_sigma_log", HGCDTE_MODULATION_PRIOR_SIGMA_LOG
        ),
        "hgcdte_k_log_modulation": getattr(
            args, "hgcdte_modulation_prior_sigma_log", HGCDTE_MODULATION_PRIOR_SIGMA_LOG
        ),
        "ice_n_log_modulation": getattr(
            args,
            "ice_n_modulation_prior_sigma_log",
            getattr(args, "ice_modulation_prior_sigma_log", ICE_MODULATION_PRIOR_SIGMA_LOG),
        )
        if getattr(args, "ice_n_modulation_prior_sigma_log", None) is not None
        else getattr(args, "ice_modulation_prior_sigma_log", ICE_MODULATION_PRIOR_SIGMA_LOG),
        "ice_k_log_modulation": getattr(
            args, "ice_modulation_prior_sigma_log", ICE_MODULATION_PRIOR_SIGMA_LOG
        ),
        "post_reset_tile_log_offset": getattr(
            args,
            "post_reset_tile_log_offset_prior_sigma",
            POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA,
        ),
        "absolute_hgcdte_n_log_scale": getattr(
            args,
            "absolute_hgcdte_scale_prior_sigma_log",
            ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG,
        ),
        "absolute_hgcdte_k_log_scale": getattr(
            args,
            "absolute_hgcdte_scale_prior_sigma_log",
            ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG,
        ),
        "complex_q_parameters": COMPLEX_Q_MAP_SCALES,
        "incidence_angle_deg": getattr(
            args,
            "incidence_angle_prior_sigma_deg",
            INCIDENCE_ANGLE_PRIOR_SIGMA_DEG,
        )
        if not getattr(args, "incidence_angle_prior_uniform", False)
        else INCIDENCE_ANGLE_MAX_DEG,
        "post_reset_ice_index": getattr(
            args,
            "post_reset_ice_index_prior_sigma",
            POST_RESET_ICE_INDEX_PRIOR_SIGMA,
        ),
        "pre_reset_ice_n": getattr(
            args, "ice_index_prior_sigma", ICE_INDEX_PRIOR_SIGMA
        ),
        "post_reset_ice_n": getattr(
            args, "ice_index_prior_sigma", ICE_INDEX_PRIOR_SIGMA
        ),
        "pre_reset_ice_k": getattr(
            args, "ice_extinction_prior_sigma", ICE_EXTINCTION_PRIOR_SIGMA
        ),
        "post_reset_ice_k": getattr(
            args, "ice_extinction_prior_sigma", ICE_EXTINCTION_PRIOR_SIGMA
        ),
        "conductive_log_thickness_nm": CONDUCTIVE_LOG_PRIOR_SIGMA,
        "conductive_log_plasma_strength": CONDUCTIVE_LOG_PRIOR_SIGMA,
        "conductive_log_damping_ratio": CONDUCTIVE_LOG_PRIOR_SIGMA,
        "qe_rbf_log_modulation": getattr(
            args, "qe_rbf_prior_sigma_log", QE_RBF_PRIOR_SIGMA_LOG
        ),
        "acm_step_log_offset": getattr(args, "acm_step_prior_sigma", 0.05),
        "late_acm_step_log_offset": getattr(
            args, "late_acm_step_prior_sigma", 0.02
        ),
        "np_node_raw": 1.0,
    }
    return {
        name: np.broadcast_to(
            np.asarray(site_scales.get(name, 1.0), dtype=float),
            np.asarray(value).shape,
        ).copy()
        for name, value in initial_values.items()
    }


def _tile_axis_for_site(name):
    return {
        "ar_z": 0,
        "ar_tile_raw": 1,
        "ar_tile_z": 1,
        "initial_thickness_log_nm": -1,
        "reset_initial_thickness_log_nm": -1,
        "post_reset_tile_log_offset": -1,
        "absolute_hgcdte_n_log_scale": -1,
        "absolute_hgcdte_k_log_scale": -1,
        "detector_log_thickness_nm": -1,
        "trajectory_rate_tile_offsets": -1,
        "trajectory_rate_tile_node_offsets": 1,
        "trajectory_rate_tile_node_log_rates": 1,
    }.get(name)


def warm_start_value_can_transfer(
    name, value, target, source_tile_ids=None, target_tile_ids=None
):
    """Return whether a source site can populate the target site shape."""
    value = np.asarray(value)
    target = np.asarray(target)
    if value.shape == target.shape:
        return True
    if source_tile_ids is None or target_tile_ids is None:
        return False
    axis = _tile_axis_for_site(name)
    if axis is None or value.ndim != target.ndim:
        return False
    axis %= value.ndim
    if value.shape[axis] != len(source_tile_ids) or target.shape[axis] != len(
        target_tile_ids
    ):
        return False
    return all(
        value.shape[index] == target.shape[index]
        for index in range(value.ndim)
        if index != axis
    )


def load_map_warm_start(
    path,
    initial_values,
    data,
    value_converters=None,
    allow_tile_subset=False,
    broadcast_tile_subset=False,
    allow_tile_template=False,
):
    """Load compatible MAP sites, retaining fresh defaults for new sites."""
    path = Path(path)
    samples_path = path / "posterior_samples.npz" if path.is_dir() else path
    if not samples_path.is_file():
        raise FileNotFoundError(f"MAP warm-start file does not exist: {samples_path}")
    with np.load(samples_path) as samples:
        if "map_only" not in samples or not bool(np.asarray(samples["map_only"]).item()):
            raise ValueError("MAP warm starts require a deterministic map-only fit")
        if "statistics_fingerprint" not in samples:
            raise ValueError("MAP warm start has no tile-statistics fingerprint")
        archived_fingerprint = str(
            np.asarray(samples["statistics_fingerprint"]).reshape(-1)[0]
        )
        source_tile_ids = None
        target_tile_ids = None
        if archived_fingerprint != statistics_fingerprint(data):
            if not allow_tile_subset:
                raise ValueError("MAP warm-start tile statistics do not match this fit")
            required_metadata = {
                "tile_ids",
                "sca_names",
                "statistics_row_fingerprints",
            }
            missing_metadata = required_metadata.difference(samples.files)
            if missing_metadata:
                raise ValueError(
                    "Tile-subset MAP warm starts require metadata: "
                    f"{sorted(missing_metadata)}"
                )
            source_tile_ids = np.asarray(samples["tile_ids"], dtype=int).reshape(-1)
            target_tile_ids = np.asarray(data["tile_ids"], dtype=int).reshape(-1)
            source_sca_names = np.asarray(samples["sca_names"]).astype(str).reshape(-1)
            target_sca_names = np.asarray(data["sca_names"]).astype(str).reshape(-1)
            if source_tile_ids.size == 0 or target_tile_ids.size == 0:
                raise ValueError("Tile-subset MAP warm starts require tile metadata")
            tile_template = (
                allow_tile_template
                and broadcast_tile_subset
                and source_tile_ids.size == 1
            )
            if not set(source_tile_ids).issubset(set(target_tile_ids)) and not tile_template:
                raise ValueError("MAP warm-start tiles are not a subset of the target tiles")
            target_sca_positions = {
                name: index for index, name in enumerate(target_sca_names)
            }
            if not set(source_sca_names).issubset(target_sca_positions):
                raise ValueError("MAP warm-start SCAs are not a subset of the target SCAs")
            source_row_fingerprints = np.asarray(
                samples["statistics_row_fingerprints"]
            ).astype(str).reshape(-1)
            target_row_fingerprints = statistics_row_fingerprints(data)
            source_row_count = source_sca_names.size * source_tile_ids.size
            target_row_count = target_sca_names.size * target_tile_ids.size
            if source_row_fingerprints.size != source_row_count:
                raise ValueError("MAP warm-start row metadata has an unexpected shape")
            if target_row_fingerprints.size != target_row_count:
                raise ValueError("Target tile statistics row metadata has an unexpected shape")
            target_tile_positions = {
                int(tile_id): index for index, tile_id in enumerate(target_tile_ids)
            }
            if not tile_template:
                for source_sca_index, sca_name in enumerate(source_sca_names):
                    target_sca_index = target_sca_positions[sca_name]
                    for source_tile_index, tile_id in enumerate(source_tile_ids):
                        source_row = source_sca_index * source_tile_ids.size + source_tile_index
                        target_row = (
                            target_sca_index * target_tile_ids.size
                            + target_tile_positions[int(tile_id)]
                        )
                        if source_row_fingerprints[source_row] != target_row_fingerprints[target_row]:
                            raise ValueError(
                                "MAP warm-start tile statistics do not match the target rows"
                            )

        warmed_values = {}
        value_converters = value_converters or {}
        for name, current_value in initial_values.items():
            source_name = name
            converter = None
            if name in value_converters:
                converter_candidates = value_converters[name]
                if (
                    isinstance(converter_candidates, tuple)
                    and len(converter_candidates) == 2
                    and isinstance(converter_candidates[0], str)
                ):
                    converter_candidates = [converter_candidates]
                for converted_source_name, converted_value in converter_candidates:
                    if converted_source_name in samples:
                        source_name = converted_source_name
                        converter = converted_value
                        break
            if source_name not in samples:
                warmed_values[name] = np.asarray(current_value).copy()
                continue
            value = np.asarray(samples[source_name], dtype=float)
            if value.shape and value.shape[0] == 1:
                value = value[0]
            if converter is not None:
                value = np.asarray(converter(value, samples), dtype=float)
            target_shape = np.asarray(current_value).shape
            if name == "acm_step_log_offset":
                if target_shape == () and value.size == 6:
                    value = np.asarray(np.mean(value))
                elif target_shape == (6,) and value.size == 1:
                    value = np.full(6, float(value.reshape(-1)[0]))
            if not warm_start_value_can_transfer(
                name,
                value,
                np.asarray(current_value),
                source_tile_ids,
                target_tile_ids,
            ):
                raise ValueError(
                    f"MAP warm-start site {name} has shape {value.shape}; "
                    f"expected {target_shape}"
                )
            if np.any(~np.isfinite(value)):
                raise ValueError(f"MAP warm-start site {name} is non-finite")
            warmed_values[name] = value.copy()
    return transfer_map_initial_values(
        warmed_values,
        initial_values,
        source_tile_ids=source_tile_ids,
        target_tile_ids=target_tile_ids,
        broadcast_tile_values=broadcast_tile_subset,
    )


def jitter_unconstrained_initial_params(initial_params, jax, seed, chains, scale):
    """Create per-chain starts in a small unconstrained neighborhood."""
    if chains < 1:
        raise ValueError("NUTS chain count must be positive")
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("NUTS initialization jitter must be finite and non-negative")
    base_key = jax.random.PRNGKey(seed)
    jittered = {}
    for index, (name, value) in enumerate(initial_params.items()):
        value = jax.numpy.asarray(value)
        batched_shape = (chains,) + value.shape
        if scale == 0.0:
            batched = jax.numpy.broadcast_to(value, batched_shape)
        else:
            noise = jax.random.normal(
                jax.random.fold_in(base_key, index),
                batched_shape,
                dtype=value.dtype,
            )
            batched = value + scale * noise
        jittered[name] = batched if chains > 1 else batched[0]
    return jittered


def build_nuts_initial_params(
    model, initial_values, jax, init_to_value, seed, chains, jitter
):
    """Transform a complete constrained start and jitter it per chain."""
    from numpyro.infer.util import initialize_model

    model_info = initialize_model(
        jax.random.PRNGKey(seed),
        model,
        init_strategy=init_to_value(values=initial_values),
    )
    return jitter_unconstrained_initial_params(
        model_info.param_info.z,
        jax,
        seed + 1,
        chains,
        jitter,
    )


def interior_nuts_initial_values(initial_values):
    """Move bounded scalar starts off exact Uniform support boundaries."""
    interior = dict(initial_values)
    if "incidence_angle_deg" in interior:
        interior["incidence_angle_deg"] = np.clip(
            np.asarray(interior["incidence_angle_deg"], dtype=float),
            1.0e-6,
            INCIDENCE_ANGLE_MAX_DEG - 1.0e-6,
        )
    return interior


def run_map_optimization(
    model,
    initial_values,
    jax,
    maxiter,
    tolerance=1.0e-9,
    ftol=0.0,
    parameter_scales=None,
    progress_interval=MAP_PROGRESS_INTERVAL,
):
    """Optimize the exact NumPyro joint log density in scaled L-BFGS-B coordinates."""
    from numpyro.infer.util import log_density
    from scipy.optimize import minimize

    flat_initial, unravel, flat_lower, flat_upper = map_parameter_bounds(
        initial_values, jax
    )
    if parameter_scales is None:
        flat_scales = np.ones_like(np.asarray(flat_initial, dtype=float))
    else:
        if set(parameter_scales) != set(initial_values):
            raise ValueError("MAP parameter scales must match the initial-value sites")
        scale_pytree = {
            name: jax.numpy.asarray(
                np.broadcast_to(
                    np.asarray(parameter_scales[name], dtype=float),
                    np.asarray(value).shape,
                )
            )
            for name, value in initial_values.items()
        }
        flat_scales, _ = jax.flatten_util.ravel_pytree(scale_pytree)
        flat_scales = np.asarray(flat_scales, dtype=float)
        if np.any(~np.isfinite(flat_scales)) or np.any(flat_scales <= 0.0):
            raise ValueError("MAP parameter scales must be finite and positive")

    flat_initial = np.asarray(flat_initial, dtype=float)
    flat_lower = np.asarray(flat_lower, dtype=float)
    flat_upper = np.asarray(flat_upper, dtype=float)
    scaled_initial = np.zeros_like(flat_initial)
    scaled_lower = (flat_lower - flat_initial) / flat_scales
    scaled_upper = (flat_upper - flat_initial) / flat_scales

    def negative_log_joint(flat_values):
        log_joint, _ = log_density(model, (), {}, unravel(flat_values))
        return -log_joint

    value_and_gradient = jax.jit(jax.value_and_grad(negative_log_joint))
    progress_interval = int(progress_interval)
    if progress_interval <= 0:
        raise ValueError("MAP progress interval must be positive")
    last_evaluation = {"values": None, "objective": None, "gradient": None}

    def scipy_objective(scaled_values):
        scaled_values = np.asarray(scaled_values, dtype=float)
        flat_values = flat_initial + flat_scales * scaled_values
        value, gradient = value_and_gradient(jax.numpy.asarray(flat_values))
        value = float(value)
        gradient = np.asarray(gradient, dtype=float)
        last_evaluation["values"] = scaled_values.copy()
        last_evaluation["objective"] = value
        last_evaluation["gradient"] = gradient.copy()
        return value, gradient * flat_scales

    def projected_gradient(flat_values, gradient, lower=flat_lower, upper=flat_upper):
        gradient = np.asarray(gradient, dtype=float).copy()
        bound_tolerance = 1.0e-10 * (1.0 + np.abs(flat_values))
        lower_active = np.isfinite(lower) & (
            flat_values <= lower + bound_tolerance
        )
        upper_active = np.isfinite(upper) & (
            flat_values >= upper - bound_tolerance
        )
        gradient[lower_active] = np.minimum(gradient[lower_active], 0.0)
        gradient[upper_active] = np.maximum(gradient[upper_active], 0.0)
        return gradient

    map_start_time = time.perf_counter()
    initial_objective, _ = scipy_objective(scaled_initial)
    initial_evaluation_seconds = time.perf_counter() - map_start_time
    objective_history = [initial_objective]

    def progress_prefix():
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.perf_counter() - map_start_time
        return f"[{timestamp} +{elapsed:.1f}s]"

    def print_map_progress(iteration):
        raw_gradient = last_evaluation["gradient"]
        scaled_gradient = raw_gradient * flat_scales
        print(
            f"{progress_prefix()} L-BFGS-B MAP progress: iteration={iteration}; "
            f"objective={last_evaluation['objective']:.6g}; "
            f"gradient_inf_norm={np.max(np.abs(raw_gradient)):.3g}; "
            f"scaled_gradient_inf_norm={np.max(np.abs(scaled_gradient)):.3g}",
            flush=True,
        )

    print_map_progress(0)

    def record_objective(flat_values):
        # SciPy has already evaluated the accepted point; avoid a duplicate
        # value-and-gradient launch just to record diagnostic history.
        if np.array_equal(flat_values, last_evaluation["values"]):
            objective_history.append(last_evaluation["objective"])
            iteration = len(objective_history) - 1
            if iteration % progress_interval == 0:
                print_map_progress(iteration)

    optimization_start = time.perf_counter()
    result = minimize(
        scipy_objective,
        scaled_initial,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(scaled_lower, scaled_upper)),
        callback=record_objective,
        options={
            "maxiter": int(maxiter),
            "ftol": float(ftol),
            "gtol": float(tolerance),
            "maxls": 50,
        },
    )
    optimization_seconds = time.perf_counter() - optimization_start
    if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
        raise ValueError("L-BFGS-B MAP optimization produced a non-finite objective")
    optimized_values = flat_initial + flat_scales * np.asarray(result.x, dtype=float)
    map_values = {
        name: np.asarray(value)
        for name, value in unravel(jax.numpy.asarray(optimized_values)).items()
    }
    scaled_gradient = np.asarray(result.jac, dtype=float)
    raw_gradient = scaled_gradient / flat_scales
    gradient_norm = float(np.max(np.abs(raw_gradient)))
    projected_gradient_norm = float(
        np.max(np.abs(projected_gradient(optimized_values, raw_gradient)))
    )
    scaled_projected_gradient_norm = float(
        np.max(
            np.abs(
                projected_gradient(
                    result.x, scaled_gradient, scaled_lower, scaled_upper
                )
            )
        )
    )
    gradient_converged = scaled_projected_gradient_norm <= tolerance
    diagnostics = {
        "optimizer": "L-BFGS-B",
        "converged": bool(result.success and gradient_converged),
        "gradient_converged": bool(gradient_converged),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "gradient_norm": gradient_norm,
        "projected_gradient_norm": projected_gradient_norm,
        "scaled_gradient_norm": float(np.max(np.abs(scaled_gradient))),
        "scaled_projected_gradient_norm": scaled_projected_gradient_norm,
        "parameter_scale_min": float(np.min(flat_scales)),
        "parameter_scale_max": float(np.max(flat_scales)),
        "function_tolerance": float(ftol),
        "objective": float(result.fun),
        "initial_objective": float(initial_objective),
        "parameter_step_norm": float(
            np.linalg.norm(optimized_values - flat_initial)
        ),
        "scaled_parameter_step_norm": float(
            np.linalg.norm(result.x - scaled_initial)
        ),
        "objective_history": np.asarray(objective_history, dtype=float),
        "initial_evaluation_seconds": float(initial_evaluation_seconds),
        "optimization_seconds": float(optimization_seconds),
    }
    print(
        f"{progress_prefix()} {diagnostics['optimizer']} MAP: "
        f"iterations={diagnostics['iterations']}; "
        f"objective={diagnostics['objective']:.6g}; "
        f"gradient_inf_norm={diagnostics['gradient_norm']:.3g}; "
        f"scaled_projected_gradient_inf_norm={diagnostics['scaled_projected_gradient_norm']:.3g}; "
        f"converged={diagnostics['converged']}; "
        f"first_evaluation_s={initial_evaluation_seconds:.2f}; "
        f"optimization_s={optimization_seconds:.2f}",
        flush=True,
    )
    return map_values, diagnostics


def _least_squares_prior_residuals(values, data, args, ar_prior_reference, jnp):
    """Return production prior and curvature residuals for the LS MAP path."""
    ar_reference, ar_lower, ar_upper = ar_prior_arrays(args.ar_layers)
    active_indices = ar_active_parameter_indices(
        args.ar_layers,
        args.fit_ar_absorption,
        getattr(args, "fixed_ar_indices", None),
    )
    ar_reference_raw_values = jnp.asarray(
        ar_reference_raw(
            ar_prior_reference,
            ar_lower,
            ar_upper,
            active_indices,
            ordered_ar_indices=getattr(args, "ordered_ar_indices", False),
        )
    )
    residuals = []

    def add_normal(name, center, scale):
        if name in values:
            residuals.append(
                ((jnp.asarray(values[name]) - jnp.asarray(center)) / float(scale)).reshape(-1)
            )

    if "ar_tile_raw" in values:
        add_normal(
            "ar_tile_raw",
            ar_reference_raw_values,
            args.ar_mean_prior_sigma,
        )
    if "ar_mu_raw" in values:
        add_normal("ar_mu_raw", ar_reference_raw_values, args.ar_mean_prior_sigma)
    if "ar_mu_z" in values:
        add_normal("ar_mu_z", 0.0, 1.0)
    if "ar_tile_z" in values:
        add_normal("ar_tile_z", 0.0, 1.0)
    if "initial_thickness_log_nm" in values:
        add_normal(
            "initial_thickness_log_nm",
            np.log(args.tile_initial_thickness_log_median_nm),
            args.tile_initial_thickness_log_sigma,
        )
    if "reset_initial_thickness_log_nm" in values:
        add_normal(
            "reset_initial_thickness_log_nm",
            np.log(args.tile_initial_thickness_log_median_nm),
            args.tile_initial_thickness_log_sigma,
        )
    if "absolute_hgcdte_n_log_scale" in values:
        add_normal(
            "absolute_hgcdte_n_log_scale",
            0.0,
            args.absolute_hgcdte_scale_prior_sigma_log,
        )
    if "absolute_hgcdte_k_log_scale" in values:
        add_normal(
            "absolute_hgcdte_k_log_scale",
            0.0,
            args.absolute_hgcdte_scale_prior_sigma_log,
        )
    if "post_reset_tile_log_offset" in values:
        add_normal(
            "post_reset_tile_log_offset",
            0.0,
            args.post_reset_tile_log_offset_prior_sigma,
        )
    if "incidence_angle_deg" in values and not args.incidence_angle_prior_uniform:
        add_normal(
            "incidence_angle_deg",
            0.0,
            args.incidence_angle_prior_sigma_deg,
        )
    trajectory_prior_rate = trajectory_prior_rate_for_data(data, args)
    trajectory_log_center = np.log(trajectory_prior_rate)
    if "log_trajectory_scale" in values:
        add_normal(
            "log_trajectory_scale",
            trajectory_log_center,
            args.trajectory_rate_prior_sigma_log,
        )
    if "np_log_rate_nodes" in values:
        add_normal(
            "np_log_rate_nodes",
            trajectory_log_center,
            args.trajectory_rate_prior_sigma_log,
        )
    if "trajectory_rate_tile_node_log_rates" in values:
        add_normal(
            "trajectory_rate_tile_node_log_rates",
            trajectory_log_center,
            args.trajectory_rate_prior_sigma_log,
        )
    if "np_log_rate_z" in values:
        add_normal("np_log_rate_z", 0.0, 1.0)
    if "np_node_raw" in values:
        add_normal("np_node_raw", 0.0, 1.0)
    if "acm_step_log_offset" in values:
        add_normal("acm_step_log_offset", 0.0, args.acm_step_prior_sigma)
    if "late_acm_step_log_offset" in values:
        add_normal("late_acm_step_log_offset", 0.0, args.late_acm_step_prior_sigma)
    if "additional_acm_step_log_offset" in values:
        add_normal(
            "additional_acm_step_log_offset",
            0.0,
            args.additional_acm_step_prior_sigma,
        )

    likelihood_scale = trajectory_rate_likelihood_scale(data, args)
    if "np_log_rate_nodes" in values and args.np_smoothness_weight > 0.0:
        curvature = jnp.asarray(rate_curvature_matrix(data["node_days"])) @ jnp.asarray(
            values["np_log_rate_nodes"]
        ).reshape(-1)
        residuals.append(
            curvature
            * jnp.sqrt(
                2.0
                * likelihood_scale
                * args.np_smoothness_weight
                / curvature.size
            )
        )
    if (
        "trajectory_rate_tile_node_log_rates" in values
        and args.tile_rate_trajectory_smoothness_weight > 0.0
    ):
        profiles = jnp.asarray(values["trajectory_rate_tile_node_log_rates"])
        curvature_matrix = jnp.asarray(rate_curvature_matrix(data["node_days"]))
        curvature = jnp.einsum("ij,...j->...i", curvature_matrix, profiles).reshape(-1)
        residuals.append(
            curvature
            * jnp.sqrt(
                2.0
                * likelihood_scale
                * args.tile_rate_trajectory_smoothness_weight
                / curvature.size
            )
        )
    if not residuals:
        return jnp.zeros(0)
    return jnp.concatenate(residuals)


def run_least_squares_map_optimization(
    model,
    initial_values,
    data,
    args,
    ar_prior_reference,
    jax,
    maxiter,
    tolerance=1.0e-9,
    parameter_scales=None,
):
    """Optimize the production SSE and Normal priors with trust-region LS."""
    from numpyro.handlers import substitute, trace
    from numpyro.infer.util import log_density
    from scipy.optimize import least_squares

    jnp = jax.numpy
    flat_initial, unravel, flat_lower, flat_upper = map_parameter_bounds(
        initial_values, jax
    )
    if parameter_scales is None:
        flat_scales = np.ones_like(np.asarray(flat_initial, dtype=float))
    else:
        scale_pytree = {
            name: jax.numpy.asarray(
                np.broadcast_to(
                    np.asarray(parameter_scales[name], dtype=float),
                    np.asarray(value).shape,
                )
            )
            for name, value in initial_values.items()
        }
        flat_scales, _ = jax.flatten_util.ravel_pytree(scale_pytree)
        flat_scales = np.asarray(flat_scales, dtype=float)
    flat_initial = np.asarray(flat_initial, dtype=float)
    flat_lower = np.asarray(flat_lower, dtype=float)
    flat_upper = np.asarray(flat_upper, dtype=float)
    scaled_lower = (flat_lower - flat_initial) / flat_scales
    scaled_upper = (flat_upper - flat_initial) / flat_scales
    likelihood_scale = jnp.asarray(
        np.sqrt(float(args.likelihood_weight) / data["n_tiles"])
        / float(args.log_noise_sigma)
    )
    counts = jnp.asarray(data["counts"])
    observed = jnp.asarray(data["sums"]) / jnp.maximum(counts, 1.0)

    def residuals(scaled_values):
        flat_values = jax.numpy.asarray(flat_initial) + jax.numpy.asarray(
            flat_scales
        ) * scaled_values
        values = unravel(flat_values)
        model_trace = trace(substitute(model, data=values)).get_trace()
        prediction = model_trace["_map_least_squares_prediction"]["value"]
        observed_values = observed
        if "post_reset_tile_log_offset" in values:
            if args.ice_reset_day is None:
                raise ValueError(
                    "Post-reset tile offsets require an ice reset day for LS MAP"
                )
            post_reset_mask = jnp.asarray(data["days"] >= args.ice_reset_day)
            observed_values = observed + (
                jnp.asarray(values["post_reset_tile_log_offset"]).reshape(-1, 1)
                * post_reset_mask[None, :]
            )
        likelihood_residual = (
            likelihood_scale
            * jnp.sqrt(counts)
            * (prediction - observed_values)
        ).reshape(-1)
        prior_residual = _least_squares_prior_residuals(
            values, data, args, ar_prior_reference, jax.numpy
        )
        return jnp.concatenate((likelihood_residual, prior_residual))

    residual_jit = jax.jit(residuals)
    jacobian_jit = jax.jit(jax.jacfwd(residuals))
    initial_scaled = np.zeros_like(flat_initial)
    initial_residual = np.asarray(residual_jit(initial_scaled), dtype=float)

    def scipy_residual(scaled_values):
        return np.asarray(residual_jit(jax.numpy.asarray(scaled_values)), dtype=float)

    def scipy_jacobian(scaled_values):
        return np.asarray(
            jacobian_jit(jax.numpy.asarray(scaled_values)), dtype=float
        )

    result = least_squares(
        scipy_residual,
        initial_scaled,
        jac=scipy_jacobian,
        bounds=(scaled_lower, scaled_upper),
        method="trf",
        x_scale=1.0,
        ftol=None,
        xtol=None,
        gtol=float(tolerance),
        max_nfev=int(maxiter),
        verbose=1,
    )
    optimized_scaled = np.asarray(result.x, dtype=float)
    optimized_values = flat_initial + flat_scales * optimized_scaled
    map_values = {
        name: np.asarray(value)
        for name, value in unravel(jax.numpy.asarray(optimized_values)).items()
    }
    final_residual = scipy_residual(optimized_scaled)
    scaled_gradient = scipy_jacobian(optimized_scaled).T @ final_residual
    raw_gradient = scaled_gradient / flat_scales
    objective = lambda values: -log_density(model, (), {}, unravel(values))[0]
    initial_objective = float(objective(jax.numpy.asarray(flat_initial)))
    final_objective = float(objective(jax.numpy.asarray(optimized_values)))
    gradient_norm = float(np.max(np.abs(raw_gradient)))
    scaled_gradient_norm = float(np.max(np.abs(scaled_gradient)))
    diagnostics = {
        "optimizer": "Trust-Region Least-Squares",
        "converged": bool(result.success and scaled_gradient_norm <= tolerance),
        "gradient_converged": bool(scaled_gradient_norm <= tolerance),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nfev),
        "function_evaluations": int(result.nfev),
        "gradient_norm": gradient_norm,
        "projected_gradient_norm": gradient_norm,
        "scaled_gradient_norm": scaled_gradient_norm,
        "scaled_projected_gradient_norm": scaled_gradient_norm,
        "parameter_scale_min": float(np.min(flat_scales)),
        "parameter_scale_max": float(np.max(flat_scales)),
        "function_tolerance": np.nan,
        "objective": final_objective,
        "initial_objective": initial_objective,
        "parameter_step_norm": float(
            np.linalg.norm(optimized_values - flat_initial)
        ),
        "objective_history": np.asarray(
            [initial_objective, final_objective], dtype=float
        ),
    }
    print(
        f"{diagnostics['optimizer']}: evaluations={diagnostics['function_evaluations']}; "
        f"objective={final_objective:.6g}; "
        f"scaled_gradient_inf_norm={scaled_gradient_norm:.3g}; "
        f"success={result.success}",
        flush=True,
    )
    return map_values, diagnostics


def run_jax_map_optimization(
    model,
    initial_values,
    jax,
    maxiter,
    tolerance=1.0e-9,
    parameter_scales=None,
):
    """Optimize the exact NumPyro joint log density with JAX-native L-BFGS-B."""
    from numpyro.infer.util import log_density
    import jaxopt

    flat_initial, unravel, flat_lower, flat_upper = map_parameter_bounds(
        initial_values, jax
    )
    if parameter_scales is None:
        flat_scales = np.ones_like(np.asarray(flat_initial, dtype=float))
    else:
        if set(parameter_scales) != set(initial_values):
            raise ValueError("MAP parameter scales must match the initial-value sites")
        scale_pytree = {
            name: jax.numpy.asarray(
                np.broadcast_to(
                    np.asarray(parameter_scales[name], dtype=float),
                    np.asarray(value).shape,
                )
            )
            for name, value in initial_values.items()
        }
        flat_scales, _ = jax.flatten_util.ravel_pytree(scale_pytree)
        flat_scales = np.asarray(flat_scales, dtype=float)
        if np.any(~np.isfinite(flat_scales)) or np.any(flat_scales <= 0.0):
            raise ValueError("MAP parameter scales must be finite and positive")
    flat_initial = np.asarray(flat_initial, dtype=float)
    flat_lower = np.asarray(flat_lower, dtype=float)
    flat_upper = np.asarray(flat_upper, dtype=float)
    scaled_initial = np.zeros_like(flat_initial)
    scaled_lower = (flat_lower - flat_initial) / flat_scales
    scaled_upper = (flat_upper - flat_initial) / flat_scales
    flat_initial_jax = jax.numpy.asarray(flat_initial)
    flat_scales_jax = jax.numpy.asarray(flat_scales)

    def negative_log_joint(scaled_values):
        flat_values = flat_initial_jax + flat_scales_jax * scaled_values
        log_joint, _ = log_density(model, (), {}, unravel(flat_values))
        return -log_joint

    initial_objective = float(negative_log_joint(jax.numpy.asarray(scaled_initial)))
    solver = jaxopt.LBFGSB(
        fun=negative_log_joint,
        maxiter=int(maxiter),
        tol=float(tolerance),
        maxls=50,
        history_size=10,
        jit=True,
        unroll=False,
    )
    result = solver.run(
        jax.numpy.asarray(scaled_initial),
        bounds=(jax.numpy.asarray(scaled_lower), jax.numpy.asarray(scaled_upper)),
    )
    scaled_values = np.asarray(jax.device_get(result.params), dtype=float)
    flat_values = flat_initial + flat_scales * scaled_values
    state = result.state
    scaled_gradient = np.asarray(jax.device_get(state.grad), dtype=float)
    raw_gradient = scaled_gradient / flat_scales
    lower = np.asarray(flat_lower, dtype=float)
    upper = np.asarray(flat_upper, dtype=float)
    projected = raw_gradient.copy()
    bound_tolerance = 1.0e-10 * (1.0 + np.abs(flat_values))
    lower_active = np.isfinite(lower) & (flat_values <= lower + bound_tolerance)
    upper_active = np.isfinite(upper) & (flat_values >= upper - bound_tolerance)
    projected[lower_active] = np.minimum(projected[lower_active], 0.0)
    projected[upper_active] = np.maximum(projected[upper_active], 0.0)
    scaled_projected = scaled_gradient.copy()
    scaled_bound_tolerance = 1.0e-10 * (1.0 + np.abs(scaled_values))
    scaled_lower_active = np.isfinite(scaled_lower) & (
        scaled_values <= scaled_lower + scaled_bound_tolerance
    )
    scaled_upper_active = np.isfinite(scaled_upper) & (
        scaled_values >= scaled_upper - scaled_bound_tolerance
    )
    scaled_projected[scaled_lower_active] = np.minimum(
        scaled_projected[scaled_lower_active], 0.0
    )
    scaled_projected[scaled_upper_active] = np.maximum(
        scaled_projected[scaled_upper_active], 0.0
    )
    gradient_norm = float(np.max(np.abs(raw_gradient)))
    projected_gradient_norm = float(np.max(np.abs(projected)))
    scaled_projected_gradient_norm = float(np.max(np.abs(scaled_projected)))
    failed_linesearch = bool(jax.device_get(state.failed_linesearch))
    gradient_converged = scaled_projected_gradient_norm <= tolerance
    converged = gradient_converged and not failed_linesearch
    iterations = int(jax.device_get(state.iter_num))
    function_evaluations = int(jax.device_get(state.num_fun_eval))
    objective = float(jax.device_get(state.value))
    diagnostics = {
        "optimizer": "JAX-LBFGS-B",
        "converged": bool(converged),
        "gradient_converged": bool(gradient_converged),
        "status": 0 if converged else 1,
        "message": (
            "Projected-gradient tolerance reached."
            if converged
            else (
                "JAXopt line search failed."
                if failed_linesearch
                else "Maximum iterations reached before projected-gradient convergence."
            )
        ),
        "iterations": iterations,
        "function_evaluations": function_evaluations,
        "gradient_norm": gradient_norm,
        "projected_gradient_norm": projected_gradient_norm,
        "scaled_gradient_norm": float(np.max(np.abs(scaled_gradient))),
        "scaled_projected_gradient_norm": scaled_projected_gradient_norm,
        "parameter_scale_min": float(np.min(flat_scales)),
        "parameter_scale_max": float(np.max(flat_scales)),
        "function_tolerance": np.nan,
        "objective": objective,
        "initial_objective": initial_objective,
        "parameter_step_norm": float(
            np.linalg.norm(flat_values - flat_initial)
        ),
        "scaled_parameter_step_norm": float(
            np.linalg.norm(scaled_values - scaled_initial)
        ),
        "objective_history": np.asarray([initial_objective, objective], dtype=float),
    }
    map_values = {
        name: np.asarray(value)
        for name, value in unravel(jax.numpy.asarray(flat_values)).items()
    }
    print(
        f"{diagnostics['optimizer']} MAP: iterations={iterations}; "
        f"objective={objective:.6g}; "
        f"gradient_inf_norm={gradient_norm:.3g}; "
        f"projected_gradient_inf_norm={projected_gradient_norm:.3g}; "
        f"converged={diagnostics['converged']}"
    )
    return map_values, diagnostics


def run_trust_exact_map_optimization(
    model,
    initial_values,
    jax,
    maxiter,
    tolerance=1.0e-9,
    parameter_scales=None,
):
    """Polish a warm MAP state with an exact-Hessian trust-region solve."""
    from numpyro.infer.util import log_density
    from scipy.optimize import minimize

    flat_initial, unravel, flat_lower, flat_upper = map_parameter_bounds(
        initial_values, jax
    )
    if parameter_scales is not None:
        raise ValueError(
            "Trust-Exact MAP optimization requires raw sample-site coordinates; "
            "use --map-parameter-scaling none"
        )
    flat_scales = np.ones_like(np.asarray(flat_initial, dtype=float))

    flat_initial = np.asarray(flat_initial, dtype=float)
    flat_lower = np.asarray(flat_lower, dtype=float)
    flat_upper = np.asarray(flat_upper, dtype=float)
    scaled_initial = np.zeros_like(flat_initial)
    scaled_lower = (flat_lower - flat_initial) / flat_scales
    scaled_upper = (flat_upper - flat_initial) / flat_scales
    flat_initial_jax = jax.numpy.asarray(flat_initial)
    flat_scales_jax = jax.numpy.asarray(flat_scales)
    scaled_initial_jax = jax.numpy.asarray(scaled_initial)

    bound_tolerance = 1.0e-10 * (1.0 + np.abs(flat_initial))
    fixed = (
        np.isfinite(flat_lower)
        & (flat_initial <= flat_lower + bound_tolerance)
    ) | (
        np.isfinite(flat_upper)
        & (flat_initial >= flat_upper - bound_tolerance)
    )
    free_indices = np.flatnonzero(~fixed)
    free_indices_jax = jax.numpy.asarray(free_indices)

    def full_scaled_values(free_values):
        return scaled_initial_jax.at[free_indices_jax].set(free_values)

    def negative_log_joint(scaled_values):
        flat_values = flat_initial_jax + flat_scales_jax * scaled_values
        log_joint, _ = log_density(model, (), {}, unravel(flat_values))
        return -log_joint

    value_and_gradient = jax.jit(jax.value_and_grad(negative_log_joint))
    hessian = jax.jit(jax.hessian(negative_log_joint))
    initial_objective = float(
        negative_log_joint(jax.numpy.asarray(scaled_initial))
    )
    map_start_time = time.perf_counter()

    if free_indices.size:

        def scipy_objective(free_values):
            value, gradient = value_and_gradient(
                full_scaled_values(jax.numpy.asarray(free_values))
            )
            return float(value), np.asarray(gradient, dtype=float)[free_indices]

        def scipy_hessian(free_values):
            full_hessian = np.asarray(
                hessian(full_scaled_values(jax.numpy.asarray(free_values))),
                dtype=float,
            )
            return full_hessian[np.ix_(free_indices, free_indices)]

        optimization_start = time.perf_counter()
        result = minimize(
            scipy_objective,
            np.asarray(scaled_initial[free_indices], dtype=float),
            jac=True,
            hess=scipy_hessian,
            method="trust-exact",
            options={"gtol": float(tolerance), "maxiter": int(maxiter)},
        )
        optimization_seconds = time.perf_counter() - optimization_start
        optimized_scaled = np.asarray(scaled_initial, dtype=float)
        optimized_scaled[free_indices] = np.asarray(result.x, dtype=float)
        status = int(result.status)
        message = str(result.message)
        iterations = int(result.nit)
        function_evaluations = int(result.nfev)
        optimizer_success = bool(result.success)
    else:
        optimization_seconds = 0.0
        optimized_scaled = np.asarray(scaled_initial, dtype=float)
        status = 0
        message = "All MAP coordinates were fixed at active bounds."
        iterations = 0
        function_evaluations = 1
        optimizer_success = True

    optimized_values = flat_initial + flat_scales * optimized_scaled
    if np.any(optimized_values < flat_lower) or np.any(optimized_values > flat_upper):
        raise ValueError("Trust-region MAP optimization left a parameter bound")
    final_objective, scaled_gradient = value_and_gradient(
        jax.numpy.asarray(optimized_scaled)
    )
    scaled_gradient = np.asarray(scaled_gradient, dtype=float)
    raw_gradient = scaled_gradient / flat_scales

    def projected_gradient(values, gradient, lower, upper):
        gradient = np.asarray(gradient, dtype=float).copy()
        bound_tolerance = 1.0e-10 * (1.0 + np.abs(values))
        lower_active = np.isfinite(lower) & (values <= lower + bound_tolerance)
        upper_active = np.isfinite(upper) & (values >= upper - bound_tolerance)
        gradient[lower_active] = np.minimum(gradient[lower_active], 0.0)
        gradient[upper_active] = np.maximum(gradient[upper_active], 0.0)
        return gradient

    projected = projected_gradient(optimized_values, raw_gradient, flat_lower, flat_upper)
    scaled_projected = projected_gradient(
        optimized_scaled, scaled_gradient, scaled_lower, scaled_upper
    )
    gradient_norm = float(np.max(np.abs(raw_gradient)))
    projected_gradient_norm = float(np.max(np.abs(projected)))
    scaled_projected_gradient_norm = float(np.max(np.abs(scaled_projected)))
    gradient_converged = scaled_projected_gradient_norm <= tolerance
    diagnostics = {
        "optimizer": "Trust-Exact",
        "converged": bool(optimizer_success and gradient_converged),
        "gradient_converged": bool(gradient_converged),
        "status": status,
        "message": message,
        "iterations": iterations,
        "function_evaluations": function_evaluations,
        "gradient_norm": gradient_norm,
        "projected_gradient_norm": projected_gradient_norm,
        "scaled_gradient_norm": float(np.max(np.abs(scaled_gradient))),
        "scaled_projected_gradient_norm": scaled_projected_gradient_norm,
        "parameter_scale_min": float(np.min(flat_scales)),
        "parameter_scale_max": float(np.max(flat_scales)),
        "function_tolerance": np.nan,
        "objective": float(final_objective),
        "initial_objective": initial_objective,
        "parameter_step_norm": float(
            np.linalg.norm(optimized_values - flat_initial)
        ),
        "scaled_parameter_step_norm": float(
            np.linalg.norm(optimized_scaled - scaled_initial)
        ),
        "objective_history": np.asarray(
            [initial_objective, float(final_objective)], dtype=float
        ),
        "initial_evaluation_seconds": float(
            time.perf_counter() - map_start_time - optimization_seconds
        ),
        "optimization_seconds": float(optimization_seconds),
    }
    map_values = {
        name: np.asarray(value)
        for name, value in unravel(jax.numpy.asarray(optimized_values)).items()
    }
    print(
        f"{diagnostics['optimizer']} MAP: "
        f"iterations={diagnostics['iterations']}; "
        f"objective={diagnostics['objective']:.6g}; "
        f"gradient_inf_norm={diagnostics['gradient_norm']:.3g}; "
        f"scaled_projected_gradient_inf_norm="
        f"{diagnostics['scaled_projected_gradient_norm']:.3g}; "
        f"converged={diagnostics['converged']}",
        flush=True,
    )
    return map_values, diagnostics


def scope_label(sca_names):
    names = []
    for value in np.asarray(sca_names).reshape(-1):
        if isinstance(value, bytes):
            value = value.decode()
        names.append(str(value))
    if len(names) == 1:
        return names[0]
    return f"{len(names)} WFI SCAs"


def require_numpyro():
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS, init_to_value
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "The NumPyro TMM fit requires jax and numpyro; use the bulge_env environment"
        ) from error
    return jax, jnp, numpyro, dist, MCMC, NUTS, init_to_value


def select_tile_ids(tile_ids, max_tiles):
    tile_ids = np.asarray(sorted(tile_ids), dtype=int)
    if max_tiles is None or max_tiles >= len(tile_ids):
        return tile_ids
    if max_tiles < 1:
        raise ValueError("max_tiles must be positive")
    indices = np.linspace(0, len(tile_ids) - 1, max_tiles).round().astype(int)
    return tile_ids[np.unique(indices)]


def statistics_paths(path):
    if path.is_dir():
        paths = sorted(path.glob("wfi*_tile_log_statistics_*.csv"))
        if not paths:
            raise ValueError(f"No completed SCA statistics CSVs found in {path}")
        return paths
    return [path]


def detector_name_from_statistics_path(path):
    prefix = path.name.split("_tile_log_statistics_", 1)[0]
    return prefix.upper()


def qe_columns_for_scas(sca_names, qe_fieldnames):
    """Map WFI detector labels to the corresponding public SCU QE columns."""
    field_to_column = {}
    for column, fieldname in enumerate(qe_fieldnames):
        normalized = str(fieldname).strip().upper()
        if normalized in field_to_column:
            raise ValueError(f"Duplicate QE field name: {fieldname}")
        field_to_column[normalized] = column

    columns = []
    for value in np.asarray(sca_names).reshape(-1):
        detector = str(value).strip().upper()
        suffix = detector[3:] if detector.startswith("WFI") else ""
        if not suffix.isdigit() or not 1 <= int(suffix) <= 18:
            raise ValueError(f"Expected a WFI01-WFI18 detector label, got {value!r}")
        fieldname = f"SCU {int(suffix)}"
        if fieldname not in field_to_column:
            raise ValueError(f"QE table does not contain the required column {fieldname}")
        columns.append(field_to_column[fieldname])

    columns = np.asarray(columns, dtype=int)
    if len(np.unique(columns)) != len(columns):
        raise ValueError("SCA statistics contain duplicate detector-to-QE assignments")
    return columns


def load_tile_statistics(
    path,
    max_tiles=None,
    max_groups=None,
    min_observation_day=None,
    additional_acm_step_repeat=None,
    pre_reset_acm_step_repeat=None,
):
    if max_groups is not None and max_groups < 1:
        raise ValueError("max_groups must be positive")
    if min_observation_day is not None and (
        not np.isfinite(min_observation_day) or min_observation_day < 0.0
    ):
        raise ValueError("Minimum observation day must be finite and non-negative")
    if additional_acm_step_repeat is not None and additional_acm_step_repeat < 0:
        raise ValueError("Additional ACM step repeat must be non-negative")
    if pre_reset_acm_step_repeat is not None and pre_reset_acm_step_repeat < 0:
        raise ValueError("Pre-reset ACM step repeat must be non-negative")
    paths = statistics_paths(path)
    sca_counts = []
    sca_sums = []
    sca_sum_squared = []
    count_fields = (
        "sum_reference_counts",
        "sum_squared_reference_counts",
        "sum_current_counts",
        "sum_squared_current_counts",
        "sum_reference_current_counts",
        "min_reference_count",
        "max_reference_count",
        "min_current_count",
        "max_current_count",
    )
    sca_count_statistics = {field: [] for field in count_fields}
    count_statistics_available = None
    tile_ids = None
    metadata = None
    groups = None
    reference = None
    sca_names = []
    for statistics_path in paths:
        with statistics_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No tile rows found in {statistics_path}")
        has_count_statistics = set(count_fields).issubset(rows[0])
        if count_statistics_available is None:
            count_statistics_available = has_count_statistics
        elif count_statistics_available != has_count_statistics:
            raise ValueError(
                "SCA statistics must consistently include or omit count moments"
            )
        all_tile_ids = sorted({int(row["tile_id"]) for row in rows})
        selected_tile_ids = select_tile_ids(all_tile_ids, max_tiles)
        if tile_ids is None:
            tile_ids = selected_tile_ids
        elif not np.array_equal(tile_ids, selected_tile_ids):
            raise ValueError("SCA statistics do not have the same tile IDs")
        selected = set(tile_ids.tolist())
        rows = [row for row in rows if int(row["tile_id"]) in selected]
        if min_observation_day is not None:
            group_min_days = {}
            for row in rows:
                group = int(row["group"])
                day = float(row["days"])
                group_min_days[group] = min(group_min_days.get(group, day), day)
            retained_groups = {
                group
                for group, day in group_min_days.items()
                if day >= min_observation_day
            }
            rows = [row for row in rows if int(row["group"]) in retained_groups]
        current_groups = sorted({int(row["group"]) for row in rows})
        if not current_groups:
            raise ValueError(
                "Observation-day cutoff removed every available observation group"
            )
        if max_groups is not None:
            current_groups = current_groups[:max_groups]
            selected_groups = set(current_groups)
            rows = [row for row in rows if int(row["group"]) in selected_groups]
        if groups is None:
            groups = current_groups
        elif groups != current_groups:
            raise ValueError("SCA statistics do not have the same observation groups")
        group_to_index = {group: index for index, group in enumerate(groups)}
        tile_to_index = {tile_id: index for index, tile_id in enumerate(tile_ids)}
        n_tiles = len(tile_ids)
        n_groups = len(groups)
        counts = np.zeros((n_tiles, n_groups), dtype=float)
        sums = np.zeros((n_tiles, n_groups), dtype=float)
        sum_squared = np.zeros((n_tiles, n_groups), dtype=float)
        count_statistics = (
            {
                field: np.zeros((n_tiles, n_groups), dtype=float)
                for field in count_fields
            }
            if count_statistics_available
            else None
        )
        led = np.zeros(n_groups, dtype=int)
        repeat = np.zeros(n_groups, dtype=int)
        days = np.zeros(n_groups, dtype=float)
        exposure_times = [None] * n_groups
        current_metadata = {}
        for row in rows:
            tile_id = int(row["tile_id"])
            tile_index = tile_to_index[tile_id]
            group = int(row["group"])
            group_index = group_to_index[group]
            counts[tile_index, group_index] = float(row["n_pair_valid"])
            sums[tile_index, group_index] = float(row["sum_log_change"])
            sum_squared[tile_index, group_index] = float(row["sum_squared_log_change"])
            if count_statistics is not None:
                for field in count_fields:
                    count_statistics[field][tile_index, group_index] = float(
                        row[field]
                    )
            led[group_index] = int(row["led"])
            repeat[group_index] = int(row["repeat"])
            days[group_index] = float(row["days"])
            exposure_times[group_index] = row["exposure_start_time"]
            current_metadata[tile_id] = {
                "tile_row": int(row["tile_row"]),
                "tile_col": int(row["tile_col"]),
                "row_start": int(row["row_start"]),
                "row_stop_exclusive": int(row["row_stop_exclusive"]),
                "col_start": int(row["col_start"]),
                "col_stop_exclusive": int(row["col_stop_exclusive"]),
            }
        if min_observation_day is None and not np.array_equal(
            current_groups, np.arange(n_groups)
        ):
            raise ValueError(
                "Tile statistics groups must span zero through the final group"
            )
        if np.any(counts <= 0.0):
            raise ValueError("Selected tiles contain empty observation groups")
        if reference is None:
            reference = (led, repeat, days, exposure_times)
            metadata = current_metadata
        else:
            reference_led, reference_repeat, reference_days, reference_times = reference
            if not np.array_equal(led, reference_led) or not np.array_equal(
                repeat, reference_repeat
            ):
                raise ValueError("SCA statistics have inconsistent LED/epoch metadata")
            if not np.allclose(days, reference_days) or exposure_times != reference_times:
                raise ValueError("SCA statistics have inconsistent time metadata")
        sca_counts.append(counts)
        sca_sums.append(sums)
        sca_sum_squared.append(sum_squared)
        if count_statistics is not None:
            for field in count_fields:
                sca_count_statistics[field].append(count_statistics[field])
        sca_names.append(detector_name_from_statistics_path(statistics_path))

    led, repeat, days, exposure_times = reference
    counts = np.concatenate(sca_counts, axis=0)
    sums = np.concatenate(sca_sums, axis=0)
    sum_squared = np.concatenate(sca_sum_squared, axis=0)
    combined_count_statistics = {
        field: np.concatenate(sca_count_statistics[field], axis=0)
        for field in count_fields
    } if count_statistics_available else {}
    n_scas = len(sca_names)
    n_tiles = len(tile_ids)
    n_groups = len(groups)

    first_by_led = []
    for led_value in range(1, 7):
        led_indices = np.flatnonzero(led == led_value)
        if not led_indices.size:
            raise ValueError(
                f"Observation-day cutoff removed every observation for LED {led_value}"
            )
        first_by_led.append(led_indices[0])
    first_by_led = np.asarray(first_by_led, dtype=int)
    after_event = np.asarray(
        [parse_mast_time(value) >= parse_mast_time(DEFAULT_STEP_TIME) for value in exposure_times],
        dtype=bool,
    )
    step_design = after_event.astype(float) - after_event[first_by_led[led - 1]]
    repeat_values = np.asarray(sorted(set(repeat.tolist())), dtype=int)
    after_late_acm_step = (repeat == repeat_values[-1]).astype(float)
    late_step_design = after_late_acm_step - after_late_acm_step[
        first_by_led[led - 1]
    ]
    if additional_acm_step_repeat is None:
        additional_step_design = np.zeros(len(repeat), dtype=float)
    else:
        before_additional_step = repeat < int(additional_acm_step_repeat)
        after_additional_step = repeat >= int(additional_acm_step_repeat)
        if not np.any(before_additional_step) or not np.any(after_additional_step):
            raise ValueError(
                "Additional ACM step repeat must have observations on both sides"
            )
        additional_step_design = after_additional_step.astype(float)
        additional_step_design -= additional_step_design[
            first_by_led[led - 1]
        ]
    if pre_reset_acm_step_repeat is None:
        pre_reset_step_design = np.zeros(len(repeat), dtype=float)
    else:
        before_pre_reset_step = repeat < int(pre_reset_acm_step_repeat)
        after_pre_reset_step = repeat >= int(pre_reset_acm_step_repeat)
        if not np.any(before_pre_reset_step) or not np.any(after_pre_reset_step):
            raise ValueError(
                "Pre-reset ACM step repeat must have observations on both sides"
            )
        pre_reset_step_design = after_pre_reset_step.astype(float)
        pre_reset_step_design -= pre_reset_step_design[
            first_by_led[led - 1]
        ]
    repeat_to_index = {repeat_value: index for index, repeat_value in enumerate(repeat_values)}
    repeat_index = np.asarray(
        [repeat_to_index[repeat_value] for repeat_value in repeat],
        dtype=int,
    )
    node_days = np.asarray(
        [np.mean(days[repeat == repeat_value]) for repeat_value in repeat_values],
        dtype=float,
    )
    if len(node_days) < 3 or np.any(np.diff(node_days) <= 0.0):
        raise ValueError("Repeat nodes must contain at least three strictly increasing times")
    result = {
        "tile_ids": tile_ids,
        "metadata": metadata,
        "counts": counts,
        "sums": sums,
        "sum_squared": sum_squared,
        "groups": np.asarray(groups, dtype=int),
        "led": led,
        "repeat": repeat,
        "days": days,
        "first_by_led": first_by_led,
        "step_design": step_design,
        "late_step_design": late_step_design,
        "additional_step_design": additional_step_design,
        "pre_reset_step_design": pre_reset_step_design,
        "additional_acm_step_repeat": (
            -1
            if additional_acm_step_repeat is None
            else int(additional_acm_step_repeat)
        ),
        "pre_reset_acm_step_repeat": (
            -1
            if pre_reset_acm_step_repeat is None
            else int(pre_reset_acm_step_repeat)
        ),
        "exposure_times": exposure_times,
        "repeat_values": repeat_values,
        "repeat_index": repeat_index,
        "node_days": node_days,
        "sca_names": np.asarray(sca_names),
        "sca_index": np.repeat(np.arange(n_scas, dtype=int), n_tiles),
        "n_scas": n_scas,
        "n_tiles": n_tiles,
        }
    result.update(combined_count_statistics)
    return result


def _fixed_correction_space(rows):
    spaces = {
        str(row.get("correction_space", "log_ratio")).strip().lower() or "log_ratio"
        for row in rows
    }
    if len(spaces) != 1 or not spaces.issubset({"log_ratio", "counts"}):
        raise ValueError(f"Unsupported fixed-correction spaces: {sorted(spaces)}")
    return next(iter(spaces))


def _anchor_count_correction_to_led_references(data, correction):
    correction = np.asarray(correction, dtype=float)
    led = np.asarray(data["led"], dtype=int).reshape(-1)
    first_by_led = np.asarray(data["first_by_led"], dtype=int).reshape(-1)
    if first_by_led.shape != (6,) or np.any(first_by_led < 0):
        raise ValueError("Count corrections require one valid reference group per LED")
    if correction.ndim == 1:
        reference = correction[first_by_led[led - 1]]
    elif correction.ndim == 2:
        reference = correction[:, first_by_led[led - 1]]
    else:
        raise ValueError("Fixed count corrections must have one or two dimensions")
    return correction - reference


def _apply_count_moment_corrections(data, absolute_correction, tile_aware):
    required = {
        "sum_reference_counts",
        "sum_squared_reference_counts",
        "sum_current_counts",
        "sum_squared_current_counts",
        "sum_reference_current_counts",
        "min_reference_count",
        "max_reference_count",
        "min_current_count",
        "max_current_count",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(
            "Count-space fixed corrections require saved count moments: "
            f"{missing}"
        )
    absolute_correction = np.asarray(absolute_correction, dtype=float)
    led = np.asarray(data["led"], dtype=int).reshape(-1)
    first_by_led = np.asarray(data["first_by_led"], dtype=int).reshape(-1)
    if tile_aware:
        reference_correction = absolute_correction[:, first_by_led[led - 1]]
    else:
        reference_correction = absolute_correction[first_by_led[led - 1]]
    reference_factor = np.exp(reference_correction)
    current_factor = np.exp(absolute_correction)
    corrected = {}
    corrected["sum_reference_counts"] = (
        np.asarray(data["sum_reference_counts"], dtype=float)
        * (reference_factor if tile_aware else reference_factor[None, :])
    )
    corrected["sum_squared_reference_counts"] = (
        np.asarray(data["sum_squared_reference_counts"], dtype=float)
        * (reference_factor if tile_aware else reference_factor[None, :]) ** 2
    )
    corrected["sum_current_counts"] = (
        np.asarray(data["sum_current_counts"], dtype=float)
        * (current_factor if tile_aware else current_factor[None, :])
    )
    corrected["sum_squared_current_counts"] = (
        np.asarray(data["sum_squared_current_counts"], dtype=float)
        * (current_factor if tile_aware else current_factor[None, :]) ** 2
    )
    corrected["sum_reference_current_counts"] = (
        np.asarray(data["sum_reference_current_counts"], dtype=float)
        * (reference_factor if tile_aware else reference_factor[None, :])
        * (current_factor if tile_aware else current_factor[None, :])
    )
    reference_scale = reference_factor if tile_aware else reference_factor[None, :]
    current_scale = current_factor if tile_aware else current_factor[None, :]
    for field in ("min_reference_count", "max_reference_count"):
        corrected[field] = np.asarray(data[field], dtype=float) * reference_scale
    for field in ("min_current_count", "max_current_count"):
        corrected[field] = np.asarray(data[field], dtype=float) * current_scale
    return corrected


def _apply_group_fixed_corrections(data, rows, by_group):
    data_groups = np.asarray(
        data.get("groups", np.arange(len(data["days"]), dtype=int)), dtype=int
    ).reshape(-1)
    if len(data_groups) != len(data["days"]):
        raise ValueError("Tile statistics groups do not match observation days")
    if len(np.unique(data_groups)) != len(data_groups):
        raise ValueError("Tile statistics contain duplicate observation groups")
    missing_groups = sorted(set(data_groups.tolist()).difference(by_group))
    if missing_groups:
        raise ValueError(
            "Fixed corrections are missing selected groups: "
            f"{missing_groups}"
        )
    correction_days = np.asarray(
        [float(by_group[group]["days"]) for group in data_groups], dtype=float
    )
    if not np.allclose(correction_days, data["days"], rtol=0.0, atol=1.0e-9):
        raise ValueError("Fixed-correction days do not match tile statistics")
    log_gain_correction = np.asarray(
        [float(by_group[group]["log_gain_correction"]) for group in data_groups],
        dtype=float,
    )
    acm_steps = np.asarray(
        [float(by_group[group]["acm_step_log_offset"]) for group in data_groups],
        dtype=float,
    )
    late_acm_steps = np.asarray(
        [
            float(by_group[group]["late_acm_step_log_offset"])
            for group in data_groups
        ],
        dtype=float,
    )
    additional_acm_steps = np.asarray(
        [
            float(by_group[group].get("additional_acm_step_log_offset") or 0.0)
            for group in data_groups
        ],
        dtype=float,
    )
    pre_reset_acm_steps = np.asarray(
        [
            float(by_group[group].get("pre_reset_acm_step_log_offset") or 0.0)
            for group in data_groups
        ],
        dtype=float,
    )
    if not np.allclose(acm_steps, acm_steps[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("Fixed ACM step must be shared across all groups")
    if not np.allclose(late_acm_steps, late_acm_steps[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("Fixed late ACM step must be shared across all groups")
    if not np.allclose(
        additional_acm_steps, additional_acm_steps[0], rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("Fixed additional ACM step must be shared across all groups")
    if not np.allclose(
        pre_reset_acm_steps, pre_reset_acm_steps[0], rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("Fixed pre-reset ACM step must be shared across all groups")
    additional_step_design = data.get("additional_step_design")
    if additional_step_design is None:
        if np.any(additional_acm_steps != 0.0):
            raise ValueError(
                "Fixed corrections contain an additional ACM step, but tile data "
                "has no additional_step_design"
            )
        additional_step_design = np.zeros(len(data_groups), dtype=float)
    additional_step_design = np.asarray(additional_step_design, dtype=float).reshape(-1)
    if len(additional_step_design) != len(data_groups):
        raise ValueError("Additional ACM step design does not match observation groups")
    pre_reset_step_design = data.get("pre_reset_step_design")
    if pre_reset_step_design is None:
        if np.any(pre_reset_acm_steps != 0.0):
            raise ValueError(
                "Fixed corrections contain a pre-reset ACM step, but tile data "
                "has no pre_reset_step_design"
            )
        pre_reset_step_design = np.zeros(len(data_groups), dtype=float)
    pre_reset_step_design = np.asarray(pre_reset_step_design, dtype=float).reshape(-1)
    if len(pre_reset_step_design) != len(data_groups):
        raise ValueError(
            "Pre-reset ACM step design does not match observation groups"
        )
    total_log_correction = (
        log_gain_correction
        - acm_steps[0] * np.asarray(data["step_design"], dtype=float)
        - late_acm_steps[0] * np.asarray(data["late_step_design"], dtype=float)
        - additional_acm_steps[0] * additional_step_design
        - pre_reset_acm_steps[0] * pre_reset_step_design
    )
    if "count_log_correction" in rows[0]:
        total_log_correction = np.asarray(
            [float(by_group[group]["count_log_correction"]) for group in data_groups],
            dtype=float,
        )
    return (
        log_gain_correction,
        float(acm_steps[0]),
        float(late_acm_steps[0]),
        float(additional_acm_steps[0]),
        float(pre_reset_acm_steps[0]),
        total_log_correction,
    )


def _apply_tile_fixed_corrections(data, rows):
    """Apply one shared-ACM correction set for every SCA/tile."""
    fieldnames = set(rows[0])
    has_detector = "detector" in fieldnames
    sca_names = np.asarray(data.get("sca_names", ["WFI01"])).reshape(-1)
    n_scas = int(data.get("n_scas", len(sca_names)))
    n_tiles = int(data["n_tiles"])
    n_rows, n_groups = np.asarray(data["counts"]).shape
    if n_rows != n_scas * n_tiles:
        raise ValueError("Tile statistics do not match the SCA/tile layout")
    if not has_detector and n_scas > 1:
        raise ValueError(
            "Tile-aware fixed corrections for multiple SCAs require a detector column"
        )

    by_key = {}
    for row in rows:
        detector = (
            str(row["detector"]).strip().upper()
            if has_detector
            else str(sca_names[0]).strip().upper()
        )
        key = (detector, int(row["tile_id"]), int(row["group"]))
        if key in by_key:
            raise ValueError(f"Duplicate tile fixed correction for {key}")
        by_key[key] = row

    data_groups = np.asarray(
        data.get("groups", np.arange(n_groups, dtype=int)), dtype=int
    ).reshape(-1)
    if len(data_groups) != n_groups:
        raise ValueError("Tile statistics groups do not match correction groups")
    tile_ids = np.asarray(data["tile_ids"], dtype=int).reshape(-1)
    row_detectors = np.repeat(
        [str(value).strip().upper() for value in sca_names], n_tiles
    )
    row_tiles = np.tile(tile_ids, n_scas)
    log_gain_correction = np.empty((n_rows, n_groups), dtype=float)
    correction_days = np.empty((n_rows, n_groups), dtype=float)
    acm_steps = np.empty(n_rows, dtype=float)
    late_acm_steps = np.empty(n_rows, dtype=float)
    additional_acm_steps = np.empty(n_rows, dtype=float)
    pre_reset_acm_steps = np.empty(n_rows, dtype=float)
    for row_index, (detector, tile_id) in enumerate(zip(row_detectors, row_tiles)):
        selected = [
            by_key.get((detector, int(tile_id), int(group)))
            for group in data_groups
        ]
        if any(row is None for row in selected):
            missing = [
                int(group)
                for group, row in zip(data_groups, selected)
                if row is None
            ]
            raise ValueError(
                f"Fixed corrections are missing {detector} tile {tile_id} groups "
                f"{missing}"
            )
        correction_days[row_index] = [float(row["days"]) for row in selected]
        log_gain_correction[row_index] = [
            float(row["log_gain_correction"]) for row in selected
        ]
        acm_values = np.asarray(
            [float(row["acm_step_log_offset"]) for row in selected], dtype=float
        )
        late_values = np.asarray(
            [float(row["late_acm_step_log_offset"]) for row in selected], dtype=float
        )
        additional_values = np.asarray(
            [
                float(row.get("additional_acm_step_log_offset") or 0.0)
                for row in selected
            ],
            dtype=float,
        )
        pre_reset_values = np.asarray(
            [
                float(row.get("pre_reset_acm_step_log_offset") or 0.0)
                for row in selected
            ],
            dtype=float,
        )
        if not np.allclose(acm_values, acm_values[0], rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"Fixed ACM step must be shared across {detector} tile {tile_id}"
            )
        if not np.allclose(late_values, late_values[0], rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "Fixed late ACM step must be shared across "
                f"{detector} tile {tile_id}"
            )
        if not np.allclose(
            additional_values, additional_values[0], rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                "Fixed additional ACM step must be shared across "
                f"{detector} tile {tile_id}"
            )
        if not np.allclose(
            pre_reset_values, pre_reset_values[0], rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                "Fixed pre-reset ACM step must be shared across "
                f"{detector} tile {tile_id}"
            )
        acm_steps[row_index] = acm_values[0]
        late_acm_steps[row_index] = late_values[0]
        additional_acm_steps[row_index] = additional_values[0]
        pre_reset_acm_steps[row_index] = pre_reset_values[0]
    if not np.allclose(
        correction_days,
        np.asarray(data["days"], dtype=float)[None, :],
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("Fixed-correction days do not match tile statistics")
    additional_step_design = data.get("additional_step_design")
    if additional_step_design is None:
        if np.any(additional_acm_steps != 0.0):
            raise ValueError(
                "Fixed corrections contain an additional ACM step, but tile data "
                "has no additional_step_design"
            )
        additional_step_design = np.zeros(len(data_groups), dtype=float)
    additional_step_design = np.asarray(additional_step_design, dtype=float).reshape(-1)
    if len(additional_step_design) != len(data_groups):
        raise ValueError("Additional ACM step design does not match observation groups")
    pre_reset_step_design = data.get("pre_reset_step_design")
    if pre_reset_step_design is None:
        if np.any(pre_reset_acm_steps != 0.0):
            raise ValueError(
                "Fixed corrections contain a pre-reset ACM step, but tile data "
                "has no pre_reset_step_design"
            )
        pre_reset_step_design = np.zeros(len(data_groups), dtype=float)
    pre_reset_step_design = np.asarray(pre_reset_step_design, dtype=float).reshape(-1)
    if len(pre_reset_step_design) != len(data_groups):
        raise ValueError(
            "Pre-reset ACM step design does not match observation groups"
        )
    total_log_correction = (
        log_gain_correction
        - acm_steps[:, None] * np.asarray(data["step_design"], dtype=float)[None, :]
        - late_acm_steps[:, None]
        * np.asarray(data["late_step_design"], dtype=float)[None, :]
        - additional_acm_steps[:, None] * additional_step_design[None, :]
        - pre_reset_acm_steps[:, None] * pre_reset_step_design[None, :]
    )
    if "count_log_correction" in rows[0]:
        count_log_correction = np.empty((n_rows, n_groups), dtype=float)
        for row_index, (detector, tile_id) in enumerate(zip(row_detectors, row_tiles)):
            count_log_correction[row_index] = [
                float(
                    by_key[(detector, int(tile_id), int(group))][
                        "count_log_correction"
                    ]
                )
                for group in data_groups
            ]
        total_log_correction = count_log_correction
    return (
        log_gain_correction,
        acm_steps,
        late_acm_steps,
        additional_acm_steps,
        pre_reset_acm_steps,
        total_log_correction,
    )


def apply_fixed_corrections(data, path):
    """Apply fixed gain and shared ACM corrections to log-ratio statistics."""
    required = {
        "group",
        "days",
        "log_gain_correction",
        "acm_step_log_offset",
        "late_acm_step_log_offset",
    }
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No fixed-correction rows found in {path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            f"Fixed-correction file is missing columns: {sorted(missing)}"
        )
    tile_aware = "tile_id" in rows[0]
    correction_space = _fixed_correction_space(rows)
    if tile_aware:
        (
            log_gain_correction,
            acm_steps,
            late_acm_steps,
            additional_acm_steps,
            pre_reset_acm_steps,
            total_log_correction,
        ) = _apply_tile_fixed_corrections(data, rows)
    else:
        by_group = {}
        for row in rows:
            group = int(row["group"])
            if group in by_group:
                raise ValueError(f"Duplicate fixed correction for group {group}")
            by_group[group] = row
        (
            log_gain_correction,
            acm_steps,
            late_acm_steps,
            additional_acm_steps,
            pre_reset_acm_steps,
            total_log_correction,
        ) = _apply_group_fixed_corrections(data, rows, by_group)
    absolute_total_log_correction = np.asarray(total_log_correction, dtype=float)
    corrected = dict(data)
    if correction_space == "counts":
        count_moments = _apply_count_moment_corrections(
            data, absolute_total_log_correction, tile_aware
        )
        corrected.update(count_moments)
        log_gain_correction = _anchor_count_correction_to_led_references(
            data, log_gain_correction
        )
        total_log_correction = _anchor_count_correction_to_led_references(
            data, absolute_total_log_correction
        )
    counts = np.asarray(data["counts"], dtype=float)
    sums = np.asarray(data["sums"], dtype=float)
    correction_for_stats = (
        total_log_correction
        if tile_aware
        else total_log_correction[None, :]
    )
    corrected["sums"] = sums + counts * correction_for_stats
    corrected["sum_squared"] = np.asarray(data["sum_squared"], dtype=float) + (
        2.0 * correction_for_stats * sums
        + counts * correction_for_stats**2
    )
    corrected["fixed_corrections_applied"] = True
    corrected["fixed_corrections_path"] = str(Path(path).expanduser().resolve())
    corrected["fixed_log_gain_correction"] = log_gain_correction
    corrected["fixed_corrections_tile_aware"] = bool(tile_aware)
    corrected["fixed_correction_space"] = correction_space
    corrected["fixed_acm_step_log_offset"] = float(np.median(acm_steps))
    corrected["fixed_late_acm_step_log_offset"] = float(np.median(late_acm_steps))
    corrected["fixed_additional_acm_step_log_offset"] = float(
        np.median(additional_acm_steps)
    )
    corrected["fixed_pre_reset_acm_step_log_offset"] = float(
        np.median(pre_reset_acm_steps)
    )
    corrected["fixed_acm_step_log_offset_by_row"] = np.asarray(acm_steps)
    corrected["fixed_late_acm_step_log_offset_by_row"] = np.asarray(late_acm_steps)
    corrected["fixed_additional_acm_step_log_offset_by_row"] = np.asarray(
        additional_acm_steps
    )
    corrected["fixed_pre_reset_acm_step_log_offset_by_row"] = np.asarray(
        pre_reset_acm_steps
    )
    corrected["fixed_total_log_correction"] = total_log_correction
    return corrected


def early_time_scale_configuration(data, cutoff_day):
    """Map early repeat nodes to shared scale-parameter indices."""
    if cutoff_day is None:
        return None
    node_days = np.asarray(data["node_days"], dtype=float)
    early_node_indices = np.flatnonzero(node_days <= float(cutoff_day))
    if not early_node_indices.size:
        raise ValueError("Early-time cutoff does not include an observation epoch")
    node_scale_index = np.full(len(node_days), -1, dtype=int)
    node_scale_index[early_node_indices] = np.arange(len(early_node_indices))
    observation_scale_index = node_scale_index[
        np.asarray(data["repeat_index"], dtype=int)
    ]
    return {
        "observation_scale_index": observation_scale_index,
        "node_indices": early_node_indices,
        "repeat": np.asarray(data["repeat_values"], dtype=int)[early_node_indices],
        "days": node_days[early_node_indices],
    }


def load_response_context(
    model_path,
    sed_mode,
    wavelength_stride,
    sca_names,
    wavelength_step_nm=None,
    ice_optical_constants_source=None,
    use_hgcdte_optical_constants=False,
    sed_weighting="photon",
    absolute_substrate_power=False,
):
    if wavelength_stride < 1:
        raise ValueError("wavelength_stride must be positive")
    if wavelength_step_nm is not None and (
        not np.isfinite(wavelength_step_nm) or wavelength_step_nm <= 0.0
    ):
        raise ValueError("wavelength_step_nm must be positive and finite")
    spectra = load_sed(SED_URL)
    qe_wavelengths, qe_values, qe_fieldnames = load_qe(QE_URL)
    del model_path  # Retained for compatibility; production TMM does not use its values.
    if wavelength_step_nm is None:
        wavelength_nodes = [] if absolute_substrate_power else [
            np.asarray([qe_wavelengths[0], qe_wavelengths[-1]])
        ]
        for spectrum in spectra.values():
            spectrum_wavelengths = np.asarray(spectrum["wavelength"])
            wavelength_nodes.append(
                spectrum_wavelengths
                if absolute_substrate_power
                else spectrum_wavelengths[
                    (spectrum_wavelengths >= qe_wavelengths[0])
                    & (spectrum_wavelengths <= qe_wavelengths[-1])
                ]
            )
        model_wavelengths = np.unique(np.concatenate(wavelength_nodes))
    else:
        wavelength_min = (
            min(np.min(spectrum["wavelength"]) for spectrum in spectra.values())
            if absolute_substrate_power
            else qe_wavelengths[0]
        )
        wavelength_max = (
            max(np.max(spectrum["wavelength"]) for spectrum in spectra.values())
            if absolute_substrate_power
            else qe_wavelengths[-1]
        )
        model_wavelengths = np.arange(
            wavelength_min,
            wavelength_max + 0.5 * wavelength_step_nm,
            wavelength_step_nm,
            dtype=float,
        )
        model_wavelengths = model_wavelengths[
            model_wavelengths <= wavelength_max
        ]
        if model_wavelengths.size == 0 or model_wavelengths[-1] < wavelength_max:
            model_wavelengths = np.append(model_wavelengths, wavelength_max)
    model_wavelengths = model_wavelengths[::wavelength_stride]
    wavelength_max = (
        max(np.max(spectrum["wavelength"]) for spectrum in spectra.values())
        if absolute_substrate_power
        else qe_wavelengths[-1]
    )
    if model_wavelengths[-1] < wavelength_max:
        model_wavelengths = np.append(model_wavelengths, wavelength_max)
    qe_columns = qe_columns_for_scas(sca_names, qe_fieldnames)
    integrations = [
        prepare_response_integration(
            model_wavelengths,
            spectra,
            qe_wavelengths,
            qe_values,
            sed_mode,
            qe_column=int(qe_column),
            sed_weighting=sed_weighting,
            include_qe=not absolute_substrate_power,
        )
        for qe_column in qe_columns
    ]
    model_weights = np.stack([integration[0] for integration in integrations])
    if absolute_substrate_power:
        active_wavelengths = np.any(model_weights != 0.0, axis=(0, 1))
        if not np.any(active_wavelengths):
            raise ValueError(
                "Absolute substrate-power integration has no nonzero wavelength weights"
            )
        model_wavelengths = model_wavelengths[active_wavelengths]
        model_weights = model_weights[:, :, active_wavelengths]
    context = {
        "wavelengths": model_wavelengths,
        "model_weights": model_weights,
        "outside_output": np.stack([integration[1] for integration in integrations]),
        "clean_output": np.stack([integration[2] for integration in integrations]),
        "qe_columns": qe_columns,
        "qe_fieldnames": np.asarray(qe_fieldnames),
        "sed_weighting": np.asarray(sed_weighting),
        "absolute_substrate_power": bool(absolute_substrate_power),
    }
    if ice_optical_constants_source is not None:
        ice_wavelengths, ice_n, ice_k = load_asw_optical_constants(
            ice_optical_constants_source
        )
        context["ice_n"] = np.interp(
            model_wavelengths, ice_wavelengths, ice_n, left=ice_n[0], right=ice_n[-1]
        )
        context["ice_k"] = np.interp(
            model_wavelengths, ice_wavelengths, ice_k, left=0.0, right=0.0
        )
        context["ice_optical_constants_source"] = np.asarray(
            str(ice_optical_constants_source)
        )
        context["ice_modulation_basis"] = ice_modulation_basis(model_wavelengths)
    if use_hgcdte_optical_constants:
        context["hgcdte_n"], context["hgcdte_k"] = (
            hgcdte_reference_optical_constants(model_wavelengths)
        )
        context["hgcdte_modulation_basis"] = hgcdte_modulation_basis(
            model_wavelengths
        )
        context["hgcdte_optical_constants_source"] = np.asarray(
            HGCDTE_OPTICAL_CONSTANTS_SOURCE
        )
    return context


def build_prediction_function(
    data,
    context,
    row_indices=None,
    fit_hgcdte=False,
    fit_hgcdte_modulation=False,
    fit_conductive_layer=False,
    finite_hgcdte_detector=False,
    detector_backside_index=1.0,
    fit_qe_rbf=False,
    qe_rbf_node_wavelengths_nm=None,
    qe_rbf_width_nm=QE_RBF_WIDTH_NM,
    use_asw_optical_constants=False,
    fit_ice_optical_modulation=False,
    use_hgcdte_optical_constants=False,
    incidence_cone_half_angle_deg=0.0,
    incidence_pupil_half_angle_deg=0.0,
    incidence_chief_ray_tilts_deg=None,
    fit_incidence_angle=False,
    early_time_cutoff_day=None,
    share_ar_across_rows=False,
    fit_late_acm_step=False,
    fit_additional_acm_step=False,
    absolute_substrate_power=None,
    share_polarization_work=False,
    post_reset_ice_reset_day=None,
    post_reset_log_offset_day=None,
    pre_reset_ice_index=None,
    fit_complex_q=False,
):
    jax, jnp, _, _, _, _, _ = require_numpyro()
    if absolute_substrate_power is None:
        absolute_substrate_power = bool(context.get("absolute_substrate_power", False))
    absolute_substrate_power = bool(absolute_substrate_power)
    if incidence_cone_half_angle_deg and incidence_pupil_half_angle_deg:
        raise ValueError("Legacy incidence cone and 2D pupil modes are mutually exclusive")
    fit_incidence_angle = bool(fit_incidence_angle)
    if fit_incidence_angle and float(incidence_cone_half_angle_deg) != 0.0:
        raise ValueError(
            "Fitted incidence angle requires the pupil/chief-ray geometry, not a solid-angle cone"
        )
    if fit_hgcdte and (fit_hgcdte_modulation or use_hgcdte_optical_constants):
        raise ValueError(
            "Fitted constant HgCdTe and wavelength-dependent HgCdTe constants "
            "are mutually exclusive"
        )
    if fit_hgcdte_modulation and not use_hgcdte_optical_constants:
        raise ValueError(
            "HgCdTe modulation requires wavelength-dependent HgCdTe constants"
        )
    if fit_ice_optical_modulation and not use_asw_optical_constants:
        raise ValueError(
            "Ice optical modulation requires wavelength-dependent ASW constants"
        )
    if fit_ice_optical_modulation and (
        post_reset_ice_reset_day is not None or pre_reset_ice_index is not None
    ):
        raise ValueError(
            "Ice optical modulation cannot be combined with segment-specific ice indices"
        )
    if finite_hgcdte_detector and fit_hgcdte_modulation:
        raise ValueError(
            "Finite HgCdTe detectors do not support free wavelength modulation"
        )
    if absolute_substrate_power and finite_hgcdte_detector:
        raise ValueError(
            "Absolute substrate power requires a semi-infinite terminal HgCdTe boundary"
        )
    if absolute_substrate_power and fit_qe_rbf:
        raise ValueError("Absolute substrate power cannot use a fitted QE modulation")
    if absolute_substrate_power and use_hgcdte_optical_constants is False:
        raise ValueError(
            "Absolute substrate power requires wavelength-dependent HgCdTe constants"
        )
    if post_reset_ice_reset_day is not None:
        if use_asw_optical_constants:
            raise ValueError(
                "Scalar segment ice indices cannot replace wavelength-dependent ASW constants"
            )
        if not np.isfinite(post_reset_ice_reset_day):
            raise ValueError("Post-reset ice reset day must be finite")
        if not np.any(
            np.asarray(data.get("days", []), dtype=float) >= post_reset_ice_reset_day
        ):
            raise ValueError("Post-reset ice reset day has no post-reset observations")
    if post_reset_log_offset_day is not None:
        if not np.isfinite(post_reset_log_offset_day):
            raise ValueError("Post-reset log-offset day must be finite")
        if not np.any(
            np.asarray(data.get("days", []), dtype=float) >= post_reset_log_offset_day
        ):
            raise ValueError("Post-reset log-offset day has no post-reset observations")
    if (
        not np.isfinite(detector_backside_index)
        or detector_backside_index <= 0.0
    ):
        raise ValueError("Detector backside index must be positive and finite")
    if early_time_cutoff_day is not None and (
        not np.isfinite(early_time_cutoff_day) or early_time_cutoff_day <= 0.0
    ):
        raise ValueError("Early-time scale cutoff must be positive and finite")
    wavelengths = jnp.asarray(context["wavelengths"])
    model_weights = jnp.asarray(context["model_weights"])
    outside_output = jnp.asarray(context["outside_output"])
    clean_output = jnp.asarray(context["clean_output"])
    n_scas = int(data["n_scas"])
    all_sca_index = np.asarray(data["sca_index"], dtype=int)
    if row_indices is None:
        row_indices = np.arange(len(all_sca_index), dtype=int)
    else:
        row_indices = np.asarray(row_indices, dtype=int).reshape(-1)
        if row_indices.size == 0 or np.any(row_indices < 0) or np.any(
            row_indices >= len(all_sca_index)
        ):
            raise ValueError("Prediction row indices are outside the tile data")
    sca_index = jnp.asarray(all_sca_index[row_indices], dtype=jnp.int32)
    if incidence_chief_ray_tilts_deg is None:
        incidence_chief_ray_tilts_deg = np.zeros((len(all_sca_index), 2), dtype=float)
    incidence_chief_ray_tilts_deg = np.asarray(
        incidence_chief_ray_tilts_deg, dtype=float
    )
    if incidence_chief_ray_tilts_deg.shape != (len(all_sca_index), 2):
        raise ValueError("Chief-ray tilts must have one x/y pair per optical row")
    selected_chief_ray_tilts = incidence_chief_ray_tilts_deg[row_indices]
    if fit_incidence_angle and np.any(incidence_chief_ray_tilts_deg != 0.0):
        raise ValueError(
            "Fitted incidence angle cannot be combined with fixed chief-ray tilts"
        )
    use_pupil_incidence = bool(
        float(incidence_pupil_half_angle_deg) > 0.0
        or np.any(selected_chief_ray_tilts != 0.0)
        or fit_incidence_angle
    )
    use_incidence_cone = bool(
        float(incidence_cone_half_angle_deg) > 0.0
        or use_pupil_incidence
        or fit_incidence_angle
    )
    if fit_incidence_angle:
        local_directions, incidence_weights_np = incidence_pupil_quadrature(
            incidence_pupil_half_angle_deg
        )
        # Validate the hard angle bound before tracing any sampled geometry.
        incidence_pupil_ray_sines(
            incidence_pupil_half_angle_deg,
            np.asarray([[INCIDENCE_ANGLE_MAX_DEG, 0.0]]),
        )
        local_directions = jnp.asarray(local_directions, dtype=float)
        local_x = local_directions[:, 0]
        local_y = local_directions[:, 1]
        local_z = local_directions[:, 2]

        def fitted_incidence_sines(angle_deg):
            angle_rad = jnp.asarray(angle_deg) * (jnp.pi / 180.0)
            sine_angle = jnp.sin(angle_rad)
            cosine_angle = jnp.cos(angle_rad)
            direction_x = local_x * cosine_angle + local_z * sine_angle
            direction_y = local_y
            sine = jnp.sqrt(jnp.maximum(direction_x * direction_x + direction_y * direction_y, 0.0))
            return sine[None, :]

        incidence_weights = jnp.asarray(incidence_weights_np, dtype=float)
        selected_incidence_sine = jnp.broadcast_to(
            fitted_incidence_sines(0.0),
            (len(row_indices), local_directions.shape[0]),
        )
    elif use_pupil_incidence:
        selected_incidence_sine, incidence_weights = incidence_pupil_ray_sines(
            incidence_pupil_half_angle_deg, selected_chief_ray_tilts
        )
    else:
        incidence_angles_deg, incidence_weights = incidence_cone_quadrature(
            incidence_cone_half_angle_deg
        )
        selected_incidence_sine = np.broadcast_to(
            np.sin(np.deg2rad(incidence_angles_deg))[None, :],
            (len(row_indices), len(incidence_angles_deg)),
        ).copy()
    incidence_weights = jnp.asarray(incidence_weights, dtype=float)
    single_incidence_ray = int(incidence_weights.shape[0]) == 1
    incidence_geometry_varies_by_row = not np.allclose(
        selected_incidence_sine, selected_incidence_sine[:1], rtol=0.0, atol=1.0e-14
    )
    selected_incidence_sine = jnp.asarray(selected_incidence_sine, dtype=float)
    if model_weights.ndim == 2:
        model_weights = jnp.broadcast_to(model_weights[None, :], (n_scas,) + model_weights.shape)
    if outside_output.ndim == 1:
        outside_output = jnp.broadcast_to(outside_output[None, :], (n_scas,) + outside_output.shape)
    if clean_output.ndim == 1:
        clean_output = jnp.broadcast_to(clean_output[None, :], (n_scas,) + clean_output.shape)
    if (
        model_weights.ndim != 3
        or outside_output.ndim != 2
        or clean_output.ndim != 2
        or model_weights.shape[0] != n_scas
        or outside_output.shape[0] != n_scas
        or clean_output.shape[0] != n_scas
    ):
        raise ValueError("Response integration arrays must have one entry per SCA")
    led_index = jnp.asarray(data["led"] - 1, dtype=jnp.int32)
    first_by_led = jnp.asarray(data["first_by_led"], dtype=jnp.int32)
    step_design = jnp.asarray(data["step_design"])
    late_step_design = jnp.asarray(
        data["late_step_design"]
        if "late_step_design" in data
        else np.zeros(len(data["led"]))
    )
    additional_step_design = jnp.asarray(
        data.get(
            "additional_step_design",
            np.zeros(len(data["led"])),
        )
    )
    early_time_scale_index = None
    if early_time_cutoff_day is not None:
        early_time_scale_index = jnp.asarray(
            early_time_scale_configuration(
                data, early_time_cutoff_day
            )["observation_scale_index"],
            dtype=jnp.int32,
        )
    shared_sca_response_inputs = n_scas == 1 and np.all(all_sca_index[row_indices] == 0)
    if shared_sca_response_inputs:
        # A single-SCA fit uses the same LED integration weights for every row.
        # Keep them shared instead of expanding them to row x group x wavelength.
        selected_model_weights = model_weights[0]
        selected_outside_output = outside_output[0]
        selected_clean_output = clean_output[0]
    else:
        row_model_weights = model_weights[sca_index]
        row_outside_output = outside_output[sca_index]
        row_clean_output = clean_output[sca_index]
        selected_model_weights = row_model_weights[:, led_index, :]
        selected_outside_output = row_outside_output[:, led_index]
        selected_clean_output = row_clean_output[:, led_index]
    if absolute_substrate_power and not np.allclose(
        np.asarray(selected_outside_output), 0.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            "Absolute substrate power cannot normalize unsupported source wavelengths"
        )
    if use_asw_optical_constants:
        if "ice_n" not in context or "ice_k" not in context:
            raise ValueError("ASW optical constants are missing from the response context")
        ice_index = jnp.asarray(context["ice_n"]) + 1j * jnp.asarray(
            context["ice_k"]
        )
        if fit_ice_optical_modulation:
            if "ice_modulation_basis" not in context:
                raise ValueError("Ice modulation basis is missing from the response context")
            ice_modulation_basis_value = jnp.asarray(
                context["ice_modulation_basis"]
            )
            ice_modulation_node_count = int(ice_modulation_basis_value.shape[1])
    else:
        ice_index = ICE_INDEX_DEFAULT + 0.0j
    if use_hgcdte_optical_constants:
        if (
            "hgcdte_n" not in context
            or "hgcdte_k" not in context
            or (
                fit_hgcdte_modulation
                and "hgcdte_modulation_basis" not in context
            )
        ):
            raise ValueError(
                "HgCdTe optical constants are missing from the response context"
            )
        wavelength_dependent_substrate_index = jnp.asarray(
            context["hgcdte_n"]
        ) + 1j * jnp.asarray(context["hgcdte_k"])
        if fit_hgcdte_modulation:
            hgcdte_modulation_basis_value = jnp.asarray(
                context["hgcdte_modulation_basis"]
            )
            hgcdte_modulation_node_count = int(
                hgcdte_modulation_basis_value.shape[1]
            )
    else:
        wavelength_dependent_substrate_index = None
    if absolute_substrate_power:
        if "counts" not in data or "sums" not in data:
            raise ValueError(
                "Absolute substrate power requires observed sufficient statistics "
                "for per-LED baseline offsets"
            )
        selected_counts = np.asarray(data["counts"])[row_indices]
        selected_sums = np.asarray(data["sums"])[row_indices]
        baseline_offsets = np.asarray(
            [
                np.average(
                    selected_sums[:, group] / selected_counts[:, group],
                    weights=selected_counts[:, group],
                )
                for group in np.asarray(data["first_by_led"], dtype=int)
            ],
            dtype=float,
        )
        baseline_offsets = jnp.asarray(baseline_offsets)
    else:
        baseline_offsets = jnp.zeros(6)
    all_group_indices = jnp.arange(led_index.shape[0], dtype=jnp.int32)
    if post_reset_ice_reset_day is None:
        post_reset_group_mask = None
    else:
        post_reset_group_mask = jnp.asarray(
            np.asarray(data["days"], dtype=float) >= float(post_reset_ice_reset_day)
        )
    if post_reset_log_offset_day is None:
        post_reset_log_offset_group_mask = None
    else:
        post_reset_log_offset_group_mask = jnp.asarray(
            np.asarray(data["days"], dtype=float) >= float(post_reset_log_offset_day)
        )
    if fit_qe_rbf:
        qe_rbf_nodes, qe_rbf_width_nm, _ = qe_rbf_configuration(
            qe_rbf_node_wavelengths_nm,
            qe_rbf_width_nm,
        )
        qe_rbf_nodes = jnp.asarray(qe_rbf_nodes)
        qe_rbf_basis = jnp.exp(
            -0.5
            * (
                (wavelengths[:, None] - qe_rbf_nodes[None, :])
                / qe_rbf_width_nm
            )
            ** 2
        )
        qe_rbf_basis = qe_rbf_basis / jnp.sum(
            qe_rbf_basis, axis=1, keepdims=True
        )

    inverse_wavelengths = 1.0 / wavelengths
    phase_factor = 2.0 * jnp.pi * inverse_wavelengths

    def oblique_layer_elements(
        index, thickness, polarization, angle_sine, phase_offset_nm=None
    ):
        index, thickness = jnp.broadcast_arrays(
            jnp.atleast_1d(index), jnp.atleast_1d(thickness)
        )
        index = index[..., None]
        thickness = thickness[..., None]
        angle_sine = angle_sine[:, :, None]
        cosine = jnp.sqrt(1.0 - (angle_sine / index[None, :, :]) ** 2)
        admittance = (
            index * cosine
            if polarization == "s"
            else index / cosine
        )
        phase = phase_factor * index * thickness * cosine
        if phase_offset_nm is not None:
            phase = phase + (
                phase_factor[None, None, :]
                * jnp.asarray(phase_offset_nm).reshape(1, -1, 1)
                * cosine
            )
        cosine_phase = jnp.cos(phase)
        sine_phase = -1j * jnp.sin(phase)
        return (
            cosine_phase,
            sine_phase / admittance,
            sine_phase * admittance,
            cosine_phase,
        )

    def wavelength_dependent_oblique_layer_elements(
        index, thickness, polarization, angle_sine, row_dependent_thickness=False
    ):
        index = jnp.asarray(index)
        if index.ndim == 0:
            index = jnp.broadcast_to(index, wavelengths.shape)
        thickness = jnp.asarray(thickness)
        if thickness.ndim == 0:
            thickness = thickness[None]
        cosine = jnp.sqrt(
            1.0 - (angle_sine[:, :, None] / index[None, None, :]) ** 2
        )
        admittance = (
            index[None, None, :] * cosine
            if polarization == "s"
            else index[None, None, :] / cosine
        )
        if row_dependent_thickness:
            phase = (
                phase_factor[None, None, :]
                * index[None, None, :]
                * thickness[None, :, None]
                * cosine
            )
        else:
            if thickness.ndim == 1:
                thickness = thickness[None, :]
            elif thickness.ndim != 2 or thickness.shape[0] != cosine.shape[1]:
                raise ValueError(
                    "Oblique layer thickness must be shared or match the optical rows"
                )
            phase = (
                phase_factor[None, None, None, :]
                * index[None, None, None, :]
                * thickness[None, :, :, None]
                * cosine[:, :, None, :]
            )
            admittance = admittance[:, :, None, :]
        cosine_phase = jnp.cos(phase)
        sine_phase = -1j * jnp.sin(phase)
        return (
            cosine_phase,
            sine_phase / admittance,
            sine_phase * admittance,
            cosine_phase,
        )

    def wavelength_dependent_oblique_layer_elements_pair(
        index, thickness, angle_sine, row_dependent_thickness=False
    ):
        """Build TE/TM layer elements while sharing phase calculations."""
        index = jnp.asarray(index)
        if index.ndim == 0:
            index = jnp.broadcast_to(index, wavelengths.shape)
        thickness = jnp.asarray(thickness)
        if thickness.ndim == 0:
            thickness = thickness[None]
        cosine = jnp.sqrt(
            1.0 - (angle_sine[:, :, None] / index[None, None, :]) ** 2
        )
        if row_dependent_thickness:
            phase = (
                phase_factor[None, None, :]
                * index[None, None, :]
                * thickness[None, :, None]
                * cosine
            )
        else:
            if thickness.ndim == 1:
                thickness = thickness[None, :]
            elif thickness.ndim != 2 or thickness.shape[0] != cosine.shape[1]:
                raise ValueError(
                    "Oblique layer thickness must be shared or match the optical rows"
                )
            phase = (
                phase_factor[None, None, None, :]
                * index[None, None, None, :]
                * thickness[None, :, :, None]
                * cosine[:, :, None, :]
            )
        cosine_phase = jnp.cos(phase)
        sine_phase = -1j * jnp.sin(phase)
        admittance_s = index[None, None, :] * cosine
        admittance_p = index[None, None, :] / cosine
        admittance_shape = (slice(None), slice(None), None, slice(None))
        if phase.ndim == 3:
            admittance_s_for_phase = admittance_s
            admittance_p_for_phase = admittance_p
        else:
            admittance_s_for_phase = admittance_s[admittance_shape]
            admittance_p_for_phase = admittance_p[admittance_shape]
        return (
            (
                cosine_phase,
                sine_phase / admittance_s_for_phase,
                sine_phase * admittance_s_for_phase,
                cosine_phase,
            ),
            (
                cosine_phase,
                sine_phase / admittance_p_for_phase,
                sine_phase * admittance_p_for_phase,
                cosine_phase,
            ),
        )

    def layer_elements(index, thickness, phase_offset_nm=None):
        index, thickness = jnp.broadcast_arrays(
            jnp.atleast_1d(index), jnp.atleast_1d(thickness)
        )
        phase = phase_factor * index[..., None] * thickness[..., None]
        if phase_offset_nm is not None:
            phase = phase + phase_factor[None, :] * jnp.asarray(
                phase_offset_nm
            ).reshape(-1, 1)
        cosine = jnp.cos(phase)
        # With n_complex = n + i*k, forward propagation decays as exp(i*phase).
        # The corresponding characteristic matrix therefore uses -i*sin(phase).
        sine = -1j * jnp.sin(phase)
        return (
            cosine,
            sine / index[..., None],
            sine * index[..., None],
            cosine,
        )

    def wavelength_dependent_layer_elements(index, thickness):
        thickness = jnp.atleast_1d(thickness)
        phase = phase_factor * thickness[..., None] * index[None, :]
        cosine = jnp.cos(phase)
        sine = -1j * jnp.sin(phase)
        return (
            cosine,
            sine / index[None, :],
            sine * index[None, :],
            cosine,
        )

    def substrate_index_for_parameters(hgcdte_parameters):
        if absolute_substrate_power:
            if hgcdte_parameters is None:
                raise ValueError(
                    "Absolute substrate power requires fitted HgCdTe scale parameters"
                )
            n_log_scale = hgcdte_parameters[..., 0]
            k_log_scale = hgcdte_parameters[..., 1]
            return jnp.real(wavelength_dependent_substrate_index) * jnp.exp(
                n_log_scale[..., None]
            ) + 1j * jnp.imag(wavelength_dependent_substrate_index) * jnp.exp(
                k_log_scale[..., None]
            )
        if fit_hgcdte_modulation:
            n_log_modulation = hgcdte_parameters[:hgcdte_modulation_node_count]
            k_log_modulation = hgcdte_parameters[hgcdte_modulation_node_count:]
            reference_n = jnp.real(wavelength_dependent_substrate_index)
            reference_k = jnp.imag(wavelength_dependent_substrate_index)
            return reference_n * jnp.exp(
                hgcdte_modulation_basis_value @ n_log_modulation
            ) + 1j * reference_k * jnp.exp(
                hgcdte_modulation_basis_value @ k_log_modulation
            )
        if use_hgcdte_optical_constants:
            return wavelength_dependent_substrate_index
        if not fit_hgcdte:
            return NOMINAL_HGCDTE[0] + 1j * NOMINAL_HGCDTE[1]
        return hgcdte_parameters[0] + 1j * hgcdte_parameters[1]

    def conductive_index_for_parameters(conductive_parameters):
        if not fit_conductive_layer:
            return None, None
        log_thickness, log_plasma_strength, log_damping_ratio = conductive_parameters
        thickness = jnp.exp(log_thickness)
        plasma_strength = jnp.exp(log_plasma_strength)
        damping_ratio = jnp.exp(log_damping_ratio)
        omega_ratio = CONDUCTIVE_REFERENCE_WAVELENGTH_NM / wavelengths
        epsilon = CONDUCTIVE_EPSILON_INF - plasma_strength / (
            omega_ratio**2 + 1j * damping_ratio * omega_ratio
        )
        return jnp.sqrt(epsilon), thickness

    def conductive_layer_elements(conductive_parameters):
        index, thickness = conductive_index_for_parameters(conductive_parameters)
        phase = phase_factor * index * thickness
        cosine = jnp.cos(phase)
        sine = -1j * jnp.sin(phase)
        return (
            cosine,
            sine / index,
            sine * index,
            cosine,
        )

    def apply_layer_elements(
        boundary_first, boundary_second, elements
    ):
        m00, m01, m10, m11 = elements
        return (
            m00 * boundary_first + m01 * boundary_second,
            m10 * boundary_first + m11 * boundary_second,
        )

    def detector_thickness_for_rows(thickness, n_rows):
        if thickness is None:
            raise ValueError(
                "Finite HgCdTe response requires detector thickness values"
            )
        thickness = jnp.asarray(thickness)
        if thickness.ndim == 0:
            return jnp.broadcast_to(thickness, (n_rows,))
        thickness = thickness.reshape(-1)
        if thickness.shape[0] == 1:
            return jnp.broadcast_to(thickness, (n_rows,))
        if thickness.shape[0] != n_rows:
            raise ValueError("Detector thickness rows do not match AR rows")
        return thickness

    def ar_boundary_terms(
        indices,
        thicknesses,
        substrate_index,
        conductive_parameters=None,
        detector_thickness_nm=None,
        phase_offset_nm=None,
    ):
        """Propagate the substrate boundary vector through the stack."""
        n_rows, n_layers = indices.shape
        substrate_index = jnp.asarray(substrate_index)
        if finite_hgcdte_detector:
            if substrate_index.ndim == 0:
                substrate_index = jnp.broadcast_to(substrate_index, wavelengths.shape)
            substrate_index = substrate_index.reshape(-1)
            boundary_shape = (n_rows, wavelengths.shape[0])
            backside_index = jnp.asarray(
                detector_backside_index, dtype=jnp.complex128
            )
            boundary_first = jnp.ones(boundary_shape, dtype=jnp.complex128)
            boundary_second = jnp.broadcast_to(backside_index, boundary_shape)
            detector_elements = wavelength_dependent_layer_elements(
                substrate_index,
                detector_thickness_for_rows(detector_thickness_nm, n_rows),
            )
            boundary_first, boundary_second = apply_layer_elements(
                boundary_first, boundary_second, detector_elements
            )
            detector_front_first = boundary_first
            detector_front_second = boundary_second
        else:
            if substrate_index.ndim == 0:
                boundary_shape = (n_rows, 1)
            else:
                boundary_shape = (n_rows, substrate_index.shape[-1])
            boundary_first = jnp.ones(boundary_shape, dtype=jnp.complex128)
            boundary_second = jnp.broadcast_to(substrate_index, boundary_shape)
            detector_front_first = boundary_first
            detector_front_second = boundary_second
        if fit_conductive_layer:
            boundary_first, boundary_second = apply_layer_elements(
                boundary_first,
                boundary_second,
                conductive_layer_elements(conductive_parameters),
            )
        # The matrix product is applied from the substrate side backwards.
        for layer_index in range(n_layers - 1, -1, -1):
            boundary_first, boundary_second = apply_layer_elements(
                boundary_first,
                boundary_second,
                layer_elements(
                    indices[:, layer_index],
                    thicknesses[:, layer_index],
                    phase_offset_nm if layer_index == 0 else None,
                ),
            )
        denominator = boundary_first + boundary_second
        if finite_hgcdte_detector:
            normalization = 2.0 / denominator
            clean = jnp.clip(
                jnp.real(
                    detector_front_first * jnp.conj(detector_front_second)
                )
                * jnp.abs(normalization) ** 2
                - jnp.real(backside_index) * jnp.abs(normalization) ** 2,
                1.0e-12,
                1.0,
            )
        else:
            clean = jnp.real(substrate_index) * jnp.abs(2.0 / denominator) ** 2
        return (
            clean,
            boundary_first,
            boundary_second,
            detector_front_first,
            detector_front_second,
        )

    def oblique_ar_boundary_terms(
        indices,
        thicknesses,
        substrate_index,
        polarization,
        angle_sine,
        conductive_parameters=None,
        detector_thickness_nm=None,
        phase_offset_nm=None,
    ):
        """Propagate one polarization through the stack at the cone angles."""
        n_rows, n_layers = indices.shape
        substrate_index = jnp.asarray(substrate_index)
        if substrate_index.ndim == 0:
            substrate_index = jnp.broadcast_to(substrate_index, wavelengths.shape)
        if substrate_index.ndim == 1:
            substrate_index = jnp.broadcast_to(
                substrate_index[None, :], (n_rows, wavelengths.shape[0])
            )
        elif substrate_index.ndim == 2:
            if substrate_index.shape[0] == 1:
                substrate_index = jnp.broadcast_to(
                    substrate_index, (n_rows, wavelengths.shape[0])
                )
            elif substrate_index.shape != (n_rows, wavelengths.shape[0]):
                raise ValueError("Row-wise substrate indices do not match AR rows")
        else:
            raise ValueError("Substrate indices must be scalar, spectral, or row-wise spectral")
        substrate_cosine = jnp.sqrt(
            1.0 - (angle_sine[:, :, None] / substrate_index[None, :, :]) ** 2
        )
        substrate_admittance = (
            substrate_index[None, :, :] * substrate_cosine
            if polarization == "s"
            else substrate_index[None, :, :] / substrate_cosine
        )
        incident_cosine = jnp.sqrt(1.0 - angle_sine**2)
        incident_admittance = (
            incident_cosine
            if polarization == "s"
            else 1.0 / incident_cosine
        )
        if finite_hgcdte_detector:
            backside_index = jnp.broadcast_to(
                jnp.asarray(detector_backside_index, dtype=jnp.complex128),
                wavelengths.shape,
            )
            backside_cosine = jnp.sqrt(
                1.0 - (angle_sine[:, :, None] / backside_index[None, None, :]) ** 2
            )
            backside_admittance = (
                backside_index[None, None, :] * backside_cosine
                if polarization == "s"
                else backside_index[None, None, :] / backside_cosine
            )
            terminal_admittance = backside_admittance
        else:
            terminal_admittance = substrate_admittance
        boundary_first = jnp.ones(
            (angle_sine.shape[0], n_rows, wavelengths.shape[0]),
            dtype=jnp.complex128,
        )
        boundary_second = jnp.broadcast_to(terminal_admittance, boundary_first.shape)
        if finite_hgcdte_detector:
            detector_elements = wavelength_dependent_oblique_layer_elements(
                substrate_index,
                detector_thickness_for_rows(detector_thickness_nm, n_rows),
                polarization,
                angle_sine,
                row_dependent_thickness=True,
            )
            boundary_first, boundary_second = apply_layer_elements(
                boundary_first, boundary_second, detector_elements
            )
            detector_front_first = boundary_first
            detector_front_second = boundary_second
        else:
            detector_front_first = boundary_first
            detector_front_second = boundary_second
        if fit_conductive_layer:
            boundary_first, boundary_second = apply_layer_elements(
                boundary_first,
                boundary_second,
                wavelength_dependent_oblique_layer_elements(
                    *conductive_index_for_parameters(conductive_parameters),
                    polarization,
                    angle_sine,
                    row_dependent_thickness=True,
                ),
            )
        # The matrix product is applied from the substrate side backwards.
        for layer_index in range(n_layers - 1, -1, -1):
            boundary_first, boundary_second = apply_layer_elements(
                boundary_first,
                boundary_second,
                oblique_layer_elements(
                    indices[:, layer_index],
                    thicknesses[:, layer_index],
                    polarization,
                    angle_sine,
                    phase_offset_nm if layer_index == 0 else None,
                ),
            )
        denominator = (
            incident_admittance[:, :, None] * boundary_first + boundary_second
        )
        if finite_hgcdte_detector:
            normalization = 2.0 * incident_admittance[:, :, None] / denominator
            clean = jnp.clip(
                (
                    jnp.real(
                        detector_front_first * jnp.conj(detector_front_second)
                    )
                    - jnp.real(backside_admittance)
                )
                * jnp.abs(normalization) ** 2
                / incident_admittance[:, :, None],
                1.0e-12,
                1.0,
            )
        else:
            clean = (
                jnp.real(terminal_admittance)
                / incident_admittance[:, :, None]
                * jnp.abs(
                    2.0 * incident_admittance[:, :, None] / denominator
                )
                ** 2
            )
        return (
            clean,
            boundary_first,
            boundary_second,
            incident_admittance,
            terminal_admittance,
            detector_front_first,
            detector_front_second,
        )

    def oblique_ar_boundary_terms_pair(
        indices, thicknesses, substrate_index, angle_sine, phase_offset_nm=None
    ):
        """Propagate both polarizations while sharing geometry and phase work."""
        n_rows, n_layers = indices.shape
        substrate_index = jnp.asarray(substrate_index)
        if substrate_index.ndim == 0:
            substrate_index = jnp.broadcast_to(substrate_index, wavelengths.shape)
        if substrate_index.ndim == 1:
            substrate_index = jnp.broadcast_to(
                substrate_index[None, :], (n_rows, wavelengths.shape[0])
            )
        elif substrate_index.ndim == 2:
            if substrate_index.shape[0] == 1:
                substrate_index = jnp.broadcast_to(
                    substrate_index, (n_rows, wavelengths.shape[0])
                )
            elif substrate_index.shape != (n_rows, wavelengths.shape[0]):
                raise ValueError("Row-wise substrate indices do not match AR rows")
        else:
            raise ValueError(
                "Substrate indices must be scalar, spectral, or row-wise spectral"
            )
        substrate_cosine = jnp.sqrt(
            1.0 - (angle_sine[:, :, None] / substrate_index[None, :, :]) ** 2
        )
        substrate_admittance_s = substrate_index[None, :, :] * substrate_cosine
        substrate_admittance_p = substrate_index[None, :, :] / substrate_cosine
        incident_cosine = jnp.sqrt(1.0 - angle_sine**2)
        incident_admittance_s = incident_cosine
        incident_admittance_p = 1.0 / incident_cosine
        boundary_shape = (angle_sine.shape[0], n_rows, wavelengths.shape[0])
        boundary_first_s = jnp.ones(boundary_shape, dtype=jnp.complex128)
        boundary_first_p = boundary_first_s
        boundary_second_s = jnp.broadcast_to(
            substrate_admittance_s, boundary_shape
        )
        boundary_second_p = jnp.broadcast_to(
            substrate_admittance_p, boundary_shape
        )
        for layer_index in range(n_layers - 1, -1, -1):
            index = jnp.asarray(indices[:, layer_index])[:, None]
            thickness = jnp.asarray(thicknesses[:, layer_index])[:, None]
            cosine = jnp.sqrt(
                1.0
                - (angle_sine[:, :, None] / index[None, :, :]) ** 2
            )
            phase = (
                phase_factor[None, None, :]
                * index[None, :, :]
                * thickness[None, :, :]
                * cosine
            )
            if layer_index == 0 and phase_offset_nm is not None:
                phase = phase + (
                    phase_factor[None, None, :]
                    * jnp.asarray(phase_offset_nm).reshape(1, n_rows, 1)
                    * cosine
                )
            cosine_phase = jnp.cos(phase)
            sine_phase = -1j * jnp.sin(phase)
            admittance_s = index[None, :, :] * cosine
            admittance_p = index[None, :, :] / cosine
            boundary_first_s, boundary_second_s = apply_layer_elements(
                boundary_first_s,
                boundary_second_s,
                (
                    cosine_phase,
                    sine_phase / admittance_s,
                    sine_phase * admittance_s,
                    cosine_phase,
                ),
            )
            boundary_first_p, boundary_second_p = apply_layer_elements(
                boundary_first_p,
                boundary_second_p,
                (
                    cosine_phase,
                    sine_phase / admittance_p,
                    sine_phase * admittance_p,
                    cosine_phase,
                ),
            )
        denominator_s = (
            incident_admittance_s[:, :, None] * boundary_first_s
            + boundary_second_s
        )
        denominator_p = (
            incident_admittance_p[:, :, None] * boundary_first_p
            + boundary_second_p
        )
        clean_s = (
            jnp.real(substrate_admittance_s)
            / incident_admittance_s[:, :, None]
            * jnp.abs(2.0 * incident_admittance_s[:, :, None] / denominator_s) ** 2
        )
        clean_p = (
            jnp.real(substrate_admittance_p)
            / incident_admittance_p[:, :, None]
            * jnp.abs(2.0 * incident_admittance_p[:, :, None] / denominator_p) ** 2
        )
        return (
            (
                clean_s,
                boundary_first_s,
                boundary_second_s,
                incident_admittance_s,
                substrate_admittance_s,
                boundary_first_s,
                boundary_second_s,
            ),
            (
                clean_p,
                boundary_first_p,
                boundary_second_p,
                incident_admittance_p,
                substrate_admittance_p,
                boundary_first_p,
                boundary_second_p,
            ),
        )

    def oblique_ar_boundary_terms_pair_single(
        indices, thicknesses, substrate_index, angle_sine, phase_offset_nm=None
    ):
        """Single-ray version of the shared-polarization boundary calculation."""
        n_rows, n_layers = indices.shape
        substrate_index = jnp.asarray(substrate_index)
        if substrate_index.ndim == 0:
            substrate_index = jnp.broadcast_to(substrate_index, wavelengths.shape)
        if substrate_index.ndim == 1:
            substrate_index = jnp.broadcast_to(
                substrate_index[None, :], (n_rows, wavelengths.shape[0])
            )
        elif substrate_index.ndim == 2:
            if substrate_index.shape[0] == 1:
                substrate_index = jnp.broadcast_to(
                    substrate_index, (n_rows, wavelengths.shape[0])
                )
            elif substrate_index.shape != (n_rows, wavelengths.shape[0]):
                raise ValueError("Row-wise substrate indices do not match AR rows")
        else:
            raise ValueError(
                "Substrate indices must be scalar, spectral, or row-wise spectral"
            )
        substrate_cosine = jnp.sqrt(
            1.0 - (angle_sine[:, None] / substrate_index) ** 2
        )
        substrate_admittance_s = substrate_index * substrate_cosine
        substrate_admittance_p = substrate_index / substrate_cosine
        incident_cosine = jnp.sqrt(1.0 - angle_sine**2)
        incident_admittance_s = incident_cosine
        incident_admittance_p = 1.0 / incident_cosine
        boundary_shape = (n_rows, wavelengths.shape[0])
        boundary_first_s = jnp.ones(boundary_shape, dtype=jnp.complex128)
        boundary_first_p = boundary_first_s
        boundary_second_s = substrate_admittance_s
        boundary_second_p = substrate_admittance_p
        for layer_index in range(n_layers - 1, -1, -1):
            index = jnp.asarray(indices[:, layer_index])[:, None]
            thickness = jnp.asarray(thicknesses[:, layer_index])[:, None]
            cosine = jnp.sqrt(1.0 - (angle_sine[:, None] / index) ** 2)
            phase = (
                phase_factor[None, :]
                * index
                * thickness
                * cosine
            )
            if layer_index == 0 and phase_offset_nm is not None:
                phase = phase + (
                    phase_factor[None, :]
                    * jnp.asarray(phase_offset_nm).reshape(n_rows, 1)
                    * cosine
                )
            cosine_phase = jnp.cos(phase)
            sine_phase = -1j * jnp.sin(phase)
            admittance_s = index * cosine
            admittance_p = index / cosine
            boundary_first_s, boundary_second_s = apply_layer_elements(
                boundary_first_s,
                boundary_second_s,
                (
                    cosine_phase,
                    sine_phase / admittance_s,
                    sine_phase * admittance_s,
                    cosine_phase,
                ),
            )
            boundary_first_p, boundary_second_p = apply_layer_elements(
                boundary_first_p,
                boundary_second_p,
                (
                    cosine_phase,
                    sine_phase / admittance_p,
                    sine_phase * admittance_p,
                    cosine_phase,
                ),
            )
        denominator_s = incident_admittance_s[:, None] * boundary_first_s + boundary_second_s
        denominator_p = incident_admittance_p[:, None] * boundary_first_p + boundary_second_p
        clean_s = (
            4.0
            * jnp.real(substrate_admittance_s)
            * incident_admittance_s[:, None]
            / jnp.abs(denominator_s) ** 2
        )
        clean_p = (
            4.0
            * jnp.real(substrate_admittance_p)
            * incident_admittance_p[:, None]
            / jnp.abs(denominator_p) ** 2
        )
        return (
            (
                clean_s,
                boundary_first_s,
                boundary_second_s,
                incident_admittance_s,
                substrate_admittance_s,
                boundary_first_s,
                boundary_second_s,
            ),
            (
                clean_p,
                boundary_first_p,
                boundary_second_p,
                incident_admittance_p,
                substrate_admittance_p,
                boundary_first_p,
                boundary_second_p,
            ),
        )

    def post_reset_oblique_layer_elements(index, thickness, polarization, angle_sine):
        """Build oblique ice elements when the index varies by group."""
        index = jnp.atleast_1d(jnp.asarray(index))
        thickness = jnp.atleast_1d(jnp.asarray(thickness))
        angle_sine = jnp.asarray(angle_sine)
        single_ray = angle_sine.ndim == 1
        if single_ray:
            angle_sine = angle_sine[None, :]
        if angle_sine.ndim != 2:
            raise ValueError("Incidence sine values must have angle and row axes")
        n_rows = angle_sine.shape[1]
        if thickness.ndim == 1:
            thickness = thickness[None, :]
        if thickness.ndim != 2:
            raise ValueError("Post-reset ice thickness must have row and group axes")
        if thickness.shape[0] == 1 and n_rows != 1:
            thickness = jnp.broadcast_to(thickness, (n_rows, thickness.shape[1]))
        elif thickness.shape[0] != n_rows:
            raise ValueError("Post-reset ice thickness rows do not match incidence rows")
        index = jnp.broadcast_to(index, thickness.shape)
        cosine = jnp.sqrt(
            1.0
            - (
                angle_sine[:, :, None, None]
                / index[None, :, :, None]
            ) ** 2
        )
        phase = (
            phase_factor[None, None, None, :]
            * index[None, :, :, None]
            * thickness[None, :, :, None]
            * cosine
        )
        cosine_phase = jnp.cos(phase)
        sine_phase = -1j * jnp.sin(phase)
        admittance = (
            index[None, :, :, None] * cosine
            if polarization == "s"
            else index[None, :, :, None] / cosine
        )
        elements = (
            cosine_phase,
            sine_phase / admittance,
            sine_phase * admittance,
            cosine_phase,
        )
        if single_ray:
            return tuple(element[0] for element in elements)
        return elements

    def post_reset_oblique_layer_elements_pair(index, thickness, angle_sine):
        elements_s = post_reset_oblique_layer_elements(
            index, thickness, "s", angle_sine
        )
        elements_p = post_reset_oblique_layer_elements(
            index, thickness, "p", angle_sine
        )
        return elements_s, elements_p

    def wavelength_dependent_oblique_layer_elements_pair_single(
        index, thickness, angle_sine
    ):
        """Single-ray ice elements with shared phase calculations."""
        index = jnp.asarray(index)
        if index.ndim == 0:
            index = jnp.broadcast_to(index, wavelengths.shape)
        thickness = jnp.asarray(thickness)
        if thickness.ndim == 0:
            thickness = thickness[None]
        cosine = jnp.sqrt(
            1.0 - (angle_sine[:, None] / index[None, :]) ** 2
        )
        if thickness.ndim == 1:
            thickness = thickness[None, :]
        elif thickness.ndim != 2 or thickness.shape[0] != cosine.shape[0]:
            raise ValueError(
                "Oblique layer thickness must be shared or match the optical rows"
            )
        phase = (
            phase_factor[None, None, :]
            * index[None, None, :]
            * thickness[:, :, None]
            * cosine[:, None, :]
        )
        cosine_phase = jnp.cos(phase)
        sine_phase = -1j * jnp.sin(phase)
        admittance_s = index[None, :] * cosine
        admittance_p = index[None, :] / cosine
        return (
            (
                cosine_phase,
                sine_phase / admittance_s[:, None, :],
                sine_phase * admittance_s[:, None, :],
                cosine_phase,
            ),
            (
                cosine_phase,
                sine_phase / admittance_p[:, None, :],
                sine_phase * admittance_p[:, None, :],
                cosine_phase,
            ),
        )

    def response_terms(
        ar_parameters,
        substrate_index,
        angle_sine,
        conductive_parameters=None,
        detector_thickness_nm=None,
        phase_offset_nm=None,
    ):
        ar_parameters = ar_parameters.reshape((ar_parameters.shape[0], -1, 3))
        ar_index = ar_parameters[:, :, 1] + 1j * ar_parameters[:, :, 2]
        if use_incidence_cone:
            if (
                share_polarization_work
                and not finite_hgcdte_detector
                and not fit_conductive_layer
            ):
                return (
                    oblique_ar_boundary_terms_pair_single(
                        ar_index,
                        100.0 * ar_parameters[:, :, 0],
                        substrate_index,
                        angle_sine[0],
                        phase_offset_nm,
                    )
                    if single_incidence_ray
                    else oblique_ar_boundary_terms_pair(
                        ar_index,
                        100.0 * ar_parameters[:, :, 0],
                        substrate_index,
                        angle_sine,
                        phase_offset_nm,
                    )
                )
            return tuple(
                oblique_ar_boundary_terms(
                    ar_index,
                    100.0 * ar_parameters[:, :, 0],
                    substrate_index,
                    polarization,
                    angle_sine,
                    conductive_parameters,
                    detector_thickness_nm,
                    phase_offset_nm,
                )
                for polarization in ("s", "p")
            )
        return ar_boundary_terms(
            ar_index,
            100.0 * ar_parameters[:, :, 0],
            substrate_index,
            conductive_parameters,
            detector_thickness_nm,
            phase_offset_nm,
        )

    def complex_q_factor(ice_thickness, ice_index_for_groups, parameters):
        """Return the effective complex-boundary multiplier for each group."""
        if parameters is None:
            raise ValueError("Complex-q parameters are required when q fitting is active")
        parameters = jnp.asarray(parameters)
        log_magnitude, magnitude_slope, phase_offset, optical_path_nm, absorption = (
            parameters
        )
        wavelength_coordinate = (wavelengths - 1400.0) / 1000.0
        magnitude = jnp.exp(log_magnitude + magnitude_slope * wavelength_coordinate)
        phase_offset = phase_offset + 2.0 * jnp.pi * optical_path_nm * (
            1.0 / wavelengths - 1.0 / 1400.0
        )
        q = magnitude * jnp.exp(1j * phase_offset)
        thickness = jnp.asarray(ice_thickness)
        if thickness.ndim == 1:
            thickness = thickness[None, :]
        elif thickness.ndim != 2:
            raise ValueError("Complex-q thickness must be one or two dimensional")
        phase_index = jnp.real(jnp.asarray(ice_index_for_groups))
        if phase_index.ndim == 0:
            phase = (
                4.0
                * jnp.pi
                * phase_index
                * thickness[..., None]
                / wavelengths
            )
        elif phase_index.ndim == 1 and phase_index.shape[0] == wavelengths.shape[0]:
            phase = (
                4.0
                * jnp.pi
                * thickness[..., None]
                * phase_index[None, None, :]
                / wavelengths
            )
        elif phase_index.ndim == 1 and phase_index.shape[0] == thickness.shape[-1]:
            phase = (
                4.0
                * jnp.pi
                * thickness[..., None]
                * phase_index[None, :, None]
                / wavelengths
            )
        else:
            raise ValueError("Complex-q ice-index shape does not match the response grid")
        numerator = jnp.abs(1.0 - q)[None, None, :] ** 2
        denominator = jnp.abs(
            1.0 - q[None, None, :] * jnp.exp(1j * phase)
        ) ** 2
        factor = jnp.exp(-absorption * thickness[..., None]) * numerator / denominator
        return factor[0] if jnp.asarray(ice_thickness).ndim == 1 else factor

    def response_curves(
        ice_thickness,
        response_terms_value,
        angle_sine,
        group_indices,
        model_weights_value,
        outside_output_value,
        clean_output_value,
        hgcdte_parameters=None,
        qe_rbf_parameters=None,
        post_reset_ice_index=None,
        pre_reset_ice_index=None,
        ice_modulation_parameters=None,
        complex_q_parameters=None,
    ):
        custom_segment_ice_index = (
            pre_reset_ice_index is not None or post_reset_ice_index is not None
        )
        if fit_ice_optical_modulation:
            if ice_modulation_parameters is None:
                raise ValueError(
                    "Ice modulation parameters are required for modulated prediction"
                )
            ice_n_log_modulation = ice_modulation_parameters[:ice_modulation_node_count]
            ice_k_log_modulation = ice_modulation_parameters[ice_modulation_node_count:]
            ice_index_for_groups = (
                jnp.real(ice_index)
                * jnp.exp(ice_modulation_basis_value @ ice_n_log_modulation)
                + 1j
                * jnp.imag(ice_index)
                * jnp.exp(ice_modulation_basis_value @ ice_k_log_modulation)
            )
        elif not custom_segment_ice_index:
            ice_index_for_groups = ice_index
        else:
            if post_reset_group_mask is None:
                raise ValueError("Segment ice indices require a reset day")
            pre_index = (
                ICE_INDEX_DEFAULT + 0.0j
                if pre_reset_ice_index is None
                else pre_reset_ice_index
            )
            post_index = (
                ICE_INDEX_DEFAULT + 0.0j
                if post_reset_ice_index is None
                else post_reset_ice_index
            )
            ice_index_for_groups = jnp.where(
                post_reset_group_mask[group_indices],
                post_index,
                pre_index,
            )
        substrate_index = (
            substrate_index_for_parameters(hgcdte_parameters)
            if not use_incidence_cone
            else None
        )
        if use_incidence_cone:
            clean_values = []
            coated_values = []
            single_shared_polarization = (
                share_polarization_work
                and single_incidence_ray
                and not finite_hgcdte_detector
                and not fit_conductive_layer
            )
            shared_ice_elements = None
            if (
                share_polarization_work
                and not finite_hgcdte_detector
                and not fit_conductive_layer
            ):
                if custom_segment_ice_index:
                    shared_ice_elements = (
                        post_reset_oblique_layer_elements_pair(
                            ice_index_for_groups,
                            ice_thickness,
                            angle_sine[0],
                        )
                        if single_incidence_ray
                        else post_reset_oblique_layer_elements_pair(
                            ice_index_for_groups,
                            ice_thickness,
                            angle_sine,
                        )
                    )
                else:
                    shared_ice_elements = (
                        wavelength_dependent_oblique_layer_elements_pair_single(
                            ice_index_for_groups
                            if use_asw_optical_constants
                            else jnp.broadcast_to(ice_index, wavelengths.shape),
                            ice_thickness,
                            angle_sine[0],
                        )
                        if single_incidence_ray
                        else wavelength_dependent_oblique_layer_elements_pair(
                            ice_index_for_groups
                            if use_asw_optical_constants
                            else jnp.broadcast_to(ice_index, wavelengths.shape),
                            ice_thickness,
                            angle_sine,
                            row_dependent_thickness=False,
                        )
                    )
            for polarization_index, (
                polarization,
                response_terms_for_polarization,
            ) in enumerate(zip(
                ("s", "p"), response_terms_value
            )):
                (
                    clean_angle,
                    ar_first,
                    ar_second,
                    incident_admittance,
                    terminal_admittance,
                    detector_front_first,
                    detector_front_second,
                ) = response_terms_for_polarization
                ice_elements = (
                    shared_ice_elements[polarization_index]
                    if shared_ice_elements is not None
                    else (
                        post_reset_oblique_layer_elements(
                            ice_index_for_groups,
                            ice_thickness,
                            polarization,
                            angle_sine,
                        )
                        if custom_segment_ice_index
                        else (
                            wavelength_dependent_oblique_layer_elements(
                                ice_index_for_groups,
                                ice_thickness,
                                polarization,
                                angle_sine,
                                row_dependent_thickness=False,
                            )
                            if use_asw_optical_constants
                            else wavelength_dependent_oblique_layer_elements(
                                jnp.broadcast_to(ice_index, wavelengths.shape),
                                ice_thickness,
                                polarization,
                                angle_sine,
                                row_dependent_thickness=False,
                            )
                        )
                    )
                )
                # Evaluate ice @ AR without materializing the full stack matrix.
                if single_shared_polarization:
                    ice_first = (
                        incident_admittance[:, None, None] * ice_elements[0]
                        + ice_elements[2]
                    )
                    ice_second = (
                        incident_admittance[:, None, None] * ice_elements[1]
                        + ice_elements[3]
                    )
                    denominator = (
                        ar_first[:, None, :] * ice_first
                        + ar_second[:, None, :] * ice_second
                    )
                    coated_angle = (
                        4.0
                        * jnp.real(terminal_admittance)[:, None, :]
                        * incident_admittance[:, None, None]
                        / jnp.abs(denominator) ** 2
                    )
                    clean_values.append(clean_angle)
                    coated_values.append(coated_angle)
                else:
                    ice_first = (
                        incident_admittance[:, :, None, None] * ice_elements[0]
                        + ice_elements[2]
                    )
                    ice_second = (
                        incident_admittance[:, :, None, None] * ice_elements[1]
                        + ice_elements[3]
                    )
                    denominator = (
                        ar_first[:, :, None, :] * ice_first
                        + ar_second[:, :, None, :] * ice_second
                    )
                    normalization = (
                        2.0
                        * incident_admittance[:, :, None, None]
                        / denominator
                    )
                    if finite_hgcdte_detector:
                        coated_angle = jnp.clip(
                            (
                                jnp.real(
                                    detector_front_first[:, :, None, :]
                                    * jnp.conj(detector_front_second[:, :, None, :])
                                )
                                - jnp.real(terminal_admittance)[:, :, None, :]
                            )
                            * jnp.abs(normalization) ** 2
                            / incident_admittance[:, :, None, None],
                            1.0e-12,
                            1.0,
                        )
                    else:
                        coated_angle = (
                            jnp.real(terminal_admittance)[:, :, None, :]
                            / incident_admittance[:, :, None, None]
                            * jnp.abs(normalization) ** 2
                        )
                    clean_values.append(
                        jnp.sum(
                            clean_angle * incidence_weights[:, None, None], axis=0
                        )
                    )
                    coated_values.append(
                        jnp.sum(
                            coated_angle * incidence_weights[:, None, None, None],
                            axis=0,
                        )
                    )
            clean = 0.5 * (clean_values[0] + clean_values[1])
            coated = 0.5 * (coated_values[0] + coated_values[1])
        else:
            (
                clean,
                ar_first,
                ar_second,
                detector_front_first,
                detector_front_second,
            ) = response_terms_value
            ice_elements = (
                wavelength_dependent_layer_elements(
                    ice_index_for_groups, ice_thickness
                )
                if use_asw_optical_constants
                else layer_elements(ice_index_for_groups, ice_thickness)
            )
            # Evaluate the denominator of ice @ AR directly instead of materializing
            # the full (tile, group, wavelength, 2, 2) product tensor.
            ice_first = ice_elements[0] + ice_elements[2]
            ice_second = ice_elements[1] + ice_elements[3]
            if ice_first.ndim == 2:
                ice_first = ice_first[None, :, :]
                ice_second = ice_second[None, :, :]
            denominator = (
                ar_first[:, None, :] * ice_first
                + ar_second[:, None, :] * ice_second
            )
            normalization = 2.0 / denominator
            if finite_hgcdte_detector:
                coated = jnp.clip(
                    (
                        jnp.real(
                            detector_front_first[:, None, :]
                            * jnp.conj(detector_front_second[:, None, :])
                        )
                        - detector_backside_index
                    )
                    * jnp.abs(normalization) ** 2,
                    1.0e-12,
                    1.0,
                )
            else:
                substrate_factor = jnp.real(substrate_index)
                if substrate_factor.ndim == 2:
                    substrate_factor = substrate_factor[:, None, :]
            coated = substrate_factor * jnp.abs(normalization) ** 2
        if fit_complex_q:
            coated = coated * complex_q_factor(
                ice_thickness, ice_index_for_groups, complex_q_parameters
            )
        if fit_qe_rbf:
            qe_multiplier = jnp.exp(qe_rbf_basis @ qe_rbf_parameters)
            if model_weights_value.ndim == 2:
                model_weights_value = model_weights_value * qe_multiplier[None, :]
            else:
                model_weights_value = (
                    model_weights_value * qe_multiplier[None, None, :]
                )
            clean_output_value = jnp.sum(model_weights_value, axis=-1) + outside_output_value
        if model_weights_value.ndim == 2:
            group_led_index = led_index[group_indices]
            model_weights = model_weights_value[group_led_index, :]
            outside_output = outside_output_value[group_led_index]
            clean_output = clean_output_value[group_led_index]
        else:
            model_weights = model_weights_value[:, group_indices, :]
            outside_output = outside_output_value[:, group_indices]
            clean_output = clean_output_value[:, group_indices]
        if absolute_substrate_power:
            clean_integrated = jnp.sum(clean[:, None, :] * model_weights, axis=-1)
            integrated = jnp.sum(coated * model_weights, axis=-1)
            return integrated / clean_integrated
        ratios = coated / clean[:, None, :]
        integrated = jnp.sum(ratios * model_weights, axis=-1)
        integrated = integrated + outside_output
        return integrated / clean_output

    def prediction_for_rows(
        ice_thickness,
        ar_parameters,
        acm_step,
        model_weights_value,
        outside_output_value,
        clean_output_value,
        incidence_sine_rows,
        hgcdte_parameters=None,
        conductive_parameters=None,
        qe_rbf_parameters=None,
        early_time_scale=None,
        late_acm_step=None,
        detector_thickness_nm=None,
        incidence_angle_deg=None,
        additional_acm_step=None,
        post_reset_ice_index=None,
        pre_reset_ice_index=None,
        ice_modulation_parameters=None,
        sca_ar_opd_nm=None,
        complex_q_parameters=None,
    ):
        if sca_ar_opd_nm is not None:
            sca_ar_opd_nm = jnp.asarray(sca_ar_opd_nm).reshape(-1)
            if sca_ar_opd_nm.shape[0] == 1 and ar_parameters.shape[0] != 1:
                sca_ar_opd_nm = jnp.broadcast_to(
                    sca_ar_opd_nm, (ar_parameters.shape[0],)
                )
            elif sca_ar_opd_nm.shape[0] != ar_parameters.shape[0]:
                raise ValueError("SCA AR OPD values must match prediction rows")
        substrate_index = substrate_index_for_parameters(hgcdte_parameters)
        if fit_incidence_angle:
            if incidence_angle_deg is None:
                raise ValueError("A fitted incidence angle is required for prediction")
            dynamic_incidence_sine = fitted_incidence_sines(incidence_angle_deg)
            incidence_sine_rows = jnp.broadcast_to(
                dynamic_incidence_sine,
                (ar_parameters.shape[0], dynamic_incidence_sine.shape[1]),
            )
        angle_sine = jnp.asarray(incidence_sine_rows).T
        share_response_terms = (
            share_ar_across_rows
            and not absolute_substrate_power
            and ar_parameters.shape[0] > 1
            and not incidence_geometry_varies_by_row
            and sca_ar_opd_nm is None
        )
        if finite_hgcdte_detector and detector_thickness_nm is not None:
            detector_thickness_shape = jnp.asarray(detector_thickness_nm).shape
            share_response_terms = share_response_terms and (
                len(detector_thickness_shape) == 0
                or int(np.prod(detector_thickness_shape)) == 1
            )
        if share_response_terms:
            terms_one_row = response_terms(
                ar_parameters[:1],
                substrate_index,
                angle_sine[:, :1],
                conductive_parameters,
                detector_thickness_nm,
                sca_ar_opd_nm,
            )
            if use_incidence_cone:
                terms = tuple(
                    tuple(
                        jnp.broadcast_to(
                            term,
                            (term.shape[0], ar_parameters.shape[0]) + term.shape[2:],
                        )
                        for term_index, term in enumerate(polarization_terms)
                    )
                    for polarization_terms in terms_one_row
                )
            else:
                terms = tuple(
                    jnp.broadcast_to(
                        term, (ar_parameters.shape[0],) + term.shape[1:]
                    )
                    for term in terms_one_row
                )
        else:
            terms = response_terms(
                ar_parameters,
                substrate_index,
                angle_sine,
                conductive_parameters,
                detector_thickness_nm,
                sca_ar_opd_nm,
            )
        response = response_curves(
            ice_thickness,
            terms,
            angle_sine,
            all_group_indices,
            model_weights_value,
            outside_output_value,
            clean_output_value,
            hgcdte_parameters,
            qe_rbf_parameters,
            post_reset_ice_index,
            pre_reset_ice_index,
            ice_modulation_parameters,
            complex_q_parameters,
        )
        current = response
        # The current response already contains every LED's reference group
        # at its exact reference thickness, so reuse those columns instead of
        # launching a second six-group TMM evaluation.
        baseline = current[:, first_by_led][:, led_index]
        acm_step = jnp.asarray(acm_step)
        observation_acm_step = (
            acm_step[led_index] if acm_step.ndim == 1 else acm_step
        )
        fitted = jnp.log(current / baseline) + observation_acm_step * step_design[None, :]
        if absolute_substrate_power:
            fitted = fitted + baseline_offsets[led_index][None, :]
        if fit_late_acm_step:
            if late_acm_step is None:
                raise ValueError("Late ACM step is required when that step is active")
            fitted = fitted + jnp.asarray(late_acm_step) * late_step_design[None, :]
        if fit_additional_acm_step:
            if additional_acm_step is None:
                raise ValueError(
                    "Additional ACM step is required when that step is active"
                )
            fitted = fitted + jnp.asarray(additional_acm_step) * additional_step_design[
                None, :
            ]
        if early_time_cutoff_day is not None:
            if early_time_scale is None:
                raise ValueError("Early-time scale is required when the cutoff is active")
            early_time_scale = jnp.asarray(early_time_scale)
            safe_scale_index = jnp.maximum(early_time_scale_index, 0)
            scale_by_observation = jnp.where(
                early_time_scale_index >= 0,
                early_time_scale[safe_scale_index],
                1.0,
            )
            # The fitted factor multiplies observed counts. Express that
            # data-side correction as the equivalent raw-space model offset.
            fitted = fitted - jnp.where(
                (early_time_scale_index >= 0)[None, :],
                jnp.log(scale_by_observation)[None, :],
                0.0,
            )
        return fitted

    def prediction(
        ice_thickness,
        ar_parameters,
        acm_step,
        hgcdte_parameters=None,
        conductive_parameters=None,
        qe_rbf_parameters=None,
        early_time_scale=None,
        late_acm_step=None,
        detector_thickness_nm=None,
        incidence_angle_deg=None,
        additional_acm_step=None,
        post_reset_ice_index=None,
        pre_reset_ice_index=None,
        ice_modulation_parameters=None,
        sca_ar_opd_nm=None,
        complex_q_parameters=None,
    ):
        row_sca_ar_opd_nm = None
        if sca_ar_opd_nm is not None:
            sca_ar_opd_nm = jnp.asarray(sca_ar_opd_nm).reshape(-1)
            if sca_ar_opd_nm.shape[0] != n_scas:
                raise ValueError("SCA AR OPD values must have one value per SCA")
            row_sca_ar_opd_nm = sca_ar_opd_nm[sca_index]
        return prediction_for_rows(
            ice_thickness,
            ar_parameters,
            acm_step,
            selected_model_weights,
            selected_outside_output,
            selected_clean_output,
            selected_incidence_sine,
            hgcdte_parameters,
            conductive_parameters,
            qe_rbf_parameters,
            early_time_scale,
            late_acm_step,
            detector_thickness_nm,
            incidence_angle_deg,
            additional_acm_step,
            post_reset_ice_index,
            pre_reset_ice_index,
            ice_modulation_parameters,
            row_sca_ar_opd_nm,
            complex_q_parameters,
        )

    def boundary_reflection(
        ar_parameters,
        hgcdte_parameters=None,
        conductive_parameters=None,
        detector_thickness_nm=None,
        incidence_angle_deg=None,
        sca_ar_opd_nm=None,
        ice_index_override=None,
    ):
        """Return physical boundary reflections from the existing TMM terms."""
        ar_parameters = jnp.asarray(ar_parameters)
        if ar_parameters.ndim == 1:
            ar_parameters = ar_parameters[None, :]
        if ar_parameters.ndim != 2 or ar_parameters.shape[0] != len(row_indices):
            raise ValueError("AR parameters must have one row per selected optical row")
        row_sca_ar_opd_nm = None
        if sca_ar_opd_nm is not None:
            sca_ar_opd_nm = jnp.asarray(sca_ar_opd_nm).reshape(-1)
            if sca_ar_opd_nm.shape[0] != n_scas:
                raise ValueError("SCA AR OPD values must have one value per SCA")
            row_sca_ar_opd_nm = sca_ar_opd_nm[sca_index]
        substrate_index = substrate_index_for_parameters(hgcdte_parameters)
        if fit_incidence_angle:
            if incidence_angle_deg is None:
                raise ValueError("A fitted incidence angle is required for reflection")
            dynamic_incidence_sine = fitted_incidence_sines(incidence_angle_deg)
            incidence_sine_rows = jnp.broadcast_to(
                dynamic_incidence_sine,
                (ar_parameters.shape[0], dynamic_incidence_sine.shape[1]),
            )
        else:
            incidence_sine_rows = selected_incidence_sine
        angle_sine = jnp.asarray(incidence_sine_rows).T
        terms = response_terms(
            ar_parameters,
            substrate_index,
            angle_sine,
            conductive_parameters,
            detector_thickness_nm,
            row_sca_ar_opd_nm,
        )
        if ice_index_override is None:
            ice_index_value = ice_index
        else:
            ice_index_value = jnp.asarray(ice_index_override)
        if jnp.asarray(ice_index_value).ndim == 0:
            ice_index_value = jnp.broadcast_to(ice_index_value, wavelengths.shape)
        if jnp.asarray(ice_index_value).shape != wavelengths.shape:
            raise ValueError("Ice index must be scalar or match the response wavelengths")

        results = {}
        if use_incidence_cone:
            for polarization, polarization_terms in zip(("s", "p"), terms):
                _, ar_first, ar_second, incident_admittance, _, _, _ = (
                    polarization_terms
                )
                cosine = jnp.sqrt(
                    1.0
                    - (
                        angle_sine[:, :, None]
                        / ice_index_value[None, None, :]
                    )
                    ** 2
                )
                ice_admittance = (
                    ice_index_value[None, None, :] * cosine
                    if polarization == "s"
                    else ice_index_value[None, None, :] / cosine
                )
                load_admittance = ar_second / ar_first
                reflection = (
                    incident_admittance[:, :, None] * ar_first - ar_second
                ) / (
                    incident_admittance[:, :, None] * ar_first + ar_second
                )
                bottom_reflection = (ice_admittance - load_admittance) / (
                    ice_admittance + load_admittance
                )
                top_reflection = (
                    ice_admittance - incident_admittance[:, :, None]
                ) / (ice_admittance + incident_admittance[:, :, None])
                results[polarization] = {
                    "reflection": reflection,
                    "load_admittance": load_admittance,
                    "bottom_reflection": bottom_reflection,
                    "top_reflection": top_reflection,
                    "round_trip_reflection": top_reflection * bottom_reflection,
                    "incident_admittance": incident_admittance[:, :, None],
                    "ice_admittance": ice_admittance,
                }
        else:
            ar_first = terms[1]
            ar_second = terms[2]
            load_admittance = ar_second / ar_first
            reflection = (ar_first - ar_second) / (ar_first + ar_second)
            ice_admittance = ice_index_value[None, :]
            bottom_reflection = (ice_admittance - load_admittance) / (
                ice_admittance + load_admittance
            )
            top_reflection = (ice_admittance - 1.0) / (ice_admittance + 1.0)
            values = {
                "reflection": reflection[None, ...],
                "load_admittance": load_admittance[None, ...],
                "bottom_reflection": bottom_reflection[None, ...],
                "top_reflection": top_reflection[None, ...],
                "round_trip_reflection": (top_reflection * bottom_reflection)[None, ...],
                "incident_admittance": jnp.ones_like(ar_first)[None, ...],
                "ice_admittance": ice_admittance[None, ...],
            }
            results["s"] = values
            results["p"] = values
        return {
            "wavelengths": wavelengths,
            "angle_sine": angle_sine,
            "s": results["s"],
            "p": results["p"],
        }

    prediction.boundary_reflection = boundary_reflection

    if all(key in data for key in ("counts", "sums", "sum_squared")):
        selected_counts = np.asarray(data["counts"])[row_indices]
        selected_sums = np.asarray(data["sums"])[row_indices]
        selected_sum_squared = np.asarray(data["sum_squared"])[row_indices]
        n_rows = int(selected_counts.shape[0])
        row_chunk_size = min(
            int(data.get("prediction_row_chunk_size", PREDICTION_ROW_CHUNK_SIZE)),
            n_rows,
        )
        n_chunks = (n_rows + row_chunk_size - 1) // row_chunk_size
        padded_rows = n_chunks * row_chunk_size
        row_padding = padded_rows - n_rows

        def pad_rows(value, mode="edge"):
            return jnp.pad(value, ((0, row_padding),) + ((0, 0),) * (value.ndim - 1), mode=mode)

        shared_response_inputs = selected_model_weights.ndim == 2
        padded_counts = pad_rows(jnp.asarray(selected_counts), mode="constant")
        padded_sums = pad_rows(jnp.asarray(selected_sums), mode="constant")
        padded_sum_squared = pad_rows(
            jnp.asarray(selected_sum_squared), mode="constant"
        )
        if not shared_response_inputs:
            padded_model_weights = pad_rows(selected_model_weights)
            padded_outside_output = pad_rows(selected_outside_output)
            padded_clean_output = pad_rows(selected_clean_output)
        padded_incidence_sine = pad_rows(selected_incidence_sine)

        def residual_sum_squares(
            ice_thickness,
            ar_parameters,
            acm_step,
            hgcdte_parameters=None,
            conductive_parameters=None,
            qe_rbf_parameters=None,
            early_time_scale=None,
            late_acm_step=None,
            detector_thickness_nm=None,
            incidence_angle_deg=None,
            additional_acm_step=None,
            post_reset_ice_index=None,
            pre_reset_ice_index=None,
            ice_modulation_parameters=None,
            post_reset_tile_log_offsets=None,
            sca_ar_opd_nm=None,
            complex_q_parameters=None,
            return_prediction=False,
        ):
            ar_rows = ar_parameters.reshape((n_rows, -1))
            padded_ar_rows = pad_rows(ar_rows)
            sca_ar_opd_rows = None
            if sca_ar_opd_nm is not None:
                sca_ar_opd_nm = jnp.asarray(sca_ar_opd_nm).reshape(-1)
                if sca_ar_opd_nm.shape[0] != n_scas:
                    raise ValueError("SCA AR OPD values must have one value per SCA")
                sca_ar_opd_rows = sca_ar_opd_nm[
                    jnp.asarray(all_sca_index[row_indices], dtype=jnp.int32)
                ]
            padded_sca_ar_opd = (
                pad_rows(sca_ar_opd_rows, mode="constant")
                if sca_ar_opd_rows is not None
                else None
            )
            detector_thickness_is_rowwise = (
                finite_hgcdte_detector
                and detector_thickness_nm is not None
                and jnp.asarray(detector_thickness_nm).ndim == 1
            )
            padded_detector_thickness = (
                pad_rows(jnp.asarray(detector_thickness_nm))
                if detector_thickness_is_rowwise
                else None
            )
            hgcdte_parameters_is_rowwise = (
                absolute_substrate_power
                and hgcdte_parameters is not None
                and jnp.asarray(hgcdte_parameters).ndim == 2
            )
            padded_hgcdte_parameters = (
                pad_rows(jnp.asarray(hgcdte_parameters))
                if hgcdte_parameters_is_rowwise
                else None
            )
            scan_values = [
                padded_ar_rows.reshape((n_chunks, row_chunk_size, -1)),
                padded_counts.reshape((n_chunks, row_chunk_size, -1)),
                padded_sums.reshape((n_chunks, row_chunk_size, -1)),
                padded_sum_squared.reshape((n_chunks, row_chunk_size, -1)),
            ]
            if not shared_response_inputs:
                scan_values.extend(
                    (
                        padded_model_weights.reshape(
                            (n_chunks, row_chunk_size) + padded_model_weights.shape[1:]
                        ),
                        padded_outside_output.reshape(
                            (n_chunks, row_chunk_size) + padded_outside_output.shape[1:]
                        ),
                        padded_clean_output.reshape(
                            (n_chunks, row_chunk_size) + padded_clean_output.shape[1:]
                        ),
                    )
                )
            scan_values.append(
                padded_incidence_sine.reshape(
                    (n_chunks, row_chunk_size) + padded_incidence_sine.shape[1:]
                )
            )
            if post_reset_tile_log_offsets is not None:
                if post_reset_log_offset_group_mask is None:
                    raise ValueError(
                        "Post-reset tile log offsets require a post-reset day"
                    )
                post_reset_tile_log_offsets = jnp.asarray(
                    post_reset_tile_log_offsets
                ).reshape(-1)
                if post_reset_tile_log_offsets.shape != (n_rows,):
                    raise ValueError(
                        "Post-reset tile log offsets must have one value per row"
                    )
                padded_post_reset_tile_log_offsets = pad_rows(
                    post_reset_tile_log_offsets, mode="constant"
                )
                scan_values.append(
                    padded_post_reset_tile_log_offsets.reshape(
                        (n_chunks, row_chunk_size)
                    )
                )
            if ice_thickness.ndim == 2:
                padded_ice_thickness = pad_rows(ice_thickness)
                scan_values.insert(
                    0,
                    padded_ice_thickness.reshape(
                        (n_chunks, row_chunk_size, -1)
                    ),
                )
            if detector_thickness_is_rowwise:
                scan_values.insert(
                    1 if ice_thickness.ndim == 2 else 0,
                    padded_detector_thickness.reshape(
                        (n_chunks, row_chunk_size)
                    ),
                )
            if padded_sca_ar_opd is not None:
                scan_values.append(
                    padded_sca_ar_opd.reshape((n_chunks, row_chunk_size))
                )
            if hgcdte_parameters_is_rowwise:
                scan_values.append(
                    padded_hgcdte_parameters.reshape(
                        (n_chunks, row_chunk_size)
                        + padded_hgcdte_parameters.shape[1:]
                    )
                )
            scan_values = tuple(scan_values)

            def scan_chunk(total_sse, values):
                value_index = 0
                if ice_thickness.ndim == 2:
                    ice_chunk = values[value_index]
                    value_index += 1
                else:
                    ice_chunk = ice_thickness
                if detector_thickness_is_rowwise:
                    detector_thickness_chunk = values[value_index]
                    value_index += 1
                else:
                    detector_thickness_chunk = detector_thickness_nm
                value_end = len(values)
                if hgcdte_parameters_is_rowwise:
                    hgcdte_parameters_chunk = values[value_end - 1]
                    value_end -= 1
                else:
                    hgcdte_parameters_chunk = hgcdte_parameters
                if padded_sca_ar_opd is not None:
                    sca_ar_opd_chunk = values[value_end - 1]
                    value_end -= 1
                else:
                    sca_ar_opd_chunk = None
                if post_reset_tile_log_offsets is not None:
                    post_reset_tile_log_offsets_chunk = values[value_end - 1]
                    value_end -= 1
                else:
                    post_reset_tile_log_offsets_chunk = None
                model_values = values[value_index:value_end]
                ar_chunk, counts_chunk, sums_chunk, sum_squared_chunk = model_values[:4]
                if shared_response_inputs:
                    model_weights_chunk = selected_model_weights
                    outside_output_chunk = selected_outside_output
                    clean_output_chunk = selected_clean_output
                    incidence_sine_chunk = model_values[4]
                else:
                    (
                        model_weights_chunk,
                        outside_output_chunk,
                        clean_output_chunk,
                        incidence_sine_chunk,
                    ) = model_values[4:]
                fitted = prediction_for_rows(
                    ice_chunk,
                    ar_chunk,
                    acm_step,
                    model_weights_chunk,
                    outside_output_chunk,
                    clean_output_chunk,
                    incidence_sine_chunk,
                    hgcdte_parameters_chunk,
                    conductive_parameters,
                    qe_rbf_parameters,
                    early_time_scale,
                    late_acm_step,
                    detector_thickness_chunk,
                    incidence_angle_deg,
                    additional_acm_step,
                    post_reset_ice_index,
                    pre_reset_ice_index,
                    ice_modulation_parameters,
                    sca_ar_opd_chunk,
                    complex_q_parameters,
                )
                safe_counts = jnp.maximum(counts_chunk, 1.0)
                observed_chunk = sums_chunk / safe_counts
                if post_reset_tile_log_offsets_chunk is not None:
                    observed_chunk = observed_chunk + (
                        post_reset_tile_log_offsets_chunk[:, None]
                        * post_reset_log_offset_group_mask[None, :]
                    )
                within_group_sse = (
                    sum_squared_chunk - sums_chunk * sums_chunk / safe_counts
                )
                residual_sse = within_group_sse + counts_chunk * (
                    fitted - observed_chunk
                ) ** 2
                return (
                    total_sse + jnp.sum(residual_sse),
                    fitted if return_prediction else None,
                )

            initial_sse = jnp.asarray(0.0, dtype=padded_counts.dtype)
            if n_chunks == 1:
                # Avoid a loop and its reverse-mode scan machinery when the
                # selected tiles already fit in a single prediction chunk.
                total_sse, fitted_padded = scan_chunk(
                    initial_sse, tuple(values[0] for values in scan_values)
                )
            else:
                if return_prediction:
                    total_sse, fitted_chunks = jax.lax.scan(
                        scan_chunk, initial_sse, scan_values
                    )
                    fitted_padded = fitted_chunks.reshape((padded_rows, -1))
                else:
                    total_sse, _ = jax.lax.scan(
                        scan_chunk, initial_sse, scan_values
                    )
            if return_prediction:
                return total_sse, fitted_padded[:n_rows]
            return total_sse

        prediction.residual_sum_squares = residual_sum_squares

    return prediction


def build_model(
    data,
    context,
    log_noise_sigma,
    likelihood_weight,
    ar_mean_prior_sigma=AR_MEAN_PRIOR_SIGMA,
    ar_prior_reference=None,
    ar_layers=1,
    np_modulation=False,
    constant_rate_modulation=False,
    constant_rate_smoothness_weight=0.0,
    fully_np_trajectory=False,
    standardize_ar_mean=False,
    rotate_ar_coordinates=False,
    whiten_np_trajectory=False,
    np_rate_likelihood_rotation=None,
    pivot_initial_thickness=False,
    map_safe_initial_thickness=False,
    np_entropy_weight=NP_ENTROPY_WEIGHT,
    np_smoothness_weight=NP_SMOOTHNESS_WEIGHT,
    np_max_node_multiplier=NP_MAX_NODE_MULTIPLIER,
    ar_sca_sigma_prior=AR_SCA_SIGMA_PRIOR,
    ar_tile_sigma_prior=AR_TILE_SIGMA_PRIOR,
        fit_sca_ar_opd=False,
        sca_ar_opd_prior_sigma_nm=SCA_AR_OPD_PRIOR_SIGMA_NM,
        shared_ar_optical_properties=False,
        shared_ar_parameters_across_tiles=False,
        independent_tile_ar_parameters=False,
        fit_ar_absorption=False,
    fit_hgcdte=False,
    fit_hgcdte_modulation=False,
    hgcdte_modulation_prior_sigma_log=HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
    fit_ice_optical_modulation=False,
    ice_modulation_prior_sigma_log=ICE_MODULATION_PRIOR_SIGMA_LOG,
    ice_n_modulation_prior_sigma_log=None,
    fit_conductive_layer=False,
    finite_hgcdte_detector=False,
    detector_thickness_nm=DETECTOR_THICKNESS_DEFAULT_NM,
    fit_detector_thickness=False,
    detector_thickness_prior_sigma_log=DETECTOR_THICKNESS_PRIOR_SIGMA_LOG,
    detector_thickness_neighbor_sigma_log=DETECTOR_THICKNESS_NEIGHBOR_SIGMA_LOG,
    detector_backside_index=1.0,
    fit_qe_rbf=False,
    qe_rbf_node_wavelengths_nm=None,
    qe_rbf_width_nm=QE_RBF_WIDTH_NM,
    qe_rbf_prior_sigma_log=QE_RBF_PRIOR_SIGMA_LOG,
    use_asw_optical_constants=False,
    use_hgcdte_optical_constants=False,
    incidence_cone_half_angle_deg=0.0,
    incidence_pupil_half_angle_deg=0.0,
    incidence_chief_ray_tilts_deg=None,
    fit_incidence_angle=False,
    incidence_angle_prior_sigma_deg=INCIDENCE_ANGLE_PRIOR_SIGMA_DEG,
    incidence_angle_prior_uniform=False,
    led_dependent_acm_step=False,
    fit_acm_step=True,
    acm_step_prior_sigma=0.05,
    fit_late_acm_step=False,
    late_acm_step_prior_sigma=0.02,
    fit_additional_acm_step=False,
    additional_acm_step_prior_sigma=0.02,
    trajectory_rate_prior_mean_nm_day=None,
    fixed_trajectory_rate_nm_day=None,
    constant_rate_trajectory=False,
    fixed_ar_indices=None,
    ordered_ar_indices=False,
    trajectory_rate_prior_sigma_log=0.75,
    tile_rate_variation=False,
    tile_rate_trajectory_variation=False,
    independent_tile_trajectories=False,
    tile_rate_trajectory_smoothness_weight=0.001,
    tile_rate_prior_sigma_log=TRAJECTORY_TILE_RATE_PRIOR_SIGMA_LOG,
    tile_rate_neighbor_sigma_log=TRAJECTORY_TILE_RATE_NEIGHBOR_SIGMA_LOG,
    tile_rate_neighbor_smoothness=True,
    fixed_initial_thickness_nm=None,
    initial_thickness_prior_mean_nm=INITIAL_THICKNESS_PRIOR_MEAN_NM,
    initial_thickness_prior_sigma_nm=INITIAL_THICKNESS_PRIOR_SIGMA_NM,
    early_time_cutoff_day=None,
    early_time_scale_prior_sigma=EARLY_TIME_SCALE_PRIOR_SIGMA,
    absolute_substrate_power=None,
    absolute_hgcdte_scale_prior_sigma_log=ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG,
    fixed_absolute_hgcdte_scales=False,
    shared_absolute_hgcdte_scales=False,
    initial_thickness_prior_uniform=False,
    tile_initial_thickness_lognormal=False,
    tile_initial_thickness_log_median_nm=10.0,
    tile_initial_thickness_log_sigma=1.5,
    ice_reset_day=None,
    fit_post_reset_tile_log_offset=False,
    post_reset_tile_log_offset_prior_sigma=POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA,
    fit_post_reset_ice_index=False,
    post_reset_ice_index_prior_mean=POST_RESET_ICE_INDEX_PRIOR_MEAN,
    post_reset_ice_index_prior_sigma=POST_RESET_ICE_INDEX_PRIOR_SIGMA,
    post_reset_ice_index_lower=POST_RESET_ICE_INDEX_LOWER,
    post_reset_ice_index_upper=POST_RESET_ICE_INDEX_UPPER,
    fit_pre_post_ice_index=False,
    ice_index_prior_mean=ICE_INDEX_PRIOR_MEAN,
    ice_index_prior_sigma=ICE_INDEX_PRIOR_SIGMA,
    ice_index_lower=ICE_INDEX_LOWER,
    ice_index_upper=ICE_INDEX_UPPER,
    ice_extinction_prior_sigma=ICE_EXTINCTION_PRIOR_SIGMA,
    ice_extinction_upper=ICE_EXTINCTION_UPPER,
    fit_complex_q=False,
    include_map_prediction=False,
):
    jax, jnp, numpyro, dist, _, _, _ = require_numpyro()
    if absolute_substrate_power is None:
        absolute_substrate_power = bool(context.get("absolute_substrate_power", False))
    absolute_substrate_power = bool(absolute_substrate_power)
    qe_rbf_nodes, qe_rbf_width_nm, qe_rbf_prior_sigma_log = qe_rbf_configuration(
        qe_rbf_node_wavelengths_nm,
        qe_rbf_width_nm,
        qe_rbf_prior_sigma_log,
    )
    ar_reference, ar_lower, ar_upper = ar_prior_arrays(ar_layers)
    ar_parameter_count = len(ar_reference)
    fixed_ar_indices = validate_fixed_ar_indices(ar_layers, fixed_ar_indices)
    ordered_ar_indices = bool(ordered_ar_indices)
    if ordered_ar_indices and fixed_ar_indices is not None:
        raise ValueError("Fixed and ordered AR indices are mutually exclusive")
    active_parameter_indices = ar_active_parameter_indices(
        ar_layers, fit_ar_absorption, fixed_ar_indices
    )
    fixed_ar_parameter_indices = (
        3 * np.arange(ar_layers, dtype=int) + 1
        if fixed_ar_indices is not None
        else np.asarray([], dtype=int)
    )
    active_parameter_count = len(active_parameter_indices)
    active_lower = jnp.asarray(ar_lower[active_parameter_indices])
    active_upper = jnp.asarray(ar_upper[active_parameter_indices])
    if sum(
        bool(mode)
        for mode in (
            np_modulation,
            constant_rate_modulation,
            fully_np_trajectory,
            constant_rate_trajectory,
        )
    ) > 1:
        raise ValueError(
            "Trajectory model modes are mutually exclusive"
        )
    if (np_modulation or constant_rate_modulation) and np_max_node_multiplier <= 1.0:
        raise ValueError("np_max_node_multiplier must be greater than one")
    if pivot_initial_thickness and not fully_np_trajectory:
        raise ValueError("Pivoted initial thickness currently requires a fully non-parametric trajectory")
    if rotate_ar_coordinates and not standardize_ar_mean:
        raise ValueError("AR coordinate rotation requires standardized AR mean coordinates")
    if ordered_ar_indices and rotate_ar_coordinates:
        raise ValueError("Ordered AR indices do not support coordinate rotation")
    if np_rate_likelihood_rotation is not None:
        if not (whiten_np_trajectory and fully_np_trajectory):
            raise ValueError(
                "Trajectory likelihood rotation requires whitened fully non-parametric rates"
            )
        np_rate_likelihood_rotation = np.asarray(
            np_rate_likelihood_rotation, dtype=float
        )
        expected_rotation_shape = (len(data["node_days"]), len(data["node_days"]))
        if (
            np_rate_likelihood_rotation.shape != expected_rotation_shape
            or not np.all(np.isfinite(np_rate_likelihood_rotation))
            or not np.allclose(
                np_rate_likelihood_rotation.T @ np_rate_likelihood_rotation,
                np.eye(expected_rotation_shape[0]),
                atol=1.0e-10,
            )
        ):
            raise ValueError("Trajectory likelihood rotation must be finite and orthogonal")
    if (
        not np.isfinite(constant_rate_smoothness_weight)
        or constant_rate_smoothness_weight < 0.0
    ):
        raise ValueError("Constant-rate smoothness weight must be finite and non-negative")
    if (
        not np.isfinite(tile_rate_trajectory_smoothness_weight)
        or tile_rate_trajectory_smoothness_weight < 0.0
    ):
        raise ValueError(
            "Tile-rate trajectory smoothness weight must be finite and non-negative"
        )
    if (
        not np.isfinite(incidence_angle_prior_sigma_deg)
        or incidence_angle_prior_sigma_deg <= 0.0
    ):
        raise ValueError("Incidence-angle prior sigma must be positive and finite")
    if fit_incidence_angle and incidence_cone_half_angle_deg != 0.0:
        raise ValueError(
            "Fitted incidence angle requires the pupil/chief-ray geometry, not a solid-angle cone"
        )
    if fit_incidence_angle and incidence_chief_ray_tilts_deg is not None and np.any(
        np.asarray(incidence_chief_ray_tilts_deg) != 0.0
    ):
        raise ValueError(
            "Fitted incidence angle cannot be combined with fixed chief-ray tilts"
        )
    if fit_post_reset_ice_index and fit_pre_post_ice_index:
        raise ValueError(
            "The scalar post-reset and complex pre/post ice-index fits are mutually exclusive"
        )
    if fit_post_reset_ice_index:
        if ice_reset_day is None:
            raise ValueError(
                "A fitted post-reset ice index requires ice_reset_day"
            )
        if use_asw_optical_constants:
            raise ValueError(
                "A scalar post-reset ice index cannot replace ASW optical constants"
            )
        if (
            not np.isfinite(post_reset_ice_index_prior_mean)
            or not np.isfinite(post_reset_ice_index_prior_sigma)
            or post_reset_ice_index_prior_sigma <= 0.0
            or not np.isfinite(post_reset_ice_index_lower)
            or not np.isfinite(post_reset_ice_index_upper)
            or post_reset_ice_index_lower >= post_reset_ice_index_upper
            or not (
                post_reset_ice_index_lower
                < post_reset_ice_index_prior_mean
                < post_reset_ice_index_upper
            )
        ):
            raise ValueError(
                "Post-reset ice-index prior must have a finite interior mean and bounds"
            )
    if fit_pre_post_ice_index:
        if ice_reset_day is None:
            raise ValueError(
                "A fitted pre/post-reset ice index requires ice_reset_day"
            )
        if use_asw_optical_constants:
            raise ValueError(
                "Scalar segment ice indices cannot replace ASW optical constants"
            )
        if (
            not np.isfinite(ice_index_prior_mean)
            or not np.isfinite(ice_index_prior_sigma)
            or ice_index_prior_sigma <= 0.0
            or not np.isfinite(ice_index_lower)
            or not np.isfinite(ice_index_upper)
            or ice_index_lower >= ice_index_upper
            or not (ice_index_lower < ice_index_prior_mean < ice_index_upper)
            or not np.isfinite(ice_extinction_prior_sigma)
            or ice_extinction_prior_sigma <= 0.0
            or not np.isfinite(ice_extinction_upper)
            or ice_extinction_upper <= 0.0
        ):
            raise ValueError(
                "Complex ice-index priors must have finite positive widths and valid bounds"
            )
    if fit_ice_optical_modulation:
        if not use_asw_optical_constants:
            raise ValueError(
                "Ice optical modulation requires ASW optical constants"
            )
        if fit_post_reset_ice_index or fit_pre_post_ice_index:
            raise ValueError(
                "Ice optical modulation is mutually exclusive with scalar ice-index fits"
            )
        if (
            not np.isfinite(ice_modulation_prior_sigma_log)
            or ice_modulation_prior_sigma_log <= 0.0
        ):
            raise ValueError(
                "Ice modulation prior sigma must be positive and finite"
            )
        if ice_n_modulation_prior_sigma_log is None:
            ice_n_modulation_prior_sigma_log = ice_modulation_prior_sigma_log
        if (
            not np.isfinite(ice_n_modulation_prior_sigma_log)
            or ice_n_modulation_prior_sigma_log <= 0.0
        ):
            raise ValueError(
                "Ice n modulation prior sigma must be positive and finite"
            )
    if fit_post_reset_tile_log_offset:
        if ice_reset_day is None:
            raise ValueError(
                "A post-reset tile log offset requires ice_reset_day"
            )
        if (
            not np.isfinite(post_reset_tile_log_offset_prior_sigma)
            or post_reset_tile_log_offset_prior_sigma <= 0.0
        ):
            raise ValueError(
                "Post-reset tile log-offset prior sigma must be positive and finite"
            )
    if ar_mean_prior_sigma <= 0.0:
        raise ValueError("AR mean prior sigma must be positive")
    if ar_sca_sigma_prior <= 0.0 or ar_tile_sigma_prior <= 0.0:
        raise ValueError("AR hierarchy sigma priors must be positive")
    if (
        not np.isfinite(sca_ar_opd_prior_sigma_nm)
        or sca_ar_opd_prior_sigma_nm <= 0.0
    ):
        raise ValueError("SCA AR OPD prior sigma must be positive and finite")
    if (
        not np.isfinite(hgcdte_modulation_prior_sigma_log)
        or hgcdte_modulation_prior_sigma_log <= 0.0
    ):
        raise ValueError("HgCdTe modulation prior sigma must be positive and finite")
    if (
        not np.isfinite(absolute_hgcdte_scale_prior_sigma_log)
        or absolute_hgcdte_scale_prior_sigma_log <= 0.0
    ):
        raise ValueError(
            "Absolute HgCdTe scale prior sigma must be positive and finite"
        )
    if absolute_substrate_power:
        if not bool(context.get("absolute_substrate_power", False)):
            raise ValueError(
                "Absolute substrate power requires an absolute-power response context"
            )
        if (
            fit_hgcdte
            or fit_hgcdte_modulation
            or not use_hgcdte_optical_constants
        ):
            raise ValueError(
                "Absolute substrate power owns the wavelength-dependent HgCdTe scales"
            )
    if fixed_absolute_hgcdte_scales and not absolute_substrate_power:
        raise ValueError(
            "Fixed absolute HgCdTe scales require absolute substrate power"
        )
    if shared_absolute_hgcdte_scales and not absolute_substrate_power:
        raise ValueError(
            "Shared absolute HgCdTe scales require absolute substrate power"
        )
    if shared_absolute_hgcdte_scales and fixed_absolute_hgcdte_scales:
        raise ValueError(
            "Shared absolute HgCdTe scales cannot be combined with fixed reference scales"
        )
    if initial_thickness_prior_uniform and pivot_initial_thickness:
        raise ValueError(
            "A uniform initial-thickness prior cannot use the pivot parameterization"
        )
    if tile_initial_thickness_lognormal and pivot_initial_thickness:
        raise ValueError(
            "Per-tile lognormal initial thickness cannot use the pivot parameterization"
        )
    if tile_initial_thickness_lognormal and initial_thickness_prior_uniform:
        raise ValueError(
            "Per-tile lognormal initial thickness cannot use the uniform prior"
        )
    if ice_reset_day is not None and pivot_initial_thickness:
        raise ValueError(
            "An ice reset cannot use the pivot initial-thickness parameterization"
        )
    if (
        not np.isfinite(tile_initial_thickness_log_median_nm)
        or tile_initial_thickness_log_median_nm <= 0.0
    ):
        raise ValueError(
            "Per-tile lognormal initial-thickness median must be positive and finite"
        )
    if (
        not np.isfinite(tile_initial_thickness_log_sigma)
        or tile_initial_thickness_log_sigma <= 0.0
    ):
        raise ValueError(
            "Per-tile lognormal initial-thickness sigma must be positive and finite"
        )
    if not np.isfinite(detector_thickness_nm) or detector_thickness_nm <= 0.0:
        raise ValueError("Detector thickness must be positive and finite")
    if (
        not np.isfinite(detector_thickness_prior_sigma_log)
        or detector_thickness_prior_sigma_log <= 0.0
    ):
        raise ValueError(
            "Detector thickness prior sigma must be positive and finite"
        )
    if (
        not np.isfinite(detector_thickness_neighbor_sigma_log)
        or detector_thickness_neighbor_sigma_log <= 0.0
    ):
        raise ValueError(
            "Detector thickness neighbor sigma must be positive and finite"
        )
    if (
        not np.isfinite(detector_backside_index)
        or detector_backside_index <= 0.0
    ):
        raise ValueError("Detector backside index must be positive and finite")
    finite_detector_active = bool(finite_hgcdte_detector or fit_detector_thickness)
    if finite_detector_active and fit_hgcdte_modulation:
        raise ValueError(
            "Finite HgCdTe detectors do not support free wavelength modulation"
        )
    if not np.isfinite(acm_step_prior_sigma) or acm_step_prior_sigma <= 0.0:
        raise ValueError("ACM step prior sigma must be positive and finite")
    if (
        not np.isfinite(late_acm_step_prior_sigma)
        or late_acm_step_prior_sigma <= 0.0
    ):
        raise ValueError("Late ACM step prior sigma must be positive and finite")
    if (
        not np.isfinite(additional_acm_step_prior_sigma)
        or additional_acm_step_prior_sigma <= 0.0
    ):
        raise ValueError(
            "Additional ACM step prior sigma must be positive and finite"
        )
    if trajectory_rate_prior_mean_nm_day is not None and (
        not np.isfinite(trajectory_rate_prior_mean_nm_day)
        or trajectory_rate_prior_mean_nm_day <= 0.0
    ):
        raise ValueError("Trajectory rate prior mean must be positive and finite")
    if fixed_trajectory_rate_nm_day is not None and (
        not np.isfinite(fixed_trajectory_rate_nm_day)
        or fixed_trajectory_rate_nm_day <= 0.0
    ):
        raise ValueError("Fixed trajectory rate must be positive and finite")
    if fixed_trajectory_rate_nm_day is not None and (
        np_modulation
        or constant_rate_modulation
        or fully_np_trajectory
        or constant_rate_trajectory
        or tile_rate_variation
        or tile_rate_trajectory_variation
        or independent_tile_trajectories
    ):
        raise ValueError(
            "A fixed trajectory rate cannot be combined with rate modulation or variation"
        )
    if constant_rate_trajectory and (
        tile_rate_variation
        or tile_rate_trajectory_variation
        or independent_tile_trajectories
    ):
        raise ValueError(
            "A constant trajectory rate cannot be combined with tile-rate variation"
        )
    if trajectory_rate_prior_sigma_log <= 0.0 or not np.isfinite(
        trajectory_rate_prior_sigma_log
    ):
        raise ValueError("Trajectory rate prior sigma must be positive and finite")
    if (
        not np.isfinite(initial_thickness_prior_mean_nm)
        or initial_thickness_prior_mean_nm < INITIAL_THICKNESS_PRIOR_LOW_NM
        or initial_thickness_prior_mean_nm > INITIAL_THICKNESS_PRIOR_HIGH_NM
    ):
        raise ValueError("Initial thickness prior mean must be between 0 and 50 nm")
    if (
        not np.isfinite(initial_thickness_prior_sigma_nm)
        or initial_thickness_prior_sigma_nm <= 0.0
    ):
        raise ValueError("Initial thickness prior sigma must be positive and finite")
    if tile_rate_prior_sigma_log <= 0.0 or not np.isfinite(
        tile_rate_prior_sigma_log
    ):
        raise ValueError("Tile trajectory rate prior sigma must be positive and finite")
    if independent_tile_trajectories and not tile_rate_trajectory_variation:
        raise ValueError(
            "Independent tile trajectories require tile-rate-trajectory-variation"
        )
    if independent_tile_trajectories and not fully_np_trajectory:
        raise ValueError(
            "Independent tile trajectories require the fully non-parametric trajectory mode"
        )
    if independent_tile_trajectories and (
        whiten_np_trajectory
        or np_rate_likelihood_rotation is not None
        or pivot_initial_thickness
    ):
        raise ValueError(
            "Independent tile trajectories do not use global trajectory whitening, "
            "rotation, or pivoted initial thickness"
        )
    if independent_tile_ar_parameters and (
        shared_ar_optical_properties or shared_ar_parameters_across_tiles
    ):
        raise ValueError(
            "Independent tile AR parameters cannot use shared AR parameter modes"
        )
    if independent_tile_ar_parameters and (
        standardize_ar_mean or rotate_ar_coordinates
    ):
        raise ValueError(
            "Independent tile AR parameters use direct raw coordinates, not global AR latents"
        )
    if tile_rate_neighbor_sigma_log <= 0.0 or not np.isfinite(
        tile_rate_neighbor_sigma_log
    ):
        raise ValueError(
            "Neighboring tile trajectory rate sigma must be positive and finite"
        )
    if fixed_initial_thickness_nm is not None and (
        not np.isfinite(fixed_initial_thickness_nm)
        or fixed_initial_thickness_nm < 0.0
        or fixed_initial_thickness_nm > 50.0
    ):
        raise ValueError("Fixed initial thickness must be between 0 and 50 nm")
    if early_time_cutoff_day is not None and (
        not np.isfinite(early_time_cutoff_day) or early_time_cutoff_day <= 0.0
    ):
        raise ValueError("Early-time scale cutoff must be positive and finite")
    if not np.isfinite(early_time_scale_prior_sigma) or early_time_scale_prior_sigma <= 0.0:
        raise ValueError("Early-time scale prior sigma must be positive and finite")
    counts = jnp.asarray(data["counts"])
    sums = jnp.asarray(data["sums"])
    sum_squared = jnp.asarray(data["sum_squared"])
    days = jnp.asarray(data["days"])
    node_days = jnp.asarray(data["node_days"])
    if ice_reset_day is not None:
        ice_reset_day = float(ice_reset_day)
        if not np.isfinite(ice_reset_day):
            raise ValueError("Ice reset day must be finite")
        if not (float(np.min(data["days"])) < ice_reset_day < float(np.max(data["days"]))):
            raise ValueError("Ice reset day must lie inside the observed day range")
        if not np.any(np.asarray(data["days"]) < ice_reset_day) or not np.any(
            np.asarray(data["days"]) >= ice_reset_day
        ):
            raise ValueError("Ice reset day must split the observed data into two segments")
    integration_times = np.unique(
        np.concatenate(
            (
                [np.min(data["days"])],
                np.asarray(data["node_days"], dtype=float),
                [np.max(data["days"])],
            )
        )
    )
    n_scas = int(data["n_scas"])
    n_tiles = int(data["n_tiles"])
    row_sca_index = jnp.asarray(data["sca_index"], dtype=jnp.int32)
    tile_rate_trajectory_variation_active = bool(
        tile_rate_trajectory_variation and n_scas * n_tiles > 1
    )
    independent_tile_trajectory_active = bool(
        independent_tile_trajectories and tile_rate_trajectory_variation_active
    )
    tile_rate_variation_active = bool(
        (tile_rate_variation or tile_rate_trajectory_variation)
        and (n_scas * n_tiles > 1)
    )
    initial_thickness_tile_active = bool(tile_initial_thickness_lognormal)
    rowwise_tile_prediction_active = bool(
        tile_rate_variation_active
        or initial_thickness_tile_active
        or fit_post_reset_tile_log_offset
    )
    neighbor_edges = tile_neighbor_edges(data)
    single_tile = is_single_tile_data(data)
    if n_tiles < 1:
        raise ValueError("At least one tile is required for the likelihood")
    # Keep the sampler curvature comparable when a fit includes more tiles.
    effective_likelihood_weight = likelihood_weight / n_tiles
    null_sse = jnp.sum(sum_squared)
    likelihood_scale = (
        0.5 * effective_likelihood_weight * null_sse / (log_noise_sigma**2)
    )
    if ar_prior_reference is None:
        ar_prior_reference = ar_reference
    ar_prior_reference = np.asarray(ar_prior_reference, dtype=float)
    if ar_prior_reference.shape != ar_reference.shape or np.any(
        ar_prior_reference <= ar_lower
    ) or np.any(ar_prior_reference >= ar_upper):
        raise ValueError("AR prior reference must match the parameter shape and bounds")
    reference_raw = jnp.asarray(
        ar_reference_raw(
            ar_prior_reference,
            ar_lower,
            ar_upper,
            active_parameter_indices,
            ordered_ar_indices=ordered_ar_indices,
        )
    )
    active_index_positions = {
        int(parameter_index): position
        for position, parameter_index in enumerate(active_parameter_indices)
    }
    ar_rotation_pairs = tuple(
        (
            active_index_positions[3 * layer],
            active_index_positions[3 * layer + 1],
        )
        for layer in range(ar_parameter_count // 3)
        if 3 * layer in active_index_positions
        and 3 * layer + 1 in active_index_positions
    )
    prediction_data = dict(data)
    aggregate_shared_ar_likelihood = bool(
        shared_ar_parameters_across_tiles
        and not tile_rate_variation_active
        and not initial_thickness_tile_active
        and not fit_post_reset_tile_log_offset
        and not absolute_substrate_power
        and not finite_detector_active
        and float(incidence_pupil_half_angle_deg) == 0.0
        and (
            incidence_chief_ray_tilts_deg is None
            or not np.any(np.asarray(incidence_chief_ray_tilts_deg) != 0.0)
        )
    )
    if aggregate_shared_ar_likelihood:
        prediction_data = aggregate_tile_sufficient_statistics(data)
    prediction = build_prediction_function(
        prediction_data,
        context,
        fit_hgcdte=fit_hgcdte,
        fit_hgcdte_modulation=fit_hgcdte_modulation,
        fit_ice_optical_modulation=fit_ice_optical_modulation,
        fit_conductive_layer=fit_conductive_layer,
        finite_hgcdte_detector=finite_detector_active,
        detector_backside_index=detector_backside_index,
        fit_qe_rbf=fit_qe_rbf,
        qe_rbf_node_wavelengths_nm=qe_rbf_nodes,
        qe_rbf_width_nm=qe_rbf_width_nm,
        use_asw_optical_constants=use_asw_optical_constants,
        use_hgcdte_optical_constants=use_hgcdte_optical_constants,
        incidence_cone_half_angle_deg=incidence_cone_half_angle_deg,
        incidence_pupil_half_angle_deg=incidence_pupil_half_angle_deg,
        incidence_chief_ray_tilts_deg=incidence_chief_ray_tilts_deg,
        fit_incidence_angle=fit_incidence_angle,
        early_time_cutoff_day=early_time_cutoff_day,
        share_ar_across_rows=shared_ar_parameters_across_tiles,
        fit_late_acm_step=fit_late_acm_step,
        fit_additional_acm_step=fit_additional_acm_step,
        absolute_substrate_power=absolute_substrate_power,
        post_reset_ice_reset_day=(
            ice_reset_day
            if (fit_post_reset_ice_index or fit_pre_post_ice_index)
            else None
        ),
        post_reset_log_offset_day=(
            ice_reset_day if fit_post_reset_tile_log_offset else None
        ),
        fit_complex_q=fit_complex_q,
    )
    residual_sum_squares = prediction.residual_sum_squares
    early_time_configuration = early_time_scale_configuration(
        data, early_time_cutoff_day
    )
    thickness_indices = np.flatnonzero(active_parameter_indices % 3 == 0)
    hierarchy_parameter_count = (
        ar_layers if shared_ar_optical_properties else active_parameter_count
    )
    trajectory_prior_rate = (
        trajectory_rate_prior_mean_nm_day
        if trajectory_rate_prior_mean_nm_day is not None
        else trajectory_rate_per_scale(data["days"]) * TRAJECTORY_SCALE_INIT
    )
    trajectory_prior_scale = trajectory_prior_rate / trajectory_rate_per_scale(
        data["days"]
    )
    pivot_node_index, _, pivot_initial_center = trajectory_pivot_configuration(
        data, trajectory_prior_rate, initial_thickness_prior_mean_nm
    )
    rate_whitener = None
    rate_likelihood_rotation_holder = {"value": None}
    map_safe_initial_thickness_holder = {"value": bool(map_safe_initial_thickness)}
    if whiten_np_trajectory and fully_np_trajectory:
        rate_whitener = jnp.asarray(
            trajectory_rate_whitener(
                data["node_days"],
                trajectory_rate_prior_sigma_log,
                float(np.asarray(likelihood_scale)),
                np_smoothness_weight,
            )
        )
        if np_rate_likelihood_rotation is not None:
            rate_likelihood_rotation_holder["value"] = jnp.asarray(
                np_rate_likelihood_rotation
            )

    def physical_ar_parameters(ar_raw):
        parameters = jnp.zeros(
            ar_raw.shape[:-1] + (ar_parameter_count,), dtype=ar_raw.dtype
        )
        if fixed_ar_indices is not None:
            parameters = parameters.at[..., fixed_ar_parameter_indices].set(
                jnp.asarray(fixed_ar_indices)
            )
        active_parameters = active_lower + (
            active_upper - active_lower
        ) * jax.nn.sigmoid(ar_raw)
        parameters = parameters.at[..., active_parameter_indices].set(active_parameters)
        if ordered_ar_indices:
            parameters = ordered_ar_indices_from_raw(
                ar_raw,
                active_parameter_indices,
                ar_lower,
                ar_upper,
                ar_parameter_count,
                jax,
                jnp,
            )
        return parameters

    def rotate_ar_latents(values):
        rotated = jnp.asarray(values)
        inverse_sqrt_two = 1.0 / jnp.sqrt(2.0)
        for thickness_position, refractive_position in ar_rotation_pairs:
            thickness = values[..., thickness_position]
            refractive_index = values[..., refractive_position]
            rotated = rotated.at[..., thickness_position].set(
                (thickness + refractive_index) * inverse_sqrt_two
            )
            rotated = rotated.at[..., refractive_position].set(
                (thickness - refractive_index) * inverse_sqrt_two
            )
        return rotated

    def log_rate_smoothness(log_rates):
        scaled_node_days = (node_days - node_days[0]) / jnp.median(
            jnp.diff(node_days)
        )
        node_slopes = jnp.diff(log_rates) / jnp.diff(scaled_node_days)
        node_curvature = 2.0 * jnp.diff(node_slopes) / (
            scaled_node_days[2:] - scaled_node_days[:-2]
        )
        return jnp.mean(node_curvature * node_curvature)

    def model():
        if fit_complex_q:
            complex_q_parameters = numpyro.sample(
                "complex_q_parameters",
                dist.Uniform(
                    jnp.asarray(COMPLEX_Q_LOWER),
                    jnp.asarray(COMPLEX_Q_UPPER),
                ).to_event(1),
            )
        else:
            complex_q_parameters = None
        if independent_tile_ar_parameters:
            ar_mu_raw = jnp.asarray(reference_raw)
            ar_sca_raw = None
        elif standardize_ar_mean:
            ar_mu_z = numpyro.sample(
                "ar_mu_z",
                dist.Normal(0.0, 1.0)
                .expand((active_parameter_count,))
                .to_event(1),
            )
            ar_mu_standardized = (
                rotate_ar_latents(ar_mu_z) if rotate_ar_coordinates else ar_mu_z
            )
            ar_mu_raw = numpyro.deterministic(
                "ar_mu_raw",
                reference_raw + ar_mean_prior_sigma * ar_mu_standardized,
            )
        else:
            ar_mu_raw = numpyro.sample(
                "ar_mu_raw",
                dist.Normal(
                    reference_raw,
                    ar_mean_prior_sigma * jnp.ones(active_parameter_count),
                ).to_event(1),
            )
        if independent_tile_ar_parameters:
            ar_tile_raw = numpyro.sample(
                "ar_tile_raw",
                dist.Normal(
                    reference_raw,
                    ar_mean_prior_sigma * jnp.ones(active_parameter_count),
                )
                .expand((n_scas, n_tiles, active_parameter_count))
                .to_event(3),
            )
            ar_parameters_all = physical_ar_parameters(ar_tile_raw)
            ar_parameters = (
                ar_parameters_all[0] if n_scas == 1 else ar_parameters_all
            )
            ar_parameters_for_prediction = ar_parameters.reshape(
                (-1, ar_parameter_count)
            )
        elif shared_ar_parameters_across_tiles:
            ar_raw_all = jnp.broadcast_to(
                ar_mu_raw[None, None, :],
                (n_scas, n_tiles, active_parameter_count),
            )
            ar_parameters_all = physical_ar_parameters(ar_raw_all)
            ar_sca_raw = jnp.broadcast_to(
                ar_mu_raw[None, :], (n_scas, active_parameter_count)
            )
            ar_parameters = (
                ar_parameters_all[0] if n_scas == 1 else ar_parameters_all
            )
            if (
                tile_rate_variation_active
                or finite_detector_active
                or absolute_substrate_power
                or initial_thickness_tile_active
                or fit_post_reset_tile_log_offset
            ):
                ar_parameters_for_prediction = ar_parameters.reshape(
                    (-1, ar_parameter_count)
                )
            else:
                ar_parameters_for_prediction = jnp.broadcast_to(
                    physical_ar_parameters(ar_mu_raw)[None, :],
                    (n_scas, ar_parameter_count),
                )
        elif shared_ar_optical_properties:
            if n_scas == 1:
                if single_tile:
                    ar_raw = ar_mu_raw[None, :]
                else:
                    ar_sigma = numpyro.sample(
                        "ar_sigma_raw",
                        dist.HalfNormal(
                            ar_tile_sigma_prior * jnp.ones(hierarchy_parameter_count)
                        ).to_event(1),
                    )
                    ar_z = numpyro.sample(
                        "ar_z",
                        dist.Normal(0.0, 1.0)
                        .expand((n_tiles, hierarchy_parameter_count))
                        .to_event(2),
                    )
                    thickness_raw = (
                        ar_mu_raw[thickness_indices][None, :]
                        + ar_sigma[None, :] * ar_z
                    )
                    ar_raw = jnp.broadcast_to(
                        ar_mu_raw, (n_tiles, active_parameter_count)
                    )
                    ar_raw = ar_raw.at[:, thickness_indices].set(thickness_raw)
                ar_parameters = physical_ar_parameters(ar_raw)
                ar_parameters_for_prediction = ar_parameters
            else:
                ar_sca_sigma = numpyro.sample(
                    "ar_sca_sigma",
                    dist.HalfNormal(
                        ar_sca_sigma_prior * jnp.ones(hierarchy_parameter_count)
                    ).to_event(1),
                )
                ar_sca_z = numpyro.sample(
                    "ar_sca_z",
                    dist.Normal(0.0, 1.0)
                    .expand((n_scas, hierarchy_parameter_count))
                    .to_event(2),
                )
                sca_thickness_raw = (
                    ar_mu_raw[thickness_indices][None, :]
                    + ar_sca_sigma[None, :] * ar_sca_z
                )
                ar_sca_raw = jnp.broadcast_to(
                    ar_mu_raw[None, :], (n_scas, active_parameter_count)
                )
                ar_sca_raw = ar_sca_raw.at[:, thickness_indices].set(
                    sca_thickness_raw
                )
                ar_tile_sigma = numpyro.sample(
                    "ar_tile_sigma",
                    dist.HalfNormal(
                        ar_tile_sigma_prior * jnp.ones(hierarchy_parameter_count)
                    ).to_event(1),
                )
                ar_tile_z = numpyro.sample(
                    "ar_tile_z",
                    dist.Normal(0.0, 1.0)
                    .expand((n_scas, n_tiles, hierarchy_parameter_count))
                    .to_event(3),
                )
                tile_thickness_raw = (
                    sca_thickness_raw[:, None, :]
                    + ar_tile_sigma[None, None, :] * ar_tile_z
                )
                ar_raw = jnp.broadcast_to(
                    ar_sca_raw[:, None, :],
                    (n_scas, n_tiles, active_parameter_count),
                )
                ar_raw = ar_raw.at[:, :, thickness_indices].set(
                    tile_thickness_raw
                )
                ar_parameters = physical_ar_parameters(ar_raw)
                ar_parameters_for_prediction = ar_parameters.reshape(
                    (-1, ar_parameter_count)
                )
        elif n_scas == 1:
            if single_tile:
                ar_raw = ar_mu_raw[None, :]
            else:
                ar_sigma = numpyro.sample(
                    "ar_sigma_raw",
                    dist.HalfNormal(
                        ar_tile_sigma_prior * jnp.ones(active_parameter_count)
                    ).to_event(1),
                )
                ar_z = numpyro.sample(
                    "ar_z",
                    dist.Normal(0.0, 1.0)
                    .expand((n_tiles, active_parameter_count))
                    .to_event(2),
                )
                ar_z_standardized = (
                    rotate_ar_latents(ar_z) if rotate_ar_coordinates else ar_z
                )
                ar_raw = ar_mu_raw[None, :] + ar_sigma[None, :] * ar_z_standardized
            ar_parameters = physical_ar_parameters(ar_raw)
            ar_parameters_for_prediction = ar_parameters
        else:
            ar_sca_sigma = numpyro.sample(
                "ar_sca_sigma",
                dist.HalfNormal(
                    ar_sca_sigma_prior * jnp.ones(active_parameter_count)
                ).to_event(1),
            )
            ar_sca_z = numpyro.sample(
                "ar_sca_z",
                dist.Normal(0.0, 1.0)
                .expand((n_scas, active_parameter_count))
                .to_event(2),
            )
            ar_sca_z_standardized = (
                rotate_ar_latents(ar_sca_z) if rotate_ar_coordinates else ar_sca_z
            )
            ar_sca_raw = (
                ar_mu_raw[None, :] + ar_sca_sigma[None, :] * ar_sca_z_standardized
            )
            ar_tile_sigma = numpyro.sample(
                "ar_tile_sigma",
                dist.HalfNormal(
                    ar_tile_sigma_prior * jnp.ones(active_parameter_count)
                ).to_event(1),
            )
            ar_tile_z = numpyro.sample(
                "ar_tile_z",
                dist.Normal(0.0, 1.0)
                .expand((n_scas, n_tiles, active_parameter_count))
                .to_event(3),
            )
            ar_tile_z_standardized = (
                rotate_ar_latents(ar_tile_z) if rotate_ar_coordinates else ar_tile_z
            )
            ar_raw = (
                ar_sca_raw[:, None, :]
                + ar_tile_sigma[None, None, :] * ar_tile_z_standardized
            )
            ar_parameters = physical_ar_parameters(ar_raw)
            ar_parameters_for_prediction = ar_parameters.reshape(
                (-1, ar_parameter_count)
            )
        if fit_sca_ar_opd:
            sca_ar_opd_nm = numpyro.sample(
                "sca_ar_opd_nm",
                dist.Normal(0.0, sca_ar_opd_prior_sigma_nm)
                .expand((n_scas,))
                .to_event(1),
            )
        else:
            sca_ar_opd_nm = None
        if tile_initial_thickness_lognormal:
            if fixed_initial_thickness_nm is None:
                initial_thickness_log_nm = numpyro.sample(
                    "initial_thickness_log_nm",
                    dist.Normal(
                        jnp.log(tile_initial_thickness_log_median_nm),
                        tile_initial_thickness_log_sigma,
                    )
                    .expand((n_scas, n_tiles))
                    .to_event(2),
                )
                d0 = numpyro.deterministic(
                    "initial_thickness_nm",
                    jnp.exp(initial_thickness_log_nm),
                )
            else:
                d0 = numpyro.deterministic(
                    "initial_thickness_nm",
                    jnp.full(
                        (n_scas, n_tiles),
                        jnp.asarray(fixed_initial_thickness_nm),
                    ),
                )
        elif fixed_initial_thickness_nm is None:
            if pivot_initial_thickness:
                initial_thickness_pivot_z = numpyro.sample(
                    "initial_thickness_pivot_z", dist.Normal(0.0, 1.0)
                )
                d0 = None
            else:
                d0 = numpyro.sample(
                    "initial_thickness_nm",
                    dist.Uniform(
                        INITIAL_THICKNESS_PRIOR_LOW_NM,
                        INITIAL_THICKNESS_PRIOR_HIGH_NM,
                    )
                    if initial_thickness_prior_uniform
                    else dist.TruncatedNormal(
                        initial_thickness_prior_mean_nm,
                        initial_thickness_prior_sigma_nm,
                        low=INITIAL_THICKNESS_PRIOR_LOW_NM,
                        high=INITIAL_THICKNESS_PRIOR_HIGH_NM,
                    ),
                )
        else:
            d0 = numpyro.deterministic(
                "initial_thickness_nm",
                jnp.asarray(fixed_initial_thickness_nm),
            )
        reset_d0 = None
        if ice_reset_day is not None:
            if tile_initial_thickness_lognormal:
                if fixed_initial_thickness_nm is None:
                    reset_initial_thickness_log_nm = numpyro.sample(
                        "reset_initial_thickness_log_nm",
                        dist.Normal(
                            jnp.log(tile_initial_thickness_log_median_nm),
                            tile_initial_thickness_log_sigma,
                        )
                        .expand((n_scas, n_tiles))
                        .to_event(2),
                    )
                    reset_d0 = numpyro.deterministic(
                        "reset_initial_thickness_nm",
                        jnp.exp(reset_initial_thickness_log_nm),
                    )
                else:
                    reset_d0 = numpyro.deterministic(
                        "reset_initial_thickness_nm",
                        jnp.full(
                            (n_scas, n_tiles),
                            jnp.asarray(fixed_initial_thickness_nm),
                        ),
                    )
            elif fixed_initial_thickness_nm is None:
                reset_d0 = numpyro.sample(
                    "reset_initial_thickness_nm",
                    dist.Uniform(
                        INITIAL_THICKNESS_PRIOR_LOW_NM,
                        INITIAL_THICKNESS_PRIOR_HIGH_NM,
                    )
                    if initial_thickness_prior_uniform
                    else dist.TruncatedNormal(
                        initial_thickness_prior_mean_nm,
                        initial_thickness_prior_sigma_nm,
                        low=INITIAL_THICKNESS_PRIOR_LOW_NM,
                        high=INITIAL_THICKNESS_PRIOR_HIGH_NM,
                    ),
                )
            else:
                reset_d0 = numpyro.deterministic(
                    "reset_initial_thickness_nm",
                    jnp.asarray(fixed_initial_thickness_nm),
                )
        pre_reset_ice_index = None
        if fit_pre_post_ice_index:
            pre_reset_ice_n = numpyro.sample(
                "pre_reset_ice_n",
                dist.TruncatedNormal(
                    ice_index_prior_mean,
                    ice_index_prior_sigma,
                    low=ice_index_lower,
                    high=ice_index_upper,
                ),
            )
            pre_reset_ice_k = numpyro.sample(
                "pre_reset_ice_k",
                dist.TruncatedNormal(
                    0.0,
                    ice_extinction_prior_sigma,
                    low=0.0,
                    high=ice_extinction_upper,
                ),
            )
            post_reset_ice_n = numpyro.sample(
                "post_reset_ice_n",
                dist.TruncatedNormal(
                    ice_index_prior_mean,
                    ice_index_prior_sigma,
                    low=ice_index_lower,
                    high=ice_index_upper,
                ),
            )
            post_reset_ice_k = numpyro.sample(
                "post_reset_ice_k",
                dist.TruncatedNormal(
                    0.0,
                    ice_extinction_prior_sigma,
                    low=0.0,
                    high=ice_extinction_upper,
                ),
            )
            pre_reset_ice_index = pre_reset_ice_n + 1j * pre_reset_ice_k
            post_reset_ice_index = post_reset_ice_n + 1j * post_reset_ice_k
        elif fit_post_reset_ice_index:
            post_reset_ice_index = numpyro.sample(
                "post_reset_ice_index",
                dist.TruncatedNormal(
                    post_reset_ice_index_prior_mean,
                    post_reset_ice_index_prior_sigma,
                    low=post_reset_ice_index_lower,
                    high=post_reset_ice_index_upper,
                ),
            )
        else:
            post_reset_ice_index = None
            pre_reset_ice_index = None
        if early_time_cutoff_day is not None:
            early_time_node_count = len(early_time_configuration["repeat"])
            if early_time_node_count > 1:
                early_time_log_scale = numpyro.sample(
                    "early_time_log_scale",
                    dist.Normal(
                        EARLY_TIME_LOG_SCALE_PRIOR_MEAN,
                        early_time_scale_prior_sigma,
                    )
                    .expand((early_time_node_count - 1,))
                    .to_event(1),
                )
            else:
                early_time_log_scale = jnp.zeros(0)
            early_time_scale = numpyro.deterministic(
                "early_time_scale",
                jnp.exp(
                    jnp.concatenate((jnp.zeros(1), early_time_log_scale))
                ),
            )
        else:
            early_time_scale = None
        if fixed_trajectory_rate_nm_day is not None:
            log_scale = jnp.log(jnp.asarray(fixed_trajectory_rate_nm_day))
            np_log_rate_nodes = None
            log_node_multipliers = None
            node_multipliers = None
        elif constant_rate_trajectory:
            log_scale = numpyro.sample(
                "log_trajectory_scale",
                dist.Normal(
                    jnp.log(trajectory_prior_rate), trajectory_rate_prior_sigma_log
                ),
            )
            np_log_rate_nodes = None
            log_node_multipliers = None
            node_multipliers = None
        elif independent_tile_trajectory_active:
            log_scale = jnp.asarray(0.0)
            np_log_rate_nodes = None
            log_node_multipliers = None
            node_multipliers = None
        elif fully_np_trajectory:
            log_scale = jnp.asarray(0.0)
            if whiten_np_trajectory:
                rate_likelihood_rotation = rate_likelihood_rotation_holder["value"]
                np_log_rate_z = numpyro.sample(
                    "np_log_rate_z",
                    dist.Normal(0.0, 1.0)
                    .expand((len(data["node_days"]),))
                    .to_event(1),
                )
                if rate_likelihood_rotation is not None:
                    np_log_rate_z = numpyro.deterministic(
                        "np_log_rate_likelihood_rotated",
                        jnp.matmul(np_log_rate_z, rate_likelihood_rotation.T),
                    )
                np_log_rate_nodes = numpyro.deterministic(
                    "np_log_rate_nodes",
                    jnp.log(trajectory_prior_rate)
                    + jnp.matmul(np_log_rate_z, rate_whitener.T),
                )
            else:
                np_log_rate_nodes = numpyro.sample(
                    "np_log_rate_nodes",
                    dist.Normal(
                        jnp.log(trajectory_prior_rate), trajectory_rate_prior_sigma_log
                    )
                    .expand((len(data["node_days"]),))
                    .to_event(1),
                )
        elif constant_rate_modulation:
            log_scale = numpyro.sample(
                "log_trajectory_scale",
                dist.Normal(
                    jnp.log(trajectory_prior_rate), trajectory_rate_prior_sigma_log
                ),
            )
            np_log_rate_nodes = None
        else:
            log_scale = numpyro.sample(
                "log_trajectory_scale",
                dist.Normal(
                    jnp.log(trajectory_prior_scale), trajectory_rate_prior_sigma_log
                ),
            )
            np_log_rate_nodes = None
        if tile_rate_variation_active and not independent_tile_trajectory_active:
            tile_log_rate_offsets = numpyro.sample(
                "trajectory_rate_tile_offsets",
                dist.Normal(0.0, tile_rate_prior_sigma_log)
                .expand((n_scas, n_tiles))
                .to_event(2),
            )
            # Keep the global trajectory scale equal to the selected-tile
            # geometric-mean rate. Without this centering, a common shift
            # between the global scale and every tile offset leaves the
            # likelihood unchanged.
            tile_log_rate_offsets = tile_log_rate_offsets - jnp.mean(
                tile_log_rate_offsets, axis=1, keepdims=True
            )
            numpyro.deterministic(
                "centered_trajectory_rate_tile_offsets",
                tile_log_rate_offsets,
            )
            if neighbor_edges.size and tile_rate_neighbor_smoothness:
                edge_indices = jnp.asarray(neighbor_edges, dtype=jnp.int32)
                neighbor_difference = (
                    jnp.take(tile_log_rate_offsets, edge_indices[:, 0], axis=1)
                    - jnp.take(tile_log_rate_offsets, edge_indices[:, 1], axis=1)
                )
                numpyro.factor(
                    "trajectory_rate_neighbor_smoothness",
                    -0.5
                    * jnp.sum(
                        (neighbor_difference / tile_rate_neighbor_sigma_log) ** 2
                    ),
                )
        else:
            tile_log_rate_offsets = jnp.zeros((n_scas, n_tiles))
        if absolute_substrate_power:
            if fixed_absolute_hgcdte_scales:
                absolute_hgcdte_n_log_scale = numpyro.deterministic(
                    "absolute_hgcdte_n_log_scale",
                    jnp.zeros((n_scas, n_tiles)),
                )
                absolute_hgcdte_k_log_scale = numpyro.deterministic(
                    "absolute_hgcdte_k_log_scale",
                    jnp.zeros((n_scas, n_tiles)),
                )
            else:
                scale_shape = (
                    (n_scas,)
                    if shared_absolute_hgcdte_scales
                    else (n_scas, n_tiles)
                )
                absolute_hgcdte_n_log_scale = numpyro.sample(
                    "absolute_hgcdte_n_log_scale",
                    dist.Normal(0.0, absolute_hgcdte_scale_prior_sigma_log)
                    .expand(scale_shape)
                    .to_event(len(scale_shape)),
                )
                absolute_hgcdte_k_log_scale = numpyro.sample(
                    "absolute_hgcdte_k_log_scale",
                    dist.Normal(0.0, absolute_hgcdte_scale_prior_sigma_log)
                    .expand(scale_shape)
                    .to_event(len(scale_shape)),
                )
            hgcdte_parameters = jnp.stack(
                (absolute_hgcdte_n_log_scale, absolute_hgcdte_k_log_scale), axis=-1
            )
            if shared_absolute_hgcdte_scales:
                hgcdte_parameters = hgcdte_parameters[row_sca_index]
            else:
                hgcdte_parameters = hgcdte_parameters.reshape((-1, 2))
        elif fit_hgcdte:
            hgcdte_n = numpyro.sample(
                "hgcdte_n",
                dist.TruncatedNormal(
                    NOMINAL_HGCDTE[0],
                    HGCDTE_PRIOR_SIGMA[0],
                    low=HGCDTE_LOWER[0],
                    high=HGCDTE_UPPER[0],
                ),
            )
            hgcdte_k = numpyro.sample(
                "hgcdte_k",
                dist.TruncatedNormal(
                    NOMINAL_HGCDTE[1],
                    HGCDTE_PRIOR_SIGMA[1],
                    low=HGCDTE_LOWER[1],
                    high=HGCDTE_UPPER[1],
                ),
            )
            hgcdte_parameters = jnp.stack((hgcdte_n, hgcdte_k))
        elif fit_hgcdte_modulation:
            hgcdte_n_log_modulation = numpyro.sample(
                "hgcdte_n_log_modulation",
                dist.Normal(
                    0.0,
                    hgcdte_modulation_prior_sigma_log,
                )
                .expand((len(HGCDTE_MODULATION_NODE_WAVELENGTHS_NM),))
                .to_event(1),
            )
            hgcdte_k_log_modulation = numpyro.sample(
                "hgcdte_k_log_modulation",
                dist.Normal(
                    0.0,
                    hgcdte_modulation_prior_sigma_log,
                )
                .expand((len(HGCDTE_MODULATION_NODE_WAVELENGTHS_NM),))
                .to_event(1),
            )
            numpyro.deterministic(
                "hgcdte_n_modulation", jnp.exp(hgcdte_n_log_modulation)
            )
            numpyro.deterministic(
                "hgcdte_k_modulation", jnp.exp(hgcdte_k_log_modulation)
            )
            hgcdte_parameters = jnp.concatenate(
                (hgcdte_n_log_modulation, hgcdte_k_log_modulation)
            )
        else:
            hgcdte_parameters = None
        if fit_ice_optical_modulation:
            ice_n_log_modulation = numpyro.sample(
                "ice_n_log_modulation",
                dist.Normal(
                    0.0,
                    ice_n_modulation_prior_sigma_log,
                )
                .expand((len(ICE_MODULATION_NODE_WAVELENGTHS_NM),))
                .to_event(1),
            )
            ice_k_log_modulation = numpyro.sample(
                "ice_k_log_modulation",
                dist.Normal(
                    0.0,
                    ice_modulation_prior_sigma_log,
                )
                .expand((len(ICE_MODULATION_NODE_WAVELENGTHS_NM),))
                .to_event(1),
            )
            numpyro.deterministic(
                "ice_n_modulation", jnp.exp(ice_n_log_modulation)
            )
            numpyro.deterministic(
                "ice_k_modulation", jnp.exp(ice_k_log_modulation)
            )
            ice_modulation_parameters = jnp.concatenate(
                (ice_n_log_modulation, ice_k_log_modulation)
            )
        else:
            ice_modulation_parameters = None
        if fit_post_reset_tile_log_offset:
            post_reset_tile_log_offsets = numpyro.sample(
                "post_reset_tile_log_offset",
                dist.Normal(0.0, post_reset_tile_log_offset_prior_sigma)
                .expand((n_scas, n_tiles))
                .to_event(2),
            )
        else:
            post_reset_tile_log_offsets = None
        if fit_incidence_angle:
            incidence_angle_prior = (
                dist.Uniform(0.0, INCIDENCE_ANGLE_MAX_DEG)
                if incidence_angle_prior_uniform
                else dist.TruncatedNormal(
                    0.0,
                    incidence_angle_prior_sigma_deg,
                    low=0.0,
                    high=INCIDENCE_ANGLE_MAX_DEG,
                )
            )
            incidence_angle_deg = numpyro.sample(
                "incidence_angle_deg",
                incidence_angle_prior,
            )
        else:
            incidence_angle_deg = None
        detector_thickness_for_prediction = None
        if finite_detector_active:
            if fit_detector_thickness:
                detector_log_thickness_nm = numpyro.sample(
                    "detector_log_thickness_nm",
                    dist.Normal(
                        jnp.log(detector_thickness_nm),
                        detector_thickness_prior_sigma_log,
                    )
                    .expand((n_scas, n_tiles))
                    .to_event(2),
                )
                if neighbor_edges.size:
                    edge_indices = jnp.asarray(neighbor_edges, dtype=jnp.int32)
                    neighbor_difference = (
                        jnp.take(
                            detector_log_thickness_nm, edge_indices[:, 0], axis=1
                        )
                        - jnp.take(
                            detector_log_thickness_nm, edge_indices[:, 1], axis=1
                        )
                    )
                    numpyro.factor(
                        "detector_thickness_neighbor_smoothness",
                        -0.5
                        * jnp.sum(
                            (
                                neighbor_difference
                                / detector_thickness_neighbor_sigma_log
                            )
                            ** 2
                        ),
                    )
                detector_thickness_values = jnp.exp(detector_log_thickness_nm)
            else:
                detector_thickness_values = jnp.full(
                    (n_scas, n_tiles), detector_thickness_nm
                )
            numpyro.deterministic(
                "detector_thickness_nm", detector_thickness_values
            )
            detector_thickness_for_prediction = detector_thickness_values.reshape(-1)
        if fit_conductive_layer:
            conductive_log_thickness_nm = numpyro.sample(
                "conductive_log_thickness_nm",
                dist.Normal(
                    CONDUCTIVE_LOG_THICKNESS_PRIOR_MEAN,
                    CONDUCTIVE_LOG_PRIOR_SIGMA,
                ),
            )
            conductive_log_plasma_strength = numpyro.sample(
                "conductive_log_plasma_strength",
                dist.Normal(
                    CONDUCTIVE_LOG_PLASMA_STRENGTH_PRIOR_MEAN,
                    CONDUCTIVE_LOG_PRIOR_SIGMA,
                ),
            )
            conductive_log_damping_ratio = numpyro.sample(
                "conductive_log_damping_ratio",
                dist.Normal(
                    CONDUCTIVE_LOG_DAMPING_RATIO_PRIOR_MEAN,
                    CONDUCTIVE_LOG_PRIOR_SIGMA,
                ),
            )
            conductive_parameters = jnp.stack(
                (
                    conductive_log_thickness_nm,
                    conductive_log_plasma_strength,
                    conductive_log_damping_ratio,
                )
            )
            numpyro.deterministic(
                "conductive_thickness_nm",
                jnp.exp(conductive_log_thickness_nm),
            )
            numpyro.deterministic(
                "conductive_plasma_strength",
                jnp.exp(conductive_log_plasma_strength),
            )
            numpyro.deterministic(
                "conductive_damping_ratio",
                jnp.exp(conductive_log_damping_ratio),
            )
        else:
            conductive_parameters = None
        if fit_qe_rbf:
            qe_rbf_log_modulation = numpyro.sample(
                "qe_rbf_log_modulation",
                dist.Normal(0.0, qe_rbf_prior_sigma_log)
                .expand((len(qe_rbf_nodes),))
                .to_event(1),
            )
            qe_rbf_parameters = qe_rbf_log_modulation
            numpyro.deterministic(
                "qe_rbf_modulation",
                jnp.exp(qe_rbf_log_modulation),
            )
        else:
            qe_rbf_parameters = None
        if not fit_acm_step:
            acm_step = numpyro.deterministic(
                "acm_step_log_offset",
                jnp.zeros(6) if led_dependent_acm_step else jnp.asarray(0.0),
            )
        elif led_dependent_acm_step:
            acm_step = numpyro.sample(
                "acm_step_log_offset",
                dist.Normal(0.0, acm_step_prior_sigma).expand((6,)).to_event(1),
            )
        else:
            acm_step = numpyro.sample(
                "acm_step_log_offset", dist.Normal(0.0, acm_step_prior_sigma)
            )
        if fit_late_acm_step:
            late_acm_step = numpyro.sample(
                "late_acm_step_log_offset",
                dist.Normal(0.0, late_acm_step_prior_sigma),
            )
        else:
            late_acm_step = jnp.asarray(0.0)
        if fit_additional_acm_step:
            additional_acm_step = numpyro.sample(
                "additional_acm_step_log_offset",
                dist.Normal(0.0, additional_acm_step_prior_sigma),
            )
        else:
            additional_acm_step = jnp.asarray(0.0)
        if independent_tile_trajectory_active:
            log_node_multipliers = jnp.zeros(len(data["repeat_values"]))
            node_multipliers = jnp.ones(len(data["repeat_values"]))
            np_smoothness = jnp.asarray(0.0)
        elif fully_np_trajectory:
            log_node_multipliers = np_log_rate_nodes - jnp.log(
                trajectory_prior_rate
            )
            node_multipliers = jnp.exp(log_node_multipliers)
            np_smoothness = log_rate_smoothness(np_log_rate_nodes)
            if not whiten_np_trajectory:
                numpyro.factor(
                    "np_smoothness_regularization",
                    -likelihood_scale * np_smoothness_weight * np_smoothness,
                )
        elif np_modulation or constant_rate_modulation:
            np_node_raw = numpyro.sample(
                "np_node_raw",
                dist.Normal(0.0, 1.0)
                .expand((len(data["repeat_values"]) - 1,))
                .to_event(1),
            )
            log_node_limit = jnp.log(np_max_node_multiplier)
            log_node_multipliers = log_node_limit * jnp.tanh(np_node_raw)
            log_node_multipliers = jnp.concatenate(
                (jnp.zeros(1), log_node_multipliers)
            )
            node_multipliers = jnp.exp(log_node_multipliers)
            np_entropy = jnp.mean(
                node_multipliers * log_node_multipliers
                - node_multipliers
                + 1.0
            )
            numpyro.factor(
                "np_entropy_regularization",
                -likelihood_scale * np_entropy_weight * np_entropy,
            )
            if np_modulation:
                np_smoothness = log_rate_smoothness(log_node_multipliers)
            else:
                np_smoothness = log_rate_smoothness(log_node_multipliers)
            smoothness_weight = (
                np_smoothness_weight
                if np_modulation
                else constant_rate_smoothness_weight
            )
            if smoothness_weight > 0.0:
                numpyro.factor(
                    "np_smoothness_regularization",
                    -likelihood_scale * smoothness_weight * np_smoothness,
                )
        else:
            log_node_multipliers = jnp.zeros(len(data["repeat_values"]))
            node_multipliers = jnp.ones(len(data["repeat_values"]))
        tile_log_rate_node_profiles = None
        if tile_rate_trajectory_variation_active:
            if independent_tile_trajectory_active:
                tile_log_rate_node_profiles = numpyro.sample(
                    "trajectory_rate_tile_node_log_rates",
                    dist.Normal(
                        jnp.log(trajectory_prior_rate), trajectory_rate_prior_sigma_log
                    )
                    .expand((n_scas, n_tiles, len(data["node_days"])))
                    .to_event(3),
                )
                tile_log_rate_offsets = jnp.zeros((n_scas, n_tiles))
                tile_mean_log_rates = jnp.mean(tile_log_rate_node_profiles, axis=-1)
                centered_tile_mean_log_rates = tile_mean_log_rates - jnp.mean(
                    tile_mean_log_rates, axis=1, keepdims=True
                )
                numpyro.deterministic(
                    "centered_trajectory_rate_tile_offsets",
                    centered_tile_mean_log_rates,
                )
            else:
                raw_tile_node_offsets = numpyro.sample(
                    "trajectory_rate_tile_node_offsets",
                    dist.Normal(0.0, tile_rate_prior_sigma_log)
                    .expand(
                        (
                            n_scas,
                            n_tiles,
                            len(data["node_days"]) - 1,
                        )
                    )
                    .to_event(3),
                )
                centered_tile_node_offsets = raw_tile_node_offsets - jnp.mean(
                    raw_tile_node_offsets, axis=1, keepdims=True
                )
                tile_log_rate_node_offsets = jnp.concatenate(
                    (
                        jnp.zeros((n_scas, n_tiles, 1)),
                        centered_tile_node_offsets,
                    ),
                    axis=-1,
                )
                tile_log_rate_node_profiles = (
                    (np_log_rate_nodes if fully_np_trajectory else log_node_multipliers)[
                        None, None, :
                    ]
                    + tile_log_rate_node_offsets
                )
                if neighbor_edges.size and tile_rate_neighbor_smoothness:
                    edge_indices = jnp.asarray(neighbor_edges, dtype=jnp.int32)
                    node_neighbor_difference = (
                        jnp.take(tile_log_rate_node_offsets, edge_indices[:, 0], axis=1)
                        - jnp.take(tile_log_rate_node_offsets, edge_indices[:, 1], axis=1)
                    )
                    numpyro.factor(
                        "trajectory_rate_tile_node_neighbor_smoothness",
                        -0.5
                        * jnp.sum(
                            (
                                node_neighbor_difference
                                / tile_rate_neighbor_sigma_log
                            )
                            ** 2
                        ),
                    )
            if tile_rate_trajectory_smoothness_weight > 0.0:
                flat_tile_log_rate_profiles = tile_log_rate_node_profiles.reshape(
                    (-1, tile_log_rate_node_profiles.shape[-1])
                )
                tile_node_curvature = jax.vmap(log_rate_smoothness)(
                    flat_tile_log_rate_profiles
                )
                numpyro.factor(
                    "trajectory_rate_tile_node_smoothness",
                    -likelihood_scale
                    * tile_rate_trajectory_smoothness_weight
                    * jnp.mean(tile_node_curvature),
                )
            if not independent_tile_trajectory_active:
                numpyro.deterministic(
                    "centered_trajectory_rate_tile_node_offsets",
                    tile_log_rate_node_offsets,
                )
        if tile_rate_trajectory_variation_active:
            flat_tile_profiles = tile_log_rate_node_profiles.reshape(
                (-1, tile_log_rate_node_profiles.shape[-1])
            )
            integrated_values = jax.vmap(
                lambda profile: integrate_rate_trajectory(
                    jax,
                    jnp,
                    days,
                    node_days,
                    integration_times,
                    0.0,
                    profile,
                    use_square_root=not (
                        fully_np_trajectory
                        or constant_rate_modulation
                        or constant_rate_trajectory
                        or fixed_trajectory_rate_nm_day is not None
                    ),
                    anchor_day=ice_reset_day,
                )
            )(flat_tile_profiles)
            if ice_reset_day is None:
                (
                    integrated_thickness,
                    base_instantaneous_rate,
                    node_integrated_thickness,
                    base_node_instantaneous_rate,
                ) = tuple(
                    values.reshape((n_scas * n_tiles, -1))
                    for values in integrated_values
                )
                reset_integrated_thickness = None
            else:
                (
                    integrated_thickness,
                    base_instantaneous_rate,
                    node_integrated_thickness,
                    base_node_instantaneous_rate,
                    reset_integrated_thickness,
                ) = integrated_values
                (
                    integrated_thickness,
                    base_instantaneous_rate,
                    node_integrated_thickness,
                    base_node_instantaneous_rate,
                ) = tuple(
                    values.reshape((n_scas * n_tiles, -1))
                    for values in (
                        integrated_thickness,
                        base_instantaneous_rate,
                        node_integrated_thickness,
                        base_node_instantaneous_rate,
                    )
                )
                reset_integrated_thickness = reset_integrated_thickness.reshape(-1)
        else:
            integrated_values = integrate_rate_trajectory(
                jax,
                jnp,
                days,
                node_days,
                integration_times,
                0.0,
                np_log_rate_nodes
                if fully_np_trajectory
                else log_node_multipliers,
                use_square_root=not (
                    fully_np_trajectory
                    or constant_rate_modulation
                    or constant_rate_trajectory
                    or fixed_trajectory_rate_nm_day is not None
                ),
                anchor_day=ice_reset_day,
            )
            if ice_reset_day is None:
                (
                    integrated_thickness,
                    base_instantaneous_rate,
                    node_integrated_thickness,
                    base_node_instantaneous_rate,
                ) = integrated_values
                reset_integrated_thickness = None
            else:
                (
                    integrated_thickness,
                    base_instantaneous_rate,
                    node_integrated_thickness,
                    base_node_instantaneous_rate,
                    reset_integrated_thickness,
                ) = integrated_values
        if pivot_initial_thickness and fixed_initial_thickness_nm is None:
            pivot_integrated_thickness = (
                node_integrated_thickness[0, pivot_node_index]
                if tile_rate_trajectory_variation_active
                else node_integrated_thickness[pivot_node_index]
            )
            pivot_thickness = (
                pivot_initial_center
                + initial_thickness_prior_sigma_nm * initial_thickness_pivot_z
            )
            d0 = pivot_thickness - pivot_integrated_thickness
            initial_thickness_prior = dist.TruncatedNormal(
                initial_thickness_prior_mean_nm,
                initial_thickness_prior_sigma_nm,
                low=INITIAL_THICKNESS_PRIOR_LOW_NM,
                high=INITIAL_THICKNESS_PRIOR_HIGH_NM,
            )
            if map_safe_initial_thickness_holder["value"]:
                base_initial_thickness_prior = dist.Normal(
                    initial_thickness_prior_mean_nm,
                    initial_thickness_prior_sigma_nm,
                )
                normalization = jnp.log(
                    base_initial_thickness_prior.cdf(
                        initial_thickness_prior.high
                    )
                    - base_initial_thickness_prior.cdf(
                        initial_thickness_prior.low
                    )
                )
                clipped_d0 = jnp.clip(
                    d0,
                    initial_thickness_prior.low,
                    initial_thickness_prior.high,
                )
                distance_outside = jnp.where(
                    d0 < initial_thickness_prior.low,
                    initial_thickness_prior.low - d0,
                    jnp.where(
                        d0 > initial_thickness_prior.high,
                        d0 - initial_thickness_prior.high,
                        0.0,
                    ),
                )
                safe_log_prob = (
                    base_initial_thickness_prior.log_prob(clipped_d0)
                    - normalization
                    - 0.5
                    * (distance_outside / 1.0e-6) ** 2
                )
                initial_thickness_log_prob = jnp.where(
                    (d0 >= initial_thickness_prior.low)
                    & (d0 <= initial_thickness_prior.high),
                    base_initial_thickness_prior.log_prob(d0) - normalization,
                    safe_log_prob,
                )
            else:
                initial_thickness_log_prob = initial_thickness_prior.log_prob(d0)
            numpyro.factor(
                "initial_thickness_prior",
                initial_thickness_log_prob
                + jnp.log(initial_thickness_prior_sigma_nm)
                - dist.Normal(0.0, 1.0).log_prob(initial_thickness_pivot_z),
            )
            numpyro.deterministic("initial_thickness_nm", d0)
        row_log_rate_scales = log_scale + tile_log_rate_offsets.reshape(-1)
        row_rate_scales = jnp.exp(row_log_rate_scales)
        d0_rows = (
            d0.reshape(-1)
            if initial_thickness_tile_active
            else jnp.broadcast_to(d0, (n_scas * n_tiles,))
        )
        integrated_for_rows = (
            integrated_thickness
            if tile_rate_trajectory_variation_active
            else integrated_thickness[None, :]
        )
        node_integrated_for_rows = (
            node_integrated_thickness
            if tile_rate_trajectory_variation_active
            else node_integrated_thickness[None, :]
        )
        if ice_reset_day is not None:
            reset_d0_rows = (
                reset_d0.reshape(-1)
                if initial_thickness_tile_active
                else jnp.broadcast_to(reset_d0, (n_scas * n_tiles,))
            )
            reset_integrated_rows = (
                reset_integrated_thickness
                if tile_rate_trajectory_variation_active
                else jnp.broadcast_to(
                    reset_integrated_thickness, (n_scas * n_tiles,)
                )
            )
            post_reset_days = days[None, :] >= ice_reset_day
            post_reset_nodes = node_days[None, :] >= ice_reset_day
            integrated_for_rows = jnp.where(
                post_reset_days,
                integrated_for_rows - reset_integrated_rows[:, None],
                integrated_for_rows,
            )
            node_integrated_for_rows = jnp.where(
                post_reset_nodes,
                node_integrated_for_rows - reset_integrated_rows[:, None],
                node_integrated_for_rows,
            )
            d0_for_rows = jnp.where(
                post_reset_days,
                reset_d0_rows[:, None],
                d0_rows[:, None],
            )
            d0_for_nodes = jnp.where(
                post_reset_nodes,
                reset_d0_rows[:, None],
                d0_rows[:, None],
            )
        else:
            d0_for_rows = d0_rows[:, None]
            d0_for_nodes = d0_rows[:, None]
        tile_ice_thickness = d0_for_rows + row_rate_scales[:, None] * (
            integrated_for_rows
        )
        tile_instantaneous_rate = row_rate_scales[:, None] * (
            base_instantaneous_rate
            if tile_rate_trajectory_variation_active
            else base_instantaneous_rate[None, :]
        )
        tile_node_ice_thickness = d0_for_nodes + row_rate_scales[:, None] * (
            node_integrated_for_rows
        )
        tile_node_instantaneous_rate = (
            row_rate_scales[:, None]
            * (
                base_node_instantaneous_rate
                if tile_rate_trajectory_variation_active
                else base_node_instantaneous_rate[None, :]
            )
        )
        if rowwise_tile_prediction_active:
            ice_thickness = tile_ice_thickness
            instantaneous_rate = jnp.mean(tile_instantaneous_rate, axis=0)
            node_ice_thickness = jnp.mean(tile_node_ice_thickness, axis=0)
            node_instantaneous_rate = jnp.mean(tile_node_instantaneous_rate, axis=0)
        else:
            ice_thickness = tile_ice_thickness[0]
            instantaneous_rate = tile_instantaneous_rate[0]
            node_ice_thickness = tile_node_ice_thickness[0]
            node_instantaneous_rate = tile_node_instantaneous_rate[0]
        if include_map_prediction:
            total_sse, map_fitted_prediction = residual_sum_squares(
                ice_thickness,
                ar_parameters_for_prediction,
                acm_step,
                hgcdte_parameters,
                conductive_parameters,
                qe_rbf_parameters,
                early_time_scale,
                late_acm_step,
                detector_thickness_for_prediction,
                incidence_angle_deg,
                additional_acm_step,
                post_reset_ice_index,
                pre_reset_ice_index,
                ice_modulation_parameters,
                post_reset_tile_log_offsets,
                sca_ar_opd_nm,
                complex_q_parameters,
                return_prediction=True,
            )
        else:
            total_sse = residual_sum_squares(
                ice_thickness,
                ar_parameters_for_prediction,
                acm_step,
                hgcdte_parameters,
                conductive_parameters,
                qe_rbf_parameters,
                early_time_scale,
                late_acm_step,
                detector_thickness_for_prediction,
                incidence_angle_deg,
                additional_acm_step,
                post_reset_ice_index,
                pre_reset_ice_index,
                ice_modulation_parameters,
                post_reset_tile_log_offsets,
                sca_ar_opd_nm,
                complex_q_parameters,
            )
        numpyro.factor(
            "tile_gaussian_log_likelihood",
            -0.5
            * effective_likelihood_weight
            * total_sse
            / (log_noise_sigma**2),
        )
        if include_map_prediction:
            numpyro.deterministic(
                "_map_least_squares_prediction", map_fitted_prediction
            )
        numpyro.deterministic("ar_parameters", ar_parameters)
        numpyro.deterministic(
            "ar_mean",
            physical_ar_parameters(ar_mu_raw),
        )
        if n_scas > 1 and ar_sca_raw is not None:
            numpyro.deterministic(
                "ar_sca_mean",
                physical_ar_parameters(ar_sca_raw),
            )
        numpyro.deterministic("ice_thickness", ice_thickness)
        numpyro.deterministic("instantaneous_rate_nm_day", instantaneous_rate)
        numpyro.deterministic(
            "node_instantaneous_rate_nm_day", node_instantaneous_rate
        )
        first_group = int(np.argmin(data["days"]))
        last_group = int(np.argmax(data["days"]))
        if rowwise_tile_prediction_active:
            numpyro.deterministic(
                "tile_ice_thickness", tile_ice_thickness
            )
            numpyro.deterministic(
                "tile_instantaneous_rate_nm_day", tile_instantaneous_rate
            )
            numpyro.deterministic(
                "tile_node_thickness", tile_node_ice_thickness
            )
            numpyro.deterministic(
                "tile_node_instantaneous_rate_nm_day",
                tile_node_instantaneous_rate,
            )
            numpyro.deterministic(
                "tile_initial_rate_nm_day", tile_instantaneous_rate[:, first_group]
            )
            numpyro.deterministic(
                "tile_final_rate_nm_day", tile_instantaneous_rate[:, last_group]
            )
            numpyro.deterministic(
                "tile_mean_rate_nm_day",
                (tile_ice_thickness[:, last_group] - tile_ice_thickness[:, first_group])
                / (jnp.max(days) - jnp.min(days)),
            )
        numpyro.deterministic(
            "initial_rate_nm_day", instantaneous_rate[first_group]
        )
        numpyro.deterministic("final_rate_nm_day", instantaneous_rate[last_group])
        if (
            np_modulation
            or constant_rate_modulation
            or (fully_np_trajectory and not independent_tile_trajectory_active)
        ):
            numpyro.deterministic("np_log_node_multipliers", log_node_multipliers)
            numpyro.deterministic("np_node_multipliers", node_multipliers)
            numpyro.deterministic(
                "np_node_thickness",
                node_ice_thickness,
            )
        if tile_rate_variation_active:
            rate_numerator = jnp.mean(
                tile_ice_thickness[:, last_group]
                - tile_ice_thickness[:, first_group]
            )
        else:
            rate_numerator = ice_thickness[last_group] - ice_thickness[first_group]
        numpyro.deterministic(
            "mean_rate_nm_day",
            rate_numerator / (jnp.max(days) - jnp.min(days)),
        )
        if np_modulation or constant_rate_modulation:
            numpyro.deterministic("np_entropy", np_entropy)
        if (
            np_modulation
            or constant_rate_modulation
            or (fully_np_trajectory and not independent_tile_trajectory_active)
        ):
            numpyro.deterministic("np_smoothness", np_smoothness)

    model.np_rate_likelihood_rotation_holder = rate_likelihood_rotation_holder
    model.map_safe_initial_thickness_holder = map_safe_initial_thickness_holder
    return model


def map_start_reference(args, ar_reference, ar_lower, ar_upper):
    map_design_wavelength_nm = getattr(args, "map_design_wavelength_nm", None)
    if map_design_wavelength_nm is None:
        return np.asarray(ar_reference).copy()
    start_reference = dielectric_ar_reference(
        args.ar_layers,
        design_wavelength_nm=map_design_wavelength_nm,
        middle_layer_eighth_wave=getattr(
            args, "map_middle_layer_eighth_wave", False
        ),
        middle_layer_half_wave=getattr(args, "map_middle_layer_half_wave", False),
    )
    if np.any(start_reference <= ar_lower) or np.any(start_reference >= ar_upper):
        raise ValueError("MAP start falls outside the AR parameter bounds")
    return start_reference


def initial_values_for_data(data, args, ar_reference, ar_lower, ar_upper):
    """Build compatible sample-site starts for a selected tile subset."""
    start_reference = map_start_reference(args, ar_reference, ar_lower, ar_upper)
    active_parameter_indices = ar_active_parameter_indices(
        args.ar_layers,
        getattr(args, "fit_ar_absorption", False),
        getattr(args, "fixed_ar_indices", None),
    )
    reference_raw = ar_reference_raw(
        start_reference,
        ar_lower,
        ar_upper,
        active_parameter_indices,
        ordered_ar_indices=getattr(args, "ordered_ar_indices", False),
    )
    if getattr(args, "independent_tile_ar_parameters", False):
        initial_values = {
            "ar_tile_raw": np.broadcast_to(
                reference_raw[None, None, :],
                (
                    data["n_scas"],
                    data["n_tiles"],
                    len(active_parameter_indices),
                ),
            ).copy()
        }
    elif getattr(args, "standardize_ar_mean", False):
        initial_values = {"ar_mu_z": np.zeros(len(active_parameter_indices))}
    else:
        initial_values = {"ar_mu_raw": reference_raw}
    if getattr(args, "fixed_initial_thickness_nm", None) is None:
        if getattr(args, "tile_initial_thickness_lognormal", False):
            initial_values["initial_thickness_log_nm"] = np.full(
                (data["n_scas"], data["n_tiles"]),
                np.log(getattr(args, "tile_initial_thickness_log_median_nm", 10.0)),
            )
            if getattr(args, "ice_reset_day", None) is not None:
                initial_values["reset_initial_thickness_log_nm"] = np.full(
                    (data["n_scas"], data["n_tiles"]),
                    np.log(
                        getattr(args, "tile_initial_thickness_log_median_nm", 10.0)
                    ),
                )
        elif not getattr(args, "pivot_initial_thickness", False):
            initial_values["initial_thickness_nm"] = getattr(
                args, "initial_thickness_init_nm", INITIAL_THICKNESS_INIT_NM
            )
            if getattr(args, "ice_reset_day", None) is not None:
                initial_values["reset_initial_thickness_nm"] = getattr(
                    args, "initial_thickness_init_nm", INITIAL_THICKNESS_INIT_NM
                )
    if getattr(args, "fit_incidence_angle", False):
        initial_values["incidence_angle_deg"] = getattr(
            args, "incidence_angle_init_deg", INCIDENCE_ANGLE_INIT_DEG
        )
    if getattr(args, "fit_complex_q", False):
        initial_values["complex_q_parameters"] = np.asarray(
            getattr(args, "complex_q_initial_parameters", COMPLEX_Q_INITIAL),
            dtype=float,
        ).copy()
    if getattr(args, "fit_post_reset_ice_index", False):
        initial_values["post_reset_ice_index"] = getattr(
            args,
            "post_reset_ice_index_prior_mean",
            POST_RESET_ICE_INDEX_PRIOR_MEAN,
        )
    if getattr(args, "fit_pre_post_ice_index", False):
        ice_n_start = getattr(args, "ice_index_prior_mean", ICE_INDEX_PRIOR_MEAN)
        initial_values.update(
            {
                "pre_reset_ice_n": ice_n_start,
                "post_reset_ice_n": ice_n_start,
                "pre_reset_ice_k": ICE_EXTINCTION_INIT,
                "post_reset_ice_k": ICE_EXTINCTION_INIT,
            }
        )
    if getattr(args, "early_time_scale_cutoff_day", None) is not None:
        early_time_configuration = early_time_scale_configuration(
            data, args.early_time_scale_cutoff_day
        )
        early_time_node_count = len(early_time_configuration["repeat"])
        if early_time_node_count > 1:
            initial_values["early_time_log_scale"] = np.full(
                early_time_node_count - 1,
                EARLY_TIME_LOG_SCALE_PRIOR_MEAN,
            )
    if getattr(args, "fixed_trajectory_rate_nm_day", None) is not None:
        pass
    elif getattr(args, "constant_rate_trajectory", False):
        trajectory_prior_rate = trajectory_prior_rate_for_data(data, args)
        initial_values["log_trajectory_scale"] = np.log(trajectory_prior_rate)
    elif getattr(args, "fully_np_trajectory", False):
        trajectory_prior_rate = trajectory_prior_rate_for_data(data, args)
        initial_log_rates = np.full(len(data["node_days"]), np.log(trajectory_prior_rate))
        if getattr(args, "independent_tile_trajectories", False):
            initial_values["trajectory_rate_tile_node_log_rates"] = np.full(
                (
                    data["n_scas"],
                    data["n_tiles"],
                    len(data["node_days"]),
                ),
                np.log(trajectory_prior_rate),
            )
        elif getattr(args, "whiten_np_trajectory", False):
            initial_values["np_log_rate_z"] = whiten_trajectory_rate_nodes(
                initial_log_rates,
                data["node_days"],
                trajectory_prior_rate,
                args.trajectory_rate_prior_sigma_log,
                trajectory_rate_likelihood_scale(data, args),
                args.np_smoothness_weight,
            )
        else:
            initial_values["np_log_rate_nodes"] = initial_log_rates
    elif getattr(args, "constant_rate_modulation", False):
        trajectory_prior_rate = (
            getattr(args, "trajectory_rate_prior_mean_nm_day", None)
            if getattr(args, "trajectory_rate_prior_mean_nm_day", None) is not None
            else trajectory_rate_per_scale(data["days"]) * TRAJECTORY_SCALE_INIT
        )
        initial_values["log_trajectory_scale"] = np.log(trajectory_prior_rate)
    else:
        initial_values["log_trajectory_scale"] = np.log(TRAJECTORY_SCALE_START)
    if getattr(args, "fit_hgcdte", False):
        initial_values.update(
            {
                "hgcdte_n": float(NOMINAL_HGCDTE[0]),
                "hgcdte_k": float(NOMINAL_HGCDTE[1]),
            }
        )
    elif getattr(args, "fit_hgcdte_modulation", False):
        initial_values.update(
            {
                "hgcdte_n_log_modulation": np.zeros(
                    len(HGCDTE_MODULATION_NODE_WAVELENGTHS_NM)
                ),
                "hgcdte_k_log_modulation": np.zeros(
                    len(HGCDTE_MODULATION_NODE_WAVELENGTHS_NM)
                ),
            }
        )
    if getattr(args, "fit_ice_optical_modulation", False):
        initial_values.update(
            {
                "ice_n_log_modulation": np.zeros(
                    len(ICE_MODULATION_NODE_WAVELENGTHS_NM)
                ),
                "ice_k_log_modulation": np.zeros(
                    len(ICE_MODULATION_NODE_WAVELENGTHS_NM)
                ),
            }
        )
    if getattr(args, "fit_post_reset_tile_log_offset", False):
        initial_values["post_reset_tile_log_offset"] = np.zeros(
            (data["n_scas"], data["n_tiles"])
        )
    if getattr(args, "absolute_substrate_power", False) and not getattr(
        args, "fixed_absolute_hgcdte_scales", False
    ):
        scale_shape = (
            (data["n_scas"],)
            if getattr(args, "shared_absolute_hgcdte_scales", False)
            else (data["n_scas"], data["n_tiles"])
        )
        initial_values.update(
            {
                "absolute_hgcdte_n_log_scale": np.zeros(scale_shape),
                "absolute_hgcdte_k_log_scale": np.zeros(scale_shape),
            }
        )
    if getattr(args, "fit_detector_thickness", False):
        initial_values["detector_log_thickness_nm"] = np.full(
            (data["n_scas"], data["n_tiles"]),
            np.log(
                getattr(
                    args,
                    "detector_thickness_nm",
                    DETECTOR_THICKNESS_DEFAULT_NM,
                )
            ),
        )
    if getattr(args, "fit_conductive_layer", False):
        initial_values.update(
            {
                "conductive_log_thickness_nm": CONDUCTIVE_LOG_THICKNESS_PRIOR_MEAN,
                "conductive_log_plasma_strength": CONDUCTIVE_LOG_PLASMA_STRENGTH_PRIOR_MEAN,
                "conductive_log_damping_ratio": CONDUCTIVE_LOG_DAMPING_RATIO_PRIOR_MEAN,
            }
        )
    if getattr(args, "fit_qe_rbf", False):
        qe_rbf_nodes, _, _ = qe_rbf_configuration(
            getattr(args, "qe_rbf_node_wavelengths_nm", None),
            getattr(args, "qe_rbf_width_nm", QE_RBF_WIDTH_NM),
            getattr(args, "qe_rbf_prior_sigma_log", QE_RBF_PRIOR_SIGMA_LOG),
        )
        initial_values["qe_rbf_log_modulation"] = np.zeros(
            len(qe_rbf_nodes)
        )
    if not getattr(args, "fixed_corrections", None) and not getattr(
        args, "corrections_already_applied", False
    ):
        initial_values["acm_step_log_offset"] = (
            np.zeros(6) if getattr(args, "led_dependent_acm_step", False) else 0.0
        )
    if getattr(args, "fit_late_acm_step", False):
        initial_values["late_acm_step_log_offset"] = np.log(0.992)
    if getattr(args, "fit_additional_acm_step", False):
        initial_values["additional_acm_step_log_offset"] = 0.0
    map_trajectory_rate_nm_day = getattr(args, "map_trajectory_rate_nm_day", None)
    if (
        map_trajectory_rate_nm_day is not None
        and getattr(args, "fixed_trajectory_rate_nm_day", None) is None
    ):
        if getattr(args, "fully_np_trajectory", False):
            map_log_rates = np.full(
                len(data["node_days"]), np.log(map_trajectory_rate_nm_day)
            )
            if getattr(args, "independent_tile_trajectories", False):
                initial_values["trajectory_rate_tile_node_log_rates"] = np.broadcast_to(
                    map_log_rates[None, None, :],
                    (
                        data["n_scas"],
                        data["n_tiles"],
                        len(data["node_days"]),
                    ),
                ).copy()
            elif getattr(args, "whiten_np_trajectory", False):
                initial_values["np_log_rate_z"] = whiten_trajectory_rate_nodes(
                    map_log_rates,
                    data["node_days"],
                    trajectory_prior_rate,
                    args.trajectory_rate_prior_sigma_log,
                    trajectory_rate_likelihood_scale(data, args),
                    args.np_smoothness_weight,
                )
            else:
                initial_values["np_log_rate_nodes"] = map_log_rates
        elif getattr(args, "constant_rate_modulation", False) or getattr(
            args, "constant_rate_trajectory", False
        ):
            initial_values["log_trajectory_scale"] = np.log(
                map_trajectory_rate_nm_day
            )
        else:
            initial_values["log_trajectory_scale"] = np.log(
                map_trajectory_rate_nm_day / trajectory_rate_per_scale(data["days"])
            )
    if (
        getattr(args, "pivot_initial_thickness", False)
        and getattr(args, "fixed_initial_thickness_nm", None) is None
    ):
        initial_values["initial_thickness_pivot_z"] = trajectory_pivot_z_from_initial_thickness(
            getattr(args, "initial_thickness_init_nm", INITIAL_THICKNESS_INIT_NM),
            data,
            args,
            trajectory_log_rates_from_initial_values(data, args, initial_values),
        )
    hierarchy_parameter_count = (
        args.ar_layers
        if args.shared_ar_optical_properties
        else len(active_parameter_indices)
    )
    if getattr(args, "fit_sca_ar_opd", False):
        initial_values["sca_ar_opd_nm"] = np.zeros(data["n_scas"])
    if (
        getattr(args, "independent_tile_ar_parameters", False)
        or is_single_tile_data(data)
        or getattr(
        args, "shared_ar_parameters_across_tiles", False
        )
    ):
        pass
    elif data["n_scas"] == 1:
        initial_values.update(
            {
                "ar_sigma_raw": np.full(
                    hierarchy_parameter_count, args.ar_tile_sigma_prior * 0.5
                ),
                "ar_z": np.zeros((len(data["tile_ids"]), hierarchy_parameter_count)),
            }
        )
    else:
        initial_values.update(
            {
                "ar_sca_sigma": np.full(
                    hierarchy_parameter_count, args.ar_sca_sigma_prior * 0.5
                ),
                "ar_sca_z": np.zeros((data["n_scas"], hierarchy_parameter_count)),
                "ar_tile_sigma": np.full(
                    hierarchy_parameter_count, args.ar_tile_sigma_prior * 0.5
                ),
                "ar_tile_z": np.zeros(
                    (
                        data["n_scas"],
                        data["n_tiles"],
                        hierarchy_parameter_count,
                    )
                ),
            }
        )
    if (
        (args.np_modulation or getattr(args, "constant_rate_modulation", False))
        and getattr(args, "fixed_trajectory_rate_nm_day", None) is None
    ):
        initial_values["np_node_raw"] = np.zeros(len(data["repeat_values"]) - 1)
    if (
        getattr(args, "tile_rate_variation", False)
        or getattr(args, "tile_rate_trajectory_variation", False)
    ) and (
        data["n_scas"] * data["n_tiles"] > 1
    ) and not getattr(args, "independent_tile_trajectories", False):
        initial_values["trajectory_rate_tile_offsets"] = np.zeros(
            (data["n_scas"], data["n_tiles"])
        )
    if getattr(args, "tile_rate_trajectory_variation", False) and (
        data["n_scas"] * data["n_tiles"] > 1
    ) and not getattr(args, "independent_tile_trajectories", False):
        initial_values["trajectory_rate_tile_node_offsets"] = np.zeros(
            (
                data["n_scas"],
                data["n_tiles"],
                len(data["node_days"]) - 1,
            )
        )
    return initial_values


def transfer_map_initial_values(
    map_values,
    full_initial_values,
    source_ar_parameter_indices=None,
    target_ar_parameter_indices=None,
    source_tile_ids=None,
    target_tile_ids=None,
    broadcast_tile_values=False,
):
    """Transfer compatible MAP sites, including AR sites with fixed indices."""
    transferred = dict(full_initial_values)
    ar_site_names = {
        "ar_mu_raw",
        "ar_tile_raw",
        "ar_sigma_raw",
        "ar_sca_sigma",
        "ar_tile_sigma",
        "ar_z",
        "ar_sca_z",
        "ar_tile_z",
    }
    hgcdte_site_names = {
        "absolute_hgcdte_n_log_scale",
        "absolute_hgcdte_k_log_scale",
    }
    if (source_ar_parameter_indices is None) != (
        target_ar_parameter_indices is None
    ):
        raise ValueError("Both AR source and target parameter indices are required")
    if (source_tile_ids is None) != (target_tile_ids is None):
        raise ValueError("Both source and target tile IDs are required")
    if source_tile_ids is not None:
        source_tile_ids = np.asarray(source_tile_ids, dtype=int).reshape(-1)
        target_tile_ids = np.asarray(target_tile_ids, dtype=int).reshape(-1)
        target_tile_positions = {
            int(tile_id): position
            for position, tile_id in enumerate(target_tile_ids)
        }
        if not set(source_tile_ids).issubset(target_tile_positions) and not (
            broadcast_tile_values and len(source_tile_ids) == 1
        ):
            raise ValueError("Source tile IDs are not a subset of target tile IDs")
    if source_ar_parameter_indices is not None:
        source_ar_parameter_indices = np.asarray(
            source_ar_parameter_indices, dtype=int
        )
        target_ar_parameter_indices = np.asarray(
            target_ar_parameter_indices, dtype=int
        )
        target_positions = {
            int(parameter_index): position
            for position, parameter_index in enumerate(target_ar_parameter_indices)
        }
    for name, value in map_values.items():
        if name not in transferred:
            continue
        value = np.asarray(value)
        target = np.asarray(transferred[name])
        if (
            name in ar_site_names
            and source_ar_parameter_indices is not None
            and value.ndim >= 1
            and target.ndim >= 1
            and value.shape[:-1] == target.shape[:-1]
            and value.shape[-1] == source_ar_parameter_indices.size
            and target.shape[-1] == target_ar_parameter_indices.size
        ):
            target = target.copy()
            for source_position, parameter_index in enumerate(
                source_ar_parameter_indices
            ):
                target_position = target_positions.get(int(parameter_index))
                if target_position is not None:
                    target[..., target_position] = value[..., source_position]
            transferred[name] = target
            continue
        tile_axis = _tile_axis_for_site(name)
        if (
            source_tile_ids is not None
            and tile_axis is not None
            and value.shape != target.shape
            and warm_start_value_can_transfer(
                name, value, target, source_tile_ids, target_tile_ids
            )
        ):
            axis = tile_axis % value.ndim
            target = target.copy()
            target_positions = (
                range(len(target_tile_ids))
                if broadcast_tile_values and len(source_tile_ids) == 1
                else [target_tile_positions[int(tile_id)] for tile_id in source_tile_ids]
            )
            for target_position in target_positions:
                source_position = 0 if broadcast_tile_values else source_tile_ids.tolist().index(
                    int(target_tile_ids[target_position])
                )
                source_index = [slice(None)] * value.ndim
                target_index = [slice(None)] * target.ndim
                source_index[axis] = source_position
                target_index[axis] = target_position
                target[tuple(target_index)] = value[tuple(source_index)]
            transferred[name] = target
            continue
        if (
            name in hgcdte_site_names
            and value.ndim + 1 == target.ndim
            and value.shape == target.shape[:-1]
        ):
            transferred[name] = np.broadcast_to(
                value[..., None], target.shape
            ).copy()
            continue
        if value.shape == target.shape:
            transferred[name] = value
    return transferred


def build_model_for_args(
    data,
    context,
    args,
    ar_prior_reference,
    incidence_chief_ray_tilts_deg,
):
    """Build one model stage from its explicit production configuration."""
    corrections_applied = bool(
        getattr(args, "corrections_already_applied", False)
        or args.fixed_corrections is not None
    )
    return build_model(
        data,
        context,
        args.log_noise_sigma,
        args.likelihood_weight,
        ar_mean_prior_sigma=args.ar_mean_prior_sigma,
        ar_prior_reference=ar_prior_reference,
        ar_layers=args.ar_layers,
        np_modulation=args.np_modulation,
        constant_rate_modulation=args.constant_rate_modulation,
        constant_rate_trajectory=getattr(args, "constant_rate_trajectory", False),
        constant_rate_smoothness_weight=args.constant_rate_smoothness_weight,
        fully_np_trajectory=args.fully_np_trajectory,
        standardize_ar_mean=args.standardize_ar_mean,
        rotate_ar_coordinates=args.rotate_ar_coordinates,
        whiten_np_trajectory=args.whiten_np_trajectory,
        np_rate_likelihood_rotation=getattr(
            args, "np_rate_likelihood_rotation", None
        ),
        pivot_initial_thickness=args.pivot_initial_thickness,
        map_safe_initial_thickness=bool(args.map_steps),
        np_entropy_weight=args.np_entropy_weight,
        np_smoothness_weight=args.np_smoothness_weight,
        np_max_node_multiplier=args.np_max_node_multiplier,
        ar_sca_sigma_prior=args.ar_sca_sigma_prior,
        ar_tile_sigma_prior=args.ar_tile_sigma_prior,
        fit_sca_ar_opd=args.fit_sca_ar_opd,
        sca_ar_opd_prior_sigma_nm=args.sca_ar_opd_prior_sigma_nm,
        shared_ar_optical_properties=args.shared_ar_optical_properties,
        shared_ar_parameters_across_tiles=args.shared_ar_parameters_across_tiles,
        independent_tile_ar_parameters=getattr(
            args, "independent_tile_ar_parameters", False
        ),
        fit_ar_absorption=args.fit_ar_absorption,
        fit_hgcdte=args.fit_hgcdte,
        fit_hgcdte_modulation=args.fit_hgcdte_modulation,
        hgcdte_modulation_prior_sigma_log=args.hgcdte_modulation_prior_sigma_log,
        fit_ice_optical_modulation=args.fit_ice_optical_modulation,
        ice_modulation_prior_sigma_log=args.ice_modulation_prior_sigma_log,
        ice_n_modulation_prior_sigma_log=args.ice_n_modulation_prior_sigma_log,
        fit_conductive_layer=args.fit_conductive_layer,
        finite_hgcdte_detector=args.finite_hgcdte_detector,
        detector_thickness_nm=args.detector_thickness_nm,
        fit_detector_thickness=args.fit_detector_thickness,
        detector_thickness_prior_sigma_log=args.detector_thickness_prior_sigma_log,
        detector_thickness_neighbor_sigma_log=args.detector_thickness_neighbor_sigma_log,
        detector_backside_index=args.detector_backside_index,
        fit_qe_rbf=args.fit_qe_rbf,
        qe_rbf_node_wavelengths_nm=args.qe_rbf_node_wavelengths_nm,
        qe_rbf_width_nm=args.qe_rbf_width_nm,
        qe_rbf_prior_sigma_log=args.qe_rbf_prior_sigma_log,
        use_asw_optical_constants=args.use_asw_optical_constants,
        use_hgcdte_optical_constants=(
            args.use_hgcdte_optical_constants or args.fit_hgcdte_modulation
        ),
        incidence_cone_half_angle_deg=args.incidence_cone_half_angle_deg,
        incidence_pupil_half_angle_deg=args.incidence_pupil_half_angle_deg,
        incidence_chief_ray_tilts_deg=incidence_chief_ray_tilts_deg,
        fit_incidence_angle=args.fit_incidence_angle,
        incidence_angle_prior_sigma_deg=args.incidence_angle_prior_sigma_deg,
        incidence_angle_prior_uniform=args.incidence_angle_prior_uniform,
        led_dependent_acm_step=args.led_dependent_acm_step,
        fit_acm_step=not corrections_applied,
        acm_step_prior_sigma=args.acm_step_prior_sigma,
        fit_late_acm_step=(
            args.fit_late_acm_step if not corrections_applied else False
        ),
        late_acm_step_prior_sigma=args.late_acm_step_prior_sigma,
        fit_additional_acm_step=(
            args.fit_additional_acm_step if not corrections_applied else False
        ),
        additional_acm_step_prior_sigma=args.additional_acm_step_prior_sigma,
        trajectory_rate_prior_mean_nm_day=args.trajectory_rate_prior_mean_nm_day,
        fixed_trajectory_rate_nm_day=args.fixed_trajectory_rate_nm_day,
        fixed_ar_indices=getattr(args, "fixed_ar_indices", None),
        ordered_ar_indices=getattr(args, "ordered_ar_indices", False),
        trajectory_rate_prior_sigma_log=args.trajectory_rate_prior_sigma_log,
        tile_rate_variation=args.tile_rate_variation,
        tile_rate_trajectory_variation=args.tile_rate_trajectory_variation,
        independent_tile_trajectories=args.independent_tile_trajectories,
        tile_rate_trajectory_smoothness_weight=(
            args.tile_rate_trajectory_smoothness_weight
        ),
        tile_rate_prior_sigma_log=args.tile_rate_prior_sigma_log,
        tile_rate_neighbor_sigma_log=args.tile_rate_neighbor_sigma_log,
        tile_rate_neighbor_smoothness=args.tile_rate_neighbor_smoothness,
        fixed_initial_thickness_nm=args.fixed_initial_thickness_nm,
        initial_thickness_prior_mean_nm=args.initial_thickness_prior_mean_nm,
        initial_thickness_prior_sigma_nm=args.initial_thickness_prior_sigma_nm,
        early_time_cutoff_day=(
            args.early_time_scale_cutoff_day if not corrections_applied else None
        ),
        early_time_scale_prior_sigma=args.early_time_scale_prior_sigma,
        absolute_hgcdte_scale_prior_sigma_log=args.absolute_hgcdte_scale_prior_sigma_log,
        fixed_absolute_hgcdte_scales=args.fixed_absolute_hgcdte_scales,
        shared_absolute_hgcdte_scales=args.shared_absolute_hgcdte_scales,
        initial_thickness_prior_uniform=args.initial_thickness_prior_uniform,
        tile_initial_thickness_lognormal=args.tile_initial_thickness_lognormal,
        tile_initial_thickness_log_median_nm=args.tile_initial_thickness_log_median_nm,
        tile_initial_thickness_log_sigma=args.tile_initial_thickness_log_sigma,
        ice_reset_day=args.ice_reset_day,
        fit_post_reset_tile_log_offset=args.fit_post_reset_tile_log_offset,
        post_reset_tile_log_offset_prior_sigma=args.post_reset_tile_log_offset_prior_sigma,
        fit_post_reset_ice_index=args.fit_post_reset_ice_index,
        post_reset_ice_index_prior_mean=args.post_reset_ice_index_prior_mean,
        post_reset_ice_index_prior_sigma=args.post_reset_ice_index_prior_sigma,
        post_reset_ice_index_lower=args.post_reset_ice_index_lower,
        post_reset_ice_index_upper=args.post_reset_ice_index_upper,
        fit_pre_post_ice_index=args.fit_pre_post_ice_index,
        ice_index_prior_mean=args.ice_index_prior_mean,
        ice_index_prior_sigma=args.ice_index_prior_sigma,
        ice_index_lower=args.ice_index_lower,
        ice_index_upper=args.ice_index_upper,
        ice_extinction_prior_sigma=args.ice_extinction_prior_sigma,
        ice_extinction_upper=args.ice_extinction_upper,
        fit_complex_q=getattr(args, "fit_complex_q", False),
        include_map_prediction=(
            getattr(args, "map_optimizer", None) == "least-squares"
        ),
    )


def run_map_stage(model, initial_values, args, jax, init_to_value, seed):
    """Run one configured MAP stage and return values plus diagnostics."""
    if not args.map_steps:
        return initial_values, None
    if args.map_optimizer == "lbfgs":
        parameter_scales = (
            map_parameter_scale_values(initial_values, args)
            if args.map_parameter_scaling == "prior"
            else None
        )
        return run_map_optimization(
            model,
            initial_values,
            jax,
            args.map_steps,
            args.map_tolerance,
            args.map_ftol,
            parameter_scales=parameter_scales,
        )
    if args.map_optimizer == "jax-lbfgsb":
        parameter_scales = (
            map_parameter_scale_values(initial_values, args)
            if args.map_parameter_scaling == "prior"
            else None
        )
        return run_jax_map_optimization(
            model,
            initial_values,
            jax,
            args.map_steps,
            args.map_tolerance,
            parameter_scales=parameter_scales,
        )
    if args.map_optimizer == "trust-exact":
        return run_trust_exact_map_optimization(
            model,
            initial_values,
            jax,
            args.map_steps,
            args.map_tolerance,
        )
    return run_map_initialization(
        model,
        initial_values,
        jax,
        init_to_value,
        seed,
        args.map_steps,
        args.map_learning_rate,
    )


def reparameterized_map_value_converters(
    data, args, ar_reference, ar_lower, ar_upper
):
    """Convert legacy constrained MAP sites into opt-in latent coordinates."""
    converters = {}
    if getattr(args, "standardize_ar_mean", False):
        start_reference = map_start_reference(args, ar_reference, ar_lower, ar_upper)
        active_parameter_indices = ar_active_parameter_indices(
            args.ar_layers,
            getattr(args, "fit_ar_absorption", False),
            getattr(args, "fixed_ar_indices", None),
        )
        reference_raw = ar_reference_raw(
            start_reference,
            ar_lower,
            ar_upper,
            active_parameter_indices,
            ordered_ar_indices=getattr(args, "ordered_ar_indices", False),
        )
        ar_mean_prior_sigma = float(args.ar_mean_prior_sigma)
        ar_rotation = ar_coordinate_rotation(
            active_parameter_indices, len(ar_reference)
        )

        def ar_mu_converter(value, _):
            standardized_raw = (
                np.asarray(value, dtype=float) - reference_raw
            ) / ar_mean_prior_sigma
            if getattr(args, "rotate_ar_coordinates", False):
                return standardized_raw @ ar_rotation
            return standardized_raw

        converters["ar_mu_z"] = (
            "ar_mu_raw",
            ar_mu_converter,
        )
    if getattr(args, "whiten_np_trajectory", False):
        trajectory_prior_rate = (
            getattr(args, "trajectory_rate_prior_mean_nm_day", None)
            if getattr(args, "trajectory_rate_prior_mean_nm_day", None) is not None
            else trajectory_rate_per_scale(data["days"]) * TRAJECTORY_SCALE_INIT
        )
        converters["np_log_rate_z"] = [
            (
                "np_log_rate_nodes",
                lambda value, _: whiten_trajectory_rate_nodes(
                    value,
                    data["node_days"],
                    trajectory_prior_rate,
                    args.trajectory_rate_prior_sigma_log,
                    trajectory_rate_likelihood_scale(data, args),
                    args.np_smoothness_weight,
                ),
            ),
            (
                "log_trajectory_scale",
                lambda value, _: whiten_trajectory_rate_nodes(
                    np.full(len(data["node_days"]), float(np.asarray(value))),
                    data["node_days"],
                    trajectory_prior_rate,
                    args.trajectory_rate_prior_sigma_log,
                    trajectory_rate_likelihood_scale(data, args),
                    args.np_smoothness_weight,
                ),
            ),
        ]
    elif getattr(args, "fully_np_trajectory", False):
        converters["np_log_rate_nodes"] = (
            "log_trajectory_scale",
            lambda value, _: np.full(
                len(data["node_days"]), float(np.asarray(value))
            ),
        )
    if getattr(args, "independent_tile_ar_parameters", False):
        converters["ar_tile_raw"] = (
            "ar_mu_raw",
            lambda value, _: np.broadcast_to(
                np.asarray(value, dtype=float).reshape(1, 1, -1),
                (
                    data["n_scas"],
                    data["n_tiles"],
                    np.asarray(value).size,
                ),
            ).copy(),
        )
    if getattr(args, "independent_tile_trajectories", False):
        converters["trajectory_rate_tile_node_log_rates"] = [
            (
                "trajectory_rate_tile_node_log_rates",
                lambda value, _: np.asarray(value, dtype=float),
            ),
            (
                "log_trajectory_scale",
                lambda value, _: np.full(
                    (
                        data["n_scas"],
                        data["n_tiles"],
                        len(data["node_days"]),
                    ),
                    float(np.asarray(value)),
                ),
            ),
        ]
    if getattr(args, "absolute_substrate_power", False) and not getattr(
        args, "shared_absolute_hgcdte_scales", False
    ):
        def tile_hgcdte_converter(value, _):
            value = np.asarray(value, dtype=float)
            target_shape = (data["n_scas"], data["n_tiles"])
            if value.shape == target_shape:
                return value.copy()
            if value.size == data["n_scas"]:
                return np.broadcast_to(value.reshape(data["n_scas"], 1), target_shape).copy()
            raise ValueError("Cannot broadcast HgCdTe warm-start values to tile sites")

        for name in (
            "absolute_hgcdte_n_log_scale",
            "absolute_hgcdte_k_log_scale",
        ):
            converters[name] = (name, tile_hgcdte_converter)
    if getattr(args, "pivot_initial_thickness", False):
        converters["initial_thickness_pivot_z"] = (
            "initial_thickness_nm",
            lambda value, samples: trajectory_pivot_z_from_initial_thickness(
                value,
                data,
                args,
                np.asarray(samples["np_log_rate_nodes"], dtype=float).reshape(-1),
            ),
        )
    return converters


def quantile(values):
    return np.percentile(np.asarray(values), [16, 50, 84])


def posterior_median_fit_diagnostics(
    data,
    context,
    samples,
    fit_hgcdte=False,
    fit_hgcdte_modulation=False,
    fit_conductive_layer=False,
    finite_hgcdte_detector=False,
    detector_backside_index=1.0,
    fit_qe_rbf=False,
    qe_rbf_node_wavelengths_nm=None,
    qe_rbf_width_nm=QE_RBF_WIDTH_NM,
    use_asw_optical_constants=False,
    fit_ice_optical_modulation=False,
    fit_post_reset_tile_log_offset=False,
    use_hgcdte_optical_constants=False,
    incidence_cone_half_angle_deg=0.0,
    incidence_pupil_half_angle_deg=0.0,
    incidence_chief_ray_tilts_deg=None,
    fit_incidence_angle=False,
    early_time_cutoff_day=None,
    fit_late_acm_step=False,
    fit_additional_acm_step=False,
    absolute_substrate_power=None,
    fit_sca_ar_opd=False,
    shared_absolute_hgcdte_scales=False,
    fit_post_reset_ice_index=False,
    post_reset_ice_reset_day=None,
    post_reset_log_offset_day=None,
    fit_pre_post_ice_index=False,
    fit_complex_q=False,
):
    """Evaluate the posterior-median parameters against the tile means."""
    _, jnp, _, _, _, _, _ = require_numpyro()
    if absolute_substrate_power is None:
        absolute_substrate_power = bool(context.get("absolute_substrate_power", False))
    sca_ar_opd_nm = None
    if fit_sca_ar_opd:
        if "sca_ar_opd_nm" not in samples:
            raise ValueError("SCA AR OPD posterior has no fitted values")
        sca_ar_opd_nm = jnp.asarray(
            np.median(np.asarray(samples["sca_ar_opd_nm"]), axis=0)
        )
    ice_thickness = np.median(np.asarray(samples["ice_thickness"]), axis=0)
    ar_parameters = np.median(np.asarray(samples["ar_parameters"]), axis=0)
    if ar_parameters.ndim == 3:
        ar_parameters = ar_parameters.reshape((-1, ar_parameters.shape[-1]))
    elif ar_parameters.ndim != 2:
        raise ValueError("Unexpected median AR parameter shape")
    n_rows = int(data["counts"].shape[0])
    if ar_parameters.shape[0] != n_rows:
        raise ValueError("Posterior AR rows do not match the selected tile data")
    detector_thickness_nm = None
    if finite_hgcdte_detector:
        if "detector_thickness_nm" not in samples:
            raise ValueError("Finite detector posterior has no thickness values")
        detector_thickness_nm = np.median(
            np.asarray(samples["detector_thickness_nm"]), axis=0
        ).reshape(-1)
        if detector_thickness_nm.size != n_rows:
            raise ValueError("Posterior detector thickness rows do not match the data")
    acm_step = np.median(
        np.asarray(samples["acm_step_log_offset"]), axis=0
    )
    hgcdte_parameters = None
    if absolute_substrate_power:
        n_log_scale = np.median(
            np.asarray(samples["absolute_hgcdte_n_log_scale"]), axis=0
        )
        k_log_scale = np.median(
            np.asarray(samples["absolute_hgcdte_k_log_scale"]), axis=0
        )
        if np.asarray(n_log_scale).ndim:
            hgcdte_parameters = np.stack((n_log_scale, k_log_scale), axis=-1)
            if shared_absolute_hgcdte_scales:
                hgcdte_parameters = hgcdte_parameters[
                    np.asarray(data["sca_index"], dtype=int)
                ]
            else:
                hgcdte_parameters = hgcdte_parameters.reshape(-1, 2)
            hgcdte_parameters = jnp.asarray(hgcdte_parameters)
        else:
            hgcdte_parameters = jnp.asarray([n_log_scale, k_log_scale])
    elif fit_hgcdte:
        hgcdte_parameters = jnp.asarray(
            [
                np.median(np.asarray(samples["hgcdte_n"])),
                np.median(np.asarray(samples["hgcdte_k"])),
            ]
        )
    elif fit_hgcdte_modulation:
        hgcdte_parameters = jnp.concatenate(
            (
                jnp.asarray(
                    np.median(
                        np.asarray(samples["hgcdte_n_log_modulation"]), axis=0
                    )
                ),
                jnp.asarray(
                    np.median(
                        np.asarray(samples["hgcdte_k_log_modulation"]), axis=0
                    )
                ),
            )
        )
    ice_modulation_parameters = None
    if fit_ice_optical_modulation:
        ice_modulation_parameters = jnp.concatenate(
            (
                jnp.asarray(
                    np.median(np.asarray(samples["ice_n_log_modulation"]), axis=0)
                ),
                jnp.asarray(
                    np.median(np.asarray(samples["ice_k_log_modulation"]), axis=0)
                ),
            )
        )
    conductive_parameters = None
    if fit_conductive_layer:
        conductive_parameters = jnp.asarray(
            [
                np.median(np.asarray(samples["conductive_log_thickness_nm"])),
                np.median(np.asarray(samples["conductive_log_plasma_strength"])),
                np.median(np.asarray(samples["conductive_log_damping_ratio"])),
            ]
        )
    qe_rbf_parameters = None
    if fit_qe_rbf:
        qe_rbf_parameters = jnp.asarray(
            np.median(np.asarray(samples["qe_rbf_log_modulation"]), axis=0)
        )
    complex_q_parameters = None
    if fit_complex_q:
        complex_q_parameters = jnp.asarray(
            np.median(np.asarray(samples["complex_q_parameters"]), axis=0)
        )
    early_time_scale = None
    if early_time_cutoff_day is not None:
        early_time_scale = jnp.asarray(
            np.median(np.asarray(samples["early_time_scale"]), axis=0)
        )
    late_acm_step = (
        jnp.asarray(np.median(np.asarray(samples["late_acm_step_log_offset"])))
        if fit_late_acm_step
        else jnp.asarray(0.0)
    )
    additional_acm_step = (
        jnp.asarray(
            np.median(np.asarray(samples["additional_acm_step_log_offset"]))
        )
        if fit_additional_acm_step
        else jnp.asarray(0.0)
    )
    incidence_angle_deg = (
        jnp.asarray(np.median(np.asarray(samples["incidence_angle_deg"])))
        if fit_incidence_angle
        else None
    )
    post_reset_ice_index = (
        jnp.asarray(np.median(np.asarray(samples["post_reset_ice_index"])))
        if fit_post_reset_ice_index
        else None
    )
    pre_reset_ice_index = None
    if fit_pre_post_ice_index:
        pre_reset_ice_index = jnp.asarray(
            np.median(np.asarray(samples["pre_reset_ice_n"]))
            + 1j * np.median(np.asarray(samples["pre_reset_ice_k"]))
        )
        post_reset_ice_index = jnp.asarray(
            np.median(np.asarray(samples["post_reset_ice_n"]))
            + 1j * np.median(np.asarray(samples["post_reset_ice_k"]))
        )
    post_reset_tile_log_offsets = None
    if fit_post_reset_tile_log_offset:
        post_reset_tile_log_offsets = np.median(
            np.asarray(samples["post_reset_tile_log_offset"]), axis=0
        ).reshape(-1)
        if post_reset_log_offset_day is None:
            raise ValueError(
                "Post-reset tile log offsets require a post-reset day"
            )
        observed_offset_mask = np.asarray(data["days"], dtype=float) >= float(
            post_reset_log_offset_day
        )
    else:
        observed_offset_mask = None
    prediction_chunks = []
    for start in range(0, n_rows, PREDICTION_ROW_CHUNK_SIZE):
        stop = min(start + PREDICTION_ROW_CHUNK_SIZE, n_rows)
        row_indices = np.arange(start, stop, dtype=int)
        prediction_chunks.append(
            np.asarray(
                build_prediction_function(
                    data,
                    context,
                    row_indices=row_indices,
                    fit_hgcdte=fit_hgcdte,
                    fit_hgcdte_modulation=fit_hgcdte_modulation,
                    fit_conductive_layer=fit_conductive_layer,
                    finite_hgcdte_detector=finite_hgcdte_detector,
                    detector_backside_index=detector_backside_index,
                    fit_qe_rbf=fit_qe_rbf,
                    qe_rbf_node_wavelengths_nm=qe_rbf_node_wavelengths_nm,
                    qe_rbf_width_nm=qe_rbf_width_nm,
                    use_asw_optical_constants=use_asw_optical_constants,
                    fit_ice_optical_modulation=fit_ice_optical_modulation,
                    use_hgcdte_optical_constants=use_hgcdte_optical_constants,
                    incidence_cone_half_angle_deg=incidence_cone_half_angle_deg,
                    incidence_pupil_half_angle_deg=incidence_pupil_half_angle_deg,
                    incidence_chief_ray_tilts_deg=incidence_chief_ray_tilts_deg,
                    fit_incidence_angle=fit_incidence_angle,
                    early_time_cutoff_day=early_time_cutoff_day,
                    fit_late_acm_step=fit_late_acm_step,
                    fit_additional_acm_step=fit_additional_acm_step,
                     absolute_substrate_power=absolute_substrate_power,
                     post_reset_ice_reset_day=post_reset_ice_reset_day,
                     fit_complex_q=fit_complex_q,
                 )(
                    jnp.asarray(
                        ice_thickness
                        if ice_thickness.ndim == 1
                        else ice_thickness[row_indices]
                    ),
                    jnp.asarray(ar_parameters[row_indices]),
                    jnp.asarray(acm_step),
                    hgcdte_parameters,
                    conductive_parameters,
                    qe_rbf_parameters,
                    early_time_scale,
                    late_acm_step,
                    jnp.asarray(detector_thickness_nm[row_indices])
                    if detector_thickness_nm is not None
                    else None,
                    incidence_angle_deg,
                    additional_acm_step,
                     post_reset_ice_index,
                     pre_reset_ice_index,
                     ice_modulation_parameters,
                     sca_ar_opd_nm,
                     complex_q_parameters,
                  )
            )
        )
    prediction = np.concatenate(prediction_chunks, axis=0)
    observed = data["sums"] / data["counts"]
    if post_reset_tile_log_offsets is not None:
        observed = observed + post_reset_tile_log_offsets[:, None] * observed_offset_mask[None, :]
    residual = observed - prediction
    non_reference = np.ones(residual.shape[1], dtype=bool)
    non_reference[data["first_by_led"]] = False
    total_count = float(np.sum(data["counts"]))
    total_sse = float(
        np.sum(
            data["sum_squared"]
            - 2.0 * prediction * data["sums"]
            + data["counts"] * prediction * prediction
        )
    )
    return {
        "group_rms_log": float(np.sqrt(np.mean(residual * residual))),
        "non_reference_group_rms_log": float(
            np.sqrt(np.mean(residual[:, non_reference] ** 2))
        ),
        "count_weighted_group_rms_log": float(
            np.sqrt(np.sum(data["counts"] * residual * residual) / total_count)
        ),
        "pixel_rms_log": float(np.sqrt(total_sse / total_count)),
        "correlation": float(np.corrcoef(observed.ravel(), prediction.ravel())[0, 1]),
        "observed": np.asarray(observed),
        "prediction": np.asarray(prediction),
        "residual": np.asarray(residual),
        "led_rms_log": np.asarray(
            [
                np.sqrt(np.mean(residual[:, data["led"] == led] ** 2))
                for led in range(1, 7)
            ]
        ),
    }


def ar_quantile_fields(ar_layers):
    fields = []
    for layer in range(ar_layers):
        prefix = "ar_" if ar_layers == 1 else f"ar_layer{layer + 1}_"
        for component in ("thickness", "n", "k"):
            fields.extend(
                f"{prefix}{component}_{percentile}"
                for percentile in ("p16", "median", "p84")
            )
    return fields


def add_tile_ar_quantiles(row, quantiles, sca_index, tile_index, ar_layers):
    for layer in range(ar_layers):
        prefix = "ar_" if ar_layers == 1 else f"ar_layer{layer + 1}_"
        for component_index, component in enumerate(("thickness", "n", "k")):
            for percentile_index, percentile in enumerate(("p16", "median", "p84")):
                row[f"{prefix}{component}_{percentile}"] = quantiles[
                    percentile_index, sca_index, tile_index, layer, component_index
                ]


def add_sca_ar_quantiles(row, quantiles, sca_index, ar_layers):
    for layer in range(ar_layers):
        prefix = "ar_" if ar_layers == 1 else f"ar_layer{layer + 1}_"
        for component_index, component in enumerate(("thickness", "n", "k")):
            for percentile_index, percentile in enumerate(("p16", "median", "p84")):
                row[f"{prefix}{component}_{percentile}"] = quantiles[
                    percentile_index, sca_index, layer, component_index
                ]


def format_command_line(argv=None, executable=None):
    """Return a shell-quoted command that reproduces the current invocation."""
    if argv is None:
        argv = sys.argv
    if executable is None:
        executable = sys.executable
    return shlex.join([executable, *argv])


def write_tile_initial_thickness_posterior(
    output_dir, data, samples, sample_name, output_name, field_prefix
):
    """Write per-tile initial-thickness quantiles for a sampled baseline."""
    thickness_samples = np.asarray(samples[sample_name], dtype=float)
    if thickness_samples.ndim == 1:
        thickness_samples = np.broadcast_to(
            thickness_samples[:, None, None],
            (thickness_samples.shape[0], data["n_scas"], data["n_tiles"]),
        )
    elif thickness_samples.ndim == 2 and data["n_scas"] == 1:
        thickness_samples = thickness_samples[:, None, :]
    if thickness_samples.ndim != 3 or thickness_samples.shape[1:] != (
        data["n_scas"],
        data["n_tiles"],
    ):
        raise ValueError("Per-tile initial-thickness posterior has an unexpected shape")
    thickness_quantiles = np.percentile(thickness_samples, [16, 50, 84], axis=0)
    with (output_dir / output_name).open("w", newline="") as handle:
        fields = [
            "detector",
            "tile_id",
            "tile_row",
            "tile_col",
            f"{field_prefix}_p16",
            f"{field_prefix}_median",
            f"{field_prefix}_p84",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sca_index, detector in enumerate(data["sca_names"]):
            for tile_index, tile_id in enumerate(data["tile_ids"]):
                tile_metadata = data["metadata"][int(tile_id)]
                writer.writerow(
                    {
                        "detector": detector,
                        "tile_id": int(tile_id),
                        "tile_row": tile_metadata["tile_row"],
                        "tile_col": tile_metadata["tile_col"],
                        f"{field_prefix}_p16": thickness_quantiles[
                            0, sca_index, tile_index
                        ],
                        f"{field_prefix}_median": thickness_quantiles[
                            1, sca_index, tile_index
                        ],
                        f"{field_prefix}_p84": thickness_quantiles[
                            2, sca_index, tile_index
                        ],
                    }
                )


def write_tile_log_offset_posterior(
    output_dir, data, samples, sample_name, output_name, field_prefix
):
    """Write per-tile log-offset quantiles for a sampled correction."""
    offset_samples = np.asarray(samples[sample_name], dtype=float)
    if offset_samples.ndim == 1:
        offset_samples = offset_samples[:, None, None]
    elif offset_samples.ndim == 2 and data["n_scas"] == 1:
        offset_samples = offset_samples[:, None, :]
    if offset_samples.ndim != 3 or offset_samples.shape[1:] != (
        data["n_scas"],
        data["n_tiles"],
    ):
        raise ValueError("Per-tile log-offset posterior has an unexpected shape")
    offset_quantiles = np.percentile(offset_samples, [16, 50, 84], axis=0)
    with (output_dir / output_name).open("w", newline="") as handle:
        fields = [
            "detector",
            "tile_id",
            "tile_row",
            "tile_col",
            f"{field_prefix}_p16",
            f"{field_prefix}_median",
            f"{field_prefix}_p84",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sca_index, detector in enumerate(data["sca_names"]):
            for tile_index, tile_id in enumerate(data["tile_ids"]):
                tile_metadata = data["metadata"][int(tile_id)]
                writer.writerow(
                    {
                        "detector": detector,
                        "tile_id": int(tile_id),
                        "tile_row": tile_metadata["tile_row"],
                        "tile_col": tile_metadata["tile_col"],
                        f"{field_prefix}_p16": offset_quantiles[
                            0, sca_index, tile_index
                        ],
                        f"{field_prefix}_median": offset_quantiles[
                            1, sca_index, tile_index
                        ],
                        f"{field_prefix}_p84": offset_quantiles[
                            2, sca_index, tile_index
                        ],
                    }
                )


def write_outputs(
    output_dir,
    data,
    samples,
    args,
    fit_diagnostics=None,
    sampler_diagnostics=None,
    map_diagnostics=None,
    warmup_map_diagnostics=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ar_layers = int(args.ar_layers)
    qe_rbf_nodes, qe_rbf_width_nm, qe_rbf_prior_sigma_log = qe_rbf_configuration(
        getattr(args, "qe_rbf_node_wavelengths_nm", None),
        getattr(args, "qe_rbf_width_nm", QE_RBF_WIDTH_NM),
        getattr(args, "qe_rbf_prior_sigma_log", QE_RBF_PRIOR_SIGMA_LOG),
    )
    command_line = format_command_line()
    metadata = {
        "tile_ids": data["tile_ids"],
        "sca_names": data["sca_names"],
        "qe_columns": data["qe_columns"],
        "ar_layers": ar_layers,
        "ordered_ar_indices": int(getattr(args, "ordered_ar_indices", False)),
        "fixed_ar_indices": (
            np.empty(0, dtype=float)
            if getattr(args, "fixed_ar_indices", None) is None
            else np.asarray(args.fixed_ar_indices, dtype=float)
        ),
        "wavelength_step_nm": (
            np.nan if args.wavelength_step_nm is None else args.wavelength_step_nm
        ),
        "wavelength_stride": args.wavelength_stride,
        "response_wavelengths_nm": np.asarray(
            data.get("response_wavelengths_nm", np.empty(0)), dtype=float
        ),
        "response_wavelength_count": int(
            data.get("response_wavelength_count", 0)
        ),
        "prediction_row_chunk_size": int(
            data.get("prediction_row_chunk_size", PREDICTION_ROW_CHUNK_SIZE)
        ),
        "likelihood_weight": args.likelihood_weight,
        "log_noise_sigma": args.log_noise_sigma,
        "statistics_path": np.asarray(str(args.tile_statistics)),
        "fixed_corrections_path": np.asarray(
            data.get(
                "fixed_corrections_path",
                "" if getattr(args, "fixed_corrections", None) is None
                else str(args.fixed_corrections),
            )
        ),
        "fixed_corrections_applied": int(data.get("fixed_corrections_applied", False)),
        "fixed_corrections_tile_aware": int(
            data.get("fixed_corrections_tile_aware", False)
        ),
        "input_corrections_already_applied": int(
            getattr(args, "corrections_already_applied", False)
        ),
        "fixed_acm_step_log_offset": float(
            data.get("fixed_acm_step_log_offset", 0.0)
        ),
        "fixed_late_acm_step_log_offset": float(
            data.get("fixed_late_acm_step_log_offset", 0.0)
        ),
        "fixed_additional_acm_step_log_offset": float(
            data.get("fixed_additional_acm_step_log_offset", 0.0)
        ),
        "fixed_pre_reset_acm_step_log_offset": float(
            data.get("fixed_pre_reset_acm_step_log_offset", 0.0)
        ),
        "fixed_acm_step_log_offset_by_row": np.asarray(
            data.get("fixed_acm_step_log_offset_by_row", np.empty(0)),
            dtype=float,
        ),
        "fixed_late_acm_step_log_offset_by_row": np.asarray(
            data.get("fixed_late_acm_step_log_offset_by_row", np.empty(0)),
            dtype=float,
        ),
        "fixed_additional_acm_step_log_offset_by_row": np.asarray(
            data.get("fixed_additional_acm_step_log_offset_by_row", np.empty(0)),
            dtype=float,
        ),
        "fixed_pre_reset_acm_step_log_offset_by_row": np.asarray(
            data.get("fixed_pre_reset_acm_step_log_offset_by_row", np.empty(0)),
            dtype=float,
        ),
        "min_observation_day": (
            np.nan
            if getattr(args, "min_observation_day", None) is None
            else args.min_observation_day
        ),
        "early_time_scale_cutoff_day": (
            np.nan
            if getattr(args, "early_time_scale_cutoff_day", None) is None
            else args.early_time_scale_cutoff_day
        ),
        "early_time_scale_prior_sigma": args.early_time_scale_prior_sigma,
        "early_time_scale_definition": np.asarray(
            "relative observed-count multiplier before per-LED normalization"
        ),
        "early_time_scale_repeat": (
            np.asarray(data["early_time_scale_repeat"], dtype=int)
            if "early_time_scale_repeat" in data
            else np.empty(0, dtype=int)
        ),
        "early_time_scale_days": (
            np.asarray(data["early_time_scale_days"], dtype=float)
            if "early_time_scale_days" in data
            else np.empty(0, dtype=float)
        ),
        "observation_early_time_scale_index": (
            np.asarray(data["early_time_scale_index"], dtype=int)
            if "early_time_scale_index" in data
            else np.full(len(data["days"]), -1, dtype=int)
        ),
        "command_line": np.asarray(command_line),
        "statistics_fingerprint": np.asarray(statistics_fingerprint(data)),
        "statistics_row_fingerprints": statistics_row_fingerprints(data),
        "sed_mode": np.asarray(args.sed_mode),
        "sed_weighting": np.asarray(args.sed_weighting),
        "max_tiles": -1 if args.max_tiles is None else args.max_tiles,
        "shared_ar_optical_properties": int(args.shared_ar_optical_properties),
        "shared_ar_parameters_across_tiles": int(
            getattr(args, "shared_ar_parameters_across_tiles", False)
        ),
        "independent_tile_ar_parameters": int(
            getattr(args, "independent_tile_ar_parameters", False)
        ),
        "fit_ar_absorption": int(args.fit_ar_absorption),
        "fit_sca_ar_opd": int(getattr(args, "fit_sca_ar_opd", False)),
        "sca_ar_opd_prior_sigma_nm": float(
            getattr(args, "sca_ar_opd_prior_sigma_nm", SCA_AR_OPD_PRIOR_SIGMA_NM)
        ),
        "sca_ar_opd_definition": np.asarray(
            "effective normal-incidence optical path added to the first AR-layer phase"
        ),
        "fit_hgcdte": int(getattr(args, "fit_hgcdte", False)),
        "fit_hgcdte_modulation": int(
            getattr(args, "fit_hgcdte_modulation", False)
        ),
        "absolute_substrate_power": int(
            getattr(args, "absolute_substrate_power", False)
        ),
        "absolute_hgcdte_tile_variation": int(
            getattr(args, "absolute_substrate_power", False)
            and not getattr(args, "shared_absolute_hgcdte_scales", False)
        ),
        "shared_absolute_hgcdte_scales": int(
            getattr(args, "shared_absolute_hgcdte_scales", False)
        ),
        "fixed_absolute_hgcdte_scales": int(
            getattr(args, "fixed_absolute_hgcdte_scales", False)
        ),
        "absolute_hgcdte_scale_prior_sigma_log": float(
            getattr(
                args,
                "absolute_hgcdte_scale_prior_sigma_log",
                ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG,
            )
        ),
        "hgcdte_modulation_prior_sigma_log": getattr(
            args,
            "hgcdte_modulation_prior_sigma_log",
            HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
        ),
        "use_hgcdte_optical_constants": int(
            getattr(args, "use_hgcdte_optical_constants", False)
        ),
        "fit_conductive_layer": int(getattr(args, "fit_conductive_layer", False)),
        "finite_hgcdte_detector": int(
            getattr(args, "finite_hgcdte_detector", False)
            or getattr(args, "fit_detector_thickness", False)
        ),
        "detector_thickness_reference_nm": float(
            getattr(args, "detector_thickness_nm", DETECTOR_THICKNESS_DEFAULT_NM)
        ),
        "fit_detector_thickness": int(
            getattr(args, "fit_detector_thickness", False)
        ),
        "detector_thickness_prior_sigma_log": float(
            getattr(
                args,
                "detector_thickness_prior_sigma_log",
                DETECTOR_THICKNESS_PRIOR_SIGMA_LOG,
            )
        ),
        "detector_thickness_neighbor_sigma_log": float(
            getattr(
                args,
                "detector_thickness_neighbor_sigma_log",
                DETECTOR_THICKNESS_NEIGHBOR_SIGMA_LOG,
            )
        ),
        "detector_backside_index": float(
            getattr(args, "detector_backside_index", 1.0)
        ),
        "fit_qe_rbf": int(getattr(args, "fit_qe_rbf", False)),
        "fit_complex_q": int(getattr(args, "fit_complex_q", False)),
        "complex_q_parameter_names": np.asarray(COMPLEX_Q_PARAMETER_NAMES),
        "complex_q_lower": COMPLEX_Q_LOWER,
        "complex_q_upper": COMPLEX_Q_UPPER,
        "use_asw_optical_constants": int(
            getattr(args, "use_asw_optical_constants", False)
        ),
        "fit_ice_optical_modulation": int(
            getattr(args, "fit_ice_optical_modulation", False)
        ),
        "ice_modulation_node_wavelengths_nm": ICE_MODULATION_NODE_WAVELENGTHS_NM,
        "ice_modulation_width_nm": float(ICE_MODULATION_WIDTH_NM),
        "ice_modulation_prior_sigma_log": float(
            getattr(
                args,
                "ice_modulation_prior_sigma_log",
                ICE_MODULATION_PRIOR_SIGMA_LOG,
            )
        ),
        "ice_n_modulation_prior_sigma_log": float(
            getattr(
                args,
                "ice_n_modulation_prior_sigma_log",
                None,
            )
            if getattr(args, "ice_n_modulation_prior_sigma_log", None) is not None
            else getattr(
                args,
                "ice_modulation_prior_sigma_log",
                ICE_MODULATION_PRIOR_SIGMA_LOG,
            )
        ),
        "ice_optical_constants_source": np.asarray(
            str(
                getattr(
                    args,
                    "ice_optical_constants_source",
                    ASW_OPTICAL_CONSTANTS_URL,
                )
            )
        ),
        "incidence_cone_half_angle_deg": float(
            getattr(args, "incidence_cone_half_angle_deg", 0.0)
        ),
        "incidence_pupil_half_angle_deg": float(
            getattr(args, "incidence_pupil_half_angle_deg", 0.0)
        ),
        "incidence_chief_ray_tilts_deg": load_incidence_chief_ray_tilts(
            getattr(args, "incidence_chief_ray_map", None),
            data,
            (
                getattr(args, "incidence_chief_ray_x_deg", 0.0),
                getattr(args, "incidence_chief_ray_y_deg", 0.0),
            ),
        ),
        "incidence_chief_ray_map": np.asarray(
            ""
            if getattr(args, "incidence_chief_ray_map", None) is None
            else str(args.incidence_chief_ray_map)
        ),
        "fit_incidence_angle": int(getattr(args, "fit_incidence_angle", False)),
        "incidence_angle_max_deg": INCIDENCE_ANGLE_MAX_DEG,
        "incidence_angle_prior_sigma_deg": float(
            getattr(
                args,
                "incidence_angle_prior_sigma_deg",
                INCIDENCE_ANGLE_PRIOR_SIGMA_DEG,
            )
        ),
        "incidence_angle_prior_uniform": int(
            getattr(args, "incidence_angle_prior_uniform", False)
        ),
        "incidence_angle_init_deg": float(
            getattr(args, "incidence_angle_init_deg", INCIDENCE_ANGLE_INIT_DEG)
        ),
        "led_dependent_acm_step": int(
            getattr(args, "led_dependent_acm_step", False)
        ),
        "acm_step_prior_sigma": getattr(args, "acm_step_prior_sigma", 0.05),
        "fit_late_acm_step": int(getattr(args, "fit_late_acm_step", False)),
        "late_acm_step_prior_sigma": getattr(
            args, "late_acm_step_prior_sigma", 0.02
        ),
        "fit_additional_acm_step": int(
            getattr(args, "fit_additional_acm_step", False)
        ),
        "additional_acm_step_repeat": int(
            data.get("additional_acm_step_repeat", -1)
        ),
        "pre_reset_acm_step_repeat": int(
            data.get("pre_reset_acm_step_repeat", -1)
        ),
        "additional_acm_step_prior_sigma": getattr(
            args, "additional_acm_step_prior_sigma", 0.02
        ),
        "initial_thickness_prior_mean_nm": float(
            getattr(
                args,
                "initial_thickness_prior_mean_nm",
                INITIAL_THICKNESS_PRIOR_MEAN_NM,
            )
        ),
        "initial_thickness_prior_sigma_nm": float(
            getattr(
                args,
                "initial_thickness_prior_sigma_nm",
                INITIAL_THICKNESS_PRIOR_SIGMA_NM,
            )
        ),
        "initial_thickness_prior_uniform": int(
            getattr(args, "initial_thickness_prior_uniform", False)
        ),
        "tile_initial_thickness_lognormal": int(
            getattr(args, "tile_initial_thickness_lognormal", False)
        ),
        "tile_initial_thickness_log_median_nm": float(
            getattr(args, "tile_initial_thickness_log_median_nm", 10.0)
        ),
        "tile_initial_thickness_log_sigma": float(
            getattr(args, "tile_initial_thickness_log_sigma", 1.5)
        ),
        "ice_reset_day": (
            np.nan
            if getattr(args, "ice_reset_day", None) is None
            else float(args.ice_reset_day)
        ),
        "fit_post_reset_tile_log_offset": int(
            getattr(args, "fit_post_reset_tile_log_offset", False)
        ),
        "post_reset_tile_log_offset_prior_sigma": float(
            getattr(
                args,
                "post_reset_tile_log_offset_prior_sigma",
                POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA,
            )
        ),
        "fit_post_reset_ice_index": int(
            getattr(args, "fit_post_reset_ice_index", False)
        ),
        "fit_pre_post_ice_index": int(
            getattr(args, "fit_pre_post_ice_index", False)
        ),
        "ice_index_prior_mean": float(
            getattr(args, "ice_index_prior_mean", ICE_INDEX_PRIOR_MEAN)
        ),
        "ice_index_prior_sigma": float(
            getattr(args, "ice_index_prior_sigma", ICE_INDEX_PRIOR_SIGMA)
        ),
        "ice_extinction_prior_sigma": float(
            getattr(
                args,
                "ice_extinction_prior_sigma",
                ICE_EXTINCTION_PRIOR_SIGMA,
            )
        ),
        "ice_extinction_upper": float(
            getattr(args, "ice_extinction_upper", ICE_EXTINCTION_UPPER)
        ),
        "initial_thickness_init_nm": float(
            getattr(args, "initial_thickness_init_nm", INITIAL_THICKNESS_INIT_NM)
        ),
        "nuts_initial_thickness_nm": float(
            getattr(
                args,
                "nuts_initial_thickness_nm",
                NUTS_INITIAL_THICKNESS_INIT_NM,
            )
        ),
        "nuts_init_jitter": float(
            getattr(args, "nuts_init_jitter", NUTS_INIT_JITTER)
        ),
        "np_modulation": int(getattr(args, "np_modulation", False)),
        "constant_rate_modulation": int(
            getattr(args, "constant_rate_modulation", False)
        ),
        "constant_rate_trajectory": int(
            getattr(args, "constant_rate_trajectory", False)
        ),
        "constant_rate_smoothness_weight": float(
            getattr(args, "constant_rate_smoothness_weight", 0.0)
        ),
        "fully_np_trajectory": int(
            getattr(args, "fully_np_trajectory", False)
        ),
        "standardize_ar_mean": int(getattr(args, "standardize_ar_mean", False)),
        "rotate_ar_coordinates": int(getattr(args, "rotate_ar_coordinates", False)),
        "whiten_np_trajectory": int(getattr(args, "whiten_np_trajectory", False)),
        "rotate_np_likelihood": int(getattr(args, "rotate_np_likelihood", False)),
        "pivot_initial_thickness": int(
            getattr(args, "pivot_initial_thickness", False)
        ),
        "qe_rbf_node_wavelengths_nm": qe_rbf_nodes,
        "qe_rbf_width_nm": qe_rbf_width_nm,
        "qe_rbf_prior_sigma_log": qe_rbf_prior_sigma_log,
        "trajectory_rate_prior_mean_nm_day": (
            trajectory_rate_per_scale(data["days"]) * TRAJECTORY_SCALE_INIT
            if args.trajectory_rate_prior_mean_nm_day is None
            else args.trajectory_rate_prior_mean_nm_day
        ),
        "trajectory_rate_prior_sigma_log": args.trajectory_rate_prior_sigma_log,
        "fixed_trajectory_rate_nm_day": (
            np.nan
            if getattr(args, "fixed_trajectory_rate_nm_day", None) is None
            else args.fixed_trajectory_rate_nm_day
        ),
        "trajectory_rate_definition": np.asarray(
            (
                "fully non-parametric instantaneous rate at each observation-epoch node"
                if getattr(args, "fully_np_trajectory", False)
                else "free constant instantaneous rate"
                if getattr(args, "constant_rate_trajectory", False)
                else "constant instantaneous rate with observation-epoch modulation"
                if getattr(args, "constant_rate_modulation", False)
                else "unmodulated instantaneous rate at the first observed time"
            )
        ),
        "trajectory_rate_reference_day": float(np.min(data["days"])),
        "tile_rate_variation": int(getattr(args, "tile_rate_variation", False)),
        "tile_rate_trajectory_variation": int(
            getattr(args, "tile_rate_trajectory_variation", False)
        ),
        "independent_tile_trajectories": int(
            getattr(args, "independent_tile_trajectories", False)
        ),
        "tile_rate_trajectory_smoothness_weight": float(
        getattr(args, "tile_rate_trajectory_smoothness_weight", 0.001)
        ),
        "tile_rate_prior_sigma_log": getattr(
            args, "tile_rate_prior_sigma_log", np.nan
        ),
        "tile_rate_neighbor_sigma_log": getattr(
            args, "tile_rate_neighbor_sigma_log", np.nan
        ),
        "tile_rate_neighbor_smoothness": int(
            getattr(args, "tile_rate_neighbor_smoothness", True)
        ),
        "tile_rate_neighbor_count": len(tile_neighbor_edges(data)),
        "map_trajectory_rate_nm_day": (
            np.nan
            if args.map_trajectory_rate_nm_day is None
            else args.map_trajectory_rate_nm_day
        ),
        "map_only": int(getattr(args, "map_only", False)),
        "map_parameter_scaling": np.asarray(
            getattr(args, "map_parameter_scaling", "none")
        ),
        "map_init_from": np.asarray(
            "" if getattr(args, "map_init_from", None) is None
            else str(getattr(args, "map_init_from"))
        ),
        "map_init_allow_tile_subset": int(
            getattr(args, "map_init_allow_tile_subset", False)
        ),
        "map_init_broadcast_tile_values": int(
            getattr(args, "map_init_broadcast_tile_values", False)
        ),
        "observation_days": data["days"],
        "observation_led": data["led"],
        "observation_repeat": data["repeat"],
        "observation_exposure_times": np.asarray(data["exposure_times"]),
        "observation_step_design": data["step_design"],
        "observation_late_step_design": np.asarray(
            data.get("late_step_design", np.zeros(len(data["days"])))
        ),
        "observation_additional_step_design": np.asarray(
            data.get("additional_step_design", np.zeros(len(data["days"])))
        ),
        "observation_pre_reset_step_design": np.asarray(
            data.get("pre_reset_step_design", np.zeros(len(data["days"])))
        ),
    }
    if map_diagnostics:
        metadata.update(
            {
                "map_optimizer": np.asarray(map_diagnostics["optimizer"]),
                "map_converged": int(map_diagnostics["converged"]),
                "map_status": int(map_diagnostics["status"]),
                "map_message": np.asarray(map_diagnostics["message"]),
                "map_iterations": int(map_diagnostics["iterations"]),
                "map_function_evaluations": int(
                    map_diagnostics["function_evaluations"]
                ),
                "map_gradient_converged": int(
                    map_diagnostics.get("gradient_converged", False)
                ),
                "map_gradient_norm": float(map_diagnostics["gradient_norm"]),
                "map_projected_gradient_norm": float(
                    map_diagnostics.get(
                        "projected_gradient_norm", map_diagnostics["gradient_norm"]
                    )
                ),
                "map_scaled_gradient_norm": float(
                    map_diagnostics.get("scaled_gradient_norm", np.nan)
                ),
                "map_scaled_projected_gradient_norm": float(
                    map_diagnostics.get("scaled_projected_gradient_norm", np.nan)
                ),
                "map_parameter_scale_min": float(
                    map_diagnostics.get("parameter_scale_min", np.nan)
                ),
                "map_parameter_scale_max": float(
                    map_diagnostics.get("parameter_scale_max", np.nan)
                ),
                "map_function_tolerance": float(
                    map_diagnostics.get("function_tolerance", np.nan)
                ),
                "map_objective": float(map_diagnostics["objective"]),
                "map_initial_objective": float(
                    map_diagnostics["initial_objective"]
                ),
                "map_parameter_step_norm": float(
                    map_diagnostics["parameter_step_norm"]
                ),
                "map_objective_history": np.asarray(
                    map_diagnostics["objective_history"], dtype=float
                ),
            }
        )
    if warmup_map_diagnostics:
        metadata.update(
            {
                "warmup_map_optimizer": np.asarray(
                    warmup_map_diagnostics["optimizer"]
                ),
                "warmup_map_converged": int(
                    warmup_map_diagnostics["converged"]
                ),
                "warmup_map_iterations": int(
                    warmup_map_diagnostics["iterations"]
                ),
                "warmup_map_objective": float(
                    warmup_map_diagnostics["objective"]
                ),
                "warmup_map_initial_objective": float(
                    warmup_map_diagnostics["initial_objective"]
                ),
                "warmup_map_gradient_norm": float(
                    warmup_map_diagnostics["gradient_norm"]
                ),
                "warmup_map_projected_gradient_norm": float(
                    warmup_map_diagnostics.get(
                        "projected_gradient_norm",
                        warmup_map_diagnostics["gradient_norm"],
                    )
                ),
                "warmup_map_objective_history": np.asarray(
                    warmup_map_diagnostics["objective_history"], dtype=float
                ),
            }
        )
    if sampler_diagnostics:
        metadata.update(
            {
                f"sampler_{key}": value
                for key, value in sampler_diagnostics.items()
            }
        )
    np.savez_compressed(
        output_dir / "posterior_samples.npz",
        **{key: np.asarray(value) for key, value in samples.items()},
        **metadata,
    )
    if fit_diagnostics is not None and "prediction" in fit_diagnostics:
        with (output_dir / "tile_group_predictions.csv").open(
            "w", newline=""
        ) as handle:
            fields = [
                "detector",
                "tile_id",
                "group",
                "days",
                "led",
                "repeat",
                "observed_log_change",
                "model_log_change",
                "residual_log",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            observed = np.asarray(fit_diagnostics["observed"], dtype=float)
            prediction = np.asarray(fit_diagnostics["prediction"], dtype=float)
            residual = np.asarray(fit_diagnostics["residual"], dtype=float)
            for row_index in range(observed.shape[0]):
                sca_index, tile_index = divmod(row_index, data["n_tiles"])
                for group_index in range(observed.shape[1]):
                    writer.writerow(
                        {
                            "detector": str(data["sca_names"][sca_index]),
                            "tile_id": int(data["tile_ids"][tile_index]),
                            "group": int(data["groups"][group_index]),
                            "days": float(data["days"][group_index]),
                            "led": int(data["led"][group_index]),
                            "repeat": int(data["repeat"][group_index]),
                            "observed_log_change": observed[row_index, group_index],
                            "model_log_change": prediction[row_index, group_index],
                            "residual_log": residual[row_index, group_index],
                        }
                    )
    if getattr(args, "fit_complex_q", False) and "complex_q_parameters" in samples:
        q_quantiles = np.percentile(
            np.asarray(samples["complex_q_parameters"], dtype=float),
            [16, 50, 84],
            axis=0,
        )
        with (output_dir / "complex_q_posterior.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "parameter",
                    "p16",
                    "median",
                    "p84",
                ],
            )
            writer.writeheader()
            for index, name in enumerate(COMPLEX_Q_PARAMETER_NAMES):
                writer.writerow(
                    {
                        "parameter": name,
                        "p16": q_quantiles[0, index],
                        "median": q_quantiles[1, index],
                        "p84": q_quantiles[2, index],
                    }
                )
    ar_samples = np.asarray(samples["ar_parameters"])
    if ar_samples.ndim == 3:
        ar_grid_samples = ar_samples[:, None, :, :]
    elif ar_samples.ndim == 4:
        ar_grid_samples = ar_samples
    else:
        raise ValueError("Unexpected AR posterior shape")
    if ar_grid_samples.shape[-1] != 3 * ar_layers:
        raise ValueError("AR posterior shape does not match the selected layer count")
    ar_grid_samples = ar_grid_samples.reshape(
        ar_grid_samples.shape[:-1] + (ar_layers, 3)
    )
    ar_quantiles = np.percentile(ar_grid_samples, [16, 50, 84], axis=0)
    with (output_dir / "tile_ar_posterior.csv").open("w", newline="") as handle:
        fields = [
            "detector",
            "tile_id",
            "tile_row",
            "tile_col",
        ] + ar_quantile_fields(ar_layers)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sca_index, detector in enumerate(data["sca_names"]):
            for tile_index, tile_id in enumerate(data["tile_ids"]):
                metadata = data["metadata"][int(tile_id)]
                row = {
                    "detector": detector,
                    "tile_id": int(tile_id),
                    "tile_row": metadata["tile_row"],
                    "tile_col": metadata["tile_col"],
                }
                add_tile_ar_quantiles(
                    row, ar_quantiles, sca_index, tile_index, ar_layers
                )
                writer.writerow(row)

    if getattr(args, "absolute_substrate_power", False):
        n_log_scale_samples = np.asarray(
            samples["absolute_hgcdte_n_log_scale"], dtype=float
        )
        k_log_scale_samples = np.asarray(
            samples["absolute_hgcdte_k_log_scale"], dtype=float
        )
        if n_log_scale_samples.ndim == 1:
            n_log_scale_samples = n_log_scale_samples[:, None, None]
            k_log_scale_samples = k_log_scale_samples[:, None, None]
        elif getattr(args, "shared_absolute_hgcdte_scales", False):
            if n_log_scale_samples.ndim != 2 or k_log_scale_samples.shape != n_log_scale_samples.shape:
                raise ValueError("Shared HgCdTe posterior has an unexpected shape")
            n_log_scale_samples = n_log_scale_samples[:, :, None]
            k_log_scale_samples = k_log_scale_samples[:, :, None]
            n_log_scale_samples = np.broadcast_to(
                n_log_scale_samples,
                (n_log_scale_samples.shape[0], data["n_scas"], data["n_tiles"]),
            )
            k_log_scale_samples = np.broadcast_to(
                k_log_scale_samples,
                (k_log_scale_samples.shape[0], data["n_scas"], data["n_tiles"]),
            )
        n_log_scale_quantiles = np.percentile(
            n_log_scale_samples, [16, 50, 84], axis=0
        )
        k_log_scale_quantiles = np.percentile(
            k_log_scale_samples, [16, 50, 84], axis=0
        )
        with (output_dir / "tile_hgcdte_posterior.csv").open(
            "w", newline=""
        ) as handle:
            fields = [
                "detector",
                "tile_id",
                "tile_row",
                "tile_col",
                "n_log_scale_p16",
                "n_log_scale_median",
                "n_log_scale_p84",
                "k_log_scale_p16",
                "k_log_scale_median",
                "k_log_scale_p84",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for sca_index, detector in enumerate(data["sca_names"]):
                for tile_index, tile_id in enumerate(data["tile_ids"]):
                    tile_metadata = data["metadata"][int(tile_id)]
                    writer.writerow(
                        {
                            "detector": detector,
                            "tile_id": int(tile_id),
                            "tile_row": tile_metadata["tile_row"],
                            "tile_col": tile_metadata["tile_col"],
                            "n_log_scale_p16": n_log_scale_quantiles[
                                0, sca_index, tile_index
                            ],
                            "n_log_scale_median": n_log_scale_quantiles[
                                1, sca_index, tile_index
                            ],
                            "n_log_scale_p84": n_log_scale_quantiles[
                                2, sca_index, tile_index
                            ],
                            "k_log_scale_p16": k_log_scale_quantiles[
                                0, sca_index, tile_index
                            ],
                            "k_log_scale_median": k_log_scale_quantiles[
                                1, sca_index, tile_index
                            ],
                            "k_log_scale_p84": k_log_scale_quantiles[
                                2, sca_index, tile_index
                            ],
                        }
                    )

    if (
        getattr(args, "tile_initial_thickness_lognormal", False)
        and "initial_thickness_nm" in samples
    ):
        write_tile_initial_thickness_posterior(
            output_dir,
            data,
            samples,
            "initial_thickness_nm",
            "tile_initial_thickness_posterior.csv",
            "initial_thickness_nm",
        )
        if (
            getattr(args, "ice_reset_day", None) is not None
            and "reset_initial_thickness_nm" in samples
        ):
            write_tile_initial_thickness_posterior(
                output_dir,
                data,
                samples,
                "reset_initial_thickness_nm",
                "tile_reset_initial_thickness_posterior.csv",
                "reset_initial_thickness_nm",
            )

    if (
        getattr(args, "fit_post_reset_tile_log_offset", False)
        and "post_reset_tile_log_offset" in samples
    ):
        write_tile_log_offset_posterior(
            output_dir,
            data,
            samples,
            "post_reset_tile_log_offset",
            "post_reset_tile_log_offset_posterior.csv",
            "post_reset_log_offset",
        )

    if "tile_initial_rate_nm_day" in samples:
        tile_initial = np.percentile(
            np.asarray(samples["tile_initial_rate_nm_day"]), [16, 50, 84], axis=0
        )
        tile_final = np.percentile(
            np.asarray(samples["tile_final_rate_nm_day"]), [16, 50, 84], axis=0
        )
        tile_mean = np.percentile(
            np.asarray(samples["tile_mean_rate_nm_day"]), [16, 50, 84], axis=0
        )
        if "centered_trajectory_rate_tile_offsets" in samples:
            tile_offset_samples = samples["centered_trajectory_rate_tile_offsets"]
            tile_offsets = np.percentile(
                np.asarray(tile_offset_samples), [16, 50, 84], axis=0
            )
        elif "trajectory_rate_tile_offsets" in samples:
            tile_offsets = np.percentile(
                np.asarray(samples["trajectory_rate_tile_offsets"]),
                [16, 50, 84],
                axis=0,
            )
        else:
            # The single-tile model omits an unidentifiable rate offset.
            tile_offsets = np.zeros(
                (3, len(data["sca_names"]), len(data["tile_ids"])),
                dtype=float,
            )
        with (output_dir / "tile_rate_posterior.csv").open("w", newline="") as handle:
            fields = [
                "detector",
                "tile_id",
                "tile_row",
                "tile_col",
                "log_rate_offset_p16",
                "log_rate_offset_median",
                "log_rate_offset_p84",
                "initial_rate_nm_day_p16",
                "initial_rate_nm_day_median",
                "initial_rate_nm_day_p84",
                "final_rate_nm_day_p16",
                "final_rate_nm_day_median",
                "final_rate_nm_day_p84",
                "mean_rate_nm_day_p16",
                "mean_rate_nm_day_median",
                "mean_rate_nm_day_p84",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for sca_index, detector in enumerate(data["sca_names"]):
                for tile_index, tile_id in enumerate(data["tile_ids"]):
                    metadata = data["metadata"][int(tile_id)]
                    writer.writerow(
                        {
                            "detector": detector,
                            "tile_id": int(tile_id),
                            "tile_row": metadata["tile_row"],
                            "tile_col": metadata["tile_col"],
                            "log_rate_offset_p16": tile_offsets[0, sca_index, tile_index],
                            "log_rate_offset_median": tile_offsets[1, sca_index, tile_index],
                            "log_rate_offset_p84": tile_offsets[2, sca_index, tile_index],
                            "initial_rate_nm_day_p16": tile_initial[0, sca_index * len(data["tile_ids"]) + tile_index],
                            "initial_rate_nm_day_median": tile_initial[1, sca_index * len(data["tile_ids"]) + tile_index],
                            "initial_rate_nm_day_p84": tile_initial[2, sca_index * len(data["tile_ids"]) + tile_index],
                            "final_rate_nm_day_p16": tile_final[0, sca_index * len(data["tile_ids"]) + tile_index],
                            "final_rate_nm_day_median": tile_final[1, sca_index * len(data["tile_ids"]) + tile_index],
                            "final_rate_nm_day_p84": tile_final[2, sca_index * len(data["tile_ids"]) + tile_index],
                            "mean_rate_nm_day_p16": tile_mean[0, sca_index * len(data["tile_ids"]) + tile_index],
                            "mean_rate_nm_day_median": tile_mean[1, sca_index * len(data["tile_ids"]) + tile_index],
                            "mean_rate_nm_day_p84": tile_mean[2, sca_index * len(data["tile_ids"]) + tile_index],
                        }
                    )

    if "detector_thickness_nm" in samples:
        detector_thickness_quantiles = np.percentile(
            np.asarray(samples["detector_thickness_nm"]), [16, 50, 84], axis=0
        )
        with (output_dir / "detector_thickness_posterior.csv").open(
            "w", newline=""
        ) as handle:
            fields = [
                "detector",
                "tile_id",
                "tile_row",
                "tile_col",
                "thickness_nm_p16",
                "thickness_nm_median",
                "thickness_nm_p84",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for sca_index, detector in enumerate(data["sca_names"]):
                for tile_index, tile_id in enumerate(data["tile_ids"]):
                    tile_metadata = data["metadata"][int(tile_id)]
                    writer.writerow(
                        {
                            "detector": detector,
                            "tile_id": int(tile_id),
                            "tile_row": tile_metadata["tile_row"],
                            "tile_col": tile_metadata["tile_col"],
                            "thickness_nm_p16": detector_thickness_quantiles[
                                0, sca_index, tile_index
                            ],
                            "thickness_nm_median": detector_thickness_quantiles[
                                1, sca_index, tile_index
                            ],
                            "thickness_nm_p84": detector_thickness_quantiles[
                                2, sca_index, tile_index
                            ],
                        }
                    )

    if "ar_sca_mean" in samples:
        sca_quantiles = np.percentile(
            np.asarray(samples["ar_sca_mean"]), [16, 50, 84], axis=0
        )
        if sca_quantiles.shape[-1] != 3 * ar_layers:
            raise ValueError("SCA AR posterior shape does not match the selected layer count")
        sca_quantiles = sca_quantiles.reshape(
            sca_quantiles.shape[:-1] + (ar_layers, 3)
        )
        with (output_dir / "sca_ar_posterior.csv").open("w", newline="") as handle:
            fields = ["detector"] + ar_quantile_fields(ar_layers)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, detector in enumerate(data["sca_names"]):
                row = {"detector": detector}
                add_sca_ar_quantiles(row, sca_quantiles, index, ar_layers)
                writer.writerow(row)

    if "sca_ar_opd_nm" in samples:
        sca_opd_quantiles = np.percentile(
            np.asarray(samples["sca_ar_opd_nm"]), [16, 50, 84], axis=0
        )
        with (output_dir / "sca_ar_opd_posterior.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "detector",
                    "opd_nm_p16",
                    "opd_nm_median",
                    "opd_nm_p84",
                ],
            )
            writer.writeheader()
            for index, detector in enumerate(data["sca_names"]):
                writer.writerow(
                    {
                        "detector": detector,
                        "opd_nm_p16": sca_opd_quantiles[0, index],
                        "opd_nm_median": sca_opd_quantiles[1, index],
                        "opd_nm_p84": sca_opd_quantiles[2, index],
                    }
                )

    if (
        args.np_modulation
        or getattr(args, "constant_rate_modulation", False)
        or (
            getattr(args, "fully_np_trajectory", False)
            and not getattr(args, "independent_tile_trajectories", False)
        )
    ):
        node_samples = np.asarray(samples["np_node_multipliers"])
        node_quantiles = np.percentile(node_samples, [16, 50, 84], axis=0)
        with (output_dir / "np_node_posterior.csv").open("w", newline="") as handle:
            node_rate_samples = np.asarray(
                samples["node_instantaneous_rate_nm_day"]
            )
            node_rate_quantiles = np.percentile(node_rate_samples, [16, 50, 84], axis=0)
            fields = [
                "repeat",
                "days",
                "node_multiplier_p16",
                "node_multiplier_median",
                "node_multiplier_p84",
                "instantaneous_rate_nm_day_p16",
                "instantaneous_rate_nm_day_median",
                "instantaneous_rate_nm_day_p84",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, repeat in enumerate(data["repeat_values"]):
                writer.writerow(
                    {
                        "repeat": int(repeat),
                        "days": data["node_days"][index],
                        "node_multiplier_p16": node_quantiles[0, index],
                        "node_multiplier_median": node_quantiles[1, index],
                        "node_multiplier_p84": node_quantiles[2, index],
                        "instantaneous_rate_nm_day_p16": node_rate_quantiles[0, index],
                        "instantaneous_rate_nm_day_median": node_rate_quantiles[1, index],
                        "instantaneous_rate_nm_day_p84": node_rate_quantiles[2, index],
                        }
                    )

    if (
        getattr(args, "early_time_scale_cutoff_day", None) is not None
        and "early_time_scale" in samples
    ):
        scale_quantiles = np.percentile(
            np.asarray(samples["early_time_scale"]), [16, 50, 84], axis=0
        )
        with (output_dir / "early_time_scale_posterior.csv").open(
            "w", newline=""
        ) as handle:
            fields = [
                "repeat",
                "days",
                "relative_count_scale_p16",
                "relative_count_scale_median",
                "relative_count_scale_p84",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, (repeat, day) in enumerate(
                zip(data["early_time_scale_repeat"], data["early_time_scale_days"])
            ):
                writer.writerow(
                    {
                        "repeat": int(repeat),
                        "days": float(day),
                        "relative_count_scale_p16": scale_quantiles[0, index],
                        "relative_count_scale_median": scale_quantiles[1, index],
                        "relative_count_scale_p84": scale_quantiles[2, index],
                    }
                )

    if getattr(args, "fit_pre_post_ice_index", False) or getattr(
        args, "fit_post_reset_ice_index", False
    ):
        if getattr(args, "fit_pre_post_ice_index", False):
            ice_index_rows = (
                ("pre-reset", "pre_reset_ice_n", "pre_reset_ice_k"),
                ("post-reset", "post_reset_ice_n", "post_reset_ice_k"),
            )
        else:
            ice_index_rows = (
                ("pre-reset", None, None),
                ("post-reset", "post_reset_ice_index", None),
            )
        with (output_dir / "ice_index_posterior.csv").open("w", newline="") as handle:
            fields = [
                "segment",
                "n_p16",
                "n_median",
                "n_p84",
                "k_p16",
                "k_median",
                "k_p84",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for segment, n_name, k_name in ice_index_rows:
                if n_name is None:
                    n_values = np.full(len(np.asarray(samples["ice_thickness"])), ICE_INDEX_DEFAULT)
                else:
                    n_values = np.asarray(samples[n_name], dtype=float)
                if k_name is None:
                    k_values = np.zeros(len(n_values))
                else:
                    k_values = np.asarray(samples[k_name], dtype=float)
                n_quantiles = np.percentile(n_values, [16, 50, 84])
                k_quantiles = np.percentile(k_values, [16, 50, 84])
                writer.writerow(
                    {
                        "segment": segment,
                        "n_p16": n_quantiles[0],
                        "n_median": n_quantiles[1],
                        "n_p84": n_quantiles[2],
                        "k_p16": k_quantiles[0],
                        "k_median": k_quantiles[1],
                        "k_p84": k_quantiles[2],
                        }
                    )

    if getattr(args, "fit_ice_optical_modulation", False):
        with (output_dir / "ice_optical_modulation_posterior.csv").open(
            "w", newline=""
        ) as handle:
            fields = [
                "parameter",
                "wavelength_nm",
                "log_modulation_p16",
                "log_modulation_median",
                "log_modulation_p84",
                "multiplier_p16",
                "multiplier_median",
                "multiplier_p84",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for parameter, sample_name in (
                ("n", "ice_n_log_modulation"),
                ("k", "ice_k_log_modulation"),
            ):
                log_quantiles = np.percentile(
                    np.asarray(samples[sample_name], dtype=float), [16, 50, 84], axis=0
                )
                multiplier_quantiles = np.exp(log_quantiles)
                for index, wavelength_nm in enumerate(
                    ICE_MODULATION_NODE_WAVELENGTHS_NM
                ):
                    writer.writerow(
                        {
                            "parameter": parameter,
                            "wavelength_nm": wavelength_nm,
                            "log_modulation_p16": log_quantiles[0, index],
                            "log_modulation_median": log_quantiles[1, index],
                            "log_modulation_p84": log_quantiles[2, index],
                            "multiplier_p16": multiplier_quantiles[0, index],
                            "multiplier_median": multiplier_quantiles[1, index],
                            "multiplier_p84": multiplier_quantiles[2, index],
                        }
                    )

    trajectory_label = (
        "fixed positive instantaneous rate trajectory"
        if getattr(args, "fixed_trajectory_rate_nm_day", None) is not None
        else "free constant positive instantaneous rate trajectory"
        if getattr(args, "constant_rate_trajectory", False)
        else "independent per-SCA/tile positive log-rate node trajectories"
        if getattr(args, "independent_tile_trajectories", False)
        else "fully non-parametric positive log-rate trajectory"
        if getattr(args, "fully_np_trajectory", False)
        else "constant positive rate with entropy-regularized node modulation"
        if getattr(args, "constant_rate_modulation", False)
        else "NP-modulated rate-integrated square-root trajectory"
        if args.np_modulation
        else "shared rate-integrated square-root trajectory"
    )
    incidence_angle_prior_description = (
        f"flat prior over 0 to {INCIDENCE_ANGLE_MAX_DEG:g} degrees"
        if getattr(args, "incidence_angle_prior_uniform", False)
        else f"truncated-Normal prior sigma `{getattr(args, 'incidence_angle_prior_sigma_deg', INCIDENCE_ANGLE_PRIOR_SIGMA_DEG)}` degrees"
    )
    ar_mean = np.asarray(samples["ar_mean"]).reshape(-1, ar_layers, 3)
    lines = [
        f"# {scope_label(data['sca_names'])} NumPyro hierarchical TMM fit",
        "",
        "## Reproduce",
        "",
        "```bash",
        command_line,
        "```",
        "",
        f"Tile statistics: `{args.tile_statistics}`.",
        (
            f"Complete observation groups before day `{args.min_observation_day}` were excluded; "
            f"the first retained observation is at day `{np.min(data['days']):.8g}`."
            if getattr(args, "min_observation_day", None) is not None
            else "All available observation groups were retained."
        ),
        (
            "Input tile statistics already contain the fixed count-space gain and ACM corrections; "
            "no early-time gain correction is fitted."
            if getattr(args, "corrections_already_applied", False)
            else (
            f"Fixed gain and ACM corrections were applied from `{args.fixed_corrections}`; "
            "no early-time gain correction is fitted."
            if getattr(args, "fixed_corrections", None) is not None
            else (
            f"One relative observed-count scale per repeat epoch, shared across all LEDs, is fitted through day "
            f"`{args.early_time_scale_cutoff_day}`; the first epoch is the unit reference and "
            f"the log-relative scale prior sigma is `{args.early_time_scale_prior_sigma}`. "
            "The scale is applied to the observed counts before their per-LED normalization."
            if getattr(args, "early_time_scale_cutoff_day", None) is not None
            else "No early-time scale correction is fitted."
            )
            )
        ),
        f"SCAs used: `{data['sca_names'].tolist()}`.",
        f"QE curves used by SCA: `{dict(zip(data['sca_names'].tolist(), (data['qe_columns'] + 1).tolist()))}`.",
        f"Coating model: `{ar_layers}` dielectric AR layer(s), ordered from air toward HgCdTe.",
        (
            "AR optical properties: one full AR parameter vector is shared across all selected tiles."
            if getattr(args, "shared_ar_parameters_across_tiles", False)
            else (
                "AR optical properties: one direct AR parameter vector is fitted independently per SCA/tile; no tile hierarchy."
                if getattr(args, "independent_tile_ar_parameters", False)
                else (
                "AR optical properties: direct single-tile parameters; no tile hierarchy."
                if is_single_tile_data(data)
                else "AR optical properties: shared n/k across all tiles; thickness varies hierarchically."
                if args.shared_ar_optical_properties
                else "AR optical properties: thickness, n, and k all vary hierarchically."
                )
            )
        ),
        (
            "AR absorption indices are fitted."
            if args.fit_ar_absorption
            else "AR layers are lossless dielectrics with k fixed to zero."
        ),
        (
            "Per-SCA AR optical-path perturbation: one zero-centered effective OPD "
            "in nm is added to the first AR-layer coherent phase with an angle-aware "
            "propagation factor; prior sigma is "
            f"`{getattr(args, 'sca_ar_opd_prior_sigma_nm', SCA_AR_OPD_PRIOR_SIGMA_NM)}` nm."
            if getattr(args, "fit_sca_ar_opd", False)
            else "No per-SCA AR optical-path perturbation is fitted."
        ),
        f"Tiles per SCA: `{len(data['tile_ids'])}`; tile IDs: `{data['tile_ids'].tolist()}`.",
        (
            "Absolute substrate-power diagnostic mode: SED-only source integration is "
            "normalized by the modeled clean substrate power, and HgCdTe n/k are fixed "
            "to the wavelength-dependent literature reference (zero log scales)."
            if getattr(args, "fixed_absolute_hgcdte_scales", False)
            else "Absolute substrate-power nuisance mode: SED-only source integration is "
            "normalized by the modeled clean substrate power, and one global positive "
            "HgCdTe n/k log-scale pair is fitted per SCA and shared across all tiles "
            "around the wavelength-dependent literature reference "
            f"with prior sigma `{getattr(args, 'absolute_hgcdte_scale_prior_sigma_log', ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG)}` "
            "and zero log-offset prior mean. No HgCdTe spatial regularization is used."
            if getattr(args, "shared_absolute_hgcdte_scales", False)
            else "Absolute substrate-power nuisance mode: SED-only source integration is "
            "normalized by the modeled clean substrate power, and global positive "
            "HgCdTe n/k log scales are fitted independently for every SCA/tile around "
            "the wavelength-dependent literature reference "
            f"with prior sigma `{getattr(args, 'absolute_hgcdte_scale_prior_sigma_log', ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG)}` "
            "and zero log-offset prior mean."
            if getattr(args, "absolute_substrate_power", False)
            else (
            "HgCdTe n/k are fitted as global wavelength-independent optical "
            f"parameters with priors centered at `{NOMINAL_HGCDTE.tolist()}`; "
            f"bounds are `{HGCDTE_LOWER.tolist()}` to `{HGCDTE_UPPER.tolist()}`."
            if getattr(args, "fit_hgcdte", False)
            else (
                "HgCdTe n/k use global smooth log-space modulation around the "
                f"80 K wavelength-dependent baseline at nodes `{HGCDTE_MODULATION_NODE_WAVELENGTHS_NM.tolist()}` nm; "
                f"node prior sigma is `{args.hgcdte_modulation_prior_sigma_log}`."
                if getattr(args, "fit_hgcdte_modulation", False)
                else (
                "HgCdTe uses the fixed 80 K wavelength-dependent optical-constant "
                f"sensitivity model from `{HGCDTE_OPTICAL_CONSTANTS_SOURCE}`."
                if getattr(args, "use_hgcdte_optical_constants", False)
                else f"HgCdTe fixed at n/k = `{NOMINAL_HGCDTE.tolist()}`."
                )
            )
            )
        ),
        (
            "Finite HgCdTe detector layer: fixed wavelength-dependent HgCdTe n/k, "
            f"thickness reference `{getattr(args, 'detector_thickness_nm', DETECTOR_THICKNESS_DEFAULT_NM)}` nm, "
            f"backside index `{getattr(args, 'detector_backside_index', 1.0)}`."
            if getattr(args, "finite_hgcdte_detector", False)
            or getattr(args, "fit_detector_thickness", False)
            else "HgCdTe is used as the semi-infinite terminal boundary."
        ),
        (
            "Detector thickness is fit independently per SCA/tile in log space with "
            f"prior sigma `{getattr(args, 'detector_thickness_prior_sigma_log', DETECTOR_THICKNESS_PRIOR_SIGMA_LOG)}` "
            "and neighboring-tile smoothness."
            if getattr(args, "fit_detector_thickness", False)
            else "Detector thickness is fixed for the finite-layer model."
        ),
        (
            "Conductive interlayer: global Drude-like film between the AR stack and HgCdTe; "
            f"reference wavelength `{CONDUCTIVE_REFERENCE_WAVELENGTH_NM}` nm and "
            f"epsilon_inf `{CONDUCTIVE_EPSILON_INF}`."
            if getattr(args, "fit_conductive_layer", False)
            else "No conductive interlayer is included."
        ),
        (
            f"QE is modulated by {len(qe_rbf_nodes)} normalized Gaussian RBF nodes at "
            f"`{qe_rbf_nodes.tolist()}` nm with width "
            f"`{qe_rbf_width_nm}` nm; log-node prior sigma is "
            f"`{qe_rbf_prior_sigma_log}`."
            if getattr(args, "fit_qe_rbf", False)
            else "QE uses the archived ground-measured wavelength weighting without a fitted RBF correction."
        ),
        (
            "SED spectral weighting: published relative energy output converted to photon rate."
            if args.sed_weighting == "photon"
            else "SED spectral weighting: table values treated as energy weights."
        ),
        (
            "Ice uses global smooth log-space n/k modulation around the published "
            "100 K amorphous-H2O optical constants at nodes "
            f"`{ICE_MODULATION_NODE_WAVELENGTHS_NM.tolist()}` nm with n prior sigma "
            f"`{getattr(args, 'ice_n_modulation_prior_sigma_log', None) if getattr(args, 'ice_n_modulation_prior_sigma_log', None) is not None else getattr(args, 'ice_modulation_prior_sigma_log', ICE_MODULATION_PRIOR_SIGMA_LOG)}` "
            f"and k prior sigma `{getattr(args, 'ice_modulation_prior_sigma_log', ICE_MODULATION_PRIOR_SIGMA_LOG)}`; "
            f"the baseline is `{args.ice_optical_constants_source}`."
            if getattr(args, "fit_ice_optical_modulation", False)
            else (
                "Ice uses the published 100 K amorphous-H2O optical constants with "
                f"wavelength-dependent n and k from `{args.ice_optical_constants_source}`."
                if getattr(args, "use_asw_optical_constants", False)
                else (
                    "Ice uses separate pre/post-reset complex indices; the real-part prior is "
                    f"Normal(`{args.ice_index_prior_mean}`, `{args.ice_index_prior_sigma}`) "
                    f"truncated to `{args.ice_index_lower}`--`{args.ice_index_upper}`, and "
                    f"the nonnegative imaginary-part prior has scale `{args.ice_extinction_prior_sigma}` "
                    f"and upper bound `{args.ice_extinction_upper}`."
                    if getattr(args, "fit_pre_post_ice_index", False)
                    else (
                        "Ice uses fixed pre-reset n=1.25 and k=0, with a fitted real post-reset "
                        "index."
                        if getattr(args, "fit_post_reset_ice_index", False)
                        else "Ice is modeled as a lossless constant-index layer with n=1.25 and k=0."
                    )
                )
            )
        ),
        (
            "One free post-reset grey log correction is fitted per SCA/tile, added to "
            "the corrected observed log ratios, with Normal prior sigma "
            f"`{getattr(args, 'post_reset_tile_log_offset_prior_sigma', POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA)}`."
            if getattr(args, "fit_post_reset_tile_log_offset", False)
            else "No free post-reset per-tile grey log correction is fitted."
        ),
        (
            "A globally shared complex-q boundary correction is fitted jointly with the "
            "production AR, ice, incidence, reset, and HgCdTe parameters. Its parameter "
            f"bounds are `{COMPLEX_Q_LOWER.tolist()}` to `{COMPLEX_Q_UPPER.tolist()}`."
            if getattr(args, "fit_complex_q", False)
            else "No complex-q boundary correction is fitted."
        ),
        (
            (
                (
                "Fitted one global non-negative chief-ray incidence angle relative to normal; "
                f"hard bound `{INCIDENCE_ANGLE_MAX_DEG}` degrees with an "
                f"{incidence_angle_prior_description}."
                    if getattr(args, "fit_incidence_angle", False)
                    else (
                        "Optical response is averaged over a two-dimensional uniformly "
                        "illuminated circular pupil with marginal-ray half-angle "
                        f"`{getattr(args, 'incidence_pupil_half_angle_deg', 0.0)}` degrees; "
                        + (
                            f"chief-ray tilts are loaded from `{args.incidence_chief_ray_map}`."
                            if getattr(args, "incidence_chief_ray_map", None) is not None
                            else "the global chief-ray x/y tilts are "
                            f"`{getattr(args, 'incidence_chief_ray_x_deg', 0.0)}` / "
                            f"`{getattr(args, 'incidence_chief_ray_y_deg', 0.0)}` degrees."
                        )
                        if (
                            getattr(args, "incidence_pupil_half_angle_deg", 0.0) != 0.0
                            or getattr(args, "incidence_chief_ray_x_deg", 0.0) != 0.0
                            or getattr(args, "incidence_chief_ray_y_deg", 0.0) != 0.0
                            or getattr(args, "incidence_chief_ray_map", None) is not None
                        )
                        else (
                            "Optical response is averaged over an unpolarized TE/TM cone "
                            "uniform in solid angle with half-angle "
                            f"`{args.incidence_cone_half_angle_deg}` degrees."
                            if getattr(args, "incidence_cone_half_angle_deg", 0.0) != 0.0
                            else "Optical response uses normal incidence."
                        )
                    )
                )
            )
        ),
        "Input tile statistics already contain all gain and ACM corrections; "
        "the ACM, late-ACM, additional-ACM, and pre-reset ACM terms are not fitted.",
        (
            f"Likelihood pixel weight before selected-tile averaging: `{args.likelihood_weight}` "
            f"({'automatic one-effective-datum normalization' if args.likelihood_weight_automatic else 'user supplied'}); "
            f"effective aggregated SSE weight: `{args.likelihood_weight / len(data['tile_ids'])}`; "
            f"log-noise sigma: `{args.log_noise_sigma}`."
        ),
        (
            f"Direct per-tile AR prior sigma in unconstrained space: `{args.ar_mean_prior_sigma}`."
            if getattr(args, "independent_tile_ar_parameters", False)
            else f"AR population-mean prior sigma in unconstrained space: `{args.ar_mean_prior_sigma}`."
        ),
        (
            "Each tile AR vector is sampled directly around the reference in raw coordinates."
            if getattr(args, "independent_tile_ar_parameters", False)
            else "Global AR mean is sampled in standard-normal coordinates with the same "
            "raw-space prior."
            if getattr(args, "standardize_ar_mean", False)
            else "Global AR mean is sampled directly in raw coordinates."
        ),
        (
            "AR thickness/index coordinates use an orthogonal full-rank rotation; "
            "the standardized Normal prior and all parameter degrees of freedom are unchanged."
            if getattr(args, "rotate_ar_coordinates", False)
            else (
                "AR layer indices are parameterized in strict increasing order, "
                "so each layer has a larger real index than the preceding layer."
                if getattr(args, "ordered_ar_indices", False)
                else "AR thickness/index coordinates use their standard ordering."
            )
        ),
        (
            "The instantaneous ice rate is fixed at "
            f"`{args.fixed_trajectory_rate_nm_day}` nm/day; no trajectory-rate "
            "parameter is sampled."
            if getattr(args, "fixed_trajectory_rate_nm_day", None) is not None
            else (
                "Free constant-rate prior: centered at the default "
                f"`{trajectory_rate_per_scale(data['days']) * TRAJECTORY_SCALE_INIT:.8g}` nm/day."
                if getattr(args, "constant_rate_trajectory", False)
                else (
                "Fully non-parametric node-rate prior: centered at the default "
                f"`{trajectory_rate_per_scale(data['days']) * TRAJECTORY_SCALE_INIT:.8g}` nm/day."
                if args.trajectory_rate_prior_mean_nm_day is None
                else "Fully non-parametric node-rate prior: centered at "
                f"`{args.trajectory_rate_prior_mean_nm_day}` nm/day."
                )
                if getattr(args, "fully_np_trajectory", False)
                else (
                    "Constant base-rate prior: centered at the default "
                    f"`{trajectory_rate_per_scale(data['days']) * TRAJECTORY_SCALE_INIT:.8g}` nm/day."
                    if args.trajectory_rate_prior_mean_nm_day is None
                    else "Constant base-rate prior: centered at "
                    f"`{args.trajectory_rate_prior_mean_nm_day}` nm/day."
                )
                if getattr(args, "constant_rate_modulation", False)
                else (
                    "Unmodulated initial instantaneous-rate prior: centered at the default "
                    f"`{trajectory_rate_per_scale(data['days']) * TRAJECTORY_SCALE_INIT:.8g}` nm/day."
                    if args.trajectory_rate_prior_mean_nm_day is None
                    else "Unmodulated initial instantaneous-rate prior: centered at "
                    f"`{args.trajectory_rate_prior_mean_nm_day}` nm/day."
                )
            )
        ),
        (
            f"Initial ice thickness is fixed at `{args.fixed_initial_thickness_nm}` nm."
            if getattr(args, "fixed_initial_thickness_nm", None) is not None
            else (
                "Each selected SCA/tile has an independent lognormal initial ice thickness "
                f"with median `{args.tile_initial_thickness_log_median_nm}` nm and log sigma "
                f"`{args.tile_initial_thickness_log_sigma}`."
                if getattr(args, "tile_initial_thickness_lognormal", False)
                else (
                    "Initial ice thickness is fitted with a flat prior over 0 to 50 nm."
                    if getattr(args, "initial_thickness_prior_uniform", False)
                    else "Initial ice thickness is fitted with a truncated Normal prior centered at "
                    f"`{getattr(args, 'initial_thickness_prior_mean_nm', INITIAL_THICKNESS_PRIOR_MEAN_NM)}` nm "
                    f"with sigma `{getattr(args, 'initial_thickness_prior_sigma_nm', INITIAL_THICKNESS_PRIOR_SIGMA_NM)}` nm over 0 to 50 nm."
                )
            )
        ),
        (
            "Initial thickness is sampled in per-tile log coordinates and transformed "
            "to positive physical thickness."
            if getattr(args, "tile_initial_thickness_lognormal", False)
            else (
                "Initial thickness is sampled in an unconstrained central-pivot coordinate; "
                "the exact truncated prior is applied to the recovered initial thickness."
                if getattr(args, "pivot_initial_thickness", False)
                else "Initial thickness is sampled directly in its constrained coordinate."
            )
        ),
        (
            f"The ice trajectory resets at day `{args.ice_reset_day}`; a separate "
            "post-reset initial thickness is fitted while sharing the rate trajectory."
            if getattr(args, "ice_reset_day", None) is not None
            else "No ice-level reset segment is modeled."
        ),
        f"Trajectory-rate prior log-scale sigma: `{args.trajectory_rate_prior_sigma_log}`.",
        (
            "No trajectory-rate nodes are sampled because the instantaneous rate is fixed."
            if getattr(args, "fixed_trajectory_rate_nm_day", None) is not None
            else "One free constant instantaneous-rate parameter is sampled; no rate nodes are used."
            if getattr(args, "constant_rate_trajectory", False)
            else "Fully non-parametric log-rate nodes are sampled in exact prior-whitened "
            "coordinates; all node degrees of freedom remain free."
            if getattr(args, "whiten_np_trajectory", False)
            else "Fully non-parametric log-rate nodes are sampled directly."
        ),
        (
            "Whitened rate latents use a full-rank orthogonal rotation from the "
            "conditional posterior Hessian; all node degrees of freedom remain free."
            if getattr(args, "rotate_np_likelihood", False)
            else "No likelihood-informed rate rotation is applied."
        ),
        "The primary trajectory quantity is a positive instantaneous rate; ice thickness is its time integral.",
        (
            "The instantaneous rate is fixed at "
            f"`{args.fixed_trajectory_rate_nm_day}` nm/day for both trajectory segments."
            if getattr(args, "fixed_trajectory_rate_nm_day", None) is not None
            else "A single free positive instantaneous rate is used for the complete trajectory."
            if getattr(args, "constant_rate_trajectory", False)
            else "Fully non-parametric rates are log-linearly interpolated between observation-epoch nodes, "
            "so the positive accumulation trajectory remains monotonic."
            if getattr(args, "fully_np_trajectory", False)
            else (
                "A constant positive base rate is log-linearly modulated between observation-epoch nodes; "
                + (
                    "generalized relative-entropy regularization is applied."
                    if args.constant_rate_smoothness_weight == 0.0
                    else "generalized relative-entropy and time-aware log-rate smoothness regularization are applied."
                )
                if getattr(args, "constant_rate_modulation", False)
                else "NP node values are log-linearly interpolated rate multipliers, so the rate is continuous at observation-epoch nodes."
            )
        ),
        (
            f"MAP-only trajectory-rate start: `{args.map_trajectory_rate_nm_day}` nm/day."
            if args.map_trajectory_rate_nm_day is not None
            else (
                f"Fixed trajectory rate: `{args.fixed_trajectory_rate_nm_day}` nm/day."
                if getattr(args, "fixed_trajectory_rate_nm_day", None) is not None
                else "MAP-only trajectory-rate start: historical initialization."
            )
        ),
        (
            "Cold MAP uses a finite smooth continuation outside the initial-thickness "
            "support for line-search stability; the exact truncated prior is restored "
            "before NUTS."
            if getattr(args, "map_steps", 0)
            and getattr(args, "pivot_initial_thickness", False)
            else "No MAP-only initial-thickness support continuation is used."
        ),
        (
            "Deterministic MAP-only evaluation; no NUTS posterior was sampled."
            if getattr(args, "map_only", False)
            else f"Warmup/samples/chains: `{args.warmup}/{args.samples}/{args.chains}`."
        ),
        (
            f"NUTS dense mass matrix: `{args.dense_mass}`; global dense mass block: "
            f"`{getattr(args, 'dense_mass_global', False)}`; maximum tree depth: "
            f"`{args.max_tree_depth}`."
            if not getattr(args, "map_only", False)
            else "NUTS settings were not used."
        ),
        (
            f"NUTS starts use the complete MAP sample-site state with per-chain "
            f"unconstrained jitter sigma `{getattr(args, 'nuts_init_jitter', NUTS_INIT_JITTER)}`; "
            f"a boundary-valued initial thickness is nudged to `{args.nuts_initial_thickness_nm}` nm "
            "only for initialization."
            if not getattr(args, "map_only", False)
            else "NUTS initialization was not used."
        ),
        (
            f"Wavelength grid: exact SED nodes / stride `{args.wavelength_stride}`."
            if args.wavelength_step_nm is None
            else f"Wavelength grid step/stride: `{args.wavelength_step_nm}` nm / `{args.wavelength_stride}`."
        ),
        f"Response integration wavelengths retained: `{data.get('response_wavelength_count', 0)}`.",
        f"Prediction likelihood row chunk size: `{data.get('prediction_row_chunk_size', PREDICTION_ROW_CHUNK_SIZE)}`.",
        (
            f"MAP optimizer/maximum iterations: `{args.map_optimizer}` / `{args.map_steps}`; "
            f"parameter scaling: `{args.map_parameter_scaling}`; "
            f"projected-gradient tolerance: `{args.map_tolerance}`; "
            f"function-reduction tolerance: `{args.map_ftol}`."
            if args.map_optimizer == "lbfgs"
            else (
                f"MAP optimizer/maximum iterations: `{args.map_optimizer}` / `{args.map_steps}`; "
                f"parameter scaling: `{args.map_parameter_scaling}`; "
                f"projected-gradient tolerance: `{args.map_tolerance}`; "
                "JAXopt line-search stopping is used."
                if args.map_optimizer == "jax-lbfgsb"
                else (
                    f"MAP optimizer/maximum iterations: `{args.map_optimizer}` / `{args.map_steps}`; "
                    f"parameter scaling: `{args.map_parameter_scaling}`; "
                    f"projected-gradient tolerance: `{args.map_tolerance}`; "
                    "exact-Hessian trust-region stopping is used."
                    if args.map_optimizer == "trust-exact"
                    else f"MAP optimizer/steps/learning rate: `{args.map_optimizer}` / "
                    f"`{args.map_steps}` / `{args.map_learning_rate}`."
                )
            )
        ),
        f"Ice trajectory: `{trajectory_label}`.",
    ]
    if getattr(args, "tile_rate_variation", False) or getattr(
        args, "tile_rate_trajectory_variation", False
    ):
        lines.extend(
            [
                "Per-SCA/tile instantaneous-rate variation is enabled.",
                (
                    f"Independent tile-node log-rate prior sigma: `{args.trajectory_rate_prior_sigma_log}`."
                    if getattr(args, "independent_tile_trajectories", False)
                    else f"Tile log-rate prior sigma: `{args.tile_rate_prior_sigma_log}`."
                ),
                (
                    f"Neighboring-tile log-rate sigma: `{args.tile_rate_neighbor_sigma_log}`; "
                    f"selected adjacency edges per SCA: `{len(tile_neighbor_edges(data))}`."
                    if getattr(args, "tile_rate_neighbor_smoothness", True)
                    else "Neighboring-tile log-rate smoothness is disabled."
                ),
            ]
        )
    if getattr(args, "tile_rate_trajectory_variation", False):
        if getattr(args, "independent_tile_trajectories", False):
            lines.extend(
                [
                    "Each SCA/tile has an independent log-rate value at every observation epoch; "
                    "there is no global trajectory or centered tile-deviation parameterization.",
                    (
                        "Within-tile time-curvature regularization is applied with weight "
                        f"`{args.tile_rate_trajectory_smoothness_weight}`; tile-neighbor "
                        "regularization remains disabled."
                        if args.tile_rate_trajectory_smoothness_weight > 0.0
                        else "No tile-neighbor or time-curvature regularization is applied to these trajectories."
                    ),
                ]
            )
        else:
            lines.extend(
                [
                    "Tile-specific log-rate deviations are fit at every observation epoch, "
                    "with zero-mean tile offsets at each node.",
                    "Tile rate curves use time-curvature regularization with weight "
                    f"`{args.tile_rate_trajectory_smoothness_weight}`.",
                ]
            )
    if args.ar_prior_from_map_start:
        lines.append(
            "Direct per-tile AR priors are centered on the physical MAP start."
            if getattr(args, "independent_tile_ar_parameters", False)
            else "AR population-mean prior is centered on the physical MAP start."
        )
    map_init_max_tiles = getattr(args, "map_init_max_tiles", None)
    if map_init_max_tiles is not None:
        lines.append(
            f"Reduced MAP initialization tile count per SCA: `{map_init_max_tiles}`."
        )
    map_init_from = getattr(args, "map_init_from", None)
    if map_init_from is not None:
        lines.append(f"MAP sample sites warm-started from `{map_init_from}`.")
    map_design_wavelength_nm = getattr(args, "map_design_wavelength_nm", None)
    if map_design_wavelength_nm is not None:
        middle_layer_start = (
            "lambda/8 middle-layer"
            if getattr(args, "map_middle_layer_eighth_wave", False)
            else "lambda/2 middle-layer"
            if getattr(args, "map_middle_layer_half_wave", False)
            else "quarter-wave"
        )
        lines.append(
            "MAP-only start uses equal log-index steps and "
            f"{middle_layer_start} optical thicknesses at "
            f"`{map_design_wavelength_nm}` nm; the posterior remains unconstrained."
        )
        if getattr(args, "map_middle_layer_eighth_wave", False):
            lines.append(
                "The three-layer MAP start uses lambda/4, lambda/8, lambda/4 "
                "optical thicknesses from air toward HgCdTe."
            )
        elif getattr(args, "map_middle_layer_half_wave", False):
            lines.append(
                "The three-layer MAP start uses lambda/4, lambda/2, lambda/4 "
                "optical thicknesses from air toward HgCdTe."
            )
    if data["n_scas"] > 1:
        lines.extend(
            [
                (
                    "AR parameters are direct independent per-SCA/tile vectors; no population or tile hierarchy."
                    if getattr(args, "independent_tile_ar_parameters", False)
                    else f"SCA AR sigma prior: `{args.ar_sca_sigma_prior}`; tile AR sigma prior: `{args.ar_tile_sigma_prior}`."
                ),
                (
                    "AR thickness uses global population shrinkage, then SCA-level offsets, "
                    "then tile-level deviations; n/k are shared globally."
                    if args.shared_ar_optical_properties
                    else "AR parameters use global population shrinkage, then SCA-level offsets, then tile-level deviations."
                    if not getattr(args, "independent_tile_ar_parameters", False)
                    else ""
                ),
            ]
        )
    if sampler_diagnostics:
        lines.extend(
            [
                f"Sampler divergences: `{sampler_diagnostics['divergences']}`.",
                f"Mean sampler acceptance probability: `{sampler_diagnostics['mean_accept_probability']:.8g}`.",
                f"Maximum NUTS leapfrog steps: `{sampler_diagnostics['max_num_steps']}`.",
                f"Tree-depth saturated transitions: `{sampler_diagnostics['tree_depth_saturated']}`.",
                f"Maximum finite r_hat: `{sampler_diagnostics['max_r_hat']:.8g}`.",
                f"Minimum effective sample size: `{sampler_diagnostics['min_effective_sample_size']:.8g}`.",
                f"Parameters with r_hat > 1.01: `{sampler_diagnostics['r_hat_gt_1_01']}`.",
                f"Parameters with effective sample size < 100: `{sampler_diagnostics['ess_lt_100']}`.",
                f"Non-finite r_hat diagnostics: `{sampler_diagnostics['r_hat_nonfinite']}`.",
                f"Non-finite effective sample size diagnostics: `{sampler_diagnostics['ess_nonfinite']}`.",
            ]
        )
    if map_diagnostics:
        lines.extend(
            [
                "",
                "## MAP optimization",
                f"- Optimizer: `{map_diagnostics['optimizer']}`.",
                f"- Reported convergence: `{map_diagnostics['converged']}`.",
                (
                    f"- Gradient convergence: `{map_diagnostics['gradient_converged']}`."
                    if "gradient_converged" in map_diagnostics
                    else "- Gradient convergence: `not assessed`."
                ),
                f"- Status: `{map_diagnostics['status']}`; message: `{map_diagnostics['message']}`.",
                f"- Iterations/function evaluations: `{map_diagnostics['iterations']}/{map_diagnostics['function_evaluations']}`.",
                f"- Initial/final objective: `{map_diagnostics['initial_objective']:.10g}` / `{map_diagnostics['objective']:.10g}`.",
                f"- Final raw gradient infinity norm: `{map_diagnostics['gradient_norm']:.10g}`.",
                f"- Final projected gradient infinity norm: `{map_diagnostics.get('projected_gradient_norm', map_diagnostics['gradient_norm']):.10g}`.",
                (
                    f"- Final scaled projected gradient infinity norm: `{map_diagnostics['scaled_projected_gradient_norm']:.10g}`."
                    if "scaled_projected_gradient_norm" in map_diagnostics
                    else "- Final scaled projected gradient infinity norm: `not assessed`."
                ),
                (
                    f"- Function-reduction tolerance: `{map_diagnostics.get('function_tolerance', np.nan):.10g}`; "
                    f"projected-gradient tolerance: `{args.map_tolerance:.10g}`."
                    if args.map_optimizer == "lbfgs"
                    else (
                        f"- Projected-gradient tolerance: `{args.map_tolerance:.10g}`; "
                        "JAXopt line-search stopping was used."
                        if args.map_optimizer == "jax-lbfgsb"
                        else (
                            f"- Projected-gradient tolerance: `{args.map_tolerance:.10g}`; "
                            "exact-Hessian trust-region stopping was used."
                            if args.map_optimizer == "trust-exact"
                            else f"- MAP steps/learning rate: `{args.map_steps}` / `{args.map_learning_rate}`."
                        )
                    )
                ),
                f"- Parameter step norm: `{map_diagnostics['parameter_step_norm']:.10g}`.",
            ]
        )
    if warmup_map_diagnostics:
        lines.extend(
            [
                "",
                "## Fixed-rate warmup",
                f"- Fixed instantaneous rate: `{WARMUP_FIXED_RATE_NM_DAY}` nm/day.",
                f"- MAP steps: `{WARMUP_MAP_STEPS}`.",
                f"- AR refractive indices fixed by layer: `{WARMUP_FIXED_AR_INDICES.tolist()}`.",
                f"- HgCdTe n/k nuisance scales fixed to literature: `{WARMUP_FIXED_HGCDTE_SCALES}`.",
                f"- HgCdTe n/k nuisance scales shared across tiles: `{WARMUP_SHARED_HGCDTE_SCALES}`.",
                f"- Reported convergence: `{warmup_map_diagnostics['converged']}`.",
                f"- Initial/final objective: `{warmup_map_diagnostics['initial_objective']:.10g}` / `{warmup_map_diagnostics['objective']:.10g}`.",
                f"- Final projected gradient infinity norm: `{warmup_map_diagnostics.get('projected_gradient_norm', warmup_map_diagnostics['gradient_norm']):.10g}`.",
            ]
        )
    elif "diagnostic_diverging" in samples:
        divergences = int(np.count_nonzero(samples["diagnostic_diverging"]))
        accept_probability = np.asarray(samples["diagnostic_accept_prob"])
        num_steps = np.asarray(samples["diagnostic_num_steps"])
        lines.extend(
            [
                f"Sampler divergences: `{divergences}`.",
                f"Mean sampler acceptance probability: `{np.mean(accept_probability):.8g}`.",
                f"Maximum NUTS leapfrog steps: `{int(np.max(num_steps))}`.",
            ]
        )
    if args.np_modulation:
        lines.extend(
            [
                f"NP entropy/smoothness weights: `{args.np_entropy_weight}` / `{args.np_smoothness_weight}`.",
                f"NP node multiplier limit: `{args.np_max_node_multiplier}`; first node fixed at one.",
                "NP modulation is regularized in log-node space with generalized relative entropy and time-aware curvature.",
            ]
        )
    if getattr(args, "constant_rate_modulation", False):
        lines.extend(
            [
                f"Constant-rate NP entropy weight: `{args.np_entropy_weight}`.",
                f"Constant-rate smoothness weight: `{args.constant_rate_smoothness_weight}`.",
                f"NP node multiplier limit: `{args.np_max_node_multiplier}`; first node fixed at one.",
                (
                    "Constant-rate modulation uses generalized relative entropy only; "
                    "rate smoothness regularization is disabled."
                    if args.constant_rate_smoothness_weight == 0.0
                    else "Constant-rate modulation uses generalized relative entropy plus time-aware log-rate smoothness."
                ),
            ]
        )
    if getattr(args, "independent_tile_trajectories", False):
        lines.extend(
            [
                f"Independent tile-node rate prior log sigma: `{args.trajectory_rate_prior_sigma_log}`.",
                "No global non-parametric trajectory is fitted.",
            ]
        )
    elif getattr(args, "fully_np_trajectory", False):
        lines.extend(
            [
                f"Fully non-parametric trajectory smoothness weight: `{args.np_smoothness_weight}`.",
                f"Node-rate prior log sigma: `{args.trajectory_rate_prior_sigma_log}`; no square-root rate law is imposed.",
                "Node rates are positive and regularized by time-aware log-rate curvature.",
            ]
        )
    lines.extend(
        [
            "",
            "| Quantity | 16th percentile | Median | 84th percentile |",
            "|---|---:|---:|---:|",
        ]
    )
    summary_quantities = [("Initial thickness (nm)", "initial_thickness_nm")]
    if (
        getattr(args, "ice_reset_day", None) is not None
        and "reset_initial_thickness_nm" in samples
    ):
        summary_quantities.append(
            ("Post-reset initial thickness (nm)", "reset_initial_thickness_nm")
        )
    if getattr(args, "fit_incidence_angle", False):
        summary_quantities.append(("Incidence angle (deg)", "incidence_angle_deg"))
    if getattr(args, "fit_hgcdte", False):
        summary_quantities.extend(
            (
                ("HgCdTe refractive index", "hgcdte_n"),
                ("HgCdTe absorption index", "hgcdte_k"),
            )
        )
    if (
        getattr(args, "finite_hgcdte_detector", False)
        or getattr(args, "fit_detector_thickness", False)
    ) and "detector_thickness_nm" in samples:
        summary_quantities.append(
            ("Detector thickness (nm)", "detector_thickness_nm")
        )
    if getattr(args, "fit_conductive_layer", False):
        summary_quantities.extend(
            (
                ("Conductive layer thickness (nm)", "conductive_thickness_nm"),
                ("Conductive plasma strength", "conductive_plasma_strength"),
                ("Conductive damping ratio", "conductive_damping_ratio"),
            )
        )
    for label, key in (
        ("Initial instantaneous rate (nm/day)", "initial_rate_nm_day"),
        ("Final instantaneous rate (nm/day)", "final_rate_nm_day"),
        ("Time-averaged instantaneous rate (nm/day)", "mean_rate_nm_day"),
    ):
        if key in samples:
            summary_quantities.append((label, key))
    for label, key in summary_quantities:
        values = np.asarray(samples[key])
        values = quantile(values)
        lines.append(f"| {label} | {values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |")
    if (
        getattr(args, "early_time_scale_cutoff_day", None) is not None
        and "early_time_scale" in samples
    ):
        scale_quantiles = np.percentile(
            np.asarray(samples["early_time_scale"]), [16, 50, 84], axis=0
        )
        lines.extend(
            [
                "",
                "### Early-time Epoch Scales",
                "",
                "| Repeat | Day | 16th percentile | Median | 84th percentile |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for index, (repeat, day) in enumerate(
            zip(data["early_time_scale_repeat"], data["early_time_scale_days"])
        ):
            values = scale_quantiles[:, index]
            lines.append(
                f"| {int(repeat)} | {float(day):.8g} | "
                f"{values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |"
            )
    if getattr(args, "fit_qe_rbf", False):
        qe_node_quantiles = np.percentile(
            np.asarray(samples["qe_rbf_modulation"]), [16, 50, 84], axis=0
        )
        for index, wavelength in enumerate(qe_rbf_nodes):
            values = qe_node_quantiles[:, index]
            lines.append(
                f"| QE RBF multiplier at {wavelength:.0f} nm | "
                f"{values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |"
            )
    acm_step_samples = np.asarray(samples["acm_step_log_offset"])
    acm_step_percent = 100.0 * np.expm1(acm_step_samples)
    if acm_step_percent.ndim == 1:
        values = quantile(acm_step_percent)
        lines.append(
            f"| ACM throughput step (%) | {values[0]:.8g} | "
            f"{values[1]:.8g} | {values[2]:.8g} |"
        )
    elif acm_step_percent.ndim == 2 and acm_step_percent.shape[1] == 6:
        values = np.percentile(acm_step_percent, [16, 50, 84], axis=0)
        for led in range(6):
            lines.append(
                f"| ACM throughput step LED {led + 1} (%) | "
                f"{values[0, led]:.8g} | {values[1, led]:.8g} | "
                f"{values[2, led]:.8g} |"
            )
    else:
        raise ValueError("Unexpected ACM step posterior shape")
    if getattr(args, "fit_complex_q", False) and "complex_q_parameters" in samples:
        q_quantiles = np.percentile(
            np.asarray(samples["complex_q_parameters"], dtype=float),
            [16, 50, 84],
            axis=0,
        )
        lines.extend(
            [
                "",
                "### Complex-q parameters",
                "",
                "| Parameter | 16th percentile | Median | 84th percentile |",
                "|---|---:|---:|---:|",
            ]
        )
        for index, name in enumerate(COMPLEX_Q_PARAMETER_NAMES):
            values = q_quantiles[:, index]
            lines.append(
                f"| {name} | {values[0]:.10g} | {values[1]:.10g} | {values[2]:.10g} |"
            )
    if "late_acm_step_log_offset" in samples:
        late_step_percent = 100.0 * np.expm1(
            np.asarray(samples["late_acm_step_log_offset"])
        )
        values = quantile(late_step_percent)
        lines.append(
            f"| Final-epoch ACM throughput step (%) | {values[0]:.8g} | "
            f"{values[1]:.8g} | {values[2]:.8g} |"
        )
    if "additional_acm_step_log_offset" in samples:
        additional_step_percent = 100.0 * np.expm1(
            np.asarray(samples["additional_acm_step_log_offset"])
        )
        values = quantile(additional_step_percent)
        lines.append(
            f"| Additional repeat-activated ACM throughput step (%) | {values[0]:.8g} | "
            f"{values[1]:.8g} | {values[2]:.8g} |"
        )
    if "tile_mean_rate_nm_day" in samples:
        values = quantile(samples["tile_mean_rate_nm_day"])
        lines.append(
            f"| Selected-tile mean rate distribution (nm/day) | {values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |"
        )
    for layer in range(ar_layers):
        values = quantile(ar_mean[:, layer, 0])
        label = "AR mean thickness scale" if ar_layers == 1 else f"AR layer {layer + 1} mean thickness scale"
        lines.append(f"| {label} | {values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |")
    if args.np_modulation or (
        getattr(args, "fully_np_trajectory", False)
        and not getattr(args, "independent_tile_trajectories", False)
    ):
        for label, key in (
            ("NP smoothness", "np_smoothness"),
        ):
            values = quantile(samples[key])
            lines.append(f"| {label} | {values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |")
        if args.np_modulation:
            values = quantile(samples["np_entropy"])
            lines.append(f"| NP entropy | {values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |")
    if fit_diagnostics is not None:
        lines.extend(
            [
                "",
                "## Posterior median predictive fit",
                "",
                f"- Group-mean RMS: `{fit_diagnostics['group_rms_log']:.10g}` log.",
                f"- Non-reference group-mean RMS: `{fit_diagnostics['non_reference_group_rms_log']:.10g}` log.",
                f"- Count-weighted group-mean RMS: `{fit_diagnostics['count_weighted_group_rms_log']:.10g}` log.",
                f"- All-pixel RMS including within-tile scatter: `{fit_diagnostics['pixel_rms_log']:.10g}` log.",
                f"- Observed/model group-mean correlation: `{fit_diagnostics['correlation']:.10g}`.",
                "",
                "| LED | Group-mean RMS (log) |",
                "|---:|---:|",
            ]
        )
        for led, rms in enumerate(fit_diagnostics["led_rms_log"], start=1):
            lines.append(f"| {led} | {rms:.10g} |")
    ice_trajectory_scope = (
        "each selected tile has its own independent positive node-rate trajectory"
        if getattr(args, "independent_tile_trajectories", False)
        else "each selected tile has its own rate-scaled ice trajectory"
        if getattr(args, "tile_rate_variation", False)
        or getattr(args, "tile_rate_trajectory_variation", False)
        else "one ice trajectory globally"
    )
    if data["n_scas"] > 1:
        model_note = (
            "This all-SCA fit shares one ice trajectory and one complete AR parameter "
            "vector across all selected tiles and SCAs."
            if getattr(args, "shared_ar_parameters_across_tiles", False)
            else (
                f"This all-SCA fit uses {ice_trajectory_scope} and direct independent "
                "AR parameters for every selected SCA/tile; no tile hierarchy."
                if getattr(args, "independent_tile_ar_parameters", False)
                else (
                f"This all-SCA fit uses {ice_trajectory_scope} and hierarchically "
                "shrinks tile AR thickness toward an SCA-specific population; dielectric "
                "n/k are shared across all tiles."
                if args.shared_ar_optical_properties
                else f"This all-SCA fit uses {ice_trajectory_scope} and hierarchically "
                "shrinks tile AR parameters toward an SCA-specific population."
                )
            )
        )
    else:
        model_note = (
            "The complete AR parameter vector is shared across all selected tiles; "
            "there is no tile-level AR variation."
            if getattr(args, "shared_ar_parameters_across_tiles", False)
            else (
                "This fit uses direct independent AR parameters for every selected tile."
                if getattr(args, "independent_tile_ar_parameters", False)
                else (
                "This is a single-tile fit. The ice trajectory and all AR parameters are "
                "directly shared by the selected tile."
                if is_single_tile_data(data)
                else f"This is a hierarchical fit; {ice_trajectory_scope}, and "
                f"the {ar_layers} dielectric AR layer(s) vary by tile under hierarchical shrinkage."
                )
            )
        )
    lines.extend(
        [
            "",
            model_note,
            "Longer chains and tile-resolution sensitivity are required before interpreting "
            "the spatial AR field physically.",
        ]
    )
    (output_dir / "numpyro_summary.md").write_text("\n".join(lines) + "\n")


def summarize_sampler_diagnostics(chain_samples, extra_fields, max_tree_depth):
    """Return compact, durable convergence diagnostics for the fit summary."""
    from numpyro.diagnostics import summary

    # Materialize GPU-backed samples once before NumPyro's diagnostic helpers.
    # This avoids repeated ROCm device transfers while calculating R-hat and ESS.
    chain_samples = {
        name: np.asarray(value) for name, value in chain_samples.items()
    }
    extra_fields = {
        name: np.asarray(value) for name, value in extra_fields.items()
    }
    posterior_summary = summary(chain_samples, group_by_chain=True)
    r_hat = np.concatenate(
        [
            np.asarray(values["r_hat"], dtype=float).reshape(-1)
            for values in posterior_summary.values()
            if "r_hat" in values
        ]
    )
    effective_sample_size = np.concatenate(
        [
            np.asarray(values["n_eff"], dtype=float).reshape(-1)
            for values in posterior_summary.values()
            if "n_eff" in values
        ]
    )
    finite_r_hat = r_hat[np.isfinite(r_hat)]
    finite_effective_sample_size = effective_sample_size[
        np.isfinite(effective_sample_size)
    ]
    num_steps = np.asarray(extra_fields["num_steps"])
    return {
        "divergences": int(np.count_nonzero(extra_fields["diverging"])),
        "mean_accept_probability": float(np.mean(extra_fields["accept_prob"])),
        "max_num_steps": int(np.max(num_steps)),
        "tree_depth_saturated": int(
            np.count_nonzero(num_steps >= 2**max_tree_depth - 1)
        ),
        "max_r_hat": float(np.max(finite_r_hat)) if finite_r_hat.size else np.nan,
        "min_effective_sample_size": (
            float(np.min(finite_effective_sample_size))
            if finite_effective_sample_size.size
            else np.nan
        ),
        "r_hat_gt_1_01": int(np.count_nonzero(finite_r_hat > 1.01)),
        "ess_lt_100": int(np.count_nonzero(finite_effective_sample_size < 100.0)),
        "r_hat_nonfinite": int(np.count_nonzero(~np.isfinite(r_hat))),
        "ess_nonfinite": int(
            np.count_nonzero(~np.isfinite(effective_sample_size))
        ),
    }


def _gpu_safe_fori_collect(
    jax,
    lower,
    upper,
    body_fun,
    init_val,
    transform=lambda value: value,
    progbar=True,
    progress_rate=None,
    return_last_val=False,
    collection_size=None,
    thinning=1,
    **progbar_opts,
):
    """Collect NumPyro states without donating the collection buffer."""
    del progbar, progress_rate, progbar_opts
    if upper < lower or thinning < 1:
        raise ValueError("Invalid sample collection bounds")
    if collection_size is None:
        collection_size = (upper - lower) // thinning

    transformed_init = transform(init_val)
    start_idx = lower + (upper - lower) % thinning
    jnp = jax.numpy

    def initialize_collection(value):
        value = jnp.asarray(value)
        return jnp.zeros((collection_size,) + value.shape, dtype=value.dtype) * value

    collection = jax.tree.map(initialize_collection, transformed_init)

    def loop_body(index, values):
        current, collected = values
        current = body_fun(current)
        sample_index = (index - start_idx) // thinning

        def update(collect_array, sample):
            return jax.lax.cond(
                sample_index >= 0,
                lambda array: array.at[sample_index].set(sample),
                lambda array: array,
                collect_array,
            )

        collected = jax.tree.map(update, collected, transform(current))
        return current, collected

    last_val, collection = jax.lax.fori_loop(
        0,
        upper,
        loop_body,
        (init_val, collection),
    )
    return (collection, last_val) if return_last_val else collection


def install_gpu_safe_numpyro_collection(jax):
    """Use the non-donating collector for GPU-backed NumPyro chains."""
    import numpyro.infer.mcmc as numpyro_mcmc

    def collect(*args, **kwargs):
        return _gpu_safe_fori_collect(jax, *args, **kwargs)

    numpyro_mcmc.fori_collect = collect


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tile-statistics",
        type=Path,
        default=DEFAULT_SINGLE_SCA_TILE_STATISTICS,
    )
    sca_group = parser.add_mutually_exclusive_group()
    sca_group.add_argument(
        "--all-sca",
        dest="all_sca",
        action="store_true",
        help="Fit all completed per-SCA statistics under the all-SCA statistics directory.",
    )
    sca_group.add_argument(
        "--single-sca",
        dest="all_sca",
        action="store_false",
        help="Force the legacy single-SCA fit.",
    )
    parser.add_argument(
        "--all-sca-statistics-dir",
        type=Path,
        default=DEFAULT_ALL_SCA_STATISTICS_DIR,
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SINGLE_SCA_OUTPUT)
    parser.add_argument(
        "--ar-layers",
        type=int,
        choices=tuple(range(1, AR_MAX_LAYERS + 1)),
        default=3,
        help="Number of stacked dielectric AR layers in the TMM coating model.",
    )
    parser.add_argument(
        "--shared-ar-optical-properties",
        action="store_true",
        help="Share each layer's refractive and absorption indices across all tiles; vary thickness hierarchically.",
    )
    parser.add_argument(
        "--shared-ar-parameters-across-tiles",
        action="store_true",
        help=(
            "Use one identical full AR parameter vector for every selected tile, "
            "removing tile-level AR variation."
        ),
    )
    parser.add_argument(
        "--independent-tile-ar-parameters",
        action="store_true",
        help=(
            "Fit a direct AR parameter vector independently for every SCA/tile, "
            "with no shared AR population mean or tile hierarchy."
        ),
    )
    parser.add_argument(
        "--fit-ar-absorption",
        action="store_true",
        help="Fit one absorption index k per AR layer instead of fixing dielectric k to zero.",
    )
    parser.add_argument(
        "--fit-sca-ar-opd",
        action="store_true",
        help=(
            "Fit one zero-centered effective optical-path perturbation in nm per SCA; "
            "the perturbation changes the first AR-layer coherent phase only."
        ),
    )
    parser.add_argument(
        "--sca-ar-opd-prior-sigma-nm",
        type=float,
        default=SCA_AR_OPD_PRIOR_SIGMA_NM,
        help="Normal prior sigma in nm for the fitted per-SCA AR optical-path perturbation.",
    )
    hgcdte_group = parser.add_mutually_exclusive_group()
    hgcdte_group.add_argument(
        "--fit-hgcdte",
        action="store_true",
        help="Fit global wavelength-independent HgCdTe n and k instead of fixing them to the nominal constants.",
    )
    hgcdte_group.add_argument(
        "--use-hgcdte-optical-constants",
        action="store_true",
        help=(
            "Use the fixed 80 K wavelength-dependent HgCdTe n/k sensitivity "
            "model instead of the nominal constant boundary."
        ),
    )
    hgcdte_group.add_argument(
        "--fit-hgcdte-modulation",
        action="store_true",
        help=(
            "Fit global smooth log-space n/k modulation around the fixed "
            "wavelength-dependent HgCdTe baseline."
        ),
    )
    hgcdte_group.add_argument(
        "--absolute-substrate-power",
        action="store_true",
        help=(
            "Fit the absolute SED-weighted power entering the semi-infinite HgCdTe "
            "boundary, with global positive n/k scale nuisance parameters."
        ),
    )
    parser.add_argument(
        "--hgcdte-modulation-prior-sigma-log",
        type=float,
        default=HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
        help="Normal prior sigma for fitted HgCdTe n/k log modulation.",
    )
    parser.add_argument(
        "--absolute-hgcdte-scale-prior-sigma-log",
        type=float,
        default=ABSOLUTE_HGCDTE_SCALE_PRIOR_SIGMA_LOG,
        help="Normal prior sigma for absolute-power HgCdTe log n/k scales.",
    )
    parser.add_argument(
        "--fix-absolute-hgcdte-reference",
        dest="fixed_absolute_hgcdte_scales",
        action="store_true",
        help=(
            "Fix absolute-power HgCdTe n/k log scales to zero, i.e. use the "
            "wavelength-dependent reference optical constants without tile scales."
        ),
    )
    parser.add_argument(
        "--share-absolute-hgcdte-scales",
        dest="shared_absolute_hgcdte_scales",
        action="store_true",
        help=(
            "Fit one free absolute-power HgCdTe n/k log-scale pair per SCA, "
            "shared across all selected tiles."
        ),
    )
    parser.add_argument(
        "--finite-hgcdte-detector",
        action="store_true",
        help=(
            "Propagate through a finite HgCdTe active layer and include the "
            "fixed-index backside boundary instead of using a semi-infinite boundary."
        ),
    )
    parser.add_argument(
        "--detector-thickness-nm",
        type=float,
        default=DETECTOR_THICKNESS_DEFAULT_NM,
        help="Fixed finite HgCdTe active-layer thickness in nm.",
    )
    parser.add_argument(
        "--fit-detector-thickness",
        action="store_true",
        help=(
            "Fit one positive log detector thickness per selected SCA/tile; "
            "this also enables --finite-hgcdte-detector."
        ),
    )
    parser.add_argument(
        "--detector-thickness-prior-sigma-log",
        type=float,
        default=DETECTOR_THICKNESS_PRIOR_SIGMA_LOG,
        help="Normal prior sigma for log finite-detector thickness.",
    )
    parser.add_argument(
        "--detector-thickness-neighbor-sigma-log",
        type=float,
        default=DETECTOR_THICKNESS_NEIGHBOR_SIGMA_LOG,
        help="Gaussian neighbor sigma for log finite-detector thickness differences.",
    )
    parser.add_argument(
        "--detector-backside-index",
        type=float,
        default=1.0,
        help="Real refractive index of the medium behind the finite HgCdTe layer.",
    )
    parser.add_argument(
        "--fit-conductive-layer",
        action="store_true",
        help=(
            "Fit a global thin Drude-like conductive layer between the AR stack "
            "and HgCdTe."
        ),
    )
    parser.add_argument(
        "--use-asw-optical-constants",
        action="store_true",
        help=(
            "Use wavelength-dependent published 100 K amorphous-H2O n/k "
            "constants for the ice layer instead of n=1.25+0j."
        ),
    )
    parser.add_argument(
        "--ice-optical-constants-source",
        default=ASW_OPTICAL_CONSTANTS_URL,
        help=(
            "Local path or URL for the ice n/k table used with "
            "--use-asw-optical-constants."
        ),
    )
    parser.add_argument(
        "--fit-ice-optical-modulation",
        action="store_true",
        help=(
            "Fit shared smooth log-space n/k modulation around the published ASW "
            "baseline; requires --use-asw-optical-constants."
        ),
    )
    parser.add_argument(
        "--ice-modulation-prior-sigma-log",
        type=float,
        default=ICE_MODULATION_PRIOR_SIGMA_LOG,
        help="Normal prior sigma for fitted ice n/k log modulation.",
    )
    parser.add_argument(
        "--ice-n-modulation-prior-sigma-log",
        type=float,
        default=None,
        help=(
            "Optional separate Normal prior sigma for the fitted ice n log modulation; "
            "the shared --ice-modulation-prior-sigma-log value is used by default."
        ),
    )
    parser.add_argument(
        "--incidence-cone-half-angle-deg",
        type=float,
        default=0.0,
        help=(
            "Average the unpolarized TE/TM response over a cone uniform in "
            "solid angle; zero retains the exact normal-incidence model."
        ),
    )
    parser.add_argument(
        "--incidence-pupil-half-angle-deg",
        type=float,
        default=0.0,
        help=(
            "Average over a uniformly illuminated circular pupil with this "
            "marginal-ray half-angle. Roman's documented f/8 beam uses about 3.576 degrees."
        ),
    )
    parser.add_argument(
        "--incidence-chief-ray-x-deg",
        type=float,
        default=0.0,
        help="Global chief-ray x tilt relative to the detector normal.",
    )
    parser.add_argument(
        "--incidence-chief-ray-y-deg",
        type=float,
        default=0.0,
        help="Global chief-ray y tilt relative to the detector normal.",
    )
    parser.add_argument(
        "--incidence-chief-ray-map",
        type=Path,
        help=(
            "CSV with detector,tile_id,chief_ray_x_deg,chief_ray_y_deg for every "
            "selected optical row; overrides the global chief-ray tilt."
        ),
    )
    parser.add_argument(
        "--fit-incidence-angle",
        action="store_true",
        help=(
            "Fit one global non-negative chief-ray incidence angle relative to normal; "
            f"the hard upper bound is {INCIDENCE_ANGLE_MAX_DEG:g} degrees."
        ),
    )
    parser.add_argument(
        "--incidence-angle-prior-sigma-deg",
        type=float,
        default=INCIDENCE_ANGLE_PRIOR_SIGMA_DEG,
        help=(
            "Half-normal-like truncated Normal prior sigma for the fitted incidence angle; "
            "the lower bound is zero."
        ),
    )
    parser.add_argument(
        "--incidence-angle-prior-uniform",
        action="store_true",
        help="Use a flat prior over the hard 0 to 15 degree incidence-angle interval.",
    )
    parser.add_argument(
        "--incidence-angle-init-deg",
        type=float,
        default=INCIDENCE_ANGLE_INIT_DEG,
        help="Positive MAP/NUTS initialization for the fitted incidence angle.",
    )
    acm_step_mode = parser.add_mutually_exclusive_group()
    acm_step_mode.add_argument(
        "--led-dependent-acm-step",
        action="store_true",
        help="Fit an independent ACM throughput step for each LED (default).",
    )
    acm_step_mode.add_argument(
        "--common-acm-step",
        dest="led_dependent_acm_step",
        action="store_false",
        help="Fit one existing ACM step shared across all LEDs.",
    )
    parser.add_argument(
        "--acm-step-prior-sigma",
        type=float,
        default=0.05,
        help="Normal prior sigma for ACM log throughput steps.",
    )
    parser.add_argument(
        "--fit-late-acm-step",
        action="store_true",
        help=(
            "Fit a second ACM step, shared across LEDs, that turns on at the "
            "final observed repeat epoch."
        ),
    )
    parser.add_argument(
        "--late-acm-step-prior-sigma",
        type=float,
        default=0.02,
        help="Normal prior sigma for the final-epoch ACM log step.",
    )
    parser.add_argument(
        "--fit-additional-acm-step",
        action="store_true",
        help=(
            "Fit one additional common ACM step activated at the repeat supplied by "
            "--additional-acm-step-repeat."
        ),
    )
    parser.add_argument(
        "--additional-acm-step-repeat",
        type=int,
        default=None,
        help=(
            "First repeat affected by the additional ACM step; for the WFI01 "
            "post-reset event this is repeat 39."
        ),
    )
    parser.add_argument(
        "--pre-reset-acm-step-repeat",
        type=int,
        default=None,
        help=(
            "First repeat affected by a pre-reset ACM correction in a fixed "
            "correction table; for WFI01 this is repeat 35 near day 11.3."
        ),
    )
    parser.add_argument(
        "--additional-acm-step-prior-sigma",
        type=float,
        default=0.02,
        help="Normal prior sigma for the additional ACM log step.",
    )
    parser.add_argument(
        "--fixed-corrections",
        type=Path,
        default=None,
        help=(
            "Apply a fixed common gain correction and fixed shared ACM steps from "
            "this correction CSV; ACM and early-time gain terms are then not fitted."
        ),
    )
    parser.add_argument(
        "--fit-qe-rbf",
        action="store_true",
        help=(
            "Fit smooth wavelength-dependent QE log-modulation nodes."
        ),
    )
    parser.add_argument(
        "--qe-rbf-node-wavelengths-nm",
        type=float,
        nargs="+",
        default=QE_RBF_NODE_WAVELENGTHS_NM.tolist(),
        help="Strictly increasing QE RBF node wavelengths in nm.",
    )
    parser.add_argument(
        "--qe-rbf-width-nm",
        type=float,
        default=QE_RBF_WIDTH_NM,
        help="Gaussian QE RBF width in nm.",
    )
    parser.add_argument(
        "--qe-rbf-prior-sigma-log",
        type=float,
        default=QE_RBF_PRIOR_SIGMA_LOG,
        help="Normal prior sigma for QE RBF log multipliers.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Tiles per SCA to fit; default is all available tiles.",
    )
    parser.add_argument(
        "--min-observation-day",
        type=float,
        default=None,
        help=(
            "Discard complete observation groups before this day; the first "
            "retained group becomes the reference time."
        ),
    )
    parser.add_argument(
        "--early-time-scale-cutoff-day",
        type=float,
        default=None,
        help=(
            "Fit one positive relative raw-count scale per repeat epoch, shared "
            "across LEDs, for epochs at or before this day; the first epoch is "
            "the unit reference."
        ),
    )
    parser.add_argument(
        "--early-time-scale-prior-sigma",
        type=float,
        default=EARLY_TIME_SCALE_PRIOR_SIGMA,
        help=(
            "Normal prior sigma for each early-time log-relative count scale "
            "(default: 0.05)."
        ),
    )
    parser.add_argument("--sed-mode", choices=("signed", "clipped", "core"), default="core")
    parser.add_argument(
        "--sed-weighting",
        choices=("photon", "energy"),
        default="photon",
        help=(
            "Use wavelength photon weighting for the published relative energy SED, "
            "or treat SED values as photon-rate weights for a sensitivity check."
        ),
    )
    parser.add_argument(
        "--wavelength-stride",
        type=int,
        default=1,
        help="Sample the QE-supported wavelength grid every N nm; 1 is the production default.",
    )
    parser.add_argument(
        "--wavelength-step-nm",
        type=float,
        default=None,
        help="Use a uniform base grid at this spacing; omit to evaluate exact SED nodes.",
    )
    parser.add_argument(
        "--prediction-row-chunk-size",
        type=int,
        default=None,
        help=(
            "Number of SCA/tile rows evaluated per likelihood scan chunk; smaller "
            "values reduce JAX compilation memory for large hierarchical fits. "
            "The default is 8 for tile-rate or finite-detector fits and 64 otherwise."
        ),
    )
    parser.add_argument("--log-noise-sigma", type=float, default=0.005)
    parser.add_argument(
        "--trajectory-rate-prior-mean-nm-day",
        type=float,
        default=None,
        help=(
            "Optional center of the unmodulated instantaneous-rate prior at the "
            "first observed time, in nm/day. If omitted, retain the historical "
            "prior center."
        ),
    )
    parser.add_argument(
        "--trajectory-rate-prior-sigma-log",
        type=float,
        default=0.75,
        help="Normal prior sigma for log trajectory scale; 1.0 is deliberately broad.",
    )
    parser.add_argument(
        "--fixed-initial-thickness-nm",
        type=float,
        default=None,
        help="Fix ice thickness at the first observed/reference time instead of fitting it.",
    )
    parser.add_argument(
        "--tile-initial-thickness-lognormal",
        action="store_true",
        help=(
            "Fit an independent positive lognormal initial-thickness value for every "
            "selected SCA/tile instead of one shared initial thickness."
        ),
    )
    parser.add_argument(
        "--tile-initial-thickness-log-median-nm",
        type=float,
        default=10.0,
        help="Median of the per-tile lognormal initial-thickness prior in nm.",
    )
    parser.add_argument(
        "--tile-initial-thickness-log-sigma",
        type=float,
        default=1.5,
        help="Log-space sigma of the per-tile lognormal initial-thickness prior.",
    )
    parser.add_argument(
        "--ice-reset-day",
        type=float,
        default=None,
        help=(
            "First observation day in a post-reset ice segment. A second initial "
            "thickness baseline is fitted after this day while sharing the rate trajectory."
        ),
    )
    parser.add_argument(
        "--fit-post-reset-tile-log-offset",
        action="store_true",
        help=(
            "Fit one zero-centered grey log correction per SCA/tile for groups on or "
            "after --ice-reset-day; positive values are additional corrections to the "
            "observed log ratios."
        ),
    )
    parser.add_argument(
        "--post-reset-tile-log-offset-prior-sigma",
        type=float,
        default=POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA,
        help="Normal prior sigma for each post-reset per-tile grey log correction.",
    )
    parser.add_argument(
        "--fit-post-reset-ice-index",
        action="store_true",
        help=(
            "Fit one shared real ice refractive index for groups on or after "
            "--ice-reset-day; pre-reset groups retain n=1.25."
        ),
    )
    parser.add_argument(
        "--post-reset-ice-index-prior-mean",
        type=float,
        default=POST_RESET_ICE_INDEX_PRIOR_MEAN,
        help="Prior mean for the shared post-reset ice refractive index.",
    )
    parser.add_argument(
        "--post-reset-ice-index-prior-sigma",
        type=float,
        default=POST_RESET_ICE_INDEX_PRIOR_SIGMA,
        help="Prior sigma for the shared post-reset ice refractive index.",
    )
    parser.add_argument(
        "--post-reset-ice-index-lower",
        type=float,
        default=POST_RESET_ICE_INDEX_LOWER,
        help="Lower bound for the shared post-reset ice refractive index.",
    )
    parser.add_argument(
        "--post-reset-ice-index-upper",
        type=float,
        default=POST_RESET_ICE_INDEX_UPPER,
        help="Upper bound for the shared post-reset ice refractive index.",
    )
    parser.add_argument(
        "--fit-pre-post-ice-index",
        action="store_true",
        help=(
            "Fit separate pre/post-reset complex ice indices, with independent real n "
            "and nonnegative imaginary k parameters; requires --ice-reset-day."
        ),
    )
    parser.add_argument(
        "--ice-index-prior-mean",
        type=float,
        default=ICE_INDEX_PRIOR_MEAN,
        help="Prior mean for the real part of each fitted ice index.",
    )
    parser.add_argument(
        "--ice-index-prior-sigma",
        type=float,
        default=ICE_INDEX_PRIOR_SIGMA,
        help="Prior sigma for the real part of each fitted ice index.",
    )
    parser.add_argument(
        "--ice-index-lower",
        type=float,
        default=ICE_INDEX_LOWER,
        help="Lower bound for the real part of each fitted ice index.",
    )
    parser.add_argument(
        "--ice-index-upper",
        type=float,
        default=ICE_INDEX_UPPER,
        help="Upper bound for the real part of each fitted ice index.",
    )
    parser.add_argument(
        "--ice-extinction-prior-sigma",
        type=float,
        default=ICE_EXTINCTION_PRIOR_SIGMA,
        help="Half-normal/truncated-normal scale for each fitted imaginary index part.",
    )
    parser.add_argument(
        "--ice-extinction-upper",
        type=float,
        default=ICE_EXTINCTION_UPPER,
        help="Upper bound for each fitted imaginary ice-index part.",
    )
    parser.add_argument(
        "--initial-thickness-prior-mean-nm",
        type=float,
        default=INITIAL_THICKNESS_PRIOR_MEAN_NM,
        help="Truncated-Normal prior mean for fitted initial ice thickness in nm.",
    )
    parser.add_argument(
        "--initial-thickness-prior-sigma-nm",
        type=float,
        default=INITIAL_THICKNESS_PRIOR_SIGMA_NM,
        help="Truncated-Normal prior sigma for fitted initial ice thickness in nm.",
    )
    parser.add_argument(
        "--initial-thickness-prior-uniform",
        action="store_true",
        help="Use a flat prior over the supported 0 to 50 nm initial-thickness interval.",
    )
    parser.add_argument(
        "--initial-thickness-init-nm",
        type=float,
        default=INITIAL_THICKNESS_INIT_NM,
        help="MAP/sampler initialization for fitted initial ice thickness in nm.",
    )
    parser.add_argument(
        "--nuts-initial-thickness-nm",
        type=float,
        default=NUTS_INITIAL_THICKNESS_INIT_NM,
        help=(
            "Initial thickness used only when NUTS has no MAP warm start; it does "
            "not change the fitted prior."
        ),
    )
    parser.add_argument(
        "--nuts-init-jitter",
        type=float,
        default=NUTS_INIT_JITTER,
        help=(
            "Per-chain standard deviation for unconstrained NUTS initialization "
            "around the complete MAP warm start; zero uses the exact MAP state."
        ),
    )
    parser.add_argument(
        "--pivot-initial-thickness",
        action="store_true",
        help=(
            "Sample an unconstrained thickness coordinate at the central trajectory "
            "pivot while retaining the exact truncated prior on initial thickness."
        ),
    )
    parser.add_argument(
        "--tile-rate-variation",
        action="store_true",
        help=(
            "Fit a log-rate offset for each selected SCA/tile, with neighboring "
            "tile offsets regularized toward one another."
        ),
    )
    parser.add_argument(
        "--tile-rate-trajectory-variation",
        action="store_true",
        help=(
            "Fit time-varying log-rate deviations at each observation epoch for every "
            "tile, centered across tiles and regularized in time and over neighbors."
        ),
    )
    parser.add_argument(
        "--independent-tile-trajectories",
        action="store_true",
        help=(
            "Fit a separate positive log-rate node trajectory for every SCA/tile; "
            "do not introduce a global trajectory or centered tile deviations."
        ),
    )
    parser.add_argument(
        "--tile-rate-trajectory-smoothness-weight",
        type=float,
        default=0.001,
        help="Curvature regularization weight for tile-specific log-rate trajectories.",
    )
    parser.add_argument(
        "--tile-rate-prior-sigma-log",
        type=float,
        default=TRAJECTORY_TILE_RATE_PRIOR_SIGMA_LOG,
        help="Prior sigma in log rate for each SCA/tile offset.",
    )
    parser.add_argument(
        "--tile-rate-neighbor-sigma-log",
        type=float,
        default=TRAJECTORY_TILE_RATE_NEIGHBOR_SIGMA_LOG,
        help="Prior sigma in log rate for neighboring tile differences.",
    )
    parser.add_argument(
        "--no-tile-rate-neighbor-smoothness",
        dest="tile_rate_neighbor_smoothness",
        action="store_false",
        help="Disable the neighboring-tile log-rate smoothness factor.",
    )
    parser.add_argument(
        "--likelihood-weight",
        type=float,
        default=None,
        help=(
            "Pixel-SSE weight before averaging across selected tiles. By default "
            "use the inverse median valid pixel count, making the aggregate "
            "tile/group contribution approximately one effective datum."
        ),
    )
    parser.add_argument(
        "--ar-mean-prior-sigma",
        type=float,
        default=AR_MEAN_PRIOR_SIGMA,
        help="Normal prior sigma for global AR raw parameters; larger values relax the AR reference center.",
    )
    parser.add_argument(
        "--standardize-ar-mean",
        action="store_true",
        help=(
            "Sample global AR mean coordinates as standard-normal latents while "
            "retaining the same raw-space prior and physical bounds."
        ),
    )
    parser.add_argument(
        "--rotate-ar-coordinates",
        action="store_true",
        help=(
            "Apply an orthogonal thickness/index rotation to standardized AR "
            "coordinates while retaining the full parameter dimension and prior."
        ),
    )
    parser.add_argument(
        "--ar-prior-from-map-start",
        action="store_true",
        help="Center the global AR prior on the physical MAP initialization instead of the historical reference.",
    )
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable NumPyro's live sampler progress display.",
    )
    chain_group = parser.add_mutually_exclusive_group()
    chain_group.add_argument(
        "--parallel-chains",
        action="store_true",
        help="Run chains with pmap on separate host devices.",
    )
    chain_group.add_argument(
        "--vectorized-chains",
        action="store_true",
        help="Run chains as one batched JAX computation on the available device.",
    )
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--target-accept-prob", type=float, default=0.85)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument(
        "--dense-mass",
        action="store_true",
        help="Adapt a dense NUTS mass matrix; useful for the correlated single-tile optical fit.",
    )
    parser.add_argument(
        "--dense-mass-global",
        action="store_true",
        help=(
            "Adapt a dense mass matrix only for global latent sites and retain diagonal "
            "mass adaptation for tile-local sites."
        ),
    )
    parser.add_argument(
        "--map-steps",
        type=int,
        default=0,
        help="Run this many full-model AutoDelta MAP steps before NUTS (default: disabled).",
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Write a deterministic MAP evaluation after --map-steps instead of running NUTS.",
    )
    parser.add_argument(
        "--map-optimizer",
        choices=("jax-lbfgsb", "lbfgs", "least-squares", "trust-exact", "adam"),
        default="lbfgs",
        help=(
            "Optimizer for --map-steps; SciPy L-BFGS-B is the default, with "
            "JAX-native bounded L-BFGS-B, exact-Hessian trust-region polishing, "
            "trust-region nonlinear least-squares, and Adam available as alternatives."
        ),
    )
    parser.add_argument(
        "--map-tolerance",
        type=float,
        default=1.0e-9,
        help="Projected-gradient tolerance for the selected MAP optimizer.",
    )
    parser.add_argument(
        "--map-ftol",
        type=float,
        default=0.0,
        help=(
            "Relative function-reduction tolerance for SciPy L-BFGS-B; zero "
            "disables this early stopping criterion."
        ),
    )
    parser.add_argument(
        "--map-parameter-scaling",
        choices=("prior", "none"),
        default="prior",
        help=(
            "Optimize SciPy L-BFGS-B in prior-width standardized coordinates "
            "(default), or retain raw sample-site coordinates."
        ),
    )
    parser.add_argument(
        "--map-learning-rate",
        type=float,
        default=0.01,
        help="Adam learning rate for optional MAP initialization.",
    )
    parser.add_argument(
        "--map-trajectory-rate-nm-day",
        type=float,
        default=None,
        help=(
            "MAP-only initial unmodulated instantaneous rate at the first observed "
            "time, in nm/day. This changes the optimizer start, not the posterior "
            "prior."
        ),
    )
    parser.add_argument(
        "--map-init-max-tiles",
        type=int,
        default=None,
        help=(
            "Run the MAP initializer first on this many tiles per SCA, then "
            "transfer compatible sites to the full selected-tile fit."
        ),
    )
    parser.add_argument(
        "--map-init-from",
        type=Path,
        default=None,
        help=(
            "Warm-start MAP sample sites from a compatible map-only posterior_samples.npz "
            "or its output directory; tile-statistics fingerprints must match unless "
            "--map-init-allow-tile-subset is enabled."
        ),
    )
    parser.add_argument(
        "--map-init-allow-tile-subset",
        action="store_true",
        help=(
            "Allow --map-init-from to load a compatible map-only fit made from a "
            "strict subset of the target tiles."
        ),
    )
    parser.add_argument(
        "--map-init-broadcast-tile-values",
        action="store_true",
        help=(
            "When loading a one-tile MAP subset, initialize every target tile-local "
            "site from that source tile."
        ),
    )
    parser.add_argument(
        "--map-design-wavelength-nm",
        type=float,
        default=None,
        help=(
            "Use a quarter-wave, equal-log-index optical stack only for MAP "
            "initialization; this does not constrain posterior sampling."
        ),
    )
    map_middle_layer_group = parser.add_mutually_exclusive_group()
    map_middle_layer_group.add_argument(
        "--map-middle-layer-eighth-wave",
        action="store_true",
        help=(
            "For a three-layer MAP start, use lambda/4, lambda/8, lambda/4 "
            "optical thicknesses instead of three quarter-wave layers."
        ),
    )
    map_middle_layer_group.add_argument(
        "--map-middle-layer-half-wave",
        action="store_true",
        help=(
            "For a three-layer MAP start, use lambda/4, lambda/2, lambda/4 "
            "optical thicknesses instead of three quarter-wave layers."
        ),
    )
    parser.add_argument(
        "--ar-sca-sigma-prior",
        type=float,
        default=AR_SCA_SIGMA_PRIOR,
        help="HalfNormal scale for SCA-to-SCA AR offsets in unconstrained parameter space.",
    )
    parser.add_argument(
        "--ar-tile-sigma-prior",
        type=float,
        default=AR_TILE_SIGMA_PRIOR,
        help="HalfNormal scale for tile AR offsets around each SCA.",
    )
    trajectory_model_group = parser.add_mutually_exclusive_group()
    trajectory_model_group.add_argument(
        "--np-modulation",
        action="store_true",
        help="Use entropy-regularized observation-epoch-node modulation of the square-root rate.",
    )
    trajectory_model_group.add_argument(
        "--constant-rate-modulation",
        action="store_true",
        help=(
            "Use a constant positive rate with observation-epoch-node modulation; "
            "apply relative-entropy regularization without rate smoothness."
        ),
    )
    trajectory_model_group.add_argument(
        "--constant-rate-trajectory",
        action="store_true",
        help="Fit one free positive instantaneous rate with no trajectory nodes.",
    )
    trajectory_model_group.add_argument(
        "--fixed-trajectory-rate-nm-day",
        type=float,
        default=None,
        help="Fix the positive instantaneous rate and do not sample a rate latent.",
    )
    trajectory_model_group.add_argument(
        "--fully-np-trajectory",
        action="store_true",
        help=(
            "Fit a positive instantaneous rate independently at every observation-epoch node, "
            "with the existing curvature smoothness regularization, instead of using "
            "the square-root trajectory (enabled by default)."
        ),
    )
    parser.add_argument(
        "--whiten-np-trajectory",
        action="store_true",
        help=(
            "Whiten the fully non-parametric log-rate nodes using the exact "
            "independent-plus-curvature Gaussian prior; all node degrees of freedom remain free."
        ),
    )
    parser.add_argument(
        "--rotate-np-likelihood",
        action="store_true",
        help=(
            "Apply a full-rank orthogonal rotation to whitened log-rate latents "
            "using the conditional posterior Hessian at the initialization point."
        ),
    )
    parser.add_argument("--np-entropy-weight", type=float, default=NP_ENTROPY_WEIGHT)
    parser.add_argument(
        "--np-smoothness-weight",
        type=float,
        default=NP_SMOOTHNESS_WEIGHT,
    )
    parser.add_argument(
        "--constant-rate-smoothness-weight",
        type=float,
        default=0.0,
        help=(
            "Optional time-aware log-rate smoothness weight for "
            "--constant-rate-modulation; zero keeps entropy-only regularization."
        ),
    )
    parser.add_argument(
        "--np-max-node-multiplier",
        type=float,
        default=NP_MAX_NODE_MULTIPLIER,
    )
    parser.set_defaults(
        all_sca=None,
        fully_np_trajectory=True,
        led_dependent_acm_step=True,
        tile_rate_neighbor_smoothness=True,
    )
    args = parser.parse_args([])
    if _PRODUCTION_STATISTICS_PATH is None:
        raise RuntimeError("Production statistics path was not selected")
    detector = _PRODUCTION_STATISTICS_PATH.name.split("_", 1)[0].upper()
    args.__dict__.update(
        {
            "all_sca": False,
            "tile_statistics": _PRODUCTION_STATISTICS_PATH,
            "output_dir": PRODUCTION_OUTPUT_ROOT / detector.lower(),
            "ar_layers": 3,
            "absolute_substrate_power": True,
            "use_hgcdte_optical_constants": True,
            "shared_absolute_hgcdte_scales": PRODUCTION_SHARED_HGCDTE_SCALES,
            "fixed_absolute_hgcdte_scales": False,
            "sed_mode": "core",
            "sed_weighting": "energy",
            "wavelength_stride": 1,
            "wavelength_step_nm": None,
            "prediction_row_chunk_size": 64,
            "log_noise_sigma": 0.005,
            "corrections_already_applied": True,
            "fixed_corrections": None,
            "fit_late_acm_step": False,
            "fit_additional_acm_step": False,
            "additional_acm_step_repeat": None,
            "pre_reset_acm_step_repeat": None,
            "early_time_scale_cutoff_day": None,
            "tile_initial_thickness_lognormal": True,
            "tile_initial_thickness_log_median_nm": 10.0,
            "tile_initial_thickness_log_sigma": 1.5,
            "fixed_initial_thickness_nm": None,
            "initial_thickness_prior_uniform": False,
            "pivot_initial_thickness": False,
            "ice_reset_day": PRODUCTION_RESET_DAY,
            "fit_post_reset_tile_log_offset": True,
            "post_reset_tile_log_offset_prior_sigma": 0.1,
            "fit_post_reset_ice_index": False,
            "fit_pre_post_ice_index": False,
            "ice_index_prior_mean": 1.25,
            "ice_index_prior_sigma": 0.03,
            "ice_index_lower": 1.15,
            "ice_index_upper": 1.40,
            "ice_extinction_prior_sigma": 0.005,
            "ice_extinction_upper": 0.05,
            "trajectory_rate_prior_mean_nm_day": 5.0,
            "fixed_trajectory_rate_nm_day": None,
            "trajectory_rate_prior_sigma_log": 1.5,
            "tile_rate_variation": True,
            "tile_rate_trajectory_variation": True,
            "independent_tile_trajectories": False,
            "tile_rate_trajectory_smoothness_weight": (
                PRODUCTION_TILE_RATE_TRAJECTORY_SMOOTHNESS_WEIGHT
            ),
            "tile_rate_prior_sigma_log": PRODUCTION_TILE_RATE_PRIOR_SIGMA_LOG,
            "tile_rate_neighbor_sigma_log": 0.05,
            "tile_rate_neighbor_smoothness": False,
            "ar_mean_prior_sigma": 10.0,
            "ar_sca_sigma_prior": 1.0,
            "ar_tile_sigma_prior": 1.0,
            "fixed_ar_indices": None,
            "ordered_ar_indices": True,
            "standardize_ar_mean": False,
            "rotate_ar_coordinates": False,
            "ar_prior_from_map_start": False,
            "fully_np_trajectory": True,
            "np_modulation": False,
            "constant_rate_modulation": False,
            "constant_rate_trajectory": False,
            "whiten_np_trajectory": True,
            "rotate_np_likelihood": False,
            "map_trajectory_rate_nm_day": 10.2,
            "np_smoothness_weight": PRODUCTION_NP_SMOOTHNESS_WEIGHT,
            "map_parameter_scaling": "prior",
            "map_optimizer": "lbfgs",
            "map_steps": 10000,
            "map_only": True,
            "map_tolerance": 1.0e-9,
            "map_ftol": 0.0,
            "map_init_from": None,
            "map_init_allow_tile_subset": False,
            "map_init_broadcast_tile_values": False,
            "map_init_max_tiles": None,
            "map_design_wavelength_nm": None,
            "map_middle_layer_eighth_wave": False,
            "map_middle_layer_half_wave": False,
            "fit_incidence_angle": True,
            "incidence_angle_prior_uniform": True,
            "incidence_angle_init_deg": 7.5,
            "incidence_cone_half_angle_deg": 0.0,
            "incidence_pupil_half_angle_deg": 0.0,
            "incidence_chief_ray_x_deg": 0.0,
            "incidence_chief_ray_y_deg": 0.0,
            "incidence_chief_ray_map": None,
            "fit_ar_absorption": False,
            "fit_sca_ar_opd": False,
            "shared_ar_optical_properties": False,
            "shared_ar_parameters_across_tiles": False,
            "independent_tile_ar_parameters": False,
            "fit_hgcdte": False,
            "fit_hgcdte_modulation": False,
            "fit_conductive_layer": False,
            "finite_hgcdte_detector": False,
            "fit_detector_thickness": False,
            "fit_qe_rbf": False,
            "use_asw_optical_constants": False,
            "fit_ice_optical_modulation": False,
            "no_progress_bar": True,
            "seed": PRODUCTION_SEED,
            "warmup": 300,
            "samples": 500,
            "chains": 2,
            "parallel_chains": False,
            "vectorized_chains": False,
            "dense_mass": False,
            "dense_mass_global": False,
        }
    )
    return args


def fixed_rate_warmup_args(args):
    """Return the explicit 500-step fixed-rate initialization configuration."""
    warmup = argparse.Namespace(**vars(args))
    warmup.fixed_trajectory_rate_nm_day = WARMUP_FIXED_RATE_NM_DAY
    warmup.trajectory_rate_prior_mean_nm_day = WARMUP_FIXED_RATE_NM_DAY
    warmup.fully_np_trajectory = False
    warmup.whiten_np_trajectory = False
    warmup.np_modulation = False
    warmup.constant_rate_modulation = False
    warmup.constant_rate_trajectory = False
    warmup.tile_rate_variation = False
    warmup.tile_rate_trajectory_variation = False
    warmup.independent_tile_trajectories = False
    warmup.map_trajectory_rate_nm_day = None
    warmup.fixed_absolute_hgcdte_scales = getattr(
        args,
        "fixed_absolute_hgcdte_scales",
        WARMUP_FIXED_HGCDTE_SCALES,
    )
    warmup.shared_absolute_hgcdte_scales = (
        False if warmup.fixed_absolute_hgcdte_scales else WARMUP_SHARED_HGCDTE_SCALES
    )
    warmup.fixed_ar_indices = WARMUP_FIXED_AR_INDICES.copy()
    warmup.ordered_ar_indices = False
    warmup.map_steps = WARMUP_MAP_STEPS
    return warmup


def _fit_one(configured_args=None):
    run_start = time.perf_counter()
    args = parse_args() if configured_args is None else configured_args
    corrections_applied = bool(
        getattr(args, "corrections_already_applied", False)
        or args.fixed_corrections is not None
    )
    if args.absolute_substrate_power:
        args.use_hgcdte_optical_constants = True
    if args.np_modulation or args.constant_rate_modulation:
        args.fully_np_trajectory = False
    if args.whiten_np_trajectory and not args.fully_np_trajectory:
        raise ValueError("--whiten-np-trajectory requires --fully-np-trajectory")
    if args.rotate_np_likelihood and not args.whiten_np_trajectory:
        raise ValueError("--rotate-np-likelihood requires --whiten-np-trajectory")
    if args.pivot_initial_thickness and not args.fully_np_trajectory:
        raise ValueError("--pivot-initial-thickness requires --fully-np-trajectory")
    if args.rotate_ar_coordinates and not args.standardize_ar_mean:
        raise ValueError("--rotate-ar-coordinates requires --standardize-ar-mean")
    if (
        not np.isfinite(args.incidence_angle_prior_sigma_deg)
        or args.incidence_angle_prior_sigma_deg <= 0.0
    ):
        raise ValueError("incidence-angle-prior-sigma-deg must be positive and finite")
    if args.fit_incidence_angle:
        if not (0.0 < args.incidence_angle_init_deg < INCIDENCE_ANGLE_MAX_DEG):
            raise ValueError(
                f"incidence-angle-init-deg must be in (0, {INCIDENCE_ANGLE_MAX_DEG:g})"
            )
        if args.incidence_cone_half_angle_deg != 0.0:
            raise ValueError(
                "--fit-incidence-angle cannot be combined with "
                "--incidence-cone-half-angle-deg"
            )
        if (
            args.incidence_chief_ray_map is not None
            or args.incidence_chief_ray_x_deg != 0.0
            or args.incidence_chief_ray_y_deg != 0.0
        ):
            raise ValueError(
                "--fit-incidence-angle cannot be combined with fixed chief-ray geometry"
            )
    if args.dense_mass and args.dense_mass_global:
        raise ValueError("--dense-mass and --dense-mass-global are mutually exclusive")
    (
        args.qe_rbf_node_wavelengths_nm,
        args.qe_rbf_width_nm,
        args.qe_rbf_prior_sigma_log,
    ) = qe_rbf_configuration(
        args.qe_rbf_node_wavelengths_nm,
        args.qe_rbf_width_nm,
        args.qe_rbf_prior_sigma_log,
    )
    if args.log_noise_sigma <= 0.0 or (
        args.likelihood_weight is not None and args.likelihood_weight <= 0.0
    ):
        raise ValueError("Noise sigma and an explicit likelihood weight must be positive")
    if args.ar_mean_prior_sigma <= 0.0:
        raise ValueError("ar-mean-prior-sigma must be positive")
    if (
        args.sca_ar_opd_prior_sigma_nm <= 0.0
        or not np.isfinite(args.sca_ar_opd_prior_sigma_nm)
    ):
        raise ValueError("sca-ar-opd-prior-sigma-nm must be positive and finite")
    if (
        args.hgcdte_modulation_prior_sigma_log <= 0.0
        or not np.isfinite(args.hgcdte_modulation_prior_sigma_log)
    ):
        raise ValueError(
            "hgcdte-modulation-prior-sigma-log must be positive and finite"
        )
    if (
        args.ice_modulation_prior_sigma_log <= 0.0
        or not np.isfinite(args.ice_modulation_prior_sigma_log)
    ):
        raise ValueError(
            "ice-modulation-prior-sigma-log must be positive and finite"
        )
    if args.ice_n_modulation_prior_sigma_log is not None and (
        args.ice_n_modulation_prior_sigma_log <= 0.0
        or not np.isfinite(args.ice_n_modulation_prior_sigma_log)
    ):
        raise ValueError(
            "ice-n-modulation-prior-sigma-log must be positive and finite"
        )
    if args.fit_ice_optical_modulation and not args.use_asw_optical_constants:
        raise ValueError(
            "--fit-ice-optical-modulation requires --use-asw-optical-constants"
        )
    if (
        args.absolute_hgcdte_scale_prior_sigma_log <= 0.0
        or not np.isfinite(args.absolute_hgcdte_scale_prior_sigma_log)
    ):
        raise ValueError(
            "absolute-hgcdte-scale-prior-sigma-log must be positive and finite"
        )
    if not np.isfinite(args.detector_thickness_nm) or args.detector_thickness_nm <= 0.0:
        raise ValueError("detector-thickness-nm must be positive and finite")
    if (
        not np.isfinite(args.detector_thickness_prior_sigma_log)
        or args.detector_thickness_prior_sigma_log <= 0.0
    ):
        raise ValueError(
            "detector-thickness-prior-sigma-log must be positive and finite"
        )
    if (
        not np.isfinite(args.detector_thickness_neighbor_sigma_log)
        or args.detector_thickness_neighbor_sigma_log <= 0.0
    ):
        raise ValueError(
            "detector-thickness-neighbor-sigma-log must be positive and finite"
        )
    if not np.isfinite(args.detector_backside_index) or args.detector_backside_index <= 0.0:
        raise ValueError("detector-backside-index must be positive and finite")
    if (args.finite_hgcdte_detector or args.fit_detector_thickness) and args.fit_hgcdte_modulation:
        raise ValueError(
            "Finite HgCdTe detectors do not support --fit-hgcdte-modulation"
        )
    if args.absolute_substrate_power and (
        args.finite_hgcdte_detector
        or args.fit_detector_thickness
        or args.fit_qe_rbf
    ):
        raise ValueError(
            "Absolute substrate power requires a semi-infinite detector and no QE RBF"
        )
    if args.fixed_absolute_hgcdte_scales and not args.absolute_substrate_power:
        raise ValueError(
            "--fix-absolute-hgcdte-reference requires --absolute-substrate-power"
        )
    if args.shared_absolute_hgcdte_scales and not args.absolute_substrate_power:
        raise ValueError(
            "--share-absolute-hgcdte-scales requires --absolute-substrate-power"
        )
    if args.shared_absolute_hgcdte_scales and args.fixed_absolute_hgcdte_scales:
        raise ValueError(
            "--share-absolute-hgcdte-scales cannot be combined with "
            "--fix-absolute-hgcdte-reference"
        )
    if args.initial_thickness_prior_uniform and args.pivot_initial_thickness:
        raise ValueError(
            "--initial-thickness-prior-uniform cannot be combined with "
            "--pivot-initial-thickness"
        )
    if args.tile_initial_thickness_lognormal and args.pivot_initial_thickness:
        raise ValueError(
            "--tile-initial-thickness-lognormal cannot be combined with "
            "--pivot-initial-thickness"
        )
    if args.tile_initial_thickness_lognormal and args.initial_thickness_prior_uniform:
        raise ValueError(
            "--tile-initial-thickness-lognormal cannot be combined with "
            "--initial-thickness-prior-uniform"
        )
    if args.ice_reset_day is not None and args.pivot_initial_thickness:
        raise ValueError(
            "--ice-reset-day cannot be combined with --pivot-initial-thickness"
        )
    if args.acm_step_prior_sigma <= 0.0 or not np.isfinite(args.acm_step_prior_sigma):
        raise ValueError("acm-step-prior-sigma must be positive and finite")
    if (
        not np.isfinite(args.late_acm_step_prior_sigma)
        or args.late_acm_step_prior_sigma <= 0.0
    ):
        raise ValueError("late-acm-step-prior-sigma must be positive and finite")
    if (
        not np.isfinite(args.additional_acm_step_prior_sigma)
        or args.additional_acm_step_prior_sigma <= 0.0
    ):
        raise ValueError(
            "additional-acm-step-prior-sigma must be positive and finite"
        )
    if args.fit_additional_acm_step and args.additional_acm_step_repeat is None:
        raise ValueError(
            "--fit-additional-acm-step requires --additional-acm-step-repeat"
        )
    if args.additional_acm_step_repeat is not None and args.additional_acm_step_repeat < 0:
        raise ValueError("additional-acm-step-repeat must be non-negative")
    if args.pre_reset_acm_step_repeat is not None and args.pre_reset_acm_step_repeat < 0:
        raise ValueError("pre-reset-acm-step-repeat must be non-negative")
    if args.pre_reset_acm_step_repeat is not None and args.fixed_corrections is None:
        raise ValueError(
            "--pre-reset-acm-step-repeat requires --fixed-corrections; "
            "the pre-reset step is fitted by build_wfi_tile_fixed_corrections.py"
        )
    if args.trajectory_rate_prior_mean_nm_day is not None and (
        not np.isfinite(args.trajectory_rate_prior_mean_nm_day)
        or args.trajectory_rate_prior_mean_nm_day <= 0.0
    ):
        raise ValueError("trajectory-rate-prior-mean-nm-day must be positive and finite")
    if args.fixed_trajectory_rate_nm_day is not None and (
        not np.isfinite(args.fixed_trajectory_rate_nm_day)
        or args.fixed_trajectory_rate_nm_day <= 0.0
    ):
        raise ValueError("fixed-trajectory-rate-nm-day must be positive and finite")
    if args.fixed_trajectory_rate_nm_day is not None and (
        args.np_modulation
        or args.constant_rate_modulation
        or args.constant_rate_trajectory
        or args.fully_np_trajectory
        or args.tile_rate_variation
        or args.tile_rate_trajectory_variation
        or args.independent_tile_trajectories
    ):
        raise ValueError(
            "A fixed trajectory rate cannot be combined with rate modulation or variation"
        )
    if args.constant_rate_trajectory and (
        args.np_modulation
        or args.constant_rate_modulation
        or args.fully_np_trajectory
        or args.tile_rate_variation
        or args.tile_rate_trajectory_variation
        or args.independent_tile_trajectories
    ):
        raise ValueError(
            "A constant trajectory rate cannot be combined with another trajectory mode"
        )
    if args.trajectory_rate_prior_sigma_log <= 0.0 or not np.isfinite(
        args.trajectory_rate_prior_sigma_log
    ):
        raise ValueError("trajectory-rate-prior-sigma-log must be positive and finite")
    if args.fixed_initial_thickness_nm is not None and (
        not np.isfinite(args.fixed_initial_thickness_nm)
        or args.fixed_initial_thickness_nm < INITIAL_THICKNESS_PRIOR_LOW_NM
        or args.fixed_initial_thickness_nm > INITIAL_THICKNESS_PRIOR_HIGH_NM
    ):
        raise ValueError("fixed-initial-thickness-nm must be between 0 and 50")
    if (
        not np.isfinite(args.initial_thickness_prior_mean_nm)
        or args.initial_thickness_prior_mean_nm < INITIAL_THICKNESS_PRIOR_LOW_NM
        or args.initial_thickness_prior_mean_nm > INITIAL_THICKNESS_PRIOR_HIGH_NM
    ):
        raise ValueError("initial-thickness-prior-mean-nm must be between 0 and 50")
    if (
        not np.isfinite(args.initial_thickness_prior_sigma_nm)
        or args.initial_thickness_prior_sigma_nm <= 0.0
    ):
        raise ValueError("initial-thickness-prior-sigma-nm must be positive and finite")
    if (
        not np.isfinite(args.tile_initial_thickness_log_median_nm)
        or args.tile_initial_thickness_log_median_nm <= 0.0
    ):
        raise ValueError(
            "tile-initial-thickness-log-median-nm must be positive and finite"
        )
    if (
        not np.isfinite(args.tile_initial_thickness_log_sigma)
        or args.tile_initial_thickness_log_sigma <= 0.0
    ):
        raise ValueError(
            "tile-initial-thickness-log-sigma must be positive and finite"
        )
    if args.ice_reset_day is not None and not np.isfinite(args.ice_reset_day):
        raise ValueError("ice-reset-day must be finite")
    if args.fit_post_reset_tile_log_offset and args.ice_reset_day is None:
        raise ValueError(
            "--fit-post-reset-tile-log-offset requires --ice-reset-day"
        )
    if (
        not np.isfinite(args.post_reset_tile_log_offset_prior_sigma)
        or args.post_reset_tile_log_offset_prior_sigma <= 0.0
    ):
        raise ValueError(
            "post-reset-tile-log-offset-prior-sigma must be positive and finite"
        )
    if args.fit_post_reset_ice_index and args.ice_reset_day is None:
        raise ValueError(
            "--fit-post-reset-ice-index requires --ice-reset-day"
        )
    if args.fit_post_reset_ice_index and args.fit_pre_post_ice_index:
        raise ValueError(
            "--fit-post-reset-ice-index and --fit-pre-post-ice-index are mutually exclusive"
        )
    if args.fit_pre_post_ice_index and args.ice_reset_day is None:
        raise ValueError(
            "--fit-pre-post-ice-index requires --ice-reset-day"
        )
    if args.fit_ice_optical_modulation and (
        args.fit_post_reset_ice_index or args.fit_pre_post_ice_index
    ):
        raise ValueError(
            "--fit-ice-optical-modulation cannot be combined with scalar ice-index fits"
        )
    if (
        not np.isfinite(args.post_reset_ice_index_prior_mean)
        or not np.isfinite(args.post_reset_ice_index_prior_sigma)
        or args.post_reset_ice_index_prior_sigma <= 0.0
        or not np.isfinite(args.post_reset_ice_index_lower)
        or not np.isfinite(args.post_reset_ice_index_upper)
        or args.post_reset_ice_index_lower
        >= args.post_reset_ice_index_upper
        or not (
            args.post_reset_ice_index_lower
            < args.post_reset_ice_index_prior_mean
            < args.post_reset_ice_index_upper
        )
    ):
        raise ValueError(
            "post-reset-ice-index prior must have a finite interior mean and bounds"
        )
    if (
        not np.isfinite(args.ice_index_prior_mean)
        or not np.isfinite(args.ice_index_prior_sigma)
        or args.ice_index_prior_sigma <= 0.0
        or not np.isfinite(args.ice_index_lower)
        or not np.isfinite(args.ice_index_upper)
        or args.ice_index_lower >= args.ice_index_upper
        or not (args.ice_index_lower < args.ice_index_prior_mean < args.ice_index_upper)
    ):
        raise ValueError(
            "ice-index prior must have a finite interior mean and bounds"
        )
    if (
        not np.isfinite(args.ice_extinction_prior_sigma)
        or args.ice_extinction_prior_sigma <= 0.0
        or not np.isfinite(args.ice_extinction_upper)
        or args.ice_extinction_upper <= 0.0
    ):
        raise ValueError(
            "ice-extinction prior must have a positive finite scale and upper bound"
        )
    if (
        not np.isfinite(args.initial_thickness_init_nm)
        or args.initial_thickness_init_nm < INITIAL_THICKNESS_PRIOR_LOW_NM
        or args.initial_thickness_init_nm > INITIAL_THICKNESS_PRIOR_HIGH_NM
    ):
        raise ValueError("initial-thickness-init-nm must be between 0 and 50")
    if (
        not np.isfinite(args.nuts_initial_thickness_nm)
        or args.nuts_initial_thickness_nm <= INITIAL_THICKNESS_PRIOR_LOW_NM
        or args.nuts_initial_thickness_nm >= INITIAL_THICKNESS_PRIOR_HIGH_NM
    ):
        raise ValueError("nuts-initial-thickness-nm must be strictly between 0 and 50")
    if not np.isfinite(args.nuts_init_jitter) or args.nuts_init_jitter < 0.0:
        raise ValueError("nuts-init-jitter must be finite and non-negative")
    if args.min_observation_day is not None and (
        not np.isfinite(args.min_observation_day)
        or args.min_observation_day < 0.0
    ):
        raise ValueError("min-observation-day must be finite and non-negative")
    if args.early_time_scale_cutoff_day is not None and (
        not np.isfinite(args.early_time_scale_cutoff_day)
        or args.early_time_scale_cutoff_day <= 0.0
    ):
        raise ValueError("early-time-scale-cutoff-day must be positive and finite")
    if args.fixed_corrections is not None and args.early_time_scale_cutoff_day is not None:
        raise ValueError(
            "--fixed-corrections cannot be combined with --early-time-scale-cutoff-day"
        )
    if (
        not np.isfinite(args.early_time_scale_prior_sigma)
        or args.early_time_scale_prior_sigma <= 0.0
    ):
        raise ValueError("early-time-scale-prior-sigma must be positive and finite")
    if args.tile_rate_prior_sigma_log <= 0.0 or not np.isfinite(
        args.tile_rate_prior_sigma_log
    ):
        raise ValueError("tile-rate-prior-sigma-log must be positive and finite")
    if args.tile_rate_neighbor_sigma_log <= 0.0 or not np.isfinite(
        args.tile_rate_neighbor_sigma_log
    ):
        raise ValueError("tile-rate-neighbor-sigma-log must be positive and finite")
    if args.warmup < 1 or args.samples < 1 or args.chains < 1 or args.max_tree_depth < 1:
        raise ValueError("Warmup, samples, chains, and max tree depth must be positive")
    if args.map_steps < 0 or args.map_learning_rate <= 0.0:
        raise ValueError("MAP steps must be non-negative and its learning rate must be positive")
    if args.map_tolerance <= 0.0 or not np.isfinite(args.map_tolerance):
        raise ValueError("map-tolerance must be positive and finite")
    if args.map_ftol < 0.0 or not np.isfinite(args.map_ftol):
        raise ValueError("map-ftol must be finite and non-negative")
    if args.map_only and args.map_steps < 1:
        raise ValueError("--map-only requires --map-steps")
    if args.map_optimizer == "adam" and args.map_parameter_scaling != "none":
        raise ValueError(
            "Prior-width parameter scaling is not supported by the Adam MAP optimizer; "
            "use --map-parameter-scaling none with Adam"
        )
    if args.map_optimizer == "trust-exact" and args.map_parameter_scaling != "none":
        raise ValueError(
            "Trust-Exact MAP optimization requires raw sample-site coordinates; "
            "use --map-parameter-scaling none"
        )
    if args.map_init_from is not None and args.map_init_max_tiles is not None:
        raise ValueError("--map-init-from cannot be combined with --map-init-max-tiles")
    if args.map_trajectory_rate_nm_day is not None and (
        not np.isfinite(args.map_trajectory_rate_nm_day)
        or args.map_trajectory_rate_nm_day <= 0.0
    ):
        raise ValueError("map-trajectory-rate-nm-day must be positive and finite")
    if args.map_init_max_tiles is not None and args.map_init_max_tiles < 1:
        raise ValueError("map-init-max-tiles must be positive")
    if args.map_design_wavelength_nm is not None and args.map_design_wavelength_nm <= 0.0:
        raise ValueError("map-design-wavelength-nm must be positive")
    if (
        args.map_middle_layer_eighth_wave or args.map_middle_layer_half_wave
    ) and args.ar_layers != 3:
        raise ValueError(
            "A special middle-layer thickness requires --ar-layers 3"
        )
    if args.wavelength_step_nm is not None and (
        not np.isfinite(args.wavelength_step_nm) or args.wavelength_step_nm <= 0.0
    ):
        raise ValueError("wavelength-step-nm must be positive and finite")
    if args.prediction_row_chunk_size is not None and args.prediction_row_chunk_size < 1:
        raise ValueError("prediction-row-chunk-size must be positive")
    incidence_cone_quadrature(args.incidence_cone_half_angle_deg)
    incidence_pupil_quadrature(args.incidence_pupil_half_angle_deg)
    if args.incidence_cone_half_angle_deg and (
        args.incidence_pupil_half_angle_deg
        or args.incidence_chief_ray_x_deg
        or args.incidence_chief_ray_y_deg
        or args.incidence_chief_ray_map is not None
    ):
        raise ValueError(
            "--incidence-cone-half-angle-deg cannot be combined with the 2D pupil geometry"
        )
    incidence_pupil_ray_sines(
        args.incidence_pupil_half_angle_deg,
        [[args.incidence_chief_ray_x_deg, args.incidence_chief_ray_y_deg]],
    )
    if (
        args.np_entropy_weight < 0.0
        or args.np_smoothness_weight < 0.0
        or args.constant_rate_smoothness_weight < 0.0
    ):
        raise ValueError("NP regularization weights must be non-negative")
    if args.np_modulation and args.np_max_node_multiplier <= 1.0:
        raise ValueError("np-max-node-multiplier must be greater than one")
    if args.ar_sca_sigma_prior <= 0.0 or args.ar_tile_sigma_prior <= 0.0:
        raise ValueError("AR hierarchy sigma priors must be positive")
    ar_reference, ar_lower, ar_upper = ar_prior_arrays(args.ar_layers)
    jax, _, numpyro, _, MCMC, NUTS, init_to_value = require_numpyro()
    if args.parallel_chains and args.chains > 1:
        numpyro.set_host_device_count(args.chains)
    jax.config.update("jax_enable_x64", True)
    if args.all_sca is None:
        args.all_sca = (
            args.tile_statistics == DEFAULT_SINGLE_SCA_TILE_STATISTICS
            and args.all_sca_statistics_dir.is_dir()
            and len(
                list(args.all_sca_statistics_dir.glob("wfi*_tile_log_statistics_*.csv"))
            )
            == 18
        )
    if args.all_sca:
        completed_sca_paths = statistics_paths(args.all_sca_statistics_dir)
        if len(completed_sca_paths) != 18:
            raise ValueError(
                "All-SCA TMM fitting requires 18 completed per-SCA statistics CSVs"
            )
    statistics_path = (
        args.all_sca_statistics_dir if args.all_sca else args.tile_statistics
    )
    args.tile_statistics = statistics_path
    if args.all_sca and args.output_dir == DEFAULT_SINGLE_SCA_OUTPUT:
        args.output_dir = DEFAULT_ALL_SCA_OUTPUT
    if args.ar_layers != 3 and args.output_dir in (
        DEFAULT_SINGLE_SCA_OUTPUT,
        DEFAULT_ALL_SCA_OUTPUT,
    ):
        args.output_dir = args.output_dir.with_name(
            f"{args.output_dir.name}_ar{args.ar_layers}layer"
        )
    data = load_tile_statistics(
        statistics_path,
        args.max_tiles,
        min_observation_day=args.min_observation_day,
        additional_acm_step_repeat=args.additional_acm_step_repeat,
        pre_reset_acm_step_repeat=args.pre_reset_acm_step_repeat,
    )
    default_chunk_size = (
        8
        if (
            (
                args.tile_rate_variation
                or args.tile_rate_trajectory_variation
            )
            and data["n_scas"] * data["n_tiles"] > 1
        )
        or args.finite_hgcdte_detector
        or args.fit_detector_thickness
        else PREDICTION_ROW_CHUNK_SIZE
    )
    data["prediction_row_chunk_size"] = (
        default_chunk_size
        if args.prediction_row_chunk_size is None
        else args.prediction_row_chunk_size
    )
    if args.fixed_corrections is not None:
        data = apply_fixed_corrections(data, args.fixed_corrections)
    incidence_chief_ray_tilts_deg = load_incidence_chief_ray_tilts(
        args.incidence_chief_ray_map,
        data,
        (args.incidence_chief_ray_x_deg, args.incidence_chief_ray_y_deg),
    )
    early_time_configuration = early_time_scale_configuration(
        data, args.early_time_scale_cutoff_day
    )
    if early_time_configuration is not None:
        data["early_time_scale_index"] = early_time_configuration[
            "observation_scale_index"
        ]
        data["early_time_scale_repeat"] = early_time_configuration["repeat"]
        data["early_time_scale_days"] = early_time_configuration["days"]
    args.likelihood_weight_automatic = args.likelihood_weight is None
    if args.likelihood_weight_automatic:
        args.likelihood_weight = default_likelihood_weight(data["counts"])
    context = load_response_context(
        args.model_path,
        args.sed_mode,
        args.wavelength_stride,
        data["sca_names"],
        args.wavelength_step_nm,
        args.ice_optical_constants_source if args.use_asw_optical_constants else None,
        args.use_hgcdte_optical_constants or args.fit_hgcdte_modulation,
        sed_weighting=args.sed_weighting,
        absolute_substrate_power=args.absolute_substrate_power,
    )
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = np.asarray(context["wavelengths"], dtype=float)
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    args.np_rate_likelihood_rotation = None
    ar_prior_reference = (
        map_start_reference(args, ar_reference, ar_lower, ar_upper)
        if args.ar_prior_from_map_start
        else ar_reference
    )

    if getattr(args, "skip_fixed_rate_warmup", False):
        warmup_args = None
        warmup_initial_values = None
        warmup_map_diagnostics = None
        print("Skipping legacy fixed-rate MAP warmup; using the requested MAP warm start.")
    else:
        warmup_args = fixed_rate_warmup_args(args)
        warmup_initial_values = initial_values_for_data(
            data, warmup_args, ar_reference, ar_lower, ar_upper
        )
        warmup_model = build_model_for_args(
            data,
            context,
            warmup_args,
            ar_prior_reference=ar_prior_reference,
            incidence_chief_ray_tilts_deg=incidence_chief_ray_tilts_deg,
        )
        warmup_initial_values, warmup_map_diagnostics = run_map_stage(
            warmup_model,
            warmup_initial_values,
            warmup_args,
            jax,
            init_to_value,
            args.seed + 1,
        )

    initial_values = initial_values_for_data(
        data, args, ar_reference, ar_lower, ar_upper
    )
    if warmup_initial_values is not None:
        initial_values = transfer_map_initial_values(
            warmup_initial_values,
            initial_values,
            source_ar_parameter_indices=ar_active_parameter_indices(
                args.ar_layers,
                args.fit_ar_absorption,
                warmup_args.fixed_ar_indices,
            ),
            target_ar_parameter_indices=ar_active_parameter_indices(
                args.ar_layers,
                args.fit_ar_absorption,
                getattr(args, "fixed_ar_indices", None),
            ),
        )
    if args.map_init_from is not None:
        initial_values = load_map_warm_start(
            args.map_init_from,
            initial_values,
            data,
            value_converters=reparameterized_map_value_converters(
                data, args, ar_reference, ar_lower, ar_upper
            ),
            allow_tile_subset=getattr(args, "map_init_allow_tile_subset", False),
            broadcast_tile_subset=getattr(
                args, "map_init_broadcast_tile_values", False
            ),
            allow_tile_template=getattr(args, "map_init_allow_tile_template", False),
        )
    model = build_model_for_args(
        data,
        context,
        args,
        ar_prior_reference=ar_prior_reference,
        incidence_chief_ray_tilts_deg=incidence_chief_ray_tilts_deg,
    )
    if args.rotate_np_likelihood:
        args.np_rate_likelihood_rotation = trajectory_rate_posterior_rotation(
            model, initial_values, jax
        )
        initial_values = apply_trajectory_rate_likelihood_rotation(
            initial_values, args
        )
        model.np_rate_likelihood_rotation_holder["value"] = jax.numpy.asarray(
            args.np_rate_likelihood_rotation
        )
    setup_seconds = time.perf_counter() - run_start
    map_diagnostics = None
    if (
        args.map_steps
        and args.map_init_max_tiles is not None
        and args.map_init_max_tiles < data["n_tiles"]
    ):
        reduced_data = load_tile_statistics(
            statistics_path,
            max_tiles=args.map_init_max_tiles,
            min_observation_day=args.min_observation_day,
            additional_acm_step_repeat=args.additional_acm_step_repeat,
            pre_reset_acm_step_repeat=args.pre_reset_acm_step_repeat,
        )
        if args.fixed_corrections is not None:
            reduced_data = apply_fixed_corrections(
                reduced_data, args.fixed_corrections
            )
        reduced_data["prediction_row_chunk_size"] = data["prediction_row_chunk_size"]
        reduced_incidence_chief_ray_tilts_deg = load_incidence_chief_ray_tilts(
            args.incidence_chief_ray_map,
            reduced_data,
            (args.incidence_chief_ray_x_deg, args.incidence_chief_ray_y_deg),
        )
        reduced_model = build_model(
            reduced_data,
            context,
            args.log_noise_sigma,
            args.likelihood_weight,
            ar_mean_prior_sigma=args.ar_mean_prior_sigma,
            ar_prior_reference=ar_prior_reference,
            ar_layers=args.ar_layers,
            np_modulation=args.np_modulation,
            constant_rate_modulation=args.constant_rate_modulation,
            constant_rate_smoothness_weight=args.constant_rate_smoothness_weight,
            fully_np_trajectory=args.fully_np_trajectory,
            standardize_ar_mean=args.standardize_ar_mean,
            rotate_ar_coordinates=args.rotate_ar_coordinates,
            whiten_np_trajectory=args.whiten_np_trajectory,
            np_rate_likelihood_rotation=args.np_rate_likelihood_rotation,
            pivot_initial_thickness=args.pivot_initial_thickness,
            map_safe_initial_thickness=bool(args.map_steps),
            np_entropy_weight=args.np_entropy_weight,
            np_smoothness_weight=args.np_smoothness_weight,
            np_max_node_multiplier=args.np_max_node_multiplier,
            ar_sca_sigma_prior=args.ar_sca_sigma_prior,
            ar_tile_sigma_prior=args.ar_tile_sigma_prior,
             fit_sca_ar_opd=args.fit_sca_ar_opd,
             sca_ar_opd_prior_sigma_nm=args.sca_ar_opd_prior_sigma_nm,
             shared_ar_optical_properties=args.shared_ar_optical_properties,
             shared_ar_parameters_across_tiles=args.shared_ar_parameters_across_tiles,
             independent_tile_ar_parameters=getattr(
                 args, "independent_tile_ar_parameters", False
             ),
             fit_ar_absorption=args.fit_ar_absorption,
            fit_hgcdte=args.fit_hgcdte,
             fit_hgcdte_modulation=args.fit_hgcdte_modulation,
             hgcdte_modulation_prior_sigma_log=args.hgcdte_modulation_prior_sigma_log,
             fit_ice_optical_modulation=args.fit_ice_optical_modulation,
             ice_modulation_prior_sigma_log=args.ice_modulation_prior_sigma_log,
             ice_n_modulation_prior_sigma_log=args.ice_n_modulation_prior_sigma_log,
             fit_conductive_layer=args.fit_conductive_layer,
            finite_hgcdte_detector=args.finite_hgcdte_detector,
            detector_thickness_nm=args.detector_thickness_nm,
            fit_detector_thickness=args.fit_detector_thickness,
            detector_thickness_prior_sigma_log=args.detector_thickness_prior_sigma_log,
            detector_thickness_neighbor_sigma_log=args.detector_thickness_neighbor_sigma_log,
            detector_backside_index=args.detector_backside_index,
            fit_qe_rbf=args.fit_qe_rbf,
            qe_rbf_node_wavelengths_nm=args.qe_rbf_node_wavelengths_nm,
            qe_rbf_width_nm=args.qe_rbf_width_nm,
            qe_rbf_prior_sigma_log=args.qe_rbf_prior_sigma_log,
            use_asw_optical_constants=args.use_asw_optical_constants,
            use_hgcdte_optical_constants=(
                args.use_hgcdte_optical_constants or args.fit_hgcdte_modulation
            ),
            incidence_cone_half_angle_deg=args.incidence_cone_half_angle_deg,
            incidence_pupil_half_angle_deg=args.incidence_pupil_half_angle_deg,
            incidence_chief_ray_tilts_deg=reduced_incidence_chief_ray_tilts_deg,
            fit_incidence_angle=args.fit_incidence_angle,
            incidence_angle_prior_sigma_deg=args.incidence_angle_prior_sigma_deg,
            incidence_angle_prior_uniform=args.incidence_angle_prior_uniform,
            led_dependent_acm_step=args.led_dependent_acm_step,
            fit_acm_step=not corrections_applied,
            acm_step_prior_sigma=args.acm_step_prior_sigma,
            fit_late_acm_step=(
                args.fit_late_acm_step if not corrections_applied else False
            ),
            late_acm_step_prior_sigma=args.late_acm_step_prior_sigma,
            fit_additional_acm_step=(
                args.fit_additional_acm_step if not corrections_applied else False
            ),
            additional_acm_step_prior_sigma=args.additional_acm_step_prior_sigma,
             trajectory_rate_prior_mean_nm_day=args.trajectory_rate_prior_mean_nm_day,
             fixed_trajectory_rate_nm_day=args.fixed_trajectory_rate_nm_day,
             constant_rate_trajectory=getattr(args, "constant_rate_trajectory", False),
             fixed_ar_indices=getattr(args, "fixed_ar_indices", None),
             ordered_ar_indices=getattr(args, "ordered_ar_indices", False),
             trajectory_rate_prior_sigma_log=args.trajectory_rate_prior_sigma_log,
            tile_rate_variation=args.tile_rate_variation,
            tile_rate_trajectory_variation=args.tile_rate_trajectory_variation,
            independent_tile_trajectories=args.independent_tile_trajectories,
            tile_rate_trajectory_smoothness_weight=(
                args.tile_rate_trajectory_smoothness_weight
            ),
            tile_rate_prior_sigma_log=args.tile_rate_prior_sigma_log,
            tile_rate_neighbor_sigma_log=args.tile_rate_neighbor_sigma_log,
            fixed_initial_thickness_nm=args.fixed_initial_thickness_nm,
            initial_thickness_prior_mean_nm=args.initial_thickness_prior_mean_nm,
            initial_thickness_prior_sigma_nm=args.initial_thickness_prior_sigma_nm,
            early_time_cutoff_day=(
                args.early_time_scale_cutoff_day
                if not corrections_applied
                else None
            ),
            early_time_scale_prior_sigma=args.early_time_scale_prior_sigma,
            absolute_hgcdte_scale_prior_sigma_log=args.absolute_hgcdte_scale_prior_sigma_log,
            fixed_absolute_hgcdte_scales=args.fixed_absolute_hgcdte_scales,
            shared_absolute_hgcdte_scales=args.shared_absolute_hgcdte_scales,
            initial_thickness_prior_uniform=args.initial_thickness_prior_uniform,
            tile_initial_thickness_lognormal=args.tile_initial_thickness_lognormal,
            tile_initial_thickness_log_median_nm=args.tile_initial_thickness_log_median_nm,
             tile_initial_thickness_log_sigma=args.tile_initial_thickness_log_sigma,
             ice_reset_day=args.ice_reset_day,
             fit_post_reset_tile_log_offset=args.fit_post_reset_tile_log_offset,
             post_reset_tile_log_offset_prior_sigma=args.post_reset_tile_log_offset_prior_sigma,
             fit_post_reset_ice_index=args.fit_post_reset_ice_index,
            post_reset_ice_index_prior_mean=args.post_reset_ice_index_prior_mean,
            post_reset_ice_index_prior_sigma=args.post_reset_ice_index_prior_sigma,
            post_reset_ice_index_lower=args.post_reset_ice_index_lower,
            post_reset_ice_index_upper=args.post_reset_ice_index_upper,
            fit_pre_post_ice_index=args.fit_pre_post_ice_index,
            ice_index_prior_mean=args.ice_index_prior_mean,
            ice_index_prior_sigma=args.ice_index_prior_sigma,
            ice_index_lower=args.ice_index_lower,
            ice_index_upper=args.ice_index_upper,
             ice_extinction_prior_sigma=args.ice_extinction_prior_sigma,
             ice_extinction_upper=args.ice_extinction_upper,
             fit_complex_q=getattr(args, "fit_complex_q", False),
             include_map_prediction=(args.map_optimizer == "least-squares"),
          )
        reduced_initial_values = initial_values_for_data(
            reduced_data, args, ar_reference, ar_lower, ar_upper
        )
        reduced_initial_values = apply_trajectory_rate_likelihood_rotation(
            reduced_initial_values, args
        )
        if args.map_optimizer == "lbfgs":
            reduced_scales = (
                map_parameter_scale_values(reduced_initial_values, args)
                if args.map_parameter_scaling == "prior"
                else None
            )
            reduced_map_values, _ = run_map_optimization(
                reduced_model,
                reduced_initial_values,
                jax,
                args.map_steps,
                args.map_tolerance,
                args.map_ftol,
                parameter_scales=reduced_scales,
            )
        elif args.map_optimizer == "least-squares":
            reduced_scales = (
                map_parameter_scale_values(reduced_initial_values, args)
                if args.map_parameter_scaling == "prior"
                else None
            )
            reduced_map_values, _ = run_least_squares_map_optimization(
                reduced_model,
                reduced_initial_values,
                reduced_data,
                args,
                ar_prior_reference,
                jax,
                args.map_steps,
                args.map_tolerance,
                parameter_scales=reduced_scales,
            )
        elif args.map_optimizer == "jax-lbfgsb":
            reduced_scales = (
                map_parameter_scale_values(reduced_initial_values, args)
                if args.map_parameter_scaling == "prior"
                else None
            )
            reduced_map_values, _ = run_jax_map_optimization(
                reduced_model,
                reduced_initial_values,
                jax,
                args.map_steps,
                args.map_tolerance,
                parameter_scales=reduced_scales,
            )
        elif args.map_optimizer == "trust-exact":
            reduced_map_values, _ = run_trust_exact_map_optimization(
                reduced_model,
                reduced_initial_values,
                jax,
                args.map_steps,
                args.map_tolerance,
            )
        else:
            reduced_map_values, _ = run_map_initialization(
                reduced_model,
                reduced_initial_values,
                jax,
                init_to_value,
                args.seed + 1,
                args.map_steps,
                args.map_learning_rate,
            )
        initial_values = transfer_map_initial_values(
            reduced_map_values,
            initial_values,
            source_tile_ids=reduced_data["tile_ids"],
            target_tile_ids=data["tile_ids"],
        )
    if args.map_steps:
        if args.map_optimizer == "lbfgs":
            parameter_scales = (
                map_parameter_scale_values(initial_values, args)
                if args.map_parameter_scaling == "prior"
                else None
            )
            initial_values, map_diagnostics = run_map_optimization(
                model,
                initial_values,
                jax,
                args.map_steps,
                args.map_tolerance,
                args.map_ftol,
                parameter_scales=parameter_scales,
            )
        elif args.map_optimizer == "least-squares":
            parameter_scales = (
                map_parameter_scale_values(initial_values, args)
                if args.map_parameter_scaling == "prior"
                else None
            )
            initial_values, map_diagnostics = run_least_squares_map_optimization(
                model,
                initial_values,
                data,
                args,
                ar_prior_reference,
                jax,
                args.map_steps,
                args.map_tolerance,
                parameter_scales=parameter_scales,
            )
        elif args.map_optimizer == "jax-lbfgsb":
            parameter_scales = (
                map_parameter_scale_values(initial_values, args)
                if args.map_parameter_scaling == "prior"
                else None
            )
            initial_values, map_diagnostics = run_jax_map_optimization(
                model,
                initial_values,
                jax,
                args.map_steps,
                args.map_tolerance,
                parameter_scales=parameter_scales,
            )
        elif args.map_optimizer == "trust-exact":
            initial_values, map_diagnostics = run_trust_exact_map_optimization(
                model,
                initial_values,
                jax,
                args.map_steps,
                args.map_tolerance,
            )
        else:
            initial_values, map_diagnostics = run_map_initialization(
                model,
                initial_values,
                jax,
                init_to_value,
                args.seed + 1,
                args.map_steps,
                args.map_learning_rate,
            )
    if args.map_steps and not args.map_only:
        model.map_safe_initial_thickness_holder["value"] = False
    if args.map_steps and args.pivot_initial_thickness:
        map_initial_thickness = trajectory_pivot_d0_from_initial_values(
            data, args, initial_values
        )
        if not (
            INITIAL_THICKNESS_PRIOR_LOW_NM < map_initial_thickness
            < INITIAL_THICKNESS_PRIOR_HIGH_NM
        ):
            raise ValueError(
                "MAP ended outside the supported initial-thickness interval: "
                f"{map_initial_thickness} nm"
            )
    if args.map_only:
        from numpyro.infer import Predictive

        postprocessing_start = time.perf_counter()
        map_posterior = {
            name: jax.numpy.asarray(value)[None, ...]
            for name, value in initial_values.items()
        }
        map_samples = Predictive(model, posterior_samples=map_posterior)(
            jax.random.PRNGKey(args.seed + 2)
        )
        samples = {
            name: np.asarray(value)
            for name, value in {**map_posterior, **map_samples}.items()
        }
        samples["chain_index"] = np.zeros(1, dtype=int)
        fit_diagnostics = posterior_median_fit_diagnostics(
            data,
            context,
            samples,
            fit_hgcdte=args.fit_hgcdte,
            fit_hgcdte_modulation=args.fit_hgcdte_modulation,
            fit_conductive_layer=args.fit_conductive_layer,
            finite_hgcdte_detector=(
                args.finite_hgcdte_detector or args.fit_detector_thickness
            ),
            detector_backside_index=args.detector_backside_index,
            fit_qe_rbf=args.fit_qe_rbf,
            qe_rbf_node_wavelengths_nm=args.qe_rbf_node_wavelengths_nm,
             qe_rbf_width_nm=args.qe_rbf_width_nm,
              use_asw_optical_constants=args.use_asw_optical_constants,
              fit_ice_optical_modulation=args.fit_ice_optical_modulation,
              fit_post_reset_tile_log_offset=args.fit_post_reset_tile_log_offset,
              use_hgcdte_optical_constants=(
                args.use_hgcdte_optical_constants or args.fit_hgcdte_modulation
            ),
            incidence_cone_half_angle_deg=args.incidence_cone_half_angle_deg,
            incidence_pupil_half_angle_deg=args.incidence_pupil_half_angle_deg,
            incidence_chief_ray_tilts_deg=incidence_chief_ray_tilts_deg,
            fit_incidence_angle=args.fit_incidence_angle,
            early_time_cutoff_day=args.early_time_scale_cutoff_day,
            fit_late_acm_step=args.fit_late_acm_step,
             fit_additional_acm_step=(
                  args.fit_additional_acm_step if not corrections_applied else False
             ),
             fit_sca_ar_opd=args.fit_sca_ar_opd,
             shared_absolute_hgcdte_scales=args.shared_absolute_hgcdte_scales,
             fit_post_reset_ice_index=args.fit_post_reset_ice_index,
              post_reset_ice_reset_day=(
                  args.ice_reset_day
                  if args.fit_post_reset_ice_index or args.fit_pre_post_ice_index
                  else None
              ),
              post_reset_log_offset_day=(
                  args.ice_reset_day
                  if args.fit_post_reset_tile_log_offset
                  else None
              ),
               fit_pre_post_ice_index=args.fit_pre_post_ice_index,
               fit_complex_q=getattr(args, "fit_complex_q", False),
          )
        postprocessing_seconds = time.perf_counter() - postprocessing_start
        write_outputs(
            args.output_dir,
            data,
            samples,
            args,
            fit_diagnostics=fit_diagnostics,
            map_diagnostics=map_diagnostics,
            warmup_map_diagnostics=warmup_map_diagnostics,
        )
        output_seconds = time.perf_counter() - postprocessing_start - postprocessing_seconds
        print(
            f"MAP stage timing: setup={setup_seconds:.2f}s; "
            f"posterior/diagnostics={postprocessing_seconds:.2f}s; "
            f"write={output_seconds:.2f}s; total={time.perf_counter() - run_start:.2f}s"
        )
        print(f"Wrote {args.output_dir / 'numpyro_summary.md'}")
        print(f"Wrote {args.output_dir / 'tile_ar_posterior.csv'}")
        if args.absolute_substrate_power:
            print(f"Wrote {args.output_dir / 'tile_hgcdte_posterior.csv'}")
        if args.fit_ice_optical_modulation:
            print(
                f"Wrote {args.output_dir / 'ice_optical_modulation_posterior.csv'}"
            )
        if args.fit_post_reset_tile_log_offset:
            print(
                f"Wrote {args.output_dir / 'post_reset_tile_log_offset_posterior.csv'}"
            )
        if args.tile_initial_thickness_lognormal:
            print(
                f"Wrote {args.output_dir / 'tile_initial_thickness_posterior.csv'}"
            )
            if args.ice_reset_day is not None:
                print(
                    f"Wrote {args.output_dir / 'tile_reset_initial_thickness_posterior.csv'}"
                )
        if data["n_scas"] > 1:
            print(f"Wrote {args.output_dir / 'sca_ar_posterior.csv'}")
        if args.fit_sca_ar_opd:
            print(f"Wrote {args.output_dir / 'sca_ar_opd_posterior.csv'}")
        if "tile_initial_rate_nm_day" in samples:
            print(f"Wrote {args.output_dir / 'tile_rate_posterior.csv'}")
        if "detector_thickness_nm" in samples:
            print(f"Wrote {args.output_dir / 'detector_thickness_posterior.csv'}")
        if (
            args.np_modulation
            or args.constant_rate_modulation
            or (
                args.fully_np_trajectory
                and not args.independent_tile_trajectories
            )
        ):
            print(f"Wrote {args.output_dir / 'np_node_posterior.csv'}")
        return
    nuts_dense_mass = args.dense_mass
    if args.dense_mass_global:
        global_site_candidates = {
            "ar_mu_raw",
            "ar_mu_z",
            "ar_sigma_raw",
            "ar_sca_sigma",
            "ar_tile_sigma",
            "sca_ar_opd_nm",
            "initial_thickness_nm",
            "initial_thickness_pivot_z",
            "incidence_angle_deg",
            "early_time_log_scale",
            "np_log_rate_nodes",
            "np_log_rate_z",
            "log_trajectory_scale",
            "np_node_raw",
            "hgcdte_n",
            "hgcdte_k",
            "hgcdte_n_log_modulation",
            "hgcdte_k_log_modulation",
            "ice_n_log_modulation",
            "ice_k_log_modulation",
            "conductive_log_thickness_nm",
            "conductive_log_plasma_strength",
            "conductive_log_damping_ratio",
            "qe_rbf_log_modulation",
            "acm_step_log_offset",
            "late_acm_step_log_offset",
            "additional_acm_step_log_offset",
            "complex_q_parameters",
        }
        global_sites = tuple(
            sorted(name for name in initial_values if name in global_site_candidates)
        )
        if not global_sites:
            raise ValueError("No global latent sites are available for dense adaptation")
        nuts_dense_mass = [global_sites]
    if (
        args.fixed_initial_thickness_nm is None
        and not args.map_init_from
        and not args.map_steps
    ):
        if args.pivot_initial_thickness:
            initial_values["initial_thickness_pivot_z"] = (
                trajectory_pivot_z_from_initial_thickness(
                    args.nuts_initial_thickness_nm,
                    data,
                    args,
                    trajectory_log_rates_from_initial_values(data, args, initial_values),
                )
            )
        else:
            initial_values["initial_thickness_nm"] = args.nuts_initial_thickness_nm
    nuts_initial_values = initial_values
    if (
        args.fixed_initial_thickness_nm is None
        and (args.map_init_from is not None or args.map_steps)
    ):
        if args.pivot_initial_thickness:
            boundary_d0 = trajectory_pivot_d0_from_initial_values(
                data, args, initial_values
            )
            if boundary_d0 <= INITIAL_THICKNESS_PRIOR_LOW_NM + 1.0e-7:
                nuts_initial_values = dict(initial_values)
                nuts_initial_values["initial_thickness_pivot_z"] = (
                    trajectory_pivot_z_from_initial_thickness(
                        args.nuts_initial_thickness_nm,
                        data,
                        args,
                        trajectory_log_rates_from_initial_values(data, args, initial_values),
                    )
                )
        elif (
            "initial_thickness_nm" in initial_values
            and float(np.asarray(initial_values["initial_thickness_nm"]))
            <= INITIAL_THICKNESS_PRIOR_LOW_NM + 1.0e-7
        ):
            # A boundary MAP value has a singular transformed geometry. Keep every
            # other MAP site intact and start d0 just inside the support; d0 remains
            # a free NUTS variable after initialization.
            nuts_initial_values = dict(initial_values)
            nuts_initial_values["initial_thickness_nm"] = args.nuts_initial_thickness_nm
    nuts_initial_values = interior_nuts_initial_values(nuts_initial_values)
    nuts_initial_params = build_nuts_initial_params(
        model,
        nuts_initial_values,
        jax,
        init_to_value,
        args.seed + 3,
        args.chains,
        args.nuts_init_jitter,
    )
    kernel = NUTS(
        model,
        target_accept_prob=args.target_accept_prob,
        max_tree_depth=args.max_tree_depth,
        dense_mass=nuts_dense_mass,
        init_strategy=init_to_value(values=nuts_initial_values),
    )
    if jax.default_backend() == "gpu":
        # NumPyro's default collector donates its collection buffer. This is
        # not host-readable after sampling on the current ROCm/JAX runtime.
        install_gpu_safe_numpyro_collection(jax)
    if not args.no_progress_bar and jax.default_backend() == "gpu":
        # ROCm can fail when NumPyro formats live JAX diagnostics for tqdm.
        # Keep the iteration/rate/ETA display, but avoid copying these values
        # from the device on every transition.
        kernel.get_diagnostics_str = lambda state: ""
    mcmc = MCMC(
        kernel,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        chain_method=(
            "parallel"
            if args.parallel_chains
            else "vectorized"
            if args.vectorized_chains
            else "sequential"
        ),
        progress_bar=not args.no_progress_bar,
    )
    mcmc.run(
        jax.random.PRNGKey(args.seed),
        init_params=nuts_initial_params,
        extra_fields=("accept_prob", "diverging", "num_steps"),
    )
    chain_samples = mcmc.get_samples(group_by_chain=True)
    extra_fields = mcmc.get_extra_fields(group_by_chain=True)
    # Force a concrete device copy before NumPyro's diagnostics or posterior
    # writers touch the arrays. The GPU collector above avoids donated buffers,
    # and this explicit copy keeps all later output operations on host arrays.
    chain_samples = {
        name: np.asarray(jax.numpy.copy(value))
        for name, value in chain_samples.items()
    }
    extra_fields = {
        name: np.asarray(jax.numpy.copy(value))
        for name, value in extra_fields.items()
    }
    sampler_diagnostics = None
    if args.samples >= 4:
        sampler_diagnostics = summarize_sampler_diagnostics(
            chain_samples,
            extra_fields,
            args.max_tree_depth,
        )
    samples = {
        name: value.reshape((-1,) + value.shape[2:])
        for name, value in chain_samples.items()
    }
    chain_index = np.repeat(np.arange(args.chains), args.samples)
    samples["chain_index"] = chain_index
    for name, value in extra_fields.items():
        samples[f"diagnostic_{name}"] = np.asarray(value).reshape(-1)
    fit_diagnostics = posterior_median_fit_diagnostics(
        data,
        context,
        samples,
        fit_hgcdte=args.fit_hgcdte,
        fit_hgcdte_modulation=args.fit_hgcdte_modulation,
        fit_conductive_layer=args.fit_conductive_layer,
        finite_hgcdte_detector=(
            args.finite_hgcdte_detector or args.fit_detector_thickness
        ),
        detector_backside_index=args.detector_backside_index,
        fit_qe_rbf=args.fit_qe_rbf,
        qe_rbf_node_wavelengths_nm=args.qe_rbf_node_wavelengths_nm,
         qe_rbf_width_nm=args.qe_rbf_width_nm,
         use_asw_optical_constants=args.use_asw_optical_constants,
         fit_ice_optical_modulation=args.fit_ice_optical_modulation,
         fit_post_reset_tile_log_offset=args.fit_post_reset_tile_log_offset,
         use_hgcdte_optical_constants=(
            args.use_hgcdte_optical_constants or args.fit_hgcdte_modulation
        ),
        incidence_cone_half_angle_deg=args.incidence_cone_half_angle_deg,
        incidence_pupil_half_angle_deg=args.incidence_pupil_half_angle_deg,
        incidence_chief_ray_tilts_deg=incidence_chief_ray_tilts_deg,
        early_time_cutoff_day=args.early_time_scale_cutoff_day,
        fit_late_acm_step=args.fit_late_acm_step,
         fit_additional_acm_step=(
             args.fit_additional_acm_step if not corrections_applied else False
         ),
         fit_sca_ar_opd=args.fit_sca_ar_opd,
         shared_absolute_hgcdte_scales=args.shared_absolute_hgcdte_scales,
         fit_post_reset_ice_index=args.fit_post_reset_ice_index,
          post_reset_ice_reset_day=(
              args.ice_reset_day
              if args.fit_post_reset_ice_index or args.fit_pre_post_ice_index
              else None
          ),
          post_reset_log_offset_day=(
              args.ice_reset_day
              if args.fit_post_reset_tile_log_offset
              else None
          ),
          fit_pre_post_ice_index=args.fit_pre_post_ice_index,
          fit_complex_q=getattr(args, "fit_complex_q", False),
      )
    write_outputs(
        args.output_dir,
        data,
        samples,
        args,
        fit_diagnostics=fit_diagnostics,
        sampler_diagnostics=sampler_diagnostics,
        map_diagnostics=map_diagnostics,
        warmup_map_diagnostics=warmup_map_diagnostics,
    )
    if args.samples >= 4:
        if jax.default_backend() == "gpu":
            print("Skipped NumPyro print_summary on GPU; durable diagnostics were written.")
        else:
            mcmc.print_summary()
    else:
        print("Skipped NumPyro summary: at least four samples are required for r_hat.")
    print(f"Wrote {args.output_dir / 'numpyro_summary.md'}")
    print(f"Wrote {args.output_dir / 'tile_ar_posterior.csv'}")
    if args.fit_ice_optical_modulation:
        print(f"Wrote {args.output_dir / 'ice_optical_modulation_posterior.csv'}")
    if args.fit_post_reset_tile_log_offset:
        print(
            f"Wrote {args.output_dir / 'post_reset_tile_log_offset_posterior.csv'}"
        )
    if data["n_scas"] > 1:
        print(f"Wrote {args.output_dir / 'sca_ar_posterior.csv'}")
    if args.fit_sca_ar_opd:
        print(f"Wrote {args.output_dir / 'sca_ar_opd_posterior.csv'}")
    if "tile_initial_rate_nm_day" in samples:
        print(f"Wrote {args.output_dir / 'tile_rate_posterior.csv'}")
    if args.tile_initial_thickness_lognormal:
        print(
            f"Wrote {args.output_dir / 'tile_initial_thickness_posterior.csv'}"
        )
        if args.ice_reset_day is not None:
            print(
                f"Wrote {args.output_dir / 'tile_reset_initial_thickness_posterior.csv'}"
            )
    if "detector_thickness_nm" in samples:
        print(f"Wrote {args.output_dir / 'detector_thickness_posterior.csv'}")
    if (
        args.np_modulation
        or args.constant_rate_modulation
        or (
            args.fully_np_trajectory
            and not args.independent_tile_trajectories
        )
    ):
        print(f"Wrote {args.output_dir / 'np_node_posterior.csv'}")


def main():
    if len(sys.argv) != 1:
        raise SystemExit(
            "fit_wfi_tmm_production.py has no command-line arguments; edit its "
            "production constants instead"
        )
    statistics_paths = sorted(
        PRODUCTION_INPUT_ROOT.glob(PRODUCTION_STATISTICS_PATTERN)
    )
    if not statistics_paths:
        raise FileNotFoundError(
            "No pre-corrected production statistics found under "
            f"{PRODUCTION_INPUT_ROOT}: {PRODUCTION_STATISTICS_PATTERN}"
        )
    PRODUCTION_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    global _PRODUCTION_STATISTICS_PATH
    for statistics_path in statistics_paths:
        _PRODUCTION_STATISTICS_PATH = statistics_path
        _fit_one()


if __name__ == "__main__":
    main()
