"""Validated, user-facing defaults for the path-driven WFI pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DETECTORS = tuple(f"WFI{number:02d}" for number in range(1, 19))

RELATIVE_RESPONSE = "relative_response"
FULL_TMM = "full_tmm"
DEFAULT_MODEL_ROUTE = RELATIVE_RESPONSE
PIPELINE_VERSION = 2

TILE_SIZE_PIXELS = 256
THINNING_SPACING_HOURS = 4.0
THIRD_DECON_DAY = 38.79784510416666
SECOND_DECON_DAY = 12.907397928240743
PRODUCTION_SED_WEIGHTING = "energy"
PRODUCTION_RATE_PRIOR_MEAN_NM_DAY = 12.0
ERIC_V3_BASE_MODEL = (
    ROOT
    / "20261111/corrected_statistics_4h_thinned/production/"
    "wavelength_response_bundle/eric_v3_reference/ice_response_model_v3.h5"
)

JOINT_EXPERIMENT_NAME = "car190_amber_joint_2day_reader_compatible_current_gray_v2"
JOINT_MODULATED_BUNDLE_NAME = "wfi_all_2nm_rank16_modulated.npz"
FINAL_H5_NAME = "joint_modulated_v2.h5"

SPECTRAL_CACHE_WAVELENGTH_MIN_NM = 470.0
SPECTRAL_CACHE_WAVELENGTH_MAX_NM = 2700.0
SPECTRAL_CACHE_WAVELENGTH_STEP_NM = 2.0
SPECTRAL_CACHE_MAX_THICKNESS_NM = 500.0
SPECTRAL_CACHE_THICKNESS_STEP_NM = 2.0
SPECTRAL_CACHE_RANK = 16

FULL_TMM_EMCEE_WALKERS = 52
FULL_TMM_EMCEE_STEPS = 1000
FULL_TMM_EMCEE_SEED = 20261004
FULL_TMM_MAP_MAX_EVALUATIONS = 10000
FULL_TMM_MAP_FTOL = 1.0e-9
FULL_TMM_MAP_XTOL = 1.0e-9

JOINT_FIT_MAXITER = 700
JOINT_FIT_FTOL = 1.0e-7
JOINT_FIT_GTOL = 1.0e-4

TMM_REFLECTANCE_MODE_COUNT = 64
TMM_REFLECTANCE_WAVELENGTH_MIN_NM = 475.0
TMM_REFLECTANCE_WAVELENGTH_MAX_NM = 2350.0
TMM_REFLECTANCE_WAVELENGTH_STEP_NM = 1.0


@dataclass(frozen=True)
class PipelineProfile:
    """Settings selected by the named full-TMM or cached-relative frontend."""

    name: str
    tile_size_pixels: int = TILE_SIZE_PIXELS
    workers: int = 2
    build_tmm_spectral_cache: bool = True
    build_six_led_cache_from_current_tmm: bool = False
    use_pinned_six_led_cache: bool = False
    fit_led_log_offsets: bool = False
    led_log_offset_prior_sigma_log: float = 0.01
    relative_batch_size: int = 1


PROFILES = {
    RELATIVE_RESPONSE: PipelineProfile(
        name=RELATIVE_RESPONSE,
        build_tmm_spectral_cache=True,
        build_six_led_cache_from_current_tmm=False,
        use_pinned_six_led_cache=True,
        fit_led_log_offsets=True,
        led_log_offset_prior_sigma_log=0.01,
        relative_batch_size=1,
    ),
    FULL_TMM: PipelineProfile(
        name=FULL_TMM,
        build_tmm_spectral_cache=True,
        build_six_led_cache_from_current_tmm=True,
        use_pinned_six_led_cache=False,
    ),
}


def profile_for(name: str) -> PipelineProfile:
    try:
        return PROFILES[str(name)]
    except KeyError as error:
        raise ValueError(
            f"Unknown WFI pipeline route {name!r}; choose {', '.join(PROFILES)}"
        ) from error


def pinned_relative_response_cache(detector: str) -> Path:
    detector = str(detector).upper()
    if detector not in DETECTORS:
        raise ValueError(f"Unknown WFI detector: {detector}")
    return (
        ROOT
        / "20261109/corrected_statistics_4h_thinned/production/"
        "relative_response_cache/tile_size_256/"
        "sensitivity_hgcdte_n_sigma_0p25"
        / f"{detector.lower()}_400nm"
    )


def output_paths(output_path, route):
    """Return the unified artifact paths under one user-selected run root."""
    root = Path(output_path).resolve()
    production = root / "production"
    profile_for(route)
    return {
        "root": root,
        "logs": root / "logs",
        "manifest": root / "pipeline_manifest.json",
        "corrected_statistics_4h": root / "corrected_statistics_4h_thinned",
        "production": production,
        "fit_root": production,
        "six_led_cache_root": production / "caches" / "six_led" / route,
        "spectral_cache_root": production / "caches" / "spectral_response" / route,
        "bundle_root": production / "wavelength_response_bundle",
        "h5_intermediate_root": production / "wavelength_response_bundle" / "intermediates" / "h5",
        "joint_fit_root": production,
        "h5_root": production / "wavelength_response_bundle" / "final",
        "plots": production / "plots" / route,
    }


def detector_fit_output_root(output_path, route, detector):
    """Return the standard per-detector output root for one selected model route."""
    profile = profile_for(route)
    detector = str(detector).lower()
    if detector.upper() not in DETECTORS:
        raise ValueError(f"Unknown WFI detector: {detector}")
    root = Path(output_path).resolve() / "production" / detector
    return root / ("relative_response" if profile.name == RELATIVE_RESPONSE else "full_tmm")


def detector_spectral_cache_root(output_path, route, detector):
    paths = output_paths(output_path, route)
    detector = str(detector).lower()
    if detector.upper() not in DETECTORS:
        raise ValueError(f"Unknown WFI detector: {detector}")
    return paths["spectral_cache_root"] / detector


def detector_six_led_cache_root(output_path, route, detector):
    profile = profile_for(route)
    detector = str(detector).upper()
    if detector not in DETECTORS:
        raise ValueError(f"Unknown WFI detector: {detector}")
    if profile.use_pinned_six_led_cache:
        return pinned_relative_response_cache(detector)
    return output_paths(output_path, route)["six_led_cache_root"] / detector.lower()
