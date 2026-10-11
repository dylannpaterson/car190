"""Run one joint WFI01 fit with no cross-tile parameter hierarchy."""

from pathlib import Path

import fit_wfi_tmm_production as production
import run_wfi_tmm_emcee_warmup as emcee_warmup
from plot_wfi_tmm_numpyro import make_plots


ROOT = Path(__file__).resolve().parent
STATISTICS = (
    ROOT / "rrn_reduction_20261005/wfi01_tile_log_statistics_512_count_corrected.csv"
)
REFERENCE_DATA_DIR = ROOT / "external_reference_data"
EMCEE_OUTPUT = (
    ROOT
    / "wfi01_absolute_substrate_power_hgcdte_prior025_all_tiles_joint_independent_emcee1000"
)
OUTPUT_DIR = (
    ROOT
    / "wfi01_absolute_substrate_power_hgcdte_prior025_all_tiles_joint_independent_map10000"
)
HGCDTE_PRIOR_SIGMA_LOG = 0.25


def main():
    emcee_warmup.run_emcee(
        STATISTICS,
        EMCEE_OUTPUT,
        walkers=52,
        steps=1000,
        seed=20261004,
        absolute_hgcdte_scale_prior_sigma_log=HGCDTE_PRIOR_SIGMA_LOG,
        fit_complex_q=False,
        reference_data_dir=REFERENCE_DATA_DIR,
    )
    production._PRODUCTION_STATISTICS_PATH = STATISTICS
    production.SED_URL = str(
        REFERENCE_DATA_DIR / "WFI_RCS_LEDP_spectral_energy_distribution.ecsv"
    )
    production.QE_URL = str(REFERENCE_DATA_DIR / "WFI_Quantum_efficiency.ecsv")
    args = production.parse_args()
    args.all_sca = False
    args.max_tiles = None
    args.tile_statistics = STATISTICS
    args.output_dir = OUTPUT_DIR
    args.map_init_from = EMCEE_OUTPUT
    args.map_steps = 10000
    args.map_only = True
    args.map_init_allow_tile_subset = True
    args.map_init_broadcast_tile_values = True
    args.fixed_absolute_hgcdte_scales = False
    args.shared_absolute_hgcdte_scales = False
    args.absolute_hgcdte_scale_prior_sigma_log = HGCDTE_PRIOR_SIGMA_LOG
    args.shared_ar_optical_properties = False
    args.shared_ar_parameters_across_tiles = False
    args.independent_tile_ar_parameters = True
    args.tile_rate_variation = True
    args.tile_rate_trajectory_variation = True
    args.independent_tile_trajectories = True
    args.tile_rate_neighbor_smoothness = False
    args.whiten_np_trajectory = False
    args.fit_complex_q = False
    args.skip_fixed_rate_warmup = True
    production._fit_one(args)
    make_plots(OUTPUT_DIR, tile_statistics=STATISTICS)


if __name__ == "__main__":
    main()
