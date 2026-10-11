"""Sensitivity MAP for the regularized HgCdTe run with a wider q slope bound."""

from pathlib import Path

import fit_wfi_tmm_production as production
from diagnose_wfi_tmm_boundary_reflection import run_diagnostic


ROOT = Path(__file__).resolve().parent
STATISTICS = (
    ROOT / "rrn_reduction_20261005/wfi01_tile_log_statistics_512_count_corrected.csv"
)
REFERENCE_DATA_DIR = ROOT / "external_reference_data"
EMCEE_OUTPUT = ROOT / "wfi01_joint_complex_q_cold_hgcdte_prior025_emcee1000"
MAP_OUTPUT = ROOT / "wfi01_joint_complex_q_cold_hgcdte_prior025_qslope_minus3_map10000"
HGCDTE_PRIOR_SIGMA_LOG = 0.25


def main():
    production._PRODUCTION_STATISTICS_PATH = STATISTICS
    production.SED_URL = str(
        REFERENCE_DATA_DIR / "WFI_RCS_LEDP_spectral_energy_distribution.ecsv"
    )
    production.QE_URL = str(REFERENCE_DATA_DIR / "WFI_Quantum_efficiency.ecsv")
    production.COMPLEX_Q_LOWER = production.COMPLEX_Q_LOWER.copy()
    production.COMPLEX_Q_LOWER[1] = -3.0
    args = production.parse_args()
    args.all_sca = False
    args.max_tiles = 1
    args.tile_statistics = STATISTICS
    args.output_dir = MAP_OUTPUT
    args.map_init_from = EMCEE_OUTPUT
    args.map_init_allow_tile_subset = False
    args.fit_complex_q = True
    args.complex_q_initial_parameters = production.COMPLEX_Q_INITIAL.copy()
    args.fixed_absolute_hgcdte_scales = False
    args.shared_absolute_hgcdte_scales = True
    args.absolute_hgcdte_scale_prior_sigma_log = HGCDTE_PRIOR_SIGMA_LOG
    args.map_steps = 10000
    args.map_only = True
    production._fit_one(args)
    run_diagnostic(
        MAP_OUTPUT,
        STATISTICS,
        REFERENCE_DATA_DIR,
        MAP_OUTPUT / "boundary_reflection_diagnostic",
    )


if __name__ == "__main__":
    main()
