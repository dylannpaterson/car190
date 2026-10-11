"""Run the fixed-HgCdTe cold single-tile MAP with all MAP stages fixed."""

from pathlib import Path

import fit_wfi_tmm_production as production


ROOT = Path(__file__).resolve().parent
STATISTICS = (
    ROOT / "rrn_reduction_20261005/wfi01_tile_log_statistics_512_count_corrected.csv"
)
EMCEE_OUTPUT = ROOT / "wfi01_joint_complex_q_cold_fixed_hgcdte_emcee1000"
MAP_OUTPUT = ROOT / "wfi01_joint_complex_q_cold_fixed_hgcdte_allstages_map10000"
REFERENCE_DATA_DIR = ROOT / "external_reference_data"


def main():
    production._PRODUCTION_STATISTICS_PATH = STATISTICS
    production.SED_URL = str(
        REFERENCE_DATA_DIR / "WFI_RCS_LEDP_spectral_energy_distribution.ecsv"
    )
    production.QE_URL = str(REFERENCE_DATA_DIR / "WFI_Quantum_efficiency.ecsv")
    args = production.parse_args()
    args.all_sca = False
    args.max_tiles = 1
    args.tile_statistics = STATISTICS
    args.output_dir = MAP_OUTPUT
    args.map_init_from = EMCEE_OUTPUT
    args.map_init_allow_tile_subset = False
    args.fit_complex_q = True
    args.complex_q_initial_parameters = production.COMPLEX_Q_INITIAL.copy()
    args.fixed_absolute_hgcdte_scales = True
    args.shared_absolute_hgcdte_scales = False
    args.map_steps = 10000
    args.map_only = True
    production._fit_one(args)


if __name__ == "__main__":
    main()
