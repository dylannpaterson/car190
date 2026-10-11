"""Run an opt-in WFI02 linear-rate-ramp EMCEE warm-up test."""

from pathlib import Path

import run_wfi_tmm_emcee_warmup as emcee_warmup


ROOT = Path(__file__).resolve().parent
STATISTICS = (
    ROOT
    / "20261008/corrected_statistics_program1063_4h/production/wfi02/"
    "tile_statistics/wfi02_tile_log_statistics_512_tile_000.csv"
)
OUTPUT_DIR = (
    ROOT
    / "20261008/corrected_statistics_program1063_4h/diagnostics/"
    "wfi02_linear_rate_emcee_warmup"
)


def main():
    emcee_warmup.run_emcee(
        STATISTICS,
        OUTPUT_DIR,
        trajectory="linear-rate-ramp",
    )


if __name__ == "__main__":
    main()
