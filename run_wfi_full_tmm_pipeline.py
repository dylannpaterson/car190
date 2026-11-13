"""Run the WFI full-TMM production pipeline."""

from wfi_pipeline.run_wfi_pipeline import main_for_route
from wfi_pipeline.wfi_pipeline_defaults import FULL_TMM


def main(argv=None):
    main_for_route(FULL_TMM, argv)


if __name__ == "__main__":
    main()
