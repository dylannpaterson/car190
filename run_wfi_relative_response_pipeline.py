"""Run the default pinned-cache WFI relative-response pipeline."""

from wfi_pipeline.run_wfi_pipeline import main_for_route
from wfi_pipeline.wfi_pipeline_defaults import RELATIVE_RESPONSE


def main(argv=None):
    main_for_route(RELATIVE_RESPONSE, argv)


if __name__ == "__main__":
    main()
