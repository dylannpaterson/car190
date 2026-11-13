# Local WFI pipeline implementation

These modules implement the local fit-to-product pipeline. Operators should use
the two repository-root frontends:

- `run_wfi_relative_response_pipeline.py` — default cached relative-response route;
- `run_wfi_full_tmm_pipeline.py` — occasional full-TMM route.

Both accept only `--input-path` and `--output-path`. The stage runners and model,
bundle, H5, and diagnostic code in this package are internal implementation
modules; the frontends orchestrate them and record their outputs/logs in one run
directory. Nexus reduction remains a separate handoff documented in
`docs/WFI_NEXUS_REDUCTION_HANDOFF.md`.

Run a lower-level stage manually only for focused development or reproduction;
ordinary production runs should start from one of the two root frontends.
