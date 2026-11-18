"""Resolve pipeline output directories used by paper analysis tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PaperRunPaths:
    run_root: Path
    statistics_dir: Path
    production_dir: Path
    input_dir: Path | None


def resolve_pipeline_run(path: Path) -> PaperRunPaths:
    """Resolve a pipeline run root, date root, or its ``production`` parent.

    Supported inputs include the pipeline output directory itself, its date
    directory (for example ``20261117``), or the date directory's
    ``production`` directory. The pipeline manifest is authoritative for the
    corresponding input-data root.
    """
    requested = Path(path).expanduser().resolve()
    candidates = (
        requested,
        requested / "production/pipeline_runs/relative_response",
        requested / "pipeline_runs/relative_response",
    )
    run_root = next(
        (candidate for candidate in candidates if (candidate / "pipeline_manifest.json").is_file()),
        None,
    )
    if run_root is None:
        raise FileNotFoundError(
            f"No relative-response pipeline manifest found at or below {requested}"
        )

    manifest = json.loads((run_root / "pipeline_manifest.json").read_text())
    statistics_dir = run_root / "corrected_statistics_4h_thinned"
    production_dir = run_root / "production"
    if not statistics_dir.is_dir():
        raise FileNotFoundError(
            f"Pipeline statistics are not ready: {statistics_dir}"
        )
    if not production_dir.is_dir():
        raise FileNotFoundError(
            f"Pipeline production outputs are not ready: {production_dir}"
        )

    input_value = manifest.get("input_path")
    input_dir = Path(input_value).expanduser().resolve() if input_value else None
    return PaperRunPaths(
        run_root=run_root,
        statistics_dir=statistics_dir,
        production_dir=production_dir,
        input_dir=input_dir,
    )
