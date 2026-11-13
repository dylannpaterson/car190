"""Make repository utilities importable for the root-flat legacy tests."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_MODULES = REPOSITORY_ROOT / "wfi_pipeline"
for path in (PIPELINE_MODULES, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
