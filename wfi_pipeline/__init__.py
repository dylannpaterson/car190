"""Implementation package for the supported local WFI pipeline frontends.

The implementation modules retain their historical flat imports internally.
Expose this directory for those imports while keeping the repository root
limited to the user-facing pipeline commands.
"""

from pathlib import Path
import sys


_MODULE_DIR = str(Path(__file__).resolve().parent)
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)
