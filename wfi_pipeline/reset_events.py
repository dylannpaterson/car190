"""Load and validate the shared WFI reset/decon event configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

import numpy as np


DEFAULT_RESET_CONFIG = Path(__file__).with_name("reset_events.yaml")
_EVENT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def load_reset_config(path=None):
    """Load reset events and their thickness-prior settings from YAML.

    The first event is the program-1063 reset found in each statistics file.
    Remaining events have explicit elapsed-day boundaries. Adding another
    event is a YAML-only operation for the production fit and H5 updater.
    """
    path = DEFAULT_RESET_CONFIG if path is None else Path(path).expanduser()
    path = path.resolve()
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Reading the WFI reset configuration requires PyYAML; "
            "install it with `python -m pip install PyYAML`."
        ) from error
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError:
        raise
    except Exception as error:
        raise ValueError(f"Could not parse reset configuration {path}: {error}") from error
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError(f"Unsupported reset configuration version in {path}")
    events = document.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("Reset configuration must contain a nonempty events list")

    normalized = []
    names = set()
    for index, raw in enumerate(events):
        if not isinstance(raw, dict):
            raise ValueError(f"Reset event {index} must be a mapping")
        event = deepcopy(raw)
        name = event.get("name")
        if not isinstance(name, str) or not _EVENT_NAME.fullmatch(name):
            raise ValueError(
                f"Reset event {index} name must match {_EVENT_NAME.pattern}"
            )
        if name in names:
            raise ValueError(f"Duplicate reset event name: {name}")
        names.add(name)

        if index == 0:
            if event.get("day_source") != "program1063" or "day" in event:
                raise ValueError(
                    "The first reset event must use day_source: program1063"
                )
        else:
            day = float(event.get("day", np.nan))
            if not np.isfinite(day):
                raise ValueError(f"Reset event {name} requires a finite day")
            event["day"] = day
            if event.get("day_source") is not None:
                raise ValueError(f"Only the first reset event may use day_source ({name})")

        prior = event.get("thickness_prior", {})
        if not isinstance(prior, dict):
            raise ValueError(f"{name}.thickness_prior must be a mapping")
        median_nm = float(prior.get("median_nm", 10.0))
        log_sigma = float(prior.get("log_sigma", 1.5))
        if not np.isfinite(median_nm) or median_nm <= 0.0:
            raise ValueError(f"{name} thickness prior median_nm must be positive")
        if not np.isfinite(log_sigma) or log_sigma <= 0.0:
            raise ValueError(f"{name} thickness prior log_sigma must be positive")
        event["thickness_prior"] = {
            "median_nm": median_nm,
            "log_sigma": log_sigma,
        }

        h5_prior = event.get("h5_prior", {})
        if not isinstance(h5_prior, dict):
            raise ValueError(f"{name}.h5_prior must be a mapping")
        h5_sigma = float(h5_prior.get("log_sigma", log_sigma))
        if not np.isfinite(h5_sigma) or h5_sigma <= 0.0:
            raise ValueError(f"{name} H5 prior log_sigma must be positive")
        h5_center = h5_prior.get("center", "configured_median")
        if h5_center not in ("configured_median", "base_h5"):
            raise ValueError(
                f"{name} H5 prior center must be configured_median or base_h5"
            )
        event["h5_prior"] = {"center": h5_center, "log_sigma": h5_sigma}

        warmup = event.get("relative_response_warmup")
        if warmup is not None:
            if not isinstance(warmup, dict):
                raise ValueError(f"{name}.relative_response_warmup must be a mapping")
            start_nm = float(warmup.get("start_nm", np.nan))
            warmup_sigma = float(warmup.get("log_sigma", np.nan))
            if not np.isfinite(start_nm) or start_nm <= 0.0:
                raise ValueError(f"{name} warmup start_nm must be positive")
            if not np.isfinite(warmup_sigma) or warmup_sigma <= 0.0:
                raise ValueError(f"{name} warmup log_sigma must be positive")
            event["relative_response_warmup"] = {
                "start_nm": start_nm,
                "log_sigma": warmup_sigma,
            }
        normalized.append(event)

    explicit_days = [event["day"] for event in normalized[1:]]
    if np.any(np.diff(explicit_days) <= 0.0):
        raise ValueError("Explicit reset-event days must be strictly increasing")
    return {
        "path": path,
        "events": tuple(normalized),
        "sha256": _sha256(path),
    }


def resolve_reset_events(config, statistics_reset_day):
    """Attach the statistics-derived first day and return ordered event records."""
    events = config["events"]
    reset_day = float(statistics_reset_day)
    if not np.isfinite(reset_day):
        raise ValueError("Statistics-derived reset day must be finite")
    resolved = []
    for index, event in enumerate(events):
        item = deepcopy(event)
        if index == 0:
            item["day"] = reset_day
        resolved.append(item)
    days = np.asarray([event["day"] for event in resolved], dtype=float)
    if np.any(np.diff(days) <= 0.0):
        raise ValueError(
            "Configured reset/decon events must follow the statistics-derived "
            "program-1063 reset in strictly increasing day order"
        )
    return tuple(resolved)


def _sha256(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
