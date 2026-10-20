"""Shared size-aware paths and tile-layout validation for WFI production tools."""

import argparse
import math
from pathlib import Path
import re


DEFAULT_TILE_SIZE = 256
DETECTOR_SIZE = 4096
PRODUCTION_DIR_NAME = "production"
TILE_DIR_PATTERN = re.compile(r"tile_(\d+)(?:_least_squares)?$")


def positive_integer(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def production_output_root(root, tile_size=DEFAULT_TILE_SIZE):
    root = Path(root) / PRODUCTION_DIR_NAME
    if tile_size == DEFAULT_TILE_SIZE:
        return root
    return root / f"tile_size_{tile_size}"


def detector_output_root(root, detector, tile_size=DEFAULT_TILE_SIZE):
    return production_output_root(root, tile_size) / str(detector).lower()


def corrected_statistics_path(statistics_dir, detector, tile_size=DEFAULT_TILE_SIZE):
    return Path(statistics_dir) / (
        f"{str(detector).lower()}_tile_log_statistics_{tile_size}_count_corrected.csv"
    )


def tile_statistics_path(statistics_root, detector, tile_size, tile_id):
    return Path(statistics_root) / (
        f"{str(detector).lower()}_tile_log_statistics_{tile_size}_tile_{tile_id:03d}.csv"
    )


def tile_grid_size_for_tile_size(tile_size):
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    return (DETECTOR_SIZE + tile_size - 1) // tile_size


def complete_tile_ids(tile_ids, label="tile IDs"):
    values = sorted({int(tile_id) for tile_id in tile_ids})
    if not values:
        raise ValueError(f"{label} are empty")
    expected = list(range(values[-1] + 1))
    if values != expected:
        raise ValueError(
            f"{label} must be contiguous from 0; found {values}"
        )
    grid_size = math.isqrt(len(values))
    if grid_size * grid_size != len(values):
        raise ValueError(
            f"{label} must describe a square detector grid; found {len(values)} tiles"
        )
    return tuple(values)


def partial_tile_grid_size(tile_ids):
    values = [int(tile_id) for tile_id in tile_ids]
    if not values or min(values) < 0:
        raise ValueError("tile IDs must contain at least one non-negative value")
    count = max(values) + 1
    grid_size = math.isqrt(count)
    if grid_size * grid_size < count:
        grid_size += 1
    return grid_size


def tile_directory_entries(fit_root):
    entries = []
    for path in sorted(Path(fit_root).glob("tile_*")):
        match = TILE_DIR_PATTERN.fullmatch(path.name)
        if match is not None:
            entries.append((int(match.group(1)), path))
    entries.sort(key=lambda item: item[0])
    return entries
