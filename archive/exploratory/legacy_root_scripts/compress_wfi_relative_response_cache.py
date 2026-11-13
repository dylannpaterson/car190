"""Write compact float32, linearly interpolated response caches."""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spacing-nm", type=float, default=2.0)
    parser.add_argument("--tile-id", type=int, action="append")
    args = parser.parse_args()
    if args.spacing_nm <= 0.0:
        raise ValueError("spacing-nm must be positive")
    return args


def knot_indices(grid, spacing_nm):
    native_step = float(np.median(np.diff(grid)))
    stride = int(round(float(spacing_nm) / native_step))
    if stride < 1:
        raise ValueError("spacing-nm is smaller than the native cache grid")
    indices = np.arange(0, grid.size, stride, dtype=int)
    if indices[-1] != grid.size - 1:
        indices = np.append(indices, grid.size - 1)
    return indices


def main():
    args = parse_args()
    source_paths = sorted(Path(args.source_root).glob("tile_*/relative_response.npz"))
    if args.tile_id is not None:
        selected = {int(tile_id) for tile_id in args.tile_id}
        source_paths = [
            path
            for path in source_paths
            if int(path.parent.name.split("_")[-1]) in selected
        ]
    if not source_paths:
        raise FileNotFoundError(f"No response caches found under {args.source_root}")

    for source_path in source_paths:
        with np.load(source_path, allow_pickle=False) as source:
            grid = np.asarray(source["grid_nm"], dtype=float)
            indices = knot_indices(grid, args.spacing_nm)
            payload = {name: np.asarray(source[name]) for name in source.files}
            payload["grid_nm"] = grid[indices].astype(np.float32)
            payload["relative_log_response"] = np.asarray(
                source["relative_log_response"], dtype=np.float32
            )[:, indices]
            payload["response_representation"] = np.asarray("linear_grid")
            payload["compression_spacing_nm"] = np.asarray(
                float(np.median(np.diff(grid[indices])))
            )
        tile_id = int(source_path.parent.name.split("_")[-1])
        output_path = (
            Path(args.output_root) / f"tile_{tile_id:03d}" / "relative_response.npz"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **payload)
        print(
            f"Wrote {output_path} ({indices.size} knots, "
            f"{output_path.stat().st_size / 1024.0:.1f} KiB)",
            flush=True,
        )


if __name__ == "__main__":
    main()
