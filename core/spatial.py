"""
Uniform Match-Point Distribution and Spatial Balancing.

Enforces grid-based candidate selection:
- Prevents correspondence clustering in high-contrast crater rims
- Promotes uniform distribution across the entire overlap footprint
- Computes genuine spatial distribution metrics (Coverage %, Occupied Cells, Mean/Max matches/cell)
"""
from typing import Tuple, List, Dict, Any, Optional
import numpy as np


def compute_grid_key(point: Tuple[float, float], shape: Tuple[int, int], grid_size: int) -> Tuple[int, int]:
    """Map a point (x, y) to grid cell (row, col) in range 0..grid_size-1."""
    h, w = shape[:2]
    x, y = point
    c = min(grid_size - 1, max(0, int(x * grid_size / max(w, 1))))
    r = min(grid_size - 1, max(0, int(y * grid_size / max(h, 1))))
    return r, c


def balance_correspondences_spatially(
    matches: List[Tuple[Tuple[float, float], Tuple[float, float], float, str]],
    source_shape: Tuple[int, int],
    reference_shape: Tuple[int, int],
    grid_size: int = 10,
    cell_limit: int = 20,
    higher_score_is_better: bool = True,
) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float], float, str]], Dict[str, Any]]:
    """Enforce uniform grid distribution on candidate correspondences.

    matches: list of (src_pt, ref_pt, score, method)
    Returns:
    - filtered_matches
    - spatial_metrics dict
    """
    total_before = len(matches)
    if total_before == 0:
        empty_metrics = {
            "grid_size": grid_size,
            "total_cells": grid_size * grid_size,
            "occupied_cells": 0,
            "spatial_coverage_pct": 0.0,
            "mean_matches_per_cell": 0.0,
            "max_matches_per_cell": 0,
            "matches_before": 0,
            "matches_after": 0,
            "cell_counts": np.zeros((grid_size, grid_size), dtype=int).tolist(),
        }
        return [], empty_metrics

    # Sort candidates by score
    sorted_matches = sorted(matches, key=lambda z: -z[2] if higher_score_is_better else z[2])

    selected = []
    scount: Dict[Tuple[int, int], int] = {}
    rcount: Dict[Tuple[int, int], int] = {}

    for s, r, score, method in sorted_matches:
        sk = compute_grid_key(s, source_shape, grid_size)
        rk = compute_grid_key(r, reference_shape, grid_size)

        if scount.get(sk, 0) >= cell_limit or rcount.get(rk, 0) >= cell_limit:
            continue

        selected.append((s, r, score, method))
        scount[sk] = scount.get(sk, 0) + 1
        rcount[rk] = rcount.get(rk, 0) + 1

    # Compute distribution statistics on selected matches
    grid_counts = np.zeros((grid_size, grid_size), dtype=int)
    for s, _, _, _ in selected:
        r_idx, c_idx = compute_grid_key(s, source_shape, grid_size)
        grid_counts[r_idx, c_idx] += 1

    occupied = int((grid_counts > 0).sum())
    total_cells = grid_size * grid_size
    coverage_fraction = float(occupied / max(total_cells, 1))
    mean_matches = float(len(selected) / max(occupied, 1)) if occupied > 0 else 0.0
    max_matches = int(grid_counts.max()) if len(selected) > 0 else 0

    spatial_metrics = {
        "grid_size": grid_size,
        "total_cells": total_cells,
        "occupied_cells": occupied,
        "spatial_coverage_fraction": coverage_fraction,
        "spatial_coverage_pct": round(coverage_fraction * 100, 1),
        "mean_matches_per_cell": round(mean_matches, 2),
        "max_matches_per_cell": max_matches,
        "matches_before": total_before,
        "matches_after": len(selected),
        "cell_counts": grid_counts.tolist(),
    }

    return selected, spatial_metrics
