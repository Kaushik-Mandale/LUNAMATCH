"""
Geographic Footprint Validation & Overlap ROI Extraction.

Implements:
1. Exact footprint bounds calculation from 4-corner polygons.
2. Strict overlap gate: rejects non-overlapping pairs with scientific rationale.
3. Common geographic overlap extraction into pixel ROIs for source and reference.
"""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np


def get_footprint_bounds(meta: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Extract (min_lat, max_lat, min_lon, max_lon) from metadata footprint."""
    if not isinstance(meta, dict):
        return None

    fp = meta.get("footprint")
    if isinstance(fp, dict):
        corners = ["upper_left", "upper_right", "lower_left", "lower_right"]
        pts = [fp.get(c) for c in corners if isinstance(fp.get(c), (list, tuple)) and len(fp.get(c)) == 2]
        if len(pts) == 4:
            lats = [float(p[0]) for p in pts]
            lons = [float(p[1]) for p in pts]
            return min(lats), max(lats), min(lons), max(lons)

    # Fallback to direct latitude / longitude ranges if available
    lat = meta.get("latitude") or meta.get("latitude_deg")
    lon = meta.get("longitude") or meta.get("longitude_deg")
    if lat is not None and lon is not None:
        try:
            flat = float(lat)
            flon = float(lon)
            return flat - 0.1, flat + 0.1, flon - 0.1, flon + 0.1
        except Exception:
            pass

    return None


def evaluate_footprint_overlap(source_meta: Dict[str, Any], reference_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate geographic footprint overlap between source and reference imagery.

    Enforces the strict gate rule:
    If no overlap:
        gate_message = "No meaningful geographic overlap detected.
Matching was not executed because the source and reference
images are geographically inconsistent."
    """
    src_bounds = get_footprint_bounds(source_meta)
    ref_bounds = get_footprint_bounds(reference_meta)

    if src_bounds is None or ref_bounds is None:
        return {
            "available": False,
            "has_overlap": None,
            "overlap_area_sq_deg": 0.0,
            "intersection_bounds": None,
            "source_bounds": src_bounds,
            "reference_bounds": ref_bounds,
            "gate_passed": True,  # Cannot gate out if footprints unavailable
            "gate_message": "Geographic coordinates unavailable for one or both products. Pipeline proceeding with cautious feature-space validation.",
            "status": "warning",
        }

    s_min_lat, s_max_lat, s_min_lon, s_max_lon = src_bounds
    r_min_lat, r_max_lat, r_min_lon, r_max_lon = ref_bounds

    inter_min_lat = max(s_min_lat, r_min_lat)
    inter_max_lat = min(s_max_lat, r_max_lat)
    inter_min_lon = max(s_min_lon, r_min_lon)
    inter_max_lon = min(s_max_lon, r_max_lon)

    lat_overlap = inter_max_lat - inter_min_lat
    lon_overlap = inter_max_lon - inter_min_lon

    has_overlap = (lat_overlap > 0) and (lon_overlap > 0)

    if not has_overlap:
        gate_msg = (
            "No meaningful geographic overlap detected.\n"
            "Matching was not executed because the source and reference\n"
            "images are geographically inconsistent."
        )
        return {
            "available": True,
            "has_overlap": False,
            "overlap_area_sq_deg": 0.0,
            "intersection_bounds": None,
            "source_bounds": src_bounds,
            "reference_bounds": ref_bounds,
            "gate_passed": False,
            "gate_message": gate_msg,
            "status": "rejected",
        }

    overlap_area = float(lat_overlap * lon_overlap)
    src_area = max((s_max_lat - s_min_lat) * (s_max_lon - s_min_lon), 1e-9)
    ref_area = max((r_max_lat - r_min_lat) * (r_max_lon - r_min_lon), 1e-9)
    overlap_fraction = float(overlap_area / min(src_area, ref_area))

    return {
        "available": True,
        "has_overlap": True,
        "overlap_area_sq_deg": overlap_area,
        "overlap_fraction": overlap_fraction,
        "intersection_bounds": (inter_min_lat, inter_max_lat, inter_min_lon, inter_max_lon),
        "source_bounds": src_bounds,
        "reference_bounds": ref_bounds,
        "gate_passed": True,
        "gate_message": f"Geographic overlap confirmed ({overlap_fraction*100:.1f}% common coverage, {overlap_area:.4f} sq. deg).",
        "status": "success",
    }


def compute_overlap_pixel_roi(
    image_shape: Tuple[int, int],
    image_bounds: Tuple[float, float, float, float],
    intersection_bounds: Tuple[float, float, float, float],
    padding_fraction: float = 0.05,
) -> Tuple[int, int, int, int]:
    """Map geographic intersection bounds to image pixel crop coordinates (y_min, y_max, x_min, x_max).

    image_bounds: (min_lat, max_lat, min_lon, max_lon)
    intersection_bounds: (inter_min_lat, inter_max_lat, inter_min_lon, inter_max_lon)
    """
    h, w = image_shape[:2]
    min_lat, max_lat, min_lon, max_lon = image_bounds
    i_min_lat, i_max_lat, i_min_lon, i_max_lon = intersection_bounds

    lat_span = max(max_lat - min_lat, 1e-9)
    lon_span = max(max_lon - min_lon, 1e-9)

    # Pixel coords (assuming North is Up, West is Left)
    # y=0 is top (max_lat), y=h is bottom (min_lat)
    y_min_norm = (max_lat - i_max_lat) / lat_span
    y_max_norm = (max_lat - i_min_lat) / lat_span

    # x=0 is left (min_lon), x=w is right (max_lon)
    x_min_norm = (i_min_lon - min_lon) / lon_span
    x_max_norm = (i_max_lon - min_lon) / lon_span

    # Add margin padding
    pad_y = (y_max_norm - y_min_norm) * padding_fraction
    pad_x = (x_max_norm - x_min_norm) * padding_fraction

    y0 = max(0, int(round((y_min_norm - pad_y) * h)))
    y1 = min(h, int(round((y_max_norm + pad_y) * h)))
    x0 = max(0, int(round((x_min_norm - pad_x) * w)))
    x1 = min(w, int(round((x_max_norm + pad_x) * w)))

    if y1 <= y0:
        y0, y1 = 0, h
    if x1 <= x0:
        x0, x1 = 0, w

    return y0, y1, x0, x1


def extract_overlap_rois(
    src_img: np.ndarray,
    ref_img: np.ndarray,
    source_meta: Dict[str, Any],
    reference_meta: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extract Source Overlap ROI and Reference Overlap ROI if geographic overlap exists."""
    overlap_info = evaluate_footprint_overlap(source_meta, reference_meta)
    if not overlap_info["has_overlap"] or overlap_info["intersection_bounds"] is None:
        return src_img, ref_img, {"roi_applied": False, "reason": "No valid overlap bounds"}

    src_bounds = overlap_info["source_bounds"]
    ref_bounds = overlap_info["reference_bounds"]
    inter_bounds = overlap_info["intersection_bounds"]

    sy0, sy1, sx0, sx1 = compute_overlap_pixel_roi(src_img.shape, src_bounds, inter_bounds)
    ry0, ry1, rx0, rx1 = compute_overlap_pixel_roi(ref_img.shape, ref_bounds, inter_bounds)

    src_roi = src_img[sy0:sy1, sx0:sx1]
    ref_roi = ref_img[ry0:ry1, rx0:rx1]

    details = {
        "roi_applied": True,
        "source_crop_box": (sy0, sy1, sx0, sx1),
        "reference_crop_box": (ry0, ry1, rx0, rx1),
        "source_roi_shape": src_roi.shape,
        "reference_roi_shape": ref_roi.shape,
        "overlap_area_sq_deg": overlap_info["overlap_area_sq_deg"],
    }

    return src_roi, ref_roi, details
