"""
Registration Engine & Visual Verification Tools.

Implements:
1. Strict Source → Reference transformation (Source warped into Reference coordinate frame).
2. Dynamic Alpha Overlay with transparency control.
3. Diagnostic Checkerboard comparison (alternating reference & registered source patches).
4. Separate inlier vs outlier correspondence rendering.
"""
from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np


def warp_source_to_reference(
    source_img: np.ndarray,
    reference_shape: Tuple[int, int],
    H: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp SOURCE / MOVING image into REFERENCE / FIXED coordinate frame.

    x_ref = H * x_src
    H maps source working pixels to reference working pixels.
    """
    ref_h, ref_w = reference_shape[:2]

    registered_source = cv2.warpPerspective(source_img, H, (ref_w, ref_h), flags=cv2.INTER_LINEAR)

    if source_mask is not None:
        valid_mask = cv2.warpPerspective(source_mask, H, (ref_w, ref_h), flags=cv2.INTER_NEAREST)
    else:
        # Create mask from non-zero transformed pixels
        valid_mask = (registered_source > 0).astype(np.uint8) * 255

    # Zero out invalid regions
    registered_source[valid_mask == 0] = 0

    return registered_source, valid_mask, H


def create_alpha_overlay(
    reference_img: np.ndarray,
    registered_source: np.ndarray,
    valid_mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend Reference image and Registered Source image with alpha transparency.

    alpha: fraction of registered source (1.0 = 100% source, 0.0 = 100% reference).
    """
    ref = reference_img.copy()
    src = registered_source.copy()

    if ref.ndim == 2:
        ref_rgb = cv2.cvtColor(ref, cv2.COLOR_GRAY2RGB)
    else:
        ref_rgb = ref

    if src.ndim == 2:
        # Tint registered source slightly (e.g. cyan/gold or natural) to make misalignment visible
        src_rgb = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    else:
        src_rgb = src

    alpha = max(0.0, min(1.0, float(alpha)))

    # Blend in valid overlap region
    overlap = (valid_mask > 0)
    blended = ref_rgb.copy()
    blended[overlap] = (
        (1.0 - alpha) * ref_rgb[overlap].astype(np.float32)
        + alpha * src_rgb[overlap].astype(np.float32)
    ).astype(np.uint8)

    return blended


def create_checkerboard_comparison(
    reference_img: np.ndarray,
    registered_source: np.ndarray,
    tile_size: int = 64,
) -> np.ndarray:
    """Generate a checkerboard visualization alternating between Reference and Registered Source.

    Allows immediate visual inspection of crater rim alignment and fracture continuation across tiles.
    """
    ref = reference_img.copy()
    src = registered_source.copy()

    if ref.ndim == 2:
        ref_rgb = cv2.cvtColor(ref, cv2.COLOR_GRAY2RGB)
    else:
        ref_rgb = ref

    if src.ndim == 2:
        src_rgb = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    else:
        src_rgb = src

    h, w = ref_rgb.shape[:2]
    # Ensure registered source matches reference dimensions
    if src_rgb.shape[:2] != (h, w):
        src_rgb = cv2.resize(src_rgb, (w, h))

    checkerboard = np.zeros_like(ref_rgb)
    n_rows = (h + tile_size - 1) // tile_size
    n_cols = (w + tile_size - 1) // tile_size

    for r in range(n_rows):
        y0 = r * tile_size
        y1 = min(h, y0 + tile_size)
        for c in range(n_cols):
            x0 = c * tile_size
            x1 = min(w, x0 + tile_size)
            if (r + c) % 2 == 0:
                checkerboard[y0:y1, x0:x1] = ref_rgb[y0:y1, x0:x1]
            else:
                checkerboard[y0:y1, x0:x1] = src_rgb[y0:y1, x0:x1]

    # Draw subtle grid lines along checker borders
    for r in range(1, n_rows):
        y = r * tile_size
        if y < h:
            cv2.line(checkerboard, (0, y), (w, y), (255, 200, 0), 1)
    for c in range(1, n_cols):
        x = c * tile_size
        if x < w:
            cv2.line(checkerboard, (x, 0), (x, h), (255, 200, 0), 1)

    return checkerboard


def draw_inlier_outlier_matches(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    source_pts: np.ndarray,
    reference_pts: np.ndarray,
    inlier_mask: np.ndarray,
    max_draw: int = 150,
) -> np.ndarray:
    """Render side-by-side correspondence plot with verified inliers (Green) and outliers (Red/Gray)."""
    src = source_img.copy()
    ref = reference_img.copy()

    if src.ndim == 2:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    if ref.ndim == 2:
        ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)

    h_src, w_src = src.shape[:2]
    h_ref, w_ref = ref.shape[:2]

    h_max = max(h_src, h_ref)
    canvas = np.zeros((h_max, w_src + w_ref, 3), dtype=np.uint8)
    canvas[:h_src, :w_src] = src
    canvas[:h_ref, w_src:w_src + w_ref] = ref

    n_pts = len(source_pts)
    if n_pts == 0:
        return canvas

    # Draw outliers first in muted red/gray, then inliers on top in vibrant green
    indices = np.arange(n_pts)
    outliers = indices[~inlier_mask]
    inliers = indices[inlier_mask]

    # Sample if too many points to avoid clutter
    if len(outliers) > max_draw // 2:
        outliers = np.random.choice(outliers, max_draw // 2, replace=False)
    if len(inliers) > max_draw:
        inliers = np.random.choice(inliers, max_draw, replace=False)

    for idx in outliers:
        sx, sy = int(round(source_pts[idx][0])), int(round(source_pts[idx][1]))
        rx, ry = int(round(reference_pts[idx][0])) + w_src, int(round(reference_pts[idx][1]))
        cv2.line(canvas, (sx, sy), (rx, ry), (80, 80, 200), 1, cv2.LINE_AA)
        cv2.circle(canvas, (sx, sy), 2, (80, 80, 200), -1)
        cv2.circle(canvas, (rx, ry), 2, (80, 80, 200), -1)

    for idx in inliers:
        sx, sy = int(round(source_pts[idx][0])), int(round(source_pts[idx][1]))
        rx, ry = int(round(reference_pts[idx][0])) + w_src, int(round(reference_pts[idx][1]))
        cv2.line(canvas, (sx, sy), (rx, ry), (0, 255, 120), 2, cv2.LINE_AA)
        cv2.circle(canvas, (sx, sy), 3, (0, 255, 120), -1)
        cv2.circle(canvas, (rx, ry), 3, (0, 255, 120), -1)

    return canvas
