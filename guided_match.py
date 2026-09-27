"""Guided coarse-to-fine matching for cross-sensor lunar registration.

ISRO-aligned design:
  1) Coarse geometry (LoFTR-ROI / SIFT) → H_coarse
  2) Warp source into reference frame (removes scale/viewpoint bulk)
  3) Tile-wise local matching on the aligned pair (uniform coverage)
  4) Map matches back to original source coordinates
  5) Fine MAGSAC + sub-pixel on the enriched set

Remote-sensing register-then-rematch cascade for OHRC↔LROC under Cloud RAM limits.
"""
from __future__ import annotations

import gc
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _as_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        g = img.astype(np.float32)
        g = g - g.min()
        g = g / max(float(g.max()), 1e-6) * 255.0
        return g.astype(np.uint8)
    return img


def illumination_normalize(gray: np.ndarray) -> np.ndarray:
    """CLAHE + mild unsharp — reduces sun-angle effects without destroying texture."""
    g = _as_gray_u8(gray)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g = clahe.apply(g)
    blur = cv2.GaussianBlur(g, (0, 0), 1.2)
    g = cv2.addWeighted(g, 1.35, blur, -0.35, 0)
    return g


def warp_source_to_reference(
    source_gray: np.ndarray,
    H: np.ndarray,
    ref_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp source into reference canvas. Returns (warped_gray, valid_mask)."""
    hr, wr = int(ref_shape[0]), int(ref_shape[1])
    H = np.asarray(H, dtype=np.float64)
    warped = cv2.warpPerspective(
        _as_gray_u8(source_gray),
        H,
        (wr, hr),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    ones = np.ones(source_gray.shape[:2], dtype=np.uint8) * 255
    mask = cv2.warpPerspective(
        ones, H, (wr, hr), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return warped, mask


def _tile_boxes(h: int, w: int, tile: int = 384, overlap: float = 0.25) -> List[Tuple[int, int, int, int]]:
    step = max(64, int(tile * (1.0 - overlap)))
    boxes = []
    y = 0
    while y < h:
        x = 0
        y1 = min(h, y + tile)
        while x < w:
            x1 = min(w, x + tile)
            if (x1 - x) >= 96 and (y1 - y) >= 96:
                boxes.append((x, y, x1, y1))
            if x1 >= w:
                break
            x += step
        if y1 >= h:
            break
        y += step
    return boxes


def tile_sift_match(
    warped_src: np.ndarray,
    reference_gray: np.ndarray,
    warped_mask: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
    tile: int = 384,
    nfeatures: int = 800,
    ratio_threshold: float = 0.78,
    max_tiles: int = 16,
) -> List[tuple]:
    """Match warped-source ↔ reference on overlapping tiles for uniform coverage."""
    a = illumination_normalize(warped_src)
    b = illumination_normalize(reference_gray)
    h, w = a.shape[:2]
    boxes = _tile_boxes(h, w, tile=tile, overlap=0.30)
    if warped_mask is not None:
        scored = []
        for box in boxes:
            x0, y0, x1, y1 = box
            m = warped_mask[y0:y1, x0:x1]
            scored.append((float(np.count_nonzero(m)), box))
        scored.sort(key=lambda t: -t[0])
        boxes = [b for s, b in scored if s > 500][:max_tiles]
    else:
        boxes = boxes[:max_tiles]

    sift = cv2.SIFT_create(nfeatures=int(nfeatures), contrastThreshold=0.02, edgeThreshold=12)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    out: List[tuple] = []
    for x0, y0, x1, y1 in boxes:
        ta = a[y0:y1, x0:x1]
        tb = b[y0:y1, x0:x1]
        ma = warped_mask[y0:y1, x0:x1] if warped_mask is not None else None
        mb = reference_mask[y0:y1, x0:x1] if reference_mask is not None else None
        kpa, da = sift.detectAndCompute(ta, ma)
        kpb, db = sift.detectAndCompute(tb, mb)
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            continue
        knn = bf.knnMatch(da, db, k=2)
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if n.distance < 1e-9:
                continue
            if m.distance / n.distance <= ratio_threshold:
                pa = kpa[m.queryIdx].pt
                pb = kpb[m.trainIdx].pt
                out.append(
                    (
                        (float(pa[0] + x0), float(pa[1] + y0)),
                        (float(pb[0] + x0), float(pb[1] + y0)),
                        float(1.0 - m.distance / max(n.distance, 1e-9)),
                    )
                )
    pool = {}
    for s, r, sc in out:
        key = (round(s[0], 0), round(s[1], 0), round(r[0], 0), round(r[1], 0))
        if key not in pool or sc > pool[key][2]:
            pool[key] = (s, r, sc)
    return sorted(pool.values(), key=lambda z: -z[2])


def map_warped_to_source(points_warped: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Inverse-project points from reference-aligned frame back to original source."""
    H = np.asarray(H, dtype=np.float64)
    Hinv = np.linalg.inv(H)
    pts = np.asarray(points_warped, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    proj = (Hinv @ np.hstack([pts, ones]).T).T
    return proj[:, :2] / np.maximum(proj[:, 2:3], 1e-12)


def guided_rematch(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    H_coarse: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
    tile: int = 384,
    nfeatures: int = 700,
    ratio_threshold: float = 0.78,
    max_tiles: int = 12,
) -> dict:
    """Full guided cascade. Correspondences in ORIGINAL source / reference coords."""
    hr, wr = reference_gray.shape[:2]
    warped, wmask = warp_source_to_reference(source_gray, H_coarse, (hr, wr))
    if reference_mask is not None:
        wmask = cv2.bitwise_and(wmask, reference_mask.astype(np.uint8))
    pairs = tile_sift_match(
        warped,
        reference_gray,
        warped_mask=wmask,
        reference_mask=reference_mask,
        tile=tile,
        nfeatures=nfeatures,
        ratio_threshold=ratio_threshold,
        max_tiles=max_tiles,
    )
    if not pairs:
        del warped, wmask
        gc.collect()
        return {"correspondences": [], "tile_matches": 0, "mode": "guided_empty"}

    pts_w = np.array([p[0] for p in pairs], dtype=np.float64)
    pts_r = np.array([p[1] for p in pairs], dtype=np.float64)
    scores = [p[2] for p in pairs]
    pts_s = map_warped_to_source(pts_w, H_coarse)

    hs, ws = source_gray.shape[:2]
    corrs = []
    for i in range(len(pts_s)):
        x, y = float(pts_s[i, 0]), float(pts_s[i, 1])
        xr, yr = float(pts_r[i, 0]), float(pts_r[i, 1])
        if 0 <= x < ws and 0 <= y < hs and 0 <= xr < wr and 0 <= yr < hr:
            corrs.append(((x, y), (xr, yr), float(scores[i])))

    del warped, wmask
    gc.collect()
    return {
        "correspondences": corrs,
        "tile_matches": len(corrs),
        "mode": "guided_warp_rematch",
    }
