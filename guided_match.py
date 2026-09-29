"""Guided coarse-to-fine matching for cross-sensor lunar registration.

1) Coarse matches → affine H
2) Warp source into reference frame
3) Tile-wise SIFT on CLAHE-aligned pair
4) Map matches back to original source coordinates
5) Merge with base matches
"""
from __future__ import annotations

import gc
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _as_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    g = np.asarray(img, dtype=np.float32)
    finite = g[np.isfinite(g)]
    if finite.size == 0:
        return np.zeros(g.shape[:2], np.uint8)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(g.shape[:2], np.uint8)
    x = np.clip((g - lo) / (hi - lo), 0, 1)
    return (x * 255.0).astype(np.uint8)


def illumination_normalize(gray: np.ndarray) -> np.ndarray:
    g = _as_gray_u8(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(g)


def to_H3(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (3, 3):
        return M
    if M.shape == (2, 3):
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = M
        return H
    raise ValueError(f"Unsupported transform shape {M.shape}")


def coarse_affine_from_corrs(
    corrs: list,
    threshold: float = 8.0,
    max_use: int = 48,
) -> Optional[np.ndarray]:
    if not corrs or len(corrs) < 6:
        return None
    ordered = sorted(corrs, key=lambda t: -float(t[2]))[:max_use]
    ps = np.array([c[0] for c in ordered], dtype=np.float32)
    pr = np.array([c[1] for c in ordered], dtype=np.float32)
    M, mask = cv2.estimateAffinePartial2D(
        ps, pr, method=cv2.RANSAC, ransacReprojThreshold=float(threshold),
        maxIters=4000, confidence=0.99,
    )
    if M is None:
        M, mask = cv2.estimateAffine2D(
            ps, pr, method=cv2.RANSAC, ransacReprojThreshold=float(threshold),
            maxIters=4000, confidence=0.99,
        )
    if M is None or mask is None or int(mask.ravel().sum()) < 4:
        return None
    return to_H3(M)


def warp_source_to_reference(
    source_gray: np.ndarray,
    H: np.ndarray,
    ref_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    hr, wr = int(ref_shape[0]), int(ref_shape[1])
    H = to_H3(H)
    warped = cv2.warpPerspective(
        _as_gray_u8(source_gray), H, (wr, hr),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    ones = np.ones(source_gray.shape[:2], dtype=np.uint8) * 255
    mask = cv2.warpPerspective(
        ones, H, (wr, hr), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return warped, mask


def _tile_boxes(h: int, w: int, tile: int = 384, overlap: float = 0.30) -> List[Tuple[int, int, int, int]]:
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
    nfeatures: int = 900,
    ratio_threshold: float = 0.80,
    max_tiles: int = 20,
) -> List[tuple]:
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
        boxes = [b for s, b in scored if s > 400][:max_tiles]
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
    H = to_H3(H)
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
    nfeatures: int = 900,
    ratio_threshold: float = 0.80,
    max_tiles: int = 20,
) -> dict:
    hr, wr = reference_gray.shape[:2]
    H_coarse = to_H3(H_coarse)
    warped, wmask = warp_source_to_reference(source_gray, H_coarse, (hr, wr))
    if reference_mask is not None:
        rm = reference_mask.astype(np.uint8)
        if rm.max() <= 1:
            rm = (rm * 255).astype(np.uint8)
        wmask = cv2.bitwise_and(wmask, rm)
    pairs = tile_sift_match(
        warped, reference_gray,
        warped_mask=wmask, reference_mask=reference_mask,
        tile=tile, nfeatures=nfeatures,
        ratio_threshold=ratio_threshold, max_tiles=max_tiles,
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


def enrich_with_guided(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    base_corrs: list,
    source_mask: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
    ransac_threshold: float = 8.0,
) -> Tuple[list, dict]:
    meta = {"guided": False, "tile_matches": 0, "mode": "none"}
    if not base_corrs or len(base_corrs) < 6:
        return list(base_corrs or []), meta
    H = coarse_affine_from_corrs(base_corrs, threshold=max(4.0, float(ransac_threshold)))
    if H is None:
        meta["mode"] = "coarse_affine_failed"
        return list(base_corrs), meta
    g = guided_rematch(
        source_gray, reference_gray, H,
        source_mask=source_mask, reference_mask=reference_mask,
        tile=384, nfeatures=900, ratio_threshold=0.80, max_tiles=20,
    )
    gcorrs = list(g.get("correspondences") or [])
    meta["tile_matches"] = len(gcorrs)
    meta["mode"] = g.get("mode", "guided")
    if len(gcorrs) < 6:
        return list(base_corrs), meta

    pool = {}
    for s, r, c in base_corrs:
        key = (round(float(s[0]), 1), round(float(s[1]), 1), round(float(r[0]), 1), round(float(r[1]), 1))
        pool[key] = (s, r, float(c))
    for s, r, c in gcorrs:
        key = (round(float(s[0]), 1), round(float(s[1]), 1), round(float(r[0]), 1), round(float(r[1]), 1))
        score = float(c) + 0.15
        if key not in pool or score > pool[key][2]:
            pool[key] = (s, r, score)
    merged = [(v[0], v[1], v[2]) for v in sorted(pool.values(), key=lambda z: -z[2])]
    meta["guided"] = True
    meta["merged"] = len(merged)
    return merged[:120], meta
