"""
ROI-gated LoFTR for LunaMatch — memory-efficient cross-sensor deep matching.

Flow:
  1) Coarse SIFT on downscaled full frames (ROI proposal)
  2) Overlap ROIs capped to roi_max_side (default 512)
  3) LoFTR only on ROI crops
  4) Map matches back to full-image coordinates
  5) Unload model + gc

Use this on Streamlit Cloud instead of full-frame LoFTR.
"""
from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from loftr_matcher import loftr_match, unload_loftr_model


def _downscale_gray(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]
    max_side = max(64, int(max_side))
    scale = min(1.0, float(max_side) / max(h, w, 1))
    if scale >= 0.999:
        return img, 1.0, 1.0
    nw = max(8, int(round(w * scale)))
    nh = max(8, int(round(h * scale)))
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, w / float(nw), h / float(nh)


def coarse_sift_correspondences(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
    nfeatures: int = 2000,
    ratio_threshold: float = 0.8,
    max_side: int = 512,
) -> list:
    """Fast downscaled SIFT for ROI proposal. Points in full-res coordinates."""
    a, sx0, sy0 = _downscale_gray(source_gray, max_side)
    b, sx1, sy1 = _downscale_gray(reference_gray, max_side)
    am = bm = None
    if source_mask is not None:
        am = cv2.resize(
            source_mask.astype(np.uint8),
            (a.shape[1], a.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    if reference_mask is not None:
        bm = cv2.resize(
            reference_mask.astype(np.uint8),
            (b.shape[1], b.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    sift = cv2.SIFT_create(nfeatures=int(nfeatures), contrastThreshold=0.02, edgeThreshold=12)
    kpa, da = sift.detectAndCompute(a, am)
    kpb, db = sift.detectAndCompute(b, bm)
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(da, db, k=2)
    out = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if n.distance < 1e-9:
            continue
        ratio = m.distance / n.distance
        if ratio <= ratio_threshold:
            p0 = kpa[m.queryIdx].pt
            p1 = kpb[m.trainIdx].pt
            out.append(
                (
                    (float(p0[0] * sx0), float(p0[1] * sy0)),
                    (float(p1[0] * sx1), float(p1[1] * sy1)),
                    float(1.0 - ratio),
                )
            )
    out.sort(key=lambda x: -x[2])
    return out[:500]


def _clamp_box(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    return x0, y0, x1, y1


def estimate_overlap_rois(
    correspondences: list,
    source_shape: Tuple[int, ...],
    reference_shape: Tuple[int, ...],
    pad_frac: float = 0.25,
    min_side: int = 256,
    max_side: int = 768,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Axis-aligned ROIs (x0,y0,x1,y1) half-open from coarse matches."""
    hs, ws = int(source_shape[0]), int(source_shape[1])
    hr, wr = int(reference_shape[0]), int(reference_shape[1])
    min_side = max(64, int(min_side))
    max_side = max(min_side, int(max_side))

    def _from_pts(pts, w: int, h: int):
        if pts is None or len(pts) < 2:
            side = min(max_side, w, h)
            cx, cy = w // 2, h // 2
            half = side // 2
            return _clamp_box(cx - half, cy - half, cx - half + side, cy - half + side, w, h)
        xs, ys = pts[:, 0], pts[:, 1]
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        x0 -= bw * pad_frac
        x1 += bw * pad_frac
        y0 -= bh * pad_frac
        y1 += bh * pad_frac
        if (x1 - x0) < min_side:
            mid = 0.5 * (x0 + x1)
            x0, x1 = mid - min_side / 2, mid + min_side / 2
        if (y1 - y0) < min_side:
            mid = 0.5 * (y0 + y1)
            y0, y1 = mid - min_side / 2, mid + min_side / 2
        if (x1 - x0) > max_side:
            mid = 0.5 * (x0 + x1)
            x0, x1 = mid - max_side / 2, mid + max_side / 2
        if (y1 - y0) > max_side:
            mid = 0.5 * (y0 + y1)
            y0, y1 = mid - max_side / 2, mid + max_side / 2
        return _clamp_box(int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1)), w, h)

    if correspondences:
        pts0 = np.array([c[0] for c in correspondences], dtype=np.float64)
        pts1 = np.array([c[1] for c in correspondences], dtype=np.float64)
    else:
        pts0 = pts1 = None
    return _from_pts(pts0, ws, hs), _from_pts(pts1, wr, hr)


def loftr_match_roi_gated(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
    min_confidence: float = 0.2,
    roi_max_side: int = 512,
    coarse_max_side: int = 512,
    pad_frac: float = 0.25,
    coarse_correspondences: Optional[list] = None,
    ratio_threshold: float = 0.8,
) -> Dict[str, Any]:
    """Coarse SIFT -> ROI crop -> LoFTR on crops only -> map to full coords -> unload."""
    start_time = time.perf_counter()
    hs, ws = source_gray.shape[:2]
    hr, wr = reference_gray.shape[:2]

    coarse = coarse_correspondences
    if not coarse:
        coarse = coarse_sift_correspondences(
            source_gray,
            reference_gray,
            source_mask=source_mask,
            reference_mask=reference_mask,
            nfeatures=2000,
            ratio_threshold=ratio_threshold,
            max_side=coarse_max_side,
        )

    src_box, ref_box = estimate_overlap_rois(
        coarse or [],
        (hs, ws),
        (hr, wr),
        pad_frac=pad_frac,
        min_side=min(256, roi_max_side),
        max_side=roi_max_side,
    )
    sx0, sy0, sx1, sy1 = src_box
    rx0, ry0, rx1, ry1 = ref_box

    src_roi = np.ascontiguousarray(source_gray[sy0:sy1, sx0:sx1])
    ref_roi = np.ascontiguousarray(reference_gray[ry0:ry1, rx0:rx1])
    sm_roi = (
        np.ascontiguousarray(source_mask[sy0:sy1, sx0:sx1]) if source_mask is not None else None
    )
    rm_roi = (
        np.ascontiguousarray(reference_mask[ry0:ry1, rx0:rx1]) if reference_mask is not None else None
    )

    gc.collect()
    try:
        out = loftr_match(
            source_gray=src_roi,
            reference_gray=ref_roi,
            source_mask=sm_roi,
            reference_mask=rm_roi,
            min_confidence=float(min_confidence),
            max_side=min(int(roi_max_side), 512),
        )
    finally:
        try:
            unload_loftr_model()
        except Exception:
            pass
        del src_roi, ref_roi, sm_roi, rm_roi
        gc.collect()

    mapped: List[tuple] = []
    for (x0, y0), (x1, y1), conf in out.get("correspondences") or []:
        mapped.append(
            (
                (float(x0) + sx0, float(y0) + sy0),
                (float(x1) + rx0, float(y1) + ry0),
                float(conf),
            )
        )

    result = dict(out)
    result["correspondences"] = mapped
    result["filtered_matches"] = len(mapped)
    result["roi_source_box"] = src_box
    result["roi_reference_box"] = ref_box
    result["coarse_match_count"] = len(coarse or [])
    result["mode"] = "roi_gated"
    result["execution_time_seconds"] = time.perf_counter() - start_time
    result["source_shape"] = (hs, ws)
    result["reference_shape"] = (hr, wr)
    return result
