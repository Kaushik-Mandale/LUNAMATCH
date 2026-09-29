"""Scale-aware full-image LoFTR for cross-sensor (OHRC ↔ LROC).

1) Percentile stretch + CLAHE (fixes near-black OHRC previews)
2) Coarse SIFT → relative scale
3) Resize source to reference scale
4) Full LoFTR on equalized, scale-matched pair
5) Map coords back to original source frame
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _to_u8(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray)
    if g.ndim > 2:
        g = g[..., 0]
    if g.dtype == np.uint8:
        return g
    g = g.astype(np.float32)
    lo, hi = np.percentile(g[np.isfinite(g)], [1.0, 99.0]) if np.isfinite(g).any() else (0.0, 1.0)
    if hi <= lo:
        lo, hi = float(np.nanmin(g)), float(np.nanmax(g))
    if hi <= lo:
        return np.zeros(g.shape, np.uint8)
    x = np.clip((g - lo) / (hi - lo), 0, 1)
    return (x * 255.0).astype(np.uint8)


def enhance_pair(src: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Strong contrast equalization for cross-sensor lunar imagery."""
    a = _to_u8(src)
    b = _to_u8(ref)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(a), clahe.apply(b)


def _downscale(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]
    max_side = max(64, int(max_side))
    s = min(1.0, float(max_side) / max(h, w, 1))
    if s >= 0.999:
        return img, 1.0, 1.0
    nw, nh = max(8, int(round(w * s))), max(8, int(round(h * s)))
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, s, s


def estimate_rel_scale(
    src: np.ndarray,
    ref: np.ndarray,
    nfeatures: int = 3000,
    ratio: float = 0.85,
    max_side: int = 800,
) -> float:
    """Median pairwise distance ratio from coarse SIFT (src/ref scale)."""
    a, sa, _ = _downscale(src, max_side)
    b, sb, _ = _downscale(ref, max_side)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    k0, d0 = sift.detectAndCompute(a, None)
    k1, d1 = sift.detectAndCompute(b, None)
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return 1.0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = bf.knnMatch(d0, d1, k=2)
    good = []
    for m in pairs:
        if len(m) < 2:
            continue
        if m[0].distance < ratio * m[1].distance:
            good.append((k0[m[0].queryIdx].pt, k1[m[0].trainIdx].pt))
    if len(good) < 8:
        return 1.0
    # scale from random pair distances
    rng = np.random.default_rng(0)
    ratios = []
    pts0 = np.array([g[0] for g in good], np.float64)
    pts1 = np.array([g[1] for g in good], np.float64)
    n = len(good)
    for _ in range(min(200, n * 3)):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        d0 = np.linalg.norm(pts0[i] - pts0[j])
        d1 = np.linalg.norm(pts1[i] - pts1[j])
        if d0 > 5 and d1 > 5:
            ratios.append(d0 / d1)
    if not ratios:
        return 1.0
    # undo downscale: pts already in downscaled coords; ratio is scale_src/scale_ref in that space
    # true src/ref scale ≈ median(ratios) * (sb/sa) wait: both downscaled independently by sa,sb
    # distance in orig_src = d0/sa, orig_ref = d1/sb → ratio_orig = (d0/sa)/(d1/sb) = (d0/d1)*(sb/sa)
    med = float(np.median(ratios))
    return float(np.clip(med * (sb / max(sa, 1e-9)), 0.15, 8.0))


def match_cross_sensor_full(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
    min_confidence: float = 0.25,
    max_side: int = 1280,
    geom_conf_floor: float = 0.32,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    src_e, ref_e = enhance_pair(source_gray, reference_gray)

    rel = estimate_rel_scale(src_e, ref_e)
    # Resize source so features are similar size to reference
    # If src features are larger (rel>1), shrink source
    h0, w0 = src_e.shape[:2]
    scale_apply = 1.0 / max(rel, 1e-6)
    # clamp extreme
    scale_apply = float(np.clip(scale_apply, 0.2, 5.0))
    nw = max(32, int(round(w0 * scale_apply)))
    nh = max(32, int(round(h0 * scale_apply)))
    src_s = cv2.resize(src_e, (nw, nh), interpolation=cv2.INTER_AREA if scale_apply < 1 else cv2.INTER_LINEAR)

    # masks scaled if present
    sm = source_mask
    if sm is not None:
        sm = cv2.resize(_to_u8(sm), (nw, nh), interpolation=cv2.INTER_NEAREST)

    import loftr_matcher as _lm

    out = _lm.loftr_match(
        source_gray=src_s,
        reference_gray=ref_e,
        source_mask=sm,
        reference_mask=reference_mask,
        min_confidence=float(min_confidence),
        max_side=int(max_side),
    )
    corrs: List[Tuple] = list(out.get("correspondences") or [])
    # Map source coords from scaled frame → original source pixels
    inv = 1.0 / max(scale_apply, 1e-9)
    mapped = []
    for s, r, c in corrs:
        s2 = (float(s[0]) * inv, float(s[1]) * inv)
        mapped.append((s2, r, float(c)))

    mapped = sorted(mapped, key=lambda t: -t[2])
    floor = max(float(geom_conf_floor), float(min_confidence))
    strict = [m for m in mapped if m[2] >= floor]
    if len(strict) >= 12:
        final = strict[:72]
    else:
        final = mapped[:min(48, len(mapped))]

    try:
        _lm.unload_loftr_model()
    except Exception:
        pass

    return {
        "correspondences": final,
        "raw_matches": out.get("raw_matches", len(mapped)),
        "filtered_matches": len(final),
        "min_confidence": floor,
        "device": out.get("device"),
        "rel_scale": rel,
        "scale_apply": scale_apply,
        "mode": "full_scale_clahe",
        "max_side": max_side,
        "execution_time_seconds": time.perf_counter() - t0,
    }
