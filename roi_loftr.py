"""ROI-gated scale-aware LoFTR for LunaMatch (Cloud memory-safe)."""
from __future__ import annotations
import gc, time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from loftr_matcher import loftr_match, unload_loftr_model

def _downscale_gray(img, max_side):
    h, w = img.shape[:2]
    max_side = max(64, int(max_side))
    scale = min(1.0, float(max_side) / max(h, w, 1))
    if scale >= 0.999:
        return img, 1.0, 1.0
    nw, nh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), w / float(nw), h / float(nh)

def coarse_sift_correspondences(source_gray, reference_gray, source_mask=None, reference_mask=None,
                                nfeatures=2500, ratio_threshold=0.82, max_side=640):
    a, sx0, sy0 = _downscale_gray(source_gray, max_side)
    b, sx1, sy1 = _downscale_gray(reference_gray, max_side)
    am = cv2.resize(source_mask.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) if source_mask is not None else None
    bm = cv2.resize(reference_mask.astype(np.uint8), (b.shape[1], b.shape[0]), interpolation=cv2.INTER_NEAREST) if reference_mask is not None else None
    sift = cv2.SIFT_create(nfeatures=int(nfeatures), contrastThreshold=0.018, edgeThreshold=12)
    kpa, da = sift.detectAndCompute(a, am)
    kpb, db = sift.detectAndCompute(b, bm)
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return []
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    out = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if n.distance < 1e-9:
            continue
        ratio = m.distance / n.distance
        if ratio <= ratio_threshold:
            p0, p1 = kpa[m.queryIdx].pt, kpb[m.trainIdx].pt
            out.append(((float(p0[0]*sx0), float(p0[1]*sy0)), (float(p1[0]*sx1), float(p1[1]*sy1)), float(1.0-ratio)))
    out.sort(key=lambda x: -x[2])
    return out[:800]

def estimate_relative_scale(correspondences, min_samples=12):
    if not correspondences or len(correspondences) < 4:
        return 1.0
    pts0 = np.asarray([c[0] for c in correspondences], np.float64)
    pts1 = np.asarray([c[1] for c in correspondences], np.float64)
    n, rng, ratios = len(pts0), np.random.default_rng(42), []
    for _ in range(min(400, max(n*3, min_samples*4))):
        i, j = rng.choice(n, size=2, replace=False)
        d0, d1 = float(np.linalg.norm(pts0[i]-pts0[j])), float(np.linalg.norm(pts1[i]-pts1[j]))
        if d0 > 8.0 and d1 > 8.0:
            ratios.append(d1/d0)
    if len(ratios) < max(3, min_samples//2):
        return 1.0
    return float(np.clip(float(np.median(ratios)), 0.15, 6.0))

def _clamp_box(x0, y0, x1, y1, w, h):
    x0 = max(0, min(x0, w-1)); y0 = max(0, min(y0, h-1))
    x1 = max(x0+1, min(x1, w)); y1 = max(y0+1, min(y1, h))
    return x0, y0, x1, y1

def estimate_overlap_rois(correspondences, source_shape, reference_shape, pad_frac=0.30, min_side=320, max_side=768, expand=1.0):
    hs, ws = int(source_shape[0]), int(source_shape[1])
    hr, wr = int(reference_shape[0]), int(reference_shape[1])
    min_side, max_side = max(64, int(min_side)), max(min_side, int(max_side))
    pad_frac = float(pad_frac) * float(expand)
    def _from_pts(pts, w, h):
        if pts is None or len(pts) < 2:
            side = min(max_side, w, h); cx, cy, half = w//2, h//2, side//2
            return _clamp_box(cx-half, cy-half, cx-half+side, cy-half+side, w, h)
        xs, ys = pts[:,0], pts[:,1]
        x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
        bw, bh = max(x1-x0, 1.0), max(y1-y0, 1.0)
        x0 -= bw*pad_frac; x1 += bw*pad_frac; y0 -= bh*pad_frac; y1 += bh*pad_frac
        if (x1-x0) < min_side:
            mid = 0.5*(x0+x1); x0, x1 = mid-min_side/2, mid+min_side/2
        if (y1-y0) < min_side:
            mid = 0.5*(y0+y1); y0, y1 = mid-min_side/2, mid+min_side/2
        if (x1-x0) > max_side:
            mid = 0.5*(x0+x1); x0, x1 = mid-max_side/2, mid+max_side/2
        if (y1-y0) > max_side:
            mid = 0.5*(y0+y1); y0, y1 = mid-max_side/2, mid+max_side/2
        return _clamp_box(int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1)), w, h)
    if correspondences:
        pts0 = np.array([c[0] for c in correspondences], np.float64)
        pts1 = np.array([c[1] for c in correspondences], np.float64)
    else:
        pts0 = pts1 = None
    return _from_pts(pts0, ws, hs), _from_pts(pts1, wr, hr)

def _run_loftr_on_rois(source_gray, reference_gray, source_mask, reference_mask, src_box, ref_box, rel_scale, min_confidence, roi_max_side):
    sx0, sy0, sx1, sy1 = src_box
    rx0, ry0, rx1, ry1 = ref_box
    src_roi = np.ascontiguousarray(source_gray[sy0:sy1, sx0:sx1])
    ref_roi = np.ascontiguousarray(reference_gray[ry0:ry1, rx0:rx1])
    sm_roi = np.ascontiguousarray(source_mask[sy0:sy1, sx0:sx1]) if source_mask is not None else None
    rm_roi = np.ascontiguousarray(reference_mask[ry0:ry1, rx0:rx1]) if reference_mask is not None else None
    scale_to_src = float(np.clip(1.0 / max(float(rel_scale), 1e-6), 0.2, 5.0))
    rh, rw = ref_roi.shape[:2]
    new_w, new_h = max(32, int(round(rw*scale_to_src))), max(32, int(round(rh*scale_to_src)))
    cap = max(64, int(roi_max_side))
    cs = min(1.0, float(cap)/max(new_w, new_h, 1))
    new_w, new_h = max(32, int(round(new_w*cs))), max(32, int(round(new_h*cs)))
    if (new_w, new_h) != (rw, rh):
        ref_roi_s = cv2.resize(ref_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rm_roi_s = cv2.resize(rm_roi.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) if rm_roi is not None else None
    else:
        ref_roi_s, rm_roi_s = ref_roi, rm_roi
    inv_sx, inv_sy = rw/float(ref_roi_s.shape[1]), rh/float(ref_roi_s.shape[0])
    gc.collect()
    try:
        out = loftr_match(source_gray=src_roi, reference_gray=ref_roi_s, source_mask=sm_roi, reference_mask=rm_roi_s,
                          min_confidence=float(min_confidence), max_side=min(int(roi_max_side), 512))
    finally:
        try: unload_loftr_model()
        except Exception: pass
        del src_roi, ref_roi, ref_roi_s, sm_roi, rm_roi, rm_roi_s
        gc.collect()
    mapped = []
    for (x0, y0), (x1, y1), conf in out.get("correspondences") or []:
        mapped.append(((float(x0)+sx0, float(y0)+sy0), (float(x1)*inv_sx+rx0, float(y1)*inv_sy+ry0), float(conf)))
    meta = {"raw_matches": out.get("raw_matches", 0), "filtered_matches": len(mapped),
            "min_confidence": out.get("min_confidence", min_confidence), "device": out.get("device"),
            "rel_scale": float(rel_scale), "roi_source_box": src_box, "roi_reference_box": ref_box}
    return mapped, meta

def loftr_match_roi_gated(source_gray, reference_gray, source_mask=None, reference_mask=None,
                          min_confidence=0.18, roi_max_side=512, coarse_max_side=640,
                          pad_frac=0.30, coarse_correspondences=None, ratio_threshold=0.82):
    t0 = time.perf_counter()
    hs, ws = source_gray.shape[:2]; hr, wr = reference_gray.shape[:2]
    coarse = list(coarse_correspondences or [])
    if len(coarse) < 8:
        extra = coarse_sift_correspondences(source_gray, reference_gray, source_mask, reference_mask,
                                            nfeatures=2500, ratio_threshold=ratio_threshold, max_side=coarse_max_side)
        seen = {(round(a[0],1), round(a[1],1), round(b[0],1), round(b[1],1)) for a,b,_ in coarse}
        for a,b,s in extra:
            k = (round(a[0],1), round(a[1],1), round(b[0],1), round(b[1],1))
            if k not in seen:
                coarse.append((a,b,s)); seen.add(k)
    rel_scale = estimate_relative_scale(coarse)
    src_box, ref_box = estimate_overlap_rois(coarse, (hs,ws), (hr,wr), pad_frac=pad_frac,
                                             min_side=min(320, roi_max_side), max_side=roi_max_side, expand=1.0)
    mapped, meta = _run_loftr_on_rois(source_gray, reference_gray, source_mask, reference_mask,
                                      src_box, ref_box, rel_scale, min_confidence, roi_max_side)
    if len(mapped) < 12:
        src_box2, ref_box2 = estimate_overlap_rois(coarse, (hs,ws), (hr,wr), pad_frac=pad_frac,
                                                   min_side=min(384, roi_max_side), max_side=min(roi_max_side+128, 768), expand=1.6)
        mapped2, meta2 = _run_loftr_on_rois(source_gray, reference_gray, source_mask, reference_mask,
                                            src_box2, ref_box2, rel_scale, max(0.12, float(min_confidence)-0.05),
                                            min(roi_max_side+128, 640))
        pool = {}
        for a,b,c in list(mapped)+list(mapped2):
            k = (round(a[0],1), round(a[1],1), round(b[0],1), round(b[1],1))
            if k not in pool or c > pool[k][2]:
                pool[k] = (a,b,c)
        mapped = sorted(pool.values(), key=lambda z: -z[2])
        meta["second_pass"] = True
        meta["roi_source_box"], meta["roi_reference_box"] = src_box2, ref_box2
        meta["raw_matches"] = int(meta.get("raw_matches",0)) + int(meta2.get("raw_matches",0))
    else:
        meta["second_pass"] = False
    return {
        "correspondences": mapped, "raw_matches": int(meta.get("raw_matches",0)),
        "filtered_matches": len(mapped), "min_confidence": float(min_confidence),
        "device": meta.get("device"), "execution_time_seconds": time.perf_counter()-t0,
        "source_shape": (hs,ws), "reference_shape": (hr,wr),
        "roi_source_box": meta.get("roi_source_box"), "roi_reference_box": meta.get("roi_reference_box"),
        "coarse_match_count": len(coarse), "rel_scale": float(rel_scale),
        "mode": "roi_gated_scale_aware", "second_pass": bool(meta.get("second_pass")),
    }
