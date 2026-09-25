"""
LoFTR (Local Feature TRansformer) inference module for LunaMatch.
Team Akatsuki Project:
"Multi-modal, Sun angle and scale invariant image
correspondence using Chandrayaan-2 optical images"

Memory-oriented for Streamlit Cloud:
- Model cached once; tensors deleted after inference
- Default max_side 768 (override via caller)
- CPU thread caps to reduce peak RSS
- Explicit gc after heavy work
"""
from __future__ import annotations

import gc
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Cap BLAS / torch threads early (safe no-ops if already set by host)
for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("TORCH_NUM_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)

_LOFTR_MODEL = None
_LOFTR_DEVICE = None
_LOFTR_PRETRAINED = "outdoor"

# Cloud-safe default; callers may raise for local high-RAM runs
DEFAULT_LOFTR_MAX_SIDE = 768


def is_loftr_available() -> Tuple[bool, str]:
    """Check if torch and kornia with LoFTR are available."""
    try:
        import torch  # noqa: F401
        import kornia  # noqa: F401
        from kornia.feature import LoFTR  # noqa: F401
        return True, "LoFTR dependencies (torch, kornia) are available."
    except ImportError as exc:
        return False, f"LoFTR dependency missing: {exc}"


def get_loftr_device() -> str:
    """Return 'cuda' if CUDA is available and functional, otherwise 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_loftr_model(
    device_str: Optional[str] = None, pretrained: str = "outdoor"
):
    """Load and cache pretrained LoFTR model (once per process)."""
    global _LOFTR_MODEL, _LOFTR_DEVICE, _LOFTR_PRETRAINED
    import torch
    from kornia.feature import LoFTR

    if device_str is None:
        device_str = get_loftr_device()

    device = torch.device(device_str)

    if (
        _LOFTR_MODEL is None
        or _LOFTR_DEVICE != device_str
        or _LOFTR_PRETRAINED != pretrained
    ):
        logger.info(
            "Loading LoFTR model (pretrained=%r) on %s...", pretrained, device_str
        )
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        model = LoFTR(pretrained=pretrained)
        model = model.to(device).eval()
        _LOFTR_MODEL = model
        _LOFTR_DEVICE = device_str
        _LOFTR_PRETRAINED = pretrained

    return _LOFTR_MODEL, _LOFTR_DEVICE


def unload_loftr_model() -> None:
    """Drop cached model to free RAM (call after pipeline on low-memory hosts)."""
    global _LOFTR_MODEL, _LOFTR_DEVICE, _LOFTR_PRETRAINED
    _LOFTR_MODEL = None
    _LOFTR_DEVICE = None
    _LOFTR_PRETRAINED = "outdoor"
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def prepare_image_for_loftr(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_side: int = DEFAULT_LOFTR_MAX_SIDE,
) -> Tuple[Any, float, float, Tuple[int, int], Tuple[int, int]]:
    """
    Prepare a 2D grayscale image for LoFTR:
    resize within max_side, dims multiples of 8, float32 [0,1] tensor.
    """
    import torch

    orig_h, orig_w = img.shape[:2]
    max_side = max(64, int(max_side))

    scale = min(1.0, float(max_side) / max(orig_h, orig_w, 1))
    target_w = max(8, int(round(orig_w * scale)))
    target_h = max(8, int(round(orig_h * scale)))
    target_w = max(8, (target_w // 8) * 8)
    target_h = max(8, (target_h // 8) * 8)

    if (target_w, target_h) != (orig_w, orig_h):
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        resized = img

    if resized.dtype == np.uint8:
        norm = resized.astype(np.float32) / 255.0
    else:
        r_min = float(np.nanmin(resized))
        r_max = float(np.nanmax(resized))
        if r_max > r_min:
            norm = ((resized - r_min) / (r_max - r_min)).astype(np.float32)
        else:
            norm = np.zeros((target_h, target_w), dtype=np.float32)

    if resized is not img and resized is not norm:
        del resized

    scale_x = orig_w / float(target_w)
    scale_y = orig_h / float(target_h)

    tensor = torch.from_numpy(np.ascontiguousarray(norm)).float().unsqueeze(0).unsqueeze(0)
    del norm
    return tensor, scale_x, scale_y, (orig_h, orig_w), (target_h, target_w)


def loftr_match(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
    min_confidence: float = 0.35,
    max_side: int = DEFAULT_LOFTR_MAX_SIDE,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute LoFTR inference with explicit tensor cleanup for low-RAM hosts."""
    import torch

    start_time = time.perf_counter()
    max_side = max(64, int(max_side))

    model, device = load_loftr_model(device_str=device_str)

    t0, sx0, sy0, orig_shape0, work_shape0 = prepare_image_for_loftr(
        source_gray, source_mask, max_side=max_side
    )
    t1, sx1, sy1, orig_shape1, work_shape1 = prepare_image_for_loftr(
        reference_gray, reference_mask, max_side=max_side
    )

    t0 = t0.to(device)
    t1 = t1.to(device)
    input_dict = {"image0": t0, "image1": t1}

    try:
        with torch.no_grad():
            output = model(input_dict)
            kpts0 = output["keypoints0"].detach().cpu().numpy()
            kpts1 = output["keypoints1"].detach().cpu().numpy()
            confidence = output["confidence"].detach().cpu().numpy()
        del output
    except Exception as exc:
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        raise RuntimeError(f"LoFTR inference execution failed: {exc}") from exc
    finally:
        del t0, t1, input_dict
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    raw_count = len(confidence)
    if raw_count == 0:
        return {
            "correspondences": [],
            "raw_matches": 0,
            "filtered_matches": 0,
            "min_confidence": float(min_confidence),
            "device": device,
            "execution_time_seconds": time.perf_counter() - start_time,
            "source_shape": orig_shape0,
            "reference_shape": orig_shape1,
            "working_source_shape": work_shape0,
            "working_reference_shape": work_shape1,
            "source_scale": (sx0, sy0),
            "reference_scale": (sx1, sy1),
        }

    pts0 = kpts0.copy()
    pts1 = kpts1.copy()
    pts0[:, 0] *= sx0
    pts0[:, 1] *= sy0
    pts1[:, 0] *= sx1
    pts1[:, 1] *= sy1

    valid_mask = np.ones(raw_count, dtype=bool)
    if source_mask is not None:
        h0, w0 = source_mask.shape[:2]
        ix0 = np.clip(np.rint(pts0[:, 0]).astype(int), 0, w0 - 1)
        iy0 = np.clip(np.rint(pts0[:, 1]).astype(int), 0, h0 - 1)
        valid_mask &= source_mask[iy0, ix0] > 0

    if reference_mask is not None:
        h1, w1 = reference_mask.shape[:2]
        ix1 = np.clip(np.rint(pts1[:, 0]).astype(int), 0, w1 - 1)
        iy1 = np.clip(np.rint(pts1[:, 1]).astype(int), 0, h1 - 1)
        valid_mask &= reference_mask[iy1, ix1] > 0

    conf_mask = (confidence >= min_confidence) & valid_mask
    selected_indices = np.where(conf_mask)[0]
    selected_indices = selected_indices[np.argsort(-confidence[selected_indices])]

    correspondences = [
        (
            (float(pts0[idx, 0]), float(pts0[idx, 1])),
            (float(pts1[idx, 0]), float(pts1[idx, 1])),
            float(confidence[idx]),
        )
        for idx in selected_indices
    ]

    del kpts0, kpts1, confidence, pts0, pts1, valid_mask, conf_mask
    gc.collect()

    return {
        "correspondences": correspondences,
        "raw_matches": raw_count,
        "filtered_matches": len(correspondences),
        "min_confidence": float(min_confidence),
        "device": device,
        "execution_time_seconds": time.perf_counter() - start_time,
        "source_shape": orig_shape0,
        "reference_shape": orig_shape1,
        "working_source_shape": work_shape0,
        "working_reference_shape": work_shape1,
        "source_scale": (sx0, sy0),
        "reference_scale": (sx1, sy1),
    }
