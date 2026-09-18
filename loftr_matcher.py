"""
LoFTR (Local Feature TRansformer) inference module for LunaMatch.
Team Akatsuki Project:
"Multi-modal, Sun angle and scale invariant image
correspondence using Chandrayaan-2 optical images"

Provides genuine LoFTR inference using kornia.feature.LoFTR.
- Model cached in memory across runs
- Automatic device detection (CUDA if available, else CPU)
- Memory-safe input resizing to multiples of 8 with scale preservation
- Output formatted as canonical correspondence tuples:
  ((sx, sy), (rx, ry), confidence)
"""
import logging
import time
from typing import Dict, Optional, Tuple, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_LOFTR_MODEL = None
_LOFTR_DEVICE = None
_LOFTR_PRETRAINED = "outdoor"


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
    """
    Load and cache pretrained LoFTR model.
    Runs once and reuses the cached model in memory.
    """
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
            "Loading LoFTR model "
            f"(pretrained='{pretrained}') on {device_str}..."
        )
        model = LoFTR(pretrained=pretrained)
        model = model.to(device).eval()
        _LOFTR_MODEL = model
        _LOFTR_DEVICE = device_str
        _LOFTR_PRETRAINED = pretrained

    return _LOFTR_MODEL, _LOFTR_DEVICE


def prepare_image_for_loftr(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_side: int = 1024,
) -> Tuple[Any, float, float, Tuple[int, int], Tuple[int, int]]:
    """
    Prepare a 2D grayscale image for LoFTR:
    1. Ensure uint8 / float in [0, 1] range.
    2. Resize to fit within max_side while ensuring dims are multiples of 8.
    3. Return (tensor [1,1,H,W], scale_x, scale_y, original_shape, work_shape).
    """
    import torch

    orig_h, orig_w = img.shape[:2]

    # Calculate scale factor to stay within max_side
    scale = min(1.0, float(max_side) / max(orig_h, orig_w, 1))
    target_w = max(8, int(round(orig_w * scale)))
    target_h = max(8, int(round(orig_h * scale)))

    # LoFTR requires dimensions to be divisible by 8
    target_w = (target_w // 8) * 8
    target_h = (target_h // 8) * 8

    if target_w < 8:
        target_w = 8
    if target_h < 8:
        target_h = 8

    if (target_w, target_h) != (orig_w, orig_h):
        resized = cv2.resize(
            img, (target_w, target_h), interpolation=cv2.INTER_AREA
        )
    else:
        resized = img.copy()

    # Normalize to [0.0, 1.0] float32
    if resized.dtype == np.uint8:
        norm = resized.astype(np.float32) / 255.0
    else:
        r_min = float(np.nanmin(resized))
        r_max = float(np.nanmax(resized))
        if r_max > r_min:
            norm = ((resized - r_min) / (r_max - r_min)).astype(np.float32)
        else:
            norm = np.zeros_like(resized, dtype=np.float32)

    # Calculate coordinate scales back to the input working image
    scale_x = orig_w / float(target_w)
    scale_y = orig_h / float(target_h)

    tensor = torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0)
    return tensor, scale_x, scale_y, (orig_h, orig_w), (target_h, target_w)


def loftr_match(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
    reference_mask: Optional[np.ndarray] = None,
    min_confidence: float = 0.35,
    max_side: int = 1024,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute genuine LoFTR inference on source and reference images.

    Returns dict containing:
    - correspondences: List of ((src_x, src_y), (ref_x, ref_y), confidence)
    - raw_matches: Total raw correspondences output by LoFTR
    - filtered_matches: Correspondences after confidence filtering
    - min_confidence: Threshold used
    - device: 'cuda' or 'cpu'
    - execution_time_seconds: Runtime in seconds
    - source_shape: (H, W)
    - reference_shape: (H, W)
    - working_source_shape: (H, W) passed to LoFTR
    - working_reference_shape: (H, W) passed to LoFTR
    """
    avail, msg = is_loftr_available()
    if not avail:
        raise RuntimeError(f"LoFTR inference unavailable: {msg}")

    import torch

    start_time = time.perf_counter()
    model, device = load_loftr_model(device_str=device_str)
    torch_device = torch.device(device)

    t0, sx0, sy0, orig_shape0, work_shape0 = prepare_image_for_loftr(
        source_gray, source_mask, max_side=max_side
    )
    t1, sx1, sy1, orig_shape1, work_shape1 = prepare_image_for_loftr(
        reference_gray, reference_mask, max_side=max_side
    )

    t0 = t0.to(torch_device)
    t1 = t1.to(torch_device)

    input_dict = {"image0": t0, "image1": t1}

    try:
        with torch.no_grad():
            output = model(input_dict)
            kpts0 = output["keypoints0"].detach().cpu().numpy()
            kpts1 = output["keypoints1"].detach().cpu().numpy()
            confidence = output["confidence"].detach().cpu().numpy()
    except Exception as exc:
        # Release GPU tensors if CUDA was used
        if device == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError(f"LoFTR inference execution failed: {exc}")

    # Clean up GPU memory if needed
    if device == "cuda":
        del t0, t1, input_dict
        torch.cuda.empty_cache()

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

    # Map coordinates back to the input working image coordinate frame
    pts0 = kpts0.copy()
    pts1 = kpts1.copy()
    pts0[:, 0] *= sx0
    pts0[:, 1] *= sy0
    pts1[:, 0] *= sx1
    pts1[:, 1] *= sy1

    # Apply mask filtering if masks provided
    valid_mask = np.ones(raw_count, dtype=bool)
    if source_mask is not None:
        h0, w0 = source_mask.shape[:2]
        ix0 = np.clip(np.rint(pts0[:, 0]).astype(int), 0, w0 - 1)
        iy0 = np.clip(np.rint(pts0[:, 1]).astype(int), 0, h0 - 1)
        valid_mask &= (source_mask[iy0, ix0] > 0)

    if reference_mask is not None:
        h1, w1 = reference_mask.shape[:2]
        ix1 = np.clip(np.rint(pts1[:, 0]).astype(int), 0, w1 - 1)
        iy1 = np.clip(np.rint(pts1[:, 1]).astype(int), 0, h1 - 1)
        valid_mask &= (reference_mask[iy1, ix1] > 0)

    # Apply confidence thresholding
    conf_mask = (confidence >= min_confidence) & valid_mask

    selected_indices = np.where(conf_mask)[0]
    # Sort by confidence descending
    selected_indices = selected_indices[
        np.argsort(-confidence[selected_indices])
    ]

    correspondences = []
    for idx in selected_indices:
        s_pt = (float(pts0[idx, 0]), float(pts0[idx, 1]))
        r_pt = (float(pts1[idx, 0]), float(pts1[idx, 1]))
        conf = float(confidence[idx])
        correspondences.append((s_pt, r_pt, conf))

    exec_time = time.perf_counter() - start_time

    return {
        "correspondences": correspondences,
        "raw_matches": raw_count,
        "filtered_matches": len(correspondences),
        "min_confidence": float(min_confidence),
        "device": device,
        "execution_time_seconds": exec_time,
        "source_shape": orig_shape0,
        "reference_shape": orig_shape1,
        "working_source_shape": work_shape0,
        "working_reference_shape": work_shape1,
        "source_scale": (sx0, sy0),
        "reference_scale": (sx1, sy1),
    }
