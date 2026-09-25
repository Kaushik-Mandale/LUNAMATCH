"""ROI-gated LoFTR loader — assembles scale-aware module from base64 parts."""
from pathlib import Path
import base64
_parts = sorted(Path(__file__).parent.glob("_roi_b64_*.txt"))
if not _parts:
    raise ImportError("missing _roi_b64_*.txt")
_src = base64.b64decode("".join(p.read_text() for p in _parts).encode("ascii")).decode("utf-8")
_ns = {}
exec(compile(_src, "roi_loftr_impl.py", "exec"), _ns)
coarse_sift_correspondences = _ns["coarse_sift_correspondences"]
estimate_overlap_rois = _ns["estimate_overlap_rois"]
estimate_relative_scale = _ns["estimate_relative_scale"]
loftr_match_roi_gated = _ns["loftr_match_roi_gated"]
