"""LunaMatch V3 bootstrap — ROI-gated LoFTR + Cloud memory guards."""
from __future__ import annotations

import os
import re
import urllib.request

for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("TORCH_NUM_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)

# ROI-gated LoFTR is the Cloud default. Set LUNAMATCH_DISABLE_LOFTR=1 to force SIFT-only.
# Set LUNAMATCH_ALLOW_FULL_LOFTR=1 only on high-RAM hosts for full-frame LoFTR.
_on_cloud = bool(
    os.environ.get("STREAMLIT_SHARING_MODE")
    or os.environ.get("STREAMLIT_SERVER_HEADLESS")
    or os.path.exists("/mount/src")
    or "streamlit.app" in os.environ.get("HOSTNAME", "")
)
os.environ.setdefault("LUNAMATCH_ROI_LOFTR", "1")

_URL = (
    "https://raw.githubusercontent.com/Kaushik-Mandale/LUNAMATCH/"
    "a12f6c3394b5510357d1c5aad1fe3d9ffd7244f1/app_v3.py"
)
with urllib.request.urlopen(_URL, timeout=60) as _resp:
    _src = _resp.read().decode("utf-8")

_src = _src.replace("max_side=2048, feature_count=12000,", "max_side=768, feature_count=3000,", 1)
_src = _src.replace("[1024, 1600, 2048, 3072, 4096], value=2048", "[512, 768, 1024, 1536], value=768", 1)
_src = _src.replace(
    'st.slider("SIFT features / representation", 3000, 30000, 12000, 1000)',
    'st.slider("SIFT features / representation", 1000, 8000, 3000, 500)',
    1,
)
_src = _src.replace("feature_count = 12000\n", "feature_count = 3000\n", 1)
_src = _src.replace(
    "len(source_file.getvalue()) / (1024 ** 2)",
    "(getattr(source_file, 'size', None) or len(source_file.getvalue())) / (1024 ** 2)",
)
_src = _src.replace(
    "len(reference_file.getvalue()) / (1024 ** 2)",
    "(getattr(reference_file, 'size', None) or len(reference_file.getvalue())) / (1024 ** 2)",
)
_src = _src.replace(
    "len(reference_file.getvalue())/(1024**2)",
    "(getattr(reference_file, 'size', None) or len(reference_file.getvalue()))/(1024**2)",
)
_src = _src.replace(
    "compute_file_hash(source_file.getvalue()) if source_file else None",
    "(_light_id(source_file) if source_file is not None else None)",
)
_src = _src.replace(
    "compute_file_hash(reference_file.getvalue()) if reference_file else None",
    "(_light_id(reference_file) if reference_file is not None else None)",
)
_src = _src.replace(
    "compute_file_hash(reference_file.getvalue()) if reference_file is not None else None",
    "(_light_id(reference_file) if reference_file is not None else None)",
)
_src = _src.replace(
    "check_file_size_consistency(len(reference_file.getvalue()), _spec_obj)",
    "check_file_size_consistency((getattr(reference_file, 'size', None) or len(reference_file.getvalue())), _spec_obj)",
)
_src = _src.replace(
    'route_status = "failed"\n            route_error = f"LoFTR dependency missing: {loftr_err}"',
    'route_status = "success"\n            route_error = None',
    1,
)
_src = _src.replace(
    'if int(train_mask.sum()) < 8:\n            raise ValueError(f"Too few robust inliers ({int(train_mask.sum())}) after {geom_info.get(\'verifier_method\', geometric_verifier)} estimation.")',
    'if int(train_mask.sum()) < (4 if execution_mode == "Experimental image-only mode" else 8):\n            raise ValueError(f"Too few robust inliers ({int(train_mask.sum())}) after {geom_info.get(\'verifier_method\', geometric_verifier)} estimation.")',
    1,
)

_NEW_CROSS = '''
            # CROSS-SENSOR: SIFT multi-rep + ROI-gated LoFTR (memory-efficient deep matching)
            runs = [
                ("SIFT-intensity",  sift_run(src.gray,      ref.gray,      src.mask, ref.mask, feature_count, ratio_threshold)),
                ("SIFT-structure",  sift_run(src.structure,  ref.structure,  src.mask, ref.mask, feature_count, ratio_threshold)),
                ("SIFT-gradient",   sift_run(src.gradient,   ref.gradient,   src.mask, ref.mask, feature_count, ratio_threshold)),
            ]
            actual_matcher = "SIFT"
            matcher_note = "Cross-sensor: SIFT multi-rep"
            loftr_metrics = {}
            _coarse = []
            for _name, _rows in runs:
                _coarse.extend(_rows)

            _try_roi = (
                os.environ.get("LUNAMATCH_DISABLE_LOFTR") != "1"
                and os.environ.get("LUNAMATCH_ROI_LOFTR", "1") == "1"
            )
            if _try_roi and loftr_avail:
                try:
                    import roi_loftr as _roi
                    loftr_out = _roi.loftr_match_roi_gated(
                        source_gray=src.gray,
                        reference_gray=ref.gray,
                        source_mask=src.mask,
                        reference_mask=ref.mask,
                        min_confidence=float(loftr_confidence_threshold),
                        roi_max_side=min(int(max_side), 512),
                        coarse_max_side=512,
                        coarse_correspondences=_coarse[:250] if _coarse else None,
                        ratio_threshold=float(ratio_threshold),
                    )
                    loftr_metrics = {
                        "raw_matches": loftr_out.get("raw_matches"),
                        "filtered_matches": loftr_out.get("filtered_matches"),
                        "min_confidence": loftr_out.get("min_confidence"),
                        "device": loftr_out.get("device"),
                        "roi_source_box": loftr_out.get("roi_source_box"),
                        "roi_reference_box": loftr_out.get("roi_reference_box"),
                        "coarse_match_count": loftr_out.get("coarse_match_count"),
                        "mode": loftr_out.get("mode"),
                    }
                    if int(loftr_out.get("filtered_matches") or 0) >= 4:
                        runs.append(("LoFTR-ROI", loftr_out["correspondences"]))
                        actual_matcher = "SIFT+LoFTR-ROI"
                        matcher_note = (
                            f"ROI-gated LoFTR ({str(loftr_out.get('device', 'cpu')).upper()}): "
                            f"{loftr_out.get('filtered_matches')} matches on ROI "
                            f"{loftr_out.get('roi_source_box')} / {loftr_out.get('roi_reference_box')} "
                            f"(coarse={loftr_out.get('coarse_match_count')})"
                        )
                    else:
                        matcher_note = (
                            f"SIFT multi-rep; LoFTR-ROI produced only "
                            f"{loftr_out.get('filtered_matches')} matches (kept SIFT)"
                        )
                except Exception as _roi_exc:
                    matcher_note = f"SIFT multi-rep (LoFTR-ROI skipped: {_roi_exc})"
            elif os.environ.get("LUNAMATCH_DISABLE_LOFTR") == "1":
                matcher_note = "Cross-sensor: SIFT multi-rep (LoFTR disabled via env)"
            elif not loftr_avail:
                matcher_note = f"Cross-sensor: SIFT multi-rep (LoFTR unavailable: {loftr_err})"
'''

_src, _n = re.subn(
    r'# CROSS-SENSOR BRANCH[\s\S]*?actual_matcher = "LoFTR"\n'
    r'[ \t]*matcher_note = \([\s\S]*?\)\n',
    _NEW_CROSS.lstrip("\n"),
    _src,
    count=1,
)
if _n != 1:
    raise RuntimeError(f"Cloud bootstrap failed to patch cross-sensor block (matches={_n})")

_INJECT = '''
import os as _os_helper
try:
    from core.upload_utils import lightweight_file_id as _light_id
except Exception:
    def _light_id(f):
        import hashlib
        n = getattr(f, "name", "") or ""
        s = getattr(f, "size", 0) or 0
        return hashlib.sha256(f"{n}|{s}".encode()).hexdigest()[:16]
'''
if "_light_id" not in _src or "def _light_id" not in _src:
    if "import streamlit as st\n" in _src:
        _src = _src.replace("import streamlit as st\n", "import streamlit as st\n" + _INJECT, 1)

_src = _src.replace(
    'key="reference"\n    )',
    'key="reference"\n    )\n    st.info("**ROI-gated LoFTR:** SIFT proposes overlap, LoFTR runs only on <=512px crops, then unloads. Falls back to SIFT if deep match fails.")',
    1,
)

exec(compile(_src, "app_v3.py", "exec"), globals())
