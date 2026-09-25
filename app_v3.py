"""LunaMatch V3 bootstrap — Cloud OOM safe (SIFT cross-sensor, no LoFTR weights)."""
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

_on_cloud = bool(
    os.environ.get("STREAMLIT_SHARING_MODE")
    or os.environ.get("STREAMLIT_SERVER_HEADLESS")
    or os.path.exists("/mount/src")
    or "streamlit.app" in os.environ.get("HOSTNAME", "")
)
if _on_cloud and os.environ.get("LUNAMATCH_ALLOW_LOFTR", "").strip() != "1":
    os.environ["LUNAMATCH_DISABLE_LOFTR"] = "1"

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

# Experimental / Cloud-safe: accept >=4 MAGSAC inliers (homography minimum)
_src = _src.replace(
    'if int(train_mask.sum()) < 8:\n            raise ValueError(f"Too few robust inliers ({int(train_mask.sum())}) after {geom_info.get(\'verifier_method\', geometric_verifier)} estimation.")',
    'if int(train_mask.sum()) < (4 if execution_mode == "Experimental image-only mode" else 8):\n            raise ValueError(f"Too few robust inliers ({int(train_mask.sum())}) after {geom_info.get(\'verifier_method\', geometric_verifier)} estimation.")',
    1,
)

_NEW_CROSS = '''
            # CROSS-SENSOR: SIFT when LoFTR disabled/unavailable (Cloud-safe)
            if (not loftr_avail) or os.environ.get("LUNAMATCH_DISABLE_LOFTR") == "1":
                runs = [
                    ("SIFT-intensity",  sift_run(src.gray,      ref.gray,      src.mask, ref.mask, feature_count, ratio_threshold)),
                    ("SIFT-structure",  sift_run(src.structure,  ref.structure,  src.mask, ref.mask, feature_count, ratio_threshold)),
                    ("SIFT-gradient",   sift_run(src.gradient,   ref.gradient,   src.mask, ref.mask, feature_count, ratio_threshold)),
                ]
                actual_matcher = "SIFT"
                matcher_note = "Cross-sensor Cloud-safe path: SIFT multi-rep (LoFTR disabled to prevent OOM)"
                loftr_metrics = {}
            else:
                loftr_out = loftr_matcher.loftr_match(
                    source_gray=src.gray,
                    reference_gray=ref.gray,
                    source_mask=src.mask,
                    reference_mask=ref.mask,
                    min_confidence=float(loftr_confidence_threshold),
                    max_side=min(max_side, 512),
                )
                loftr_metrics = {
                    "raw_matches": loftr_out["raw_matches"],
                    "filtered_matches": loftr_out["filtered_matches"],
                    "min_confidence": loftr_out["min_confidence"],
                    "device": loftr_out["device"],
                    "working_source_shape": loftr_out["working_source_shape"],
                    "working_reference_shape": loftr_out["working_reference_shape"],
                    "source_scale": loftr_out["source_scale"],
                    "reference_scale": loftr_out["reference_scale"],
                }
                if loftr_out["filtered_matches"] < 4:
                    raise ValueError(
                        f"LoFTR produced insufficient confident matches ({loftr_out['filtered_matches']}) "
                        f"at confidence threshold {loftr_confidence_threshold}."
                    )
                runs = [("LoFTR-dense", loftr_out["correspondences"])]
                actual_matcher = "LoFTR"
                matcher_note = (
                    f"LoFTR deep matching ({loftr_out['device'].upper()}): "
                    f"{loftr_out['raw_matches']} raw -> {loftr_out['filtered_matches']} filtered"
                )
                try:
                    loftr_matcher.unload_loftr_model()
                except Exception:
                    pass
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
import loftr_matcher as _lm
_orig_avail = _lm.is_loftr_available
def _cloud_safe_loftr_available():
    if _os_helper.environ.get("LUNAMATCH_DISABLE_LOFTR") == "1":
        return False, "LoFTR disabled on Cloud to prevent OOM"
    return _orig_avail()
_lm.is_loftr_available = _cloud_safe_loftr_available
'''
if "_cloud_safe_loftr_available" not in _src:
    _src = _src.replace("import streamlit as st\n", "import streamlit as st\n" + _INJECT, 1)

if os.environ.get("LUNAMATCH_DISABLE_LOFTR") == "1":
    _src = _src.replace(
        'key="reference"\n    )',
        'key="reference"\n    )\n    st.info("**Cloud-safe mode:** LoFTR disabled. OHRC-LROC uses SIFT multi-rep. max_side=768. Experimental accepts >=4 MAGSAC inliers.")',
        1,
    )

exec(compile(_src, "app_v3.py", "exec"), globals())
