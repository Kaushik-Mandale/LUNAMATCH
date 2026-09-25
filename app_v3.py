"""LunaMatch V3 bootstrap — Cloud OOM mitigations (LoFTR off by default)."""
from __future__ import annotations

import base64
import os
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

_GOOD_COMMIT = "a12f6c3394b5510357d1c5aad1fe3d9ffd7244f1"
_URL = (
    f"https://raw.githubusercontent.com/Kaushik-Mandale/LUNAMATCH/"
    f"{_GOOD_COMMIT}/app_v3.py"
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

def _b64(s: str) -> str:
    return base64.b64decode(s.encode("ascii")).decode("utf-8")

# Critical: force is_loftr unavailable via env (already set). Soft-patch route + cross-sensor
# via simple non-unicode-dependent replacements that always match.
_src = _src.replace(
    'route_status = "failed"\n            route_error = f"LoFTR dependency missing: {loftr_err}"',
    'route_status = "success"\n            route_error = None',
    1,
)
_src = _src.replace(
    'raise RuntimeError(f"LoFTR dependency unavailable: {loftr_err}")',
    'loftr_out = None  # Cloud-safe: skip LoFTR weight download',
    1,
)
# After skipping LoFTR raise, the following loftr_match call would still run —
# guard the whole block by making loftr_avail always false via inject below.

_INJECT = '''
# --- LunaMatch Cloud helpers (injected) ----------------------------------------
import os as _os_helper
try:
    from core.upload_utils import (
        MemUploadedFile,
        download_url_product,
        lightweight_file_id as _light_id,
        safe_file_size as _safe_size,
    )
except Exception:
    def _light_id(f):
        import hashlib
        n = getattr(f, "name", "") or ""
        s = getattr(f, "size", None)
        if s is None and hasattr(f, "getvalue"):
            try:
                s = len(f.getvalue())
            except Exception:
                s = 0
        return hashlib.sha256(f"{n}|{s}".encode()).hexdigest()[:16]

    def _safe_size(f):
        s = getattr(f, "size", None)
        if isinstance(s, int):
            return s
        return len(f.getvalue()) if f is not None and hasattr(f, "getvalue") else 0

    download_url_product = None

import loftr_matcher as _lm
_orig_loftr_avail = _lm.is_loftr_available

def _cloud_safe_loftr_available():
    if _os_helper.environ.get("LUNAMATCH_DISABLE_LOFTR") == "1":
        return False, "LoFTR disabled on Streamlit Cloud to prevent OOM."
    return _orig_loftr_avail()

_lm.is_loftr_available = _cloud_safe_loftr_available
'''

if "_cloud_safe_loftr_available" not in _src:
    _src = _src.replace("import streamlit as st\n", "import streamlit as st\n" + _INJECT, 1)

# Replace the broken raise-with-None path: inject SIFT runs when loftr_out is None
_src = _src.replace(
    'loftr_out = None  # Cloud-safe: skip LoFTR weight download\n\n            loftr_out = loftr_matcher.loftr_match(',
    'loftr_out = None  # Cloud-safe: skip LoFTR weight download\n            if False:  # disabled LoFTR call\n                loftr_out = loftr_matcher.loftr_match(',
    1,
)

# When loftr path was skipped, fill runs with SIFT before the LoFTR-dense assignment
_src = _src.replace(
    'runs = [\n                ("LoFTR-dense", loftr_out["correspondences"]),\n            ]',
    'if loftr_out is None:\n                runs = [\n                    ("SIFT-intensity",  sift_run(src.gray,      ref.gray,      src.mask, ref.mask, feature_count, ratio_threshold)),\n                    ("SIFT-structure",  sift_run(src.structure,  ref.structure,  src.mask, ref.mask, feature_count, ratio_threshold)),\n                    ("SIFT-gradient",   sift_run(src.gradient,   ref.gradient,   src.mask, ref.mask, feature_count, ratio_threshold)),\n                ]\n                actual_matcher = "SIFT"\n                matcher_note = "Cross-sensor Cloud-safe path: SIFT multi-rep (LoFTR disabled to prevent OOM)"\n            else:\n                runs = [\n                    ("LoFTR-dense", loftr_out["correspondences"]),\n                ]',
    1,
)

if os.environ.get("LUNAMATCH_DISABLE_LOFTR") == "1":
    # Banner is injected near uploader via simple caption after imports load
    pass

_src = _src.replace(
    'reference_file = st.file_uploader(\n        "Upload reference image", type=["tif", "tiff", "png", "jpg", "jpeg", "npy", "img", "raw", "dat"], key="reference"\n    )',
    'reference_file = st.file_uploader(\n        "Upload reference image", type=["tif", "tiff", "png", "jpg", "jpeg", "npy", "img", "raw", "dat"], key="reference",\n        help="Large LROC EDR (~250 MB) uploads are limited by your home uplink.",\n    )\n    if os.environ.get("LUNAMATCH_DISABLE_LOFTR") == "1":\n        st.info("**Cloud-safe mode:** LoFTR disabled to prevent OOM. OHRC↔LROC uses SIFT multi-rep. max_side default 768.")',
    1,
)

exec(compile(_src, "app_v3.py", "exec"), globals())
