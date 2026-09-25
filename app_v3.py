"""LunaMatch V3 bootstrap — Cloud memory / OOM hard-fail mitigations.

Root cause of Streamlit "Oh no" after model download reaches 100%:
  Torch + kornia LoFTR weights + large IMG exceed free-tier RAM and the
  process is killed. Fix: disable LoFTR on Cloud by default and route
  cross-sensor pairs through the SIFT multi-representation path.
"""
from __future__ import annotations

import os
import urllib.request

# Thread caps before NumPy / Torch initialize
for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("TORCH_NUM_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)

# Detect Streamlit Cloud / Sharing and disable LoFTR unless explicitly allowed.
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

# --- Cloud processing defaults -------------------------------------------------
_src = _src.replace(
    "max_side=2048, feature_count=12000,",
    "max_side=768, feature_count=3000,",
    1,
)
_src = _src.replace(
    "[1024, 1600, 2048, 3072, 4096], value=2048",
    "[512, 768, 1024, 1536], value=768",
    1,
)
_src = _src.replace(
    'st.slider("SIFT features / representation", 3000, 30000, 12000, 1000)',
    'st.slider("SIFT features / representation", 1000, 8000, 3000, 500)',
    1,
)
_src = _src.replace("feature_count = 12000\n", "feature_count = 3000\n", 1)

# --- Avoid loading entire 252 MB buffer for captions / identity ----------------
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

# --- Cloud-safe matcher routing: never OOM on LoFTR weights --------------------
# Stage 05: report SIFT fallback instead of hard-fail when LoFTR is disabled
_src = _src.replace(
    'matcher_method = f"DIFFERENT SENSOR ({source_sensor} \u2194 {reference_sensor}) \u2192 LoFTR requested but unavailable"\n'
    '            route_status = "failed"\n'
    '            route_error = f"LoFTR dependency missing: {loftr_err}"',
    'matcher_method = f"DIFFERENT SENSOR ({source_sensor} \u2194 {reference_sensor}) \u2192 SIFT multi-rep (Cloud-safe; LoFTR disabled to avoid OOM)"\n'
    '            route_status = "success"\n'
    '            route_error = None',
)
# Also match non-unicode arrows if present in source
_src = _src.replace(
    'matcher_method = f"DIFFERENT SENSOR ({source_sensor} \u2194 {reference_sensor}) \u2192 LoFTR requested but unavailable"',
    'matcher_method = f"DIFFERENT SENSOR ({source_sensor} \u2194 {reference_sensor}) \u2192 SIFT multi-rep (Cloud-safe)"',
)

# Stage 06 cross-sensor: use SIFT path instead of loading LoFTR when disabled
_OLD_CROSS = '''# CROSS-SENSOR BRANCH \u2192 Genuine LoFTR inference (kornia.feature.LoFTR)
            if not loftr_avail:
                raise RuntimeError(f"LoFTR dependency unavailable: {loftr_err}")

            loftr_out = loftr_matcher.loftr_match(
                source_gray=src.gray,
                reference_gray=ref.gray,
                source_mask=src.mask,
                reference_mask=ref.mask,
                min_confidence=float(loftr_confidence_threshold),
                max_side=min(max_side, 1024),
            )'''

_NEW_CROSS = '''# CROSS-SENSOR BRANCH \u2192 LoFTR when allowed; else SIFT multi-rep (Cloud-safe)
            if not loftr_avail or os.environ.get("LUNAMATCH_DISABLE_LOFTR") == "1":
                # Avoid torch/kornia weight download that OOM-kills Streamlit Cloud
                runs = [
                    ("SIFT-intensity",  sift_run(src.gray,      ref.gray,      src.mask, ref.mask, feature_count, ratio_threshold)),
                    ("SIFT-structure",  sift_run(src.structure,  ref.structure,  src.mask, ref.mask, feature_count, ratio_threshold)),
                    ("SIFT-gradient",   sift_run(src.gradient,   ref.gradient,   src.mask, ref.mask, feature_count, ratio_threshold)),
                ]
                actual_matcher = "SIFT"
                matcher_note = "Cross-sensor Cloud-safe path: SIFT on intensity/structure/gradient (LoFTR disabled to prevent OOM)"
                loftr_out = None
            else:
                loftr_out = loftr_matcher.loftr_match(
                    source_gray=src.gray,
                    reference_gray=ref.gray,
                    source_mask=src.mask,
                    reference_mask=ref.mask,
                    min_confidence=float(loftr_confidence_threshold),
                    max_side=min(max_side, 512),
                )
                try:
                    loftr_matcher.unload_loftr_model()
                except Exception:
                    pass'''

# The good-commit source uses unicode arrows — try both encodings
for _old, _new in (
    (_OLD_CROSS, _NEW_CROSS),
    (
        _OLD_CROSS.replace("\\u2192", "\u2192"),
        _NEW_CROSS.replace("\\u2192", "\u2192"),
    ),
):
    if _old in _src:
        _src = _src.replace(_old, _new, 1)
        break
else:
    # Fallback: simpler gate before loftr_match call
    _src = _src.replace(
        "max_side=min(max_side, 1024),",
        "max_side=min(max_side, 512),",
        1,
    )

# After LoFTR metrics block, skip runs assignment if already set by SIFT fallback
_src = _src.replace(
    'runs = [\n                ("LoFTR-dense", loftr_out["correspondences"]),\n            ]',
    'if loftr_out is not None:\n                runs = [\n                    ("LoFTR-dense", loftr_out["correspondences"]),\n                ]',
    1,
)

# Inject helpers after streamlit import
_INJECT = '''
# --- LunaMatch large-product + Cloud helpers (injected) ------------------------
import os as _os_helper
try:
    from core.upload_utils import (
        MemUploadedFile,
        download_url_product,
        lightweight_file_id as _light_id,
        safe_file_size as _safe_size,
    )
except Exception:  # pragma: no cover
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

# Force LoFTR availability off when Cloud OOM guard is active
import loftr_matcher as _lm
_orig_loftr_avail = _lm.is_loftr_available

def _cloud_safe_loftr_available():
    if _os_helper.environ.get("LUNAMATCH_DISABLE_LOFTR") == "1":
        return False, "LoFTR disabled on Streamlit Cloud to prevent OOM (set LUNAMATCH_ALLOW_LOFTR=1 to override)."
    return _orig_loftr_avail()

_lm.is_loftr_available = _cloud_safe_loftr_available
'''

if "_cloud_safe_loftr_available" not in _src:
    _src = _src.replace(
        "import streamlit as st\n",
        "import streamlit as st\n" + _INJECT,
        1,
    )

# URL fetch UI (optional speed path for 250 MB products)
_OLD_UPLOADER = '''    reference_file = st.file_uploader(
        "Upload reference image", type=["tif", "tiff", "png", "jpg", "jpeg", "npy", "img", "raw", "dat"], key="reference"
    )'''
_NEW_UPLOADER = '''    reference_file = st.file_uploader(
        "Upload reference image", type=["tif", "tiff", "png", "jpg", "jpeg", "npy", "img", "raw", "dat"], key="reference",
        help="Large LROC EDR (~250 MB) uploads are limited by your home uplink.",
    )
    if os.environ.get("LUNAMATCH_DISABLE_LOFTR") == "1":
        st.info(
            "**Cloud-safe mode:** LoFTR deep matching is disabled to prevent out-of-memory crashes. "
            "Cross-sensor pairs (e.g. OHRC \u2194 LROC) use the SIFT multi-representation path. "
            "Default max side is 768."
        )
    with st.expander("Faster: load reference from direct URL (server-side fetch)", expanded=False):
        st.caption("Paste a direct HTTP(S) link to the .IMG. Server download is often faster than home upload.")
        _ref_url = st.text_input("Reference product URL", key="_ref_url_input", placeholder="https://\u2026/M1438615574LE.IMG")
        if st.button("Fetch reference on server", key="_ref_url_fetch") and _ref_url and download_url_product is not None:
            try:
                with st.spinner("Server downloading reference product\u2026"):
                    reference_file = download_url_product(_ref_url)
                st.session_state["_ref_url_file"] = reference_file
                st.success(f"Loaded **{reference_file.name}** ({reference_file.size/1024/1024:.1f} MB).")
            except Exception as _url_exc:
                st.error(f"URL fetch failed: {_url_exc}")
        elif st.session_state.get("_ref_url_file") is not None and reference_file is None:
            reference_file = st.session_state["_ref_url_file"]
            st.caption(f"Using server-fetched reference: **{reference_file.name}**")'''

if _OLD_UPLOADER in _src:
    _src = _src.replace(_OLD_UPLOADER, _NEW_UPLOADER, 1)

exec(compile(_src, "app_v3.py", "exec"), globals())
