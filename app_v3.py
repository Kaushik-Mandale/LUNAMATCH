"""LunaMatch V3 bootstrap — memory + large-product transfer optimisations.

Loads the last complete app_v3 from git history, then soft-patches:
  - lower Cloud defaults (max_side=1024, features=5000)
  - avoid full getvalue() for size captions / file identity on big IMGs
  - optional server-side URL fetch for reference products (faster than home uplink)
"""
from __future__ import annotations

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
    "max_side=1024, feature_count=5000,",
    1,
)
_src = _src.replace(
    "[1024, 1600, 2048, 3072, 4096], value=2048",
    "[512, 768, 1024, 1536, 2048], value=1024",
    1,
)
_src = _src.replace(
    'st.slider("SIFT features / representation", 3000, 30000, 12000, 1000)',
    'st.slider("SIFT features / representation", 2000, 15000, 5000, 500)',
    1,
)
_src = _src.replace("feature_count = 12000\n", "feature_count = 5000\n", 1)

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
# Session invalidation: do not SHA-256 the full 252 MB on every rerun
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

# Inject lightweight helpers + URL loader right after streamlit import block
_INJECT = '''
# --- LunaMatch large-product helpers (injected) --------------------------------
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
'''

if "lightweight_file_id as _light_id" not in _src:
    _src = _src.replace(
        "import streamlit as st\n",
        "import streamlit as st\n" + _INJECT,
        1,
    )

# Inject URL-based reference load UI after the reference file_uploader
_OLD_UPLOADER = '''    reference_file = st.file_uploader(
        "Upload reference image", type=["tif", "tiff", "png", "jpg", "jpeg", "npy", "img", "raw", "dat"], key="reference"
    )'''

_NEW_UPLOADER = '''    reference_file = st.file_uploader(
        "Upload reference image", type=["tif", "tiff", "png", "jpg", "jpeg", "npy", "img", "raw", "dat"], key="reference",
        help="Large LROC EDR (~250 MB) uploads are limited by your home uplink. Prefer the URL option below when possible.",
    )
    with st.expander("Faster: load reference from direct URL (server-side fetch)", expanded=False):
        st.caption(
            "Paste a direct HTTP(S) link to the .IMG (PDS / LROC / your own host). "
            "The Streamlit server downloads it — usually faster than uploading 250 MB from home."
        )
        _ref_url = st.text_input(
            "Reference product URL",
            value=st.session_state.get("_ref_url_input", ""),
            key="_ref_url_input",
            placeholder="https://…/M1438615574LE.IMG",
        )
        _fetch = st.button("Fetch reference on server", key="_ref_url_fetch")
        if _fetch and _ref_url and download_url_product is not None:
            try:
                _bar = st.progress(0, text="Downloading reference on server…")
                def _cb(n):
                    # indeterminate-ish progress from bytes (cap display at 250 MB)
                    _bar.progress(min(n / (252 * 1024 * 1024), 0.99), text=f"Downloaded {n/1024/1024:.1f} MB…")
                with st.spinner("Server downloading reference product…"):
                    reference_file = download_url_product(_ref_url, progress_cb=_cb)
                _bar.progress(1.0, text="Download complete")
                st.session_state["_ref_url_file"] = reference_file
                st.success(f"Loaded **{reference_file.name}** ({reference_file.size/1024/1024:.1f} MB) via server fetch.")
            except Exception as _url_exc:
                st.error(f"URL fetch failed: {_url_exc}")
        elif st.session_state.get("_ref_url_file") is not None and reference_file is None:
            reference_file = st.session_state["_ref_url_file"]
            st.caption(f"Using server-fetched reference: **{reference_file.name}**")'''

if _OLD_UPLOADER in _src:
    _src = _src.replace(_OLD_UPLOADER, _NEW_UPLOADER, 1)
else:
    # Fallback: still provide helpers even if uploader text drifted
    pass

exec(compile(_src, "app_v3.py", "exec"), globals())
