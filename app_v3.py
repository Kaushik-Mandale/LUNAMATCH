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

# Patches embedded as base64 so unicode arrows survive transport
_src = _src.replace(_b64(open.__doc__ and ""), "", 0)  # no-op keep structure
