"""LunaMatch V3 — bootstrap from last known-good commit while memory-optimized
helpers (loftr_matcher, core/sr_load) stay on main.

Streamlit Cloud defaults to lower processing limits via the restored UI; prefer
max_side=1024 and feature_count<=5000 on free tier.
"""
from __future__ import annotations

import os
import urllib.request

# Thread caps before torch/numpy initialize
for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("TORCH_NUM_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)

# Last complete app_v3 on main before the size-limit push issues
_GOOD_COMMIT = "a12f6c3394b5510357d1c5aad1fe3d9ffd7244f1"
_URL = (
    f"https://raw.githubusercontent.com/Kaushik-Mandale/LUNAMATCH/"
    f"{_GOOD_COMMIT}/app_v3.py"
)

with urllib.request.urlopen(_URL, timeout=60) as _resp:
    _src = _resp.read().decode("utf-8")

# Soft-patch cloud-friendly defaults into the loaded source
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
_src = _src.replace(
    "feature_count = 12000\n",
    "feature_count = 5000\n",
    1,
)

# Avoid reading entire large scientific binaries into RAM just to show a size caption.
# Streamlit UploadedFile exposes .size; fall back only when missing.
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

exec(compile(_src, "app_v3.py", "exec"), globals())
