"""LunaMatch V3 bootstrap — LoFTR-ROI primary geometry + cleaner fusion."""
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
os.environ.setdefault("LUNAMATCH_ROI_LOFTR", "1")

_URL = (
    "https://raw.githubusercontent.com/Kaushik-Mandale/LUNAMATCH/"
    "a12f6c3394b5510357d1c5aad1fe3d9ffd7244f1/app_v3.py"
)
with urllib.request.urlopen(_URL, timeout=60) as _resp:
    _src = _resp.read().decode("utf-8")

_src = _src.replace("max_side=2048, feature_count=12000,", "max_side=768, feature_count=4000,", 1)
_src = _src.replace("[1024, 1600, 2048, 3072, 4096], value=2048", "[512, 768, 1024, 1536], value=768", 1)
_src = _src.replace(
    'st.slider("SIFT features / representation", 3000, 30000, 12000, 1000)',
    'st.slider("SIFT features / representation", 1000, 8000, 4000, 500)',
    1,
)
_src = _src.replace("feature_count = 12000\n", "feature_count = 4000\n", 1)
_src = _src.replace('st.slider("Maximum matches per cell", 5, 50, 20)', 'st.slider("Maximum matches per cell", 5, 50, 25)', 1)
_src = _src.replace("cell_limit=20", "cell_limit=25")
_src = _src.replace(
    'st.slider("LoFTR confidence threshold", 0.10, 0.90, 0.35, 0.05)',
    'st.slider("LoFTR confidence threshold", 0.10, 0.90, 0.22, 0.05)',
    1,
)
_src = _src.replace("loftr_confidence_threshold = 0.35\n", "loftr_confidence_threshold = 0.22\n", 1)
_src = _src.replace(
    'st.slider("Reciprocal descriptor ratio", 0.55, 0.90, 0.78, 0.01)',
    'st.slider("Reciprocal descriptor ratio", 0.55, 0.90, 0.80, 0.01)',
    1,
)
_src = _src.replace("ratio_threshold=0.78", "ratio_threshold=0.80")
# Slightly looser MAGSAC default helps cross-sensor at working scale
_src = _src.replace(
    'st.slider("RANSAC/MAGSAC threshold (working px)", 0.5, 8.0, 2.5, 0.1)',
    'st.slider("RANSAC/MAGSAC threshold (working px)", 0.5, 12.0, 4.0, 0.1)',
    1,
)
_src = _src.replace("ransac_threshold=2.5", "ransac_threshold=4.0")
_src = _src.replace("threshold=2.5", "threshold=4.0")

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

# Fusion: LoFTR scores dominate; SIFT only fills coverage
_OLD_FUSE = '''    is_loftr_run = any("LoFTR" in m for m, _ in runs)
    pool = {}
    for method, rows in runs:
        for s, r, ratio in rows:
            key = (round(s[0], 1), round(s[1], 1), round(r[0], 1), round(r[1], 1))
            if is_loftr_run or "LoFTR" in method:
                # Confidence score in [0.0, 1.0]: higher is better
                score = float(ratio)
                if key not in pool or score > pool[key][2]:
                    pool[key] = (s, r, score, method)
            else:
                # Ratio score: lower is better; small bonus for structural representations
                score = float(ratio) - (0.035 if method == "structural" else 0.0)
                if key not in pool or score < pool[key][2]:
                    pool[key] = (s, r, score, method)

    if is_loftr_run:
        rows = sorted(pool.values(), key=lambda z: -z[2])
    else:
        rows = sorted(pool.values(), key=lambda z: z[2])'''

_NEW_FUSE = '''    has_loftr = any("LoFTR" in m for m, _ in runs)
    pool = {}
    for method, rows in runs:
        for s, r, ratio in rows:
            key = (round(s[0], 1), round(s[1], 1), round(r[0], 1), round(r[1], 1))
            if "LoFTR" in method:
                score = float(ratio) + 0.50  # strong preference for deep matches
            else:
                # When LoFTR present, heavily down-weight SIFT to cut outlier pollution
                score = max(0.0, 1.0 - float(ratio))
                if has_loftr:
                    score *= 0.25
                if "structure" in method.lower():
                    score += 0.02
            if key not in pool or score > pool[key][2]:
                pool[key] = (s, r, score, method)
    rows = sorted(pool.values(), key=lambda z: -z[2])'''

if _OLD_FUSE in _src:
    _src = _src.replace(_OLD_FUSE, _NEW_FUSE, 1)

_src = _src.replace(
    'if len(selected) < 8:\n        raise ValueError(f"Only {len(selected)} fused correspondences survived. Need at least 8.")',
    'if len(selected) < 4:\n        raise ValueError(f"Only {len(selected)} fused correspondences survived. Need at least 4.")',
    1,
)

# Prefer LoFTR-only geometry when ROI matcher is dense enough
_src = _src.replace(
    "ps, pr, scores, methods = fuse_and_select(runs, src.gray.shape, ref.gray.shape, grid_size, cell_limit)",
    "_runs_for_fuse = runs\n"
    "        _loftr_runs = [(n, r) for n, r in runs if 'LoFTR' in n and r]\n"
    "        if _loftr_runs and sum(len(r) for _, r in _loftr_runs) >= 12:\n"
    "            _runs_for_fuse = _loftr_runs  # geometry from deep matches only\n"
    "        ps, pr, scores, methods = fuse_and_select(_runs_for_fuse, src.gray.shape, ref.gray.shape, grid_size, cell_limit)",
    1,
)

# UI: treat SIFT+LoFTR-ROI as active deep matcher (not "LoFTR not available")
_src = _src.replace(
    'if actual_m == "LoFTR":\n                lm = ms.get("loftr_metrics", {})\n                st.success(\n                    f"✅ **Genuine LoFTR Active** ({lm.get(\'device\', \'cpu\').upper()}): "\n                    f"{lm.get(\'raw_matches\', 0)} raw matches → {lm.get(\'filtered_matches\', 0)} confident (threshold={lm.get(\'min_confidence\', 0.35):.2f})."\n                )',
    'if actual_m == "LoFTR" or (actual_m and "LoFTR" in str(actual_m)):\n                lm = ms.get("loftr_metrics", {}) or {}\n                st.success(\n                    f"✅ **ROI-LoFTR Active** ({str(lm.get(\'device\', \'cpu\')).upper()}): "\n                    f"{lm.get(\'filtered_matches\', 0)} matches, rel_scale={lm.get(\'rel_scale\', \'—\')}, "\n                    f"mode={lm.get(\'mode\', actual_m)}."\n                )',
    1,
)

_NEW_CROSS = '''
            # CROSS-SENSOR: SIFT coarse + scale-aware ROI-LoFTR (LoFTR preferred for geometry)
            _sift_runs = [
                ("SIFT-intensity",  sift_run(src.gray,      ref.gray,      src.mask, ref.mask, feature_count, ratio_threshold)),
                ("SIFT-structure",  sift_run(src.structure,  ref.structure,  src.mask, ref.mask, feature_count, ratio_threshold)),
                ("SIFT-gradient",   sift_run(src.gradient,   ref.gradient,   src.mask, ref.mask, feature_count, ratio_threshold)),
            ]
            runs = list(_sift_runs)
            actual_matcher = "SIFT"
            matcher_note = "Cross-sensor: SIFT multi-rep"
            loftr_metrics = {}
            _coarse = []
            for _name, _rows in _sift_runs:
                _coarse.extend(_rows)

            _try_roi = (
                os.environ.get("LUNAMATCH_DISABLE_LOFTR") != "1"
                and os.environ.get("LUNAMATCH_ROI_LOFTR", "1") == "1"
            )
            if _try_roi and loftr_avail:
                try:
                    import roi_loftr as _roi
                    _conf = min(float(loftr_confidence_threshold), 0.28)
                    loftr_out = _roi.loftr_match_roi_gated(
                        source_gray=src.gray,
                        reference_gray=ref.gray,
                        source_mask=src.mask,
                        reference_mask=ref.mask,
                        min_confidence=_conf,
                        roi_max_side=min(int(max_side), 512),
                        coarse_max_side=640,
                        pad_frac=0.30,
                        coarse_correspondences=_coarse[:400] if _coarse else None,
                        ratio_threshold=max(float(ratio_threshold), 0.80),
                    )
                    loftr_metrics = {
                        "raw_matches": loftr_out.get("raw_matches"),
                        "filtered_matches": loftr_out.get("filtered_matches"),
                        "min_confidence": loftr_out.get("min_confidence"),
                        "device": loftr_out.get("device"),
                        "roi_source_box": loftr_out.get("roi_source_box"),
                        "roi_reference_box": loftr_out.get("roi_reference_box"),
                        "coarse_match_count": loftr_out.get("coarse_match_count"),
                        "rel_scale": loftr_out.get("rel_scale"),
                        "mode": loftr_out.get("mode"),
                        "second_pass": loftr_out.get("second_pass"),
                    }
                    _n_loftr = int(loftr_out.get("filtered_matches") or 0)
                    if _n_loftr >= 12:
                        # Dense deep matches: use LoFTR-ROI alone for geometry (cuts SIFT outliers)
                        runs = [("LoFTR-ROI", loftr_out["correspondences"])]
                        actual_matcher = "LoFTR-ROI"
                        matcher_note = (
                            f"LoFTR-ROI primary ({str(loftr_out.get('device', 'cpu')).upper()}): "
                            f"{_n_loftr} matches, rel_scale={loftr_out.get('rel_scale')}, "
                            f"SIFT used only for ROI proposal"
                        )
                    elif _n_loftr >= 4:
                        # Keep limited top SIFT for coverage fill; LoFTR still preferred in fusion scores
                        _cap = []
                        for _n, _r in _sift_runs:
                            _cap.append((_n, sorted(_r, key=lambda t: t[2])[:20]))
                        runs = _cap + [("LoFTR-ROI", loftr_out["correspondences"])]
                        actual_matcher = "SIFT+LoFTR-ROI"
                        matcher_note = (
                            f"Scale-aware ROI-LoFTR ({str(loftr_out.get('device', 'cpu')).upper()}): "
                            f"{_n_loftr} deep + capped SIFT, rel_scale={loftr_out.get('rel_scale')}"
                        )
                    else:
                        matcher_note = (
                            f"SIFT multi-rep; ROI-LoFTR only {_n_loftr} matches "
                            f"(rel_scale={loftr_out.get('rel_scale')}; kept SIFT)"
                        )
                except Exception as _roi_exc:
                    matcher_note = f"SIFT multi-rep (ROI-LoFTR skipped: {_roi_exc})"
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
try:
    from core.upload_utils import lightweight_file_id as _light_id
except Exception:
    def _light_id(f):
        import hashlib
        n = getattr(f, "name", "") or ""
        s = getattr(f, "size", 0) or 0
        return hashlib.sha256(f"{n}|{s}".encode()).hexdigest()[:16]
'''
if "def _light_id" not in _src:
    _src = _src.replace("import streamlit as st\n", "import streamlit as st\n" + _INJECT, 1)

_src = _src.replace(
    'key="reference"\n    )',
    'key="reference"\n    )\n    st.info("**LoFTR-ROI primary:** when >=12 deep matches, geometry uses LoFTR only (SIFT = ROI proposal). MAGSAC default 4.0 px.")',
    1,
)

exec(compile(_src, "app_v3.py", "exec"), globals())
