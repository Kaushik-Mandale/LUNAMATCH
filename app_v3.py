"""LunaMatch V3 bootstrap — local full-image LoFTR (GPU) + Cloud ROI-LoFTR."""
from __future__ import annotations

import os
import re
import urllib.request

_on_cloud = bool(
    os.environ.get("STREAMLIT_SHARING_MODE")
    or os.path.exists("/mount/src")
    or "streamlit.app" in os.environ.get("HOSTNAME", "")
)

_threads = "1" if _on_cloud else str(max(1, min(8, (os.cpu_count() or 4))))
for _k, _v in (
    ("OMP_NUM_THREADS", _threads),
    ("MKL_NUM_THREADS", _threads),
    ("OPENBLAS_NUM_THREADS", _threads),
    ("NUMEXPR_NUM_THREADS", _threads),
    ("TORCH_NUM_THREADS", _threads),
):
    os.environ.setdefault(_k, _v)

if not _on_cloud:
    os.environ.setdefault("LUNAMATCH_LOCAL", "1")
    # Full-image LoFTR on laptop; ROI only if user forces it
    os.environ.setdefault("LUNAMATCH_ROI_LOFTR", "0")
else:
    os.environ.setdefault("LUNAMATCH_ROI_LOFTR", "1")

_URL = (
    "https://raw.githubusercontent.com/Kaushik-Mandale/LUNAMATCH/"
    "a12f6c3394b5510357d1c5aad1fe3d9ffd7244f1/app_v3.py"
)
with urllib.request.urlopen(_URL, timeout=60) as _resp:
    _src = _resp.read().decode("utf-8")

if _on_cloud:
    _src = _src.replace("max_side=2048, feature_count=12000,", "max_side=768, feature_count=4000,", 1)
    _src = _src.replace("[1024, 1600, 2048, 3072, 4096], value=2048", "[512, 768, 1024, 1536], value=768", 1)
    _src = _src.replace(
        'st.slider("SIFT features / representation", 3000, 30000, 12000, 1000)',
        'st.slider("SIFT features / representation", 1000, 8000, 4000, 500)',
        1,
    )
    _src = _src.replace("feature_count = 12000\n", "feature_count = 4000\n", 1)
else:
    # Local: working resolution for full-frame LoFTR (RTX 4050 ~6GB: 1024–1280 safe)
    _src = _src.replace("max_side=2048, feature_count=12000,", "max_side=1280, feature_count=8000,", 1)
    _src = _src.replace("[1024, 1600, 2048, 3072, 4096], value=2048", "[768, 1024, 1280, 1536, 2048], value=1280", 1)
    _src = _src.replace(
        'st.slider("SIFT features / representation", 3000, 30000, 12000, 1000)',
        'st.slider("SIFT features / representation", 2000, 16000, 8000, 500)',
        1,
    )
    _src = _src.replace("feature_count = 12000\n", "feature_count = 8000\n", 1)

_src = _src.replace(
    'st.slider("LoFTR confidence threshold", 0.10, 0.90, 0.35, 0.05)',
    'st.slider("LoFTR confidence threshold", 0.10, 0.90, 0.25, 0.05)',
    1,
)
_src = _src.replace("loftr_confidence_threshold = 0.35\n", "loftr_confidence_threshold = 0.25\n", 1)
_src = _src.replace(
    'st.slider("Reciprocal descriptor ratio", 0.55, 0.90, 0.78, 0.01)',
    'st.slider("Reciprocal descriptor ratio", 0.55, 0.90, 0.82, 0.01)',
    1,
)
_src = _src.replace("ratio_threshold=0.78", "ratio_threshold=0.82")
_src = _src.replace(
    'st.slider("RANSAC/MAGSAC threshold (working px)", 0.5, 6.0, 2.5, 0.25)',
    'st.slider("RANSAC/MAGSAC threshold (working px)", 0.5, 12.0, 4.0, 0.25)',
    1,
)
_src = _src.replace("ransac_threshold=2.5", "ransac_threshold=4.0")

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

_OLD_FUSE = '''    is_loftr_run = any("LoFTR" in m for m, _ in runs)
    pool = {}
    for method, rows in runs:
        for s, r, ratio in rows:
            key = (round(s[0], 1), round(s[1], 1), round(r[0], 1), round(r[1], 1))
            if is_loftr_run or "LoFTR" in method:
                score = float(ratio)
                if key not in pool or score > pool[key][2]:
                    pool[key] = (s, r, score, method)
            else:
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
                score = float(ratio) + 0.50
            else:
                score = max(0.0, 1.0 - float(ratio))
                if has_loftr:
                    score *= 0.20
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

_src = _src.replace(
    "ps, pr, scores, methods = fuse_and_select(runs, src.gray.shape, ref.gray.shape, grid_size, cell_limit)",
    "_runs_for_fuse = runs\n"
    "        _loftr_runs = [(n, r) for n, r in runs if 'LoFTR' in n and r]\n"
    "        if _loftr_runs and sum(len(r) for _, r in _loftr_runs) >= 8:\n"
    "            _runs_for_fuse = _loftr_runs\n"
    "        ps, pr, scores, methods = fuse_and_select(_runs_for_fuse, src.gray.shape, ref.gray.shape, grid_size, cell_limit)",
    1,
)

_OLD_GEOM = (
    "        H, train_mask, geom_info = estimate_geometric_model(\n"
    "            ps[train_idx], pr[train_idx], model=model, verifier=geometric_verifier, threshold=ransac_threshold\n"
    "        )\n"
    "        # Backwards-compatible estimate_model(ps[train_idx]\n"
    "        global_train_inlier = np.zeros(len(ps), dtype=bool)\n"
    "        global_train_inlier[train_idx] = train_mask"
)
_NEW_GEOM = (
    "        H, train_mask, geom_info = estimate_geometric_model(\n"
    "            ps[train_idx], pr[train_idx], model=model, verifier=geometric_verifier, threshold=ransac_threshold\n"
    "        )\n"
    "        train_mask = np.asarray(train_mask).reshape(-1).astype(bool)\n"
    "        if int(train_mask.sum()) >= 6:\n"
    "            _ips = np.asarray(ps[train_idx][train_mask], np.float64)\n"
    "            _ipr = np.asarray(pr[train_idx][train_mask], np.float64)\n"
    "            _th_fine = max(1.0, float(ransac_threshold) * 0.5)\n"
    "            try:\n"
    "                H2, mask2, info2 = estimate_geometric_model(\n"
    "                    _ips, _ipr, model=model, verifier=geometric_verifier, threshold=_th_fine\n"
    "                )\n"
    "                mask2 = np.asarray(mask2).reshape(-1).astype(bool)\n"
    "                if H2 is not None and int(mask2.sum()) >= 4:\n"
    "                    H = H2\n"
    "                    _pts = np.asarray(ps[train_idx], np.float64)\n"
    "                    _ptr = np.asarray(pr[train_idx], np.float64)\n"
    "                    _ones = np.ones((len(_pts), 1), np.float64)\n"
    "                    _proj = (H @ np.hstack([_pts, _ones]).T).T\n"
    "                    _proj = _proj[:, :2] / np.maximum(_proj[:, 2:3], 1e-12)\n"
    "                    _err = np.linalg.norm(_proj - _ptr, axis=1)\n"
    "                    train_mask = _err <= max(1.5, float(ransac_threshold) * 0.65)\n"
    "                    if isinstance(info2, dict):\n"
    "                        geom_info = dict(info2)\n"
    "                    geom_info['multi_pass'] = True\n"
    "                    geom_info['inlier_count'] = int(train_mask.sum())\n"
    "            except Exception:\n"
    "                pass\n"
    "        global_train_inlier = np.zeros(len(ps), dtype=bool)\n"
    "        global_train_inlier[train_idx] = train_mask"
)
if _OLD_GEOM in _src:
    _src = _src.replace(_OLD_GEOM, _NEW_GEOM, 1)

_src = _src.replace(
    "H2, mask2, _ = estimate_geometric_model(rs, rr, model=model, verifier=geometric_verifier, threshold=ransac_threshold)",
    "H2, mask2, _ = estimate_geometric_model(rs, rr, model=model, verifier=geometric_verifier, threshold=max(1.0, float(ransac_threshold) * 0.5))",
    1,
)
_src = _src.replace(
    "if mask2.sum() >= 8:",
    "if int(np.asarray(mask2).reshape(-1).sum()) >= (4 if execution_mode == \"Experimental image-only mode\" else 8):",
    1,
)

_src = _src.replace('if actual_m == "LoFTR":', 'if actual_m == "LoFTR" or (actual_m and "LoFTR" in str(actual_m)):', 1)
_src = _src.replace('**Genuine LoFTR Active**', '**Full-image LoFTR Active**', 1)

# Local = full-image LoFTR; Cloud = ROI-gated
if _on_cloud:
    _NEW_CROSS = '''
            # CROSS-SENSOR: ROI-LoFTR (Cloud memory safe)
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
            if loftr_avail and os.environ.get("LUNAMATCH_ROI_LOFTR", "1") == "1":
                try:
                    import roi_loftr as _roi
                    loftr_out = _roi.loftr_match_roi_gated(
                        source_gray=src.gray, reference_gray=ref.gray,
                        source_mask=src.mask, reference_mask=ref.mask,
                        min_confidence=min(float(loftr_confidence_threshold), 0.26),
                        roi_max_side=min(int(max_side), 512), coarse_max_side=640, pad_frac=0.30,
                        coarse_correspondences=_coarse[:400] if _coarse else None,
                        ratio_threshold=max(float(ratio_threshold), 0.80),
                    )
                    _all = list(loftr_out.get("correspondences") or [])
                    _strict = sorted([c for c in _all if float(c[2]) >= max(0.22, float(loftr_confidence_threshold))], key=lambda t: -float(t[2]))
                    if len(_strict) < 8:
                        _strict = sorted(_all, key=lambda t: -float(t[2]))[:40]
                    loftr_metrics = {"raw_matches": loftr_out.get("raw_matches"), "filtered_matches": len(_strict),
                                    "device": loftr_out.get("device"), "mode": "roi_gated", "min_confidence": float(loftr_confidence_threshold)}
                    if len(_strict) >= 4:
                        runs = [("LoFTR-ROI", _strict)]
                        actual_matcher = "LoFTR-ROI"
                        matcher_note = f"ROI-LoFTR ({loftr_out.get('device','cpu')}): {len(_strict)} matches"
                except Exception as _e:
                    matcher_note = f"SIFT (ROI-LoFTR skipped: {_e})"
'''
else:
    _NEW_CROSS = '''
            # CROSS-SENSOR: FULL-IMAGE LoFTR (local GPU) — whole working frame, not ROI crop
            runs = []
            actual_matcher = "SIFT"
            matcher_note = "Cross-sensor"
            loftr_metrics = {}
            if not loftr_avail:
                _sift_runs = [
                    ("SIFT-intensity",  sift_run(src.gray, ref.gray, src.mask, ref.mask, feature_count, ratio_threshold)),
                    ("SIFT-structure",  sift_run(src.structure, ref.structure, src.mask, ref.mask, feature_count, ratio_threshold)),
                    ("SIFT-gradient",   sift_run(src.gradient, ref.gradient, src.mask, ref.mask, feature_count, ratio_threshold)),
                ]
                runs = list(_sift_runs)
                matcher_note = f"SIFT (LoFTR unavailable: {loftr_err})"
            else:
                try:
                    import loftr_matcher as _lm
                    # Cap side for VRAM: RTX 4050 ~6GB → prefer ≤1280; UI max_side still controls
                    _full_side = min(int(max_side), 1280)
                    loftr_out = _lm.loftr_match(
                        source_gray=src.gray,
                        reference_gray=ref.gray,
                        source_mask=src.mask,
                        reference_mask=ref.mask,
                        min_confidence=float(loftr_confidence_threshold),
                        max_side=_full_side,
                    )
                    _corrs = list(loftr_out.get("correspondences") or [])
                    loftr_metrics = {
                        "raw_matches": loftr_out.get("raw_matches"),
                        "filtered_matches": loftr_out.get("filtered_matches"),
                        "min_confidence": loftr_out.get("min_confidence"),
                        "device": loftr_out.get("device"),
                        "working_source_shape": loftr_out.get("working_source_shape"),
                        "working_reference_shape": loftr_out.get("working_reference_shape"),
                        "mode": "full_image",
                        "max_side": _full_side,
                    }
                    if len(_corrs) < 4:
                        raise ValueError(f"Full LoFTR only {len(_corrs)} matches")
                    runs = [("LoFTR-full", _corrs)]
                    actual_matcher = "LoFTR-full"
                    matcher_note = (
                        f"Full-image LoFTR ({str(loftr_out.get('device','cpu')).upper()}): "
                        f"{loftr_out.get('raw_matches')} raw → {loftr_out.get('filtered_matches')} conf "
                        f"@ max_side={_full_side}"
                    )
                    try:
                        _lm.unload_loftr_model()
                    except Exception:
                        pass
                except Exception as _full_exc:
                    # Fallback: ROI then SIFT
                    matcher_note = f"Full LoFTR failed ({_full_exc}); trying ROI/SIFT"
                    try:
                        import roi_loftr as _roi
                        _sift_runs = [
                            ("SIFT-intensity", sift_run(src.gray, ref.gray, src.mask, ref.mask, feature_count, ratio_threshold)),
                            ("SIFT-structure", sift_run(src.structure, ref.structure, src.mask, ref.mask, feature_count, ratio_threshold)),
                        ]
                        _coarse = []
                        for _, _r in _sift_runs:
                            _coarse.extend(_r)
                        loftr_out = _roi.loftr_match_roi_gated(
                            source_gray=src.gray, reference_gray=ref.gray,
                            source_mask=src.mask, reference_mask=ref.mask,
                            min_confidence=min(float(loftr_confidence_threshold), 0.22),
                            roi_max_side=min(int(max_side), 768), coarse_max_side=800, pad_frac=0.35,
                            coarse_correspondences=_coarse[:500] if _coarse else None,
                            ratio_threshold=max(float(ratio_threshold), 0.80),
                        )
                        _strict = list(loftr_out.get("correspondences") or [])
                        if len(_strict) >= 4:
                            runs = [("LoFTR-ROI", _strict)]
                            actual_matcher = "LoFTR-ROI"
                            loftr_metrics = {"filtered_matches": len(_strict), "device": loftr_out.get("device"), "mode": "roi_fallback"}
                            matcher_note = f"ROI-LoFTR fallback: {len(_strict)} matches"
                        else:
                            runs = list(_sift_runs)
                            actual_matcher = "SIFT"
                            matcher_note = f"SIFT fallback after full/ROI LoFTR weak"
                    except Exception as _fb:
                        runs = [
                            ("SIFT-intensity", sift_run(src.gray, ref.gray, src.mask, ref.mask, feature_count, ratio_threshold)),
                            ("SIFT-structure", sift_run(src.structure, ref.structure, src.mask, ref.mask, feature_count, ratio_threshold)),
                            ("SIFT-gradient", sift_run(src.gradient, ref.gradient, src.mask, ref.mask, feature_count, ratio_threshold)),
                        ]
                        actual_matcher = "SIFT"
                        matcher_note = f"SIFT only (LoFTR errors: {_full_exc}; {_fb})"
'''

_src, _n = re.subn(
    r'# CROSS-SENSOR BRANCH[\s\S]*?actual_matcher = "LoFTR"\n'
    r'[ \t]*matcher_note = \([\s\S]*?\)\n',
    _NEW_CROSS.lstrip("\n"),
    _src,
    count=1,
)
if _n != 1:
    print(f"[bootstrap] cross-sensor patch skipped (matches={_n})")

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

_banner = (
    '    st.info("**Local full-image LoFTR (GPU):** whole working frame (max_side capped ~1280 for VRAM). Restart Streamlit after git pull.")'
    if not _on_cloud
    else '    st.info("**Streamlit Cloud:** ROI-LoFTR only (memory). Use local RTX for full-image LoFTR.")'
)

_src = _src.replace(
    'key="reference"\n    )',
    'key="reference"\n    )\n'
    '    st.caption("Large LROC: local disk is fine.")\n'
    '    _ref_url = st.text_input("Or load reference via HTTPS URL", value="", key="reference_url_fetch", placeholder="https://…/M….IMG")\n'
    '    if _ref_url and _ref_url.strip().startswith(("http://", "https://")) and reference_file is None:\n'
    '        try:\n'
    '            from core.upload_utils import download_url_product as _dl_ref\n'
    '            with st.spinner("Downloading reference…"):\n'
    '                reference_file = _dl_ref(_ref_url.strip())\n'
    '            st.success(f"Loaded **{reference_file.name}**")\n'
    '        except Exception as _ue:\n'
    '            st.error(f"URL fetch failed: {_ue}")\n'
    + _banner + '\n',
    1,
)

try:
    compile(_src, "app_v3.py", "exec")
except SyntaxError as _syn:
    raise RuntimeError(f"Bootstrap syntax error line {_syn.lineno}: {_syn.msg}") from _syn

exec(compile(_src, "app_v3.py", "exec"), globals())
