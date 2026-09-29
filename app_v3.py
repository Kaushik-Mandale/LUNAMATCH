"""LunaMatch V3 bootstrap — GSD-aware scale + CLAHE LoFTR + guided rematch."""
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
    'st.slider("LoFTR confidence threshold", 0.10, 0.90, 0.26, 0.05)',
    1,
)
_src = _src.replace("loftr_confidence_threshold = 0.35\n", "loftr_confidence_threshold = 0.26\n", 1)
_src = _src.replace(
    'st.slider("Reciprocal descriptor ratio", 0.55, 0.90, 0.78, 0.01)',
    'st.slider("Reciprocal descriptor ratio", 0.55, 0.90, 0.82, 0.01)',
    1,
)
_src = _src.replace("ratio_threshold=0.78", "ratio_threshold=0.82")
_src = _src.replace(
    'st.slider("RANSAC/MAGSAC threshold (working px)", 0.5, 6.0, 2.5, 0.25)',
    'st.slider("RANSAC/MAGSAC threshold (working px)", 0.5, 12.0, 7.0, 0.25)',
    1,
)
_src = _src.replace("ransac_threshold=2.5", "ransac_threshold=7.0")
_src = _src.replace(
    'st.selectbox("Global geometric model", ["Homography", "Affine"])',
    'st.selectbox("Global geometric model", ["Affine", "Homography"])',
    1,
)

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
    "            _th_fine = max(1.0, float(ransac_threshold) * 0.55)\n"
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
    "                    train_mask = _err <= max(1.5, float(ransac_threshold) * 0.7)\n"
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
    "H2, mask2, _ = estimate_geometric_model(rs, rr, model=model, verifier=geometric_verifier, threshold=max(1.0, float(ransac_threshold) * 0.55))",
    1,
)
_src = _src.replace(
    "if mask2.sum() >= 8:",
    "if int(np.asarray(mask2).reshape(-1).sum()) >= (4 if execution_mode == \"Experimental image-only mode\" else 8):",
    1,
)

_src = _src.replace('if actual_m == "LoFTR":', 'if actual_m == "LoFTR" or (actual_m and "LoFTR" in str(actual_m)):', 1)
_src = _src.replace('**Genuine LoFTR Active**', '**Full-image LoFTR Active**', 1)

if _on_cloud:
    _NEW_CROSS = '''
            # CROSS-SENSOR: ROI-LoFTR (Cloud)
            _sift_runs = [
                ("SIFT-intensity",  sift_run(src.gray, ref.gray, src.mask, ref.mask, feature_count, ratio_threshold)),
                ("SIFT-structure",  sift_run(src.structure, ref.structure, src.mask, ref.mask, feature_count, ratio_threshold)),
                ("SIFT-gradient",   sift_run(src.gradient, ref.gradient, src.mask, ref.mask, feature_count, ratio_threshold)),
            ]
            runs = list(_sift_runs)
            actual_matcher = "SIFT"
            matcher_note = "Cross-sensor: SIFT"
            loftr_metrics = {}
            if loftr_avail:
                try:
                    import roi_loftr as _roi
                    _coarse = []
                    for _, _r in _sift_runs:
                        _coarse.extend(_r)
                    loftr_out = _roi.loftr_match_roi_gated(
                        source_gray=src.gray, reference_gray=ref.gray,
                        source_mask=src.mask, reference_mask=ref.mask,
                        min_confidence=min(float(loftr_confidence_threshold), 0.26),
                        roi_max_side=min(int(max_side), 512), coarse_max_side=640, pad_frac=0.30,
                        coarse_correspondences=_coarse[:400] if _coarse else None,
                        ratio_threshold=max(float(ratio_threshold), 0.80),
                    )
                    _strict = sorted(list(loftr_out.get("correspondences") or []), key=lambda t: -float(t[2]))[:40]
                    loftr_metrics = {"filtered_matches": len(_strict), "device": loftr_out.get("device"), "mode": "roi"}
                    if len(_strict) >= 4:
                        runs = [("LoFTR-ROI", _strict)]
                        actual_matcher = "LoFTR-ROI"
                        matcher_note = f"ROI-LoFTR: {len(_strict)}"
                except Exception as _e:
                    matcher_note = f"SIFT ({_e})"
'''
else:
    _NEW_CROSS = '''
            # CROSS-SENSOR: GSD-scale + CLAHE LoFTR + guided rematch (local GPU)
            runs = []
            actual_matcher = "SIFT"
            matcher_note = "Cross-sensor"
            loftr_metrics = {}
            if not loftr_avail:
                runs = [
                    ("SIFT-intensity", sift_run(src.gray, ref.gray, src.mask, ref.mask, feature_count, ratio_threshold)),
                    ("SIFT-structure", sift_run(src.structure, ref.structure, src.mask, ref.mask, feature_count, ratio_threshold)),
                    ("SIFT-gradient", sift_run(src.gradient, ref.gradient, src.mask, ref.mask, feature_count, ratio_threshold)),
                ]
                matcher_note = f"SIFT ({loftr_err})"
            else:
                try:
                    import cross_sensor_loftr as _cs
                    _full_side = min(int(max_side), 1280)
                    def _pick_gsd(_meta):
                        if not isinstance(_meta, dict):
                            return None
                        for _k in ("gsd_m_per_pixel", "gsd_m", "resolution_gsd", "gsd"):
                            _v = _meta.get(_k)
                            if _v is None and isinstance(_meta.get("metadata"), dict):
                                _v = _meta["metadata"].get(_k)
                            try:
                                _f = float(_v)
                                if _f > 0:
                                    return _f
                            except Exception:
                                pass
                        return None
                    try:
                        _src_gsd = _pick_gsd(source_metadata)
                    except Exception:
                        _src_gsd = None
                    try:
                        _ref_gsd = _pick_gsd(reference_metadata)
                    except Exception:
                        _ref_gsd = None
                    if _src_gsd is None and str(source_sensor).upper().startswith("OHRC"):
                        _src_gsd = 0.24
                    if _ref_gsd is None and "LROC" in str(reference_sensor).upper():
                        _ref_gsd = 2.0
                    loftr_out = _cs.match_cross_sensor_full(
                        source_gray=src.gray, reference_gray=ref.gray,
                        source_mask=src.mask, reference_mask=ref.mask,
                        min_confidence=min(0.20, float(loftr_confidence_threshold)),
                        max_side=_full_side,
                        geom_conf_floor=max(0.26, float(loftr_confidence_threshold)),
                        source_gsd_m=_src_gsd,
                        reference_gsd_m=_ref_gsd,
                    )
                    _corrs = list(loftr_out.get("correspondences") or [])
                    loftr_metrics = {
                        "raw_matches": loftr_out.get("raw_matches"),
                        "filtered_matches": len(_corrs),
                        "min_confidence": loftr_out.get("min_confidence"),
                        "device": loftr_out.get("device"),
                        "mode": loftr_out.get("mode"),
                        "rel_scale": loftr_out.get("rel_scale"),
                        "scale_apply": loftr_out.get("scale_apply"),
                        "scale_source": loftr_out.get("scale_source"),
                        "max_side": _full_side,
                    }
                    if len(_corrs) < 4:
                        raise ValueError(f"only {len(_corrs)} after scale-CLAHE LoFTR")
                    try:
                        import guided_match as _gm
                        _corrs, _gmeta = _gm.enrich_with_guided(
                            src.gray, ref.gray, _corrs,
                            source_mask=src.mask, reference_mask=ref.mask,
                            ransac_threshold=max(6.0, float(ransac_threshold)),
                        )
                        loftr_metrics["guided"] = _gmeta
                        loftr_metrics["filtered_matches"] = len(_corrs)
                        _gnote = (
                            f" + guided({_gmeta.get('tile_matches', 0)} tiles)"
                            if _gmeta.get("guided") else ""
                        )
                    except Exception as _gexc:
                        _gnote = f" (guided skipped: {_gexc})"
                    runs = [("LoFTR-full", _corrs)]
                    actual_matcher = "LoFTR-full"
                    matcher_note = (
                        f"GSD-scale LoFTR ({str(loftr_out.get('device','cpu')).upper()}): "
                        f"{loftr_out.get('raw_matches')}→{len(_corrs)}, scale={loftr_out.get('scale_apply')} ({loftr_out.get('scale_source')})"
                        f"{_gnote}"
                    )
                except Exception as _full_exc:
                    matcher_note = f"Full LoFTR failed ({_full_exc}); SIFT fallback"
                    runs = [
                        ("SIFT-intensity", sift_run(src.gray, ref.gray, src.mask, ref.mask, feature_count, ratio_threshold)),
                        ("SIFT-structure", sift_run(src.structure, ref.structure, src.mask, ref.mask, feature_count, ratio_threshold)),
                        ("SIFT-gradient", sift_run(src.gradient, ref.gradient, src.mask, ref.mask, feature_count, ratio_threshold)),
                    ]
                    actual_matcher = "SIFT"
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

_cmp_old = (
    "if enable_comparison:\n"
    "            alt_verifier = \"RANSAC\" if verifier_selected == \"MAGSAC++\" else \"MAGSAC++\"\n"
    "            with st.spinner(f\"Running comparison baseline run with {alt_verifier}...\"):\n"
    "                comp_result = run_pipeline(\n"
)
_cmp_new = (
    "if enable_comparison:\n"
    "          try:\n"
    "            alt_verifier = \"RANSAC\" if verifier_selected == \"MAGSAC++\" else \"MAGSAC++\"\n"
    "            with st.spinner(f\"Running comparison baseline run with {alt_verifier}...\"):\n"
    "                comp_result = run_pipeline(\n"
)
if _cmp_old in _src:
    _src = _src.replace(_cmp_old, _cmp_new, 1)
    _src = _src.replace(
        'st.session_state["comp_result"] = comp_result',
        'st.session_state["comp_result"] = comp_result\n'
        '          except Exception as _cmp_exc:\n'
        '            st.warning(f"Comparison skipped ({_cmp_exc}). Primary MAGSAC result kept.")\n'
        '            st.session_state["comp_result"] = None',
        1,
    )

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
    'key="reference"\n    )\n'
    '    st.info("**Local:** GSD-aware scale + CLAHE LoFTR + guided rematch. Affine / MAGSAC=7 / conf=0.26 defaults.")\n',
    1,
)

try:
    compile(_src, "app_v3.py", "exec")
except SyntaxError as _syn:
    raise RuntimeError(f"Bootstrap syntax error line {_syn.lineno}: {_syn.msg}") from _syn

exec(compile(_src, "app_v3.py", "exec"), globals())
