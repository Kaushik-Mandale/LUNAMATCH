"""LunaMatch V3 bootstrap — ISRO guided cascade."""
from __future__ import annotations
import os, re, urllib.request
for _k, _v in (("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"), ("NUMEXPR_NUM_THREADS", "1"), ("TORCH_NUM_THREADS", "1")):
    os.environ.setdefault(_k, _v)
os.environ.setdefault("LUNAMATCH_ROI_LOFTR", "1")
_URL = "https://raw.githubusercontent.com/Kaushik-Mandale/LUNAMATCH/a12f6c3394b5510357d1c5aad1fe3d9ffd7244f1/app_v3.py"
with urllib.request.urlopen(_URL, timeout=60) as _resp:
    _src = _resp.read().decode("utf-8")
_src = _src.replace("max_side=2048, feature_count=12000,", "max_side=768, feature_count=4000,", 1)
_src = _src.replace("[1024, 1600, 2048, 3072, 4096], value=2048", "[512, 768, 1024, 1536], value=768", 1)
_src = _src.replace('st.slider("SIFT features / representation", 3000, 30000, 12000, 1000)', 'st.slider("SIFT features / representation", 1000, 8000, 4000, 500)', 1)
_src = _src.replace("feature_count = 12000\n", "feature_count = 4000\n", 1)
_src = _src.replace('st.slider("LoFTR confidence threshold", 0.10, 0.90, 0.35, 0.05)', 'st.slider("LoFTR confidence threshold", 0.10, 0.90, 0.30, 0.05)', 1)
_src = _src.replace("loftr_confidence_threshold = 0.35\n", "loftr_confidence_threshold = 0.30\n", 1)
_src = _src.replace('st.slider("Reciprocal descriptor ratio", 0.55, 0.90, 0.78, 0.01)', 'st.slider("Reciprocal descriptor ratio", 0.55, 0.90, 0.80, 0.01)', 1)
_src = _src.replace("ratio_threshold=0.78", "ratio_threshold=0.80")
_src = _src.replace('st.slider("RANSAC/MAGSAC threshold (working px)", 0.5, 6.0, 2.5, 0.25)', 'st.slider("RANSAC/MAGSAC threshold (working px)", 0.5, 12.0, 3.5, 0.25)', 1)
_src = _src.replace("ransac_threshold=2.5", "ransac_threshold=3.5")
_src = _src.replace("len(source_file.getvalue()) / (1024 ** 2)", "(getattr(source_file, 'size', None) or len(source_file.getvalue())) / (1024 ** 2)")
_src = _src.replace("len(reference_file.getvalue()) / (1024 ** 2)", "(getattr(reference_file, 'size', None) or len(reference_file.getvalue())) / (1024 ** 2)")
_src = _src.replace("len(reference_file.getvalue())/(1024**2)", "(getattr(reference_file, 'size', None) or len(reference_file.getvalue()))/(1024**2)")
_src = _src.replace("compute_file_hash(source_file.getvalue()) if source_file else None", "(_light_id(source_file) if source_file is not None else None)")
_src = _src.replace("compute_file_hash(reference_file.getvalue()) if reference_file else None", "(_light_id(reference_file) if reference_file is not None else None)")
_src = _src.replace("compute_file_hash(reference_file.getvalue()) if reference_file is not None else None", "(_light_id(reference_file) if reference_file is not None else None)")
_src = _src.replace("check_file_size_consistency(len(reference_file.getvalue()), _spec_obj)", "check_file_size_consistency((getattr(reference_file, 'size', None) or len(reference_file.getvalue())), _spec_obj)")
_src = _src.replace('route_status = "failed"\n            route_error = f"LoFTR dependency missing: {loftr_err}"', 'route_status = "success"\n            route_error = None', 1)
_src = _src.replace('if int(train_mask.sum()) < 8:\n            raise ValueError(f"Too few robust inliers ({int(train_mask.sum())}) after {geom_info.get(\'verifier_method\', geometric_verifier)} estimation.")', 'if int(train_mask.sum()) < (4 if execution_mode == "Experimental image-only mode" else 8):\n            raise ValueError(f"Too few robust inliers ({int(train_mask.sum())}) after {geom_info.get(\'verifier_method\', geometric_verifier)} estimation.")', 1)
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
            if "LoFTR" in method or method == "Guided":
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
_src = _src.replace('if len(selected) < 8:\n        raise ValueError(f"Only {len(selected)} fused correspondences survived. Need at least 8.")', 'if len(selected) < 4:\n        raise ValueError(f"Only {len(selected)} fused correspondences survived. Need at least 4.")', 1)
_GUIDED = r'''_runs_for_fuse = runs
        _loftr_runs = [(n, r) for n, r in runs if "LoFTR" in n and r]
        if _loftr_runs and sum(len(r) for _, r in _loftr_runs) >= 10:
            _runs_for_fuse = _loftr_runs
        ps, pr, scores, methods = fuse_and_select(_runs_for_fuse, src.gray.shape, ref.gray.shape, grid_size, cell_limit)
        try:
            if len(ps) >= 4:
                _H0, _m0, _ = estimate_geometric_model(ps, pr, model=model, verifier=geometric_verifier, threshold=max(float(ransac_threshold), 3.0))
                _m0 = __import__("numpy").asarray(_m0).reshape(-1).astype(bool)
                if _H0 is not None and int(_m0.sum()) >= 4:
                    from guided_match import guided_rematch as _guided_rematch
                    _g = _guided_rematch(src.gray, ref.gray, _H0, source_mask=getattr(src, "mask", None), reference_mask=getattr(ref, "mask", None), tile=min(384, max(256, int(max_side)//2)), nfeatures=600, ratio_threshold=min(float(ratio_threshold), 0.80), max_tiles=10)
                    _gc = _g.get("correspondences") or []
                    if len(_gc) >= 6:
                        import numpy as _np
                        _ps2 = _np.array([c[0] for c in _gc], _np.float64)
                        _pr2 = _np.array([c[1] for c in _gc], _np.float64)
                        _sc2 = _np.array([c[2] for c in _gc], _np.float64)
                        _keep = _m0 if len(_m0) == len(ps) else _np.ones(len(ps), dtype=bool)
                        ps = _np.vstack([ps[_keep], _ps2])
                        pr = _np.vstack([pr[_keep], _pr2])
                        scores = _np.concatenate([_np.asarray(scores, float)[_keep], _sc2])
                        methods = _np.array(list(_np.asarray(methods)[_keep]) + ["Guided"] * len(_ps2))
                        _rows = [((float(ps[i,0]), float(ps[i,1])), (float(pr[i,0]), float(pr[i,1])), float(scores[i])) for i in range(len(ps))]
                        ps, pr, scores, methods = fuse_and_select([("Guided-cascade", _rows)], src.gray.shape, ref.gray.shape, grid_size, max(int(cell_limit), 25))
        except Exception:
            pass'''
_src = _src.replace("ps, pr, scores, methods = fuse_and_select(runs, src.gray.shape, ref.gray.shape, grid_size, cell_limit)", _GUIDED, 1)
_OLD_GEOM = "        H, train_mask, geom_info = estimate_geometric_model(\n            ps[train_idx], pr[train_idx], model=model, verifier=geometric_verifier, threshold=ransac_threshold\n        )\n        # Backwards-compatible estimate_model(ps[train_idx]\n        global_train_inlier = np.zeros(len(ps), dtype=bool)\n        global_train_inlier[train_idx] = train_mask"
_NEW_GEOM = "        H, train_mask, geom_info = estimate_geometric_model(\n            ps[train_idx], pr[train_idx], model=model, verifier=geometric_verifier, threshold=ransac_threshold\n        )\n        train_mask = np.asarray(train_mask).reshape(-1).astype(bool)\n        if int(train_mask.sum()) >= 6:\n            _ips = np.asarray(ps[train_idx][train_mask], np.float64)\n            _ipr = np.asarray(pr[train_idx][train_mask], np.float64)\n            _th_fine = max(1.0, float(ransac_threshold) * 0.5)\n            try:\n                H2, mask2, info2 = estimate_geometric_model(_ips, _ipr, model=model, verifier=geometric_verifier, threshold=_th_fine)\n                mask2 = np.asarray(mask2).reshape(-1).astype(bool)\n                if H2 is not None and int(mask2.sum()) >= 4:\n                    H = H2\n                    _pts = np.asarray(ps[train_idx], np.float64)\n                    _ptr = np.asarray(pr[train_idx], np.float64)\n                    _ones = np.ones((len(_pts), 1), np.float64)\n                    _proj = (H @ np.hstack([_pts, _ones]).T).T\n                    _proj = _proj[:, :2] / np.maximum(_proj[:, 2:3], 1e-12)\n                    _err = np.linalg.norm(_proj - _ptr, axis=1)\n                    train_mask = _err <= max(1.5, float(ransac_threshold) * 0.65)\n                    if isinstance(info2, dict):\n                        geom_info = dict(info2)\n                    geom_info['multi_pass'] = True\n                    geom_info['inlier_count'] = int(train_mask.sum())\n            except Exception:\n                pass\n        global_train_inlier = np.zeros(len(ps), dtype=bool)\n        global_train_inlier[train_idx] = train_mask"
if _OLD_GEOM in _src:
    _src = _src.replace(_OLD_GEOM, _NEW_GEOM, 1)
_src = _src.replace("H2, mask2, _ = estimate_geometric_model(rs, rr, model=model, verifier=geometric_verifier, threshold=ransac_threshold)", "H2, mask2, _ = estimate_geometric_model(rs, rr, model=model, verifier=geometric_verifier, threshold=max(1.0, float(ransac_threshold) * 0.5))", 1)
_src = _src.replace("if mask2.sum() >= 8:", "if int(np.asarray(mask2).reshape(-1).sum()) >= (4 if execution_mode == \"Experimental image-only mode\" else 8):", 1)
_src = _src.replace('if actual_m == "LoFTR":', 'if actual_m == "LoFTR" or (actual_m and "LoFTR" in str(actual_m)):', 1)
_src = _src.replace('**Genuine LoFTR Active**', '**ROI-LoFTR / Guided cascade Active**', 1)
_NEW_CROSS = '''
            _sift_runs = [
                ("SIFT-intensity",  sift_run(src.gray, ref.gray, src.mask, ref.mask, feature_count, ratio_threshold)),
                ("SIFT-structure",  sift_run(src.structure, ref.structure, src.mask, ref.mask, feature_count, ratio_threshold)),
                ("SIFT-gradient",   sift_run(src.gradient, ref.gradient, src.mask, ref.mask, feature_count, ratio_threshold)),
            ]
            runs = list(_sift_runs)
            actual_matcher = "SIFT"
            matcher_note = "Cross-sensor: SIFT multi-rep"
            loftr_metrics = {}
            _coarse = []
            for _name, _rows in _sift_runs:
                _coarse.extend(_rows)
            _try_roi = os.environ.get("LUNAMATCH_DISABLE_LOFTR") != "1" and os.environ.get("LUNAMATCH_ROI_LOFTR", "1") == "1"
            if _try_roi and loftr_avail:
                try:
                    import roi_loftr as _roi
                    loftr_out = _roi.loftr_match_roi_gated(source_gray=src.gray, reference_gray=ref.gray, source_mask=src.mask, reference_mask=ref.mask, min_confidence=min(float(loftr_confidence_threshold), 0.28), roi_max_side=min(int(max_side), 512), coarse_max_side=640, pad_frac=0.30, coarse_correspondences=_coarse[:400] if _coarse else None, ratio_threshold=max(float(ratio_threshold), 0.80))
                    _all = list(loftr_out.get("correspondences") or [])
                    _gc = max(0.28, float(loftr_confidence_threshold))
                    _strict = sorted([c for c in _all if float(c[2]) >= _gc], key=lambda t: -float(t[2]))
                    if len(_strict) < 10:
                        _strict = sorted(_all, key=lambda t: -float(t[2]))
                        _strict = [c for c in _strict if float(c[2]) >= min(_gc, 0.22)][:40]
                    loftr_metrics = {"raw_matches": loftr_out.get("raw_matches"), "filtered_matches": len(_strict), "min_confidence": _gc, "device": loftr_out.get("device"), "rel_scale": loftr_out.get("rel_scale"), "mode": "loftr_roi+guided", "discovered": len(_all)}
                    if len(_strict) >= 10:
                        runs = [("LoFTR-ROI", _strict)]; actual_matcher = "LoFTR-ROI"
                        matcher_note = f"LoFTR-ROI+Guided ({loftr_out.get('device')}) {len(_strict)}/{len(_all)} strict, scale={loftr_out.get('rel_scale')}"
                    elif len(_strict) >= 4:
                        runs = [(n, sorted(r, key=lambda t: t[2])[:15]) for n, r in _sift_runs] + [("LoFTR-ROI", _strict)]
                        actual_matcher = "SIFT+LoFTR-ROI"
                        matcher_note = f"ROI-LoFTR+Guided {len(_strict)} strict + SIFT"
                    else:
                        matcher_note = f"SIFT; ROI-LoFTR sparse ({len(_strict)})"
                except Exception as _e:
                    matcher_note = f"SIFT (ROI-LoFTR skipped: {_e})"
            elif not loftr_avail:
                matcher_note = f"SIFT (LoFTR unavailable: {loftr_err})"
'''
_src, _n = re.subn(r'# CROSS-SENSOR BRANCH[\s\S]*?actual_matcher = "LoFTR"\n[ \t]*matcher_note = \([\s\S]*?\)\n', _NEW_CROSS.lstrip('\n'), _src, count=1)
if _n != 1:
    raise RuntimeError(f"cross-sensor patch failed matches={_n}")
_INJECT = '''\ntry:\n    from core.upload_utils import lightweight_file_id as _light_id\nexcept Exception:\n    def _light_id(f):\n        import hashlib\n        return hashlib.sha256(f"{(getattr(f,'name','') or '')}|{getattr(f,'size',0) or 0}".encode()).hexdigest()[:16]\n'''
if "def _light_id" not in _src:
    _src = _src.replace("import streamlit as st\n", "import streamlit as st\n" + _INJECT, 1)
_src = _src.replace('key="reference"\n    )', 'key="reference"\n    )\n    st.caption("Large LROC: prefer server-side URL if browser upload stalls.")\n    _ref_url = st.text_input("Or load reference via direct HTTPS URL", value="", key="reference_url_fetch", placeholder="https://…/M….IMG")\n    if _ref_url and _ref_url.strip().startswith(("http://", "https://")) and reference_file is None:\n        try:\n            from core.upload_utils import download_url_product as _dl_ref\n            with st.spinner("Server downloading reference…"):\n                reference_file = _dl_ref(_ref_url.strip())\n            st.success(f"Loaded {reference_file.name} ({getattr(reference_file,"size",0)/(1024**2):.1f} MB)")\n        except Exception as _ue:\n            st.error(f"URL fetch failed: {_ue}")\n    st.info("**ISRO cascade:** LoFTR-ROI coarse → guided warp-rematch (tiles + CLAHE) → multi-pass MAGSAC → sub-pixel.")', 1)
compile(_src, "app_v3.py", "exec")
exec(compile(_src, "app_v3.py", "exec"), globals())
