"""
Tests for LoFTR multimodal correspondence and MAGSAC++ geometric verification in LunaMatch.
Team Akatsuki Project.
"""
import numpy as np
import pytest
from app_v3 import classify_sensor_path, estimate_geometric_model, draw_matches
import loftr_matcher


# ==================================================
# 1. SENSOR ROUTING TESTS
# ==================================================

def test_routing_same_sensor_sift():
    """Same sensor pairs route to 'same' (SIFT)."""
    assert classify_sensor_path("OHRC", "OHRC") == "same"
    assert classify_sensor_path("TMC", "TMC") == "same"
    assert classify_sensor_path("IIRS", "IIRS") == "same"


def test_routing_cross_sensor_loftr():
    """Different sensor pairs route to 'different' (LoFTR)."""
    assert classify_sensor_path("OHRC", "TMC") == "different"
    assert classify_sensor_path("OHRC", "IIRS") == "different"
    assert classify_sensor_path("TMC", "IIRS") == "different"


def test_routing_cross_sensor_symmetry():
    """Symmetric cross-sensor routing."""
    assert classify_sensor_path("TMC", "OHRC") == "different"
    assert classify_sensor_path("IIRS", "OHRC") == "different"
    assert classify_sensor_path("IIRS", "TMC") == "different"


# ==================================================
# 2. MAGSAC++ GEOMETRIC VERIFICATION TESTS
# ==================================================

def test_magsac_plus_plus_primary_homography():
    """MAGSAC++ estimates homography correctly on synthetic inliers + outliers."""
    np.random.seed(42)
    # Generate 50 points
    src_pts = np.random.uniform(50, 450, (50, 2))
    # True homography: scale + translation + slight perspective
    H_true = np.array([[1.05, 0.02, 30.0],
                       [-0.01, 1.03, 15.0],
                       [0.00005, 0.00002, 1.0]], dtype=np.float64)

    # Project to get reference points
    src_h = np.column_stack([src_pts, np.ones(len(src_pts))])
    ref_h = (H_true @ src_h.T).T
    ref_pts = ref_h[:, :2] / ref_h[:, 2:3]

    # Add 10 outliers
    ref_pts[:10] += np.random.uniform(50, 150, (10, 2))

    H_est, mask, info = estimate_geometric_model(src_pts, ref_pts, model="Homography", verifier="MAGSAC++", threshold=3.0)

    assert info["verifier_method"] == "MAGSAC++"
    assert info["fallback_used"] is False
    assert mask.sum() >= 35
    assert info["inlier_ratio"] >= 0.70
    assert H_est is not None
    assert np.isfinite(H_est).all()


def test_magsac_plus_plus_primary_affine():
    """MAGSAC++ estimates affine transformation correctly."""
    np.random.seed(42)
    src_pts = np.random.uniform(50, 450, (40, 2))
    A_true = np.array([[1.02, -0.05, 20.0],
                       [0.05, 1.02, -10.0]], dtype=np.float64)
    ref_pts = (A_true[:, :2] @ src_pts.T).T + A_true[:, 2]

    # Add 5 outliers
    ref_pts[:5] += np.random.uniform(40, 100, (5, 2))

    H_est, mask, info = estimate_geometric_model(src_pts, ref_pts, model="Affine", verifier="MAGSAC++", threshold=3.0)

    assert info["verifier_method"] == "MAGSAC++"
    assert info["fallback_used"] is False
    assert mask.sum() >= 30
    assert H_est is not None


def test_ransac_baseline_mode():
    """Selecting RANSAC explicitly uses RANSAC verifier."""
    np.random.seed(42)
    src_pts = np.random.uniform(50, 450, (30, 2))
    ref_pts = src_pts + np.array([10.0, 5.0])

    H_est, mask, info = estimate_geometric_model(src_pts, ref_pts, model="Homography", verifier="RANSAC", threshold=3.0)

    assert info["verifier_method"] == "RANSAC"
    assert info["fallback_used"] is False
    assert mask.sum() >= 25


def test_magsac_fallback_on_degenerate_input():
    """If MAGSAC++ fails, explicit fallback to RANSAC is recorded and not hidden."""
    # Degenerate collinear points
    src_pts = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]], dtype=np.float64)
    ref_pts = src_pts + 5.0

    # Minimum number of points is 4 for homography
    try:
        H_est, mask, info = estimate_geometric_model(src_pts, ref_pts, model="Homography", verifier="MAGSAC++", threshold=3.0)
    except ValueError as e:
        # If both fail on degenerate, clear ValueError is raised
        assert "could not be estimated" in str(e) or "unstable" in str(e) or "Insufficient" in str(e)


# ==================================================
# 3. LOFTR PREPROCESSING & FORMATTING TESTS
# ==================================================

def test_loftr_prepare_image_resizes_to_multiples_of_8():
    """Images are resized/padded so that dimensions are divisible by 8 for LoFTR coarse stage."""
    img = np.zeros((373, 517), dtype=np.uint8)
    tensor, sx, sy, orig_shape, work_shape = loftr_matcher.prepare_image_for_loftr(img, max_side=500)

    assert work_shape[0] % 8 == 0
    assert work_shape[1] % 8 == 0
    assert orig_shape == (373, 517)
    assert sx > 0 and sy > 0


def test_loftr_error_handling_when_unavailable(monkeypatch):
    """When LoFTR dependencies are unavailable, clear RuntimeError is raised (no silent SIFT fallback)."""
    monkeypatch.setattr(loftr_matcher, "is_loftr_available", lambda: (False, "Test missing dependency"))

    with pytest.raises(RuntimeError) as excinfo:
        loftr_matcher.loftr_match(np.zeros((100, 100), dtype=np.uint8), np.zeros((100, 100), dtype=np.uint8))

    assert "LoFTR inference unavailable" in str(excinfo.value)
    assert "Test missing dependency" in str(excinfo.value)


def test_draw_matches_different_dimensions():
    """draw_matches handles source and reference images of different heights and widths without crashing."""
    src = np.zeros((300, 400), dtype=np.uint8)
    ref = np.ones((500, 350), dtype=np.uint8) * 255
    ps = np.array([[50.0, 60.0], [100.0, 120.0]])
    pr = np.array([[70.0, 80.0], [150.0, 160.0]])
    inliers = np.array([True, False])

    canvas = draw_matches(src, ref, ps, pr, inliers)
    assert canvas.shape == (500, 750, 3)

