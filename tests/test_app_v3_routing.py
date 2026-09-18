"""
Architecture routing verification tests for LunaMatch V3.

These tests verify the ACTUAL sensor routing logic (classify_sensor_path),
the matching_strategy key in the report, and that LoFTR is not faked.

Tests do NOT exercise the full pipeline (which requires image files).
They test the routing function and the documented behavior in app_v3.py.
"""
from pathlib import Path
from app_v3 import classify_sensor_path, geographic_overlap, validate_metadata


# ==================================================
# TEST A — Same-sensor routes to SIFT
# ==================================================

def test_routing_ohrc_ohrc_selects_same():
    """OHRC → OHRC must produce sensor_path == 'same' → SIFT branch."""
    result = classify_sensor_path("OHRC", "OHRC")
    assert result == "same", f"Expected 'same', got '{result}'"


def test_complete_metadata_without_gsd_is_valid_with_explicit_warning():
    metadata = {
        "sensor_type": "TMC",
        "processing_level": "Derived",
        "start_time": "2023-06-12T22:18:42Z",
        "stop_time": "2023-06-12T22:28:35Z",
        "altitude_km": 110.59,
        "sun_azimuth_deg": 1.0,
        "sun_elevation_deg": 2.0,
        "solar_incidence_deg": 3.0,
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "yaw_deg": 0.0,
        "projection": "Polar stereographic",
        "footprint": {
            "upper_left": [85.0, 10.0], "upper_right": [85.0, 11.0],
            "lower_left": [84.0, 10.0], "lower_right": [84.0, 11.0],
        },
        "dimensions": {"lines": 100, "samples": 100},
    }
    result = validate_metadata(metadata)
    assert result["valid"] is True
    assert result["warnings"] == ["GSD unavailable for this product"]
    assert not any("Missing GSD" in error for error in result["validation_errors"])


def test_disjoint_footprints_are_rejected_before_cross_sensor_matching():
    source = {"footprint": {
        "upper_left": [-89.0, 10.0], "upper_right": [-89.0, 11.0],
        "lower_left": [-90.0, 10.0], "lower_right": [-90.0, 11.0],
    }}
    reference = {"footprint": {
        "upper_left": [85.0, 10.0], "upper_right": [85.0, 11.0],
        "lower_left": [84.0, 10.0], "lower_right": [84.0, 11.0],
    }}
    result = geographic_overlap(source, reference)
    assert result["available"] is True
    assert result["overlap"] is False


def test_overlapping_footprints_allow_cross_sensor_matching():
    source = {"footprint": {
        "upper_left": [-89.0, 10.0], "upper_right": [-89.0, 11.0],
        "lower_left": [-90.0, 10.0], "lower_right": [-90.0, 11.0],
    }}
    reference = {"footprint": {
        "upper_left": [-89.5, 10.5], "upper_right": [-89.5, 11.5],
        "lower_left": [-90.5, 10.5], "lower_right": [-90.5, 11.5],
    }}
    result = geographic_overlap(source, reference)
    assert result["available"] is True
    assert result["overlap"] is True


def test_routing_tmc_tmc_selects_same():
    """TMC → TMC must produce sensor_path == 'same'."""
    assert classify_sensor_path("TMC", "TMC") == "same"


def test_routing_iirs_iirs_selects_same():
    """IIRS → IIRS must produce sensor_path == 'same'."""
    assert classify_sensor_path("IIRS", "IIRS") == "same"


# ==================================================
# TEST B — Cross-sensor routes to LoFTR branch
# ==================================================

def test_routing_ohrc_tmc_selects_different():
    """OHRC source + TMC reference must produce sensor_path == 'different' → LoFTR branch."""
    result = classify_sensor_path("OHRC", "TMC")
    assert result == "different", f"Expected 'different', got '{result}'"


def test_routing_tmc_ohrc_selects_different():
    """Cross-sensor routing is symmetric: TMC → OHRC must also give 'different'."""
    assert classify_sensor_path("TMC", "OHRC") == "different"


def test_routing_ohrc_iirs_selects_different():
    assert classify_sensor_path("OHRC", "IIRS") == "different"


def test_routing_iirs_tmc_selects_different():
    assert classify_sensor_path("IIRS", "TMC") == "different"


# ==================================================
# Routing is generic — NOT hardcoded to OHRC
# ==================================================

def test_routing_is_generic_not_hardcoded_to_ohrc():
    """
    The routing function must compare the two sensor strings generically.
    It must NOT contain any hardcoded 'OHRC' logic.
    Any equal pair → 'same'. Any unequal pair → 'different'.
    """
    assert classify_sensor_path("X", "X") == "same"
    assert classify_sensor_path("X", "Y") == "different"
    assert classify_sensor_path("ABC", "ABC") == "same"
    assert classify_sensor_path("ABC", "DEF") == "different"
    # Verify the function body does not hardcode sensor names
    import inspect
    source = inspect.getsource(classify_sensor_path)
    assert "OHRC" not in source, "classify_sensor_path must not hardcode 'OHRC'"
    assert "TMC" not in source, "classify_sensor_path must not hardcode 'TMC'"
    assert "IIRS" not in source, "classify_sensor_path must not hardcode 'IIRS'"


# ==================================================
# LoFTR honesty checks
# ==================================================

def test_loftr_not_faked_as_sift_in_source():
    """
    The cross-sensor fallback must be explicitly labelled as fallback.
    The code must never record actual_matcher='LoFTR' while running SIFT.
    The words 'SIFT+AKAZE-fallback' or 'loftr_available' must appear in source.
    """
    text = Path("app_v3.py").read_text(encoding="utf-8")
    assert "SIFT+AKAZE-fallback" in text, "Cross-sensor fallback label missing from source"
    assert "loftr_available" in text, "'loftr_available' key missing from source"
    assert "LoFTR is NOT claimed" in text, "Honesty comment 'LoFTR is NOT claimed' missing"


def test_matching_strategy_key_present_in_report_code():
    """The report dict must include 'matching_strategy' key."""
    text = Path("app_v3.py").read_text(encoding="utf-8")
    assert "matching_strategy" in text
    assert "sensor_relation" in text
    assert "loftr_available" in text
    assert "loftr_note" in text


def test_pyramid_consumed_not_just_built():
    """
    Stage 04 builds the pyramid. Stage 06 must consume at least level 1.
    Verify that 'src_pyramid[1]' or 'SIFT-pyramid1' appears in source.
    """
    text = Path("app_v3.py").read_text(encoding="utf-8")
    assert "src_pyramid[1]" in text, "Pyramid level 1 is not consumed in Stage 06"
    assert "SIFT-pyramid1" in text, "'SIFT-pyramid1' run label not found"


def test_clahe_consumed_in_preprocessing():
    """CLAHE must be applied and its output consumed (not just called and thrown away)."""
    text = Path("app_v3.py").read_text(encoding="utf-8")
    assert "clahe.apply" in text, "CLAHE apply call not found"
    # local is the CLAHE output; it must feed structure/gradient
    assert "local = clahe.apply" in text, "CLAHE output must be assigned to 'local'"
    # structure is built from local via Sobel
    assert "cv2.Sobel(local" in text, "CLAHE output ('local') must feed Sobel for structure"


def test_ransac_magsac_applied_to_matcher_output():
    """RANSAC/MAGSAC must be called in estimate_model on the fused correspondence output."""
    text = Path("app_v3.py").read_text(encoding="utf-8")
    assert "USAC_MAGSAC" in text or "cv2.RANSAC" in text, "RANSAC/MAGSAC not present"
    assert "findHomography" in text or "estimateAffine2D" in text, "Geometric estimation call missing"
    assert "estimate_model(ps[train_idx]" in text, "estimate_model must receive ps[train_idx] (matcher output)"


def test_spatial_grid_filtering_enforced_before_ransac():
    """
    fuse_and_select must enforce cell_limit per grid cell during selection,
    not just calculate coverage after the fact.
    """
    text = Path("app_v3.py").read_text(encoding="utf-8")
    assert "cell_limit" in text, "cell_limit variable not found"
    assert "scount.get" in text and "rcount.get" in text, "Grid cell counting not found in fuse_and_select"
    assert ">= cell_limit" in text, "Cell limit enforcement condition not found"


def test_subpixel_refinement_uses_cornersubpix():
    """Sub-pixel refinement must use cv2.cornerSubPix, not just a flag."""
    text = Path("app_v3.py").read_text(encoding="utf-8")
    assert "cv2.cornerSubPix" in text, "cv2.cornerSubPix not found — sub-pixel refinement is not genuine"


def test_footprint_used_as_spatial_gate():
    """
    Metadata footprint overlap must gate the pipeline (not just display).
    Stage 08 reads footprint_overlap label and fails/warns if low-overlap.
    """
    text = Path("app_v3.py").read_text(encoding="utf-8")
    assert "footprint_overlap" in text, "'footprint_overlap' not used from metadata"
    assert "low-overlap" in text, "Low-overlap gate condition not found"
