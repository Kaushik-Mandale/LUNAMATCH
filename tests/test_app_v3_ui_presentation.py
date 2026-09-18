from pathlib import Path
import numpy as np
from app_v3 import make_display_preview


def test_make_display_preview_grayscale_and_diagnostics():
    # Test 2D grayscale image
    img = np.random.randint(20, 200, (600, 800), dtype=np.uint8)
    disp, diag = make_display_preview(img, max_dim=400)

    assert disp.ndim == 3
    assert disp.shape[2] == 3
    assert max(disp.shape[:2]) <= 400
    assert diag["dtype"] == "uint8"
    assert diag["shape"] == [600, 800]
    assert diag["min"] >= 0
    assert diag["max"] <= 255


def test_make_display_preview_bgr_to_rgb():
    # Test 3D BGR image (Blue = channel 0, Green = channel 1, Red = channel 2)
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # Blue channel in BGR
    disp, diag = make_display_preview(img, max_dim=300, is_bgr=True)

    # Converted BGR -> RGB: Blue moves to channel 2, channel 0 is Red
    assert disp.shape[2] == 3
    assert disp[0, 0, 0] == 0    # Red in RGB is 0
    assert disp[0, 0, 2] == 255  # Blue in RGB is 255


def test_make_display_preview_none_and_dark_images():
    # Test None image
    disp_none, diag_none = make_display_preview(None)
    assert disp_none.shape == (200, 200, 3)
    assert diag_none["dtype"] == "none"

    # Test dark image display stretch
    dark_img = np.ones((100, 100), dtype=np.uint8) * 5
    disp_dark, diag_dark = make_display_preview(dark_img, max_dim=200)
    assert disp_dark.max() > 5  # Stretched for display visibility


def test_ui_sections_and_tab_elements_in_app_v3():
    app_text = Path("app_v3.py").read_text(encoding="utf-8")

    # Verify top metric cards
    assert "Fused Matches" in app_text
    assert "Training Inliers" in app_text
    assert "Inlier Ratio" in app_text
    assert "Holdout RMSE" in app_text
    assert "Overlap" in app_text
    assert "Spatial Coverage" in app_text

    # Verify tab navigation
    assert '"Overview"' in app_text
    assert '"Correspondences"' in app_text
    assert '"Registration"' in app_text
    assert '"Metrics"' in app_text
    assert '"Report"' in app_text

    # Verify image registration sections
    assert "IMAGE REGISTRATION" in app_text
    assert "SOURCE / MOVING IMAGE" in app_text
    assert "REFERENCE / FIXED IMAGE" in app_text
    assert "REGISTERED IMAGE" in app_text
    assert "Source → Reference" in app_text
    assert "CORRESPONDENCES" in app_text

    # Verify horizontal pipeline trace & collapsed technical details
    assert "Pipeline Stage Trace" in app_text
    assert "TECHNICAL PIPELINE DETAILS ▾" in app_text

    # Verify export buttons
    assert "Registered PNG" in app_text
    assert "Validity mask" in app_text
    assert "Correspondences CSV" in app_text
    assert "Research report JSON" in app_text
