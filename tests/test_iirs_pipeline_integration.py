"""
Integration tests for IIRS pipeline support in LunaMatch V3.
Tests cover:
- Sensor auto-detection (filename and metadata)
- _canonical_sensor token normalisation
- IIRS ↔ IIRS, IIRS ↔ OHRC, IIRS ↔ TMC-2 routing
- Resolution scale ratio calculation
- Geographic footprint gating
- IIRS preprocessing diagnostics structure
- load_iirs_working_image (WorkingImage output)
"""

from __future__ import annotations

import io
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app_v3 import (
    _canonical_sensor,
    classify_sensor_path,
    detect_sensor_from_filename,
    geographic_overlap,
    load_iirs_working_image,
)
import iirs_preprocessing as ip


# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────

def _make_multiband_tiff(bands: int = 16, h: int = 80, w: int = 64) -> bytes:
    """Create an in-memory multi-band GeoTIFF that simulates an IIRS cube."""
    np.random.seed(7)
    data = np.random.uniform(50, 500, (bands, h, w)).astype(np.float32)
    transform = from_origin(0, 0, 80.0, 80.0)
    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=bands,
        dtype="float32",
        transform=transform,
    ) as dst:
        dst.write(data)
    return buf.getvalue()


def _make_single_band_tiff(h: int = 60, w: int = 50) -> bytes:
    """Create a single-band GeoTIFF (simulates OHRC or TMC-2 product)."""
    np.random.seed(3)
    data = np.random.uniform(10, 230, (1, h, w)).astype(np.float32)
    buf = io.BytesIO()
    with rasterio.open(buf, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32") as dst:
        dst.write(data)
    return buf.getvalue()


# ──────────────────────────────────────────────────────
# 1. Sensor token normalisation
# ──────────────────────────────────────────────────────

class TestCanonicalSensor:
    def test_ohrc_unchanged(self):
        assert _canonical_sensor("OHRC") == "OHRC"

    def test_tmc2_normalised(self):
        assert _canonical_sensor("TMC-2") == "TMC"

    def test_tmc_unchanged(self):
        assert _canonical_sensor("TMC") == "TMC"

    def test_iirs_unchanged(self):
        assert _canonical_sensor("IIRS") == "IIRS"

    def test_auto_detect_resolves_unknown(self):
        assert _canonical_sensor("Auto Detect") == "Unknown"

    def test_case_insensitive(self):
        assert _canonical_sensor("tmc-2") == "TMC"
        assert _canonical_sensor("iirs") == "IIRS"
        assert _canonical_sensor("ohrc") == "OHRC"


# ──────────────────────────────────────────────────────
# 2. Filename-based sensor detection
# ──────────────────────────────────────────────────────

class TestDetectSensorFromFilename:
    def test_ohrc_product(self):
        assert detect_sensor_from_filename("ch2_ohr_ncp_20241115T1326321339_d_img_d18.xml") == "OHRC"

    def test_tmc_product(self):
        assert detect_sensor_from_filename("ch2_tmc_ndn_20230612T2218425665_d_oth_n18.tif") == "TMC"

    def test_iirs_product(self):
        assert detect_sensor_from_filename("ch2_iir_ncp_20200101T000000_l2_rd_v1.img") == "IIRS"

    def test_iirs_generic_name(self):
        assert detect_sensor_from_filename("iirs_cube.npy") == "IIRS"

    def test_unknown_filename(self):
        assert detect_sensor_from_filename("random_file.tif") == "Unknown"


# ──────────────────────────────────────────────────────
# 3. Sensor path routing (confirm IIRS cases)
# ──────────────────────────────────────────────────────

class TestIIRSRouting:
    def test_iirs_iirs_same(self):
        assert classify_sensor_path("IIRS", "IIRS") == "same"

    def test_iirs_ohrc_different(self):
        assert classify_sensor_path("IIRS", "OHRC") == "different"

    def test_ohrc_iirs_different(self):
        assert classify_sensor_path("OHRC", "IIRS") == "different"

    def test_iirs_tmc_different(self):
        assert classify_sensor_path("IIRS", "TMC") == "different"

    def test_tmc_iirs_different(self):
        assert classify_sensor_path("TMC", "IIRS") == "different"


# ──────────────────────────────────────────────────────
# 4. Resolution scale ratio
# ──────────────────────────────────────────────────────

class TestResolutionScaleRatio:
    def test_iirs_vs_tmc2(self):
        res = ip.compute_resolution_scale_ratio(80.0, 5.0)
        assert res["scale_ratio"] == 16.0
        assert res["coarser_sensor"] == "source"

    def test_iirs_vs_ohrc(self):
        res = ip.compute_resolution_scale_ratio(80.0, 0.25)
        assert res["scale_ratio"] == 320.0
        assert res["coarser_sensor"] == "source"

    def test_same_resolution(self):
        res = ip.compute_resolution_scale_ratio(80.0, 80.0)
        assert res["scale_ratio"] == 1.0
        assert res["coarser_sensor"] == "equal"

    def test_reference_coarser(self):
        res = ip.compute_resolution_scale_ratio(5.0, 80.0)
        assert res["coarser_sensor"] == "reference"


# ──────────────────────────────────────────────────────
# 5. Geographic footprint overlap gating
# ──────────────────────────────────────────────────────

class TestFootprintOverlapGating:
    def _meta(self, ul, ur, ll, lr):
        return {"footprint": {"upper_left": ul, "upper_right": ur, "lower_left": ll, "lower_right": lr}}

    def test_disjoint_footprints_rejected(self):
        src = self._meta([-89.0, 10.0], [-89.0, 11.0], [-90.0, 10.0], [-90.0, 11.0])
        ref = self._meta([85.0, 10.0], [85.0, 11.0], [84.0, 10.0], [84.0, 11.0])
        result = geographic_overlap(src, ref)
        assert result["available"] is True
        assert result["overlap"] is False

    def test_overlapping_footprints_accepted(self):
        src = self._meta([-89.0, 10.0], [-89.0, 11.0], [-90.0, 10.0], [-90.0, 11.0])
        ref = self._meta([-89.5, 10.5], [-89.5, 11.5], [-90.5, 10.5], [-90.5, 11.5])
        result = geographic_overlap(src, ref)
        assert result["available"] is True
        assert result["overlap"] is True

    def test_missing_footprint_unavailable(self):
        result = geographic_overlap({}, {})
        assert result["available"] is False


# ──────────────────────────────────────────────────────
# 6. load_iirs_working_image integration
# ──────────────────────────────────────────────────────

class TestLoadIIRSWorkingImage:
    def test_multiband_tiff_produces_working_image(self):
        tiff_bytes = _make_multiband_tiff(bands=16, h=80, w=64)
        from app_v3 import WorkingImage
        working, diag = load_iirs_working_image(tiff_bytes, "ch2_iir_test.tif", max_side=256)
        assert isinstance(working, WorkingImage)
        assert working.gray.ndim == 2
        assert working.mask.ndim == 2
        assert working.gray.dtype == np.uint8
        assert working.structure.ndim == 2
        assert working.gradient.ndim == 2
        assert working.to_original.shape == (3, 3)

    def test_diag_fields_present(self):
        tiff_bytes = _make_multiband_tiff(bands=8, h=60, w=48)
        _, diag = load_iirs_working_image(tiff_bytes, "iirs_cube.tif", max_side=128)
        assert "cube_bands" in diag
        assert diag["cube_bands"] == 8
        assert "valid_pixel_pct" in diag
        assert 0.0 <= diag["valid_pixel_pct"] <= 100.0
        assert "representation" in diag
        assert "normalization_method" in diag

    def test_single_band_tiff_is_handled(self):
        """Single-band IIRS input should process correctly as selected-band fallback."""
        tiff_bytes = _make_single_band_tiff(h=60, w=48)
        working, diag = load_iirs_working_image(tiff_bytes, "iirs_single.tif", max_side=128)
        assert working.gray.ndim == 2
        assert diag["cube_bands"] == 1

    def test_pca_representation_output(self):
        tiff_bytes = _make_multiband_tiff(bands=16, h=80, w=64)
        working, diag = load_iirs_working_image(
            tiff_bytes, "iirs.tif", max_side=256,
            iirs_representation="pca", pca_component=1
        )
        assert working.gray.ndim == 2
        assert "PC1" in diag.get("representation", "")

    def test_selected_band_representation_output(self):
        tiff_bytes = _make_multiband_tiff(bands=16, h=80, w=64)
        working, diag = load_iirs_working_image(
            tiff_bytes, "iirs.tif", max_side=256,
            iirs_representation="selected_band"
        )
        assert working.gray.ndim == 2

    def test_gradient_representation_output(self):
        tiff_bytes = _make_multiband_tiff(bands=16, h=80, w=64)
        working, diag = load_iirs_working_image(
            tiff_bytes, "iirs.tif", max_side=256,
            iirs_representation="gradient"
        )
        assert working.gray.ndim == 2
        assert "Gradient" in diag.get("representation", "")
