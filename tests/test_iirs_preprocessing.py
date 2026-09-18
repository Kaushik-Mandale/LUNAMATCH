"""
Unit tests for iirs_preprocessing.py module in LunaMatch V3.
Verifies hyperspectral cube ingestion, metadata inspection, invalid pixel masking,
normalization, PCA computation, composite/selected-band/gradient 2D generation,
and resolution scale ratio calculation.
"""

import io
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import iirs_preprocessing as ip


@pytest.fixture
def synthetic_iirs_cube():
    """Create a synthetic 16-band IIRS hyperspectral cube with craters and NoData borders."""
    np.random.seed(42)
    bands = 16
    h, w = 120, 100
    cube = np.zeros((bands, h, w), dtype=np.float32)

    # Base lunar crater spatial pattern
    y, x = np.ogrid[:h, :w]
    crater_center = (60, 50)
    dist = np.sqrt((y - crater_center[0]) ** 2 + (x - crater_center[1]) ** 2)
    base_crater = np.sin(dist / 10.0) * 50.0 + 100.0

    # Fill bands with spectral variations
    for b in range(bands):
        spectral_factor = 1.0 + 0.05 * b + 0.1 * np.sin(b / 2.0)
        noise = np.random.normal(0, 2.0, (h, w)).astype(np.float32)
        cube[b] = base_crater * spectral_factor + noise

    # Add invalid / NoData borders & sporadic NaNs
    cube[:, :5, :] = -9999.0
    cube[:, -5:, :] = np.nan
    cube[:, :, :5] = 1e7  # uncalibrated extreme
    return cube


def test_inspect_iirs_metadata_geotiff(tmp_path):
    """Test inspect_iirs_metadata on a multi-band GeoTIFF with band tags."""
    tif_path = tmp_path / "test_iirs.tif"
    data = np.ones((8, 64, 64), dtype=np.float32) * 128.0
    transform = from_origin(0, 0, 80.0, 80.0)

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=8,
        dtype="float32",
        transform=transform,
    ) as dst:
        dst.write(data)
        for b_idx in range(1, 9):
            dst.update_tags(b_idx, WAVELENGTH=str(800 + b_idx * 100))

    meta = ip.inspect_iirs_metadata(tif_path)
    assert meta["sensor_type"] == "IIRS"
    assert meta["bands"] == 8
    assert meta["lines"] == 64
    assert meta["samples"] == 64
    assert len(meta["wavelengths_nm"]) == 8
    assert meta["wavelength_range_nm"] == (900.0, 1600.0)
    assert meta["valid"] is True


def test_inspect_iirs_metadata_single_band_warning():
    """Inspect single-band image and verify appropriate warning is recorded."""
    meta = ip.inspect_iirs_metadata(b"dummy", name="single_band.png")
    assert any("warning" in str(w).lower() or "spectral" in str(w).lower() for w in meta["warnings"])


def test_remove_invalid_pixels(synthetic_iirs_cube):
    """Verify NoData (-9999), NaNs, and extreme values are correctly masked."""
    cleaned, valid_mask = ip.remove_invalid_pixels(synthetic_iirs_cube, no_data_value=-9999.0)

    assert valid_mask.ndim == 2
    assert valid_mask.shape == synthetic_iirs_cube.shape[1:]
    assert not np.any(valid_mask[:5, :])  # top border was -9999
    assert not np.any(valid_mask[-5:, :])  # bottom border was NaN
    assert np.all(np.isfinite(cleaned))
    assert np.all(valid_mask[20:80, 20:80])  # central crater region should be valid


def test_validate_iirs_cube(synthetic_iirs_cube):
    """Validate spatial, spectral, and numerical report."""
    report = ip.validate_iirs_cube(synthetic_iirs_cube)
    assert report["bands"] == 16
    assert report["lines"] == 120
    assert report["samples"] == 100
    assert report["valid_pixel_percentage"] > 70.0
    assert report["valid"] is True


def test_normalize_iirs(synthetic_iirs_cube):
    """Test percentile and standardization normalization."""
    cleaned, mask = ip.remove_invalid_pixels(synthetic_iirs_cube, no_data_value=-9999.0)

    # Percentile
    norm_p, _ = ip.normalize_iirs(cleaned, mask, method="percentile")
    assert norm_p.shape == synthetic_iirs_cube.shape
    valid_vals = norm_p[0, mask]
    assert np.all(valid_vals >= 0.0) and np.all(valid_vals <= 1.0)

    # Standardization
    norm_s, _ = ip.normalize_iirs(cleaned, mask, method="standardization")
    assert norm_s.shape == synthetic_iirs_cube.shape
    assert abs(np.mean(norm_s[0, mask])) < 1e-4


def test_generate_iirs_pca(synthetic_iirs_cube):
    """Test PCA calculation, PC1/PC2/PC3 generation, and variance ratio."""
    cleaned, mask = ip.remove_invalid_pixels(synthetic_iirs_cube, no_data_value=-9999.0)
    pca_res = ip.generate_iirs_pca(cleaned, mask, n_components=3)

    assert "PC1" in pca_res["components"]
    assert "PC2" in pca_res["components"]
    assert "PC3" in pca_res["components"]
    pc1 = pca_res["components"]["PC1"]
    assert pc1.dtype == np.uint8
    assert pc1.shape == (120, 100)
    assert len(pca_res["explained_variance_ratio"]) == 3
    # First component should capture dominant variance
    assert pca_res["explained_variance_ratio"][0] > pca_res["explained_variance_ratio"][1]


def test_generate_iirs_2d_representation_modes(synthetic_iirs_cube):
    """Test all representation modes: automatic, pca, selected_band, composite, gradient."""
    cleaned, mask = ip.remove_invalid_pixels(synthetic_iirs_cube, no_data_value=-9999.0)

    # Automatic (should choose PCA for 16-band cube)
    rep_auto, diag_auto = ip.generate_iirs_2d_representation(cleaned, mask, method="automatic")
    assert rep_auto.shape == (120, 100)
    assert rep_auto.dtype == np.uint8
    assert "PCA" in diag_auto["method_applied"]

    # PCA PC1
    rep_pc1, diag_pc1 = ip.generate_iirs_2d_representation(cleaned, mask, method="pca", pca_component=1)
    assert rep_pc1.shape == (120, 100)
    assert diag_pc1["method_applied"] == "PCA (PC1)"

    # Selected Band
    rep_band, diag_band = ip.generate_iirs_2d_representation(cleaned, mask, method="selected_band", selected_band=5)
    assert rep_band.shape == (120, 100)
    assert "Band 5" in diag_band["method_applied"]

    # Composite
    rep_comp, diag_comp = ip.generate_iirs_2d_representation(cleaned, mask, method="composite", composite_bands=(2, 8, 14))
    assert rep_comp.shape == (120, 100)
    assert "Composite" in diag_comp["method_applied"]

    # Gradient
    rep_grad, diag_grad = ip.generate_iirs_2d_representation(cleaned, mask, method="gradient")
    assert rep_grad.shape == (120, 100)
    assert "Gradient" in diag_grad["method_applied"]


def test_compute_resolution_scale_ratio():
    """Verify scale ratio calculation between IIRS (~80m), TMC-2 (~5m), and OHRC (~0.25m)."""
    # IIRS vs TMC-2
    res_tmc = ip.compute_resolution_scale_ratio(80.0, 5.0)
    assert res_tmc["scale_ratio"] == 16.0
    assert res_tmc["coarser_sensor"] == "source"
    assert res_tmc["recommended_downsample_factor"] == 16

    # IIRS vs OHRC
    res_ohrc = ip.compute_resolution_scale_ratio(80.0, 0.25)
    assert res_ohrc["scale_ratio"] == 320.0
    assert res_ohrc["coarser_sensor"] == "source"
    assert res_ohrc["recommended_downsample_factor"] == 320

    # IIRS vs IIRS
    res_same = ip.compute_resolution_scale_ratio(80.0, 80.0)
    assert res_same["scale_ratio"] == 1.0
    assert res_same["coarser_sensor"] == "equal"
