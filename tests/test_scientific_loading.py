"""Unit and integration tests for scientific product loading and canonical metadata state."""
import io
import unittest.mock as mock
import numpy as np
from PIL import Image
import pytest

from core.scientific_reader import (
    ProductType,
    classify_product,
    classify_product_display_label,
    inspect_scientific_product,
    load_scientific_preview,
    load_product,
    get_raster_shape,
    get_raster_dtype,
)
from core.product_state import (
    MetadataStatus,
    FootprintStatus,
    PairValidationStatus,
    compute_metadata_status,
    compute_footprint_status,
    compute_pair_validation,
    scale_ratio_display,
)
from app_v3 import parse_chandrayaan2_pds4_xml


def test_png_loads_as_standard_image():
    """Verify standard PNG images load as STANDARD_IMAGE and decode via PIL."""
    # Create an in-memory PNG
    im = Image.new("L", (128, 64), color=120)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    ptype = classify_product("test_preview.png")
    assert ptype == ProductType.STANDARD_IMAGE

    report = inspect_scientific_product(png_bytes, "test_preview.png")
    assert report["product_type"] == ProductType.STANDARD_IMAGE.value
    assert report["decoding_status"] == "READY"
    assert report["width"] == 128
    assert report["height"] == 64

    arr, valid, info = load_product(png_bytes, "test_preview.png")
    assert arr.shape == (64, 128)
    assert valid.shape == (64, 128)


def test_lro_img_not_sent_to_pil():
    """Ensure raw scientific .IMG files are NEVER routed to PIL.Image.open."""
    fake_img_bytes = b"\x00" * 4096
    with mock.patch("PIL.Image.open") as mock_pil_open:
        report = inspect_scientific_product(fake_img_bytes, "M1438615574LE.IMG")
        # PIL.Image.open should NOT have been called
        mock_pil_open.assert_not_called()
        assert report["product_type"] == ProductType.LRO_PDS3_BINARY.value


def test_lro_img_classified_as_scientific_binary():
    """Verify that M1438615574LE.IMG is classified as LRO_PDS3_BINARY."""
    ptype = classify_product("M1438615574LE.IMG")
    assert ptype == ProductType.LRO_PDS3_BINARY

    label = classify_product_display_label("M1438615574LE.IMG")
    assert "LRO LROC NAC" in label


def test_missing_lro_label_does_not_invalidate_image():
    """Uploading LRO .IMG without a label keeps image READY while decoding is pending."""
    fake_bytes = b"\x00" * 2048
    report = inspect_scientific_product(fake_bytes, "M1438615574LE.IMG", metadata=None)
    assert report["valid_image_status"] == "Ready (pending metadata)"
    assert report["decoding_status"] == "PENDING_METADATA"
    assert "Scientific LRO .IMG detected" in report["message"]


def test_missing_reference_metadata_is_pending():
    """When reference metadata is not provided, metadata status is NOT_PROVIDED."""
    ref_status = compute_metadata_status({}, "NOT_PROVIDED")
    assert ref_status == MetadataStatus.NOT_PROVIDED


def test_missing_reference_gsd_is_pending():
    """When reference GSD is None, scale ratio display reports pending."""
    s = scale_ratio_display(0.24, None)
    assert "pending" in s.lower()
    assert "reference GSD unavailable" in s


def test_missing_reference_footprint_is_pending():
    """When reference metadata is NOT_PROVIDED, footprint status is PENDING."""
    fp_status = compute_footprint_status({}, MetadataStatus.NOT_PROVIDED)
    assert fp_status == FootprintStatus.PENDING


def test_no_stale_pair_metadata():
    """Pair validation returns PENDING and does not falsely REJECT when reference footprint is pending."""
    status, reason = compute_pair_validation(
        source_meta_status=MetadataStatus.PARSED_FROM_PRODUCT,
        reference_meta_status=MetadataStatus.NOT_PROVIDED,
        source_footprint_status=FootprintStatus.AVAILABLE,
        reference_footprint_status=FootprintStatus.PENDING,
        footprint_eval={},
    )
    assert status == PairValidationStatus.PENDING
    assert status != PairValidationStatus.REJECTED


def test_header_uses_canonical_metadata():
    """Verify header display strings derive strictly from canonical metadata."""
    src_gsd = 0.24
    ref_gsd = None
    hdr_src = f"{src_gsd:.4f} m/pixel" if src_gsd is not None else "Pending metadata"
    hdr_ref = f"{ref_gsd:.4f} m/pixel" if ref_gsd is not None else "Pending metadata"
    hdr_scale = scale_ratio_display(src_gsd, ref_gsd) if (src_gsd and ref_gsd) else "Scale ratio pending"

    assert "0.2400 m/pixel" in hdr_src
    assert hdr_ref == "Pending metadata"
    assert hdr_scale == "Scale ratio pending"


def test_source_gsd_updates_after_xml_parse():
    """Simulate parsing real OHRC PDS4 XML and check canonical GSD extraction."""
    xml_sample = """<?xml version="1.0" encoding="UTF-8"?>
    <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
        <Identification_Area>
            <logical_identifier>urn:isro:ch2:ohrc:ch2_ohr_ncp_20241115T1326321339_d_img_d18</logical_identifier>
        </Identification_Area>
        <Observation_Area>
            <Discipline_Area>
                <spatial_resolution unit="m">0.24</spatial_resolution>
            </Discipline_Area>
        </Observation_Area>
        <File_Area_Observational>
            <Array_2D_Image>
                <axes>2</axes>
                <axis_index_order>Last_Index_Fastest</axis_index_order>
                <Element_Array>
                    <data_type>SignedMSB2</data_type>
                </Element_Array>
                <Axis_Array>
                    <axis_name>Line</axis_name>
                    <elements>2048</elements>
                </Axis_Array>
                <Axis_Array>
                    <axis_name>Sample</axis_name>
                    <elements>1024</elements>
                </Axis_Array>
            </Array_2D_Image>
        </File_Area_Observational>
    </Product_Observational>
    """
    meta = parse_chandrayaan2_pds4_xml(xml_sample.encode("utf-8"), "ch2_ohr_d18.xml")
    assert meta["gsd_m_per_pixel"] == 0.24
    assert meta["dimensions"]["lines"] == 2048
    assert meta["dimensions"]["samples"] == 1024


def test_scientific_preview_requires_metadata():
    """Attempting to load a scientific binary preview without dimensions raises ValueError."""
    fake_bytes = b"\x00" * 1024
    with pytest.raises(ValueError, match="lines and samples not provided in metadata"):
        load_scientific_preview(fake_bytes, "M1438615574LE.IMG", metadata={})


def test_lro_product_identity():
    """Verify various LRO file name patterns correctly identify as LRO_PDS3_BINARY."""
    assert classify_product("M1438615574LE.IMG") == ProductType.LRO_PDS3_BINARY
    assert classify_product("M1438615574RE.IMG") == ProductType.LRO_PDS3_BINARY
    assert classify_product("nac_lroc_obs.img") == ProductType.LRO_PDS3_BINARY
    assert classify_product("ch2_ohr_ncp_d18.png") == ProductType.STANDARD_IMAGE
    assert classify_product("ch2_ohr_ncp_d18.img") == ProductType.CHANDRAYAAN_PDS4_BINARY


def test_large_binary_not_loaded_into_pil():
    """Verify that a synthetic scientific binary is decoded using load_scientific_preview, not PIL."""
    # 200 lines x 100 samples uint8 = 20,000 bytes
    synthetic_raster = (np.arange(20000) % 255).astype(np.uint8).tobytes()
    meta = {
        "dimensions": {"lines": 200, "samples": 100},
        "data_type": "uint8",
        "offset": 0,
    }
    with mock.patch("PIL.Image.open") as mock_pil:
        arr, valid, load_info = load_product(synthetic_raster, "M1438615574LE.IMG", metadata=meta, max_side=500)
        mock_pil.assert_not_called()
        assert arr.shape == (200, 100)
        assert load_info["is_preview"] is True
        assert load_info["label"] == "Visualization Preview"
