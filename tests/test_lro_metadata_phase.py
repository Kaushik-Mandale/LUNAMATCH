import os

import numpy as np
import pytest

from core.footprint import (
    calculate_overlap_area,
    evaluate_footprint_overlap,
    normalize_longitudes,
)
from core.pds_parser import parse_lro_pds3_label, verify_product_id
from core.product_state import FootprintStatus, MetadataStatus, compute_footprint_status
from core.product_state import PairValidationStatus, compute_pair_validation
from core.scientific_reader import (
    ScientificRasterSpec,
    check_file_size_consistency,
    load_scientific_preview,
    open_scientific_memmap,
    persist_product,
    scientific_raster_spec,
)
from app_v3 import lroc_catalog_metadata_record


def lro_label(product_id="M1438615574LE", pointer=1):
    return f'''PDS_VERSION_ID = PDS3
RECORD_BYTES = 4
^IMAGE = {pointer}
PRODUCT_ID = "{product_id}"
INSTRUMENT_ID = "LROC"
LINES = 4
LINE_SAMPLES = 5
SAMPLE_BITS = 8
SAMPLE_TYPE = UNSIGNED_INTEGER
SCALED_PIXEL_WIDTH = 2.0092 <METERS/PIXEL>
CORNER1_LATITUDE = -89.0
CORNER1_LONGITUDE = 359.0
CORNER2_LATITUDE = -89.0
CORNER2_LONGITUDE = 1.0
CORNER3_LATITUDE = -90.0
CORNER3_LONGITUDE = 1.0
CORNER4_LATITUDE = -90.0
CORNER4_LONGITUDE = 359.0
END
'''


def test_lro_label_parser_builds_explicit_raster_spec():
    metadata = parse_lro_pds3_label(lro_label())
    spec = scientific_raster_spec(metadata)
    assert metadata["metadata_source"] == "PDS3_LBL"
    assert spec.lines == 4
    assert spec.samples == 5
    assert spec.sample_bits == 8
    assert spec.dtype == "uint8"
    assert spec.image_offset == 0
    assert spec.record_bytes == 4


def test_pds3_byte_pointer_is_not_treated_as_record_number():
    metadata = parse_lro_pds3_label(lro_label().replace("^IMAGE = 1", "^IMAGE = 12 <BYTES>"))
    assert metadata["image_pointer_unit"] == "BYTES"
    assert metadata["image_offset"] == 12


def test_lro_product_id_verification():
    assert verify_product_id("M1438615574LE.IMG", "M1438615574LE")["status"] == "VERIFIED"
    result = verify_product_id("M1438615574LE.IMG", "OTHER_PRODUCT")
    assert result["status"] == "MISMATCH"
    assert "different products" in result["message"]


def test_lroc_catalog_metadata_record_for_m1438615574le():
    record = lroc_catalog_metadata_record("M1438615574LE.IMG")
    assert record["product_id"] == "M1438615574LE"
    assert record["sensor_type"] == "LROC_NAC"
    assert record["dimensions"]["lines"] > 0
    assert record["dimensions"]["samples"] > 0
    assert record["gsd_m_per_pixel"] == 2.00920205325772
    assert record["gsd_provenance"] == "LROC_PRODUCT_RECORD"
    assert record["start_time"]
    assert record["footprint"]["upper_left"]
    assert record.get("metadata_source") == "LROC_CATALOG"
    assert record.get("record_bytes") is None
    assert record.get("image_offset") is None
    assert record.get("sample_type") is None


def test_lroc_catalog_gsd_is_not_replaced_by_derived_value():
    record = lroc_catalog_metadata_record("M1438615574LE.IMG")
    assert record["catalog_gsd_m_per_pixel"] == 2.00920205325772
    assert record["gsd_m_per_pixel"] == record["catalog_gsd_m_per_pixel"]
    assert record.get("derived_gsd_m_per_pixel") is None
    assert scientific_raster_spec(record).is_decodable() is False
    assert record["gsd_m_per_pixel"] / 0.24 == pytest.approx(8.371675221907166)


def test_file_size_consistency_states():
    spec = ScientificRasterSpec(lines=4, samples=5, sample_bits=8, dtype="uint8", image_offset=4)
    assert check_file_size_consistency(24, spec)["status"] == "VERIFIED"
    assert check_file_size_consistency(28, spec)["status"] == "PARTIAL"
    assert check_file_size_consistency(10, spec)["status"] == "MISMATCH"


def test_memmap_reader_and_preview_do_not_require_pil():
    raw = bytes(range(20))
    path = persist_product(raw, suffix=".IMG")
    try:
        spec = ScientificRasterSpec(lines=4, samples=5, sample_bits=8, dtype="uint8", image_offset=0)
        mapped = open_scientific_memmap(path, spec)
        assert isinstance(mapped, np.memmap)
        assert mapped.shape == (4, 5)
        assert int(mapped[3, 4]) == 19
        preview, mask, info = load_scientific_preview(path, "M1438615574LE.IMG", {"raster_spec": spec.to_dict()}, max_side=3)
        assert preview.shape == (2, 3)
        assert mask.shape == preview.shape
        assert info["label"] == "Visualization Preview"
        del mapped
    finally:
        os.unlink(path)


def test_missing_lro_metadata_remains_pending():
    assert compute_footprint_status({}, MetadataStatus.NOT_PROVIDED) == FootprintStatus.PENDING


def test_longitude_wrap_polygon_overlap_is_calculated():
    source = {"footprint": {
        "upper_left": [-89.0, 359.0], "upper_right": [-89.0, 1.0],
        "lower_right": [-90.0, 1.0], "lower_left": [-90.0, 359.0],
    }, "projection": "Polar stereographic"}
    reference = {"footprint": {
        "upper_left": [-89.2, 0.0], "upper_right": [-89.2, 2.0],
        "lower_right": [-89.8, 2.0], "lower_left": [-89.8, 0.0],
    }, "projection": "Polar stereographic"}
    result = evaluate_footprint_overlap(source, reference)
    assert result["status"] == "validated"
    assert result["has_overlap"] is True
    assert result["overlap_area"] > 0
    assert result["overlap_percentage_source"] > 0


def test_missing_reference_footprint_is_not_evaluated():
    result = evaluate_footprint_overlap(source_meta={"footprint": {}}, reference_meta={})
    assert result["status"] == "pending"
    assert result["has_overlap"] is None
    assert result["overlap_area_sq_deg"] == 0.0


def test_expected_file_structure_and_read_preview_aliases():
    spec = ScientificRasterSpec(lines=4, samples=5, sample_bits=8, dtype="uint8", image_offset=0, record_bytes=4)
    structure = spec.expected_file_structure()
    assert structure["expected_raster_bytes"] == 20
    assert structure["status"] == "VERIFIED"
    assert hasattr(spec, "read_preview")


def test_longitude_normalization_and_overlap_area_helpers():
    wrapped = normalize_longitudes([359.0, 0.0, 1.0, 359.5])
    assert wrapped[0] == 359.0
    assert wrapped[1] == 0.0
    polygon = [(0.0, 359.0), (0.0, 1.0), (1.0, 1.0), (1.0, 359.0)]
    assert calculate_overlap_area(polygon) > 0.0


def test_overlap_below_threshold_is_not_pair_validated():
    source = {"footprint": {
        "upper_left": [0.0, 0.0], "upper_right": [0.0, 10.0],
        "lower_right": [10.0, 10.0], "lower_left": [10.0, 0.0],
    }, "projection": "Equirectangular"}
    reference = {"footprint": {
        "upper_left": [0.0, 9.998], "upper_right": [0.0, 20.0],
        "lower_right": [10.0, 20.0], "lower_left": [10.0, 9.998],
    }, "projection": "Equirectangular"}
    result = evaluate_footprint_overlap(source, reference, min_overlap_percentage=0.1)
    assert result["status"] == "insufficient_overlap"
    assert result["gate_passed"] is False
    assert result["overlap_percentage_smaller"] < 0.1
    status, _ = compute_pair_validation(
        MetadataStatus.CATALOG_METADATA,
        MetadataStatus.CATALOG_METADATA,
        FootprintStatus.AVAILABLE,
        FootprintStatus.AVAILABLE,
        result,
    )
    assert status == PairValidationStatus.INSUFFICIENT_OVERLAP
