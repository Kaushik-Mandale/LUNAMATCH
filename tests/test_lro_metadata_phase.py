import os

import numpy as np
import pytest

from core.footprint import (
    calculate_overlap_area,
    evaluate_footprint_overlap,
    normalize_longitudes,
    to_polar_stereographic,
)
from core.pds_parser import parse_lro_pds3_label, verify_product_id
from core.product_state import FootprintStatus, MetadataStatus, compute_footprint_status
from core.product_state import PairValidationStatus, compute_pair_validation
from core.scientific_reader import (
    LROC_NAC_EDR_PROFILE,
    apply_raster_profile,
    ScientificRasterSpec,
    check_file_size_consistency,
    inspect_scientific_product,
    load_scientific_preview,
    open_scientific_memmap,
    persist_product,
    scientific_raster_spec,
)
from app_v3 import lroc_catalog_metadata_record, _lroc_nac_edr_catalog_raw_record


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
    assert record["dimensions"] == {"lines": 52224, "samples": 5064}
    assert record["gsd_m_per_pixel"] == 2.00920205325772
    assert record["gsd_provenance"] == "LROC_PRODUCT_RECORD"
    assert record["start_time"]
    assert record["footprint"]["upper_left"]
    assert record.get("metadata_source") == "LROC_CATALOG"
    # After apply_raster_profile() the LROC_NAC_EDR_PROFILE binary layout is present:
    assert record.get("record_bytes") == 5064
    assert record.get("sample_type") == "UNSIGNED_BYTE"
    assert record.get("image_offset") == 5064
    assert record.get("format_profile") == "LROC_NAC_EDR_PDS3"
    # Raw catalog record (pre-profile) has None for binary layout fields:
    raw = _lroc_nac_edr_catalog_raw_record("M1438615574LE")
    assert raw.get("record_bytes") is None
    assert raw.get("sample_type") is None
    assert raw.get("image_offset") is None
    assert record["footprint"] == {
        "upper_left": [-88.07, 211.75],
        "upper_right": [-88.26, 197.21],
        "lower_left": [-88.96, 293.28],
        "lower_right": [-89.40, 311.20],
    }


def test_lroc_nac_edr_profile_is_generic_and_populates_raster_contract():
    record = lroc_catalog_metadata_record("M1438615574LE.IMG")
    profiled = apply_raster_profile(record, "M1438615574LE.IMG")
    spec = scientific_raster_spec(profiled)
    assert LROC_NAC_EDR_PROFILE.name == "LROC_NAC_EDR_PDS3"
    assert profiled["format_profile"] == "LROC_NAC_EDR_PDS3"
    assert profiled["raster_spec_provenance"] == "LROC EDR archive format specification"
    assert profiled["data_set_id"] == "LRO-L-LROC-2-EDR-V1.0"
    assert profiled["record_type"] == "FIXED_LENGTH"
    assert profiled["label_records"] == 1
    assert profiled["image_pointer"] == 2
    assert spec.lines == 52224
    assert spec.samples == 5064
    assert spec.sample_bits == 8
    assert spec.dtype == "uint8"
    assert spec.sample_type == "UNSIGNED_BYTE"
    assert spec.record_bytes == 5064
    assert spec.image_offset == 5064


def test_lroc_nac_edr_expected_structure_and_tolerance_states():
    profiled = apply_raster_profile(lroc_catalog_metadata_record("M1438615574LE.IMG"), "M1438615574LE.IMG")
    spec = scientific_raster_spec(profiled)
    expected_payload = 52224 * 5064
    expected_file = 52225 * 5064
    check = check_file_size_consistency(expected_file, spec)
    assert check["status"] == "VERIFIED"
    assert check["expected_raster_bytes"] == expected_payload
    assert check["expected_record_count"] == 52225
    assert check["expected_file_size_from_records"] == expected_file
    assert check_file_size_consistency(expected_file - 1, spec)["status"] == "PARTIAL"
    assert check_file_size_consistency(expected_file - 5064 * 2, spec)["status"] == "MISMATCH"


def test_lroc_catalog_profile_is_ready_without_label_but_binary_sanity_is_separate():
    # Build a realistic LROC NAC EDR metadata dict for a second product.
    # Dimensions come from an imaginary catalog entry; the profile supplies the binary layout.
    generic_meta = {
        "mission": "Lunar Reconnaissance Orbiter",
        "instrument": "LROC NAC",
        "sensor_type": "LROC_NAC",
        "product_id": "M1534362340LE",
        "data_set_id": "LRO-L-LROC-2-EDR-V1.0",
        "product_type": "EDR",
        "metadata_source": "LROC_CATALOG",
        "dimensions": {"lines": 52224, "samples": 5064},
        "raster_spec": {"lines": 52224, "samples": 5064},
        "valid": True,
        "validation_errors": [],
    }
    profiled = apply_raster_profile(generic_meta, "M1534362340LE.IMG")
    assert profiled["format_profile"] == "LROC_NAC_EDR_PDS3"
    assert scientific_raster_spec(profiled).is_decodable() is True
    report = inspect_scientific_product(b"", "M1534362340LE.IMG", profiled)
    assert report["format_profile"] == "LROC_NAC_EDR_PDS3"
    assert report["decoding_status"] in {"PARTIAL", "MISMATCH", "RASTER_DECODE_ERROR"}


def test_lroc_catalog_gsd_is_not_replaced_by_derived_value():
    record = lroc_catalog_metadata_record("M1438615574LE.IMG")
    assert record["catalog_gsd_m_per_pixel"] == 2.00920205325772
    assert record["gsd_m_per_pixel"] == record["catalog_gsd_m_per_pixel"]
    assert record.get("derived_gsd_m_per_pixel") is None
    # The raw pre-profile catalog record is NOT decodable (no binary layout):
    raw = _lroc_nac_edr_catalog_raw_record("M1438615574LE")
    assert scientific_raster_spec(raw).is_decodable() is False
    # The profile-applied record IS decodable (correct behavior):
    assert scientific_raster_spec(record).is_decodable() is True
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
    assert result["intersection_area_km2"] == pytest.approx(result["overlap_area_km2"])


def test_polar_overlap_areas_use_projected_geometry_not_lon_lat_bbox():
    source = {"footprint": {
        "upper_left": [-80.0, 359.0], "upper_right": [-80.0, 1.0],
        "lower_right": [-82.0, 1.0], "lower_left": [-82.0, 359.0],
    }, "projection": "Polar stereographic"}
    reference = {"footprint": {
        "upper_left": [-80.0, 0.0], "upper_right": [-80.0, 2.0],
        "lower_right": [-82.0, 2.0], "lower_left": [-82.0, 0.0],
    }, "projection": "Polar stereographic"}
    result = evaluate_footprint_overlap(source, reference, min_overlap_percentage=0.1)
    projected_source = [to_polar_stereographic(lat, lon) for lat, lon in (
        (-80.0, 359.0), (-80.0, 1.0), (-82.0, 1.0), (-82.0, 359.0)
    )]
    projected_source_area = calculate_overlap_area(projected_source)
    assert result["projection_status"]["source"]["status"] == "VALID"
    assert result["source_area_km2"] > 0.0
    assert result["source_area_km2"] == pytest.approx(projected_source_area)
    assert result["overlap_polygon"]
    assert all(0.0 <= point[1] < 360.0 for point in result["overlap_polygon"])


def test_fixed_ohrc_lroc_pair_is_deterministic_and_reports_geometry_status():
    source = {"footprint": {
        "upper_left": [-89.199860, 222.259328],
        "upper_right": [-89.209060, 229.728973],
        "lower_left": [-89.908842, 110.268353],
        "lower_right": [-89.946885, 22.453651],
    }, "projection": "Polar stereographic"}
    reference = {"footprint": {
        "upper_left": [-88.07, 211.75],
        "upper_right": [-88.26, 197.21],
        "lower_left": [-88.96, 293.28],
        "lower_right": [-89.40, 311.20],
    }, "projection": "Polar stereographic"}
    first = evaluate_footprint_overlap(source, reference)
    second = evaluate_footprint_overlap(source, reference)
    assert first == second
    assert first["geometry_status"] == "VALID_INTERSECTION"
    assert first["projected_source_polygon"]
    assert first["projected_reference_polygon"]
    assert first["projected_overlap_polygon"]
    assert first["overlap_percentage_smaller"] > 0.0
    assert first["overlap_percentage_smaller"] < 100.0


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


def test_acceptance_overlap_0_095_percent_is_blocked_and_not_validated():
    footprint_eval = {
        "status": "insufficient_overlap",
        "has_overlap": True,
        "gate_passed": False,
        "overlap_percentage_smaller": 0.095,
        "minimum_overlap_percentage": 0.1,
        "overlap_area_km2": 1.234567,
    }
    status, reason = compute_pair_validation(
        MetadataStatus.CATALOG_METADATA,
        MetadataStatus.CATALOG_METADATA,
        FootprintStatus.AVAILABLE,
        FootprintStatus.AVAILABLE,
        footprint_eval,
    )
    assert status == PairValidationStatus.INSUFFICIENT_OVERLAP
    assert "0.095%" in reason
    assert "0.100%" in reason
    assert status != PairValidationStatus.VALID
