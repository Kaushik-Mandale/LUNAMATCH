"""
Unit tests for PAIR_001 configuration, PDS3 LRO NAC label parsing,
and scientific lunar product binary loading.
"""
import numpy as np
import pytest
from core.data_models import (
    PAIR_001_CONFIG,
    PAIR_001_SOURCE,
    PAIR_001_REFERENCE,
    calculate_scale_ratio,
)
from core.pds_parser import parse_lro_pds3_label, read_scientific_binary
import app_v3


def test_pair_001_configuration_and_scale_ratio():
    """Verify PAIR_001 specification: OHRC -> LRO NAC (~7.8x scale variation)."""
    src = PAIR_001_SOURCE
    ref = PAIR_001_REFERENCE

    assert src.mission == "Chandrayaan-2"
    assert src.instrument == "OHRC"
    assert src.product_id == "ch2_ohr_ncp_20190906T2241285714_d_img_gds"
    assert src.calibration_state == "Calibrated"
    assert src.resolution_gsd_m == 0.24

    assert ref.mission == "Lunar Reconnaissance Orbiter"
    assert ref.instrument == "LROC NAC"
    assert ref.product_id == "M1534362340LE"
    assert ref.calibration_state == "EDR"
    assert ref.resolution_gsd_m == 1.8669

    ratio, scale_str = calculate_scale_ratio(src.resolution_gsd_m, ref.resolution_gsd_m)
    assert ratio is not None
    assert round(ratio, 2) == 7.78
    assert scale_str == "Scale Difference ≈ 7.8×"


def test_parse_lro_pds3_label_extracts_correct_fields():
    """Test parsing an attached or detached LRO LROC NAC PDS3 label."""
    sample_lbl = """
PDS_VERSION_ID = PDS3
RECORD_TYPE = FIXED_LENGTH
RECORD_BYTES = 5064
FILE_RECORDS = 52224
^IMAGE = 1
PRODUCT_ID = "M1534362340LE"
INSTRUMENT_HOST_NAME = "LUNAR RECONNAISSANCE ORBITER"
INSTRUMENT_ID = "LROC"
PRODUCT_TYPE = "EDR"
START_TIME = "2021-03-12T14:15:30.000"
STOP_TIME = "2021-03-12T14:15:45.000"
SPACECRAFT_ALTITUDE = 50.4 <KM>
SCALED_PIXEL_WIDTH = 1.8669 <METERS/PIXEL>
RESOLUTION = 1.8669 <METERS/PIXEL>
INCIDENCE_ANGLE = 71.5 <DEGREE>
EMISSION_ANGLE = 2.1 <DEGREE>
PHASE_ANGLE = 53.2 <DEGREE>
SUB_SOLAR_AZIMUTH = 72.1 <DEGREE>
LINES = 52224
LINE_SAMPLES = 5064
SAMPLE_BITS = 8
SAMPLE_TYPE = UNSIGNED_INTEGER
UPPER_LEFT_LATITUDE = -70.60
UPPER_LEFT_LONGITUDE = 22.40
UPPER_RIGHT_LATITUDE = -70.62
UPPER_RIGHT_LONGITUDE = 23.80
LOWER_LEFT_LATITUDE = -71.70
LOWER_LEFT_LONGITUDE = 22.35
LOWER_RIGHT_LATITUDE = -71.72
LOWER_RIGHT_LONGITUDE = 23.75
END
"""
    meta = parse_lro_pds3_label(sample_lbl, "M1534362340LE.LBL")

    assert meta["valid"] is True
    assert meta["mission"] == "LUNAR RECONNAISSANCE ORBITER"
    assert "NAC" in meta["instrument"]
    assert meta["product_id"] == "M1534362340LE"
    assert meta["gsd_m_per_pixel"] == 1.8669
    assert meta["dimensions"]["lines"] == 52224
    assert meta["dimensions"]["samples"] == 5064
    assert meta["solar_incidence_deg"] == 71.5
    assert meta["sun_elevation_deg"] == 90.0 - 71.5
    assert meta["sun_azimuth_deg"] == 72.1
    assert meta["footprint"]["upper_left"] == [-70.60, 22.40]


def test_parse_metadata_xml_routes_to_lro_pds3_when_given_lbl():
    """parse_metadata_xml routes PDS3 labels to parse_lro_pds3_label seamlessly."""
    pds3_text = b"PDS_VERSION_ID = PDS3\nPRODUCT_ID = \"M1534362340LE\"\nLINES = 1000\nLINE_SAMPLES = 500\nEND"
    meta = app_v3.parse_metadata_xml(pds3_text, "M1534362340LE.IMG")
    assert meta["valid"] is True
    assert meta["product_id"] == "M1534362340LE"
    assert meta["dimensions"]["lines"] == 1000
    assert meta["dimensions"]["samples"] == 500


def test_read_scientific_binary_safe_loading():
    """Verify reading raw 2D image binary bytes safely with memory limit."""
    # Synthetic 200x300 uint8 binary image
    lines, samples = 200, 300
    orig_data = np.arange(lines * samples, dtype=np.uint8).reshape((lines, samples))
    raw_bytes = orig_data.tobytes()

    arr, valid, load_meta = read_scientific_binary(raw_bytes, lines, samples, data_type="UnsignedByte", max_side=150)

    assert arr.ndim == 2
    assert max(arr.shape) <= 150
    assert load_meta["original_shape"] == (200, 300)
    assert load_meta["stride"] == 2
    assert arr.dtype == np.float32
    assert valid.shape == arr.shape
