import numpy as np
import pytest
from core.scientific_reader import (
    LROC_NAC_EDR_PROFILE, ScientificRasterSpec, apply_raster_profile,
    check_file_size_consistency, is_lroc_nac_edr, load_scientific_preview,
    open_scientific_memmap, scientific_raster_spec, validate_scientific_raster_access,
)
from app_v3 import (
    _lroc_nac_edr_catalog_raw_record, _lroc_nac_edr_family_record,
    lroc_catalog_metadata_record,
)


def _make_meta(product_id=None, lines=52224, samples=5064):
    if product_id is None:
        product_id = "M1438615574LE"
    raw = {
        "sensor_type": "LROC_NAC", "product_id": product_id,
        "data_set_id": "LRO-L-LROC-2-EDR-V1.0", "product_type": "EDR",
        "metadata_source": "LROC_CATALOG",
        "dimensions": {"lines": lines, "samples": samples},
        "raster_spec": {"lines": lines, "samples": samples},
        "valid": True, "validation_errors": [],
    }
    return apply_raster_profile(raw, product_id + ".IMG")


def _synth(lines, samples, offset):
    hdr = b"\x00" * offset
    row = (bytes(range(256)) * (samples // 256 + 1))[:samples]
    return hdr + b"".join(row for _ in range(lines))


# 1. Profile recognition
def test_recognized_via_dataset_and_instrument():
    assert is_lroc_nac_edr({"data_set_id": "LRO-L-LROC-2-EDR-V1.0", "instrument": "LROC NAC", "product_type": "EDR"}) is True

def test_recognized_via_catalog_source():
    assert is_lroc_nac_edr({"metadata_source": "LROC_CATALOG", "sensor_type": "LROC_NAC", "instrument": "LROC NAC"}, "M1438615574LE.IMG") is True

def test_not_recognized_unrelated():
    assert is_lroc_nac_edr({"data_set_id": "X", "instrument": "OHRC"}) is False

def test_not_recognized_empty():
    assert is_lroc_nac_edr({}, "ch2_ohr.img") is False


# 2. Product identification
def test_m1438615574le_product_id():
    r = lroc_catalog_metadata_record("M1438615574LE.IMG")
    assert r["product_id"] == "M1438615574LE"
    assert r["sensor_type"] == "LROC_NAC"
    assert r["data_set_id"] == "LRO-L-LROC-2-EDR-V1.0"

def test_family_recognition_m1534362340le():
    f = _lroc_nac_edr_family_record("M1534362340LE")
    assert f is not None and f["data_set_id"] == "LRO-L-LROC-2-EDR-V1.0" and f["sensor_type"] == "LROC_NAC"

def test_raw_record_none_unknown():
    assert _lroc_nac_edr_catalog_raw_record("M9999999999LE") is None

def test_family_none_non_nac():
    assert _lroc_nac_edr_family_record("ch2_ohrc_something") is None


# 3. RECORD_BYTES = 5064
def test_record_bytes_is_5064():
    assert LROC_NAC_EDR_PROFILE.record_bytes == 5064


# 4. LABEL_RECORDS = 1
def test_label_records_is_1():
    assert LROC_NAC_EDR_PROFILE.label_records == 1


# 5. IMAGE pointer = 2
def test_image_pointer_is_2():
    assert LROC_NAC_EDR_PROFILE.image_pointer == 2


# 6. image offset = 5064
def test_image_offset_is_5064():
    spec = scientific_raster_spec(_make_meta())
    assert spec.image_offset == 5064
    assert spec.image_offset == (LROC_NAC_EDR_PROFILE.image_pointer - 1) * LROC_NAC_EDR_PROFILE.record_bytes


# 7. sample bits = 8
def test_sample_bits_is_8():
    assert LROC_NAC_EDR_PROFILE.sample_bits == 8
    assert scientific_raster_spec(_make_meta()).sample_bits == 8


# 8. dtype = uint8
def test_dtype_is_uint8():
    assert LROC_NAC_EDR_PROFILE.dtype == "uint8"
    spec = scientific_raster_spec(_make_meta())
    assert spec.dtype == "uint8"
    assert np.dtype(spec.dtype) == np.dtype("uint8")


# 9. lines = 52224
def test_lines_is_52224():
    r = lroc_catalog_metadata_record("M1438615574LE.IMG")
    assert r["dimensions"]["lines"] == 52224
    assert scientific_raster_spec(r).lines == 52224


# 10. samples = 5064
def test_samples_is_5064():
    r = lroc_catalog_metadata_record("M1438615574LE.IMG")
    assert r["dimensions"]["samples"] == 5064
    assert scientific_raster_spec(r).samples == 5064


# 11. Expected image payload
def test_expected_image_payload():
    spec = scientific_raster_spec(_make_meta())
    assert spec.lines * spec.samples * (spec.sample_bits // 8) == 264_462_336


# 12. Expected record count
def test_expected_record_count():
    spec = scientific_raster_spec(_make_meta())
    r = check_file_size_consistency(52225 * 5064, spec)
    assert r["expected_record_count"] == 52225
    assert r["expected_file_size_from_records"] == 52225 * 5064


# 13. File-size consistency
def test_verified_nominal():
    assert check_file_size_consistency(52225 * 5064, scientific_raster_spec(_make_meta()))["status"] == "VERIFIED"

def test_partial_one_missing():
    assert check_file_size_consistency(52225 * 5064 - 1, scientific_raster_spec(_make_meta()))["status"] == "PARTIAL"

def test_mismatch_too_small():
    assert check_file_size_consistency(52225 * 5064 - 5064 * 2, scientific_raster_spec(_make_meta()))["status"] == "MISMATCH"

def test_partial_extra_no_record_bytes():
    spec = ScientificRasterSpec(lines=4, samples=5, sample_bits=8, dtype="uint8", image_offset=4)
    assert check_file_size_consistency(28, spec)["status"] == "PARTIAL"

def test_verified_exact_no_record_bytes():
    spec = ScientificRasterSpec(lines=4, samples=5, sample_bits=8, dtype="uint8", image_offset=4)
    assert check_file_size_consistency(24, spec)["status"] == "VERIFIED"


# 14. memmap opening
def test_memmap_opens(tmp_path):
    L, S, OFF = 16, 5064, 5064
    f = tmp_path / "M_t14_LE.IMG"
    f.write_bytes(_synth(L, S, OFF))
    spec = ScientificRasterSpec(lines=L, samples=S, sample_bits=8, dtype="uint8", image_offset=OFF, record_bytes=5064)
    mm = open_scientific_memmap(str(f), spec)
    assert isinstance(mm, np.memmap) and mm.shape == (L, S) and mm.dtype == np.dtype("uint8")
    del mm


# 15. First-line read
def test_first_line_read(tmp_path):
    L, S, OFF = 32, 5064, 5064
    f = tmp_path / "M_t15_LE.IMG"
    f.write_bytes(_synth(L, S, OFF))
    mm = open_scientific_memmap(str(f), ScientificRasterSpec(lines=L, samples=S, sample_bits=8, dtype="uint8", image_offset=OFF))
    row = np.asarray(mm[0, :])
    assert row.shape == (S,) and row.dtype == np.dtype("uint8") and row.size > 0
    del mm


# 16. Middle-line read
def test_middle_line_read(tmp_path):
    L, S, OFF = 32, 5064, 5064
    f = tmp_path / "M_t16_LE.IMG"
    f.write_bytes(_synth(L, S, OFF))
    mm = open_scientific_memmap(str(f), ScientificRasterSpec(lines=L, samples=S, sample_bits=8, dtype="uint8", image_offset=OFF))
    row = np.asarray(mm[L // 2, :])
    assert row.shape == (S,) and row.dtype == np.dtype("uint8")
    del mm


# 17. Last-line read
def test_last_line_read(tmp_path):
    L, S, OFF = 32, 5064, 5064
    f = tmp_path / "M_t17_LE.IMG"
    f.write_bytes(_synth(L, S, OFF))
    mm = open_scientific_memmap(str(f), ScientificRasterSpec(lines=L, samples=S, sample_bits=8, dtype="uint8", image_offset=OFF))
    row = np.asarray(mm[L - 1, :])
    assert row.shape == (S,) and row.dtype == np.dtype("uint8")
    del mm


# 18. Preview generation
def test_preview_generation(tmp_path):
    L, S, OFF = 64, 5064, 5064
    f = tmp_path / "M_t18_LE.IMG"
    f.write_bytes(_synth(L, S, OFF))
    spec = ScientificRasterSpec(lines=L, samples=S, sample_bits=8, dtype="uint8", image_offset=OFF)
    preview, mask, info = load_scientific_preview(str(f), "M_t18_LE.IMG", {"raster_spec": spec.to_dict()}, max_side=128)
    assert preview.ndim == 2 and preview.dtype == np.dtype("float32")
    assert mask.shape == preview.shape
    assert info["label"] == "Visualization Preview" and info["is_preview"] is True
    assert info["original_shape"] == (L, S)


# 19. Invalid/truncated file rejection
def test_truncated_file_mismatch(tmp_path):
    spec = ScientificRasterSpec(lines=52224, samples=5064, sample_bits=8, dtype="uint8", image_offset=5064, record_bytes=5064)
    f = tmp_path / "trunc.IMG"
    f.write_bytes(b"\x00" * 100)
    assert validate_scientific_raster_access(str(f), spec)["status"] in {"MISMATCH", "PARTIAL"}

def test_empty_file_rejected(tmp_path):
    spec = ScientificRasterSpec(lines=4, samples=5, sample_bits=8, dtype="uint8", image_offset=0)
    f = tmp_path / "empty.IMG"
    f.write_bytes(b"")
    assert validate_scientific_raster_access(str(f), spec)["status"] in {"MISMATCH", "PARTIAL"}

def test_incomplete_spec_rejected():
    spec = ScientificRasterSpec(lines=4, samples=5, sample_bits=8, image_offset=0)
    r = validate_scientific_raster_access(b"\x00" * 20, spec)
    assert r["status"] == "PARTIAL" and r["error_class"] == "RASTER_FORMAT_UNKNOWN"


# 20. Generic support for another LROC NAC EDR product
def test_profile_second_product():
    meta = _make_meta("M1534362340LE", 52224, 5064)
    spec = scientific_raster_spec(meta)
    assert meta["format_profile"] == "LROC_NAC_EDR_PDS3"
    assert spec.lines == 52224 and spec.samples == 5064
    assert spec.dtype == "uint8" and spec.image_offset == 5064
    assert spec.is_decodable() is True

def test_profile_different_lines():
    meta = _make_meta("M9999999999RE", lines=10000, samples=5064)
    spec = scientific_raster_spec(meta)
    assert spec.lines == 10000 and spec.samples == 5064 and spec.image_offset == 5064

def test_is_lroc_nac_edr_generic():
    for pid in ("M1438615574LE", "M1534362340LE", "M9999999999RE"):
        assert is_lroc_nac_edr({"data_set_id": "LRO-L-LROC-2-EDR-V1.0", "instrument": "LROC NAC", "product_type": "EDR"}, pid + ".IMG") is True

def test_family_record_identifiers():
    f = _lroc_nac_edr_family_record("M9999999999RE")
    assert f["data_set_id"] == "LRO-L-LROC-2-EDR-V1.0"
    assert f["sensor_type"] == "LROC_NAC" and f["product_type"] == "EDR"
    assert f["dimensions"]["lines"] is None and f["dimensions"]["samples"] is None
