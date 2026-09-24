"""LROC NAC EDR PDS3 generic raster profile.

Resolves sample_type, dtype, record_bytes, and image_offset for LROC NAC EDR
products from verified catalog dimensions + the standard PDS3 layout:
  RECORD_BYTES = LINE_SAMPLES (no line prefix/suffix)
  LABEL_RECORDS = 1
  ^IMAGE = 2  (record number)
  image_offset = (pointer - 1) * RECORD_BYTES
  SAMPLE_BITS = 8, SAMPLE_TYPE = UNSIGNED_INTEGER -> uint8

Explicit LBL values always take precedence. Never fabricates an LBL.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

LROC_NAC_EDR_PDS3_PROFILE_ID = "LROC_NAC_EDR_PDS3"


def is_lroc_nac_edr_product(metadata: Optional[Dict[str, Any]]) -> bool:
    """Detect LROC NAC EDR products from available metadata (not product ID alone)."""
    if not isinstance(metadata, dict):
        return False
    sensor = str(
        metadata.get("sensor_type") or metadata.get("instrument") or ""
    ).upper().replace("-", " ").replace("_", " ")
    is_lroc_nac = any(token in sensor for token in ("LROC NAC", "LROC_NAC", "LROC", "NAC")) or str(
        metadata.get("sensor_type", "")
    ).upper() in {"LROC_NAC", "LROC", "NAC"}
    level = str(
        metadata.get("processing_level")
        or metadata.get("product_type")
        or metadata.get("calibration_state")
        or ""
    ).upper()
    is_edr = (not level) or ("EDR" in level) or level in {"RAW", "PDS3_RAW", "ORIGINAL"}
    product_id = str(metadata.get("product_id") or "").upper()
    name_looks_lroc = bool(re.match(r"^M\d+[LR]E$", product_id))
    return bool((is_lroc_nac or name_looks_lroc) and is_edr)


def apply_lroc_nac_edr_pds3_profile(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fill missing PDS3 raster-contract fields for recognized LROC NAC EDR products."""
    meta = dict(metadata or {})
    if not is_lroc_nac_edr_product(meta):
        return meta

    dims = dict(meta.get("dimensions") or {})
    raw = dict(meta.get("raster_spec") or {})

    lines = raw.get("lines") or dims.get("lines") or meta.get("lines")
    samples = raw.get("samples") or dims.get("samples") or meta.get("samples")
    if not lines or not samples:
        return meta

    lines = int(lines)
    samples = int(samples)

    sample_bits = int(raw.get("sample_bits") or meta.get("sample_bits") or 8)
    sample_type = (
        raw.get("sample_type")
        or meta.get("sample_type")
        or meta.get("data_type")
        or "UNSIGNED_INTEGER"
    )
    dtype = raw.get("dtype") or ("uint8" if sample_bits == 8 else "uint8")

    record_bytes = raw.get("record_bytes") or meta.get("record_bytes")
    if record_bytes is None:
        bytes_per_sample = max(1, sample_bits // 8)
        record_bytes = samples * bytes_per_sample
    record_bytes = int(record_bytes)

    image_pointer = raw.get("image_pointer")
    if image_pointer is None:
        image_pointer = meta.get("image_pointer")
    image_pointer_unit = raw.get("image_pointer_unit") or meta.get("image_pointer_unit")
    image_offset = raw.get("image_offset")
    if image_offset is None:
        image_offset = meta.get("image_offset", meta.get("offset"))

    if image_offset is None:
        pointer = int(image_pointer) if image_pointer is not None else 2
        unit = str(image_pointer_unit or "RECORDS").upper()
        if "BYTE" in unit:
            image_offset = pointer
        else:
            image_offset = max(pointer - 1, 0) * record_bytes
        if image_pointer is None:
            image_pointer = 2
            image_pointer_unit = "RECORDS"

    image_offset = int(image_offset)
    profile_provenance = "LROC EDR archive format specification"

    raster_spec = {
        "lines": lines,
        "samples": samples,
        "sample_bits": sample_bits,
        "dtype": dtype,
        "sample_type": sample_type,
        "byte_order": raw.get("byte_order") or meta.get("byte_order"),
        "image_offset": image_offset,
        "image_pointer": image_pointer,
        "image_pointer_unit": image_pointer_unit or "RECORDS",
        "record_bytes": record_bytes,
        "line_prefix_bytes": raw.get("line_prefix_bytes") or meta.get("line_prefix_bytes") or 0,
        "line_suffix_bytes": raw.get("line_suffix_bytes") or meta.get("line_suffix_bytes") or 0,
        "product_id": raw.get("product_id") or meta.get("product_id"),
        "profile_id": LROC_NAC_EDR_PDS3_PROFILE_ID,
        "provenance": profile_provenance,
    }

    meta["dimensions"] = {"lines": lines, "samples": samples}
    meta["sample_bits"] = sample_bits
    meta["sample_type"] = sample_type
    meta["data_type"] = sample_type
    meta["image_offset"] = image_offset
    meta["image_pointer"] = image_pointer
    meta["image_pointer_unit"] = image_pointer_unit or "RECORDS"
    meta["record_bytes"] = record_bytes
    meta["raster_spec"] = raster_spec
    meta["raster_profile_id"] = LROC_NAC_EDR_PDS3_PROFILE_ID
    meta["raster_spec_provenance"] = profile_provenance
    if not meta.get("sensor_type"):
        meta["sensor_type"] = "LROC_NAC"
    if not meta.get("processing_level"):
        meta["processing_level"] = "EDR"
    return meta


def unresolved_raster_contract_message(spec) -> str:
    missing = []
    if not getattr(spec, "lines", None):
        missing.append("lines")
    if not getattr(spec, "samples", None):
        missing.append("samples")
    if not getattr(spec, "sample_bits", None):
        missing.append("sample bits")
    if not getattr(spec, "dtype", None) and not getattr(spec, "sample_type", None):
        missing.append("sample type")
    if getattr(spec, "image_offset", None) is None:
        missing.append("image offset")
    if not missing:
        return "Scientific raster contract unresolved"
    return "Scientific raster contract unresolved. Missing: " + ", ".join(missing)
