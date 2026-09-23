"""
Scientific Lunar Product & Label Parsers for Chandrayaan-2 and LRO NAC.

Supports:
- Chandrayaan-2 PDS4 XML labels and raw/calibrated binary rasters
- LRO LROC NAC PDS3 labels (attached or detached .LBL) and raw EDR rasters
- Multi-tier scientific representation:
  RAW/ORIGINAL -> SCIENTIFIC IMAGE DATA -> VISUALIZATION COPY -> PROCESSING COPY
- Memory-safe bounded loading
"""
import math
import re
from typing import Dict, Any, Optional, Tuple, List
import numpy as np


def _normalise_product_id(value: str) -> str:
    return re.sub(r"\.(?:img|lbl|xml)$", "", (value or "").strip().strip('"\''), flags=re.IGNORECASE).upper()


def verify_product_id(image_name: str, metadata_product_id: str) -> Dict[str, Any]:
    """Verify that supplied metadata identifies the uploaded product."""
    image_id = _normalise_product_id((image_name or "").replace("\\", "/").rsplit("/", 1)[-1])
    metadata_id = _normalise_product_id(metadata_product_id)
    verified = bool(image_id and metadata_id and image_id == metadata_id)
    return {
        "status": "VERIFIED" if verified else "MISMATCH",
        "image_product_id": image_id or None,
        "metadata_product_id": metadata_id or None,
        "message": "Product ID verified." if verified else "Reference image and metadata refer to different products.",
    }


def raster_dtype_from_pds(sample_type: Optional[str], sample_bits: Optional[int], byte_order: Optional[str] = None):
    """Map explicit PDS sample fields to a NumPy dtype; return None when unknown."""
    if not sample_type or not sample_bits:
        return None
    token = re.sub(r"[^A-Z0-9]", "", str(sample_type).upper())
    endian = ">" if byte_order and str(byte_order).upper() in {"MSB", "BIG_ENDIAN", "BIGENDIAN"} else "<"
    if token in {"UNSIGNEDINTEGER", "UNSIGNEDINT", "UNSIGNEDBYTE"} and sample_bits == 8:
        return np.dtype("u1")
    if token in {"UNSIGNEDINTEGER", "UNSIGNEDINT"} and sample_bits == 16:
        return np.dtype(f"{endian}u2")
    if token in {"SIGNEDINTEGER", "INTEGER", "SIGNEDINT"} and sample_bits == 16:
        return np.dtype(f"{endian}i2")
    if token in {"IEEEREAL", "IEEE754", "IEEE754MSBSINGLE", "IEEE754LSBSINGLE"} and sample_bits == 32:
        return np.dtype(f"{endian}f4")
    return None


def parse_lro_pds3_label(data: bytes | str, filename: str = "") -> Dict[str, Any]:
    """Parse LRO LROC NAC PDS3 label (attached or detached .LBL / .IMG header).

    Extracts:
    - Product ID, Mission, Instrument
    - Resolution / GSD (scaled pixel width)
    - Image dimensions (LINES, LINE_SAMPLES)
    - Sun and camera geometry (INCIDENCE, EMISSION, PHASE, SUB_SOLAR_AZIMUTH)
    - Bounding latitudes and longitudes / corner coordinates
    """
    if isinstance(data, bytes):
        # PDS3 labels are ASCII text
        text = data[:65536].decode("ascii", errors="replace")
    else:
        text = str(data)[:65536]

    # Helper to extract value by key regex
    def get_val(key: str, default: Optional[str] = None) -> Optional[str]:
        # Handle formats like KEY = VALUE or KEY = "VALUE"
        pattern = rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*$"
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            v = match.group(1).strip().strip('"\'')
            # Strip trailing comments or units like <METERS/PIXEL>
            v = re.sub(r"<.*?>", "", v).strip()
            return v
        return default

    def get_float(key: str, default: Optional[float] = None) -> Optional[float]:
        val = get_val(key)
        if val is not None:
            try:
                # Remove non-numeric except dot, minus, plus, e
                m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", val)
                if m:
                    return float(m.group(0))
            except Exception:
                pass
        return default

    def get_int(key: str, default: Optional[int] = None) -> Optional[int]:
        val = get_val(key)
        if val is not None:
            try:
                m = re.search(r"\d+", val)
                if m:
                    return int(m.group(0))
            except Exception:
                pass
        return default

    def get_pointer(key: str) -> tuple[Optional[int], Optional[str]]:
        match = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^\r\n]+)", text, re.MULTILINE | re.IGNORECASE)
        if not match:
            return None, None
        raw = match.group(1).strip()
        if not raw:
            return None, None
        if raw.startswith("("):
            raw = raw.split(")", 1)[1].strip() if ")" in raw else raw
        number_match = re.search(r"\d+", raw)
        if not number_match:
            return None, None
        number_value = int(number_match.group(0))
        unit_match = re.search(r"<\s*([^>]+)>", raw)
        unit = unit_match.group(1).upper() if unit_match else "RECORDS"
        if "BYTE" in unit.upper() or "BYTES" in unit.upper():
            return number_value, "BYTES"
        if "RECORD" in unit.upper():
            return number_value, "RECORDS"
        if "IMAGE" in unit.upper() or "OFFSET" in unit.upper():
            return number_value, "BYTES"
        return number_value, unit

    def first_float(*keys: str) -> Optional[float]:
        for key in keys:
            value = get_float(key)
            if value is not None:
                return value
        return None

    product_id = get_val("PRODUCT_ID")
    data_set_id = get_val("DATA_SET_ID")
    record_type = get_val("RECORD_TYPE")
    file_records = get_int("FILE_RECORDS")
    label_records = get_int("LABEL_RECORDS")
    inst_host = get_val("INSTRUMENT_HOST_NAME") or get_val("MISSION_NAME")
    instrument_id = get_val("INSTRUMENT_ID") or get_val("INSTRUMENT_NAME")
    product_type = get_val("PRODUCT_TYPE")
    target_name = get_val("TARGET_NAME") or get_val("TARGET")
    start_time = get_val("START_TIME")
    stop_time = get_val("STOP_TIME")

    lines = get_int("LINES")
    samples = get_int("LINE_SAMPLES")
    sample_bits = get_int("SAMPLE_BITS")
    sample_type = get_val("SAMPLE_TYPE")
    record_bytes = get_int("RECORD_BYTES")
    image_pointer, image_pointer_unit = get_pointer("^IMAGE")
    byte_order = get_val("MSB") or get_val("BYTE_ORDER")
    line_prefix_bytes = get_int("LINE_PREFIX_BYTES") or get_int("LINE_PREFIX")
    line_suffix_bytes = get_int("LINE_SUFFIX_BYTES") or get_int("LINE_SUFFIX")
    image_header = get_int("IMAGE_HEADER")
    image_offset = None
    if image_pointer is not None:
        unit = str(image_pointer_unit or "").upper()
        if "BYTE" in unit:
            image_offset = image_pointer
        elif "RECORD" in unit:
            image_offset = max(image_pointer - 1, 0) * int(record_bytes or 0)
        elif record_bytes:
            image_offset = max(image_pointer - 1, 0) * int(record_bytes)
        elif image_pointer and image_pointer > 1:
            image_offset = max(image_pointer - 1, 0)

    resolution = get_float("SCALED_PIXEL_WIDTH") or get_float("RESOLUTION")

    # Geometry
    incidence = get_float("INCIDENCE_ANGLE")
    emission = get_float("EMISSION_ANGLE")
    phase = get_float("PHASE_ANGLE")
    sun_azimuth = get_float("SUB_SOLAR_AZIMUTH")
    altitude = get_float("SPACECRAFT_ALTITUDE")

    # Footprint corners
    # LRO labels often have MINIMUM_LATITUDE, MAXIMUM_LATITUDE, WESTERNMOST_LONGITUDE, EASTERNMOST_LONGITUDE
    # or CORNER1_LATITUDE ...
    ul_lat = first_float("UPPER_LEFT_LATITUDE", "CORNER1_LATITUDE", "MAXIMUM_LATITUDE")
    ul_lon = first_float("UPPER_LEFT_LONGITUDE", "CORNER1_LONGITUDE", "WESTERNMOST_LONGITUDE")
    ur_lat = first_float("UPPER_RIGHT_LATITUDE", "CORNER2_LATITUDE", "MAXIMUM_LATITUDE")
    ur_lon = first_float("UPPER_RIGHT_LONGITUDE", "CORNER2_LONGITUDE", "EASTERNMOST_LONGITUDE")
    lr_lat = first_float("LOWER_RIGHT_LATITUDE", "CORNER3_LATITUDE", "MINIMUM_LATITUDE")
    lr_lon = first_float("LOWER_RIGHT_LONGITUDE", "CORNER3_LONGITUDE", "EASTERNMOST_LONGITUDE")
    ll_lat = first_float("LOWER_LEFT_LATITUDE", "CORNER4_LATITUDE", "MINIMUM_LATITUDE")
    ll_lon = first_float("LOWER_LEFT_LONGITUDE", "CORNER4_LONGITUDE", "WESTERNMOST_LONGITUDE")

    footprint = {
        "upper_left": [ul_lat, ul_lon] if ul_lat is not None and ul_lon is not None else [],
        "upper_right": [ur_lat, ur_lon] if ur_lat is not None and ur_lon is not None else [],
        "lower_left": [ll_lat, ll_lon] if ll_lat is not None and ll_lon is not None else [],
        "lower_right": [lr_lat, lr_lon] if lr_lat is not None and lr_lon is not None else [],
    }

    # Sun elevation is 90 - incidence angle
    sun_elevation = (90.0 - incidence) if incidence is not None else None

    # Keep ``valid`` as the historical parser-recognized-record flag. The app
    # separately requires complete resolution and footprint metadata before
    # scientific validation can run.
    is_valid = bool(product_id and lines and samples)

    return {
        "mission": inst_host,
        "instrument": f"{instrument_id} NAC" if instrument_id and "LROC" in instrument_id.upper() else instrument_id,
        "sensor_type": "LROC_NAC" if instrument_id and "LROC" in instrument_id.upper() else instrument_id,
        "product_id": product_id,
        "data_set_id": data_set_id,
        "record_type": record_type,
        "file_records": file_records,
        "label_records": label_records,
        "target": target_name,
        "processing_level": product_type,
        "start_time": start_time,
        "stop_time": stop_time,
        "gsd_m_per_pixel": float(resolution) if resolution is not None else None,
        "altitude_km": float(altitude) if altitude is not None else None,
        "roll_deg": None,
        "pitch_deg": None,
        "yaw_deg": None,
        "sun_azimuth_deg": float(sun_azimuth) if sun_azimuth is not None else None,
        "sun_elevation_deg": float(sun_elevation) if sun_elevation is not None else None,
        "solar_incidence_deg": float(incidence) if incidence is not None else None,
        "emission_deg": float(emission) if emission is not None else None,
        "phase_deg": float(phase) if phase is not None else None,
        "projection": None,
        "area": None,
        "footprint": footprint,
        "dimensions": {"lines": lines, "samples": samples},
        "data_type": sample_type,
        "sample_bits": sample_bits,
        "sample_type": sample_type,
        "byte_order": byte_order,
        "image_offset": image_offset,
        "image_pointer": image_pointer,
        "image_pointer_unit": image_pointer_unit,
        "record_bytes": record_bytes,
        "image_header_bytes": image_header,
        "line_prefix_bytes": line_prefix_bytes,
        "line_suffix_bytes": line_suffix_bytes,
        "raster_spec": {
            "lines": lines,
            "samples": samples,
            "sample_bits": sample_bits,
            "dtype": str(raster_dtype_from_pds(sample_type, sample_bits, byte_order)) if raster_dtype_from_pds(sample_type, sample_bits, byte_order) else None,
            "sample_type": sample_type,
            "byte_order": byte_order,
            "image_offset": image_offset,
            "image_pointer": image_pointer,
            "image_pointer_unit": image_pointer_unit,
            "record_bytes": record_bytes,
            "line_prefix_bytes": line_prefix_bytes,
            "line_suffix_bytes": line_suffix_bytes,
            "product_id": product_id,
        },
        "metadata_source": "PDS3_LBL",
        "valid": is_valid,
        "validation_errors": [] if is_valid else ["Reference metadata is incomplete"],
        "validation_warnings": [],
    }


def read_scientific_binary(
    data: bytes,
    lines: int,
    samples: int,
    data_type: str = "UnsignedByte",
    offset: int = 0,
    max_side: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Read a scientific raster from binary bytes safely into memory.

    Returns:
    - scientific_arr: 2D float32 array preserving calibrated values
    - mask: boolean valid pixel mask
    - meta: dictionary of loading details
    """
    dtype_map = {
        "unsignedbyte": np.dtype("uint8"),
        "byte": np.dtype("uint8"),
        "uint8": np.dtype("uint8"),
        "signedmsb2": np.dtype(">i2"),
        "signedlsb2": np.dtype("<i2"),
        "int16": np.dtype("int16"),
        "ieee754msbsingle": np.dtype(">f4"),
        "ieee754lsbsingle": np.dtype("<f4"),
        "float32": np.dtype("float32"),
    }

    dt = dtype_map.get(data_type.lower().replace("_", "").replace(" ", ""), np.dtype("uint8"))
    bytes_per_sample = dt.itemsize
    expected_bytes = lines * samples * bytes_per_sample

    available_bytes = len(data) - offset
    if available_bytes < expected_bytes:
        # Check if file has enough bytes; if not, raise informative error
        raise ValueError(
            f"Insufficient binary data: header specified {lines}×{samples} ({expected_bytes} bytes), "
            f"but only {available_bytes} bytes available at offset {offset}."
        )

    # Subsampling factor for memory-safe processing
    stride = max(1, math.ceil(max(lines, samples) / max_side))
    out_lines = lines // stride
    out_samples = samples // stride

    # Use buffer view with strided reading
    raw_flat = np.frombuffer(data, dtype=dt, count=lines * samples, offset=offset)
    arr_2d = raw_flat.reshape((lines, samples))

    if stride > 1:
        arr_sub = arr_2d[::stride, ::stride].astype(np.float32)
    else:
        arr_sub = arr_2d.astype(np.float32)

    valid_mask = np.isfinite(arr_sub) & (arr_sub > 0)

    load_meta = {
        "original_shape": (lines, samples),
        "loaded_shape": arr_sub.shape,
        "stride": stride,
        "dtype": str(dt),
        "data_type": data_type,
        "processing_limit_max_side": max_side,
    }

    return arr_sub, valid_mask, load_meta
