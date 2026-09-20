"""
Scientific Lunar Product & Label Parsers for Chandrayaan-2 and LRO NAC.

Supports:
- Chandrayaan-2 PDS4 XML labels and raw/calibrated binary rasters
- LRO LROC NAC PDS3 labels (attached or detached .LBL) and raw EDR rasters
- Multi-tier scientific representation:
  RAW/ORIGINAL -> SCIENTIFIC IMAGE DATA -> VISUALIZATION COPY -> PROCESSING COPY
- Memory-safe bounded loading
"""
import io
import math
import re
from typing import Dict, Any, Optional, Tuple, List
import numpy as np


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
        pattern = rf"^\s*{key}\s*=\s*(.+?)\s*$"
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

    product_id = get_val("PRODUCT_ID", filename)
    inst_host = get_val("INSTRUMENT_HOST_NAME", "LUNAR RECONNAISSANCE ORBITER")
    instrument_id = get_val("INSTRUMENT_ID", "LROC")
    product_type = get_val("PRODUCT_TYPE", "EDR")
    start_time = get_val("START_TIME", "")
    stop_time = get_val("STOP_TIME", "")

    lines = get_int("LINES")
    samples = get_int("LINE_SAMPLES")
    sample_bits = get_int("SAMPLE_BITS", 8)
    sample_type = get_val("SAMPLE_TYPE", "UNSIGNED_INTEGER")

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
    ul_lat = get_float("UPPER_LEFT_LATITUDE") or get_float("CORNER1_LATITUDE") or get_float("MAXIMUM_LATITUDE")
    ul_lon = get_float("UPPER_LEFT_LONGITUDE") or get_float("CORNER1_LONGITUDE") or get_float("WESTERNMOST_LONGITUDE")
    ur_lat = get_float("UPPER_RIGHT_LATITUDE") or get_float("CORNER2_LATITUDE") or get_float("MAXIMUM_LATITUDE")
    ur_lon = get_float("UPPER_RIGHT_LONGITUDE") or get_float("CORNER2_LONGITUDE") or get_float("EASTERNMOST_LONGITUDE")
    lr_lat = get_float("LOWER_RIGHT_LATITUDE") or get_float("CORNER3_LATITUDE") or get_float("MINIMUM_LATITUDE")
    lr_lon = get_float("LOWER_RIGHT_LONGITUDE") or get_float("CORNER3_LONGITUDE") or get_float("EASTERNMOST_LONGITUDE")
    ll_lat = get_float("LOWER_LEFT_LATITUDE") or get_float("CORNER4_LATITUDE") or get_float("MINIMUM_LATITUDE")
    ll_lon = get_float("LOWER_LEFT_LONGITUDE") or get_float("CORNER4_LONGITUDE") or get_float("WESTERNMOST_LONGITUDE")

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
        "instrument": f"{instrument_id} NAC" if "LROC" in instrument_id else instrument_id,
        "sensor_type": "LROC_NAC" if "LROC" in instrument_id else instrument_id,
        "product_id": product_id or filename,
        "processing_level": product_type,
        "start_time": start_time,
        "stop_time": stop_time,
        "gsd_m_per_pixel": float(resolution) if resolution is not None else None,
        "altitude_km": float(altitude) if altitude is not None else None,
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "yaw_deg": 0.0,
        "sun_azimuth_deg": float(sun_azimuth) if sun_azimuth is not None else None,
        "sun_elevation_deg": float(sun_elevation) if sun_elevation is not None else None,
        "solar_incidence_deg": float(incidence) if incidence is not None else None,
        "emission_deg": float(emission) if emission is not None else None,
        "phase_deg": float(phase) if phase is not None else None,
        "projection": "Equirectangular / Polar Stereographic",
        "area": "Lunar Surface",
        "footprint": footprint,
        "dimensions": {"lines": lines, "samples": samples},
        "data_type": sample_type,
        "sample_bits": sample_bits,
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
