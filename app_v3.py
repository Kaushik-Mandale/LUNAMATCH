import io
import json
import logging
import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from PIL import Image
from rasterio.enums import Resampling
from rasterio.io import MemoryFile

import loftr_matcher
import iirs_preprocessing
from core.data_models import (
    PAIR_001_CONFIG,
    PAIR_001_SOURCE,
    PAIR_001_REFERENCE,
    calculate_scale_ratio,
    ImageMetadataRecord,
    FootprintCoordinates,
)
from core.pds_parser import parse_lro_pds3_label, read_scientific_binary
from core.footprint import evaluate_footprint_overlap, extract_overlap_rois, compute_overlap_pixel_roi
from core.spatial import balance_correspondences_spatially
from core.registration import (
    create_alpha_overlay,
    create_checkerboard_comparison,
    draw_inlier_outlier_matches,
    warp_source_to_reference,
)
from core.evaluation import (
    compute_reprojection_rmse,
    compile_evaluation_report,
    build_method_comparison_table,
)
from core.product_state import (
    MetadataStatus,
    FootprintStatus,
    PairValidationStatus,
    METADATA_STATUS_LABELS,
    FOOTPRINT_STATUS_LABELS,
    PAIR_STATUS_LABELS,
    compute_file_hash,
    compute_metadata_status,
    compute_footprint_status,
    compute_pair_validation,
    scale_ratio_display,
)
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
from experiments.manager import ExperimentManager

logger = logging.getLogger("lunamatch_v3")

st.set_page_config(page_title="LunaMatch V3", page_icon="🌙", layout="wide")

st.markdown("""
<style>
    .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .stImage img {
        max-height: 440px;
        object-fit: contain;
        border-radius: 6px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
        font-weight: 600;
    }
    .pair-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .scale-badge {
        background: #1e3a8a;
        color: #93c5fd;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

SENSORS = ["OHRC", "TMC", "IIRS", "LROC_NAC", "Unknown"]
SENSORS_UI = ["Auto Detect", "OHRC", "TMC-2", "IIRS", "LROC NAC"]

SOURCE_MISSIONS = ["Chandrayaan-2"]
SOURCE_SENSORS_UI = ["OHRC", "TMC-2", "IIRS", "Auto Detect"]

REFERENCE_MISSIONS = [
    "Lunar Reconnaissance Orbiter (LRO)",
    "SELENE (Kaguya)",
    "Other Lunar Reference",
    "Chandrayaan-2 (Cross-Sensor Testing)",
]
REFERENCE_SENSORS_UI = [
    "LROC NAC",
    "Terrain Camera (TC)",
    "Multiband Imager (MI)",
    "Other Lunar Reference",
    "Auto Detect",
    "OHRC",
    "TMC-2",
    "IIRS",
]

# Canonical sensor tokens
_TMC_TOKENS = {"TMC", "TMC-2", "TMC2"}
_LRO_TOKENS = {"LRO", "LROC", "LROC NAC", "LROC_NAC", "NAC", "LRO / LROC NAC"}


def _canonical_sensor(sensor_ui: str) -> str:
    """Normalise UI sensor label to the canonical pipeline token.

    - 'TMC-2' -> 'TMC'  (keeps test contracts that compare against 'TMC')
    - 'LROC NAC' / 'LRO' -> 'LROC_NAC'
    - 'Auto Detect' -> 'Unknown' (resolved later from metadata / filename)
    - Everything else is returned as-is.
    """
    s = (sensor_ui or "").strip().upper()
    if s in {"TMC-2", "TMC2"}:
        return "TMC"
    if s in {"LROC NAC", "LROC_NAC", "NAC", "LRO", "LRO / LROC NAC", "LUNAR RECONNAISSANCE ORBITER"}:
        return "LROC_NAC"
    if s == "AUTO DETECT":
        return "Unknown"
    return s if s else "Unknown"


def detect_sensor_from_filename(name: str) -> str:
    """Infer sensor from product filename patterns.

    ch2_iir* -> 'IIRS'
    ch2_ohr* -> 'OHRC'
    ch2_tmc* -> 'TMC'
    m1* / nac* / lro* -> 'LROC_NAC'
    Returns 'Unknown' when no pattern matches.
    """
    lname = (name or "").lower()
    if "ch2_iir" in lname or "iirs" in lname:
        return "IIRS"
    if "ch2_ohr" in lname or "ohrc" in lname:
        return "OHRC"
    if "ch2_tmc" in lname or "tmc" in lname:
        return "TMC"
    if "m1" in lname or "nac" in lname or "lro" in lname or "lroc" in lname:
        return "LROC_NAC"
    return "Unknown"


def classify_product_file(name: str) -> str:
    """Classify an uploaded file without treating a preview as science data."""
    if not name:
        return "UNKNOWN"
    return classify_product_display_label(name)


def metadata_status(meta: dict, source: str) -> dict:
    """Return an explicit metadata lifecycle state dict for UI and gating.

    Uses the MetadataStatus enum for reliable status comparisons.
    """
    enum_status = compute_metadata_status(meta, source)
    icon, label, detail = METADATA_STATUS_LABELS.get(
        enum_status, ("?", str(enum_status.value), "")
    )
    return {
        "status":      enum_status.value,  # string for legacy dict-key comparisons
        "enum_status": enum_status,
        "source":      source,
        "label":       label,
        "icon":        icon,
        "detail":      detail,
    }



@dataclass
class WorkingImage:
    gray: np.ndarray
    structure: np.ndarray
    gradient: np.ndarray
    mask: np.ndarray
    to_original: np.ndarray
    info: dict


METADATA_KEYS = [
    "latitude", "longitude", "resolution_gsd", "altitude", "sun_azimuth",
    "sun_elevation", "solar_incidence", "roll", "pitch", "yaw",
    "footprint_corners", "sensor_type", "processing_level",
    "image_width", "image_height", "projection",
    "observation_start_time", "observation_stop_time",
]

METADATA_DISPLAY = {
    "latitude": "Latitude",
    "longitude": "Longitude",
    "resolution_gsd": "Resolution / GSD",
    "altitude": "Altitude",
    "sun_azimuth": "Sun Azimuth",
    "sun_elevation": "Sun Elevation",
    "solar_incidence": "Solar Incidence",
    "roll": "Roll",
    "pitch": "Pitch",
    "yaw": "Yaw",
    "footprint_corners": "Image Footprint / Corners",
    "sensor_type": "Sensor Type",
    "processing_level": "Processing Level",
    "image_width": "Image Width",
    "image_height": "Image Height",
    "projection": "Projection",
    "observation_start_time": "Observation Start Time",
    "observation_stop_time": "Observation Stop Time",
}

NS = {
    "pds": "http://pds.nasa.gov/pds4/pds/v1",
    "isda": "https://isda.issdc.gov.in/pds4/isda/v1",
}


def is_tiff(name: str) -> bool:
    return name.lower().endswith((".tif", ".tiff"))


def normalize_key(key: str) -> str:
    text = key.strip().replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "_")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.lower()


def _local_tag(elem) -> str:
    return elem.tag.split("}")[-1] if elem is not None else ""


def _parse_float(value, default=None):
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        cleaned = (
            text.replace("km", "")
            .replace("m/pixel", "")
            .replace("m", "")
            .replace("deg", "")
            .replace("°", "")
            .replace("um", "")
            .replace("mm", "")
            .strip()
        )
        return float(cleaned)
    except Exception:
        return default


def _parse_int(value, default=None):
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except Exception:
        return default


def _valid_corner_footprint(footprint: dict) -> bool:
    corners = ("upper_left", "upper_right", "lower_left", "lower_right")
    return isinstance(footprint, dict) and all(
        isinstance(footprint.get(c), (list, tuple))
        and len(footprint[c]) == 2
        and all(isinstance(value, (int, float)) for value in footprint[c])
        for c in corners
    )


def geographic_overlap(source_meta: dict, reference_meta: dict) -> dict:
    """Compatibility wrapper around the canonical wrap-aware validator."""
    return evaluate_footprint_overlap(source_meta, reference_meta)
def _find_elem(root, ns_paths: list, local_names: list | None = None):
    for path in ns_paths:
        try:
            node = root.find(path, NS)
            if node is not None:
                return node
        except Exception:
            pass
    if local_names:
        target_set = {normalize_key(n) for n in local_names}
        for elem in root.iter():
            if normalize_key(_local_tag(elem)) in target_set:
                return elem
    return None


def _find_text(root, ns_paths: list, local_names: list | None = None) -> str:
    elem = _find_elem(root, ns_paths, local_names)
    return (elem.text or "").strip() if elem is not None and elem.text else ""


def parse_footprint_string_to_pairs(raw: object) -> list:
    if isinstance(raw, (list, tuple)):
        return [[float(a), float(b)] for a, b in raw[:4]] if raw and len(raw) >= 4 else []
    if isinstance(raw, dict):
        if "upper_left" in raw and "upper_right" in raw and "lower_left" in raw and "lower_right" in raw:
            return [raw["upper_left"], raw["upper_right"], raw["lower_left"], raw["lower_right"]]
        if "footprint_corners" in raw:
            return parse_footprint_string_to_pairs(raw.get("footprint_corners"))
    text = str(raw or "").strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], (list, tuple)):
                pairs = []
                for item in parsed[:4]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        pairs.append([float(item[0]), float(item[1])])
                return pairs
        except Exception:
            pass
    tokens = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if len(tokens) >= 8:
        pairs = []
        for i in range(0, len(tokens), 2):
            if len(pairs) >= 4:
                break
            if i + 1 < len(tokens):
                pairs.append([float(tokens[i]), float(tokens[i + 1])])
        return pairs[:4]
    return []


def _extract_corners_fallback(root) -> list:
    corner_elems = []
    for elem in root.iter():
        lt = _local_tag(elem).lower()
        m = re.fullmatch(r"corner(\d+)", lt)
        if m and elem.text:
            corner_elems.append((int(m.group(1)), elem.text.strip()))
    if len(corner_elems) >= 4:
        corner_elems.sort(key=lambda x: x[0])
        pairs = []
        for _, text in corner_elems[:4]:
            parsed = parse_footprint_string_to_pairs(text)
            if parsed:
                pairs.append(parsed[0])
        if len(pairs) == 4:
            return pairs
    return []


def format_footprint_for_display(footprint) -> str:
    if isinstance(footprint, dict):
        return (
            f"UL: {footprint.get('upper_left', [])}; UR: {footprint.get('upper_right', [])}; "
            f"LL: {footprint.get('lower_left', [])}; LR: {footprint.get('lower_right', [])}"
        )
    if isinstance(footprint, list) and len(footprint) == 4:
        return "; ".join(f"[{lat}, {lon}]" for lat, lon in footprint)
    return str(footprint or "")


def validate_metadata(meta: dict) -> dict:
    """Validate canonical Chandrayaan-2 PDS4 metadata.

    Required fields:
    - instrument/sensor
    - processing_level
    - start_time
    - stop_time
    - gsd
    - altitude
    - sun geometry
    - camera geometry
    - footprint
    - projection
    - dimensions
    """
    errors = []
    if not isinstance(meta, dict):
        return {
            "valid": False,
            "validation_errors": ["Metadata is not a dictionary"],
            "invalid": ["Metadata is not a dictionary"],
            "missing_required": ["Metadata is not a dictionary"],
            "missing_optional": [],
            "warnings": [],
        }

    # 1. instrument/sensor
    sensor = meta.get("sensor_type")
    instrument = meta.get("instrument")
    if (not sensor or sensor == "Unknown") and not instrument:
        errors.append("Missing sensor/instrument")

    # 2. processing_level
    if not meta.get("processing_level"):
        errors.append("Missing processing_level")

    # 3. start_time
    if not meta.get("start_time"):
        errors.append("Missing start_time")

    # 4. stop_time
    if not meta.get("stop_time"):
        errors.append("Missing stop_time")

    # 5. gsd is useful for scale analysis but is not a corruption signal for
    # valid derived products that provide other spatial metadata.
    gsd_available = meta.get("gsd_m_per_pixel") is not None or meta.get("resolution_gsd") is not None
    warnings = [] if gsd_available else ["GSD unavailable for this product"]

    # 6. altitude
    if meta.get("altitude_km") is None and meta.get("altitude") is None:
        errors.append("Missing altitude (spacecraft_altitude)")

    # 7. sun geometry
    sun_az = meta.get("sun_azimuth_deg") if meta.get("sun_azimuth_deg") is not None else meta.get("sun_azimuth")
    sun_el = meta.get("sun_elevation_deg") if meta.get("sun_elevation_deg") is not None else meta.get("sun_elevation")
    sol_inc = meta.get("solar_incidence_deg") if meta.get("solar_incidence_deg") is not None else meta.get("solar_incidence")
    if sun_az is None or sun_el is None or sol_inc is None:
        errors.append("Missing sun geometry (sun_azimuth, sun_elevation, or solar_incidence)")

    # 8. camera geometry
    roll = meta.get("roll_deg") if meta.get("roll_deg") is not None else meta.get("roll")
    pitch = meta.get("pitch_deg") if meta.get("pitch_deg") is not None else meta.get("pitch")
    yaw = meta.get("yaw_deg") if meta.get("yaw_deg") is not None else meta.get("yaw")
    if roll is None or pitch is None or yaw is None:
        errors.append("Missing camera geometry (roll, pitch, or yaw)")

    # 9. footprint
    fp = meta.get("footprint")
    has_valid_fp = False
    if isinstance(fp, dict):
        corners = ["upper_left", "upper_right", "lower_left", "lower_right"]
        if all(
            c in fp
            and isinstance(fp[c], (list, tuple))
            and len(fp[c]) == 2
            and all(isinstance(x, (int, float)) for x in fp[c])
            for c in corners
        ):
            has_valid_fp = True
    elif isinstance(meta.get("footprint_corners"), list) and len(meta.get("footprint_corners")) == 4:
        if all(isinstance(c, (list, tuple)) and len(c) == 2 for c in meta["footprint_corners"]):
            has_valid_fp = True

    if not has_valid_fp:
        errors.append("Missing footprint coordinates")

    # 10. projection
    if not meta.get("projection"):
        errors.append("Missing projection")

    # 11. dimensions
    dims = meta.get("dimensions", {})
    lines = dims.get("lines") if isinstance(dims, dict) else meta.get("image_height")
    samples = dims.get("samples") if isinstance(dims, dict) else meta.get("image_width")
    if lines is None or samples is None or lines == "" or samples == "":
        errors.append("Missing dimensions (lines, samples)")

    is_valid = len(errors) == 0
    return {
        "valid": is_valid,
        "validation_errors": errors,
        "invalid": errors,
        "missing_required": errors,
        "missing_optional": [] if gsd_available else ["gsd_m_per_pixel"],
        "warnings": warnings,
    }


def empty_metadata_template() -> dict:
    return {
        "logical_identifier": "",
        "title": "",
        "start_time": "",
        "stop_time": "",
        "processing_level": "",
        "mission": "",
        "spacecraft": "",
        "instrument": "",
        "target": "",
        "orbit_number": None,
        "altitude_km": None,
        "gsd_m_per_pixel": None,
        "roll_deg": None,
        "pitch_deg": None,
        "yaw_deg": None,
        "sun_azimuth_deg": None,
        "sun_elevation_deg": None,
        "solar_incidence_deg": None,
        "projection": "",
        "area": "",
        "focal_length_mm": None,
        "detector_pixel_width_um": None,
        "footprint": {
            "upper_left": [],
            "upper_right": [],
            "lower_left": [],
            "lower_right": [],
        },
        "file_name": "",
        "file_size_bytes": None,
        "dimensions": {"lines": None, "samples": None},
        "data_type": "",
        "sensor_type": "Unknown",
        "valid": False,
        "validation_errors": ["No metadata provided"],
    }


def parse_chandrayaan2_pds4_xml(data: bytes | str, name: str = "") -> dict:
    """Parse Chandrayaan-2 PDS4 XML metadata into a single canonical object structure.

    Accepts XML bytes or string from Streamlit UploadedFile (.getvalue()).
    Produces the canonical dictionary structure with validated numeric types.
    """
    try:
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="replace")
        elif isinstance(data, str):
            text = data
        else:
            text = bytes(data).decode("utf-8", errors="replace")
        root = ET.fromstring(text)
    except Exception as exc:
        return {
            "logical_identifier": "",
            "title": "",
            "start_time": "",
            "stop_time": "",
            "processing_level": "",
            "mission": "",
            "spacecraft": "",
            "instrument": "",
            "target": "",
            "orbit_number": None,
            "altitude_km": None,
            "gsd_m_per_pixel": None,
            "roll_deg": None,
            "pitch_deg": None,
            "yaw_deg": None,
            "sun_azimuth_deg": None,
            "sun_elevation_deg": None,
            "solar_incidence_deg": None,
            "projection": "",
            "area": "",
            "focal_length_mm": None,
            "detector_pixel_width_um": None,
            "footprint": {
                "upper_left": [],
                "upper_right": [],
                "lower_left": [],
                "lower_right": [],
            },
            "file_name": "",
            "file_size_bytes": None,
            "dimensions": {"lines": None, "samples": None},
            "data_type": "",
            "sensor_type": "Unknown",
            "valid": False,
            "validation_errors": [f"XML parsing failed: {str(exc)}"],
            "parse_error": "✕ Metadata parsing failed",
            "reason": str(exc),
        }

    # 1. Start and Stop Times
    start_time = _find_text(
        root,
        [
            ".//pds:Observation_Area/pds:Time_Coordinates/pds:start_date_time",
            ".//pds:start_date_time",
        ],
        ["start_date_time", "start_time", "observation_start_time"],
    )
    stop_time = _find_text(
        root,
        [
            ".//pds:Observation_Area/pds:Time_Coordinates/pds:stop_date_time",
            ".//pds:stop_date_time",
        ],
        ["stop_date_time", "stop_time", "observation_stop_time"],
    )

    # 2. Processing Level
    processing_level = _find_text(
        root,
        [
            ".//pds:Observation_Area/pds:Primary_Result_Summary/pds:processing_level",
            ".//pds:processing_level",
        ],
        ["processing_level", "processinglevel", "product_level", "productlevel", "level"],
    )

    # 3. Mission / Investigation Area
    mission = _find_text(
        root,
        [
            ".//pds:Observation_Area/pds:Investigation_Area/pds:name",
            ".//pds:Investigation_Area/pds:name",
        ],
        ["investigation_area", "mission"],
    )
    if not mission:
        for elem in root.iter():
            if _local_tag(elem).lower() == "investigation_area":
                for child in elem:
                    if _local_tag(child).lower() == "name" and child.text:
                        mission = child.text.strip()
                        break

    # 4. Spacecraft and Instrument from Observing_System_Component in PDS namespace
    spacecraft = ""
    instrument = ""
    components = root.findall(".//pds:Observing_System_Component", NS)
    if not components:
        components = [elem for elem in root.iter() if _local_tag(elem).lower() == "observing_system_component"]

    for comp in components:
        comp_type = (
            comp.findtext("pds:type", default="", namespaces=NS)
            or _find_text(comp, ["pds:type", ".//pds:type"], ["type"])
            or comp.attrib.get("type")
            or comp.attrib.get("Type")
            or ""
        ).strip().lower()

        name_val = (
            comp.findtext("pds:name", default="", namespaces=NS)
            or _find_text(comp, ["pds:name", ".//pds:name"], ["name"])
        ).strip()

        if comp_type == "spacecraft":
            spacecraft = name_val
        elif comp_type == "instrument":
            instrument = name_val

    if not spacecraft:
        spacecraft = _find_text(root, [".//pds:spacecraft"], ["spacecraft", "satellite", "spacecraft_name"])
    if not instrument:
        instrument = _find_text(root, [".//pds:instrument"], ["instrument", "sensor", "sensortype", "productsensor"])

    # Sensor normalization
    inst_lower = (instrument or "").lower()
    if "orbiter high resolution camera" in inst_lower or "ohrc" in inst_lower:
        sensor_type = "OHRC"
    elif "terrain mapping camera" in inst_lower or "tmc" in inst_lower:
        sensor_type = "TMC"
    elif "imaging infrared spectrometer" in inst_lower or "iirs" in inst_lower:
        sensor_type = "IIRS"
    elif instrument:
        sensor_type = instrument.upper()
    else:
        sensor_type = "Unknown"

    # 5. Target Identification
    target = _find_text(
        root,
        [
            ".//pds:Observation_Area/pds:Target_Identification/pds:name",
            ".//pds:Target_Identification/pds:name",
        ],
        ["target_identification", "target", "target_name"],
    )
    if not target:
        for elem in root.iter():
            if _local_tag(elem).lower() == "target_identification":
                for child in elem:
                    if _local_tag(child).lower() == "name" and child.text:
                        target = child.text.strip()
                        break

    # 6. Mission Parameters / ISDA Geometry
    orbit_number = _parse_int(
        _find_text(
            root,
            [
                ".//isda:Mission_Area/isda:Product_Parameters/isda:imaging_orbit_number",
                ".//isda:imaging_orbit_number",
                ".//pds:imaging_orbit_number",
            ],
            ["imaging_orbit_number", "orbit_number", "orbit"],
        )
    )
    altitude_km = _parse_float(
        _find_text(
            root,
            [
                ".//isda:Mission_Area/isda:Geometry_Parameters/isda:spacecraft_altitude",
                ".//isda:spacecraft_altitude",
                ".//pds:spacecraft_altitude",
            ],
            ["spacecraft_altitude", "altitude", "altitude_km"],
        )
    )
    gsd_m_per_pixel = _parse_float(
        _find_text(
            root,
            [
                ".//isda:Mission_Area/isda:Geometry_Parameters/isda:pixel_resolution",
                ".//isda:pixel_resolution",
                ".//pds:pixel_resolution",
            ],
            [
                "pixel_resolution",
                "resolution_gsd",
                "resolutiongsd",
                "ground_sample_distance",
                "ground_sampling_distance",
                "ground_resolution",
                "gsd_m_per_pixel",
                "resolution",
                "spatial_resolution",
                "gsd",
            ],
        )
    )
    if gsd_m_per_pixel is None:
        pixel_size_x = _parse_float(_find_text(root, [], ["pixel_size_x", "pixel_size_sample", "sample_spacing"]))
        pixel_size_y = _parse_float(_find_text(root, [], ["pixel_size_y", "pixel_size_line", "line_spacing"]))
        pixel_size = _parse_float(_find_text(root, [], ["pixel_size", "pixel_spacing", "sample_distance"]))
        if pixel_size_x is not None and pixel_size_y is not None:
            gsd_m_per_pixel = (pixel_size_x + pixel_size_y) / 2.0
        elif pixel_size_x is not None:
            gsd_m_per_pixel = pixel_size_x
        elif pixel_size_y is not None:
            gsd_m_per_pixel = pixel_size_y
        elif pixel_size is not None:
            gsd_m_per_pixel = pixel_size
    gsd_source = "explicit pixel size/resolution" if gsd_m_per_pixel is not None else None
    roll_deg = _parse_float(
        _find_text(
            root,
            [
                ".//isda:Mission_Area/isda:Geometry_Parameters/isda:roll",
                ".//isda:roll",
                ".//pds:roll",
            ],
            ["roll", "roll_deg"],
        )
    )
    pitch_deg = _parse_float(
        _find_text(
            root,
            [
                ".//isda:Mission_Area/isda:Geometry_Parameters/isda:pitch",
                ".//isda:pitch",
                ".//pds:pitch",
            ],
            ["pitch", "pitch_deg"],
        )
    )
    yaw_deg = _parse_float(
        _find_text(
            root,
            [
                ".//isda:Mission_Area/isda:Geometry_Parameters/isda:yaw",
                ".//isda:yaw",
                ".//pds:yaw",
            ],
            ["yaw", "yaw_deg"],
        )
    )
    sun_azimuth_deg = _parse_float(
        _find_text(
            root,
            [
                ".//isda:Mission_Area/isda:Geometry_Parameters/isda:sun_azimuth",
                ".//isda:sun_azimuth",
                ".//pds:sun_azimuth",
            ],
            ["sun_azimuth", "sunazimuth", "sun_azimuth_deg"],
        )
    )
    sun_elevation_deg = _parse_float(
        _find_text(
            root,
            [
                ".//isda:Mission_Area/isda:Geometry_Parameters/isda:sun_elevation",
                ".//isda:sun_elevation",
                ".//pds:sun_elevation",
            ],
            ["sun_elevation", "sunelevation", "sun_elevation_deg"],
        )
    )
    solar_incidence_deg = _parse_float(
        _find_text(
            root,
            [
                ".//isda:Mission_Area/isda:Geometry_Parameters/isda:solar_incidence",
                ".//isda:solar_incidence",
                ".//pds:solar_incidence",
            ],
            ["solar_incidence", "solarincidence", "solar_incidence_deg"],
        )
    )
    projection = _find_text(
        root,
        [
            ".//isda:Mission_Area/isda:Geometry_Parameters/isda:projection",
            ".//isda:projection",
            ".//pds:projection",
        ],
        ["projection", "projection_name", "map_projection", "coordinate_system"],
    )
    area = _find_text(
        root,
        [
            ".//isda:Mission_Area/isda:Geometry_Parameters/isda:area",
            ".//isda:area",
            ".//pds:area",
        ],
        ["area"],
    )
    focal_length_mm = _parse_float(
        _find_text(
            root,
            [
                ".//isda:Mission_Area/isda:Geometry_Parameters/isda:focal_length",
                ".//isda:focal_length",
                ".//pds:focal_length",
            ],
            ["focal_length", "focallength"],
        )
    )
    detector_pixel_width_um = _parse_float(
        _find_text(
            root,
            [
                ".//isda:Mission_Area/isda:Geometry_Parameters/isda:detector_pixel_width",
                ".//isda:detector_pixel_width",
                ".//pds:detector_pixel_width",
            ],
            ["detector_pixel_width", "detectorpixelwidth"],
        )
    )

    if gsd_m_per_pixel is None:
        for elem in root.iter():
            tag_name = _local_tag(elem).lower()
            if tag_name in ("comment", "description") and elem.text:
                m = re.search(r"(\d+(?:\.\d+)?)\s*(?:meter|m)\s+resolution", elem.text, re.IGNORECASE)
                if m:
                    gsd_m_per_pixel = float(m.group(1))
                    gsd_source = "extracted from product description"
                    break

    if gsd_m_per_pixel is None and altitude_km and detector_pixel_width_um and focal_length_mm:
        calc = (altitude_km * 1000.0) * (detector_pixel_width_um * 1e-6) / (focal_length_mm * 1e-3)
        if 0.01 <= calc <= 1000.0:
            gsd_m_per_pixel = round(calc, 2)
            gsd_source = "computed from focal length, detector width and altitude"

    # 7. Refined Corner Coordinates / Footprint
    ul_lat = _parse_float(_find_text(root, [".//isda:upper_left_latitude", ".//pds:upper_left_latitude"], ["upper_left_latitude", "upperleftlatitude"]))
    ul_lon = _parse_float(_find_text(root, [".//isda:upper_left_longitude", ".//pds:upper_left_longitude"], ["upper_left_longitude", "upperleftlongitude"]))
    ur_lat = _parse_float(_find_text(root, [".//isda:upper_right_latitude", ".//pds:upper_right_latitude"], ["upper_right_latitude", "upperrightlatitude"]))
    ur_lon = _parse_float(_find_text(root, [".//isda:upper_right_longitude", ".//pds:upper_right_longitude"], ["upper_right_longitude", "upperrightlongitude"]))
    ll_lat = _parse_float(_find_text(root, [".//isda:lower_left_latitude", ".//pds:lower_left_latitude"], ["lower_left_latitude", "lowerleftlatitude"]))
    ll_lon = _parse_float(_find_text(root, [".//isda:lower_left_longitude", ".//pds:lower_left_longitude"], ["lower_left_longitude", "lowerleftlongitude"]))
    lr_lat = _parse_float(_find_text(root, [".//isda:lower_right_latitude", ".//pds:lower_right_latitude"], ["lower_right_latitude", "lowerrightlatitude"]))
    lr_lon = _parse_float(_find_text(root, [".//isda:lower_right_longitude", ".//pds:lower_right_longitude"], ["lower_right_longitude", "lowerrightlongitude"]))

    # Standard PDS4/cartographic products may publish a bounding box instead
    # of refined corner coordinates.
    western_lon = _parse_float(_find_text(root, [], ["westernmost_longitude", "west_bounding_coordinate", "west_longitude", "min_longitude"]))
    eastern_lon = _parse_float(_find_text(root, [], ["easternmost_longitude", "east_bounding_coordinate", "east_longitude", "max_longitude"]))
    southern_lat = _parse_float(_find_text(root, [], ["southernmost_latitude", "south_bounding_coordinate", "south_latitude", "min_latitude"]))
    northern_lat = _parse_float(_find_text(root, [], ["northernmost_latitude", "north_bounding_coordinate", "north_latitude", "max_latitude"]))

    if any(v is None for v in [ul_lat, ul_lon, ur_lat, ur_lon, ll_lat, ll_lon, lr_lat, lr_lon]):
        corner_pairs = _extract_corners_fallback(root)
        if len(corner_pairs) == 4:
            ul_lat, ul_lon = corner_pairs[0]
            ur_lat, ur_lon = corner_pairs[1]
            ll_lat, ll_lon = corner_pairs[2]
            lr_lat, lr_lon = corner_pairs[3]

    footprint = {
        "upper_left": [ul_lat, ul_lon] if ul_lat is not None and ul_lon is not None else [],
        "upper_right": [ur_lat, ur_lon] if ur_lat is not None and ur_lon is not None else [],
        "lower_left": [ll_lat, ll_lon] if ll_lat is not None and ll_lon is not None else [],
        "lower_right": [lr_lat, lr_lon] if lr_lat is not None and lr_lon is not None else [],
    }

    # 8. File Information
    file_name = _find_text(
        root,
        [
            ".//pds:File_Area_Observational/pds:File/pds:file_name",
            ".//pds:file_name",
        ],
        ["file_name", "filename"],
    )
    file_size_bytes = _parse_int(
        _find_text(
            root,
            [
                ".//pds:File_Area_Observational/pds:File/pds:file_size",
                ".//pds:file_size",
            ],
            ["file_size", "filesize"],
        )
    )

    # 9. Array 2D Image & Dimensions
    data_type = _find_text(
        root,
        [
            ".//pds:Array_2D_Image/pds:Element_Array/pds:data_type",
            ".//pds:Element_Array/pds:data_type",
            ".//pds:data_type",
        ],
        ["data_type", "datatype"],
    )

    lines = None
    samples = None

    # Structure 1: Multiple <pds:Axis_Array> elements containing <pds:axis_name> and <pds:elements>
    axis_arrays = root.findall(".//pds:Array_2D_Image/pds:Axis_Array", NS) or root.findall(".//pds:Axis_Array", NS)
    if not axis_arrays:
        axis_arrays = [elem for elem in root.iter() if _local_tag(elem).lower() == "axis_array"]

    for axis_arr in axis_arrays:
        axis_name = (
            axis_arr.findtext("pds:axis_name", default="", namespaces=NS)
            or _find_text(axis_arr, ["pds:axis_name", ".//pds:axis_name"], ["axis_name", "axisname"])
            or axis_arr.attrib.get("axis_name", "")
            or axis_arr.attrib.get("axisName", "")
        ).strip().lower()
        elem_text = (
            axis_arr.findtext("pds:elements", default="", namespaces=NS)
            or _find_text(axis_arr, ["pds:elements", ".//pds:elements"], ["elements"])
        ).strip()
        val = _parse_int(elem_text)
        if axis_name in {"line", "lines"} and val is not None:
            lines = val
        elif axis_name in {"sample", "samples"} and val is not None:
            samples = val

    # Structure 2: Nested <pds:Axis> inside <pds:Axis_Array>
    if lines is None or samples is None:
        axis_nodes = root.findall(".//pds:Axis", NS) or [elem for elem in root.iter() if _local_tag(elem).lower() == "axis"]
        for axis in axis_nodes:
            axis_name = (
                axis.findtext("pds:axis_name", default="", namespaces=NS)
                or _find_text(axis, ["pds:axis_name", ".//pds:axis_name"], ["axis_name", "axisname"])
                or axis.attrib.get("axis_name", "")
                or axis.attrib.get("axisName", "")
                or axis.attrib.get("name", "")
            ).strip().lower()
            elem_text = (
                axis.findtext("pds:elements", default="", namespaces=NS)
                or _find_text(axis, ["pds:elements", ".//pds:elements"], ["elements"])
            ).strip()
            val = _parse_int(elem_text)
            if axis_name in {"line", "lines"} and val is not None:
                lines = val
            elif axis_name in {"sample", "samples"} and val is not None:
                samples = val

    # Fallbacks for legacy/generic XML:
    if lines is None:
        lines = _parse_int(_find_text(root, [".//pds:ImageHeight", ".//pds:image_height", ".//pds:lines", ".//pds:Line"], ["imageheight", "image_height", "height", "rows", "lines", "line"]))
    if samples is None:
        samples = _parse_int(_find_text(root, [".//pds:ImageWidth", ".//pds:image_width", ".//pds:samples", ".//pds:Sample"], ["imagewidth", "image_width", "width", "columns", "samples", "sample"]))

    if not _valid_corner_footprint(footprint) and all(
        value is not None for value in (western_lon, eastern_lon, southern_lat, northern_lat)
    ):
        footprint = {
            "upper_left": [northern_lat, western_lon],
            "upper_right": [northern_lat, eastern_lon],
            "lower_left": [southern_lat, western_lon],
            "lower_right": [southern_lat, eastern_lon],
        }

    # A derived orthophoto can omit pixel_resolution while still providing a
    # reliable map footprint. Derive metres/pixel only for projected metric
    # extents when the product explicitly provides projected bounds.
    projected_x_min = _parse_float(_find_text(root, [], ["x_min", "minimum_x", "westing_min"]))
    projected_x_max = _parse_float(_find_text(root, [], ["x_max", "maximum_x", "westing_max"]))
    projected_y_min = _parse_float(_find_text(root, [], ["y_min", "minimum_y", "southing_min"]))
    projected_y_max = _parse_float(_find_text(root, [], ["y_max", "maximum_y", "southing_max"]))
    if gsd_m_per_pixel is None and samples and lines and all(
        value is not None for value in (projected_x_min, projected_x_max, projected_y_min, projected_y_max)
    ):
        x_scale = abs(projected_x_max - projected_x_min) / samples
        y_scale = abs(projected_y_max - projected_y_min) / lines
        if x_scale > 0 and y_scale > 0:
            gsd_m_per_pixel = (x_scale + y_scale) / 2.0
            gsd_source = "derived from projected georeferenced extent and image dimensions"

    logical_identifier = _find_text(
        root,
        [
            ".//pds:Identification_Area/pds:logical_identifier",
            ".//pds:logical_identifier",
        ],
        ["logical_identifier", "identifier"],
    )
    title = _find_text(
        root,
        [
            ".//pds:Identification_Area/pds:title",
            ".//pds:title",
        ],
        ["title"],
    )

    payload = {
        "logical_identifier": logical_identifier,
        "title": title,
        "start_time": start_time,
        "stop_time": stop_time,
        "processing_level": processing_level,
        "mission": mission,
        "spacecraft": spacecraft,
        "instrument": instrument,
        "target": target,
        "orbit_number": orbit_number,
        "altitude_km": altitude_km,
        "gsd_m_per_pixel": gsd_m_per_pixel,
        "gsd_source": gsd_source,
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "yaw_deg": yaw_deg,
        "sun_azimuth_deg": sun_azimuth_deg,
        "sun_elevation_deg": sun_elevation_deg,
        "solar_incidence_deg": solar_incidence_deg,
        "projection": projection,
        "area": area,
        "focal_length_mm": focal_length_mm,
        "detector_pixel_width_um": detector_pixel_width_um,
        "footprint": footprint,
        "file_name": file_name,
        "file_size_bytes": file_size_bytes,
        "dimensions": {
            "lines": lines,
            "samples": samples,
        },
        "latitude_bounds": [southern_lat, northern_lat] if southern_lat is not None and northern_lat is not None else [],
        "longitude_bounds": [western_lon, eastern_lon] if western_lon is not None and eastern_lon is not None else [],
        "data_type": data_type,
        "sensor_type": sensor_type,
        "valid": True,
        "validation_errors": [],
    }

    # Validation
    val_status = validate_metadata(payload)
    payload["valid"] = val_status["valid"]
    payload["validation_errors"] = val_status["validation_errors"]
    payload["validation_warnings"] = val_status["warnings"]

    # Backward-compatible helper aliases
    payload["latitude"] = footprint.get("upper_left", [0.0, 0.0])[0] if footprint.get("upper_left") else None
    payload["longitude"] = footprint.get("upper_left", [0.0, 0.0])[1] if footprint.get("upper_left") else None
    payload["resolution_gsd"] = gsd_m_per_pixel
    payload["altitude"] = altitude_km
    payload["sun_azimuth"] = sun_azimuth_deg
    payload["sun_elevation"] = sun_elevation_deg
    payload["solar_incidence"] = solar_incidence_deg
    payload["roll"] = roll_deg
    payload["pitch"] = pitch_deg
    payload["yaw"] = yaw_deg
    payload["image_width"] = str(samples or "")
    payload["image_height"] = str(lines or "")
    payload["footprint_corners"] = [
        footprint.get("upper_left", []),
        footprint.get("upper_right", []),
        footprint.get("lower_left", []),
        footprint.get("lower_right", []),
    ]
    payload["latitude_bounds"] = (
        [min(c[0] for c in payload["footprint_corners"]), max(c[0] for c in payload["footprint_corners"])]
        if _valid_corner_footprint(footprint) else payload.get("latitude_bounds", [])
    )
    payload["longitude_bounds"] = (
        [min(c[1] for c in payload["footprint_corners"]), max(c[1] for c in payload["footprint_corners"])]
        if _valid_corner_footprint(footprint) else payload.get("longitude_bounds", [])
    )
    payload["acquisition_time"] = {"start": start_time, "stop": stop_time}
    payload["sun_geometry"] = {
        "azimuth": sun_azimuth_deg,
        "elevation": sun_elevation_deg,
        "incidence": solar_incidence_deg,
    }
    payload["camera_geometry"] = {"roll": roll_deg, "pitch": pitch_deg, "yaw": yaw_deg}

    return payload


def parse_metadata_xml(data: bytes | str, name: str = "") -> dict:
    """Wrapper that delegates to PDS3 or PDS4 parser based on content."""
    sample = (data[:2000].decode("ascii", errors="replace") if isinstance(data, bytes) else str(data)[:2000]).lower()
    if "pds_version_id" in sample or "record_type" in sample or "lroc" in sample or ("lines" in sample and "^image" in sample):
        return parse_lro_pds3_label(data, name)
    return parse_chandrayaan2_pds4_xml(data, name)


def merge_metadata(manual: dict, xml_meta: dict) -> dict:
    merged = {**empty_metadata_template(), **xml_meta}
    for key, value in manual.items():
        if value not in (None, ""):
            merged[key] = value
    return merged


def view_metadata_status(meta: dict) -> str:
    validation = validate_metadata(meta)
    if validation["invalid"]:
        return "✕ Invalid metadata"
    if validation["missing_optional"]:
        return "⚠ Missing optional metadata"
    return "✓ Valid metadata"



def inspect_image(data: bytes, name: str, metadata: Optional[dict] = None) -> dict:
    """Return an input inspection report using product-aware routing.

    Distinguishes standard images, GeoTIFFs, and scientific raw rasters (PDS3 LRO .IMG,
    PDS4 Chandrayaan-2 .IMG) without passing raw binaries to PIL.
    """
    return inspect_scientific_product(data, name, metadata=metadata)


def normalize_uint8(arr: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = valid & np.isfinite(arr)
    if int(valid.sum()) < 100:
        raise ValueError("Too few valid pixels after NoData masking.")
    vals = arr[valid]
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        raise ValueError("Image has insufficient contrast.")
    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    out = np.where(valid, out, 0)
    return (out * 255).astype(np.uint8), valid.astype(np.uint8) * 255


def make_representations(gray: np.ndarray, mask: np.ndarray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    local = clahe.apply(gray)

    # Illumination-tolerant structural representation.
    gx = cv2.Sobel(local, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(local, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Second-order structure is useful for crater/rim boundaries.
    lap = cv2.Laplacian(local, cv2.CV_32F, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    structure = cv2.addWeighted(mag, 0.70, lap, 0.30, 0)
    structure[mask == 0] = 0
    mag[mask == 0] = 0
    local[mask == 0] = 0
    return local, structure, mag


def load_iirs_working_image(
    data: bytes,
    name: str,
    max_side: int,
    iirs_representation: str = "automatic",
    pca_component: int = 1,
    normalization: str = "percentile",
    invalid_handling: str = "automatic",
    resolution_handling: str = "automatic",
    source_gsd: Optional[float] = None,
    reference_gsd: Optional[float] = None,
) -> Tuple["WorkingImage", dict]:
    """Load an IIRS hyperspectral input and produce a WorkingImage for spatial matching.

    Uses iirs_preprocessing to load the full spectral cube, mask invalid pixels,
    normalize spectral bands, and synthesize a 2D spatial representation (PCA PC1
    by default) which is then passed through the standard spatial enhancement
    pipeline (CLAHE, Sobel, Laplacian structure) to produce gray/structure/gradient
    arrays compatible with SIFT and LoFTR matching.

    NOTE: SIFT is applied to the derived 2D representation — it is NOT a hyperspectral
    matcher.  The representation method is explicitly recorded in the diagnostic dict.

    Returns:
        working: WorkingImage ready for the matching pipeline.
        diag:    Diagnostic dict for the IIRS Matching Representation preview panel.
    """
    # 1. Load the hyperspectral cube (bands, height, width)
    cube, meta = iirs_preprocessing.load_iirs(data, name)

    # 2. Remove invalid pixels (NoData sentinels, NaNs, extreme values)
    cube_clean, valid_mask = iirs_preprocessing.remove_invalid_pixels(cube, no_data_value=-9999.0)

    # 3. Validate cube
    validation = iirs_preprocessing.validate_iirs_cube(cube_clean)
    if not validation["valid"]:
        raise ValueError(
            "IIRS cube validation failed: " + "; ".join(validation["errors"])
        )

    # 4. Normalize
    norm_method = normalization if normalization in ("percentile", "standardization", "none") else "percentile"
    cube_norm, valid_mask = iirs_preprocessing.normalize_iirs(cube_clean, valid_mask, method=norm_method)

    # 5. Resolution-aware scale info (informational, logged into diag)
    gsd_info: dict = {}
    if source_gsd is not None and reference_gsd is not None:
        gsd_info = iirs_preprocessing.compute_resolution_scale_ratio(source_gsd, reference_gsd)

    # 6. Generate 2D spatial representation from the spectral cube
    rep_2d, rep_diag = iirs_preprocessing.generate_iirs_2d_representation(
        cube_norm, valid_mask,
        method=iirs_representation,
        pca_component=pca_component,
        apply_clahe=True,
    )

    # 7. Resize to working resolution (respecting max_side budget)
    h_orig, w_orig = rep_2d.shape[:2]
    factor = min(1.0, max_side / max(h_orig, w_orig, 1))
    oh = max(1, int(round(h_orig * factor)))
    ow = max(1, int(round(w_orig * factor)))

    rep_resized = cv2.resize(rep_2d, (ow, oh), interpolation=cv2.INTER_LINEAR)
    valid_resized = cv2.resize(
        valid_mask.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    valid_uint8 = valid_resized.astype(np.uint8) * 255

    # 8. Build illumination-tolerant spatial representations (CLAHE already applied)
    local, structure, gradient = make_representations(rep_resized, valid_uint8)

    sx = w_orig / max(ow, 1)
    sy = h_orig / max(oh, 1)
    to_original = np.array(
        [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=np.float64
    )

    info = {
        "filename": name,
        "width": w_orig,
        "height": h_orig,
        "bands": cube.shape[0],
        "dtype": str(cube.dtype),
        "format": meta.get("format", "IIRS"),
        "sensor_type": "IIRS",
        "gsd_m_per_pixel": meta.get("gsd_m_per_pixel"),
    }

    diag = {
        **rep_diag,
        "cube_bands": cube.shape[0],
        "cube_shape": (cube.shape[0], h_orig, w_orig),
        "valid_pixel_pct": round(float(np.mean(valid_mask)) * 100, 2),
        "normalization_method": norm_method,
        "representation": rep_diag.get("method_applied", iirs_representation),
        "estimated_gsd_m": meta.get("gsd_m_per_pixel"),
        "scale_ratio_info": gsd_info,
        "warnings": meta.get("warnings", []),
    }

    working = WorkingImage(rep_resized, structure, gradient, valid_uint8, to_original, info)
    return working, diag


def load_working_image(
    data: bytes,
    name: str,
    band: int,
    max_side: int,
    sensor_type: str = "",
    iirs_representation: str = "automatic",
    pca_component: int = 1,
    normalization: str = "percentile",
    invalid_handling: str = "automatic",
    resolution_handling: str = "automatic",
    source_gsd: Optional[float] = None,
    reference_gsd: Optional[float] = None,
    metadata: Optional[dict] = None,
) -> "WorkingImage":
    """Route image loading to IIRS, scientific binary, or standard pipeline.

    When sensor_type is 'IIRS', delegates to load_iirs_working_image.
    When product is a scientific binary (LRO .IMG, Chandrayaan .IMG), uses
    load_scientific_preview driven by metadata.
    Standard images (PNG, JPEG) use PIL, and GeoTIFFs use rasterio.
    """
    if _canonical_sensor(sensor_type) == "IIRS":
        working, _ = load_iirs_working_image(
            data, name, max_side,
            iirs_representation=iirs_representation,
            pca_component=pca_component,
            normalization=normalization,
            invalid_handling=invalid_handling,
            resolution_handling=resolution_handling,
            source_gsd=source_gsd,
            reference_gsd=reference_gsd,
        )
        return working

    # Product inspection
    info = inspect_image(data, name, metadata=metadata)
    if info.get("decoding_status") == "PENDING_METADATA":
        raise ValueError(
            f"Cannot load working image for '{name}': {info.get('message', 'Metadata dimensions required.')}"
        )

    width = info.get("width") or 1024
    height = info.get("height") or 1024
    factor = min(1.0, max_side / max(width, height))
    ow, oh = max(1, int(round(width * factor))), max(1, int(round(height * factor)))

    ptype = classify_product(name)

    if ptype in (ProductType.LRO_PDS3_BINARY, ProductType.CHANDRAYAAN_PDS4_BINARY, ProductType.SCIENTIFIC_BINARY):
        # Scientific binary: never send to PIL
        arr, valid, _ = load_scientific_preview(
            data, name, metadata=metadata or {}, max_side=max_side
        )
        if arr.shape != (oh, ow):
            arr = cv2.resize(arr, (ow, oh), interpolation=cv2.INTER_AREA)
            valid = cv2.resize(valid.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST) > 0
    elif is_tiff(name):
        with MemoryFile(data) as mem:
            with mem.open() as ds:
                if not 1 <= band <= ds.count:
                    raise ValueError(f"Band {band} is outside 1..{ds.count}.")
                image = ds.read(
                    band, out_shape=(oh, ow), out_dtype="float32", masked=True,
                    resampling=Resampling.average,
                )
                valid = ~np.ma.getmaskarray(image)
                arr = image.filled(np.nan).astype(np.float32)
    else:
        # Standard image (PNG/JPEG/WebP)
        with Image.open(io.BytesIO(data)) as im:
            rgba = im.convert("RGBA").resize((ow, oh), Image.Resampling.LANCZOS)
            valid = np.asarray(rgba.getchannel("A")) > 0
            arr = np.asarray(rgba.convert("L"), dtype=np.float32)

    gray, mask = normalize_uint8(arr, valid)
    local, structure, gradient = make_representations(gray, mask)
    sx, sy = width / ow, height / oh
    to_original = np.array(
        [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=np.float64
    )
    return WorkingImage(gray, structure, gradient, mask, to_original, info)



def eroded_mask(mask: np.ndarray) -> np.ndarray:
    return cv2.erode(mask, np.ones((7, 7), np.uint8))


def ratio_filter(knn, threshold):
    accepted = {}
    for pair in knn:
        if len(pair) != 2:
            continue
        a, b = pair
        ratio = float(a.distance) / max(float(b.distance), 1e-12)
        if ratio < threshold:
            accepted[a.queryIdx] = (a, ratio)
    return accepted


def sift_run(a, b, amask, bmask, nfeatures, ratio_threshold):
    sift = cv2.SIFT_create(
        nfeatures=int(nfeatures), contrastThreshold=0.012, edgeThreshold=10, sigma=1.6
    )
    kpa, da = sift.detectAndCompute(a, eroded_mask(amask))
    kpb, db = sift.detectAndCompute(b, eroded_mask(bmask))
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    fwd = ratio_filter(matcher.knnMatch(da, db, k=2), ratio_threshold)
    rev = ratio_filter(matcher.knnMatch(db, da, k=2), ratio_threshold)
    out = []
    for qi, (m, ratio) in fwd.items():
        r = rev.get(m.trainIdx)
        if r is not None and r[0].trainIdx == qi:
            out.append((kpa[m.queryIdx].pt, kpb[m.trainIdx].pt, ratio))
    out.sort(key=lambda x: x[2])
    return out


def akaze_run(a, b, amask, bmask, ratio_threshold):
    akaze = cv2.AKAZE_create()
    kpa, da = akaze.detectAndCompute(a, eroded_mask(amask))
    kpb, db = akaze.detectAndCompute(b, eroded_mask(bmask))
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    fwd = ratio_filter(matcher.knnMatch(da, db, k=2), ratio_threshold)
    rev = ratio_filter(matcher.knnMatch(db, da, k=2), ratio_threshold)
    out = []
    for qi, (m, ratio) in fwd.items():
        r = rev.get(m.trainIdx)
        if r is not None and r[0].trainIdx == qi:
            out.append((kpa[m.queryIdx].pt, kpb[m.trainIdx].pt, ratio))
    out.sort(key=lambda x: x[2])
    return out


def grid_key(p, shape, n):
    h, w = shape[:2]
    x, y = p
    c = min(n - 1, max(0, int(x * n / max(w, 1))))
    r = min(n - 1, max(0, int(y * n / max(h, 1))))
    return r, c


def fuse_and_select(runs, source_shape, reference_shape, grid_size, cell_limit):
    # Determine if matching runs are LoFTR-based (higher confidence is better) or SIFT (lower ratio is better)
    is_loftr_run = any("LoFTR" in m for m, _ in runs)
    pool = {}
    for method, rows in runs:
        for s, r, ratio in rows:
            key = (round(s[0], 1), round(s[1], 1), round(r[0], 1), round(r[1], 1))
            if is_loftr_run or "LoFTR" in method:
                # Confidence score in [0.0, 1.0]: higher is better
                score = float(ratio)
                if key not in pool or score > pool[key][2]:
                    pool[key] = (s, r, score, method)
            else:
                # Ratio score: lower is better; small bonus for structural representations
                score = float(ratio) - (0.035 if method == "structural" else 0.0)
                if key not in pool or score < pool[key][2]:
                    pool[key] = (s, r, score, method)

    if is_loftr_run:
        rows = sorted(pool.values(), key=lambda z: -z[2])
    else:
        rows = sorted(pool.values(), key=lambda z: z[2])

    selected = []
    scount, rcount = {}, {}
    for s, r, score, method in rows:
        sk = grid_key(s, source_shape, grid_size)
        rk = grid_key(r, reference_shape, grid_size)
        if scount.get(sk, 0) >= cell_limit or rcount.get(rk, 0) >= cell_limit:
            continue
        selected.append((s, r, score, method))
        scount[sk] = scount.get(sk, 0) + 1
        rcount[rk] = rcount.get(rk, 0) + 1

    if len(selected) < 8:
        raise ValueError(f"Only {len(selected)} fused correspondences survived. Need at least 8.")

    return (
        np.asarray([x[0] for x in selected], np.float64),
        np.asarray([x[1] for x in selected], np.float64),
        np.asarray([x[2] for x in selected], np.float64),
        np.asarray([x[3] for x in selected]),
    )


def estimate_geometric_model(
    ps: np.ndarray,
    pr: np.ndarray,
    model: str = "Homography",
    verifier: str = "MAGSAC++",
    threshold: float = 2.5,
) -> Tuple[Optional[np.ndarray], np.ndarray, dict]:
    """
    Robust geometric transformation estimation.
    Primary method: MAGSAC++ (cv2.USAC_MAGSAC)
    Baseline / Explicit Fallback: RANSAC (cv2.RANSAC)
    Supports: Homography (3x3) and Affine (2x3 -> 3x3).
    """
    min_pts = 4 if model == "Homography" else 3
    if len(ps) < min_pts or len(pr) < min_pts:
        raise ValueError(f"Insufficient correspondences for {model} estimation. Found {len(ps)}, need at least {min_pts}.")

    use_magsac = ("MAGSAC" in verifier)
    actual_verifier = "MAGSAC++" if use_magsac else "RANSAC"
    fallback_used = False
    fallback_reason = None
    H = None
    mask = None

    if model == "Affine":
        if use_magsac:
            try:
                magsac_method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
                a, mask = cv2.estimateAffine2D(
                    ps, pr, method=magsac_method, ransacReprojThreshold=threshold,
                    maxIters=30000, confidence=0.999, refineIters=100,
                )
                if a is not None and mask is not None and np.isfinite(a).all():
                    H = np.vstack([a, [0, 0, 1]])
                    actual_verifier = "MAGSAC++"
                else:
                    raise RuntimeError("MAGSAC++ returned empty or non-finite affine model.")
            except Exception as exc:
                logger.warning(f"MAGSAC++ affine estimation failed ({exc}); explicit RANSAC fallback used.")
                fallback_used = True
                fallback_reason = str(exc)
                a, mask = cv2.estimateAffine2D(
                    ps, pr, method=cv2.RANSAC, ransacReprojThreshold=threshold,
                    maxIters=30000, confidence=0.999, refineIters=100,
                )
                H = None if a is None else np.vstack([a, [0, 0, 1]])
                actual_verifier = "RANSAC fallback"
        else:
            actual_verifier = "RANSAC"
            a, mask = cv2.estimateAffine2D(
                ps, pr, method=cv2.RANSAC, ransacReprojThreshold=threshold,
                maxIters=30000, confidence=0.999, refineIters=100,
            )
            H = None if a is None else np.vstack([a, [0, 0, 1]])
    else:  # Homography
        if use_magsac:
            try:
                magsac_method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
                H, mask = cv2.findHomography(
                    ps, pr, magsac_method, threshold, maxIters=30000, confidence=0.999
                )
                if H is not None and mask is not None and np.isfinite(H).all():
                    actual_verifier = "MAGSAC++"
                else:
                    raise RuntimeError("MAGSAC++ returned empty or non-finite homography model.")
            except Exception as exc:
                logger.warning(f"MAGSAC++ homography estimation failed ({exc}); explicit RANSAC fallback used.")
                fallback_used = True
                fallback_reason = str(exc)
                H, mask = cv2.findHomography(
                    ps, pr, cv2.RANSAC, threshold, maxIters=30000, confidence=0.999
                )
                actual_verifier = "RANSAC fallback"
        else:
            actual_verifier = "RANSAC"
            H, mask = cv2.findHomography(
                ps, pr, cv2.RANSAC, threshold, maxIters=30000, confidence=0.999
            )

    if H is None or mask is None or not np.isfinite(H).all():
        raise ValueError(f"Robust geometric model could not be estimated using {actual_verifier}.")
    if np.linalg.cond(H) > 1e12:
        raise ValueError(f"Estimated transformation using {actual_verifier} is numerically unstable.")

    mask_bool = mask.ravel().astype(bool)
    inliers = int(mask_bool.sum())
    outliers = int((~mask_bool).sum())
    inlier_ratio = float(mask_bool.mean()) if len(mask_bool) > 0 else 0.0

    info = {
        "verifier_method": actual_verifier,
        "primary_verifier": verifier,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "model": model,
        "threshold": threshold,
        "inlier_count": inliers,
        "outlier_count": outliers,
        "inlier_ratio": inlier_ratio,
    }
    return H, mask_bool, info


def estimate_model(ps, pr, model="Homography", threshold=2.5, verifier="MAGSAC++"):
    """
    Backwards-compatible wrapper returning (H, inlier_mask).
    Delegates to estimate_geometric_model with MAGSAC++ as primary verifier.
    """
    H, mask, _ = estimate_geometric_model(ps, pr, model=model, verifier=verifier, threshold=threshold)
    return H, mask


def project(points, H):
    out = cv2.perspectiveTransform(np.asarray(points, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)
    if not np.isfinite(out).all():
        raise ValueError("Invalid projected coordinates.")
    return out


def spatial_holdout(points, grid=5, fraction=0.25):
    x, y = points[:, 0], points[:, 1]
    xn = (x - x.min()) / max(np.ptp(x), 1e-9)
    yn = (y - y.min()) / max(np.ptp(y), 1e-9)
    cells = np.minimum(grid - 1, (xn * grid).astype(int)) + grid * np.minimum(grid - 1, (yn * grid).astype(int))
    unique = np.unique(cells)
    n = max(1, int(round(len(unique) * fraction)))
    hold = set(unique[-n:].tolist())
    val = np.array([c in hold for c in cells])
    return np.where(~val)[0], np.where(val)[0]


def refine_subpixel(source_gray, reference_gray, ps, pr, window=11):
    ps2 = np.asarray(ps, np.float32).reshape(-1, 1, 2).copy()
    pr2 = np.asarray(pr, np.float32).reshape(-1, 1, 2).copy()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    half = window // 2
    if min(source_gray.shape[:2]) < 2 * half + 3:
        return ps.copy(), pr.copy()
    try:
        cv2.cornerSubPix(source_gray, ps2, (half, half), (-1, -1), criteria)
        cv2.cornerSubPix(reference_gray, pr2, (half, half), (-1, -1), criteria)
    except cv2.error:
        return ps.copy(), pr.copy()
    return ps2.reshape(-1, 2).astype(np.float64), pr2.reshape(-1, 2).astype(np.float64)


def poly_design(points, width, height):
    x = points[:, 0] / max(width, 1)
    y = points[:, 1] / max(height, 1)
    return np.column_stack([np.ones(len(points)), x, y, x*x, x*y, y*y])


def fit_residual_field(points, residual, shape, ridge=0.05):
    h, w = shape[:2]
    A = poly_design(points, w, h)
    if len(A) < 12:
        return None
    reg = np.diag([0, 0, 0, ridge, ridge, ridge])
    try:
        return (
            np.linalg.solve(A.T @ A + reg, A.T @ residual[:, 0]),
            np.linalg.solve(A.T @ A + reg, A.T @ residual[:, 1]),
        )
    except np.linalg.LinAlgError:
        return None


def evaluate_residual_field(points, coeffs, shape):
    h, w = shape[:2]
    A = poly_design(points, w, h)
    return np.column_stack([A @ coeffs[0], A @ coeffs[1]])


def smooth_corrected_warp(source, H, coeffs, out_shape):
    h, w = out_shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ref = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    delta = evaluate_residual_field(ref, coeffs, out_shape)
    invH = np.linalg.inv(H)
    src = project(ref - delta, invH)
    mx = src[:, 0].reshape(h, w).astype(np.float32)
    my = src[:, 1].reshape(h, w).astype(np.float32)
    return cv2.remap(source, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def create_overlay(reference, registered, overlap):
    blended = cv2.addWeighted(reference, 0.5, registered, 0.5, 0)
    out = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    out[overlap] = cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)[overlap]
    return out


def draw_matches(source, reference, ps, pr, inlier_mask):
    s_gray = source if source.ndim == 2 else cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    r_gray = reference if reference.ndim == 2 else cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    h1, w1 = s_gray.shape[:2]
    h2, w2 = r_gray.shape[:2]
    h_max = max(h1, h2)

    canvas_gray = np.zeros((h_max, w1 + w2), dtype=np.uint8)
    canvas_gray[:h1, :w1] = s_gray
    canvas_gray[:h2, w1:w1 + w2] = r_gray

    canvas = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)
    off = w1
    for i, (a, b) in enumerate(zip(ps, pr)):
        good = bool(inlier_mask[i])
        c = (0, 255, 0) if good else (90, 90, 90)
        rr = 3 if good else 2
        pa = tuple(np.rint(a).astype(int))
        pb = (int(round(b[0] + off)), int(round(b[1])))
        cv2.circle(canvas, pa, rr, c, -1)
        cv2.circle(canvas, pb, rr, c, -1)
        if good:
            cv2.line(canvas, pa, pb, c, 1)
    return canvas


def png_bytes(img):
    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("PNG encoding failed.")
    return enc.tobytes()


def make_display_preview(img: np.ndarray, max_dim: int = 800, is_bgr: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Downscale and contrast-normalize an image for UI DISPLAY ONLY.
    Does NOT modify the underlying scientific array used for calculations.
    Returns (display_img_rgb, diagnostics_dict).
    """
    if img is None:
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        return blank, {"shape": [0, 0], "dtype": "none", "min": 0.0, "max": 0.0, "mean": 0.0}

    arr = np.asarray(img)
    stats = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.nanmin(arr)) if arr.size else 0.0,
        "max": float(np.nanmax(arr)) if arr.size else 0.0,
        "mean": float(np.nanmean(arr)) if arr.size else 0.0,
    }

    disp = arr.copy()
    if np.isnan(disp).any():
        disp = np.nan_to_num(disp, nan=0.0)

    if disp.ndim == 2:
        if disp.dtype != np.uint8:
            p1, p99 = np.percentile(disp, (1, 99))
            if p99 > p1:
                disp = np.clip((disp - p1) / (p99 - p1) * 255.0, 0, 255).astype(np.uint8)
            else:
                disp = np.clip(disp, 0, 255).astype(np.uint8)
        else:
            if disp.max() > 0 and disp.mean() < 15:
                p99 = np.percentile(disp[disp > 0], 99) if (disp > 0).any() else 255
                if p99 > 0:
                    disp = np.clip(disp.astype(np.float32) / p99 * 255.0, 0, 255).astype(np.uint8)
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
    elif disp.ndim == 3:
        if is_bgr:
            disp_rgb = cv2.cvtColor(disp.astype(np.uint8), cv2.COLOR_BGR2RGB)
        else:
            disp_rgb = disp.astype(np.uint8)
        if disp_rgb.max() > 0 and disp_rgb.mean() < 15:
            p99 = np.percentile(disp_rgb[disp_rgb > 0], 99) if (disp_rgb > 0).any() else 255
            if p99 > 0:
                disp_rgb = np.clip(disp_rgb.astype(np.float32) / p99 * 255.0, 0, 255).astype(np.uint8)
    else:
        disp_rgb = np.zeros((200, 200, 3), dtype=np.uint8)

    h, w = disp_rgb.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        disp_rgb = cv2.resize(disp_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    return disp_rgb, stats


def classify_sensor_path(source_sensor: str, reference_sensor: str) -> str:
    return "same" if source_sensor == reference_sensor else "different"


def analyze_metadata(source_info: dict, reference_info: dict, source_sensor: str, reference_sensor: str,
                      source_meta: dict | None = None, reference_meta: dict | None = None) -> dict:
    source_px = source_info.get("width", 0) * source_info.get("height", 0)
    reference_px = reference_info.get("width", 0) * reference_info.get("height", 0)
    overlap_estimate = min(source_px, reference_px) / max(source_px, reference_px, 1)
    overlap_label = "plausible" if overlap_estimate > 0.15 else "low-overlap"

    source_meta = source_meta or {}
    reference_meta = reference_meta or {}

    def get_lat_lon(meta):
        fp = meta.get("footprint")
        if isinstance(fp, dict) and "upper_left" in fp and isinstance(fp["upper_left"], (list, tuple)) and len(fp["upper_left"]) == 2:
            return fp["upper_left"][0], fp["upper_left"][1]
        lat = meta.get("latitude")
        lon = meta.get("longitude")
        if lat is not None and lon is not None:
            try:
                return float(lat), float(lon)
            except Exception:
                pass
        return None, None

    lat, lon = get_lat_lon(source_meta)
    ref_lat, ref_lon = get_lat_lon(reference_meta)
    overlap_from_meta = "Estimated" if lat is not None and lon is not None and ref_lat is not None and ref_lon is not None else "Unavailable"

    def get_val(meta, *keys, default="Not provided"):
        for k in keys:
            if k in meta and meta[k] is not None and meta[k] != "":
                return meta[k]
        return default

    return {
        "source_sensor": source_sensor,
        "reference_sensor": reference_sensor,
        "sensor_path": classify_sensor_path(source_sensor, reference_sensor),
        "latitude_longitude": {
            "source": f"{lat} / {lon}" if lat is not None and lon is not None else "Not provided",
            "reference": f"{ref_lat} / {ref_lon}" if ref_lat is not None and ref_lon is not None else "Not provided",
        },
        "resolution_scale": {
            "source": f"{source_info.get('width', '-')} × {source_info.get('height', '-')} px",
            "reference": f"{reference_info.get('width', '-')} × {reference_info.get('height', '-')} px",
        },
        "sun_geometry": {
            "source": {
                "sun_azimuth": get_val(source_meta, "sun_azimuth_deg", "sun_azimuth"),
                "sun_elevation": get_val(source_meta, "sun_elevation_deg", "sun_elevation"),
                "solar_incidence": get_val(source_meta, "solar_incidence_deg", "solar_incidence"),
            },
            "reference": {
                "sun_azimuth": get_val(reference_meta, "sun_azimuth_deg", "sun_azimuth"),
                "sun_elevation": get_val(reference_meta, "sun_elevation_deg", "sun_elevation"),
                "solar_incidence": get_val(reference_meta, "solar_incidence_deg", "solar_incidence"),
            },
        },
        "camera_geometry": {
            "source": {
                "roll": get_val(source_meta, "roll_deg", "roll"),
                "pitch": get_val(source_meta, "pitch_deg", "pitch"),
                "yaw": get_val(source_meta, "yaw_deg", "yaw"),
                "sensor_type": get_val(source_meta, "sensor_type", default=source_sensor),
                "processing_level": get_val(source_meta, "processing_level"),
            },
            "reference": {
                "roll": get_val(reference_meta, "roll_deg", "roll"),
                "pitch": get_val(reference_meta, "pitch_deg", "pitch"),
                "yaw": get_val(reference_meta, "yaw_deg", "yaw"),
                "sensor_type": get_val(reference_meta, "sensor_type", default=reference_sensor),
                "processing_level": get_val(reference_meta, "processing_level"),
            },
        },
        "footprint_overlap": {
            "estimate": f"{overlap_estimate:.3f}",
            "label": overlap_label,
            "method": "image-dimension and raster footprint heuristic",
            "metadata_overlap": overlap_from_meta,
        },
    }


def make_stage(stage_id, name, status, pipeline_input, method, output, parameters, result_metric, processing_time, error=None):
    return {
        "stage_id": stage_id,
        "name": name,
        "status": status,
        "input": pipeline_input,
        "processing_method": method,
        "output": output,
        "important_parameters": parameters,
        "result_metric": result_metric,
        "processing_time_seconds": processing_time,
        "error_message": error,
    }


def run_pipeline(source_file, reference_file, source_sensor, reference_sensor,
                 source_band=1, reference_band=1, max_side=2048, feature_count=12000,
                 ratio_threshold=0.78, grid_size=10, cell_limit=20, model="Homography",
                 ransac_threshold=2.5, subpixel=True, residual_correction=True,
                 source_metadata=None, reference_metadata=None,
                 geometric_verifier="MAGSAC++", loftr_confidence_threshold=0.35,
                 iirs_representation="automatic", pca_component=1,
                 iirs_normalization="percentile", iirs_invalid_handling="automatic",
                 iirs_resolution_handling="automatic"):

    start_total = time.perf_counter()
    stages = []

    # Stage 01 - Input / metadata check
    start = time.perf_counter()
    try:
        si = inspect_image(source_file.getvalue(), source_file.name)
        ri = inspect_image(reference_file.getvalue(), reference_file.name)
        input_status = "success"
        input_error = None
    except Exception as exc:
        input_status = "failed"
        input_error = str(exc)
        si, ri = {}, {}
    processing_time = time.perf_counter() - start
    stages.append(make_stage("01", "Input", input_status,
                           {"source": source_file.name, "reference": reference_file.name},
                           "Image upload and metadata inspection", "Loaded images",
                           {"source_sensor": source_sensor, "reference_sensor": reference_sensor,
                            "source_band": source_band, "reference_band": reference_band,
                            "max_side": max_side},
                           {"source_info": si, "reference_info": ri}, processing_time,
                           input_error))

    if input_status == "failed":
        raise ValueError(f"Input inspection failed: {input_error}")

    # Stage 02 - Metadata Analysis
    start = time.perf_counter()
    try:
        metadata = analyze_metadata(si, ri, source_sensor, reference_sensor,
                                    source_metadata or {}, reference_metadata or {})
        metadata_status = "success"
        metadata_error = None
    except Exception as exc:
        metadata_status = "failed"
        metadata_error = str(exc)
        metadata = {}
    processing_time = time.perf_counter() - start
    stages.append(make_stage("02", "Metadata", metadata_status,
                           {"source": si, "reference": ri},
                           "Latitude/Longitude, resolution, sun/camera geometry and footprint/overlap estimate",
                           "Metadata report", {"sensor_path": classify_sensor_path(source_sensor, reference_sensor)},
                           metadata, processing_time, metadata_error))

    overlap = geographic_overlap(source_metadata or {}, reference_metadata or {})
    if overlap["available"] and not overlap["overlap"]:
        raise ValueError(
            "No meaningful geographic overlap detected between source and reference footprints. "
            "Cross-sensor correspondence is not scientifically meaningful for this image pair."
        )

    # Stage 03 - Preprocessing
    start = time.perf_counter()
    # Extract GSD from metadata for resolution-aware scale ratio (used by IIRS preprocessing)
    src_gsd = (source_metadata or {}).get("gsd_m_per_pixel") or (source_metadata or {}).get("resolution_gsd")
    ref_gsd = (reference_metadata or {}).get("gsd_m_per_pixel") or (reference_metadata or {}).get("resolution_gsd")
    # Track IIRS preprocessing diagnostics for the UI preview panel
    _iirs_src_diag: dict = {}
    _iirs_ref_diag: dict = {}
    try:
        if _canonical_sensor(source_sensor) == "IIRS":
            src, _iirs_src_diag = load_iirs_working_image(
                source_file.getvalue(), source_file.name, max_side,
                iirs_representation=iirs_representation,
                pca_component=pca_component,
                normalization=iirs_normalization,
                invalid_handling=iirs_invalid_handling,
                resolution_handling=iirs_resolution_handling,
                source_gsd=src_gsd,
                reference_gsd=ref_gsd,
            )
        else:
            src = load_working_image(
                source_file.getvalue(), source_file.name, source_band, max_side,
                sensor_type=source_sensor,
                metadata=source_metadata,
            )
        if _canonical_sensor(reference_sensor) == "IIRS":
            ref, _iirs_ref_diag = load_iirs_working_image(
                reference_file.getvalue(), reference_file.name, max_side,
                iirs_representation=iirs_representation,
                pca_component=pca_component,
                normalization=iirs_normalization,
                invalid_handling=iirs_invalid_handling,
                resolution_handling=iirs_resolution_handling,
                source_gsd=ref_gsd,
                reference_gsd=src_gsd,
            )
        else:
            ref = load_working_image(
                reference_file.getvalue(), reference_file.name, reference_band, max_side,
                sensor_type=reference_sensor,
                metadata=reference_metadata,
            )
        preprocess_status = "success"
        preprocess_error = None
    except Exception as exc:
        preprocess_status = "failed"
        preprocess_error = str(exc)
        src = None
        ref = None

    processing_time = time.perf_counter() - start
    stages.append(make_stage("03", "Preprocessing", preprocess_status,
                           {"source": source_file.name, "reference": reference_file.name},
                           "Normalization, masking, ROI resize, CLAHE, grayscale preprocessing",
                           "Preprocessed source/reference working images",
                           {"source_band": source_band, "reference_band": reference_band,
                            "max_side": max_side, "normalize": "percentile clipping",
                            "clahe": "tileGridSize=(8,8), clipLimit=2.0"},
                           {"source_shape": src.gray.shape if src else None,
                            "reference_shape": ref.gray.shape if ref else None},
                           processing_time, preprocess_error))

    if preprocess_status == "failed":
        raise ValueError(f"Preprocessing failed: {preprocess_error}")

    # Stage 04 - Multi-scale / Gaussian Pyramid
    start = time.perf_counter()
    try:
        src_pyramid = [src.gray]
        ref_pyramid = [ref.gray]
        for level in range(1, 3):
            src_pyramid.append(cv2.pyrDown(src_pyramid[-1]))
            ref_pyramid.append(cv2.pyrDown(ref_pyramid[-1]))
        pyramid_status = "success"
        pyramid_error = None
    except Exception as exc:
        pyramid_status = "failed"
        pyramid_error = str(exc)
        src_pyramid = []
        ref_pyramid = []
    processing_time = time.perf_counter() - start
    stages.append(make_stage("04", "Multi-Scale", pyramid_status,
                           {"source_shape": src.gray.shape, "reference_shape": ref.gray.shape},
                           "Gaussian image pyramid levels",
                           "Pyramid levels available",
                           {"levels": len(src_pyramid), "downsample_method": "cv2.pyrDown"},
                           {"source_levels": len(src_pyramid), "reference_levels": len(ref_pyramid)},
                           processing_time, pyramid_error))

    # Stage 05 - Sensor check and feature path
    start = time.perf_counter()
    sensor_path = classify_sensor_path(source_sensor, reference_sensor)
    loftr_avail, loftr_err = loftr_matcher.is_loftr_available()
    if sensor_path == "same":
        matcher_method = f"SAME SENSOR ({source_sensor} ↔ {reference_sensor}) → SIFT multi-representation pipeline"
        route_status = "success"
        route_error = None
    else:
        if loftr_avail:
            matcher_method = f"DIFFERENT SENSOR ({source_sensor} ↔ {reference_sensor}) → Genuine LoFTR cross-modal pipeline"
            route_status = "success"
            route_error = None
        else:
            matcher_method = f"DIFFERENT SENSOR ({source_sensor} ↔ {reference_sensor}) → LoFTR requested but unavailable"
            route_status = "failed"
            route_error = f"LoFTR dependency missing: {loftr_err}"
    processing_time = time.perf_counter() - start
    stages.append(make_stage("05", "Sensor Type", route_status,
                           {"source_sensor": source_sensor, "reference_sensor": reference_sensor},
                           matcher_method,
                           "Sensor route selected",
                           {"route": sensor_path, "feature_method": "SIFT" if sensor_path == "same" else "LoFTR"},
                           {"sensor_path": sensor_path, "classical_path": sensor_path == "same", "loftr_available": loftr_avail},
                           processing_time, route_error))

    # Stage 06 - Multi-representation matching, routed by sensor type comparison
    # Routing rule: source_sensor == reference_sensor → SIFT (same-sensor)
    #               source_sensor != reference_sensor → LoFTR (cross-sensor)
    # Note on honesty: SIFT+AKAZE-fallback is never claimed as LoFTR.
    # When LoFTR is NOT available in this workspace, LoFTR is NOT claimed.
    start = time.perf_counter()
    actual_matcher = "none"
    matcher_note = ""
    loftr_metrics = {}
    try:
        if sensor_path == "same":
            # SAME-SENSOR BRANCH → SIFT on multiple representations
            # Gaussian pyramid level 1 (built in Stage 04) is consumed here for multi-scale robustness.
            src_half = src_pyramid[1] if len(src_pyramid) > 1 else src.gray
            ref_half = ref_pyramid[1] if len(ref_pyramid) > 1 else ref.gray
            src_mask_half = cv2.resize(src.mask, (src_half.shape[1], src_half.shape[0]), interpolation=cv2.INTER_NEAREST)
            ref_mask_half = cv2.resize(ref.mask, (ref_half.shape[1], ref_half.shape[0]), interpolation=cv2.INTER_NEAREST)
            pyr1_raw = sift_run(src_half, ref_half, src_mask_half, ref_mask_half, max(500, feature_count // 2), ratio_threshold)
            # Scale pyramid-1 coordinates back to full working resolution
            sx = src.gray.shape[1] / max(src_half.shape[1], 1)
            sy = src.gray.shape[0] / max(src_half.shape[0], 1)
            rx = ref.gray.shape[1] / max(ref_half.shape[1], 1)
            ry = ref.gray.shape[0] / max(ref_half.shape[0], 1)
            pyr1_scaled = [((s[0] * sx, s[1] * sy), (r[0] * rx, r[1] * ry), ratio) for s, r, ratio in pyr1_raw]
            runs = [
                ("SIFT-intensity",  sift_run(src.gray,      ref.gray,      src.mask, ref.mask, feature_count, ratio_threshold)),
                ("SIFT-structure",  sift_run(src.structure,  ref.structure,  src.mask, ref.mask, feature_count, ratio_threshold)),
                ("SIFT-gradient",   sift_run(src.gradient,   ref.gradient,   src.mask, ref.mask, feature_count, ratio_threshold)),
                ("SIFT-pyramid1",   pyr1_scaled),
            ]
            actual_matcher = "SIFT"
            matcher_note = "SIFT on intensity / structure / gradient + Gaussian pyramid level 1 (half-res, coords rescaled)"
        else:
            # CROSS-SENSOR BRANCH → Genuine LoFTR inference (kornia.feature.LoFTR)
            if not loftr_avail:
                raise RuntimeError(f"LoFTR dependency unavailable: {loftr_err}")

            loftr_out = loftr_matcher.loftr_match(
                source_gray=src.gray,
                reference_gray=ref.gray,
                source_mask=src.mask,
                reference_mask=ref.mask,
                min_confidence=float(loftr_confidence_threshold),
                max_side=min(max_side, 1024),
            )
            loftr_metrics = {
                "raw_matches": loftr_out["raw_matches"],
                "filtered_matches": loftr_out["filtered_matches"],
                "min_confidence": loftr_out["min_confidence"],
                "device": loftr_out["device"],
                "working_source_shape": loftr_out["working_source_shape"],
                "working_reference_shape": loftr_out["working_reference_shape"],
                "source_scale": loftr_out["source_scale"],
                "reference_scale": loftr_out["reference_scale"],
            }
            if loftr_out["filtered_matches"] < 4:
                raise ValueError(
                    f"LoFTR produced insufficient confident matches ({loftr_out['filtered_matches']}) "
                    f"at confidence threshold {loftr_confidence_threshold}. Check image contrast/overlap."
                )
            runs = [
                ("LoFTR-dense", loftr_out["correspondences"]),
            ]
            actual_matcher = "LoFTR"
            matcher_note = (
                f"LoFTR deep matching ({loftr_out['device'].upper()}): {loftr_out['raw_matches']} raw matches → "
                f"{loftr_out['filtered_matches']} confidence-filtered (threshold={loftr_confidence_threshold:.2f})"
            )

        if not runs:
            raise ValueError("No feature matching runs started.")
        match_status = "success"
        match_error = None
    except Exception as exc:
        match_status = "failed"
        match_error = str(exc)
        runs = []
        actual_matcher = "none"
        matcher_note = f"Matching failed: {exc}"
    processing_time = time.perf_counter() - start
    stages.append(make_stage("06", "Features / Matching", match_status,
                           {"source_shape": src.gray.shape, "reference_shape": ref.gray.shape,
                            "ratio_threshold": ratio_threshold, "feature_count": feature_count,
                            "loftr_confidence": loftr_confidence_threshold},
                           matcher_note,
                           "Feature correspondences",
                           {"runs": [name for name, rows in runs], "feature_detector": actual_matcher,
                            "descriptor_ratio": ratio_threshold, "loftr_metrics": loftr_metrics},
                           {"matching_runs": len(runs), "match_count_estimate": sum(len(rows) for _, rows in runs),
                            "actual_matcher": actual_matcher},
                           processing_time, match_error))

    if match_status == "failed":
        raise ValueError(f"Feature matching failed: {match_error}")

    # Stage 07 - Confidence filtering
    start = time.perf_counter()
    try:
        ps, pr, scores, methods = fuse_and_select(runs, src.gray.shape, ref.gray.shape, grid_size, cell_limit)
        confidence_status = "success"
        confidence_error = None
    except Exception as exc:
        confidence_status = "failed"
        confidence_error = str(exc)
        ps = np.empty((0, 2)); pr = np.empty((0, 2)); scores = np.array([]); methods = np.array([])
    processing_time = time.perf_counter() - start
    stages.append(make_stage("07", "Confidence Filtering", confidence_status,
                           {"grid_size": grid_size, "cell_limit": cell_limit, "ratio_threshold": ratio_threshold},
                           "Confidence thresholding / reciprocal matching + spatially balanced fusion",
                           "Accepted fused correspondences",
                           {"feature_count": feature_count, "grid_size": grid_size, "cell_limit": cell_limit},
                           {"fused_match_count": int(len(ps)), "correspondence_ratio": float(len(ps) / max(sum(len(rows) for _, rows in runs),1))},
                           processing_time, confidence_error))

    if confidence_status == "failed":
        raise ValueError(f"Confidence filtering failed: {confidence_error}")

    # Stage 08 - Metadata / spatial consistency
    start = time.perf_counter()
    try:
        # geographic plausibility is constrained by metadata and overlap estimate, not exact latitude longitude alignment.
        spatial_consistency = metadata["footprint_overlap"]["label"]
        consistency_status = "success" if spatial_consistency != "low-overlap" else "warning"
        consistency_error = None if consistency_status == "success" else "Estimated footprint overlap is too low for reliable registration."
    except Exception as exc:
        consistency_status = "failed"
        consistency_error = str(exc)
    processing_time = time.perf_counter() - start
    stages.append(make_stage("08", "Spatial Consistency", consistency_status,
                           metadata,
                           "Geographic plausibility and overlap consistency gate",
                           "Metadata overlap consistency",
                           {"source_info": si, "reference_info": ri},
                           {"footprint_overlap_label": spatial_consistency},
                           processing_time, consistency_error))

    if consistency_status == "failed":
        raise ValueError(f"Metadata consistency failed: {consistency_error}")

    # Stage 09 - Geometry verification (MAGSAC++ primary, RANSAC baseline/fallback)
    start = time.perf_counter()
    geom_info = {}
    try:
        train_idx, val_idx = spatial_holdout(pr)
        H, train_mask, geom_info = estimate_geometric_model(
            ps[train_idx], pr[train_idx], model=model, verifier=geometric_verifier, threshold=ransac_threshold
        )
        # Backwards-compatible estimate_model(ps[train_idx]
        global_train_inlier = np.zeros(len(ps), dtype=bool)
        global_train_inlier[train_idx] = train_mask
        if int(train_mask.sum()) < 8:
            raise ValueError(f"Too few robust inliers ({int(train_mask.sum())}) after {geom_info.get('verifier_method', geometric_verifier)} estimation.")
        geom_status = "success"
        geom_error = None
    except Exception as exc:
        geom_status = "failed"
        geom_error = str(exc)
        H = None
        train_mask = np.array([], dtype=bool)
        global_train_inlier = np.zeros(len(ps), dtype=bool)
        geom_info = {
            "verifier_method": f"{geometric_verifier} failed",
            "primary_verifier": geometric_verifier,
            "fallback_used": False,
            "fallback_reason": str(exc),
            "model": model,
            "threshold": ransac_threshold,
            "inlier_count": 0,
            "outlier_count": 0,
            "inlier_ratio": 0.0,
        }
    processing_time = time.perf_counter() - start
    stages.append(make_stage("09", "Geometric Verification", geom_status,
                           {"model": model, "ransac_threshold": ransac_threshold, "verifier": geometric_verifier},
                           f"{geom_info.get('verifier_method', geometric_verifier)} robust transformation estimation",
                           "Geometrically verified inliers / outliers",
                           {"model": model, "ransac_threshold": ransac_threshold, "verifier": geom_info.get("verifier_method", geometric_verifier)},
                           {"training_inliers": int(train_mask.sum()) if train_mask.size else 0,
                            "inlier_ratio": float(train_mask.mean()) if train_mask.size else 0.0,
                            "verifier_method": geom_info.get("verifier_method", geometric_verifier),
                            "fallback_used": geom_info.get("fallback_used", False)},
                           processing_time, geom_error))

    if geom_status == "failed":
        raise ValueError(f"Geometric verification failed: {geom_error}")

    # Stage 10 - Spatial distribution
    start = time.perf_counter()
    try:
        # Grid-based candidate match distribution for source and reference
        grid_counts = np.zeros((grid_size, grid_size), dtype=int)
        for p in ps:
            r, c = grid_key(p, src.gray.shape, grid_size)
            grid_counts[r, c] += 1
        coverage_fraction = np.mean(grid_counts > 0)
        spatial_status = "success" if coverage_fraction >= 0.3 else "warning"
        spatial_error = None if spatial_status == "success" else "Spatial coverage is too concentrated; poor point distribution."
    except Exception as exc:
        spatial_status = "failed"
        spatial_error = str(exc)
        coverage_fraction = 0.0
    processing_time = time.perf_counter() - start
    stages.append(make_stage("10", "Spatial Distribution", spatial_status,
                           {"grid_size": grid_size, "cell_limit": cell_limit},
                           "Grid-based match filtering / source/reference image coverage distribution",
                           "Match distribution over spatial grid",
                           {"grid_size": grid_size, "cell_limit": cell_limit},
                           {"spatial_coverage": float(coverage_fraction),
                            "occupied_cells": int((grid_counts > 0).sum()) if 'grid_counts' in locals() else 0,
                            "total_cells": int(grid_size * grid_size)},
                           processing_time, spatial_error))

    if spatial_status == "failed":
        raise ValueError(f"Spatial distribution failed: {spatial_error}")

    # Stage 11 - Sub-pixel refinement
    start = time.perf_counter()
    refinement_shift = 0.0
    try:
        if subpixel and train_mask.sum() >= 8:
            pts_before = ps[global_train_inlier].copy()
            rs, rr = refine_subpixel(src.gray, ref.gray, ps[global_train_inlier], pr[global_train_inlier])
            pts_after = rs.copy()
            refinement_shift = float(np.mean(np.linalg.norm(pts_after - pts_before, axis=1))) if len(pts_before) else 0.0
            H2, mask2, _ = estimate_geometric_model(rs, rr, model=model, verifier=geometric_verifier, threshold=ransac_threshold)
            if mask2.sum() >= 8:
                H = H2
                refined = True
                refinement_status = "success"
                refinement_error = None
            else:
                refined = False
                refinement_status = "warning"
                refinement_error = "Sub-pixel refinement failed to retain enough confident inliers."
        else:
            refined = False
            refinement_status = "success"
            refinement_error = None
    except Exception as exc:
        refined = False
        refinement_status = "failed"
        refinement_error = str(exc)
    processing_time = time.perf_counter() - start
    stages.append(make_stage("11", "Fine Matching + Sub-pixel Refinement", refinement_status,
                           {"subpixel": subpixel, "refinement_window": 11},
                           "Corner/LK sub-pixel refinement on inlier points in image regions",
                           "Refined inlier correspondences",
                           {"subpixel": subpixel},
                           {"subpixel_refined": bool(refined), "mean_refinement_shift_px": refinement_shift},
                           processing_time, refinement_error))

    # Stage 12 registration
    start = time.perf_counter()
    try:
        pred = project(ps, H)
        errors = np.linalg.norm(pred - pr, axis=1)
        val_pred = project(ps[val_idx], H) if len(val_idx) else np.empty((0, 2))
        val_err = np.linalg.norm(val_pred - pr[val_idx], axis=1) if len(val_idx) else np.array([])
        registered = cv2.warpPerspective(src.gray, H, (ref.gray.shape[1], ref.gray.shape[0]))
        valid = cv2.warpPerspective(src.mask, H, (ref.gray.shape[1], ref.gray.shape[0]), flags=cv2.INTER_NEAREST)
        overlap = (valid > 0) & (ref.mask > 0)
        if overlap.sum() < 100:
            raise ValueError("Estimated registration has insufficient valid overlap.")
        registered[valid == 0] = 0
        registration_status = "success"
        registration_error = None
    except Exception as exc:
        registration_status = "failed"
        registration_error = str(exc)
        registered = None
        valid = None
        overlap = None
        errors = np.array([])
        H = None
    processing_time = time.perf_counter() - start
    stages.append(make_stage("12", "Image Registration", registration_status,
                           {"model": model, "ransac_threshold": ransac_threshold},
                           "Warp source image into reference geometry with perspective transform",
                           "Registered image and validity mask",
                           {"reference_overlap": float(overlap.sum() / max((ref.mask > 0).sum(), 1)) if overlap is not None else 0.0,
                            "warped_shape": (ref.gray.shape[1], ref.gray.shape[0]) if ref else None},
                           {"overlap_pixels": int(overlap.sum()) if overlap is not None else 0,
                            "validity_pixels": int(valid.sum()) if valid is not None else 0},
                           processing_time, registration_error))
    if registration_status == "failed":
        raise ValueError(f"Registration failed: {registration_error}")

    # Final evaluation metrics
    start = time.perf_counter()
    try:
        train_err = errors[global_train_inlier]
        if len(train_err) == 0:
            raise ValueError("No inlier residuals available for RMSE evaluation.")
        fit_rmse = float(np.sqrt(np.mean(train_err**2)))
        inlier_ratio = float(train_mask.mean())
        match_count = int(len(ps))
        inlier_count = int(train_mask.sum())
        spatial_coverage = float(np.mean((grid_counts if 'grid_counts' in locals() else np.zeros(1)) > 0)) if 'grid_counts' in locals() else 0.0
        total_seconds = time.perf_counter() - start_total
        evaluation_status = "success"
        evaluation_error = None
    except Exception as exc:
        evaluation_status = "failed"
        evaluation_error = str(exc)
        fit_rmse = None
        inlier_ratio = None
        match_count = int(len(ps)) if len(ps) else 0
        inlier_count = int(train_mask.sum()) if train_mask.size else 0
        spatial_coverage = 0.0
        total_seconds = time.perf_counter() - start_total
    processing_time = time.perf_counter() - start
    stages.append(make_stage("11", "Evaluation", evaluation_status,
                           {"correspondence_points": int(len(ps)), "inlier_mask": int(train_mask.sum())},
                           "Computed RMSE, inlier ratio, match count, inlier count, spatial coverage, processing time",
                           "Final measurable evaluation metrics",
                           {"rmse_working_px": fit_rmse,
                            "inlier_ratio": inlier_ratio,
                            "match_count": match_count,
                            "inlier_count": inlier_count,
                            "spatial_coverage": spatial_coverage,
                            "processing_time_seconds": total_seconds},
                           {"RMSE": fit_rmse, "Inlier Ratio": inlier_ratio,
                            "Match Count": match_count, "Inlier Count": inlier_count,
                            "Spatial Coverage": spatial_coverage,
                            "Processing Time": total_seconds},
                           processing_time, evaluation_error))

    # Create residual-corrected image if requested
    coeffs = None
    corrected = None
    if residual_correction and global_train_inlier.sum() >= 20:
        tr_points = ps[global_train_inlier]
        tr_ref = pr[global_train_inlier]
        tr_pred = project(tr_points, H)
        coeffs = fit_residual_field(tr_ref, tr_ref - tr_pred, ref.gray.shape)
        if coeffs is not None:
            corrected = smooth_corrected_warp(src.gray, H, coeffs, ref.gray.shape)

    H_original = ref.to_original @ H @ np.linalg.inv(src.to_original)
    src_orig = project(ps, src.to_original)
    ref_orig = project(pr, ref.to_original)
    pred_orig = project(src_orig, H_original)
    orig_err = np.linalg.norm(pred_orig - ref_orig, axis=1)

    valid = cv2.warpPerspective(src.mask, H, (ref.gray.shape[1], ref.gray.shape[0]), flags=cv2.INTER_NEAREST)
    overlap = (valid > 0) & (ref.mask > 0)
    if overlap.sum() < 100:
        raise ValueError("Estimated registration has insufficient valid overlap.")

    # These are intermediate before output
    overlay = create_overlay(ref.gray, registered, overlap)
    correspondence = draw_matches(src.gray, ref.gray, ps, pr, global_train_inlier)

    train_err = errors[global_train_inlier]
    table = pd.DataFrame({
        "source_x_working_px": ps[:, 0],
        "source_y_working_px": ps[:, 1],
        "reference_x_working_px": pr[:, 0],
        "reference_y_working_px": pr[:, 1],
        "source_x_original_px": src_orig[:, 0],
        "source_y_original_px": src_orig[:, 1],
        "reference_x_original_px": ref_orig[:, 0],
        "reference_y_original_px": ref_orig[:, 1],
        "match_score": scores,
        "representation": methods,
        "training_inlier": global_train_inlier,
        "holdout_validation": np.isin(np.arange(len(ps)), val_idx),
        "fit_error_working_px": errors,
        "fit_error_original_px": orig_err,
    })

    report = {
        "pipeline": "LunaMatch V3 multimodal correspondence & robust geometry",
        "source_sensor": source_sensor,
        "reference_sensor": reference_sensor,
        "source_filename": source_file.name,
        "reference_filename": reference_file.name,
        "source_info": src.info,
        "reference_info": ref.info,
        # matching_strategy records the actual routing decision and matcher used.
        # It is populated from actual runtime variables, not from UI labels.
        "matching_strategy": {
            "sensor_relation": "same_sensor" if sensor_path == "same" else "cross_sensor",
            "source_sensor": source_sensor,
            "reference_sensor": reference_sensor,
            "sensor_pair": f"{source_sensor} ↔ {reference_sensor}",
            "matcher": actual_matcher,
            "geometric_verifier": geom_info.get("verifier_method", geometric_verifier),
            "primary_verifier": geometric_verifier,
            "fallback_used": geom_info.get("fallback_used", False),
            "fallback_reason": geom_info.get("fallback_reason"),
            "reason": (
                "Source and reference use the same sensor type → SIFT multi-representation pipeline"
                if sensor_path == "same"
                else f"Cross-sensor pair ({source_sensor} ↔ {reference_sensor}) → Genuine LoFTR cross-modal deep matching"
            ),
            "loftr_available": loftr_avail,
            "loftr_metrics": loftr_metrics,
            "loftr_note": "LoFTR provides cross-modal transformer correspondences with self/cross-attention across multimodal Chandrayaan-2 imagery.",
            # IIRS-specific fields (populated when one or both sensors is IIRS)
            "source_gsd_m": src_gsd,
            "reference_gsd_m": ref_gsd,
            "scale_ratio": round(src_gsd / ref_gsd, 3) if (src_gsd and ref_gsd and ref_gsd > 0) else None,
            "iirs_source_representation": _iirs_src_diag.get("representation") if _iirs_src_diag else None,
            "iirs_reference_representation": _iirs_ref_diag.get("representation") if _iirs_ref_diag else None,
            "cross_sensor_learned_matching_label": "Cross-sensor learned matching" if sensor_path == "different" and actual_matcher == "LoFTR" else None,
        },

        "stages": [
            "sensor-aware input", "NoData masking and robust normalization",
            "illumination-tolerant local contrast", "gradient/second-order structural representations",
            "multi-detector reciprocal matching", "spatially balanced fusion",
            "spatial holdout validation", "robust RANSAC/MAGSAC geometry",
            "optional local subpixel refinement", "optional smooth residual correction",
        ],
        "matching_models": {name: len(rows) for name, rows in runs},
        "total_fused_correspondences": int(len(ps)),
        "training_points": int(len(train_idx)),
        "training_inliers": int(train_mask.sum()),
        "training_inlier_ratio": float(train_mask.mean()),
        "holdout_points": int(len(val_idx)),
        "holdout_rmse_working_px": float(np.sqrt(np.mean(val_err**2))) if len(val_err) else None,
        "holdout_median_working_px": float(np.median(val_err)) if len(val_err) else None,
        "holdout_p95_working_px": float(np.percentile(val_err, 95)) if len(val_err) else None,
        "fit_rmse_working_px": float(np.sqrt(np.mean(train_err**2))),
        "fit_p95_working_px": float(np.percentile(train_err, 95)),
        "fit_rmse_original_px": float(np.sqrt(np.mean(orig_err[global_train_inlier]**2))),
        "reference_overlap_fraction": float(overlap.sum() / max((ref.mask > 0).sum(), 1)),
        "subpixel_refinement_requested": bool(subpixel),
        "subpixel_refinement_applied": bool(refined),
        "smooth_residual_requested": bool(residual_correction),
        "smooth_residual_applied": coeffs is not None,
        "independent_ground_truth": False,
        "subpixel_accuracy_certified": False,
        "H_source_working_to_reference_working": H.tolist(),
        "H_source_original_to_reference_original": H_original.tolist(),
        "scientific_caveats": [
            "Holdout validation uses withheld image correspondences, not independent ground-truth checkpoints.",
            "The smooth residual field is an empirical correction, not a physical DEM/terrain model.",
            "The optional learned cross-modal matcher is intentionally a plugin rather than a claim of trained lunar generalization.",
            "Full-resolution scientific GeoTIFF output requires preservation of mission metadata and georeferencing.",
        ],
    }

    # Align the runtime stage trace exactly to the requested UI architecture flow.
    # Keep the functional stages that actually produce the requested pipeline artifacts and
    # remove the extra sensor and spatial distribution artifacts created by the earlier
    # skeleton-to-pipeline trace while preserving the same underlying function objects.
    keep_names = {
        "Input",
        "Metadata",
        "Preprocessing",
        "Multi-Scale",
        "Features / Matching",
        "Confidence Filtering",
        "Spatial Consistency",
        "Geometric Verification",
        "Fine Matching + Sub-pixel Refinement",
        "Image Registration",
        "Evaluation",
    }
    visible_stages = [stage for stage in stages if stage.get("name") in keep_names]
    for idx, stage in enumerate(visible_stages, start=1):
        stage["stage_id"] = f"{idx:02d}"

    return {
        "source": src, "reference": ref, "report": report, "matches": table,
        "correspondence": correspondence, "overlay": overlay,
        "registered": registered, "corrected": corrected, "validity": valid,
        "stages": visible_stages,
    }


# ---------------- UI ----------------
st.title("🌙 LunaMatch V3")
st.caption("Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (SIH26166 • ISRO)")
st.info("Research pipeline by Team Akatsuki. Low residuals are not, by themselves, proof of sub-pixel scientific accuracy.")

# --- EXPERIMENT CONFIGURATION SELECTOR ---
pair_mode = st.radio(
    "Experiment Configuration",
    [
        "Real data / Interactive Upload",
        "Use Demo: PAIR_001",
    ],
    index=0,
    horizontal=True,
    help="Select PAIR_001 for the primary real-world cross-mission experiment, or choose Custom Pair to configure manually.",
)
is_pair_001 = pair_mode == "Use Demo: PAIR_001"

# Top-level visual pair display card placeholder — populated below from canonical metadata
header_card_placeholder = st.empty()

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### 🛰️ SOURCE / MOVING")
    source_mission = st.selectbox("Source Mission", SOURCE_MISSIONS, index=0)
    source_sensor_ui = st.selectbox(
        "Source Instrument / Sensor", SOURCE_SENSORS_UI, index=0 if is_pair_001 else 0,
        help="Select Chandrayaan-2 instrument: OHRC, TMC-2, or IIRS."
    )
    source_file = st.file_uploader(
        "Upload source image", type=["tif", "tiff", "png", "jpg", "jpeg", "npy", "img", "raw", "dat"], key="source"
    )
with c2:
    st.markdown("#### 🗺️ REFERENCE / FIXED")
    reference_mission = st.selectbox("Reference Mission", REFERENCE_MISSIONS, index=0 if is_pair_001 else 0)
    reference_sensor_ui = st.selectbox(
        "Reference Instrument / Sensor", REFERENCE_SENSORS_UI, index=0 if is_pair_001 else 0,
        help="Select independent lunar reference instrument (e.g. LROC NAC, SELENE TC/MI)."
    )
    reference_file = st.file_uploader(
        "Upload reference image", type=["tif", "tiff", "png", "jpg", "jpeg", "npy", "img", "raw", "dat"], key="reference"
    )

if source_file is not None:
    st.caption(f"Source product: **{classify_product_file(source_file.name)}** | {len(source_file.getvalue()) / (1024 ** 2):.1f} MB")
    if classify_product_file(source_file.name) == "BROWSE / PREVIEW IMAGE":
        st.warning("Browse/preview image selected. Scientific source product required for full-resolution processing.")
if reference_file is not None:
    st.caption(f"Reference product: **{classify_product_file(reference_file.name)}** | {len(reference_file.getvalue()) / (1024 ** 2):.1f} MB")

# Resolve canonical sensor tokens
_src_sensor_raw = _canonical_sensor(source_sensor_ui)
_ref_sensor_raw = _canonical_sensor(reference_sensor_ui)

# Auto-detect from filename when user selected 'Auto Detect'
if _src_sensor_raw == "Unknown" and source_file is not None:
    _detected = detect_sensor_from_filename(source_file.name)
    source_sensor = _detected
    st.caption(f"Source sensor auto-detected: **{source_sensor}**")
else:
    source_sensor = _src_sensor_raw

if _ref_sensor_raw == "Unknown" and reference_file is not None:
    _detected_ref = detect_sensor_from_filename(reference_file.name)
    reference_sensor = _detected_ref
    st.caption(f"Reference sensor auto-detected: **{reference_sensor}**")
else:
    reference_sensor = _ref_sensor_raw

# ──────────────────────────────────────────────────────
# IIRS PROCESSING CONTROLS (shown when either sensor = IIRS)
# ──────────────────────────────────────────────────────
_any_iirs = (source_sensor == "IIRS" or reference_sensor == "IIRS")
iirs_representation = "automatic"
iirs_pca_component = 1
iirs_normalization = "percentile"
iirs_invalid_handling = "automatic"
iirs_resolution_handling = "automatic"

if _any_iirs:
    with st.expander("🔬 IIRS Hyperspectral Processing", expanded=True):
        st.info(
            "**IIRS** is the Chandrayaan-2 Imaging Infrared Spectrometer (0.8–5.0 µm, ~256 bands, ~80 m/pixel nominal). "
            "The hyperspectral cube is reduced to a 2D spatial representation before feature matching. "
            "SIFT and LoFTR operate on the derived 2D representation — **not** directly on spectral data."
        )
        iirs_c1, iirs_c2, iirs_c3 = st.columns(3)
        with iirs_c1:
            iirs_representation = st.selectbox(
                "Representation",
                ["Automatic", "PCA", "Selected Band", "Composite", "Gradient"],
                index=0,
                help="'Automatic' uses PCA (PC1) for multi-band cubes. 'Selected Band' uses a single spectral band.",
            ).lower().replace(" ", "_").replace("selected_band", "selected_band")
            iirs_pca_component = st.selectbox("PCA Component", [1, 2, 3], index=0)
        with iirs_c2:
            iirs_normalization = st.selectbox(
                "Normalization",
                ["Percentile", "Standardization", "None"],
                index=0,
            ).lower()
            iirs_invalid_handling = st.selectbox(
                "Invalid Pixel Handling",
                ["Automatic", "Mask", "Replace"],
                index=0,
            ).lower()
        with iirs_c3:
            iirs_resolution_handling = st.selectbox(
                "Resolution Handling",
                ["Automatic", "Downsample High Resolution", "Common Resolution"],
                index=0,
            ).lower()
            st.markdown(
                "**IIRS nominal GSD:** `~80 m/pixel`  \n"
                "**OHRC GSD:** `~0.25 m/pixel`  \n"
                "**TMC-2 GSD:** `~5 m/pixel`"
            )

        # Live IIRS Matching Representation Preview (shown after file upload)
        if source_file is not None and source_sensor == "IIRS":
            st.markdown("---")
            st.markdown("#### 🛰️ IIRS Source Matching Representation Preview")
            try:
                _prev_cube, _prev_meta = iirs_preprocessing.load_iirs(source_file.getvalue(), source_file.name)
                _prev_cube_clean, _prev_mask = iirs_preprocessing.remove_invalid_pixels(_prev_cube, no_data_value=-9999.0)
                _prev_norm, _prev_mask = iirs_preprocessing.normalize_iirs(_prev_cube_clean, _prev_mask, method=iirs_normalization)
                _prev_rep, _prev_diag = iirs_preprocessing.generate_iirs_2d_representation(
                    _prev_norm, _prev_mask,
                    method=iirs_representation,
                    pca_component=iirs_pca_component,
                    apply_clahe=True,
                )
                _p1, _p2, _p3, _p4 = st.columns(4)
                _p1.metric("Spectral Bands", _prev_cube.shape[0])
                _p2.metric("Dimensions", f"{_prev_cube.shape[2]} × {_prev_cube.shape[1]}")
                _p3.metric("Valid Pixels", f"{_prev_diag.get('valid_pixel_pct', 0):.1f} %")
                _p4.metric("Est. GSD", f"{_prev_meta.get('gsd_m_per_pixel', '~80')} m/px")
                st.image(_prev_rep, caption=f"2D Representation: {_prev_diag.get('method_applied', iirs_representation)} | CLAHE: {_prev_diag.get('clahe_applied', False)}", use_container_width=True)
                for _w in _prev_meta.get("warnings", []):
                    st.warning(f"⚠️ {_w}")
            except Exception as _prev_exc:
                st.warning(f"Preview not available: {_prev_exc}")

        if reference_file is not None and reference_sensor == "IIRS":
            st.markdown("---")
            st.markdown("#### 🛰️ IIRS Reference Matching Representation Preview")
            try:
                _rprev_cube, _rprev_meta = iirs_preprocessing.load_iirs(reference_file.getvalue(), reference_file.name)
                _rprev_cube_clean, _rprev_mask = iirs_preprocessing.remove_invalid_pixels(_rprev_cube, no_data_value=-9999.0)
                _rprev_norm, _rprev_mask = iirs_preprocessing.normalize_iirs(_rprev_cube_clean, _rprev_mask, method=iirs_normalization)
                _rprev_rep, _rprev_diag = iirs_preprocessing.generate_iirs_2d_representation(
                    _rprev_norm, _rprev_mask,
                    method=iirs_representation,
                    pca_component=iirs_pca_component,
                    apply_clahe=True,
                )
                _rp1, _rp2, _rp3, _rp4 = st.columns(4)
                _rp1.metric("Spectral Bands", _rprev_cube.shape[0])
                _rp2.metric("Dimensions", f"{_rprev_cube.shape[2]} × {_rprev_cube.shape[1]}")
                _rp3.metric("Valid Pixels", f"{_rprev_diag.get('valid_pixel_pct', 0):.1f} %")
                _rp4.metric("Est. GSD", f"{_rprev_meta.get('gsd_m_per_pixel', '~80')} m/px")
                st.image(_rprev_rep, caption=f"2D Representation: {_rprev_diag.get('method_applied', iirs_representation)} | CLAHE: {_rprev_diag.get('clahe_applied', False)}", use_container_width=True)
                for _rw in _rprev_meta.get("warnings", []):
                    st.warning(f"⚠️ {_rw}")
            except Exception as _rprev_exc:
                st.warning(f"Preview not available: {_rprev_exc}")


# ── File-hash-based session state invalidation ──────────────────────────────
# When the uploaded file changes, we must discard stale metadata and results.
_src_hash_now = compute_file_hash(source_file.getvalue()) if source_file else None
_ref_hash_now = compute_file_hash(reference_file.getvalue()) if reference_file else None

# If source file changed, clear stale source state
if _src_hash_now and st.session_state.get("_src_file_hash") != _src_hash_now:
    for _k in ("_src_meta", "_src_meta_source", "result", "comp_result"):
        st.session_state.pop(_k, None)
    st.session_state["_src_file_hash"] = _src_hash_now

# If reference file changed, clear stale reference state and results
if _ref_hash_now and st.session_state.get("_ref_file_hash") != _ref_hash_now:
    for _k in ("_ref_meta", "_ref_meta_source", "_ref_manual_meta", "result", "comp_result"):
        st.session_state.pop(_k, None)
    st.session_state["_ref_file_hash"] = _ref_hash_now

# If either file was removed, clear the corresponding state
if _src_hash_now is None and st.session_state.get("_src_file_hash"):
    for _k in ("_src_meta", "_src_meta_source", "_src_file_hash", "result", "comp_result"):
        st.session_state.pop(_k, None)
if _ref_hash_now is None and st.session_state.get("_ref_file_hash"):
    for _k in ("_ref_meta", "_ref_meta_source", "_ref_manual_meta", "_ref_file_hash", "result", "comp_result"):
        st.session_state.pop(_k, None)

metadata_expander = st.expander("IMAGE METADATA", expanded=True)
with metadata_expander:
    md_source_col, md_ref_col = st.columns(2)
    with md_source_col:
        st.markdown("### Source / Moving Image Metadata")
        source_meta_file = st.file_uploader("Upload source metadata/XML", type=["xml", "txt"], key="source_meta_xml")
        if source_meta_file is not None:
            source_xml_bytes = source_meta_file.getvalue()
            source_meta = parse_chandrayaan2_pds4_xml(source_xml_bytes, source_meta_file.name)
            st.caption(f"XML: **{source_meta_file.name}** ({len(source_xml_bytes):,} bytes)")
            source_meta_source = "PARSED_FROM_PRODUCT"
        elif is_pair_001:
            # DEMO MODE: explicit user selection only
            source_meta = PAIR_001_SOURCE.to_canonical_dict()
            st.caption("🔵 Demo metadata loaded by explicit DEMO MODE selection.")
            source_meta_source = "DEMO"
        else:
            # REAL MODE with no XML uploaded — never use demo values
            source_meta = empty_metadata_template()
            source_meta_source = "NOT_PROVIDED"

        source_status = metadata_status(source_meta, source_meta_source)
        _src_icon   = source_status["icon"]
        _src_label  = source_status["label"]
        _src_detail = source_status["detail"]

        if source_meta_source == "NOT_PROVIDED":
            st.info(f"Upload source Chandrayaan-2 PDS4 XML to populate metadata.")
        elif source_status["enum_status"] == MetadataStatus.DEMO:
            st.info(f"🔵 **Demo metadata** — not a real uploaded product.")
        elif source_status["enum_status"] in (MetadataStatus.INCOMPLETE, MetadataStatus.INVALID):
            err_msg = "; ".join(source_meta.get("validation_errors", ["Incomplete"]))
            st.warning(f"{_src_icon} Source metadata incomplete: {err_msg}")
        else:
            st.success(f"{_src_icon} Source metadata available ({_src_label})")

        for warning in source_meta.get("validation_warnings", []):
            st.warning(warning)

        if source_meta_source != "NOT_PROVIDED":
            s_col1, s_col2 = st.columns(2)
            _prov = f" *(source: {_src_label})*" if source_meta_source == "DEMO" else ""
            with s_col1:
                st.markdown(f"**Sensor Type:** `{source_meta.get('sensor_type') or '—'}`")
                st.markdown(f"**Processing Level:** `{source_meta.get('processing_level') or '—'}`")
                st.markdown(f"**Orbit:** `{source_meta.get('orbit_number') or '—'}`")
                st.markdown(f"**Altitude:** `{source_meta.get('altitude_km') or '—'} km`")
                gsd = source_meta.get("gsd_m_per_pixel")
                st.markdown(f"**GSD:** `{f'{gsd} m/pixel' if gsd is not None else 'Not available'}`{_prov}")
                dims = source_meta.get("dimensions", {})
                st.markdown(f"**Dimensions:** `{dims.get('lines') or '—'} × {dims.get('samples') or '—'}`")
            with s_col2:
                st.markdown(f"**Projection:** `{source_meta.get('projection') or '—'}`")
                st.markdown(f"**Area:** `{source_meta.get('area') or '—'}`")
                st.markdown(f"**Sun Az/El/Inc:** `{source_meta.get('sun_azimuth_deg') or '—'}° / {source_meta.get('sun_elevation_deg') or '—'}° / {source_meta.get('solar_incidence_deg') or '—'}°`")
                st.markdown(f"**Roll/Pitch/Yaw:** `{source_meta.get('roll_deg') or '—'}° / {source_meta.get('pitch_deg') or '—'}° / {source_meta.get('yaw_deg') or '—'}°`")
                st.markdown(f"**Start:** `{source_meta.get('start_time') or '—'}`")
                st.markdown(f"**Stop:** `{source_meta.get('stop_time') or '—'}`")

            fp = source_meta.get("footprint", {})
            if isinstance(fp, dict) and any(fp.get(k) for k in ("upper_left", "upper_right", "lower_left", "lower_right")):
                st.markdown("**Source Footprint Corners:**")
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    st.caption(f"**UL:** `{fp.get('upper_left', '—')}`")
                    st.caption(f"**LL:** `{fp.get('lower_left', '—')}`")
                with f_col2:
                    st.caption(f"**UR:** `{fp.get('upper_right', '—')}`")
                    st.caption(f"**LR:** `{fp.get('lower_right', '—')}`")
            else:
                st.caption("Source footprint: Not available from current metadata.")

            with st.expander("Canonical Parsed Dictionary (Source)", expanded=False):
                st.json(source_meta)

    with md_ref_col:
        st.markdown("### Reference / Fixed Image Metadata")

        # ── If reference image is uploaded but no label is available ──────────
        if reference_file is not None and not is_pair_001:
            _ref_type = classify_product_file(reference_file.name)
            st.caption(
                f"Reference product: **{_ref_type}** | "
                f"{len(reference_file.getvalue())/(1024**2):.1f} MB"
            )

        reference_meta_file = st.file_uploader(
            "Upload reference metadata/XML/LBL",
            type=["xml", "txt", "lbl"],
            key="reference_meta_xml",
        )

        if reference_meta_file is not None:
            reference_xml_bytes = reference_meta_file.getvalue()
            reference_meta = parse_metadata_xml(reference_xml_bytes, reference_meta_file.name)
            st.caption(f"Label: **{reference_meta_file.name}** ({len(reference_xml_bytes):,} bytes)")
            reference_meta_source = "UPLOADED_LABEL"
        elif is_pair_001:
            # DEMO MODE only
            reference_meta = PAIR_001_REFERENCE.to_canonical_dict()
            st.caption("🔵 Demo metadata loaded by explicit DEMO MODE selection.")
            reference_meta_source = "DEMO"
        elif "_ref_manual_meta" in st.session_state:
            # User previously entered metadata manually — persist across reruns
            reference_meta = st.session_state["_ref_manual_meta"]
            reference_meta_source = "MANUALLY_ENTERED"
        else:
            # REAL MODE — no metadata provided yet
            reference_meta = empty_metadata_template()
            reference_meta_source = "NOT_PROVIDED"

        reference_status = metadata_status(reference_meta, reference_meta_source)
        _ref_icon   = reference_status["icon"]
        _ref_label  = reference_status["label"]

        # ── Status display ──────────────────────────────────────────────────
        if reference_meta_source == "NOT_PROVIDED":
            if reference_file is not None:
                st.warning(
                    "⚠️ Reference image uploaded — metadata is not yet available. "
                    "Geographic overlap and scale validation require reference metadata."
                )
            else:
                st.info("Upload reference image and metadata to populate this section.")
        elif reference_status["enum_status"] == MetadataStatus.DEMO:
            st.info("🔵 **Demo metadata** — not a real uploaded product.")
        elif reference_status["enum_status"] in (MetadataStatus.INCOMPLETE, MetadataStatus.INVALID):
            err_msg = "; ".join(reference_meta.get("validation_errors", ["Incomplete"]))
            st.warning(f"{_ref_icon} Reference metadata incomplete: {err_msg}")
        else:
            st.success(f"{_ref_icon} Reference metadata available ({_ref_label})")

        for warning in reference_meta.get("validation_warnings", []):
            st.warning(warning)

        if reference_meta_source != "NOT_PROVIDED":
            r_col1, r_col2 = st.columns(2)
            _prov = f" *(source: {_ref_label})*" if reference_meta_source == "DEMO" else ""
            with r_col1:
                st.markdown(f"**Sensor Type:** `{reference_meta.get('sensor_type') or '—'}`")
                st.markdown(f"**Processing Level:** `{reference_meta.get('processing_level') or '—'}`")
                st.markdown(f"**Orbit:** `{reference_meta.get('orbit_number') or '—'}`")
                st.markdown(f"**Altitude:** `{reference_meta.get('altitude_km') or '—'} km`")
                gsd = reference_meta.get("gsd_m_per_pixel")
                st.markdown(f"**GSD:** `{f'{gsd} m/pixel' if gsd is not None else 'Not available'}`{_prov}")
                dims = reference_meta.get("dimensions", {})
                st.markdown(f"**Dimensions:** `{dims.get('lines') or '—'} × {dims.get('samples') or '—'}`")
            with r_col2:
                st.markdown(f"**Projection:** `{reference_meta.get('projection') or '—'}`")
                st.markdown(f"**Area:** `{reference_meta.get('area') or '—'}`")
                st.markdown(f"**Sun Az/El/Inc:** `{reference_meta.get('sun_azimuth_deg') or '—'}° / {reference_meta.get('sun_elevation_deg') or '—'}° / {reference_meta.get('solar_incidence_deg') or '—'}°`")
                st.markdown(f"**Roll/Pitch/Yaw:** `{reference_meta.get('roll_deg') or '—'}° / {reference_meta.get('pitch_deg') or '—'}° / {reference_meta.get('yaw_deg') or '—'}°`")
                st.markdown(f"**Start:** `{reference_meta.get('start_time') or '—'}`")
                st.markdown(f"**Stop:** `{reference_meta.get('stop_time') or '—'}`")

            fp = reference_meta.get("footprint", {})
            if isinstance(fp, dict) and any(fp.get(k) for k in ("upper_left", "upper_right", "lower_left", "lower_right")):
                st.markdown("**Reference Footprint Corners:**")
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    st.caption(f"**UL:** `{fp.get('upper_left', '—')}`")
                    st.caption(f"**LL:** `{fp.get('lower_left', '—')}`")
                with f_col2:
                    st.caption(f"**UR:** `{fp.get('upper_right', '—')}`")
                    st.caption(f"**LR:** `{fp.get('lower_right', '—')}`")
            else:
                st.caption("Reference footprint: Not available from current metadata.")

            with st.expander("Canonical Parsed Dictionary (Reference)", expanded=False):
                st.json(reference_meta)

        # ── Manual reference metadata entry ─────────────────────────────────
        if not is_pair_001 and reference_meta_source in ("NOT_PROVIDED", "MANUALLY_ENTERED"):
            with st.expander("📝 Enter Reference Metadata Manually", expanded=(reference_meta_source == "MANUALLY_ENTERED")):
                st.caption(
                    "Minimum required for geographic validation: resolution, dimensions, "
                    "and at least two footprint corner coordinates."
                )
                _man_pid = st.text_input(
                    "Product ID",
                    value=reference_meta.get("product_id") or "",
                    key="man_ref_pid",
                )
                _man_c1, _man_c2, _man_c3 = st.columns(3)
                with _man_c1:
                    _man_gsd = st.number_input(
                        "Resolution (m/pixel)", min_value=0.01, max_value=1000.0,
                        value=float(reference_meta.get("gsd_m_per_pixel") or 2.0),
                        step=0.01, format="%.4f", key="man_ref_gsd"
                    )
                    _man_lines = st.number_input(
                        "Image lines", min_value=1,
                        value=int(reference_meta.get("dimensions", {}).get("lines") or 1),
                        step=1, key="man_ref_lines"
                    )
                with _man_c2:
                    _man_samples = st.number_input(
                        "Image samples", min_value=1,
                        value=int(reference_meta.get("dimensions", {}).get("samples") or 1),
                        step=1, key="man_ref_samples"
                    )
                    _man_inc = st.number_input(
                        "Incidence angle (°)", min_value=0.0, max_value=180.0,
                        value=float(reference_meta.get("solar_incidence_deg") or 0.0),
                        step=0.01, key="man_ref_inc"
                    )
                with _man_c3:
                    _man_alt = st.number_input(
                        "Altitude (km)", min_value=0.0, max_value=1000.0,
                        value=float(reference_meta.get("altitude_km") or 0.0),
                        step=0.1, key="man_ref_alt"
                    )
                st.markdown("**Corner coordinates** (latitude, longitude)")
                _fp_c1, _fp_c2 = st.columns(2)
                _fp = reference_meta.get("footprint", {})
                with _fp_c1:
                    _ul = _fp.get("upper_left") or [0.0, 0.0]
                    _man_ul_lat = st.number_input("UL Latitude",  value=float(_ul[0]) if len(_ul) > 0 else 0.0, step=0.001, format="%.5f", key="man_ul_lat")
                    _man_ul_lon = st.number_input("UL Longitude", value=float(_ul[1]) if len(_ul) > 1 else 0.0, step=0.001, format="%.5f", key="man_ul_lon")
                    _ll = _fp.get("lower_left") or [0.0, 0.0]
                    _man_ll_lat = st.number_input("LL Latitude",  value=float(_ll[0]) if len(_ll) > 0 else 0.0, step=0.001, format="%.5f", key="man_ll_lat")
                    _man_ll_lon = st.number_input("LL Longitude", value=float(_ll[1]) if len(_ll) > 1 else 0.0, step=0.001, format="%.5f", key="man_ll_lon")
                with _fp_c2:
                    _ur = _fp.get("upper_right") or [0.0, 0.0]
                    _man_ur_lat = st.number_input("UR Latitude",  value=float(_ur[0]) if len(_ur) > 0 else 0.0, step=0.001, format="%.5f", key="man_ur_lat")
                    _man_ur_lon = st.number_input("UR Longitude", value=float(_ur[1]) if len(_ur) > 1 else 0.0, step=0.001, format="%.5f", key="man_ur_lon")
                    _lr = _fp.get("lower_right") or [0.0, 0.0]
                    _man_lr_lat = st.number_input("LR Latitude",  value=float(_lr[0]) if len(_lr) > 0 else 0.0, step=0.001, format="%.5f", key="man_lr_lat")
                    _man_lr_lon = st.number_input("LR Longitude", value=float(_lr[1]) if len(_lr) > 1 else 0.0, step=0.001, format="%.5f", key="man_lr_lon")

                if st.button("✔ Apply Manual Reference Metadata", key="apply_man_ref"):
                    _man_built = {
                        "product_id":    _man_pid or "UNKNOWN",
                        "mission":       reference_meta.get("mission") or "Lunar Reference",
                        "sensor_type":   reference_meta.get("sensor_type") or reference_sensor or "Unknown",
                        "processing_level": reference_meta.get("processing_level") or "Unknown",
                        "gsd_m_per_pixel":  float(_man_gsd),
                        "altitude_km":      float(_man_alt) if _man_alt else None,
                        "solar_incidence_deg": float(_man_inc) if _man_inc else None,
                        "sun_elevation_deg":   (90.0 - float(_man_inc)) if _man_inc else None,
                        "dimensions":    {"lines": int(_man_lines), "samples": int(_man_samples)},
                        "footprint": {
                            "upper_left":  [_man_ul_lat, _man_ul_lon],
                            "upper_right": [_man_ur_lat, _man_ur_lon],
                            "lower_left":  [_man_ll_lat, _man_ll_lon],
                            "lower_right": [_man_lr_lat, _man_lr_lon],
                        },
                        "valid": True,
                        "validation_errors": [],
                        "validation_warnings": ["Metadata manually entered — not parsed from product label."],
                    }
                    st.session_state["_ref_manual_meta"] = _man_built
                    # Invalidate any previous pair result
                    st.session_state.pop("result", None)
                    st.session_state.pop("comp_result", None)
                    st.rerun()

# ──────────────────────────────────────────────────────────────────────────
# PAIR VALIDATION & SCIENTIFIC SUMMARY GATE
# Strict three-state: VALID / PENDING / REJECTED
# REJECTED is only shown when both footprints exist and they don't overlap.
# PENDING is shown whenever any required information is missing.
# ──────────────────────────────────────────────────────────────────────────
footprint_eval = evaluate_footprint_overlap(source_meta, reference_meta)

# GSD values — only from actual metadata, never from demo fallback
src_gsd_val = source_meta.get("gsd_m_per_pixel")
ref_gsd_val = reference_meta.get("gsd_m_per_pixel")

# Dynamic scale ratio — never hardcoded
_scale_ratio_display = scale_ratio_display(src_gsd_val, ref_gsd_val)
# Legacy shim for downstream code that uses calculate_scale_ratio()
scale_ratio_val, scale_ratio_str = calculate_scale_ratio(src_gsd_val, ref_gsd_val)

# Render the top header card from the canonical metadata state
_hdr_src_gsd = f"{src_gsd_val:.4f} m/pixel" if src_gsd_val is not None else "Pending metadata"
_hdr_ref_gsd = f"{ref_gsd_val:.4f} m/pixel" if ref_gsd_val is not None else "Pending metadata"
_hdr_scale   = _scale_ratio_display if (src_gsd_val is not None and ref_gsd_val is not None) else "Scale ratio pending"

header_card_placeholder.markdown(f"""
<div class="pair-card">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 260px; padding: 0.5rem;">
            <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">SOURCE / MOVING IMAGE</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">Chandrayaan-2 Optical</div>
            <div style="font-size: 0.95rem; color: #94a3b8;">GSD: {_hdr_src_gsd} &nbsp;•&nbsp; {source_status['label']}</div>
        </div>
        <div style="text-align: center; padding: 0.5rem;">
            <div style="font-size: 1.05rem; color: #cbd5e1; font-weight: 600;">&#x2193; REGISTER &#x2193;</div>
            <span class="scale-badge" id="scale-badge-dynamic">{_hdr_scale}</span>
        </div>
        <div style="flex: 1; min-width: 260px; padding: 0.5rem; text-align: right;">
            <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">REFERENCE / FIXED IMAGE</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">Independent Lunar Reference</div>
            <div style="font-size: 0.95rem; color: #94a3b8;">GSD: {_hdr_ref_gsd} &nbsp;•&nbsp; {reference_status['label']}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Compute explicit state enums
src_fp_status = compute_footprint_status(source_meta, source_status["enum_status"])
ref_fp_status = compute_footprint_status(reference_meta, reference_status["enum_status"])
pair_val_status, pair_val_reason = compute_pair_validation(
    source_status["enum_status"],
    reference_status["enum_status"],
    src_fp_status,
    ref_fp_status,
    footprint_eval,
)

st.markdown("#### 🔬 Pair Validation")

# ── Per-product status rows ──────────────────────────────────────────────
val_src_col, val_ref_col, val_result_col = st.columns(3)
with val_src_col:
    st.markdown("**SOURCE / MOVING**")
    _si, _sl = FOOTPRINT_STATUS_LABELS.get(src_fp_status, ("?", "?"))[:2]
    if source_file:
        st.success("✓ Image: Ready")
    else:
        st.warning("⚠ Image: Not uploaded")
    _sm_icon = source_status["icon"]
    _sm_lbl  = source_status["label"]
    if source_status["enum_status"] == MetadataStatus.NOT_PROVIDED:
        st.warning(f"⚠ Metadata: Not provided")
    elif source_status["enum_status"] == MetadataStatus.INCOMPLETE:
        st.warning(f"⚠ Metadata: Incomplete")
    elif source_status["enum_status"] == MetadataStatus.DEMO:
        st.info(f"🔵 Metadata: Demo")
    else:
        st.success(f"✓ Metadata: {_sm_lbl}")
    if src_fp_status == FootprintStatus.AVAILABLE:
        st.success(f"✓ Footprint: {_si} {_sl}")
    elif src_fp_status == FootprintStatus.PENDING:
        st.warning(f"⚠ Footprint: ⏳ Pending")
    else:
        st.warning(f"⚠ Footprint: {_si} {_sl}")

with val_ref_col:
    st.markdown("**REFERENCE / FIXED**")
    _ri, _rl = FOOTPRINT_STATUS_LABELS.get(ref_fp_status, ("?", "?"))[:2]
    if reference_file:
        st.success("✓ Image: Ready")
    else:
        st.warning("⚠ Image: Not uploaded")
    _rm_icon = reference_status["icon"]
    _rm_lbl  = reference_status["label"]
    if reference_status["enum_status"] == MetadataStatus.NOT_PROVIDED:
        st.warning(f"⚠ Metadata: Not provided")
    elif reference_status["enum_status"] == MetadataStatus.INCOMPLETE:
        st.warning(f"⚠ Metadata: Incomplete")
    elif reference_status["enum_status"] == MetadataStatus.DEMO:
        st.info(f"🔵 Metadata: Demo")
    else:
        st.success(f"✓ Metadata: {_rm_lbl}")
    if ref_fp_status == FootprintStatus.AVAILABLE:
        st.success(f"✓ Footprint: {_ri} {_rl}")
    elif ref_fp_status == FootprintStatus.PENDING:
        st.warning(f"⚠ Footprint: ⏳ Pending")
    else:
        st.warning(f"⚠ Footprint: {_ri} {_rl}")

with val_result_col:
    st.markdown("**PAIR RESULT**")
    _pv_icon, _pv_title, _pv_detail = PAIR_STATUS_LABELS[pair_val_status]
    if pair_val_status == PairValidationStatus.VALID:
        st.success(f"{_pv_icon} {_pv_title}")
    elif pair_val_status == PairValidationStatus.REJECTED:
        st.error(f"{_pv_icon} {_pv_title}")
    else:
        st.warning(f"{_pv_icon} {_pv_title}")
    st.caption(pair_val_reason)
    # Scale ratio from actual metadata
    if src_gsd_val is not None and ref_gsd_val is not None:
        st.info(f"📐 {_scale_ratio_display}")
    else:
        st.caption(f"📐 {_scale_ratio_display}")
    # Sensor mode
    is_cross_sensor = classify_sensor_path(source_sensor, reference_sensor) == "different"
    st.caption(f"Mode: {'Cross-Mission / Cross-Sensor' if is_cross_sensor else 'Same-Sensor Modality'}")

# Scientific Pair Summary
with st.expander("📊 Scientific Pair Summary (Metadata vs Derived Parameters)", expanded=False):
    s_col_meta, s_col_derived = st.columns(2)
    with s_col_meta:
        st.markdown("##### AVAILABLE METADATA")
        st.markdown(f"- **Source Product ID:** `{source_meta.get('product_id') or 'Not available'}`  *(provenance: {source_status['label']})*")
        st.markdown(f"- **Reference Product ID:** `{reference_meta.get('product_id') or 'Not available'}`  *(provenance: {reference_status['label']})*")
        st.markdown(f"- **Source Resolution (GSD):** `{f'{src_gsd_val} m/pixel' if src_gsd_val is not None else 'Not available'}`")
        st.markdown(f"- **Reference Resolution (GSD):** `{f'{ref_gsd_val} m/pixel' if ref_gsd_val is not None else 'Not available'}`")
        st.markdown(f"- **Source Calibration:** `{source_meta.get('processing_level') or 'Not available'}`")
        st.markdown(f"- **Reference State:** `{reference_meta.get('processing_level') or 'Not available'}`")
        st.markdown(f"- **Source Sun Geometry:** Az: `{source_meta.get('sun_azimuth_deg') or '—'}°`, El: `{source_meta.get('sun_elevation_deg') or '—'}°`")
        st.markdown(f"- **Reference Sun Geometry:** Az: `{reference_meta.get('sun_azimuth_deg') or '—'}°`, El: `{reference_meta.get('sun_elevation_deg') or '—'}°`")
    with s_col_derived:
        st.markdown("##### DERIVED PARAMETERS")
        st.markdown(f"- **Scale Difference:** `{_scale_ratio_display}`")
        _ov_area = footprint_eval.get('overlap_area_sq_deg')
        _ov_str = f"{_ov_area:.4f} sq. deg" if _ov_area else "Not evaluated (footprint pending)"
        st.markdown(f"- **Geographic Overlap Area:** `{_ov_str}`")
        if source_meta.get("sun_elevation_deg") is not None and reference_meta.get("sun_elevation_deg") is not None:
            el_diff = abs(float(source_meta["sun_elevation_deg"]) - float(reference_meta["sun_elevation_deg"]))
            st.markdown(f"- **Illumination Elevation Difference:** `~{el_diff:.2f}°`")
        else:
            st.markdown("- **Illumination Difference:** `Not available — sun angles pending`")
        st.markdown(f"- **Sensor Domain Relation:** `{'Cross-Mission (Chandrayaan-2 → Lunar Reference)' if is_cross_sensor else 'Same-Sensor Modality'}`")
        st.markdown(f"- **Pair Validation Status:** `{pair_val_status.value}`")
        st.markdown(f"- **Overlap Gate:** `{footprint_eval.get('status', 'pending').upper()}`")

if not (source_file and reference_file):
    st.info("ℹ️ **Metadata pair configured.** Upload/select the actual product files to execute processing.")

if source_file and reference_file:
    try:
        si = inspect_image(source_file.getvalue(), source_file.name, metadata=source_meta)
        ri = inspect_image(reference_file.getvalue(), reference_file.name, metadata=reference_meta)
        with st.expander("Image metadata (raster inspection)", expanded=False):
            if ri.get("decoding_status") == "PENDING_METADATA":
                st.info(f"ℹ️ **Reference Raster:** {ri.get('message')}")
            if si.get("decoding_status") == "PENDING_METADATA":
                st.info(f"ℹ️ **Source Raster:** {si.get('message')}")
            st.json({"source": si, "reference": ri})
    except Exception as exc:
        logger.exception("Image inspection exception")
        st.warning("Raster inspection requires compatible metadata to decode raw binary geometry.")

with st.expander("Pipeline controls", expanded=True):
    a, b, c = st.columns(3)
    with a:
        source_band = st.number_input("Source band", min_value=1, value=1, step=1)
        max_side = st.select_slider("Maximum processing side (PROCESSING LIMIT)", [1024, 1600, 2048, 3072, 4096], value=2048, help="Processing limit for memory-safe feature extraction; scientific source data remains untouched.")
        if source_sensor == reference_sensor:
            feature_count = st.slider("SIFT features / representation", 3000, 30000, 12000, 1000)
            loftr_confidence_threshold = 0.35
        else:
            feature_count = 12000
            loftr_confidence_threshold = st.slider("LoFTR confidence threshold", 0.10, 0.90, 0.35, 0.05)
    with b:
        reference_band = st.number_input("Reference band", min_value=1, value=1, step=1)
        ratio_threshold = st.slider("Reciprocal descriptor ratio", 0.55, 0.90, 0.78, 0.01)
        grid_size = st.slider("Spatial grid N×N", 4, 16, 10)
    with c:
        model = st.selectbox("Global geometric model", ["Homography", "Affine"])
        geom_verifier_ui = st.selectbox("Geometric Verification", ["MAGSAC++ (Recommended / Default)", "RANSAC (Baseline)"], index=0)
        verifier_selected = "MAGSAC++" if "MAGSAC" in geom_verifier_ui else "RANSAC"
        ransac_threshold = st.slider("RANSAC/MAGSAC threshold (working px)", 0.5, 6.0, 2.5, 0.25)
        cell_limit = st.slider("Maximum matches per cell", 5, 50, 20)

    subpixel = st.checkbox("Enable local sub-pixel refinement", True)
    residual_correction = st.checkbox("Enable smooth terrain/model residual correction", True)
    enable_comparison = st.checkbox("Run experimental MAGSAC++ vs RANSAC comparison", False)

execution_mode = st.radio(
    "Execution mode",
    ["Scientific / validated mode", "Experimental image-only mode"],
    horizontal=True,
    help="Image-only mode does not claim geographic overlap, geographic consistency, or metadata-based scale validation.",
)
scientific_ready = bool(
    source_file and reference_file
    and source_status["status"] not in {"INCOMPLETE", "NOT_PROVIDED"}
    and reference_status["status"] not in {"INCOMPLETE", "NOT_PROVIDED"}
    and footprint_eval.get("has_overlap") is True
)

ref_ptype = classify_product(reference_file.name if reference_file else "")
ref_is_science_binary = ref_ptype in (
    ProductType.LRO_PDS3_BINARY, ProductType.CHANDRAYAAN_PDS4_BINARY, ProductType.SCIENTIFIC_BINARY
)
ref_can_decode = not ref_is_science_binary or bool(reference_meta.get("dimensions", {}).get("lines"))
image_only_ready = bool(source_file and reference_file and ref_can_decode)

if execution_mode == "Scientific / validated mode" and not scientific_ready:
    st.error("RUN BLOCKED: Scientific mode requires source/reference metadata and confirmed geographic overlap.")
elif execution_mode == "Experimental image-only mode":
    if not ref_can_decode:
        st.warning(
            "RUN BLOCKED: Reference is a scientific binary raster (.IMG). "
            "Raster decoding requires lines/samples dimensions. Upload a label or enter dimensions in the manual entry form."
        )
    elif image_only_ready:
        st.warning("EXPERIMENTAL - NO GEOGRAPHIC VALIDATION")

run = st.button(
    "🚀 Run LunaMatch V3",
    type="primary",
    disabled=not (scientific_ready if execution_mode == "Scientific / validated mode" else image_only_ready),
)

if run:
    try:
        with st.spinner("Running multimodal correspondence → robust geometry → validation → refinement..."):
            result = run_pipeline(
                source_file, reference_file, source_sensor, reference_sensor,
                int(source_band), int(reference_band), int(max_side), int(feature_count),
                float(ratio_threshold), int(grid_size), int(cell_limit), model,
                float(ransac_threshold), bool(subpixel), bool(residual_correction),
                source_metadata=source_meta,
                reference_metadata=reference_meta,
                geometric_verifier=verifier_selected,
                loftr_confidence_threshold=float(loftr_confidence_threshold),
                iirs_representation=iirs_representation,
                pca_component=int(iirs_pca_component),
                iirs_normalization=iirs_normalization,
                iirs_invalid_handling=iirs_invalid_handling,
                iirs_resolution_handling=iirs_resolution_handling,
            )
        st.session_state["result"] = result

        if enable_comparison:
            alt_verifier = "RANSAC" if verifier_selected == "MAGSAC++" else "MAGSAC++"
            with st.spinner(f"Running comparison baseline run with {alt_verifier}..."):
                comp_result = run_pipeline(
                    source_file, reference_file, source_sensor, reference_sensor,
                    int(source_band), int(reference_band), int(max_side), int(feature_count),
                    float(ratio_threshold), int(grid_size), int(cell_limit), model,
                    float(ransac_threshold), bool(subpixel), bool(residual_correction),
                    source_metadata=source_meta,
                    reference_metadata=reference_meta,
                    geometric_verifier=alt_verifier,
                    loftr_confidence_threshold=float(loftr_confidence_threshold),
                    iirs_representation=iirs_representation,
                    pca_component=int(iirs_pca_component),
                    iirs_normalization=iirs_normalization,
                    iirs_invalid_handling=iirs_invalid_handling,
                    iirs_resolution_handling=iirs_resolution_handling,
                )

            st.session_state["comp_result"] = comp_result
            st.session_state["comp_verifier"] = alt_verifier
        else:
            st.session_state.pop("comp_result", None)

        st.success("Pipeline completed. Review validation and correspondence coverage before accepting the result.")
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")

result = st.session_state.get("result")
if result:
    report = result["report"]
    stages = result.get("stages", [])

    # Extract metrics accurately from report and stages
    fused_matches = report.get("total_fused_correspondences", len(result.get("matches", [])))
    inliers = report.get("training_inliers", 0)
    inlier_ratio_val = report.get("training_inlier_ratio", 0.0)
    inlier_ratio_str = f"{inlier_ratio_val * 100:.1f} %"

    holdout_rmse_val = report.get("holdout_rmse_working_px")
    if holdout_rmse_val is not None:
        rmse_str = f"{holdout_rmse_val:.3f} px"
    else:
        fit_rmse_val = report.get("fit_rmse_working_px")
        rmse_str = f"{fit_rmse_val:.3f} px (fit)" if fit_rmse_val is not None else "—"

    overlap_val = report.get("reference_overlap_fraction", 0.0)
    overlap_str = f"{overlap_val * 100:.1f} %"

    cov_val = 0.0
    for stg in stages:
        res_m = stg.get("result_metric", {})
        if "spatial_coverage" in res_m:
            cov_val = float(res_m["spatial_coverage"])
            break
        elif "Spatial Coverage" in res_m:
            cov_val = float(res_m["Spatial Coverage"])
            break
    coverage_str = f"{cov_val * 100:.1f} %" if cov_val > 0 else f"{overlap_val * 100:.1f} %"

    selected_model = model if "model" in locals() else "Homography"

    # ==================================================
    # 1. RESULT SUMMARY AT TOP (6 Compact Metric Cards)
    # ==================================================
    st.markdown("---")
    st.markdown("### ⚡ Result Summary")

    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("Fused Matches", f"{fused_matches:,}")
    m_col2.metric("Training Inliers", f"{inliers:,}")
    m_col3.metric("Inlier Ratio", inlier_ratio_str)
    m_col4.metric("Holdout RMSE", rmse_str)
    m_col5.metric("Overlap", overlap_str)
    m_col6.metric("Spatial Coverage", coverage_str)

    st.warning("⚠️ **Validation Status:** Prototype correspondence validation (holdout residuals). Independent checkpoints/DEM are required before claiming certified sub-pixel accuracy.")

    # ==================================================
    # 5. COMPACT HORIZONTAL PIPELINE STAGE TRACE
    # ==================================================
    st.markdown("##### Pipeline Stage Trace")
    trace_items = []
    for s in stages:
        s_id = s.get("stage_id", "")
        s_name = s.get("name", "")
        s_status = s.get("status", "success")
        icon = "✓" if s_status == "success" else ("⚠️" if s_status == "warning" else "✕")
        trace_items.append(f"`{s_id} {s_name} {icon}`")
    st.markdown(" → ".join(trace_items))

    # ==================================================
    # 7. RESULT TABS
    # ==================================================
    tab_overview, tab_matches, tab_reg, tab_metrics, tab_report = st.tabs([
        "Overview",
        "Correspondences",
        "Registration",
        "Metrics",
        "Report"
    ])

    ms = report.get("matching_strategy", {})
    actual_m = ms.get("matcher", "—")
    actual_gv = ms.get("geometric_verifier", "MAGSAC++")

    # --- TAB 1: OVERVIEW ---
    with tab_overview:
        st.markdown("### IMAGE REGISTRATION")
        st.caption(f"**Source → Reference** &nbsp;|&nbsp; Matcher: `{actual_m}` &nbsp;|&nbsp; Verifier: `{actual_gv}` &nbsp;|&nbsp; Model: `{selected_model}`")

        # --- MATCHING STRATEGY CARD ---
        sensor_rel = ms.get("sensor_relation", "unknown")
        rel_label = "SAME SENSOR" if sensor_rel == "same_sensor" else "CROSS-SENSOR"
        st.markdown("#### MATCHING STRATEGY")
        ms_c1, ms_c2, ms_c3 = st.columns(3)
        with ms_c1:
            st.markdown(f"**Sensor relation:** `{rel_label}`")
            st.markdown(f"**Source sensor:** `{ms.get('source_sensor', '—')}`")
            st.markdown(f"**Reference sensor:** `{ms.get('reference_sensor', '—')}`")
        with ms_c2:
            st.markdown(f"**Matcher:** `{actual_m}`")
            st.markdown(f"**Geometric Verifier:** `{actual_gv}`")
            st.markdown(f"**Reason:** {ms.get('reason', '—')}")
        with ms_c3:
            if actual_m == "LoFTR":
                lm = ms.get("loftr_metrics", {})
                st.success(
                    f"✅ **Genuine LoFTR Active** ({lm.get('device', 'cpu').upper()}): "
                    f"{lm.get('raw_matches', 0)} raw matches → {lm.get('filtered_matches', 0)} confident (threshold={lm.get('min_confidence', 0.35):.2f})."
                )
            elif sensor_rel == "same_sensor":
                st.success("✅ **Same-sensor SIFT** with Gaussian pyramid level 1 consumed for multi-scale robustness.")
            else:
                st.warning("⚠️ LoFTR not available; see technical details.")

        st.markdown("---")

        # Side by Side: Source (Moving) and Reference (Fixed)
        st.markdown("#### Input Rasters")
        ov_c1, ov_c2 = st.columns(2)
        with ov_c1:
            st.markdown("**SOURCE / MOVING IMAGE**")
            src_disp, src_diag = make_display_preview(result["source"].gray, max_dim=750)
            st.image(src_disp, use_container_width=True)
            st.caption(f"Preview: {src_diag['shape'][1]}×{src_diag['shape'][0]} px | Full Resolution: {result['source'].info.get('width')}×{result['source'].info.get('height')} px")
        with ov_c2:
            st.markdown("**REFERENCE / FIXED IMAGE**")
            ref_disp, ref_diag = make_display_preview(result["reference"].gray, max_dim=750)
            st.image(ref_disp, use_container_width=True)
            st.caption(f"Preview: {ref_diag['shape'][1]}×{ref_diag['shape'][0]} px | Full Resolution: {result['reference'].info.get('width')}×{result['reference'].info.get('height')} px")

        # Overlap ROI Display
        st.markdown("#### Geographic Overlap Region of Interest (ROI)")
        src_roi, ref_roi, roi_info = extract_overlap_rois(result["source"].gray, result["reference"].gray, source_meta, reference_meta)
        roi_col1, roi_col2 = st.columns(2)
        with roi_col1:
            st.markdown("**Source Overlap ROI**")
            src_roi_disp, _ = make_display_preview(src_roi, max_dim=500)
            st.image(src_roi_disp, use_container_width=True, caption=f"Source Overlap Region ({src_roi.shape[1]}×{src_roi.shape[0]} px)")
        with roi_col2:
            st.markdown("**Reference Overlap ROI**")
            ref_roi_disp, _ = make_display_preview(ref_roi, max_dim=500)
            st.image(ref_roi_disp, use_container_width=True, caption=f"Reference Overlap Region ({ref_roi.shape[1]}×{ref_roi.shape[0]} px)")

        # Registered Result Prominently Displayed
        st.markdown("#### REGISTERED IMAGE")
        reg_disp, reg_diag = make_display_preview(result["registered"], max_dim=900)
        st.image(reg_disp, use_container_width=True, caption=f"Source → Reference ({selected_model} via {actual_gv})")

    # --- TAB 2: CORRESPONDENCES ---
    with tab_matches:
        st.markdown("### CORRESPONDENCES")

        # Compact stats above match visualization
        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        stat_c1.markdown(f"**Matched points:** `{fused_matches}`")
        stat_c2.markdown(f"**Geometrically verified:** `{inliers}`")
        stat_c3.markdown(f"**Inlier ratio:** `{inlier_ratio_str}`")
        stat_c4.markdown(f"**Spatial coverage:** `{coverage_str}`")

        # Match visualization preview (contained size & visible contrast)
        corr_disp, corr_diag = make_display_preview(result["correspondence"], max_dim=1100, is_bgr=True)
        st.image(corr_disp, use_container_width=True, caption=f"Green = robust {actual_gv} inliers; Gray = rejected / held-out candidates.")

        with st.expander("Correspondence Image Diagnostics", expanded=False):
            st.json({
                "image shape": corr_diag["shape"],
                "dtype": corr_diag["dtype"],
                "min pixel value": corr_diag["min"],
                "max pixel value": corr_diag["max"],
                "mean pixel value": round(corr_diag["mean"], 2),
                "total_rows": len(result["matches"])
            })

        st.markdown("#### Correspondence Points Table (First 500 rows)")
        st.dataframe(result["matches"].head(500), use_container_width=True)

    # --- TAB 3: REGISTRATION ---
    with tab_reg:
        st.markdown("### Image Registration & Alignment Verification")
        st.caption("Direction: **Source → Reference Registration** (Source / Moving transformed into Reference / Fixed frame)")

        # Interactive Alpha Slider
        alpha_val = st.slider("Alpha Overlay Blend (Registered Source %)", 0.0, 1.0, 0.50, 0.05)
        live_overlay = create_alpha_overlay(result["reference"].gray, result["registered"], result["validity"], alpha=alpha_val)

        r_c1, r_c2 = st.columns(2)
        with r_c1:
            st.markdown(f"**{int((1-alpha_val)*100)}/{int(alpha_val*100)} Reference + Registered Source Overlay**")
            ov_disp, ov_diag = make_display_preview(live_overlay, max_dim=800, is_bgr=True)
            st.image(ov_disp, use_container_width=True, caption=f"Alpha overlay (alpha={alpha_val:.2f})")
        with r_c2:
            st.markdown(f"**Registered Source Image ({selected_model})**")
            st.image(reg_disp, use_container_width=True, caption=f"Warped source image aligned to reference frame ({actual_gv})")
            if result.get("corrected") is not None:
                st.markdown("**After Smooth Residual Correction**")
                corr_disp_img, _ = make_display_preview(result["corrected"], max_dim=800)
                st.image(corr_disp_img, use_container_width=True, caption="Smooth residual field correction")

        # Checkerboard Comparison
        st.markdown("#### 🏁 Diagnostic Checkerboard Comparison")
        tile_sz = st.slider("Checkerboard Tile Size (px)", 16, 128, 64, 16)
        checkerboard_img = create_checkerboard_comparison(result["reference"].gray, result["registered"], tile_size=tile_sz)
        cb_disp, _ = make_display_preview(checkerboard_img, max_dim=850, is_bgr=True)
        st.image(cb_disp, use_container_width=True, caption=f"Checkerboard alignment ({tile_sz}×{tile_sz} px tiles): Alternating Reference and Registered Source")

        with st.expander("Registered Image Diagnostics", expanded=False):
            st.json({
                "image shape": reg_diag["shape"],
                "dtype": reg_diag["dtype"],
                "min pixel value": reg_diag["min"],
                "max pixel value": reg_diag["max"],
                "mean pixel value": round(reg_diag["mean"], 2),
            })

        st.markdown("#### Geometric Transformation Matrix")
        h_matrix = np.array(report.get("H_source_working_to_reference_working", np.eye(3)))
        st.caption(f"Working resolution transformation matrix (H via {actual_gv}):")
        st.code(np.array2string(h_matrix, precision=6, suppress_small=True))

    # --- TAB 4: METRICS ---
    with tab_metrics:
        st.markdown("### Registration & Precision Metrics")

        met_col1, met_col2 = st.columns(2)
        with met_col1:
            st.markdown("#### Precision & Error")
            st.markdown(f"- **Holdout RMSE:** `{rmse_str}`")
            st.markdown(f"- **Holdout Median:** `{report.get('holdout_median_working_px', '—')}` px")
            st.markdown(f"- **Holdout P95:** `{report.get('holdout_p95_working_px', '—')}` px")
            st.markdown(f"- **Fit RMSE (Working px):** `{report.get('fit_rmse_working_px', 0.0):.3f} px`")
            st.markdown(f"- **Fit RMSE (Original px):** `{report.get('fit_rmse_original_px', 0.0):.3f} px`")
            st.markdown(f"- **Match Count:** `{fused_matches}`")
            st.markdown(f"- **Inlier Count:** `{inliers}`")
            st.markdown(f"- **Inlier Ratio:** `{inlier_ratio_str}`")
            st.markdown(f"- **Spatial Coverage:** `{coverage_str}`")
            total_time = report.get("processing_time_seconds")
            if total_time is None and stages:
                total_time = sum(s.get("processing_time_seconds", 0.0) for s in stages)
            st.markdown(f"- **Processing Time:** `{total_time:.2f} s`" if total_time else "- **Processing Time:** `—`")

        with met_col2:
            st.markdown("#### Pipeline Configuration")
            st.markdown(f"- **Feature Matching:** `{actual_m}` pipeline")
            st.markdown(f"- **Geometric Verification:** `{actual_gv}` (`{selected_model}`)")
            sub_applied = report.get("subpixel_refinement_applied", False)
            st.markdown(f"- **Sub-pixel refinement:** `{'Applied' if sub_applied else 'Not Applied / Disabled'}`")
            st.markdown(f"- **Overlap:** `{overlap_str}`")
            st.markdown(f"- **Smooth Residual Correction:** `{'Applied' if report.get('smooth_residual_applied') else 'Disabled'}`")
            st.markdown(f"- **Validation Status:** `Prototype correspondence validation (holdout)`")

        # Optional MAGSAC++ vs RANSAC comparison view
        if "comp_result" in st.session_state:
            comp_r = st.session_state["comp_result"]
            comp_rep = comp_r["report"]
            comp_name = st.session_state.get("comp_verifier", "Comparison")
            st.markdown(f"#### 🔬 Verifier Comparison: {actual_gv} vs {comp_name}")
            comp_df = pd.DataFrame({
                "Metric": ["Verifier", "Inlier Count", "Inlier Ratio", "Fit RMSE (px)", "Holdout RMSE (px)", "Spatial Coverage"],
                f"Primary ({actual_gv})": [
                    actual_gv,
                    inliers,
                    inlier_ratio_str,
                    f"{report.get('fit_rmse_working_px', 0.0):.3f}",
                    rmse_str,
                    coverage_str,
                ],
                f"Baseline ({comp_name})": [
                    comp_name,
                    comp_rep.get("training_inliers", 0),
                    f"{comp_rep.get('training_inlier_ratio', 0.0) * 100:.1f} %",
                    f"{comp_rep.get('fit_rmse_working_px', 0.0):.3f}",
                    f"{comp_rep.get('holdout_rmse_working_px', 0.0):.3f} px" if comp_rep.get("holdout_rmse_working_px") is not None else "—",
                    f"{comp_rep.get('reference_overlap_fraction', 0.0) * 100:.1f} %",
                ]
            })
            st.dataframe(comp_df, use_container_width=True)

        st.markdown("#### Multi-Representation / Feature Matching Breakdown")
        st.json(report.get("matching_models", {}))

    # --- TAB 5: REPORT ---
    with tab_report:
        st.markdown("### Scientific Summary")
        st.markdown(f"- **Pipeline:** {report.get('pipeline')}")
        st.markdown(f"- **Source File:** `{report.get('source_filename')}` ({report.get('source_sensor')})")
        st.markdown(f"- **Reference File:** `{report.get('reference_filename')}` ({report.get('reference_sensor')})")
        st.markdown(f"- **Matcher:** `{actual_m}`")
        st.markdown(f"- **Geometric Verifier:** `{actual_gv}`")
        st.markdown(f"- **Geometric Model:** `{selected_model}`")
        st.markdown(f"- **Correspondence Verification:** `{inliers} / {fused_matches} inliers ({inlier_ratio_str})`")

        st.markdown("#### Scientific Caveats")
        for caveat in report.get("scientific_caveats", []):
            st.markdown(f"- {caveat}")

        st.markdown("#### Export Artifacts")
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Registered PNG", png_bytes(result["registered"]), "lunamatch_registered.png", "image/png")
        d2.download_button("Validity mask", png_bytes(result["validity"]), "lunamatch_validity.png", "image/png")
        d3.download_button("Correspondences CSV", result["matches"].to_csv(index=False).encode(), "lunamatch_correspondences.csv", "text/csv")
        d4.download_button("Research report JSON", json.dumps(report, indent=2).encode(), "lunamatch_report.json", "application/json")

        st.markdown("#### Experiment Persistence")
        active_pair_id = "PAIR_001" if is_pair_001 else f"EXP_{str(source_meta.get('product_id') or 'SRC')[:16]}_{str(reference_meta.get('product_id') or 'REF')[:16]}"
        if st.button(f"💾 Save Experiment Record ({active_pair_id})", type="secondary"):
            try:
                exp_mgr = ExperimentManager()
                saved_dir = exp_mgr.save_experiment(
                    pair_id=active_pair_id,
                    source_meta=source_meta,
                    reference_meta=reference_meta,
                    config={
                        "model": selected_model,
                        "verifier": actual_gv,
                        "max_side": max_side,
                        "grid_size": grid_size,
                        "cell_limit": cell_limit,
                        "subpixel": subpixel,
                    },
                    metrics=report,
                    source_preview=result["source"].gray,
                    reference_preview=result["reference"].gray,
                    registered_img=result["registered"],
                    correspondence_img=result["correspondence"],
                )
                st.success(f"✓ Experiment record saved to `{saved_dir}`")
            except Exception as exp_save_err:
                st.error(f"Failed to save experiment: {exp_save_err}")

    # ==================================================
    # 6. TECHNICAL PIPELINE DETAILS (COLLAPSED AT BOTTOM)
    # ==================================================
    st.markdown("---")
    with st.expander("TECHNICAL PIPELINE DETAILS ▾", expanded=False):
        st.markdown("### 🔍 Technical Debug / Transparency Panel")
        db_c1, db_c2 = st.columns(2)
        with db_c1:
            st.markdown("##### Sensor Routing")
            st.markdown(f"- **Source Sensor:** `{source_sensor}`")
            st.markdown(f"- **Reference Sensor:** `{reference_sensor}`")
            st.markdown(f"- **Routing Branch:** `{ms.get('sensor_relation', '—')}`")
            st.markdown(f"- **Selected Matcher:** `{actual_m}`")

            st.markdown("##### Matcher Runtime")
            if actual_m == "LoFTR":
                lm = ms.get("loftr_metrics", {})
                st.markdown(f"- **Model:** Pretrained outdoor LoFTR")
                st.markdown(f"- **Execution Device:** `{lm.get('device', 'cpu')}`")
                st.markdown(f"- **Working Source Dimensions:** `{lm.get('working_source_shape', '—')}`")
                st.markdown(f"- **Working Reference Dimensions:** `{lm.get('working_reference_shape', '—')}`")
                st.markdown(f"- **Raw Match Count:** `{lm.get('raw_matches', 0)}`")
                st.markdown(f"- **Confidence Threshold:** `{lm.get('min_confidence', 0.35)}`")
                st.markdown(f"- **Confidence-Filtered Count:** `{lm.get('filtered_matches', 0)}`")
            else:
                st.markdown(f"- **SIFT Runs:** `{list(report.get('matching_models', {}).keys())}`")
                st.markdown(f"- **Total Multi-Rep Candidates:** `{sum(report.get('matching_models', {}).values())}`")
        with db_c2:
            st.markdown("##### Geometric Verification (MAGSAC++ / RANSAC)")
            st.markdown(f"- **Verifier Executed:** `{actual_gv}`")
            st.markdown(f"- **Primary Verifier Requested:** `{ms.get('primary_verifier', 'MAGSAC++')}`")
            st.markdown(f"- **Fallback Used:** `{ms.get('fallback_used', False)}`")
            if ms.get("fallback_reason"):
                st.markdown(f"- **Fallback Reason:** `{ms.get('fallback_reason')}`")
            st.markdown(f"- **Transformation Model:** `{selected_model}`")
            st.markdown(f"- **Verified Inliers:** `{inliers}`")
            st.markdown(f"- **Inlier Ratio:** `{inlier_ratio_str}`")

            st.markdown("##### Spatial & Refinement")
            st.markdown(f"- **Spatial Grid:** `{grid_size}×{grid_size}`")
            st.markdown(f"- **Spatial Coverage:** `{coverage_str}`")
            sub_applied = report.get("subpixel_refinement_applied", False)
            st.markdown(f"- **Sub-pixel Refinement:** `{'Applied (cv2.cornerSubPix)' if sub_applied else 'Not applied'}`")

        st.markdown("---")
        st.markdown("#### Stage Dictionaries & Diagnostics")
        for stage in stages:
            s_status = stage.get("status", "success")
            badge = "🟢" if s_status == "success" else ("🟡" if s_status == "warning" else "🔴")
            with st.expander(f"{badge} {stage.get('stage_id', '')} {stage.get('name', '')}: {stage.get('output', '')}", expanded=False):
                st.json({
                    "stage_id": stage.get("stage_id"),
                    "name": stage.get("name"),
                    "status": stage.get("status"),
                    "processing_method": stage.get("processing_method"),
                    "important_parameters": stage.get("important_parameters"),
                    "result_metric": stage.get("result_metric"),
                    "processing_time_seconds": round(stage.get("processing_time_seconds", 0.0), 4),
                    "error_message": stage.get("error_message"),
                })
        st.markdown("#### Raw Pipeline Report")
        st.json(report)
