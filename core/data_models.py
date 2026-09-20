"""
Data models for LunaMatch V3 (SIH26166).

Defines the core data contracts for:
- Source / Moving image (Chandrayaan-2: OHRC, TMC-2, IIRS)
- Reference / Fixed image (Independent Lunar Reference: LRO NAC, SELENE, etc.)
- Pair validation and scale relationship
- Preconfigured demo pair PAIR_001
- Experiment serialization records
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import json


@dataclass
class FootprintCoordinates:
    upper_left: List[float] = field(default_factory=list)
    upper_right: List[float] = field(default_factory=list)
    lower_left: List[float] = field(default_factory=list)
    lower_right: List[float] = field(default_factory=list)

    def is_valid(self) -> bool:
        corners = [self.upper_left, self.upper_right, self.lower_left, self.lower_right]
        return all(isinstance(c, (list, tuple)) and len(c) == 2 for c in corners)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "upper_left": list(self.upper_left),
            "upper_right": list(self.upper_right),
            "lower_left": list(self.lower_left),
            "lower_right": list(self.lower_right),
        }


@dataclass
class SunGeometry:
    azimuth_deg: Optional[float] = None
    elevation_deg: Optional[float] = None
    incidence_deg: Optional[float] = None
    phase_deg: Optional[float] = None


@dataclass
class CameraGeometry:
    roll_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    yaw_deg: Optional[float] = None
    altitude_km: Optional[float] = None
    emission_deg: Optional[float] = None


@dataclass
class ImageMetadataRecord:
    mission: str = "Unknown"
    instrument: str = "Unknown"
    product_id: str = ""
    product_type: str = ""
    acquisition_time: str = ""
    resolution_gsd_m: Optional[float] = None
    lines: Optional[int] = None
    samples: Optional[int] = None
    footprint: FootprintCoordinates = field(default_factory=FootprintCoordinates)
    sun_geometry: SunGeometry = field(default_factory=SunGeometry)
    camera_geometry: CameraGeometry = field(default_factory=CameraGeometry)
    data_format: str = "Unknown"
    calibration_state: str = "Unknown"
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Convert to the canonical dictionary format expected by app_v3."""
        fp_dict = self.footprint.to_dict()
        return {
            "mission": self.mission,
            "instrument": self.instrument,
            "sensor_type": self.instrument.upper() if self.instrument else "Unknown",
            "product_id": self.product_id,
            "processing_level": self.calibration_state or self.product_type,
            "start_time": self.acquisition_time,
            "stop_time": self.acquisition_time,
            "gsd_m_per_pixel": self.resolution_gsd_m,
            "altitude_km": self.camera_geometry.altitude_km,
            "roll_deg": self.camera_geometry.roll_deg,
            "pitch_deg": self.camera_geometry.pitch_deg,
            "yaw_deg": self.camera_geometry.yaw_deg,
            "sun_azimuth_deg": self.sun_geometry.azimuth_deg,
            "sun_elevation_deg": self.sun_geometry.elevation_deg,
            "solar_incidence_deg": self.sun_geometry.incidence_deg,
            "footprint": fp_dict,
            "dimensions": {"lines": self.lines, "samples": self.samples},
            "data_type": self.data_format,
            "valid": True,
            "validation_errors": [],
            "validation_warnings": [],
        }


@dataclass
class PairValidationResult:
    is_valid: bool
    has_overlap: bool
    overlap_area_sq_deg: float
    scale_ratio: Optional[float]
    scale_difference_str: str
    source_resolution_m: Optional[float]
    reference_resolution_m: Optional[float]
    sun_elevation_diff_deg: Optional[float]
    sun_azimuth_diff_deg: Optional[float]
    gate_message: str
    validation_card: Dict[str, Any]
    derived_parameters: Dict[str, Any]
    available_metadata: Dict[str, Any]


@dataclass
class SpatialBalancingMetrics:
    grid_rows: int
    grid_cols: int
    total_cells: int
    occupied_cells: int
    coverage_fraction: float
    mean_matches_per_cell: float
    max_matches_per_cell: int
    matches_before: int
    matches_after: int


def calculate_scale_ratio(source_gsd: Optional[float], ref_gsd: Optional[float]) -> Tuple[Optional[float], str]:
    """Calculate scale ratio (ref_gsd / source_gsd) and format string."""
    if source_gsd and ref_gsd and source_gsd > 0 and ref_gsd > 0:
        ratio = ref_gsd / source_gsd
        return ratio, f"Scale Difference ≈ {ratio:.1f}×"
    return None, "Scale Difference unavailable"


# ==================================================
# REAL EXPERIMENT DEMO PAIR: PAIR_001
# ==================================================
# Source: Chandrayaan-2 OHRC calibrated product ch2_ohr_ncp_20190906T2241285714_d_img_gds (~0.24 m/px)
# Reference: LRO LROC NAC EDR M1534362340LE (~1.8669 m/px)
# Scale ratio: 1.8669 / 0.24 ≈ 7.78 -> Display as "Scale Difference ≈ 7.8×"

PAIR_001_SOURCE = ImageMetadataRecord(
    mission="Chandrayaan-2",
    instrument="OHRC",
    product_id="ch2_ohr_ncp_20190906T2241285714_d_img_gds",
    product_type="IMG",
    acquisition_time="2019-09-06T22:41:28.5714Z",
    resolution_gsd_m=0.24,
    lines=80000,
    samples=12000,
    footprint=FootprintCoordinates(
        upper_left=[-70.85, 22.75],
        upper_right=[-70.86, 23.45],
        lower_left=[-71.45, 22.70],
        lower_right=[-71.46, 23.40],
    ),
    sun_geometry=SunGeometry(
        azimuth_deg=65.4,
        elevation_deg=14.8,
        incidence_deg=75.2,
    ),
    camera_geometry=CameraGeometry(
        roll_deg=0.12,
        pitch_deg=0.08,
        yaw_deg=-0.02,
        altitude_km=100.2,
    ),
    data_format="UnsignedByte",
    calibration_state="Calibrated",
)

PAIR_001_REFERENCE = ImageMetadataRecord(
    mission="Lunar Reconnaissance Orbiter",
    instrument="LROC NAC",
    product_id="M1534362340LE",
    product_type="EDR",
    acquisition_time="2021-03-12T14:15:30.000Z",
    resolution_gsd_m=1.8669,
    lines=52224,
    samples=5064,
    footprint=FootprintCoordinates(
        upper_left=[-70.60, 22.40],
        upper_right=[-70.62, 23.80],
        lower_left=[-71.70, 22.35],
        lower_right=[-71.72, 23.75],
    ),
    sun_geometry=SunGeometry(
        azimuth_deg=72.1,
        elevation_deg=18.5,
        incidence_deg=71.5,
        phase_deg=53.2,
    ),
    camera_geometry=CameraGeometry(
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        altitude_km=50.4,
        emission_deg=2.1,
    ),
    data_format="PDS3_RAW",
    calibration_state="EDR",
)

PAIR_001_CONFIG = {
    "pair_id": "PAIR_001",
    "name": "Chandrayaan-2 OHRC → LRO LROC NAC (~7.8× Scale Variation)",
    "description": "Cross-mission optical correspondence test evaluating scale invariance across ~7.8× resolution ratio.",
    "source": PAIR_001_SOURCE,
    "reference": PAIR_001_REFERENCE,
    "approx_scale_ratio": 1.8669 / 0.24,
    "scale_display": "Scale Difference ≈ 7.8×",
}
