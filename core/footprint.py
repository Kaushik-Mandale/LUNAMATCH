"""
Geographic Footprint Validation & Overlap ROI Extraction.

Implements:
1. Exact footprint bounds calculation from 4-corner polygons.
2. Strict overlap gate: rejects non-overlapping pairs with scientific rationale.
3. Common geographic overlap extraction into pixel ROIs for source and reference.
"""
from dataclasses import asdict, dataclass
import math
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

try:
    from rasterio.crs import CRS
    from rasterio.warp import transform as _crs_transform
    _HAS_PROJ = True
except ImportError:
    CRS = None
    _crs_transform = None
    _HAS_PROJ = False

LUNAR_RADIUS_KM = 1737.4
LUNAR_RADIUS_PROVENANCE = "STANDARD_LUNAR_CONSTANT"


def validate_projection(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Validate projection identity and expose lunar-radius provenance."""
    projection = str(meta.get("projection") or "").strip().lower() if isinstance(meta, dict) else ""
    if not projection:
        return {"status": "NOT_PROVIDED", "projection": None, "provenance": "UNKNOWN", "message": "Projection metadata is not available."}
    if not any(token in projection for token in ("polar", "stereographic", "equirectangular")):
        return {"status": "UNSUPPORTED", "projection": meta.get("projection"), "provenance": "PRODUCT_METADATA", "message": f"Projection '{meta.get('projection')}' is not supported by the lunar footprint engine."}
    radius = meta.get("planetary_radius_km") or meta.get("lunar_radius_km") or LUNAR_RADIUS_KM
    provenance = "PRODUCT_METADATA" if meta.get("planetary_radius_km") or meta.get("lunar_radius_km") else LUNAR_RADIUS_PROVENANCE
    parameters = meta.get("projection_parameters") if isinstance(meta.get("projection_parameters"), dict) else {}
    if not _HAS_PROJ:
        return {"status": "PROJECTION_ERROR", "projection": meta.get("projection"), "provenance": provenance, "radius_km": float(radius), "message": "The PROJ projection engine is unavailable."}
    return {
        "status": "VALID",
        "projection": meta.get("projection"),
        "provenance": provenance,
        "radius_km": float(radius),
        "parameters": parameters,
        "crs_engine": "PROJ via rasterio",
        "message": "Lunar polar stereographic projection is available through the PROJ engine.",
    }


@dataclass
class FootprintValidationResult:
    source_polygon: List[Tuple[float, float]]
    reference_polygon: List[Tuple[float, float]]
    overlap_polygon: List[Tuple[float, float]]
    source_area: float
    reference_area: float
    overlap_area: float
    overlap_percentage_source: float
    overlap_percentage_reference: float
    status: str
    gate_passed: bool
    gate_message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def to_polar_stereographic(lat: float, lon: float, r_moon: float = LUNAR_RADIUS_KM) -> Tuple[float, float]:
    """Project lunar South Polar latitude and longitude to planar coordinates (x, y) in km.

    Conformal South Polar Stereographic projection centered at the South Pole (-90 deg).
    """
    co_lat = max(0.0, 90.0 + lat)
    r = 2.0 * r_moon * math.tan(math.radians(co_lat / 2.0))
    rad_lon = math.radians(lon)
    x = r * math.sin(rad_lon)
    y = -r * math.cos(rad_lon)
    return (float(x), float(y))


def from_polar_stereographic(x: float, y: float, r_moon: float = LUNAR_RADIUS_KM) -> Tuple[float, float]:
    """Inverse project from planar coordinates (x, y) in km to (lat, lon)."""
    r = math.hypot(x, y)
    if r < 1e-12:
        return (-90.0, 0.0)
    co_lat = 2.0 * math.degrees(math.atan(r / (2.0 * r_moon)))
    lat = -90.0 + co_lat
    lon = math.degrees(math.atan2(x, -y)) % 360.0
    if math.isclose(lon, 360.0, abs_tol=1e-12):
        lon = 0.0
    return (float(lat), float(lon))


def normalize_longitudes(values: List[float]) -> List[float]:
    """Normalize longitudes into the [0, 360) domain while preserving local continuity."""
    normalized = []
    for value in values:
        if value is None:
            normalized.append(None)
            continue
        wrapped = float(value) % 360.0
        normalized.append(wrapped)
    return normalized


def calculate_overlap_area(polygon: List[Tuple[float, float]]) -> float:
    """Return the polygon area in square degrees for a simple (lat, lon) footprint polygon."""
    if len(polygon) < 3:
        return 0.0
    return abs(_polygon_area(polygon))


def _polygon_area(polygon: List[Tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    ) / 2.0)


def _signed_polygon_area(polygon: List[Tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    ) / 2.0


def _clip_polygon(subject, clip):
    """Sutherland-Hodgman clipping for the convex four-corner footprints."""
    if not subject or not clip:
        return []
    orientation = 1.0 if _signed_polygon_area(clip) >= 0 else -1.0

    def inside(point, start, end):
        cross = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])
        return orientation * cross >= -1e-9

    def intersection(first, second, start, end):
        x1, y1 = first
        x2, y2 = second
        x3, y3 = start
        x4, y4 = end
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-12:
            return second
        factor = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        return [x1 + factor * (x2 - x1), y1 + factor * (y2 - y1)]

    output = list(subject)
    for index, clip_start in enumerate(clip):
        clip_end = clip[(index + 1) % len(clip)]
        input_polygon = output
        output = []
        if not input_polygon:
            break
        previous = input_polygon[-1]
        for current in input_polygon:
            current_inside = inside(current, clip_start, clip_end)
            previous_inside = inside(previous, clip_start, clip_end)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current, clip_start, clip_end))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current, clip_start, clip_end))
            previous = current
    return output


def get_footprint_bounds(meta: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Extract (min_lat, max_lat, min_lon, max_lon) from metadata footprint."""
    if not isinstance(meta, dict):
        return None

    fp = meta.get("footprint")
    if isinstance(fp, dict):
        corners = ["upper_left", "upper_right", "lower_left", "lower_right"]
        pts = [fp.get(c) for c in corners if isinstance(fp.get(c), (list, tuple)) and len(fp.get(c)) == 2]
        if len(pts) == 4:
            lats = [float(p[0]) for p in pts]
            lons = [float(p[1]) for p in pts]
            return min(lats), max(lats), min(lons), max(lons)

    # Fallback to direct latitude / longitude ranges if available
    lat = meta.get("latitude") or meta.get("latitude_deg")
    lon = meta.get("longitude") or meta.get("longitude_deg")
    if lat is not None and lon is not None:
        try:
            flat = float(lat)
            flon = float(lon)
            return flat - 0.1, flat + 0.1, flon - 0.1, flon + 0.1
        except Exception:
            pass

    return None


def _footprint_polygon(meta: Dict[str, Any]) -> Optional[List[Tuple[float, float]]]:
    """Return an ordered lat/lon polygon with longitudes unwrapped locally."""
    fp = meta.get("footprint") if isinstance(meta, dict) else None
    if not isinstance(fp, dict):
        return None
    corners = [fp.get(name) for name in ("upper_left", "upper_right", "lower_right", "lower_left")]
    if any(not isinstance(point, (list, tuple)) or len(point) != 2 for point in corners):
        return None

    polygon = [(float(point[0]), float(point[1]) % 360.0) for point in corners]
    unwrapped = [polygon[0]]
    for lat, lon in polygon[1:]:
        previous_lon = unwrapped[-1][1]
        while lon - previous_lon > 180.0:
            lon -= 360.0
        while lon - previous_lon < -180.0:
            lon += 360.0
        unwrapped.append((lat, lon))
    return unwrapped


def _project_polar_polygon(
    polygon: List[Tuple[float, float]],
    projection_status: Dict[str, Any],
) -> Tuple[Optional[List[Tuple[float, float]]], Dict[str, Any]]:
    """Project a lat/lon footprint through the shared PROJ polar CRS in km."""
    if not polygon or projection_status.get("status") != "VALID" or not _HAS_PROJ:
        return None, {"status": "PROJECTION_ERROR", "message": "Polar projection is unavailable."}

    params = projection_status.get("parameters") or {}
    radius_m = float(projection_status.get("radius_km", LUNAR_RADIUS_KM)) * 1000.0
    try:
        proj4 = (
            "+proj=stere +lat_0={lat_0} +lat_ts={lat_ts} +lon_0={lon_0} "
            "+x_0={x_0} +y_0={y_0} +a={a} +b={b} +units=m +no_defs"
        ).format(
            lat_0=float(params.get("latitude_of_origin", params.get("lat_0", -90.0))),
            lat_ts=float(params.get("latitude_true_scale", params.get("lat_ts", -90.0))),
            lon_0=float(params.get("central_meridian", params.get("lon_0", 0.0))),
            x_0=float(params.get("false_easting", params.get("x_0", 0.0))),
            y_0=float(params.get("false_northing", params.get("y_0", 0.0))),
            a=float(params.get("semi_major_axis_m", params.get("a", radius_m))),
            b=float(params.get("semi_minor_axis_m", params.get("b", radius_m))),
        )
        crs = CRS.from_proj4(proj4)
        geographic_crs = CRS.from_proj4(
            "+proj=longlat +R={radius} +no_defs".format(radius=radius_m)
        )
        lats = [point[0] for point in polygon]
        lons = [point[1] % 360.0 for point in polygon]
        xs, ys = _crs_transform(geographic_crs, crs, lons, lats)
        projected = [(float(x) / 1000.0, float(y) / 1000.0) for x, y in zip(xs, ys)]
        return projected, {
            "status": "VALID",
            "engine": "PROJ via rasterio",
            "crs": crs.to_wkt(),
            "proj4": proj4,
            "vertices_km": projected,
        }
    except Exception as exc:
        return None, {"status": "PROJECTION_ERROR", "message": str(exc)}


def _orientation(a, b, c) -> float:
    return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])


def _segments_intersect(a, b, c, d) -> bool:
    def sign(value):
        return (value > 1e-9) - (value < -1e-9)

    o1, o2 = sign(_orientation(a, b, c)), sign(_orientation(a, b, d))
    o3, o4 = sign(_orientation(c, d, a)), sign(_orientation(c, d, b))
    return o1 != o2 and o3 != o4


def _point_in_polygon(point, polygon) -> bool:
    inside = False
    px, py = point
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        if (y1 > py) != (y2 > py):
            x_at_y = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < x_at_y:
                inside = not inside
    return inside


def _polygons_intersect(first, second) -> bool:
    for index, first_start in enumerate(first):
        first_end = first[(index + 1) % len(first)]
        for second_index, second_start in enumerate(second):
            second_end = second[(second_index + 1) % len(second)]
            if _segments_intersect(first_start, first_end, second_start, second_end):
                return True
    return _point_in_polygon(first[0], second) or _point_in_polygon(second[0], first)


def _polygon_is_valid(polygon) -> bool:
    """Validate a simple polygon before clipping it."""
    if not polygon or len(polygon) < 3 or abs(_signed_polygon_area(polygon)) <= 1e-12:
        return False
    for first_index in range(len(polygon)):
        first_start = polygon[first_index]
        first_end = polygon[(first_index + 1) % len(polygon)]
        if math.hypot(first_end[0] - first_start[0], first_end[1] - first_start[1]) <= 1e-12:
            continue
        for second_index in range(first_index + 1, len(polygon)):
            if second_index in {first_index, (first_index - 1) % len(polygon), (first_index + 1) % len(polygon)}:
                continue
            second_start = polygon[second_index]
            second_end = polygon[(second_index + 1) % len(polygon)]
            if math.hypot(second_end[0] - second_start[0], second_end[1] - second_start[1]) <= 1e-12:
                continue
            if any(
                math.hypot(first_point[0] - second_point[0], first_point[1] - second_point[1]) <= 1e-12
                for first_point in (first_start, first_end)
                for second_point in (second_start, second_end)
            ):
                continue
            if _segments_intersect(first_start, first_end, second_start, second_end):
                return False
    return True


def evaluate_footprint_overlap(
    source_meta: Dict[str, Any],
    reference_meta: Dict[str, Any],
    min_overlap_percentage: float = 0.1,
) -> Dict[str, Any]:
    """Evaluate geographic footprint overlap between source and reference imagery.

    Supports polar stereographic projection for lunar South Polar footprints (lat <= -60 deg).
    Enforces strict gating with configurable minimum overlap threshold.
    """
    src_polygon = _footprint_polygon(source_meta)
    ref_polygon = _footprint_polygon(reference_meta)
    source_projection = validate_projection(source_meta)
    reference_projection = validate_projection(reference_meta)

    def metrics(
        source_area: float,
        reference_area: float,
        intersection_area: float,
        source_area_km2: Optional[float] = None,
        reference_area_km2: Optional[float] = None,
        intersection_area_km2: Optional[float] = None,
    ) -> Dict[str, float]:
        deg_to_km = 1737.4 * math.pi / 180.0
        source_percent = intersection_area / max(source_area, 1e-9) * 100.0
        reference_percent = intersection_area / max(reference_area, 1e-9) * 100.0
        smaller_percent = intersection_area / max(min(source_area, reference_area), 1e-9) * 100.0
        return {
            "source_area": source_area,
            "reference_area": reference_area,
            "overlap_area": intersection_area,
            "source_area_km2": source_area_km2 if source_area_km2 is not None else source_area * (deg_to_km * deg_to_km),
            "reference_area_km2": reference_area_km2 if reference_area_km2 is not None else reference_area * (deg_to_km * deg_to_km),
            "intersection_area_km2": intersection_area_km2 if intersection_area_km2 is not None else intersection_area * (deg_to_km * deg_to_km),
            "overlap_percentage_source": source_percent,
            "overlap_percentage_reference": reference_percent,
            "overlap_percentage_smaller": smaller_percent,
            "intersection_percent_of_source": source_percent,
            "intersection_percent_of_reference": reference_percent,
            "intersection_percent_of_smaller": smaller_percent,
        }

    if src_polygon is None or ref_polygon is None:
        return {
            "available": False,
            "has_overlap": None,
            "overlap": None,
            "overlap_area_sq_deg": 0.0,
            "overlap_area_km2": 0.0,
            "intersection_bounds": None,
            "source_bounds": None,
            "reference_bounds": None,
            "gate_passed": False,
            "gate_message": "Geographic overlap cannot yet be evaluated; one or both footprints are unavailable.",
            "status": "pending",
            "geometry_status": "INSUFFICIENT_GEOMETRY_METADATA",
            "minimum_overlap_percentage": min_overlap_percentage,
            "projection_status": {"source": source_projection, "reference": reference_projection},
        }

    # Detect if footprints are near the lunar South Pole
    projection_names = (
        str(source_projection.get("projection") or "").lower(),
        str(reference_projection.get("projection") or "").lower(),
    )
    is_polar = (
        all(p[0] <= -60.0 for p in src_polygon + ref_polygon)
        and all("polar" in name or "stereographic" in name for name in projection_names)
    )

    if is_polar:
        # Polar stereographic planar space (km)
        if source_projection["status"] != "VALID" or reference_projection["status"] != "VALID":
            return {
                "available": True, "has_overlap": None, "overlap": None,
                "overlap_area_sq_deg": 0.0, "overlap_area_km2": 0.0,
                "intersection_bounds": None, "source_bounds": None, "reference_bounds": None,
                "gate_passed": False,
                "gate_message": "Projection validation is incomplete; geographic overlap was not evaluated.",
                "status": "geometry_error",
                "geometry_status": "PROJECTION_ERROR",
                "minimum_overlap_percentage": min_overlap_percentage,
                "projection_status": {"source": source_projection, "reference": reference_projection},
            }
        src_proj, src_projection_debug = _project_polar_polygon(src_polygon, source_projection)
        ref_proj, ref_projection_debug = _project_polar_polygon(ref_polygon, reference_projection)
        if src_proj is None or ref_proj is None:
            return {
                "available": True,
                "has_overlap": None,
                "overlap": None,
                "overlap_area_sq_deg": 0.0,
                "overlap_area_km2": 0.0,
                "intersection_bounds": None,
                "source_bounds": None,
                "reference_bounds": None,
                "gate_passed": False,
                "gate_message": "Projection failed; geographic overlap was not evaluated.",
                "status": "geometry_error",
                "geometry_status": "PROJECTION_ERROR",
                "projection_status": {"source": source_projection, "reference": reference_projection},
                "projected_source_polygon": src_projection_debug.get("vertices_km", []),
                "projected_reference_polygon": ref_projection_debug.get("vertices_km", []),
                "projection_debug": {"source": src_projection_debug, "reference": ref_projection_debug},
                "minimum_overlap_percentage": min_overlap_percentage,
            }
        if (
            not _polygon_is_valid(src_proj)
            or not _polygon_is_valid(ref_proj)
            or source_projection.get("radius_km") != reference_projection.get("radius_km")
            or source_projection.get("parameters", {}) != reference_projection.get("parameters", {})
        ):
            return {
                "available": True,
                "has_overlap": None,
                "overlap": None,
                "overlap_area_sq_deg": 0.0,
                "overlap_area_km2": 0.0,
                "intersection_bounds": None,
                "source_bounds": None,
                "reference_bounds": None,
                "gate_passed": False,
                "gate_message": "Footprint geometry or shared projection parameters are invalid.",
                "status": "geometry_error",
                "geometry_status": "INVALID_GEOMETRY",
                "projection_status": {"source": source_projection, "reference": reference_projection},
                "projected_source_polygon": src_proj,
                "projected_reference_polygon": ref_proj,
                "projection_debug": {"source": src_projection_debug, "reference": ref_projection_debug},
                "minimum_overlap_percentage": min_overlap_percentage,
            }
        src_radius = source_projection.get("radius_km", LUNAR_RADIUS_KM)
        has_overlap = _polygons_intersect(src_proj, ref_proj)

        src_bounds = (
            min(point[0] for point in src_polygon), max(point[0] for point in src_polygon),
            min(point[1] for point in src_polygon), max(point[1] for point in src_polygon),
        )
        ref_bounds = (
            min(point[0] for point in ref_polygon), max(point[0] for point in ref_polygon),
            min(point[1] for point in ref_polygon), max(point[1] for point in ref_polygon),
        )

        source_area_km2 = float(_polygon_area(src_proj))
        reference_area_km2 = float(_polygon_area(ref_proj))
        deg_to_km = 1737.4 * math.pi / 180.0
        km2_to_sqdeg = 1.0 / (deg_to_km * deg_to_km)
        source_area = source_area_km2 * km2_to_sqdeg
        reference_area = reference_area_km2 * km2_to_sqdeg

        if not has_overlap:
            gate_msg = (
                "No meaningful geographic overlap detected.\n"
                "Matching was not executed because the source and reference\n"
                "images are geographically inconsistent."
            )
            return {
                "available": True,
                "has_overlap": False,
                "overlap": False,
                "overlap_area_sq_deg": 0.0,
                "overlap_area_km2": 0.0,
                "intersection_bounds": None,
                "source_bounds": src_bounds,
                "reference_bounds": ref_bounds,
                "gate_passed": False,
                "gate_message": gate_msg,
                "status": "rejected",
                "geometry_status": "VALID_NO_INTERSECTION",
                "source_polygon": src_polygon,
                "reference_polygon": ref_polygon,
                "projected_source_polygon": src_proj,
                "projected_reference_polygon": ref_proj,
                "overlap_polygon": [],
                "projection_debug": {"source": src_projection_debug, "reference": ref_projection_debug},
                "source_area": source_area,
                "reference_area": reference_area,
                "overlap_area": 0.0,
                **metrics(source_area, reference_area, 0.0),
                "minimum_overlap_percentage": min_overlap_percentage,
            }

        overlap_proj = _clip_polygon(src_proj, ref_proj)
        overlap_area_km2 = float(_polygon_area(overlap_proj))
        overlap_area = overlap_area_km2 * km2_to_sqdeg
        overlap_polygon = [from_polar_stereographic(x, y, src_radius) for x, y in overlap_proj]
        overlap_fraction = float(overlap_area / max(min(source_area, reference_area), 1e-9))
        overlap_metrics = metrics(source_area, reference_area, overlap_area)
        pct_src = overlap_metrics["overlap_percentage_source"]
        pct_ref = overlap_metrics["overlap_percentage_reference"]
        pct_smaller = overlap_metrics["overlap_percentage_smaller"]

        inter_lats = [p[0] for p in overlap_polygon] if overlap_polygon else []
        inter_lons = [p[1] for p in overlap_polygon] if overlap_polygon else []
        inter_bounds = (min(inter_lats), max(inter_lats), min(inter_lons), max(inter_lons)) if inter_lats else None

        if pct_smaller < min_overlap_percentage:
            return {
                "available": True,
                "has_overlap": True,
                "overlap": False,
                "overlap_area_sq_deg": overlap_area,
                "overlap_area_km2": overlap_area_km2,
                "overlap_fraction": overlap_fraction,
                "source_polygon": src_polygon,
                "reference_polygon": ref_polygon,
                "overlap_polygon": overlap_polygon,
                **metrics(source_area, reference_area, 0.0, source_area_km2, reference_area_km2, 0.0),
                "intersection_bounds": inter_bounds,
                "source_bounds": src_bounds,
                "reference_bounds": ref_bounds,
                "gate_passed": False,
                "minimum_overlap_percentage": min_overlap_percentage,
                "gate_message": f"Insufficient geographic overlap detected ({pct_smaller:.3f}% of the smaller footprint is below the {min_overlap_percentage:.3f}% threshold). Matching blocked.",
                "status": "insufficient_overlap",
                "geometry_status": "VALID_INTERSECTION",
                "projection_status": {"source": source_projection, "reference": reference_projection},
                "projected_source_polygon": src_proj,
                "projected_reference_polygon": ref_proj,
                "projected_overlap_polygon": overlap_proj,
                "projection_debug": {"source": src_projection_debug, "reference": ref_projection_debug},
            }

        return {
            "available": True,
            "has_overlap": True,
            "overlap": True,
            "overlap_area_sq_deg": overlap_area,
            "overlap_area_km2": overlap_area_km2,
            "overlap_fraction": overlap_fraction,
            "source_polygon": src_polygon,
            "reference_polygon": ref_polygon,
            "overlap_polygon": overlap_polygon,
            **overlap_metrics,
            "intersection_bounds": inter_bounds,
            "source_bounds": src_bounds,
            "reference_bounds": ref_bounds,
            "gate_passed": True,
            "minimum_overlap_percentage": min_overlap_percentage,
            "gate_message": f"Geographic overlap confirmed ({pct_smaller:.3f}% of the smaller footprint, {overlap_area:.6f} sq. deg).",
            "status": "validated",
            "geometry_status": "VALID_INTERSECTION",
            "projection_status": {"source": source_projection, "reference": reference_projection},
            "projected_source_polygon": src_proj,
            "projected_reference_polygon": ref_proj,
            "projected_overlap_polygon": overlap_proj,
            "projection_debug": {"source": src_projection_debug, "reference": ref_projection_debug},
        }

    # Standard lat/lon space (equatorial or non-polar)
    src_center = sum(point[1] for point in src_polygon) / len(src_polygon)
    ref_center = sum(point[1] for point in ref_polygon) / len(ref_polygon)
    longitude_shift = round((src_center - ref_center) / 360.0) * 360.0
    ref_polygon = [(lat, lon + longitude_shift) for lat, lon in ref_polygon]

    src_bounds = (
        min(point[0] for point in src_polygon), max(point[0] for point in src_polygon),
        min(point[1] for point in src_polygon), max(point[1] for point in src_polygon),
    )
    ref_bounds = (
        min(point[0] for point in ref_polygon), max(point[0] for point in ref_polygon),
        min(point[1] for point in ref_polygon), max(point[1] for point in ref_polygon),
    )

    has_overlap = _polygons_intersect(src_polygon, ref_polygon)
    s_min_lat, s_max_lat, s_min_lon, s_max_lon = src_bounds
    r_min_lat, r_max_lat, r_min_lon, r_max_lon = ref_bounds

    inter_min_lat = max(s_min_lat, r_min_lat)
    inter_max_lat = min(s_max_lat, r_max_lat)
    inter_min_lon = max(s_min_lon, r_min_lon)
    inter_max_lon = min(s_max_lon, r_max_lon)

    if not has_overlap:
        gate_msg = (
            "No meaningful geographic overlap detected.\n"
            "Matching was not executed because the source and reference\n"
            "images are geographically inconsistent."
        )
        return {
            "available": True,
            "has_overlap": False,
            "overlap": False,
            "overlap_area_sq_deg": 0.0,
            "overlap_area_km2": 0.0,
            "intersection_bounds": None,
            "source_bounds": src_bounds,
            "reference_bounds": ref_bounds,
            "gate_passed": False,
            "gate_message": gate_msg,
            "status": "rejected",
            "geometry_status": "VALID_NO_INTERSECTION",
            "source_polygon": src_polygon,
            "reference_polygon": ref_polygon,
            "projected_source_polygon": None,
            "projected_reference_polygon": None,
            "overlap_polygon": [],
            "source_area": _polygon_area(src_polygon),
            "reference_area": _polygon_area(ref_polygon),
            "overlap_area": 0.0,
            **metrics(_polygon_area(src_polygon), _polygon_area(ref_polygon), 0.0),
            "minimum_overlap_percentage": min_overlap_percentage,
        }

    overlap_polygon = _clip_polygon(src_polygon, ref_polygon)
    overlap_area = float(_polygon_area(overlap_polygon))
    deg_to_km = 1737.4 * math.pi / 180.0
    overlap_area_km2 = float(overlap_area * (deg_to_km * deg_to_km))
    source_area = float(_polygon_area(src_polygon))
    reference_area = float(_polygon_area(ref_polygon))
    overlap_fraction = float(overlap_area / max(min(source_area, reference_area), 1e-9))
    overlap_metrics = metrics(
        source_area,
        reference_area,
        overlap_area,
        intersection_area_km2=overlap_area_km2,
    )
    pct_src = overlap_metrics["overlap_percentage_source"]
    pct_ref = overlap_metrics["overlap_percentage_reference"]
    pct_smaller = overlap_metrics["overlap_percentage_smaller"]

    if pct_smaller < min_overlap_percentage:
        return {
            "available": True,
            "has_overlap": True,
            "overlap": False,
            "overlap_area_sq_deg": overlap_area,
            "overlap_area_km2": overlap_area_km2,
            "overlap_fraction": overlap_fraction,
            "source_polygon": src_polygon,
            "reference_polygon": ref_polygon,
            "overlap_polygon": overlap_polygon,
            **overlap_metrics,
            "intersection_bounds": (inter_min_lat, inter_max_lat, inter_min_lon, inter_max_lon),
            "source_bounds": src_bounds,
            "reference_bounds": ref_bounds,
            "gate_passed": False,
            "minimum_overlap_percentage": min_overlap_percentage,
            "gate_message": f"Insufficient geographic overlap detected ({pct_smaller:.3f}% of the smaller footprint is below the {min_overlap_percentage:.3f}% threshold). Matching blocked.",
            "status": "insufficient_overlap",
            "geometry_status": "VALID_INTERSECTION",
            "projection_status": {"source": source_projection, "reference": reference_projection},
        }

    return {
        "available": True,
        "has_overlap": True,
        "overlap": True,
        "overlap_area_sq_deg": overlap_area,
        "overlap_area_km2": overlap_area_km2,
        "overlap_fraction": overlap_fraction,
        "source_polygon": src_polygon,
        "reference_polygon": ref_polygon,
        "overlap_polygon": overlap_polygon,
        **overlap_metrics,
        "intersection_bounds": (inter_min_lat, inter_max_lat, inter_min_lon, inter_max_lon),
        "source_bounds": src_bounds,
        "reference_bounds": ref_bounds,
        "gate_passed": True,
        "minimum_overlap_percentage": min_overlap_percentage,
        "gate_message": f"Geographic overlap confirmed ({pct_smaller:.3f}% of the smaller footprint, {overlap_area:.6f} sq. deg).",
        "status": "validated",
        "geometry_status": "VALID_INTERSECTION",
        "projection_status": {"source": source_projection, "reference": reference_projection},
    }


def compute_overlap_pixel_roi(
    image_shape: Tuple[int, int],
    image_bounds: Tuple[float, float, float, float],
    intersection_bounds: Tuple[float, float, float, float],
    padding_fraction: float = 0.05,
) -> Tuple[int, int, int, int]:
    """Map geographic intersection bounds to image pixel crop coordinates (y_min, y_max, x_min, x_max).

    image_bounds: (min_lat, max_lat, min_lon, max_lon)
    intersection_bounds: (inter_min_lat, inter_max_lat, inter_min_lon, inter_max_lon)
    """
    h, w = image_shape[:2]
    min_lat, max_lat, min_lon, max_lon = image_bounds
    i_min_lat, i_max_lat, i_min_lon, i_max_lon = intersection_bounds

    lat_span = max(max_lat - min_lat, 1e-9)
    lon_span = max(max_lon - min_lon, 1e-9)

    # Pixel coords (assuming North is Up, West is Left)
    # y=0 is top (max_lat), y=h is bottom (min_lat)
    y_min_norm = (max_lat - i_max_lat) / lat_span
    y_max_norm = (max_lat - i_min_lat) / lat_span

    # x=0 is left (min_lon), x=w is right (max_lon)
    x_min_norm = (i_min_lon - min_lon) / lon_span
    x_max_norm = (i_max_lon - min_lon) / lon_span

    # Add margin padding
    pad_y = (y_max_norm - y_min_norm) * padding_fraction
    pad_x = (x_max_norm - x_min_norm) * padding_fraction

    y0 = max(0, int(round((y_min_norm - pad_y) * h)))
    y1 = min(h, int(round((y_max_norm + pad_y) * h)))
    x0 = max(0, int(round((x_min_norm - pad_x) * w)))
    x1 = min(w, int(round((x_max_norm + pad_x) * w)))

    if y1 <= y0:
        y0, y1 = 0, h
    if x1 <= x0:
        x0, x1 = 0, w

    return y0, y1, x0, x1


def extract_overlap_rois(
    src_img: np.ndarray,
    ref_img: np.ndarray,
    source_meta: Dict[str, Any],
    reference_meta: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extract Source Overlap ROI and Reference Overlap ROI if geographic overlap exists."""
    overlap_info = evaluate_footprint_overlap(source_meta, reference_meta)
    if not overlap_info["has_overlap"] or overlap_info["intersection_bounds"] is None:
        return src_img, ref_img, {"roi_applied": False, "reason": "No valid overlap bounds"}

    src_bounds = overlap_info["source_bounds"]
    ref_bounds = overlap_info["reference_bounds"]
    inter_bounds = overlap_info["intersection_bounds"]

    sy0, sy1, sx0, sx1 = compute_overlap_pixel_roi(src_img.shape, src_bounds, inter_bounds)
    ry0, ry1, rx0, rx1 = compute_overlap_pixel_roi(ref_img.shape, ref_bounds, inter_bounds)

    src_roi = src_img[sy0:sy1, sx0:sx1]
    ref_roi = ref_img[ry0:ry1, rx0:rx1]

    details = {
        "roi_applied": True,
        "source_crop_box": (sy0, sy1, sx0, sx1),
        "reference_crop_box": (ry0, ry1, rx0, rx1),
        "source_roi_shape": src_roi.shape,
        "reference_roi_shape": ref_roi.shape,
        "overlap_area_sq_deg": overlap_info["overlap_area_sq_deg"],
    }

    return src_roi, ref_roi, details


def render_footprint_overlap_map(
    footprint_eval: Dict[str, Any],
    width: int = 700,
    height: int = 420,
) -> Optional[Any]:
    """Render a visual footprint overlap diagnostic graphic using PIL.

    Renders:
    - Source footprint polygon (cyan)
    - Reference footprint polygon (magenta)
    - Overlap polygon (green shaded)
    - Coordinate grid and diagnostic legend
    """
    from PIL import Image, ImageDraw

    src_poly = footprint_eval.get("source_polygon")
    ref_poly = footprint_eval.get("reference_polygon")
    overlap_poly = footprint_eval.get("overlap_polygon")

    if not src_poly or not ref_poly:
        return None

    # Determine if polar stereographic projection should be used
    is_polar = all(p[0] <= -60.0 for p in src_poly + ref_poly)

    if is_polar:
        src_coords = footprint_eval.get("projected_source_polygon") or []
        ref_coords = footprint_eval.get("projected_reference_polygon") or []
        ov_coords = footprint_eval.get("projected_overlap_polygon") or []
    else:
        src_coords = [(lon, lat) for lat, lon in src_poly]
        ref_coords = [(lon, lat) for lat, lon in ref_poly]
        ov_coords = [(lon, lat) for lat, lon in overlap_poly] if overlap_poly else []

    if not src_coords or not ref_coords:
        return None

    all_pts = src_coords + ref_coords
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max(max_x - min_x, 1e-4)
    span_y = max(max_y - min_y, 1e-4)

    pad_x = span_x * 0.15
    pad_y = span_y * 0.15
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y
    span_x = max_x - min_x
    span_y = max_y - min_y

    margin_left = 60
    margin_right = 40
    margin_top = 50
    margin_bottom = 50
    draw_w = width - margin_left - margin_right
    draw_h = height - margin_top - margin_bottom

    def to_canvas(x, y):
        cx = margin_left + int((x - min_x) / span_x * draw_w)
        cy = height - margin_bottom - int((y - min_y) / span_y * draw_h)
        return (cx, cy)

    src_px = [to_canvas(x, y) for x, y in src_coords]
    ref_px = [to_canvas(x, y) for x, y in ref_coords]
    ov_px = [to_canvas(x, y) for x, y in ov_coords] if ov_coords else []

    # Base background: deep space dark slate
    img = Image.new("RGBA", (width, height), (15, 23, 42, 255))
    draw = ImageDraw.Draw(img)

    # Subtle grid lines
    grid_steps = 4
    for i in range(grid_steps + 1):
        gx = margin_left + int(i * draw_w / grid_steps)
        gy = margin_top + int(i * draw_h / grid_steps)
        draw.line([(gx, margin_top), (gx, height - margin_bottom)], fill=(30, 41, 59, 180), width=1)
        draw.line([(margin_left, gy), (width - margin_right, gy)], fill=(30, 41, 59, 180), width=1)

    # Frame
    draw.rectangle(
        [(margin_left, margin_top), (width - margin_right, height - margin_bottom)],
        outline=(51, 65, 85, 255),
        width=1,
    )

    # Draw Source footprint (Cyan outline)
    draw.polygon(src_px, outline=(6, 182, 212, 255), width=3)

    # Draw Reference footprint (Magenta outline)
    draw.polygon(ref_px, outline=(236, 72, 153, 255), width=3)

    # Draw Overlap polygon (Green shaded overlay with alpha)
    if ov_px and len(ov_px) >= 3:
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        ov_draw.polygon(ov_px, fill=(34, 197, 94, 90), outline=(16, 185, 129, 255), width=2)
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)

    # Title & Coordinate System Banner
    cs_label = "Lunar South Polar Stereographic (km)" if is_polar else "Lunar Coordinates (Lon/Lat  deg)"
    draw.text((margin_left, 16), f"Footprint Overlap Diagnostic -- {cs_label}", fill=(241, 245, 249, 255))

    # Legend at bottom
    leg_y = height - 32
    # Cyan: Source
    draw.rectangle([(margin_left, leg_y + 4), (margin_left + 16, leg_y + 12)], outline=(6, 182, 212, 255), width=2)
    draw.text((margin_left + 22, leg_y), "Source (OHRC)", fill=(6, 182, 212, 255))
    # Magenta: Reference
    draw.rectangle([(margin_left + 160, leg_y + 4), (margin_left + 176, leg_y + 12)], outline=(236, 72, 153, 255), width=2)
    draw.text((margin_left + 182, leg_y), "Reference (LRO NAC)", fill=(236, 72, 153, 255))
    # Green: Overlap
    draw.rectangle([(margin_left + 360, leg_y + 4), (margin_left + 376, leg_y + 12)], fill=(34, 197, 94, 120), outline=(16, 185, 129, 255), width=2)
    draw.text((margin_left + 382, leg_y), "Overlap Region", fill=(34, 197, 94, 255))

    return img

