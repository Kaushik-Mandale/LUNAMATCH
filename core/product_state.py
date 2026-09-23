"""
Product State Machine for LunaMatch V3.

Explicit lifecycle states for image uploads, metadata, footprints, and pair
validation. Prevents stale demo values from contaminating real experiments.

Zero-fabrication policy: every state is derived from what is actually present.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# State Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class MetadataStatus(str, Enum):
    """Source and completeness of metadata for a single product."""
    NOT_PROVIDED    = "NOT_PROVIDED"     # No file or manual input at all
    PARSED_FROM_PRODUCT = "PARSED_FROM_PRODUCT"  # Embedded label or sidecar XML parsed
    UPLOADED_LABEL  = "UPLOADED_LABEL"   # User uploaded a .LBL / .XML separately
    CATALOG_METADATA = "CATALOG_METADATA"  # From a known mission catalog record
    MANUALLY_ENTERED = "MANUALLY_ENTERED"  # User typed values into a form
    INCOMPLETE      = "INCOMPLETE"       # Parsed but required fields missing
    DEMO            = "DEMO"             # Loaded from a demo/fixture configuration
    INVALID         = "INVALID"          # Parse succeeded but result is nonsensical


class FootprintStatus(str, Enum):
    """Geographic footprint availability."""
    AVAILABLE   = "AVAILABLE"    # Four valid corner coordinates exist
    UNAVAILABLE = "UNAVAILABLE"  # Corners missing even though metadata was supplied
    PENDING     = "PENDING"      # Metadata not yet provided; footprint not evaluable


class PairValidationStatus(str, Enum):
    """Three-state pair validation result."""
    VALID    = "VALID"    # Footprints overlap; all required metadata present
    PENDING  = "PENDING"  # Cannot yet determine — metadata / footprint missing
    REJECTED = "REJECTED" # Both footprints known and they do NOT overlap


# Human-readable labels used in the UI
METADATA_STATUS_LABELS = {
    MetadataStatus.NOT_PROVIDED:      ("⚠️", "Not provided", "No metadata has been supplied."),
    MetadataStatus.PARSED_FROM_PRODUCT: ("✓", "Parsed from product", "Metadata was extracted from the uploaded product label."),
    MetadataStatus.UPLOADED_LABEL:    ("✓", "Uploaded label", "Metadata was supplied as a separate label file."),
    MetadataStatus.CATALOG_METADATA:  ("✓", "Catalog record", "Metadata comes from a mission catalog."),
    MetadataStatus.MANUALLY_ENTERED:  ("✓", "Manually entered", "Metadata was typed by the user."),
    MetadataStatus.INCOMPLETE:        ("⚠️", "Incomplete", "Some required fields are missing."),
    MetadataStatus.DEMO:              ("🔵", "Demo dataset", "Metadata comes from a demo configuration — not a real product."),
    MetadataStatus.INVALID:           ("✕", "Invalid", "Metadata could not be parsed correctly."),
}

FOOTPRINT_STATUS_LABELS = {
    FootprintStatus.AVAILABLE:   ("✓", "Available"),
    FootprintStatus.UNAVAILABLE: ("⚠️", "Unavailable — provide corner coordinates"),
    FootprintStatus.PENDING:     ("⏳", "Pending — supply metadata first"),
}

PAIR_STATUS_LABELS = {
    PairValidationStatus.VALID:    ("🟢", "PAIR VALIDATED", "Geographic overlap confirmed."),
    PairValidationStatus.PENDING:  ("🟡", "VALIDATION PENDING", "Provide reference metadata to evaluate geographic overlap."),
    PairValidationStatus.REJECTED: ("🔴", "PAIR REJECTED", "No meaningful geographic overlap detected."),
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_file_hash(data: bytes) -> str:
    """Return the first 16 hex chars of the SHA-256 of a file's bytes.

    Used to detect file replacement and invalidate stale session state.
    """
    return hashlib.sha256(data).hexdigest()[:16]


def _valid_footprint(fp) -> bool:
    """Return True iff fp has four corners each with two numeric coordinates."""
    if not isinstance(fp, dict):
        return False
    for key in ("upper_left", "upper_right", "lower_left", "lower_right"):
        corner = fp.get(key)
        if not isinstance(corner, (list, tuple)) or len(corner) != 2:
            return False
        try:
            float(corner[0])
            float(corner[1])
        except (TypeError, ValueError):
            return False
    return True


def compute_metadata_status(meta: dict, source_label: str) -> MetadataStatus:
    """Derive the MetadataStatus enum value from a parsed metadata dict.

    Parameters
    ----------
    meta:
        Canonical metadata dict (from parse_chandrayaan2_pds4_xml or
        parse_lro_pds3_label). May be an empty template.
    source_label:
        One of "NOT_PROVIDED", "PARSED_FROM_PRODUCT", "UPLOADED_LABEL",
        "CATALOG_METADATA", "MANUALLY_ENTERED", "DEMO".
    """
    if not meta or source_label == "NOT_PROVIDED":
        return MetadataStatus.NOT_PROVIDED

    if source_label == "DEMO":
        return MetadataStatus.DEMO

    # Map raw source strings to enum
    _map = {
        "PARSED_FROM_PRODUCT": MetadataStatus.PARSED_FROM_PRODUCT,
        "UPLOADED_LABEL":      MetadataStatus.UPLOADED_LABEL,
        "CATALOG_METADATA":    MetadataStatus.CATALOG_METADATA,
        "MANUALLY_ENTERED":    MetadataStatus.MANUALLY_ENTERED,
    }
    base_status = _map.get(source_label, MetadataStatus.INCOMPLETE)

    if not meta.get("valid", False) or meta.get("validation_errors"):
        return MetadataStatus.INCOMPLETE

    # Require at least lines + samples for a metadata record to be non-INCOMPLETE
    dims = meta.get("dimensions", {})
    if not dims.get("lines") or not dims.get("samples"):
        return MetadataStatus.INCOMPLETE

    # LRO scientific validation needs the declared raster contract, measured
    # resolution, and real corner coordinates. A recognized PDS record alone
    # must not unlock geographic validation.
    if str(meta.get("sensor_type", "")).upper() == "LROC_NAC":
        if meta.get("gsd_m_per_pixel") is None or not _valid_footprint(meta.get("footprint")):
            return MetadataStatus.INCOMPLETE
        raster_spec = meta.get("raster_spec") or {}
        if not raster_spec.get("dtype") or raster_spec.get("image_offset") is None:
            return MetadataStatus.INCOMPLETE

    return base_status


def compute_footprint_status(meta: dict, meta_status: MetadataStatus) -> FootprintStatus:
    """Derive FootprintStatus from parsed metadata and its current status."""
    if meta_status in (MetadataStatus.NOT_PROVIDED, MetadataStatus.INVALID):
        return FootprintStatus.PENDING

    fp = meta.get("footprint") if isinstance(meta, dict) else None
    if _valid_footprint(fp):
        return FootprintStatus.AVAILABLE

    # Metadata exists but corners are missing
    if meta_status != MetadataStatus.NOT_PROVIDED:
        return FootprintStatus.UNAVAILABLE

    return FootprintStatus.PENDING


def compute_pair_validation(
    source_meta_status: MetadataStatus,
    reference_meta_status: MetadataStatus,
    source_footprint_status: FootprintStatus,
    reference_footprint_status: FootprintStatus,
    footprint_eval: dict,
) -> tuple[PairValidationStatus, str]:
    """Derive pair validation status from component states.

    Returns
    -------
    (PairValidationStatus, human_readable_reason)
    """
    # If either footprint is not yet available → PENDING (never REJECTED)
    if (source_footprint_status != FootprintStatus.AVAILABLE or
            reference_footprint_status != FootprintStatus.AVAILABLE):
        missing = []
        if source_footprint_status != FootprintStatus.AVAILABLE:
            missing.append("source footprint")
        if reference_footprint_status != FootprintStatus.AVAILABLE:
            missing.append("reference footprint")
        return (
            PairValidationStatus.PENDING,
            f"Pair validation pending — {' and '.join(missing)} not yet available. "
            "Provide reference metadata to evaluate geographic overlap.",
        )

    # Both footprints available — use the evaluated result
    eval_status = footprint_eval.get("status", "pending")
    if eval_status == "rejected" or footprint_eval.get("has_overlap") is False:
        return (
            PairValidationStatus.REJECTED,
            "No meaningful geographic overlap detected. "
            "Matching was not executed because the source and reference "
            "images are geographically inconsistent.",
        )
    if eval_status == "validated" or footprint_eval.get("has_overlap") is True:
        area = footprint_eval.get("overlap_area_sq_deg", 0.0)
        pct = footprint_eval.get("overlap_fraction", 0.0) * 100
        return (
            PairValidationStatus.VALID,
            f"Geographic overlap confirmed ({pct:.1f}% of smaller image, {area:.4f} sq. deg).",
        )

    # Footprints were valid but overlap engine returned an ambiguous state
    return (
        PairValidationStatus.PENDING,
        "Geographic overlap evaluation is in progress.",
    )


def scale_ratio_display(src_gsd: Optional[float], ref_gsd: Optional[float]) -> str:
    """Return a human-readable scale ratio string from actual GSD values.

    Never returns a hardcoded value.  Returns a pending string if either GSD
    is unknown.
    """
    if src_gsd is None or ref_gsd is None:
        return "Scale ratio pending — reference GSD unavailable"
    if src_gsd <= 0 or ref_gsd <= 0:
        return "Scale ratio unavailable — invalid GSD value"
    ratio = ref_gsd / src_gsd
    return f"Scale Difference ≈ {ratio:.2f}×  ({src_gsd:.4f} m/px → {ref_gsd:.4f} m/px)"
