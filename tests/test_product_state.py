"""Unit tests for core/product_state.py state machine and validation."""
import pytest
from core.product_state import (
    MetadataStatus,
    FootprintStatus,
    PairValidationStatus,
    compute_file_hash,
    compute_metadata_status,
    compute_footprint_status,
    compute_pair_validation,
    scale_ratio_display,
)


def test_compute_file_hash():
    data1 = b"test payload 1"
    data2 = b"test payload 2"
    h1 = compute_file_hash(data1)
    h2 = compute_file_hash(data2)
    assert len(h1) == 16
    assert h1 == compute_file_hash(data1)
    assert h1 != h2


def test_compute_metadata_status_not_provided():
    assert compute_metadata_status({}, "NOT_PROVIDED") == MetadataStatus.NOT_PROVIDED
    assert compute_metadata_status(None, "NOT_PROVIDED") == MetadataStatus.NOT_PROVIDED


def test_compute_metadata_status_demo():
    meta = {"product_id": "DEMO_1", "valid": True}
    assert compute_metadata_status(meta, "DEMO") == MetadataStatus.DEMO


def test_compute_metadata_status_valid_and_incomplete():
    complete_meta = {
        "product_id": "TEST_PROD",
        "valid": True,
        "validation_errors": [],
        "dimensions": {"lines": 1000, "samples": 500},
    }
    assert compute_metadata_status(complete_meta, "UPLOADED_LABEL") == MetadataStatus.UPLOADED_LABEL
    assert compute_metadata_status(complete_meta, "PARSED_FROM_PRODUCT") == MetadataStatus.PARSED_FROM_PRODUCT

    # Incomplete if dimensions missing
    no_dims = {"product_id": "TEST_PROD", "valid": True, "dimensions": {}}
    assert compute_metadata_status(no_dims, "UPLOADED_LABEL") == MetadataStatus.INCOMPLETE


def test_compute_footprint_status():
    assert compute_footprint_status({}, MetadataStatus.NOT_PROVIDED) == FootprintStatus.PENDING

    valid_fp = {
        "upper_left": [-10.0, 40.0],
        "upper_right": [-10.0, 41.0],
        "lower_left": [-11.0, 40.0],
        "lower_right": [-11.0, 41.0],
    }
    meta_with_fp = {"footprint": valid_fp}
    assert compute_footprint_status(meta_with_fp, MetadataStatus.PARSED_FROM_PRODUCT) == FootprintStatus.AVAILABLE

    meta_no_fp = {"footprint": None}
    assert compute_footprint_status(meta_no_fp, MetadataStatus.UPLOADED_LABEL) == FootprintStatus.UNAVAILABLE


def test_compute_pair_validation_three_states():
    # State 1: PENDING when either footprint is missing
    status, reason = compute_pair_validation(
        MetadataStatus.PARSED_FROM_PRODUCT,
        MetadataStatus.NOT_PROVIDED,
        FootprintStatus.AVAILABLE,
        FootprintStatus.PENDING,
        {},
    )
    assert status == PairValidationStatus.PENDING
    assert "pending" in reason.lower()

    # State 2: REJECTED when footprints are evaluated and have no overlap
    rejected_eval = {"status": "rejected", "has_overlap": False}
    status_rej, reason_rej = compute_pair_validation(
        MetadataStatus.PARSED_FROM_PRODUCT,
        MetadataStatus.UPLOADED_LABEL,
        FootprintStatus.AVAILABLE,
        FootprintStatus.AVAILABLE,
        rejected_eval,
    )
    assert status_rej == PairValidationStatus.REJECTED
    assert "No meaningful geographic overlap" in reason_rej

    # State 3: VALID when overlap is confirmed
    valid_eval = {
        "status": "validated",
        "has_overlap": True,
        "overlap_area_sq_deg": 0.05,
        "overlap_fraction": 0.35,
    }
    status_val, reason_val = compute_pair_validation(
        MetadataStatus.PARSED_FROM_PRODUCT,
        MetadataStatus.UPLOADED_LABEL,
        FootprintStatus.AVAILABLE,
        FootprintStatus.AVAILABLE,
        valid_eval,
    )
    assert status_val == PairValidationStatus.VALID
    assert "Geographic overlap confirmed" in reason_val


def test_scale_ratio_display():
    assert "pending" in scale_ratio_display(0.25, None).lower()
    assert "unavailable" in scale_ratio_display(0.25, 0.0).lower()
    s = scale_ratio_display(0.25, 2.0)
    assert "8.00×" in s
    assert "0.2500 m/px → 2.0000 m/px" in s
