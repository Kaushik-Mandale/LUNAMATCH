"""Scientific Raster Reader & Product Loader for LunaMatch V3."""
from __future__ import annotations

import io
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.io import MemoryFile
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

from core.lroc_profile import (
    LROC_NAC_EDR_PDS3_PROFILE_ID,
    is_lroc_nac_edr_product,
    apply_lroc_nac_edr_pds3_profile,
    unresolved_raster_contract_message as _unresolved_raster_contract_message,
)


class ProductType(str, Enum):
    STANDARD_IMAGE = "STANDARD_IMAGE"
    CHANDRAYAAN_PDS4_BINARY = "CHANDRAYAAN_PDS4_BINARY"
    LRO_PDS3_BINARY = "LRO_PDS3_BINARY"
    SCIENTIFIC_BINARY = "SCIENTIFIC_BINARY"
    GEOTIFF = "GEOTIFF"
    METADATA_LABEL = "METADATA_LABEL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ScientificRasterSpec:
    lines: Optional[int] = None
    samples: Optional[int] = None
    sample_bits: Optional[int] = None
    dtype: Optional[str] = None
    sample_type: Optional[str] = None
    byte_order: Optional[str] = None
    image_offset: Optional[int] = None
    record_bytes: Optional[int] = None
    line_prefix_bytes: Optional[int] = None
    line_suffix_bytes: Optional[int] = None
    scaling: Optional[Dict[str, Any]] = None
    product_id: Optional[str] = None
    provenance: Optional[str] = None

    def is_decodable(self) -> bool:
        return bool(self.lines and self.samples and self.dtype and self.image_offset is not None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def expected_file_structure(self, file_size_bytes: Optional[int] = None) -> Dict[str, Any]:
        if not self.lines or not self.samples or not self.sample_bits:
            return {
                "status": "UNKNOWN",
                "expected_raster_bytes": None,
                "actual_file_bytes": file_size_bytes,
                "image_offset": self.image_offset,
            }
        bps = max(1, int(self.sample_bits) // 8)
        lp = int(self.line_prefix_bytes or 0)
        ls = int(self.line_suffix_bytes or 0)
        rec = int(self.record_bytes) if self.record_bytes else (lp + self.samples * bps + ls)
        expected_raster = int(self.lines) * rec
        offset = int(self.image_offset or 0)
        expected_total = offset + expected_raster
        actual = file_size_bytes
        status = "UNKNOWN"
        if actual is not None:
            if actual == expected_total:
                status = "VERIFIED"
            elif actual >= expected_total:
                status = "PARTIAL"
            else:
                status = "MISMATCH"
        return {
            "status": status,
            "expected_raster_bytes": expected_raster,
            "expected_total_bytes": expected_total,
            "actual_file_bytes": actual,
            "image_offset": offset,
            "record_bytes": rec,
            "bytes_per_sample": bps,
        }


def scientific_raster_spec(metadata: Optional[Dict[str, Any]]) -> ScientificRasterSpec:
    metadata = apply_lroc_nac_edr_pds3_profile(metadata or {})
    raw = metadata.get("raster_spec") or {}
    dims = metadata.get("dimensions") or {}
    raw_dtype = raw.get("dtype") or metadata.get("dtype") or metadata.get("data_type")
    sample_bits = raw.get("sample_bits") or metadata.get("sample_bits")
    sample_type = raw.get("sample_type") or metadata.get("sample_type") or raw_dtype
    image_offset = raw.get("image_offset")
    if image_offset is None:
        image_offset = metadata.get("image_offset", metadata.get("offset"))
    record_bytes = raw.get("record_bytes") or metadata.get("record_bytes")
    lines = raw.get("lines") or dims.get("lines") or metadata.get("lines")
    samples = raw.get("samples") or dims.get("samples") or metadata.get("samples")

    dtype_str = None
    if isinstance(raw_dtype, str):
        dt = raw_dtype.lower().replace("_", "").replace(" ", "")
        if dt in ("uint8", "unsignedbyte", "byte", "unsignedinteger"):
            dtype_str = "uint8"
        elif dt in ("int16", "signedmsb2", "signedlsb2"):
            dtype_str = "int16"
        elif dt in ("float32", "ieee754msbsingle", "ieee754lsbsingle"):
            dtype_str = "float32"
        else:
            dtype_str = raw_dtype
    elif sample_bits == 8:
        dtype_str = "uint8"

    return ScientificRasterSpec(
        lines=int(lines) if lines else None,
        samples=int(samples) if samples else None,
        sample_bits=int(sample_bits) if sample_bits else None,
        dtype=dtype_str,
        sample_type=str(sample_type) if sample_type else None,
        byte_order=raw.get("byte_order") or metadata.get("byte_order"),
        image_offset=int(image_offset) if image_offset is not None else None,
        record_bytes=int(record_bytes) if record_bytes else None,
        line_prefix_bytes=int(raw.get("line_prefix_bytes") or metadata.get("line_prefix_bytes") or 0),
        line_suffix_bytes=int(raw.get("line_suffix_bytes") or metadata.get("line_suffix_bytes") or 0),
        scaling=raw.get("scaling") or metadata.get("scaling"),
        product_id=raw.get("product_id") or metadata.get("product_id"),
        provenance=raw.get("provenance") or metadata.get("raster_spec_provenance"),
    )


def check_file_size_consistency(file_size_bytes: int, spec: ScientificRasterSpec) -> Dict[str, Any]:
    structure = spec.expected_file_structure(file_size_bytes)
    msgs = {
        "VERIFIED": "Binary size matches raster contract",
        "PARTIAL": "File larger than expected raster region (extra header/trailer OK)",
        "MISMATCH": "File smaller than expected raster region",
        "UNKNOWN": "Cannot verify size without full raster contract",
    }
    return {
        "status": structure.get("status", "UNKNOWN"),
        "expected_raster_bytes": structure.get("expected_raster_bytes"),
        "expected_total_bytes": structure.get("expected_total_bytes"),
        "actual_file_bytes": file_size_bytes,
        "image_offset": structure.get("image_offset"),
        "record_bytes": structure.get("record_bytes"),
        "message": msgs.get(structure.get("status", "UNKNOWN"), "Size check incomplete"),
    }
