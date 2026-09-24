"""
Scientific Raster Reader & Product Loader for LunaMatch V3.

Provides product-aware loading that distinguishes standard images (PNG, JPEG),
GeoTIFFs, and scientific raw rasters (PDS3 LRO .IMG, PDS4 Chandrayaan-2 .IMG).

Never passes raw scientific binaries to generic decoders like PIL.Image.open().
Uses metadata-driven dimensions and numpy.memmap / strided reading to prevent
loading large rasters (e.g. 252 MB) repeatedly into memory.
"""
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

from core.lroc_profile import (
    apply_lroc_nac_edr_pds3_profile,
    unresolved_raster_contract_message,
)

try:
    import rasterio
    from rasterio.io import MemoryFile
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False


class ProductType(str, Enum):
    """Classification of an uploaded or selected lunar product file."""
    STANDARD_IMAGE           = "STANDARD_IMAGE"
    CHANDRAYAAN_PDS4_BINARY  = "CHANDRAYAAN_PDS4_BINARY"
    LRO_PDS3_BINARY          = "LRO_PDS3_BINARY"
    SCIENTIFIC_BINARY        = "SCIENTIFIC_BINARY"
    GEOTIFF                  = "GEOTIFF"
    METADATA_LABEL           = "METADATA_LABEL"
    UNKNOWN                  = "UNKNOWN"


@dataclass(frozen=True)
class ScientificRasterSpec:
    """Explicit PDS raster contract; unknown fields remain None."""
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
                "record_bytes": self.record_bytes,
                "message": "Raster structure cannot be verified until lines, samples, and sample_bits are known.",
            }
        total_pixels = int(self.lines) * int(self.samples)
        bytes_per_sample = max(1, int(self.sample_bits) // 8)
        expected_raster_bytes = total_pixels * bytes_per_sample
        actual_file_bytes = int(file_size_bytes) if file_size_bytes is not None else None
        image_offset = int(self.image_offset) if self.image_offset is not None else 0
        available_after_offset = None if actual_file_bytes is None else max(0, actual_file_bytes - image_offset)
        if actual_file_bytes is None:
            status = "VERIFIED"
            message = "Raster structure is internally consistent with the parsed label; file-size validation is pending actual file data."
        elif available_after_offset == expected_raster_bytes:
            status = "VERIFIED"
            message = "Raster structure and file size are consistent with the parsed label."
        elif available_after_offset > expected_raster_bytes:
            status = "PARTIAL"
            message = "Raster parameters are understood, but the file contains extra header/record structure beyond the raw image payload."
        elif available_after_offset < expected_raster_bytes:
            status = "MISMATCH"
            message = "File is smaller than the declared scientific raster structure."
        else:
            status = "UNKNOWN"
            message = "File structure could not be fully validated from the available metadata."
        return {
            "status": status,
            "expected_raster_bytes": expected_raster_bytes,
            "actual_file_bytes": actual_file_bytes,
            "available_raster_bytes": available_after_offset,
            "image_offset": image_offset,
            "record_bytes": self.record_bytes,
            "message": message,
        }

    def read_preview(self, file_or_bytes: Union[bytes, Any], name: str, max_side: int = 2048) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        return load_scientific_preview(file_or_bytes, name, {"raster_spec": self.to_dict()}, max_side=max_side)


def scientific_raster_spec(metadata: Optional[Dict[str, Any]]) -> ScientificRasterSpec:
    """Normalize parser metadata into one explicit raster specification.

    For recognized LROC NAC EDR products, missing sample_type / image_offset /
    record_bytes are resolved from the generic LROC_NAC_EDR_PDS3 archive profile.
    Explicit LBL values always win.
    """
    metadata = apply_lroc_nac_edr_pds3_profile(metadata or {})
    raw = metadata.get("raster_spec") or {}
    dims = metadata.get("dimensions") or {}
    raw_dtype = raw.get("dtype")
    if raw_dtype is None:
        legacy_type = str(raw.get("sample_type", metadata.get("sample_type", metadata.get("data_type", "")))).lower().replace("_", "").replace(" ", "")
        raw_dtype = {
            "unsignedbyte": "uint8", "uint8": "uint8", "byte": "uint8",
            "unsignedinteger": "uint8",
            "signedmsb2": ">i2", "signedlsb2": "<i2", "int16": "int16",
            "ieee754msbsingle": ">f4", "ieee754lsbsingle": "<f4", "float32": "float32",
        }.get(legacy_type)
        if raw_dtype is None and metadata.get("sample_bits") == 8:
            raw_dtype = "uint8"
    return ScientificRasterSpec(
        lines=raw.get("lines", dims.get("lines")),
        samples=raw.get("samples", dims.get("samples")),
        sample_bits=raw.get("sample_bits", metadata.get("sample_bits")),
        dtype=raw_dtype,
        sample_type=raw.get("sample_type", metadata.get("sample_type", metadata.get("data_type"))),
        byte_order=raw.get("byte_order", metadata.get("byte_order")),
        image_offset=raw.get("image_offset", metadata.get("image_offset", metadata.get("offset"))),
        record_bytes=raw.get("record_bytes", metadata.get("record_bytes")),
        line_prefix_bytes=raw.get("line_prefix_bytes", metadata.get("line_prefix_bytes")),
        line_suffix_bytes=raw.get("line_suffix_bytes", metadata.get("line_suffix_bytes")),
        scaling=raw.get("scaling", metadata.get("scaling")),
        product_id=raw.get("product_id", metadata.get("product_id")),
        provenance=raw.get("provenance") or metadata.get("raster_spec_provenance"),
    )
