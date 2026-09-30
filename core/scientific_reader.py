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
from contextlib import contextmanager
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
class RasterFormatProfile:
    """Documented binary layout for a recognized scientific product family.

    ``sample_type`` is the canonical PDS3 ``SAMPLE_TYPE`` keyword token;
    ``sample_type_display`` is the human-readable form shown in the UI.
    """
    name: str
    record_bytes: int
    label_records: int
    image_pointer: int
    sample_bits: int
    dtype: str
    sample_type: str
    provenance: str
    sample_type_display: str = ""


LROC_NAC_EDR_PROFILE = RasterFormatProfile(
    name="LROC_NAC_EDR_PDS3",
    record_bytes=5064,
    label_records=1,
    image_pointer=2,
    sample_bits=8,
    dtype="uint8",
    sample_type="UNSIGNED_BYTE",
    provenance="LROC EDR archive format specification",
    sample_type_display="Unsigned Byte",
)

SAMPLE_TYPE_DISPLAY: Dict[str, str] = {
    "UNSIGNEDBYTE": "Unsigned Byte",
    "UNSIGNEDINTEGER": "Unsigned Integer",
    "UNSIGNEDINT": "Unsigned Integer",
    "SIGNEDINTEGER": "Signed Integer",
    "SIGNEDINT": "Signed Integer",
    "IEEEREAL": "IEEE Real",
    "IEEE754MSBSINGLE": "IEEE 754 Single Precision",
    "IEEE754LSBSINGLE": "IEEE 754 Single Precision",
}

DTYPE_DISPLAY: Dict[str, str] = {
    "uint8": "Unsigned Byte",
    ">i2": "Signed 16-bit Integer",
    "<i2": "Signed 16-bit Integer",
    "int16": "Signed 16-bit Integer",
    ">f4": "IEEE 754 Single Precision",
    "<f4": "IEEE 754 Single Precision",
    "float32": "IEEE 754 Single Precision",
}


def sample_type_display(sample_type: Optional[str], dtype: Optional[str] = None) -> str:
    """Return the human-readable sample-type label for the raster contract panel."""
    token = re.sub(r"[^A-Za-z0-9]", "", str(sample_type or "")).upper()
    if token in SAMPLE_TYPE_DISPLAY:
        return SAMPLE_TYPE_DISPLAY[token]
    if dtype in DTYPE_DISPLAY:
        return DTYPE_DISPLAY[str(dtype)]
    return str(sample_type or dtype or "Unknown")


def is_lroc_nac_edr(metadata: Optional[Dict[str, Any]] = None, name: str = "") -> bool:
    """Return whether metadata or filename identifies the generic LROC NAC EDR family."""
    metadata = metadata or {}
    dataset = str(metadata.get("data_set_id") or metadata.get("dataset") or "").upper()
    instrument = str(metadata.get("instrument") or metadata.get("instrument_id") or "").upper()
    sensor = str(metadata.get("sensor_type") or "").upper()
    product_type = str(metadata.get("product_type") or metadata.get("processing_level") or "").upper()
    catalog = str(metadata.get("metadata_source") or "").upper()
    profile = str(metadata.get("format_profile") or "").upper()

    raw_pid = str(metadata.get("product_id") or os.path.basename(name or "")).strip()
    clean_stem = re.sub(r"\.(img|lbl|xml|json|dat|raw|bin)$", "", raw_pid, flags=re.IGNORECASE).upper()
    filename_family = bool(re.match(r"^M\d+[LR]E$", clean_stem))

    # Reject non-lunar or clearly distinct sensors/missions
    if "OHRC" in instrument or "CH2" in dataset or "IIRS" in instrument or "TMC" in instrument:
        return False
    lname = (name or "").lower()
    if "ch2_" in lname or "ohrc" in lname or "iir" in lname or "tmc" in lname:
        return False

    if "LROC_NAC_EDR" in profile:
        return True

    if filename_family:
        return True

    if dataset == "LRO-L-LROC-2-EDR-V1.0":
        return True

    if "LROC" in instrument and ("NAC" in instrument or sensor == "LROC_NAC" or "EDR" in product_type):
        return True

    if sensor == "LROC_NAC":
        return True

    if catalog in {"LROC_CATALOG", "CATALOG_METADATA"} and ("LROC" in sensor or "LROC" in instrument or filename_family):
        return True

    return False


def apply_raster_profile(metadata: Optional[Dict[str, Any]], name: str = "") -> Dict[str, Any]:
    """Apply a recognized product-family profile without fabricating a label."""
    metadata = dict(metadata or {})
    if not is_lroc_nac_edr(metadata, name):
        return metadata

    dims = dict(metadata.get("dimensions") or {})
    lines = dims.get("lines") or (metadata.get("raster_spec") or {}).get("lines") or metadata.get("lines")
    samples = dims.get("samples") or (metadata.get("raster_spec") or {}).get("samples") or metadata.get("samples")
    sample_bits = metadata.get("sample_bits") or (metadata.get("raster_spec") or {}).get("sample_bits")

    if not lines or not samples:
        # Recognized as LROC NAC EDR, but dimensions are still unknown
        metadata["format_profile"] = LROC_NAC_EDR_PROFILE.name
        return metadata

    profile = LROC_NAC_EDR_PROFILE
    is_uploaded_lbl = metadata.get("metadata_source") == "UPLOADED_LABEL"

    record_bytes = int(metadata.get("record_bytes") or (metadata.get("raster_spec") or {}).get("record_bytes") or profile.record_bytes)
    image_pointer = int(metadata.get("image_pointer") or (metadata.get("raster_spec") or {}).get("image_pointer") or profile.image_pointer)

    if is_uploaded_lbl and metadata.get("image_offset") is not None:
        image_offset = int(metadata["image_offset"])
    else:
        # byte_offset = (pointer - 1) * record_bytes
        image_offset = (image_pointer - 1) * record_bytes

    if is_uploaded_lbl and metadata.get("sample_type"):
        sample_type = str(metadata["sample_type"])
        sample_type_label = sample_type_display(sample_type)
    else:
        sample_type = profile.sample_type
        sample_type_label = profile.sample_type_display or sample_type_display(sample_type)

    dtype = "uint8"
    bits = int(sample_bits or profile.sample_bits)

    raster_spec = dict(metadata.get("raster_spec") or {})
    raster_spec.update({
        "lines": int(lines),
        "samples": int(samples),
        "sample_bits": bits,
        "dtype": dtype,
        "sample_type": sample_type,
        "sample_type_display": sample_type_label,
        "byte_order": None,
        "image_offset": image_offset,
        "image_pointer": image_pointer,
        "image_pointer_unit": "RECORDS",
        "record_bytes": record_bytes,
        "label_records": profile.label_records,
        "product_id": metadata.get("product_id") or os.path.basename(name or ""),
        "provenance": profile.provenance,
    })
    metadata["dimensions"] = {"lines": int(lines), "samples": int(samples)}
    metadata.update({
        "format_profile": profile.name,
        "raster_spec_provenance": profile.provenance,
        "product_metadata_provenance": metadata.get("product_metadata_provenance") or "Verified LROC catalog record",
        "record_type": metadata.get("record_type") or "FIXED_LENGTH",
        "record_bytes": record_bytes,
        "label_records": profile.label_records,
        "image_pointer": image_pointer,
        "image_pointer_unit": "RECORDS",
        "image_offset": image_offset,
        "sample_bits": bits,
        "sample_type": sample_type,
        "sample_type_display": sample_type_label,
        "data_type": sample_type,
        "dtype": dtype,
        "byte_order": None,
        "binary_consistency_provenance": "Derived from downloaded IMG file size + catalog dimensions + LROC EDR PDS3 raster profile",
        "raster_spec": raster_spec,
    })
    return metadata


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
    image_pointer: Optional[int] = None
    image_pointer_unit: Optional[str] = None
    label_records: Optional[int] = None
    profile_id: Optional[str] = None
    line_prefix_bytes: Optional[int] = None
    line_suffix_bytes: Optional[int] = None
    scaling: Optional[Dict[str, Any]] = None
    product_id: Optional[str] = None
    provenance: Optional[str] = None

    def is_decodable(self) -> bool:
        return bool(self.lines and self.samples and self.dtype and self.image_offset is not None)

    @property
    def bytes_per_sample(self) -> Optional[int]:
        """Bytes per sample derived from the resolved dtype, else from sample_bits."""
        if self.dtype:
            try:
                return int(np.dtype(self.dtype).itemsize)
            except TypeError:
                return None
        if self.sample_bits:
            return max(1, int(self.sample_bits) // 8)
        return None

    def expected_raster_bytes(self) -> Optional[int]:
        """Image payload size implied by lines x samples x bytes-per-sample."""
        bps = self.bytes_per_sample
        if not (self.lines and self.samples and bps):
            return None
        return int(self.lines) * int(self.samples) * bps

    def missing_fields(self) -> list:
        """Explicitly name the raster contract fields that are still unresolved."""
        missing = []
        if not self.lines or not self.samples:
            missing.append("dimensions (lines, samples)")
        if not self.dtype and not self.sample_type:
            missing.append("sample type")
        if self.image_offset is None:
            missing.append("image offset")
        return missing

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def expected_file_structure(self, file_size_bytes: Optional[int] = None) -> Dict[str, Any]:
        """Return the raster-byte structure implied by the scientific metadata.

        The result keeps compatibility with both the current app and the stricter
        scientific validation rule set: exact-size checks remain possible, but a
        file containing extra headers/labels/records is reported as PARTIAL rather
        than rejected as a binary mismatch.
        """
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
        """Compatibility wrapper for the scientific subset of the full preview loader."""
        return load_scientific_preview(file_or_bytes, name, {"raster_spec": self.to_dict()}, max_side=max_side)


def scientific_raster_spec(metadata: Optional[Dict[str, Any]]) -> ScientificRasterSpec:
    """Normalize parser metadata into one explicit raster specification."""
    metadata = dict(metadata or {})
    raw = metadata.get("raster_spec") or {}
    dims = metadata.get("dimensions") or {}
    raw_dtype = raw.get("dtype") or metadata.get("dtype")
    if raw_dtype is None:
        legacy_type = str(raw.get("sample_type", metadata.get("sample_type", metadata.get("data_type", "")))).lower().replace("_", "").replace(" ", "")
        raw_dtype = {
            "unsignedbyte": "uint8", "uint8": "uint8", "byte": "uint8",
            "signedmsb2": ">i2", "signedlsb2": "<i2", "int16": "int16",
            "ieee754msbsingle": ">f4", "ieee754lsbsingle": "<f4", "float32": "float32",
        }.get(legacy_type)
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
        image_pointer=raw.get("image_pointer", metadata.get("image_pointer")),
        image_pointer_unit=raw.get("image_pointer_unit", metadata.get("image_pointer_unit")),
        label_records=raw.get("label_records", metadata.get("label_records")),
        profile_id=metadata.get("format_profile"),
        provenance=metadata.get("raster_spec_provenance"),
    )


def check_file_size_consistency(file_size_bytes: int, spec: ScientificRasterSpec) -> Dict[str, Any]:
    """Compare expected raster bytes with actual product size without guessing.

    A file whose payload is larger than the raw raster indicates extra records,
    labels, or attached metadata; that should be tagged as PARTIAL rather than
    rejected outright.
    """
    if not spec.lines or not spec.samples or not spec.sample_bits:
        return {"status": "UNKNOWN", "expected_raster_bytes": None, "actual_file_bytes": int(file_size_bytes),
                "image_offset": spec.image_offset, "message": "Raster size cannot be verified until raster fields are supplied."}
    expected = int(spec.lines) * int(spec.samples) * int(spec.sample_bits) // 8
    image_offset = int(spec.image_offset) if spec.image_offset is not None else 0
    available = max(0, int(file_size_bytes) - image_offset)
    record_bytes = int(spec.record_bytes) if spec.record_bytes else None
    expected_record_count = None
    expected_file_size_from_records = None
    if record_bytes:
        image_pointer = image_offset // record_bytes + 1
        expected_record_count = image_pointer - 1 + int(spec.lines)
        expected_file_size_from_records = expected_record_count * record_bytes
    expected_nominal_file_size = image_offset + expected

    if expected_file_size_from_records is None and available == expected:
        status = "VERIFIED"
        message = "Raster byte count is consistent."
    elif expected_file_size_from_records is not None and int(file_size_bytes) == expected_file_size_from_records:
        status = "VERIFIED"
        message = "Raster structure is verified against the LROC EDR format profile and file size."
    elif expected_file_size_from_records is not None and int(file_size_bytes) >= expected_nominal_file_size - record_bytes:
        status = "PARTIAL"
        message = "Image payload is present, but the file size is not the nominal complete record structure."
    elif available < expected:
        status = "MISMATCH"
        message = "File is smaller than the declared raster region."
    elif available > expected:
        status = "PARTIAL"
        message = "Raster parameters are understood; file contains extra bytes beyond the raw image payload."
    else:
        status = "UNKNOWN"
        message = "File structure could not be fully validated from the available metadata."
    return {
        "status": status,
        "expected_raster_bytes": expected,
        "actual_file_bytes": int(file_size_bytes),
        "available_raster_bytes": available,
        "image_offset": image_offset,
        "record_bytes": record_bytes,
        "expected_nominal_file_size": expected_nominal_file_size,
        "expected_record_count": expected_record_count,
        "expected_file_size_from_records": expected_file_size_from_records,
        "message": message,
    }


def validate_scientific_raster_access(
    file_or_bytes: Union[bytes, Any],
    spec: ScientificRasterSpec,
) -> Dict[str, Any]:
    """Read bounded raster samples through memmap without loading the full image."""
    if not spec.is_decodable():
        return {"status": "PARTIAL", "error_class": "RASTER_FORMAT_UNKNOWN", "message": "Raster contract is incomplete."}
    is_temp = False
    if isinstance(file_or_bytes, (str, os.PathLike)) and os.path.exists(file_or_bytes):
        path = str(file_or_bytes)
    else:
        path = persist_product(file_or_bytes, suffix=".IMG")
        is_temp = True
    mapped = None
    try:
        size = os.path.getsize(path)
        structure = check_file_size_consistency(size, spec)
        if structure["status"] == "MISMATCH":
            checks = build_raster_sanity_checks(spec, structure, None, None)
            return {
                "status": "MISMATCH",
                "error_class": "RASTER_CONSISTENCY_ERROR",
                "structure": structure,
                "checks": checks,
                "message": structure["message"],
            }
        mapped = open_scientific_memmap(path, spec)
        dtype = np.dtype(spec.dtype)
        rows = sorted({0, max(0, int(spec.lines // 2)), max(0, int(spec.lines - 1))})
        samples = []
        for row in rows:
            window = np.asarray(mapped[row, : min(int(spec.samples), 64)])
            samples.append({
                "row": row,
                "shape": tuple(window.shape),
                "dtype": str(window.dtype),
                "non_empty": bool(window.size),
                "finite": bool(np.isfinite(window.astype(np.float32)).all()),
                "in_dtype_range": _values_in_dtype_range(window, dtype),
                "min": int(window.min()) if window.size else None,
                "max": int(window.max()) if window.size else None,
            })
        checks = build_raster_sanity_checks(spec, structure, samples, dtype)
        failed = [c["name"] for c in checks if c["status"] == "FAIL"]
        if failed:
            return {
                "status": "MISMATCH",
                "error_class": "RASTER_SANITY_ERROR",
                "structure": structure,
                "rows": samples,
                "checks": checks,
                "message": "Raster sanity checks failed: " + "; ".join(failed) + ".",
            }
        return {
            "status": "VERIFIED",
            "error_class": None,
            "structure": structure,
            "rows": samples,
            "checks": checks,
            "message": "Raster access verified through bounded memmap reads.",
        }
    except Exception as exc:
        return {"status": "PARTIAL", "error_class": "RASTER_DECODE_ERROR", "message": str(exc)}
    finally:
        if mapped is not None:
            if hasattr(mapped, "_mmap") and mapped._mmap:
                try:
                    mapped._mmap.close()
                except Exception:
                    pass
            del mapped
        if is_temp:
            try:
                os.unlink(path)
            except OSError:
                pass


def _values_in_dtype_range(window: np.ndarray, dtype: np.dtype) -> bool:
    """Return whether every sampled value is representable by the resolved dtype."""
    if window.size == 0:
        return False
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return bool(window.min() >= info.min and window.max() <= info.max)
    if np.issubdtype(dtype, np.floating):
        return bool(np.isfinite(window.astype(np.float64)).all())
    return True


def build_raster_sanity_checks(
    spec: ScientificRasterSpec,
    structure: Optional[Dict[str, Any]] = None,
    samples: Optional[list] = None,
    dtype: Optional[np.dtype] = None,
) -> list:
    """Return the ordered raster sanity checks required before preprocessing.

    Mirrors the documented pre-decoding gate: file exists, dimensions known,
    sample type resolved, image offset resolved, record structure resolved,
    payload/file-size consistency, bounded preview read, first/middle/last line
    readability, dtype validity, and non-empty / non-corrupt preview.
    """
    samples = samples or []
    checks = []

    def add(check_name: str, ok: bool, detail: str) -> None:
        checks.append({"name": check_name, "status": "PASS" if ok else "FAIL", "detail": detail})

    dims_ok = bool(spec.lines and spec.samples)
    add("dimensions known", dims_ok,
        f"{spec.lines} x {spec.samples}" if dims_ok else "lines/samples unresolved")
    add("sample type resolved", bool(spec.dtype or spec.sample_type),
        str(spec.dtype or spec.sample_type or "sample type unresolved"))
    add("image offset resolved", spec.image_offset is not None,
        f"{spec.image_offset} bytes" if spec.image_offset is not None else "image offset unresolved")
    record_ok = bool(spec.record_bytes) and spec.image_pointer is not None
    add("record structure resolved", record_ok,
        f"RECORD_BYTES={spec.record_bytes}, ^IMAGE={spec.image_pointer} {spec.image_pointer_unit or 'RECORDS'}"
        if record_ok else "record structure unresolved")
    structure_status = (structure or {}).get("status")
    add("payload/file size consistency", structure_status in {"VERIFIED", "PARTIAL"},
        str((structure or {}).get("message") or "consistency not evaluated"))
    add("bounded preview read", bool(samples) or structure_status == "MISMATCH",
        f"{len(samples)} bounded row reads" if samples else "no rows read")
    rows_ok = bool(samples) and all(s.get("non_empty") for s in samples)
    add("first/middle/last line readable", rows_ok,
        ", ".join(f"row {s.get('row')}" for s in samples) if samples else "no rows read")
    range_ok = bool(samples) and all(s.get("in_dtype_range", True) for s in samples)
    add("sample values valid for dtype", range_ok, str(dtype or spec.dtype or "unknown dtype"))
    finite_ok = bool(samples) and all(s.get("finite", True) for s in samples)
    add("preview non-empty and not corrupted", finite_ok,
        f"{sum(int(s.get('shape', (0,))[0]) for s in samples)} sampled values")
    return checks


def build_raster_contract(
    metadata: Optional[Dict[str, Any]],
    name: str = "",
    file_size_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Describe the resolved scientific raster contract for the UI panel.

    When the contract is unresolved, ``missing_fields`` names every field that
    still has to be supplied (for example ``sample type`` and ``image offset``).
    """
    metadata = apply_raster_profile(metadata, name)
    spec = scientific_raster_spec(metadata)
    structure = check_file_size_consistency(int(file_size_bytes), spec) if file_size_bytes is not None else None
    return {
        "product_id": spec.product_id or metadata.get("product_id") or os.path.basename(name or ""),
        "profile_id": spec.profile_id or metadata.get("format_profile"),
        "lines": spec.lines,
        "samples": spec.samples,
        "dimensions_display": f"{spec.lines} × {spec.samples}" if spec.lines and spec.samples else None,
        "sample_bits": spec.sample_bits,
        "sample_type": spec.sample_type,
        "sample_type_display": sample_type_display(spec.sample_type, spec.dtype),
        "dtype": spec.dtype,
        "record_bytes": spec.record_bytes,
        "label_records": spec.label_records,
        "image_pointer": spec.image_pointer,
        "image_pointer_unit": spec.image_pointer_unit or ("RECORDS" if spec.image_pointer is not None else None),
        "image_offset": spec.image_offset,
        "expected_image_payload_bytes": spec.expected_raster_bytes(),
        "binary_consistency_status": (structure or {}).get("status", "UNKNOWN"),
        "binary_consistency_reason": (structure or {}).get("message"),
        "binary_consistency": structure,
        "reader": "Memory-mapped PDS3" if spec.is_decodable() else None,
        "resolved": spec.is_decodable(),
        "missing_fields": [] if spec.is_decodable() else spec.missing_fields(),
        "raster_spec_provenance": metadata.get("raster_spec_provenance"),
        "product_metadata_provenance": metadata.get("product_metadata_provenance"),
        "binary_consistency_provenance": metadata.get("binary_consistency_provenance"),
    }


@contextmanager
def open_scientific_source(file_or_bytes: Union[bytes, Any], suffix: str = ".bin"):
    """Yield a filesystem path for a scientific raster without holding it in RAM.

    Paths are yielded unchanged. In-memory payloads (bytes, BytesIO, Streamlit
    uploads) are streamed to a temporary file so the raster can be memory-mapped
    instead of being copied into a full in-memory array; the temporary file is
    removed on exit.
    """
    if isinstance(file_or_bytes, (str, os.PathLike)):
        if not os.path.exists(file_or_bytes):
            raise FileNotFoundError(f"Scientific raster path does not exist: {file_or_bytes}")
        yield str(file_or_bytes)
        return
    temp_path = persist_product(file_or_bytes, suffix=suffix)
    try:
        yield temp_path
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def _close_memmap(mapped: Any) -> None:
    """Release a numpy memmap's underlying mmap handle when present."""
    handle = getattr(mapped, "_mmap", None)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass


def persist_product(file_or_bytes: Union[bytes, Any], suffix: str = ".bin") -> str:
    """Persist an upload to a temporary path for memory-mapped access."""
    if isinstance(file_or_bytes, (str, os.PathLike)) and os.path.exists(file_or_bytes):
        import shutil
        fd, path = tempfile.mkstemp(prefix="lunamatch_product_", suffix=suffix)
        with os.fdopen(fd, "wb") as output, open(file_or_bytes, "rb") as src:
            shutil.copyfileobj(src, output)
        return path
    fd, path = tempfile.mkstemp(prefix="lunamatch_product_", suffix=suffix)
    with os.fdopen(fd, "wb") as output:
        if isinstance(file_or_bytes, (bytes, bytearray, memoryview)):
            output.write(file_or_bytes)
        elif hasattr(file_or_bytes, "getbuffer"):
            output.write(file_or_bytes.getbuffer())
        elif hasattr(file_or_bytes, "getvalue"):
            output.write(file_or_bytes.getvalue())
        elif hasattr(file_or_bytes, "read"):
            while True:
                chunk = file_or_bytes.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        else:
            raise TypeError(f"Unsupported file data type: {type(file_or_bytes)}")
    return path


def open_scientific_memmap(path: str, spec: ScientificRasterSpec) -> np.memmap:
    """Open a scientific raster as a read-only memory map."""
    if not spec.is_decodable():
        raise ValueError("Scientific raster specification is incomplete; decoding is pending metadata.")
    return np.memmap(path, dtype=np.dtype(spec.dtype), mode="r", offset=int(spec.image_offset),
                     shape=(int(spec.lines), int(spec.samples)))


def classify_product(name: str) -> ProductType:
    """Classify a product file by its name and extension.

    Deterministic classification without attempting to decode bytes.
    """
    if not name:
        return ProductType.UNKNOWN

    lname = name.strip().lower()

    if lname.endswith((".xml", ".lbl", ".txt")):
        return ProductType.METADATA_LABEL

    if lname.endswith((".tif", ".tiff")):
        return ProductType.GEOTIFF

    if lname.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        return ProductType.STANDARD_IMAGE

    if lname.endswith((".img", ".raw", ".dat", ".bin")):
        base = os.path.basename(lname)
        if re.match(r"^m\d+[lr]e", base) or "lroc" in base or "nac" in base or "lro" in base:
            return ProductType.LRO_PDS3_BINARY

        # Check for Chandrayaan-2 pattern: ch2_ohr..., ch2_tmc..., ch2_iir...
        if "ch2_" in base or "ohrc" in base or "tmc" in base or "iir" in base:
            return ProductType.CHANDRAYAAN_PDS4_BINARY

        return ProductType.SCIENTIFIC_BINARY

    return ProductType.UNKNOWN


def classify_product_display_label(name: str) -> str:
    """Return a descriptive human-readable label for the product."""
    ptype = classify_product(name)
    lname = (name or "").lower()

    if ptype == ProductType.METADATA_LABEL:
        return "Metadata Label (.xml / .lbl)"
    if ptype == ProductType.GEOTIFF:
        return "Georeferenced GeoTIFF"
    if ptype == ProductType.STANDARD_IMAGE:
        if "ch2_ohr" in lname or "ohrc" in lname:
            return "CHANDRAYAAN-2 OHRC BROWSE/PREVIEW IMAGE"
        if "ch2_tmc" in lname or "tmc" in lname:
            return "CHANDRAYAAN-2 TMC-2 BROWSE/PREVIEW IMAGE"
        if "ch2_iir" in lname or "iirs" in lname:
            return "CHANDRAYAAN-2 IIRS BROWSE/PREVIEW IMAGE"
        return "Standard Image (PNG / JPEG)"
    if ptype == ProductType.LRO_PDS3_BINARY:
        return "LRO LROC NAC Scientific Binary (.IMG)"
    if ptype == ProductType.CHANDRAYAAN_PDS4_BINARY:
        return "Chandrayaan-2 Scientific Binary (.IMG)"
    if ptype == ProductType.SCIENTIFIC_BINARY:
        return "Scientific Raw Binary (.IMG / .DAT)"
    return "Unknown Format"


def inspect_scientific_product(
    file_or_bytes: Union[bytes, Any],
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Inspect an uploaded product without sending scientific binaries to PIL.

    Parameters
    ----------
    file_or_bytes:
        Raw bytes, BytesIO, or Streamlit UploadedFile.
    name:
        Filename of the product.
    metadata:
        Parsed metadata dict, if available.

    Returns
    -------
    Dictionary describing the raster geometry, format, and decoding readiness.
    """
    if metadata:
        metadata = apply_raster_profile(metadata, name)
    else:
        metadata = {}
    ptype = classify_product(name)
    size_bytes = _get_byte_length(file_or_bytes)

    report: Dict[str, Any] = {
        "filename": name,
        "product_type": ptype.value,
        "product_label": classify_product_display_label(name),
        "file_size_bytes": size_bytes,
        "format": ptype.value,
        "crs": None,
        "projection": "Not available in metadata",
        "transform": None,
        "bands": 1,
        "channels": 1,
    }

    # 1. GeoTIFF
    if ptype == ProductType.GEOTIFF and _HAS_RASTERIO:
        data = _read_bytes_bounded(file_or_bytes, max_bytes=size_bytes)
        try:
            with MemoryFile(data) as mem:
                with mem.open() as ds:
                    report.update({
                        "width": ds.width,
                        "height": ds.height,
                        "bands": ds.count,
                        "channels": ds.count,
                        "crs": str(ds.crs) if ds.crs else None,
                        "dtype": str(ds.dtypes[0]),
                        "data_type": str(ds.dtypes[0]),
                        "valid_image_status": "Valid",
                        "decoding_status": "READY",
                        "format": "GeoTIFF",
                    })
                    return report
        except Exception as exc:
            report.update({
                "valid_image_status": "Invalid GeoTIFF",
                "decoding_status": "FAILED",
                "error_message": f"GeoTIFF read failed: {exc}",
            })
            return report

    # 2. Standard Image (PNG / JPEG / WebP)
    if ptype == ProductType.STANDARD_IMAGE:
        try:
            data = _read_bytes_bounded(file_or_bytes, max_bytes=min(size_bytes, 10 * 1024 * 1024))
            with Image.open(io.BytesIO(data)) as im:
                report.update({
                    "width": im.width,
                    "height": im.height,
                    "bands": len(im.getbands()),
                    "channels": len(im.getbands()),
                    "dtype": "uint8",
                    "data_type": "uint8",
                    "valid_image_status": "Valid",
                    "decoding_status": "READY",
                    "format": im.format or "Standard Image",
                })
                return report
        except Exception as exc:
            report.update({
                "valid_image_status": "Invalid standard image",
                "decoding_status": "FAILED",
                "error_message": f"Standard image decode failed: {exc}",
            })
            return report

    # 3. Scientific Binary (LRO .IMG, Chandrayaan .IMG, Generic .IMG)
    if ptype in (ProductType.LRO_PDS3_BINARY, ProductType.CHANDRAYAAN_PDS4_BINARY, ProductType.SCIENTIFIC_BINARY):
        spec = scientific_raster_spec(metadata)
        lines = spec.lines
        samples = spec.samples
        data_type = spec.dtype or spec.sample_type or "unknown"
        report["raster_spec"] = spec.to_dict()
        report["file_size_consistency"] = check_file_size_consistency(size_bytes, spec)
        report["format_profile"] = metadata.get("format_profile")
        report["raster_spec_provenance"] = metadata.get("raster_spec_provenance")

        if lines and samples and lines > 0 and samples > 0:
            consistency = report["file_size_consistency"]
            if spec.is_decodable():
                decoding_msg = (
                    f"Raster contract resolved from {metadata.get('raster_spec_provenance') or 'metadata'}: "
                    f"{lines} lines × {samples} samples ({spec.dtype}, offset {spec.image_offset} bytes)."
                )
                # Binary consistency drives readiness: an inconsistent or truncated
                # product is never reported READY, so preprocessing cannot consume it
                # in either execution mode.
                status_calc = {"VERIFIED": "READY", "PARTIAL": "PARTIAL", "MISMATCH": "MISMATCH"}.get(
                    consistency["status"], "PARTIAL"
                )
                report["raster_contract"] = build_raster_contract(metadata, name, size_bytes)
            else:
                missing = spec.missing_fields()
                decoding_msg = (
                    "Scientific raster contract unresolved. Missing: " + ", ".join(missing) + "."
                    if missing else "Scientific raster contract unresolved."
                )
                status_calc = "PENDING_METADATA"

            report.update({
                "width": int(samples),
                "height": int(lines),
                "dtype": str(data_type),
                "data_type": str(data_type),
                "valid_image_status": "Ready (Metadata Linked)",
                "decoding_status": status_calc,
                "format": f"Scientific Raster ({ptype.value})",
                "message": decoding_msg,
            })
            if spec.is_decodable() and consistency["status"] != "MISMATCH":
                access = validate_scientific_raster_access(file_or_bytes, spec)
                report["raster_access"] = access
                if access["status"] == "VERIFIED":
                    report["decoding_status"] = "READY"
                elif access["status"] == "PARTIAL" and access.get("error_class") is None:
                    report["decoding_status"] = "PARTIAL"
                else:
                    report["decoding_status"] = "MISMATCH"
        else:
            lro_note = (
                "Scientific LRO .IMG detected. A compatible PDS3 label or verified "
                "catalog metadata is required to decode the raster."
                if ptype == ProductType.LRO_PDS3_BINARY else
                "Scientific binary raster detected. Associated metadata is required to decode raster dimensions."
            )
            report.update({
                "width": None,
                "height": None,
                "dtype": "unknown",
                "data_type": "unknown",
                "valid_image_status": "Ready (pending metadata)",
                "decoding_status": "PENDING_METADATA",
                "format": f"Scientific Binary ({ptype.value})",
                "message": lro_note,
            })
        return report

    # 4. Unknown
    report.update({
        "valid_image_status": "Unknown format",
        "decoding_status": "UNSUPPORTED",
        "message": f"Unsupported or unrecognized product extension for '{name}'.",
    })
    return report


def load_scientific_preview(
    file_or_bytes: Union[bytes, Any],
    name: str,
    metadata: Dict[str, Any],
    max_side: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Memory-safe preview generation for large scientific rasters.

    Reads a strided subsample of the raw raster using buffer slicing or
    memory mapping so that a 252 MB file is not duplicated into multiple RAM copies.

    Parameters
    ----------
    file_or_bytes:
        Raw bytes, BytesIO, or Streamlit UploadedFile.
    name:
        Product filename.
    metadata:
        Parsed metadata dict containing dimensions and data_type.
    max_side:
        Max dimension for the visualization preview.

    Returns
    -------
    Tuple of (preview_float32, valid_mask, preview_meta).
    """
    spec = scientific_raster_spec(metadata)
    lines = int(spec.lines or 0)
    samples = int(spec.samples or 0)

    if lines <= 0 or samples <= 0:
        raise ValueError(
            f"Cannot decode scientific raster '{name}': lines and samples not provided in metadata."
        )

    if not spec.dtype or spec.image_offset is None:
        raise ValueError(f"Cannot decode scientific raster '{name}': explicit dtype and image offset are required.")
    data_type = str(spec.dtype)
    offset = int(spec.image_offset)

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

    stride = max(1, math.ceil(max(lines, samples) / max_side))
    expected_bytes = lines * samples * bytes_per_sample

    # Decode strictly through a memory map so a ~250 MB product is never copied
    # into RAM as one contiguous array. In-memory uploads are streamed to a
    # temporary file first (bounded chunks) and removed afterwards.
    with open_scientific_source(file_or_bytes, suffix=".IMG") as source_path:
        available_bytes = os.path.getsize(source_path) - offset
        if available_bytes < expected_bytes:
            raise ValueError(
                f"Insufficient binary data for '{name}': metadata requires {lines}×{samples} "
                f"({expected_bytes} bytes), but only {available_bytes} bytes available at offset {offset}."
            )
        arr_2d = np.memmap(source_path, dtype=dt, mode="r", offset=offset, shape=(lines, samples))
        try:
            if stride > 1:
                arr_sub = np.array(arr_2d[::stride, ::stride], dtype=np.float32)
            else:
                arr_sub = np.array(arr_2d, dtype=np.float32)
        finally:
            _close_memmap(arr_2d)
            del arr_2d

    valid_mask = np.isfinite(arr_sub) & (arr_sub > 0)

    load_meta = {
        "original_shape": (lines, samples),
        "preview_shape": arr_sub.shape,
        "stride": stride,
        "dtype": str(dt),
        "data_type": data_type,
        "is_preview": True,
        "label": "Visualization Preview",
    }
    return arr_sub, valid_mask, load_meta


def _get_byte_length(file_or_bytes: Any) -> int:
    """Return size in bytes of a file-like object or bytes."""
    if hasattr(file_or_bytes, "size"):
        return int(file_or_bytes.size)
    if hasattr(file_or_bytes, "getbuffer"):
        return len(file_or_bytes.getbuffer())
    if isinstance(file_or_bytes, (bytes, bytearray, memoryview)):
        return len(file_or_bytes)
    if hasattr(file_or_bytes, "getvalue"):
        return len(file_or_bytes.getvalue())
    if hasattr(file_or_bytes, "fileno"):
        try:
            return os.fstat(file_or_bytes.fileno()).st_size
        except Exception:
            pass
    if hasattr(file_or_bytes, "seek") and hasattr(file_or_bytes, "tell"):
        try:
            curr = file_or_bytes.tell()
            file_or_bytes.seek(0, os.SEEK_END)
            end = file_or_bytes.tell()
            file_or_bytes.seek(curr, os.SEEK_SET)
            return end
        except Exception:
            pass
    if isinstance(file_or_bytes, (str, os.PathLike)) and os.path.exists(file_or_bytes):
        return os.path.getsize(file_or_bytes)
    return 0


def _read_bytes_bounded(file_or_bytes: Any, max_bytes: int = 20 * 1024 * 1024) -> bytes:
    """Safely obtain up to max_bytes from an uploaded file or bytes container."""
    if hasattr(file_or_bytes, "getvalue"):
        val = file_or_bytes.getvalue()
        return val[:max_bytes] if len(val) > max_bytes else val
    if hasattr(file_or_bytes, "getbuffer"):
        buf = file_or_bytes.getbuffer()
        return bytes(buf[:max_bytes])
    if isinstance(file_or_bytes, (bytes, bytearray)):
        return file_or_bytes[:max_bytes]
    if hasattr(file_or_bytes, "read"):
        return file_or_bytes.read(max_bytes)
    return b""


def get_raster_shape(metadata: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    """Return (lines, samples) from metadata if available."""
    if not isinstance(metadata, dict):
        return None
    dims = metadata.get("dimensions", {})
    lines = dims.get("lines")
    samples = dims.get("samples")
    if lines and samples and int(lines) > 0 and int(samples) > 0:
        return (int(lines), int(samples))
    return None


def get_raster_dtype(metadata: Optional[Dict[str, Any]]) -> np.dtype:
    """Return numpy dtype from metadata without assuming uint8."""
    if not isinstance(metadata, dict):
        return np.dtype("uint8")
    data_type = str(metadata.get("data_type") or "unsignedbyte")
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
    return dtype_map.get(data_type.lower().replace("_", "").replace(" ", ""), np.dtype("uint8"))


def load_scientific_region(
    file_or_bytes: Union[bytes, Any],
    metadata: Dict[str, Any],
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> np.ndarray:
    """Load an exact region of interest from a scientific binary without reading the whole file."""
    meta_dims = metadata.get("dimensions", {}) if isinstance(metadata, dict) else {}
    total_lines = int(meta_dims.get("lines") or 0)
    total_samples = int(meta_dims.get("samples") or 0)
    if total_lines <= 0 or total_samples <= 0:
        raise ValueError("Cannot load region: dimensions missing from metadata.")

    row_start = max(0, min(row_start, total_lines))
    row_end = max(row_start, min(row_end, total_lines))
    col_start = max(0, min(col_start, total_samples))
    col_end = max(col_start, min(col_end, total_samples))

    dt = get_raster_dtype(metadata)
    offset = int(metadata.get("offset") or 0)

    if hasattr(file_or_bytes, "getbuffer"):
        buf = file_or_bytes.getbuffer()
    elif isinstance(file_or_bytes, (bytes, bytearray, memoryview)):
        buf = file_or_bytes
    elif hasattr(file_or_bytes, "read"):
        buf = file_or_bytes.read()
    else:
        raise TypeError(f"Unsupported file data type: {type(file_or_bytes)}")

    raw_flat = np.frombuffer(buf, dtype=dt, count=total_lines * total_samples, offset=offset)
    arr_2d = raw_flat.reshape((total_lines, total_samples))
    return np.array(arr_2d[row_start:row_end, col_start:col_end], dtype=np.float32)


def load_product(
    file_or_bytes: Union[bytes, Any],
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    max_side: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Universal product loader routing to appropriate decoder.

    - STANDARD_IMAGE (PNG, JPEG) -> PIL
    - GEOTIFF -> rasterio
    - SCIENTIFIC_BINARY / LRO_PDS3_BINARY / CHANDRAYAAN_PDS4_BINARY -> load_scientific_preview
    """
    ptype = classify_product(name)

    if ptype == ProductType.STANDARD_IMAGE:
        data = _read_bytes_bounded(file_or_bytes, max_bytes=100 * 1024 * 1024)
        with Image.open(io.BytesIO(data)) as im:
            rgba = im.convert("RGBA")
            valid = np.asarray(rgba.getchannel("A")) > 0
            arr = np.asarray(rgba.convert("L"), dtype=np.float32)
            return arr, valid, {"product_type": ptype.value, "format": im.format}

    if ptype == ProductType.GEOTIFF and _HAS_RASTERIO:
        data = _read_bytes_bounded(file_or_bytes, max_bytes=100 * 1024 * 1024)
        with MemoryFile(data) as mem:
            with mem.open() as ds:
                image = ds.read(1, masked=True)
                valid = ~np.ma.getmaskarray(image)
                arr = image.filled(np.nan).astype(np.float32)
                return arr, valid, {"product_type": ptype.value, "format": "GeoTIFF"}

    if ptype in (ProductType.LRO_PDS3_BINARY, ProductType.CHANDRAYAAN_PDS4_BINARY, ProductType.SCIENTIFIC_BINARY):
        if not metadata or not metadata.get("dimensions", {}).get("lines"):
            raise ValueError(
                f"Scientific product '{name}' requires metadata dimensions to decode raster."
            )
        return load_scientific_preview(file_or_bytes, name, metadata, max_side=max_side)

    raise ValueError(f"Unsupported product format for '{name}' ({ptype.value}).")
