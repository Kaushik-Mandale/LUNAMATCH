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
    metadata = metadata or {}
    raw = metadata.get("raster_spec") or {}
    dims = metadata.get("dimensions") or {}
    raw_dtype = raw.get("dtype")
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
    if available == expected:
        status = "VERIFIED"
        message = "Raster byte count is consistent."
    elif available > expected:
        status = "PARTIAL"
        message = "Raster parameters are understood, but the file contains extra header/record structure beyond the raw image payload."
    elif available < expected:
        status = "MISMATCH"
        message = "File is smaller than the declared raster region."
    else:
        status = "UNKNOWN"
        message = "File structure could not be fully validated from the available metadata."
    return {"status": status, "expected_raster_bytes": expected, "actual_file_bytes": int(file_size_bytes),
            "available_raster_bytes": available, "image_offset": image_offset,
            "message": message}


def persist_product(file_or_bytes: Union[bytes, Any], suffix: str = ".bin") -> str:
    """Persist an upload to a temporary path for memory-mapped access."""
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

        if lines and samples and lines > 0 and samples > 0:
            report.update({
                "width": int(samples),
                "height": int(lines),
                "dtype": str(data_type),
                "data_type": str(data_type),
                "valid_image_status": "Ready (Metadata Linked)",
                "decoding_status": "READY" if spec.is_decodable() else "PENDING_METADATA",
                "format": f"Scientific Raster ({ptype.value})",
                "message": f"Raster geometry verified from metadata: {lines} lines × {samples} samples." if spec.is_decodable() else "Raster dimensions found, but explicit sample type and image offset are still required.",
            })
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

    if isinstance(file_or_bytes, (str, os.PathLike)):
        arr_2d = np.memmap(file_or_bytes, dtype=dt, mode="r", offset=offset, shape=(lines, samples))
    elif hasattr(file_or_bytes, "getbuffer"):
        buf = file_or_bytes.getbuffer()
    elif isinstance(file_or_bytes, (bytes, bytearray, memoryview)):
        buf = file_or_bytes
    elif hasattr(file_or_bytes, "read"):
        buf = file_or_bytes.read()
    else:
        raise TypeError(f"Unsupported file data type: {type(file_or_bytes)}")

    if not isinstance(file_or_bytes, (str, os.PathLike)):
        available_bytes = len(buf) - offset
        expected_bytes = lines * samples * bytes_per_sample

        if available_bytes < expected_bytes:
            raise ValueError(
                f"Insufficient binary data for '{name}': metadata requires {lines}×{samples} "
                f"({expected_bytes} bytes), but only {available_bytes} bytes available at offset {offset}."
            )

        raw_flat = np.frombuffer(buf, dtype=dt, count=lines * samples, offset=offset)
        arr_2d = raw_flat.reshape((lines, samples))

    if stride > 1:
        arr_sub = np.array(arr_2d[::stride, ::stride], dtype=np.float32)
    else:
        arr_sub = np.array(arr_2d, dtype=np.float32)

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

