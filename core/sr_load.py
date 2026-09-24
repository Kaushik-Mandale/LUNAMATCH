"""Scientific product classification and loading."""
from __future__ import annotations
import io, math, os, re, tempfile
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
    apply_lroc_nac_edr_pds3_profile,
    unresolved_raster_contract_message as _unresolved_raster_contract_message,
)
from core.sr_spec import (
    ProductType, ScientificRasterSpec, scientific_raster_spec, check_file_size_consistency,
)


def classify_product(name: str) -> ProductType:
    if not name:
        return ProductType.UNKNOWN
    lname = name.lower()
    if lname.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
        return ProductType.STANDARD_IMAGE
    if lname.endswith((".tif", ".tiff")):
        return ProductType.GEOTIFF
    if lname.endswith((".xml", ".lbl", ".pds", ".lblx")):
        return ProductType.METADATA_LABEL
    if lname.endswith((".img", ".dat", ".bin")):
        base = os.path.basename(lname)
        if re.match(r"^m\d+[lr]e", base) or "lroc" in base or "nac" in base:
            return ProductType.LRO_PDS3_BINARY
        if "ch2_" in base or "ohrc" in base or "tmc" in base or "iirs" in base:
            return ProductType.CHANDRAYAAN_PDS4_BINARY
        return ProductType.SCIENTIFIC_BINARY
    return ProductType.UNKNOWN


def classify_product_display_label(name: str) -> str:
    labels = {
        ProductType.STANDARD_IMAGE: "Standard Image",
        ProductType.GEOTIFF: "GeoTIFF",
        ProductType.LRO_PDS3_BINARY: "LRO PDS3 Binary",
        ProductType.CHANDRAYAAN_PDS4_BINARY: "Chandrayaan-2 PDS4 Binary",
        ProductType.SCIENTIFIC_BINARY: "Scientific Binary",
        ProductType.METADATA_LABEL: "Metadata Label",
        ProductType.UNKNOWN: "Unknown",
    }
    return labels.get(classify_product(name), "Unknown")


def get_raster_shape(metadata: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    if not isinstance(metadata, dict):
        return None
    dims = metadata.get("dimensions") or {}
    lines = dims.get("lines") or metadata.get("lines")
    samples = dims.get("samples") or metadata.get("samples")
    if lines and samples:
        return (int(lines), int(samples))
    return None


def get_raster_dtype(metadata: Optional[Dict[str, Any]]) -> np.dtype:
    if not isinstance(metadata, dict):
        return np.dtype("uint8")
    data_type = str(metadata.get("data_type") or metadata.get("dtype") or "unsignedbyte")
    dtype_map = {
        "unsignedbyte": np.dtype("uint8"), "byte": np.dtype("uint8"), "uint8": np.dtype("uint8"),
        "unsignedinteger": np.dtype("uint8"), "unsigned_integer": np.dtype("uint8"),
        "signedmsb2": np.dtype(">i2"), "signedlsb2": np.dtype("<i2"), "int16": np.dtype("int16"),
        "ieee754msbsingle": np.dtype(">f4"), "ieee754lsbsingle": np.dtype("<f4"),
        "float32": np.dtype("float32"),
    }
    key = data_type.lower().replace("_", "").replace(" ", "")
    return dtype_map.get(key, np.dtype("uint8"))


def _get_byte_length(file_or_bytes: Any) -> int:
    if isinstance(file_or_bytes, (bytes, bytearray, memoryview)):
        return len(file_or_bytes)
    if hasattr(file_or_bytes, "getbuffer"):
        return len(file_or_bytes.getbuffer())
    if hasattr(file_or_bytes, "seek") and hasattr(file_or_bytes, "tell"):
        pos = file_or_bytes.tell()
        file_or_bytes.seek(0, 2)
        size = file_or_bytes.tell()
        file_or_bytes.seek(pos)
        return size
    return 0


def _read_bytes_bounded(file_or_bytes: Any, max_bytes: int = 20 * 1024 * 1024) -> bytes:
    if isinstance(file_or_bytes, (bytes, bytearray)):
        return bytes(file_or_bytes[:max_bytes])
    if hasattr(file_or_bytes, "getvalue"):
        return file_or_bytes.getvalue()[:max_bytes]
    if hasattr(file_or_bytes, "read"):
        pos = file_or_bytes.tell() if hasattr(file_or_bytes, "tell") else None
        data = file_or_bytes.read(max_bytes)
        if pos is not None and hasattr(file_or_bytes, "seek"):
            file_or_bytes.seek(pos)
        return data if isinstance(data, bytes) else bytes(data)
    raise TypeError(f"Unsupported file data type: {type(file_or_bytes)}")


def inspect_scientific_product(file_or_bytes: Any, name: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metadata = apply_lroc_nac_edr_pds3_profile(metadata or {})
    ptype = classify_product(name)
    report: Dict[str, Any] = {
        "product_type": ptype.value,
        "display_label": classify_product_display_label(name),
        "name": name,
        "file_size_bytes": _get_byte_length(file_or_bytes),
    }
    if ptype in (ProductType.LRO_PDS3_BINARY, ProductType.CHANDRAYAAN_PDS4_BINARY, ProductType.SCIENTIFIC_BINARY):
        spec = scientific_raster_spec(metadata)
        report["raster_spec"] = spec.to_dict()
        report["is_decodable"] = spec.is_decodable()
        report["raster_profile_id"] = metadata.get("raster_profile_id")
        report["raster_spec_provenance"] = metadata.get("raster_spec_provenance")
        if spec.is_decodable():
            report["binary_consistency"] = check_file_size_consistency(report["file_size_bytes"], spec)
            report["valid_image_status"] = "Ready (Metadata Linked)"
            report["decoding_status"] = "READY"
        else:
            report["binary_consistency"] = {"status": "UNKNOWN"}
            report["valid_image_status"] = "Blocked"
            report["decoding_status"] = "BLOCKED"
            report["message"] = _unresolved_raster_contract_message(spec)
    elif ptype in (ProductType.STANDARD_IMAGE, ProductType.GEOTIFF):
        report["valid_image_status"] = "Ready"
        report["decoding_status"] = "READY"
        report["is_decodable"] = True
    else:
        report["valid_image_status"] = "Unknown"
        report["decoding_status"] = "UNKNOWN"
        report["is_decodable"] = False
    return report


def load_scientific_preview(file_or_bytes: Any, name: str, metadata: Optional[Dict[str, Any]] = None, max_side: int = 2048):
    metadata = apply_lroc_nac_edr_pds3_profile(metadata or {})
    spec = scientific_raster_spec(metadata)
    if not spec.is_decodable():
        raise ValueError(_unresolved_raster_contract_message(spec))
    info: Dict[str, Any] = {
        "product_type": classify_product(name).value,
        "raster_spec": spec.to_dict(),
        "raster_profile_id": metadata.get("raster_profile_id"),
    }
    path = None
    try:
        if isinstance(file_or_bytes, str) and os.path.isfile(file_or_bytes):
            path = file_or_bytes
        else:
            data = _read_bytes_bounded(file_or_bytes, max_bytes=300 * 1024 * 1024)
            fd, path = tempfile.mkstemp(suffix=".img")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            info["temp_path"] = path
        dtype = np.dtype(spec.dtype or "uint8")
        offset = int(spec.image_offset or 0)
        shape = (int(spec.lines), int(spec.samples))
        mm = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)
        step_y = max(1, int(math.ceil(shape[0] / max_side)))
        step_x = max(1, int(math.ceil(shape[1] / max_side)))
        preview = np.array(mm[::step_y, ::step_x], dtype=np.float32)
        valid = np.ones(preview.shape, dtype=bool)
        info["preview_shape"] = preview.shape
        info["full_shape"] = shape
        info["downsample"] = (step_y, step_x)
        return preview, valid, info
    finally:
        if info.get("temp_path") and os.path.isfile(info["temp_path"]):
            try:
                os.unlink(info["temp_path"])
            except OSError:
                pass


def load_product(file_or_bytes, name: str, metadata=None, max_side: int = 2048):
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
        if not metadata or not (metadata.get("dimensions") or {}).get("lines"):
            raise ValueError(f"Scientific product '{name}' requires metadata dimensions to decode raster.")
        return load_scientific_preview(file_or_bytes, name, metadata, max_side=max_side)
    raise ValueError(f"Unsupported product format for '{name}' ({ptype.value}).")
