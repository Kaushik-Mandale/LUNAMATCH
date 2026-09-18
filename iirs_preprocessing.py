"""
iirs_preprocessing.py — Chandrayaan-2 IIRS (Imaging Infrared Spectrometer) Preprocessing Module.

LunaMatch V3 Hyperspectral Processing Engine:
- Hyperspectral cube ingestion (Multi-band GeoTIFF, ENVI HDR/DAT, PDS4 Array_3D, NumPy arrays, standard rasters)
- Robust metadata inspection & validation
- NoData / invalid pixel masking & cleaning
- Spectral normalization (percentile clipping, standardization, per-band scaling)
- 2D representation synthesis for spatial correspondence (PCA PC1/PC2/PC3, selected band, multi-band composite, gradient)
- Resolution-aware spatial scaling & GSD calculation
"""

from __future__ import annotations

import io
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from PIL import Image

# ISRO Chandrayaan-2 IIRS specifications
# Spectral Range: 0.8 - 5.0 µm (~800 - 5000 nm), 256 contiguous spectral bands
# Nominal Spatial Resolution (GSD): ~80 m/pixel from 100 km circular lunar polar orbit
IIRS_NOMINAL_GSD_M = 80.0
IIRS_SPECTRAL_BANDS_NOMINAL = 256
IIRS_WAVELENGTH_MIN_NM = 800.0
IIRS_WAVELENGTH_MAX_NM = 5000.0


def inspect_iirs_metadata(
    data: Union[bytes, str, Path],
    name: str = "",
    xml_data: Optional[Union[bytes, str]] = None,
) -> Dict[str, Any]:
    """Inspect and extract metadata from an IIRS data source and optional PDS4 XML label.

    Does not fabricate metadata. If spectral band details cannot be determined,
    sets a clear warning in the returned dictionary.
    """
    metadata: Dict[str, Any] = {
        "filename": name or (str(data) if isinstance(data, (str, Path)) else "iirs_input"),
        "sensor_type": "IIRS",
        "instrument": "Imaging Infrared Spectrometer",
        "bands": None,
        "lines": None,
        "samples": None,
        "dtype": None,
        "wavelengths_nm": [],
        "wavelength_range_nm": None,
        "gsd_m_per_pixel": IIRS_NOMINAL_GSD_M,
        "gsd_source": "nominal_sensor_specification",
        "footprint": {},
        "valid": False,
        "warnings": [],
        "format": "Unknown",
    }

    # 1. Inspect XML metadata if supplied
    if xml_data is not None:
        _parse_iirs_pds4_xml(xml_data, metadata)

    # 2. Inspect file/data structure
    if isinstance(data, (str, Path)):
        file_path = Path(data)
        if file_path.exists():
            metadata["filename"] = file_path.name
            raw_bytes = file_path.read_bytes()
        else:
            raw_bytes = b""
    elif isinstance(data, bytes):
        raw_bytes = data
    else:
        raw_bytes = bytes(data)

    if raw_bytes:
        # Check GeoTIFF / TIFF
        if raw_bytes.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")) or (name and name.lower().endswith((".tif", ".tiff"))):
            try:
                with MemoryFile(raw_bytes) as mem:
                    with mem.open() as ds:
                        metadata["format"] = "GeoTIFF"
                        metadata["bands"] = ds.count
                        metadata["lines"] = ds.height
                        metadata["samples"] = ds.width
                        metadata["dtype"] = str(ds.dtypes[0])
                        if ds.crs:
                            metadata["crs"] = str(ds.crs)
                        # Check band tags for wavelengths
                        wls = []
                        for b_idx in range(1, ds.count + 1):
                            tag_dict = ds.tags(b_idx)
                            wl_val = tag_dict.get("WAVELENGTH") or tag_dict.get("wavelength") or tag_dict.get("CENTER_WAVELENGTH")
                            if wl_val:
                                try:
                                    wls.append(float(wl_val))
                                except ValueError:
                                    pass
                        if wls:
                            metadata["wavelengths_nm"] = wls
                            metadata["wavelength_range_nm"] = (min(wls), max(wls))
            except Exception as exc:
                metadata["warnings"].append(f"TIFF raster inspection note: {exc}")

        # Check ENVI Header in same directory or companion
        elif name and name.lower().endswith(".hdr"):
            _parse_envi_header(raw_bytes.decode("utf-8", errors="replace"), metadata)

        # Check NumPy array
        elif raw_bytes.startswith(b"\x93NUMPY"):
            try:
                arr = np.load(io.BytesIO(raw_bytes))
                metadata["format"] = "NumPy"
                metadata["dtype"] = str(arr.dtype)
                if arr.ndim == 3:
                    # Determine whether (B, H, W) or (H, W, B)
                    if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
                        metadata["bands"], metadata["lines"], metadata["samples"] = arr.shape
                    else:
                        metadata["lines"], metadata["samples"], metadata["bands"] = arr.shape
                elif arr.ndim == 2:
                    metadata["bands"] = 1
                    metadata["lines"], metadata["samples"] = arr.shape
            except Exception as exc:
                metadata["warnings"].append(f"NumPy load note: {exc}")

    # Check spectral band availability
    if metadata["bands"] is None or metadata["bands"] <= 0:
        metadata["warnings"].append(
            "Spectral bands could not be automatically determined from input container. "
            "Input will be treated as single-band or parsed via structural fallback."
        )
    elif metadata["bands"] == 1:
        metadata["warnings"].append(
            "Single-band input supplied for IIRS. Hyperspectral PCA will fall back to intensity/selected band representation."
        )
    elif not metadata["wavelengths_nm"]:
        metadata["warnings"].append(
            f"Input contains {metadata['bands']} spectral bands, but explicit per-band wavelength calibration tags were not found in raster metadata."
        )

    # Determine validity
    if metadata["lines"] and metadata["samples"] and metadata["lines"] > 0 and metadata["samples"] > 0:
        metadata["valid"] = True

    return metadata


def _parse_envi_header(header_text: str, metadata: Dict[str, Any]) -> None:
    """Parse ENVI header text into metadata dictionary."""
    metadata["format"] = "ENVI"
    lines_match = re.search(r"lines\s*=\s*(\d+)", header_text, re.IGNORECASE)
    samples_match = re.search(r"samples\s*=\s*(\d+)", header_text, re.IGNORECASE)
    bands_match = re.search(r"bands\s*=\s*(\d+)", header_text, re.IGNORECASE)
    if lines_match:
        metadata["lines"] = int(lines_match.group(1))
    if samples_match:
        metadata["samples"] = int(samples_match.group(1))
    if bands_match:
        metadata["bands"] = int(bands_match.group(1))

    # Wavelength list
    wl_match = re.search(r"wavelength\s*=\s*\{([^}]+)\}", header_text, re.IGNORECASE | re.DOTALL)
    if wl_match:
        vals = [float(v.strip()) for v in wl_match.group(1).split(",") if v.strip()]
        metadata["wavelengths_nm"] = vals
        if vals:
            metadata["wavelength_range_nm"] = (min(vals), max(vals))


def _parse_iirs_pds4_xml(xml_content: Union[bytes, str], metadata: Dict[str, Any]) -> None:
    """Parse ISRO/PDS4 XML metadata specific to Chandrayaan-2 IIRS."""
    try:
        text = xml_content.decode("utf-8", errors="replace") if isinstance(xml_content, bytes) else str(xml_content)
        root = ET.fromstring(text)
    except Exception as exc:
        metadata["warnings"].append(f"PDS4 XML parse error: {exc}")
        return

    # Extract instrument
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if tag in ("name", "instrument") and elem.text:
            t = elem.text.strip().lower()
            if "iirs" in t or "imaging infrared spectrometer" in t:
                metadata["sensor_type"] = "IIRS"
                metadata["instrument"] = elem.text.strip()
                break

    # Extract 3D Array dimensions (Array_3D_Spectrum, Array_3D_Image)
    for arr_elem in root.iter():
        tag = arr_elem.tag.split("}")[-1]
        if "Array_3D" in tag or "Array_2D" in tag:
            axes = arr_elem.findall(".//{*}Axis_Array")
            for ax in axes:
                name_elem = ax.find(".//{*}axis_name")
                elem_elem = ax.find(".//{*}elements")
                if name_elem is not None and elem_elem is not None and elem_elem.text:
                    ax_name = (name_elem.text or "").strip().lower()
                    count = int(elem_elem.text.strip())
                    if "band" in ax_name or "spectrum" in ax_name or "spectral" in ax_name:
                        metadata["bands"] = count
                    elif "line" in ax_name:
                        metadata["lines"] = count
                    elif "sample" in ax_name:
                        metadata["samples"] = count

    # Extract GSD / pixel resolution if present
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if tag in ("pixel_resolution", "pixelresolution") and elem.text:
            m = re.search(r"([\d.]+)", elem.text)
            if m:
                metadata["gsd_m_per_pixel"] = float(m.group(1))
                metadata["gsd_source"] = "pds4_geometry_parameters"
                break


def load_iirs(
    data: Union[bytes, str, Path],
    name: str = "",
    target_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load an IIRS hyperspectral data cube into a 3D float32 numpy array.

    Canonical output format: (bands, lines/height, samples/width).
    Supports GeoTIFF/TIFF, NumPy binary (.npy), and raw buffers.
    """
    if isinstance(data, (str, Path)):
        file_path = Path(data)
        if not file_path.exists():
            raise FileNotFoundError(f"IIRS file not found: {data}")
        raw_bytes = file_path.read_bytes()
        name = name or file_path.name
    elif isinstance(data, bytes):
        raw_bytes = data
    else:
        raw_bytes = bytes(data)

    metadata = inspect_iirs_metadata(raw_bytes, name)

    # 1. Load via rasterio (Multi-band TIFF / GeoTIFF)
    if raw_bytes.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")) or (name and name.lower().endswith((".tif", ".tiff"))):
        with MemoryFile(raw_bytes) as mem:
            with mem.open() as ds:
                bands_count = ds.count
                if target_shape is not None:
                    out_h, out_w = target_shape
                    cube = ds.read(
                        out_shape=(bands_count, out_h, out_w),
                        out_dtype="float32",
                        resampling=Resampling.bilinear,
                    )
                else:
                    cube = ds.read(out_dtype="float32")
                # cube is (bands, height, width)
                metadata["bands"] = cube.shape[0]
                metadata["lines"] = cube.shape[1]
                metadata["samples"] = cube.shape[2]
                return cube.astype(np.float32), metadata

    # 2. Load via NumPy array
    if raw_bytes.startswith(b"\x93NUMPY"):
        arr = np.load(io.BytesIO(raw_bytes)).astype(np.float32)
        if arr.ndim == 2:
            cube = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= arr.shape[2]:
                cube = arr
            else:
                cube = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported NumPy array dimensions for IIRS cube: {arr.shape}")

        if target_shape is not None and (cube.shape[1] != target_shape[0] or cube.shape[2] != target_shape[1]):
            resized_bands = [
                cv2.resize(b, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
                for b in cube
            ]
            cube = np.stack(resized_bands, axis=0)

        metadata["bands"] = cube.shape[0]
        metadata["lines"] = cube.shape[1]
        metadata["samples"] = cube.shape[2]
        return cube.astype(np.float32), metadata

    # 3. Load via PIL Image (e.g. RGB or Grayscale raster)
    try:
        with Image.open(io.BytesIO(raw_bytes)) as im:
            if target_shape is not None:
                im = im.resize((target_shape[1], target_shape[0]), Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.float32)
            if arr.ndim == 2:
                cube = arr[np.newaxis, :, :]
            elif arr.ndim == 3:
                # PIL (H, W, Channels) -> (Channels, H, W)
                cube = np.transpose(arr, (2, 0, 1))
            else:
                cube = arr[np.newaxis, :, :]
            metadata["bands"] = cube.shape[0]
            metadata["lines"] = cube.shape[1]
            metadata["samples"] = cube.shape[2]
            return cube.astype(np.float32), metadata
    except Exception as exc:
        raise ValueError(f"Unable to decode IIRS data stream into hyperspectral cube: {exc}")


def remove_invalid_pixels(
    cube: np.ndarray,
    no_data_value: Optional[float] = None,
    extreme_min: float = -1e4,
    extreme_max: float = 1e6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify and mask out invalid pixels (NaNs, infs, extreme uncalibrated values, NoData).

    Parameters:
        cube: 3D array (bands, height, width).
        no_data_value: Optional sentinel value (e.g. -9999.0, 65535).
        extreme_min: Lower bound for physically plausible radiance/reflectance.
        extreme_max: Upper bound for plausible values.

    Returns:
        cleaned_cube: 3D float32 array with invalid pixels zeroed or filled with finite median.
        valid_mask: 2D boolean array (height, width) indicating spatially valid pixels.
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube (bands, height, width), got shape {cube.shape}")

    # Spatial validity: a pixel is valid if finite across majority of bands
    finite_mask = np.isfinite(cube)
    plausible_mask = (cube > extreme_min) & (cube < extreme_max)
    valid_3d = finite_mask & plausible_mask

    if no_data_value is not None:
        valid_3d = valid_3d & (cube != no_data_value)

    # 2D spatial validity: valid if at least 50% of bands are good
    band_count = cube.shape[0]
    valid_band_sum = np.sum(valid_3d, axis=0)
    spatial_valid_mask = valid_band_sum >= max(1, band_count // 2)

    cleaned_cube = np.copy(cube)
    for b in range(band_count):
        band = cleaned_cube[b]
        b_valid = valid_3d[b] & spatial_valid_mask
        if np.any(b_valid):
            med_val = float(np.median(band[b_valid]))
            band[~b_valid] = med_val
        else:
            band[:] = 0.0
        cleaned_cube[b] = band

    return cleaned_cube.astype(np.float32), spatial_valid_mask.astype(bool)


def validate_iirs_cube(
    cube: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate spatial, spectral, and numerical properties of an IIRS cube."""
    report = {
        "valid": False,
        "bands": 0,
        "lines": 0,
        "samples": 0,
        "total_pixels": 0,
        "valid_pixels": 0,
        "valid_pixel_percentage": 0.0,
        "dynamic_range": (0.0, 0.0),
        "errors": [],
        "warnings": [],
    }
    if not isinstance(cube, np.ndarray):
        report["errors"].append("Cube is not a numpy array.")
        return report

    if cube.ndim != 3:
        report["errors"].append(f"Expected 3D array (bands, height, width), got {cube.ndim}D.")
        return report

    b, h, w = cube.shape
    total_pix = h * w
    report["bands"] = b
    report["lines"] = h
    report["samples"] = w
    report["total_pixels"] = total_pix

    if b == 0 or h == 0 or w == 0:
        report["errors"].append("Zero dimension encountered in cube.")
        return report

    finite_count = int(np.sum(np.isfinite(cube)))
    if finite_count == 0:
        report["errors"].append("Cube contains zero finite values.")
        return report

    _, valid_mask = remove_invalid_pixels(cube)
    valid_count = int(np.sum(valid_mask))
    report["valid_pixels"] = valid_count
    val_pct = (valid_count / max(total_pix, 1)) * 100.0
    report["valid_pixel_percentage"] = round(val_pct, 2)

    if val_pct < 5.0:
        report["errors"].append(f"Insufficient valid pixels in IIRS cube: {val_pct:.1f}% valid.")
        return report
    elif val_pct < 50.0:
        report["warnings"].append(f"High invalid/NoData pixel proportion: {100.0 - val_pct:.1f}% invalid.")

    # Dynamic range of valid pixels
    valid_vals = cube[:, valid_mask]
    if valid_vals.size > 0:
        lo, hi = float(np.min(valid_vals)), float(np.max(valid_vals))
        report["dynamic_range"] = (lo, hi)
        if hi <= lo:
            report["errors"].append("Zero contrast in valid IIRS pixels.")
            return report

    report["valid"] = len(report["errors"]) == 0
    return report


def normalize_iirs(
    cube: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    method: str = "percentile",
    clip_percentiles: Tuple[float, float] = (1.0, 99.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply spectral and spatial normalization to an IIRS hyperspectral cube.

    Methods:
    - 'percentile': Per-band robust 1-99 percentile stretch to [0.0, 1.0].
    - 'standardization': Z-score standardization (mean=0, std=1) across valid pixels.
    - 'none': Leaves pixel values unscaled (clipped to finite).
    """
    if valid_mask is None:
        cube, valid_mask = remove_invalid_pixels(cube)

    norm_cube = np.zeros_like(cube, dtype=np.float32)
    method_clean = method.lower().strip()

    for b in range(cube.shape[0]):
        band = cube[b]
        band_valid = valid_mask & np.isfinite(band)
        if not np.any(band_valid):
            continue

        vals = band[band_valid]

        if method_clean == "percentile":
            lo, hi = np.percentile(vals, clip_percentiles)
            if hi > lo:
                scaled = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
            else:
                scaled = np.zeros_like(band)
            scaled[~band_valid] = 0.0
            norm_cube[b] = scaled

        elif method_clean == "standardization":
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            if std_val > 1e-7:
                scaled = (band - mean_val) / std_val
            else:
                scaled = band - mean_val
            scaled[~band_valid] = 0.0
            norm_cube[b] = scaled

        else:  # 'none'
            scaled = np.copy(band)
            scaled[~band_valid] = 0.0
            norm_cube[b] = scaled

    return norm_cube, valid_mask


def generate_iirs_pca(
    cube: np.ndarray,
    valid_mask: np.ndarray,
    n_components: int = 3,
) -> Dict[str, Any]:
    """Compute Principal Component Analysis (PCA) on valid spectral vectors of the IIRS cube.

    Centers valid spectral pixels, computes covariance matrix, solves eigen-decomposition,
    and projects pixels back to spatial PC component maps (PC1, PC2, PC3).
    Maps are normalized to [0, 255] uint8.
    """
    bands, height, width = cube.shape
    n_comp = min(n_components, bands)

    valid_indices = np.where(valid_mask)
    if len(valid_indices[0]) < 10:
        raise ValueError("Too few valid pixels to compute IIRS PCA.")

    # Spectral vectors: shape (N_valid, bands)
    X = cube[:, valid_mask].T  # (N_valid, bands)
    mean_spectrum = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean_spectrum

    # Numerical covariance
    cov = np.cov(X_centered, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[cov]])

    # Eigen-decomposition (eigenvalues in ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    total_variance = np.sum(eigenvalues)
    var_ratio = (eigenvalues / max(total_variance, 1e-12)).tolist() if total_variance > 0 else [1.0 / bands] * bands

    # Project valid pixels onto components
    projected = np.dot(X_centered, eigenvectors[:, :n_comp])  # (N_valid, n_comp)

    components: Dict[str, np.ndarray] = {}
    for comp_idx in range(n_comp):
        comp_name = f"PC{comp_idx + 1}"
        comp_vals = projected[:, comp_idx]
        lo, hi = np.percentile(comp_vals, [1, 99])
        if hi > lo:
            scaled = np.clip((comp_vals - lo) / (hi - lo), 0.0, 1.0) * 255.0
        else:
            scaled = np.zeros_like(comp_vals)

        pc_img = np.zeros((height, width), dtype=np.uint8)
        pc_img[valid_indices] = scaled.astype(np.uint8)
        components[comp_name] = pc_img

    return {
        "components": components,
        "eigenvalues": eigenvalues[:n_comp].tolist(),
        "explained_variance_ratio": var_ratio[:n_comp],
        "n_components": n_comp,
    }


def generate_iirs_band_composite(
    cube: np.ndarray,
    valid_mask: np.ndarray,
    bands: Optional[Union[List[int], Tuple[int, ...]]] = None,
) -> np.ndarray:
    """Generate a multi-band composite grayscale image from the IIRS cube.

    Parameters:
        cube: 3D float array (bands, height, width).
        valid_mask: 2D boolean mask.
        bands: 1-indexed list/tuple of band indices to composite.
               Defaults to 3 evenly spaced bands across the spectral range.
    """
    total_bands, height, width = cube.shape
    if total_bands == 1:
        img = cube[0]
        lo, hi = np.percentile(img[valid_mask], [1, 99]) if np.any(valid_mask) else (0, 1)
        scaled = np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0
        out = scaled.astype(np.uint8)
        out[~valid_mask] = 0
        return out

    if bands is None:
        # 3 representative bands (e.g. VIS-NIR boundary ~1.0µm, hydration band ~2.8µm, thermal ~4.0µm)
        b1 = max(1, total_bands // 6)
        b2 = max(1, total_bands // 2)
        b3 = min(total_bands, (5 * total_bands) // 6)
        bands = (b1, b2, b3)

    selected = []
    for b in bands:
        b_idx = int(np.clip(b - 1, 0, total_bands - 1))
        b_data = cube[b_idx]
        if np.any(valid_mask):
            lo, hi = np.percentile(b_data[valid_mask], [1, 99])
            norm_b = np.clip((b_data - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        else:
            norm_b = np.zeros_like(b_data)
        selected.append(norm_b)

    composite = np.mean(selected, axis=0) * 255.0
    out = np.clip(composite, 0, 255).astype(np.uint8)
    out[~valid_mask] = 0
    return out


def generate_iirs_2d_representation(
    cube: np.ndarray,
    valid_mask: np.ndarray,
    method: str = "automatic",
    selected_band: int = 1,
    pca_component: int = 1,
    apply_clahe: bool = True,
    composite_bands: Optional[Tuple[int, ...]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Synthesize a high-contrast 2D image from the IIRS hyperspectral cube.

    Options:
    - 'automatic': Defaults to PCA (PC1) for multi-band cubes (>1 band) with >85% variance,
                   or selected band for single band.
    - 'pca': Principal component image (PC1, PC2, or PC3).
    - 'selected_band': Single spectral band index (1-indexed).
    - 'composite': Multi-band weighted average composite.
    - 'gradient': Multi-scale Sobel gradient magnitude image for rim/crater terrain relief.

    Returns:
        rep_image: 2D uint8 grayscale image (height, width).
        diag: Diagnostic summary dictionary.
    """
    bands, height, width = cube.shape
    method_clean = method.lower().strip()

    diag: Dict[str, Any] = {
        "method_requested": method,
        "method_applied": "",
        "cube_shape": (bands, height, width),
        "valid_pixel_pct": float(np.mean(valid_mask) * 100.0),
        "clahe_applied": False,
    }

    # Decide method if automatic
    if method_clean == "automatic":
        if bands > 1:
            method_clean = "pca"
        else:
            method_clean = "selected_band"

    # 1. PCA representation
    if method_clean == "pca" and bands > 1:
        pca_res = generate_iirs_pca(cube, valid_mask, n_components=3)
        comp_key = f"PC{int(np.clip(pca_component, 1, pca_res['n_components']))}"
        rep = pca_res["components"].get(comp_key, pca_res["components"]["PC1"])
        diag["method_applied"] = f"PCA ({comp_key})"
        diag["explained_variance"] = pca_res["explained_variance_ratio"]

    # 2. Selected Band
    elif method_clean == "selected_band" or (method_clean == "pca" and bands == 1):
        b_idx = int(np.clip(selected_band - 1, 0, bands - 1))
        b_img = cube[b_idx]
        if np.any(valid_mask):
            lo, hi = np.percentile(b_img[valid_mask], [1, 99])
            scaled = np.clip((b_img - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0
        else:
            scaled = np.zeros_like(b_img)
        rep = scaled.astype(np.uint8)
        rep[~valid_mask] = 0
        diag["method_applied"] = f"Selected Band (Band {b_idx + 1})"

    # 3. Composite
    elif method_clean == "composite":
        rep = generate_iirs_band_composite(cube, valid_mask, bands=composite_bands)
        diag["method_applied"] = "Multi-band Composite"

    # 4. Gradient Representation
    elif method_clean == "gradient":
        # Base on PC1 if multi-band, or band 1 if single band
        if bands > 1:
            base_rep, _ = generate_iirs_2d_representation(cube, valid_mask, method="pca", apply_clahe=False)
        else:
            base_rep, _ = generate_iirs_2d_representation(cube, valid_mask, method="selected_band", apply_clahe=False)

        gx = cv2.Sobel(base_rep, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(base_rep, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        rep = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        rep[~valid_mask] = 0
        diag["method_applied"] = "Spectral Gradient Magnitude"

    else:
        # Fallback to selected band
        b_idx = int(np.clip(selected_band - 1, 0, bands - 1))
        rep = np.clip(cube[b_idx], 0, 255).astype(np.uint8)
        rep[~valid_mask] = 0
        diag["method_applied"] = f"Selected Band {b_idx + 1} (Fallback)"

    # Optional CLAHE on the synthesized 2D image
    if apply_clahe and np.any(valid_mask):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        rep_clahe = clahe.apply(rep)
        rep_clahe[~valid_mask] = 0
        rep = rep_clahe
        diag["clahe_applied"] = True

    return rep, diag


def compute_resolution_scale_ratio(
    source_gsd: float,
    reference_gsd: float,
) -> Dict[str, Any]:
    """Compute spatial resolution ratio and multi-scale recommendations.

    scale_ratio = source_gsd / reference_gsd.
    Example: IIRS (~80 m) vs TMC-2 (~5 m) -> scale_ratio = 16.0
             IIRS (~80 m) vs OHRC (~0.25 m) -> scale_ratio = 320.0
    """
    s_gsd = max(float(source_gsd), 1e-4)
    r_gsd = max(float(reference_gsd), 1e-4)
    ratio = s_gsd / r_gsd

    octaves = int(round(np.log2(max(ratio, 1.0 / ratio)))) if ratio != 1.0 else 0

    return {
        "source_gsd_m": s_gsd,
        "reference_gsd_m": r_gsd,
        "scale_ratio": round(ratio, 3),
        "coarser_sensor": "source" if ratio > 1.0 else ("reference" if ratio < 1.0 else "equal"),
        "recommended_octaves": octaves,
        "recommended_downsample_factor": max(1, int(round(ratio if ratio > 1.0 else 1.0 / ratio))),
    }
