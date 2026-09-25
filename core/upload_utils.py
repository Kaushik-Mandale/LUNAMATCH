"""Upload helpers optimised for large LROC scientific products on Streamlit Cloud.

Browser	o Cloud upload of ~250 MB EDR files is limited by the user's uplink.
These helpers:
  1. Avoid reading full bytes just to show size or detect file identity.
  2. Allow the *server* to fetch a product from a direct HTTP(S) URL
     (PDS / LROC / user-hosted), which is often much faster than a home uplink.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from typing import Any, Optional
from urllib.parse import urlparse

# Soft limit for URL downloads on Streamlit free tier (bytes)
DEFAULT_URL_MAX_BYTES = 400 * 1024 * 1024  # 400 MB


class MemUploadedFile:
    """Minimal stand-in for st.UploadedFile (name / size / getvalue / read)."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self.size = len(data)
        self._data = data

    def getvalue(self) -> bytes:
        return self._data

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self._data
        return self._data[:size]

    def seek(self, pos: int, whence: int = 0) -> int:
        return 0

    def tell(self) -> int:
        return 0


def safe_file_size(file_obj: Any) -> int:
    """Return byte size without forcing a full getvalue() when .size exists."""
    if file_obj is None:
        return 0
    size = getattr(file_obj, "size", None)
    if isinstance(size, int) and size >= 0:
        return size
    if hasattr(file_obj, "getvalue"):
        try:
            return len(file_obj.getvalue())
        except Exception:
            return 0
    return 0


def lightweight_file_id(file_obj: Any) -> str:
    """Stable short id from name+size (no full-body hash for large scientific rasters).

    Full SHA-256 of a 252 MB buffer doubles peak RAM on Streamlit Cloud.
    Name+size is enough to detect replacement for session-cache invalidation.
    """
    if file_obj is None:
        return ""
    name = getattr(file_obj, "name", "") or ""
    size = safe_file_size(file_obj)
    raw = f"{name}|{size}".encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:16]


def _filename_from_url(url: str) -> str:
    path = urlparse(url).path or ""
    base = os.path.basename(path)
    if base and "." in base:
        return base
    return "remote_product.img"


def download_url_product(
    url: str,
    *,
    max_bytes: int = DEFAULT_URL_MAX_BYTES,
    timeout: int = 600,
    progress_cb=None,
) -> MemUploadedFile:
    """Download a product on the *server* and wrap it as an UploadedFile-like object.

    Raises ValueError on invalid URL / size / HTTP errors.
    """
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "LunaMatch/3 (Streamlit scientific pipeline)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # Honour Content-Length when present
            cl = resp.headers.get("Content-Length")
            if cl is not None:
                try:
                    declared = int(cl)
                    if declared > max_bytes:
                        raise ValueError(
                            f"Remote file is {declared / (1024**2):.1f} MB; "
                            f"limit is {max_bytes / (1024**2):.0f} MB."
                        )
                except ValueError as exc:
                    if "limit is" in str(exc):
                        raise
            chunks = []
            total = 0
            while True:
                block = resp.read(1024 * 1024)  # 1 MB chunks
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise ValueError(
                        f"Download exceeded {max_bytes / (1024**2):.0f} MB limit."
                    )
                chunks.append(block)
                if progress_cb is not None:
                    progress_cb(total)
            data = b"".join(chunks)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"HTTP {exc.code} fetching URL: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"Could not reach URL: {exc.reason}") from exc

    if not data:
        raise ValueError("Downloaded file is empty.")

    name = _filename_from_url(url)
    return MemUploadedFile(name, data)


def write_temp_product(file_obj: Any, suffix: str = ".img") -> str:
    """Write uploaded bytes to a temp file path (caller should unlink when done)."""
    data = file_obj.getvalue() if hasattr(file_obj, "getvalue") else bytes(file_obj)
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path
