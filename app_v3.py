"""Bootstrap loader — assembles app_v3 soft-patcher from b64 parts."""
from pathlib import Path
import base64
_parts = sorted(Path(__file__).parent.glob("_boot_b64_*.txt"))
if not _parts:
    raise ImportError("missing _boot_b64_*.txt — redeploy required")
_src = base64.b64decode("".join(p.read_text() for p in _parts).encode("ascii")).decode("utf-8")
exec(compile(_src, "app_v3_boot_impl.py", "exec"), globals())
