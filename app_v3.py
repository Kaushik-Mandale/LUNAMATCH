"""LunaMatch V3 — Cloud-safe bootstrap (base64 parts)."""
from pathlib import Path
import base64
parts = sorted(Path(__file__).parent.glob("_b64boot_*.txt"))
if not parts:
    raise ImportError("missing _b64boot_*.txt")
raw = base64.b64decode("".join(p.read_text() for p in parts).encode("ascii"))
exec(compile(raw.decode("utf-8"), "app_v3_bootstrap.py", "exec"), globals())
