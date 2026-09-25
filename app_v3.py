"""LunaMatch V3 entry — assembles bootstrap from parts then runs it."""
from pathlib import Path
_parts = sorted(Path(__file__).parent.glob("_boot_part_*.txt"))
if not _parts:
    raise ImportError("missing _boot_part_*.txt — Cloud OOM bootstrap parts not deployed")
_src = "".join(p.read_text(encoding="utf-8") for p in _parts)
exec(compile(_src, "app_v3_bootstrap.py", "exec"), globals())
