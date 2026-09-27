"""LunaMatch V3 entry — load soft-patch bootstrap from app_v3_boot.py."""
from pathlib import Path
_p = Path(__file__).with_name("app_v3_boot.py")
if not _p.exists():
    raise FileNotFoundError("app_v3_boot.py missing — redeploy required")
exec(compile(_p.read_text(encoding="utf-8"), str(_p), "exec"), globals())
