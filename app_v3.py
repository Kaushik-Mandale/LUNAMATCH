"""LunaMatch V3 entrypoint - loads chunked source for size-limited deploys."""
from pathlib import Path
_chunks = sorted(Path(__file__).parent.glob("_app_chunk_*.txt"))
if not _chunks:
    raise ImportError("app_v3.py: no _app_chunk_*.txt files found next to this loader.")
_src = "".join(p.read_text(encoding="utf-8") for p in _chunks)
exec(compile(_src, str(Path(__file__).resolve()), "exec"), globals())
