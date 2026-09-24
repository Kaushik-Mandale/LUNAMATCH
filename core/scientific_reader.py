"""Scientific Raster Reader - loads chunked source for size-limited deploys."""
from pathlib import Path
_chunks = sorted(Path(__file__).parent.glob("_sr_chunk_*.txt"))
if not _chunks:
    raise ImportError("core/scientific_reader.py: no _sr_chunk_*.txt files found.")
_src = "".join(p.read_text(encoding="utf-8") for p in _chunks)
exec(compile(_src, str(Path(__file__).resolve()), "exec"), globals())
