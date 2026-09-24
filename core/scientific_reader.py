"""Scientific Raster Reader - loads chunked source for size-limited deploys."""
from pathlib import Path
_chunks = sorted(Path(__file__).parent.glob("_sr_chunk_*.txt"))
_src = "".join(p.read_text() for p in _chunks)
exec(compile(_src, str(Path(__file__).resolve()), "exec"), globals())
