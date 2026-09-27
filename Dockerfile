# LunaMatch — Hugging Face Spaces (Docker + Streamlit)
# Free CPU Basic: ~2 vCPU, ~16 GB RAM — enough for multi-tile / larger-ROI LoFTR
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    LUNAMATCH_ROI_LOFTR=1 \
    LUNAMATCH_HF_SPACE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TORCH_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# HF Docker Spaces expect the app on port 7860
EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app_v3.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.maxUploadSize=1024"]
