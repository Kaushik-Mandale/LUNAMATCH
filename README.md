<div align="center">

<a href="https://lunamatch.streamlit.app" target="_blank"><img src="https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"/></a>
<img src="https://img.shields.io/badge/Team-Akatsuki-red?style=for-the-badge&logo=rocket&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Deep%20Learning-LoFTR-purple?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenCV-4.10-green?style=for-the-badge&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/Tests-40%20passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white"/>

# 🌙 LunaMatch — Lunar Image Correspondence Engine

### *Multi-modal · Sun-angle Invariant · Scale Invariant*
### *Chandrayaan-2 Optical Image Correspondence — OHRC · TMC-2 · IIRS*

</div>

---

## 🚀 Overview

**LunaMatch** is a research-grade image correspondence pipeline built by **Team Akatsuki** for processing Chandrayaan-2 lunar optical imagery. It solves the hard problem of finding accurate correspondences between:

- 📡 **Same-sensor images** — OHRC ↔ OHRC, TMC-2 ↔ TMC-2  
- 🔀 **Cross-sensor (multimodal) images** — OHRC ↔ TMC-2, OHRC ↔ IIRS

The engine automatically selects the best matching strategy based on sensor type, then runs a full multi-stage geometric verification, sub-pixel refinement, and registration pipeline.

---

## 🏗️ Architecture

```
Chandrayaan-2 Image Pair
         │
         ▼
┌─────────────────────────────┐
│     Metadata Analysis       │  ← XML parsing: lat/lon, GSD, Sun geometry,
│  Sensor Type Detection      │     orbit, footprint, camera angles
└──────────────┬──────────────┘
               │
       ┌───────┴───────┐
       │ Sensor Router │
       └───────┬───────┘
        Same?  │  Different?
       ┌───────┘  └───────┐
       ▼                  ▼
┌────────────┐     ┌─────────────────┐
│   SIFT     │     │  LoFTR (Deep    │
│ Classical  │     │  Learning)      │
│ Pipeline   │     │  Transformer    │
│ CLAHE +    │     │  kornia.feature │
│ Multi-scale│     │  .LoFTR         │
│ Pyramid    │     └────────┬────────┘
└─────┬──────┘              │
      └──────────┬──────────┘
                 ▼
     Confidence Filtering + Spatial Gating
                 ▼
      ┌──────────────────────┐
      │  Geometric Verifier  │
      │  MAGSAC++ (primary)  │  ← cv2.USAC_MAGSAC (OpenCV 4.10)
      │  RANSAC   (baseline) │
      └──────────┬───────────┘
                 ▼
     Spatial Distribution Grid Filter
                 ▼
     Sub-Pixel Refinement (cv2.cornerSubPix)
                 ▼
     Homography Warp Registration
                 ▼
     Evaluation Metrics + Scientific Report
```

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| 🧠 **LoFTR Deep Matching** | Genuine transformer-based dense matching for cross-sensor pairs. No fake SIFT fallback. |
| 🔬 **SIFT Classical Pipeline** | Multi-scale pyramid + CLAHE + ratio test for same-sensor pairs. |
| 🛡️ **MAGSAC++ Verification** | Primary geometric verifier (`cv2.USAC_MAGSAC`). Superior to RANSAC in high-outlier scenarios. |
| 📐 **Sub-Pixel Refinement** | `cv2.cornerSubPix` iterative corner refinement on inlier keypoints. |
| 🗺️ **Footprint Spatial Gating** | Lat/Lon footprint overlap used to pre-filter geometrically implausible matches. |
| 📊 **Scientific Quality Report** | Fit RMSE, Holdout RMSE, Inlier Ratio, Spatial Distribution Score, Overlap %. |
| 🔭 **Chandrayaan-2 Native** | Parses PDS4 XML metadata natively — no external dependencies for label parsing. |
| 🖥️ **Streamlit UI** | Interactive web dashboard with upload, metric cards, visualisations, and debug panel. |

---

## 📈 Benchmark Results

### Same-Sensor: OHRC ↔ OHRC (SIFT + MAGSAC++)

| Metric | Value |
|---|---|
| Candidate Matches | 588 |
| MAGSAC++ Inliers | 356 |
| Inlier Ratio | **80.91%** |
| Fit RMSE | **1.26 px** |
| Holdout RMSE | **1.35 px** |
| Overlap Area | **94.12%** |
| Runtime | 4.14s |

### Cross-Sensor: OHRC ↔ TMC-2 (LoFTR + MAGSAC++)

| Metric | Value |
|---|---|
| LoFTR Raw Matches | 2,016 |
| Confidence-Filtered | 1,882 |
| MAGSAC++ Inliers | 757 |
| Inlier Ratio | **98.44%** |
| Fit RMSE | **0.65 px** |
| Holdout RMSE | **0.99 px** |
| Sub-Pixel Refinement | ✅ Applied |
| Runtime | 7.02s |

---

## 🛠️ Installation

### Prerequisites

- Python 3.12+
- Windows / Linux / macOS
- 4 GB RAM minimum (8 GB recommended for LoFTR on CPU)
- CUDA-capable GPU optional (CPU inference fully supported)

### Setup

```powershell
# Clone the repository
git clone https://github.com/<your-org>/lunamatch.git
cd lunamatch

# Create a virtual environment
python -m venv .venv
.\.venv\Scripts\activate       # Windows
# source .venv/bin/activate    # Linux/macOS

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements_v3.txt
```

### LoFTR Pretrained Weights

The LoFTR `loftr_outdoor` weights are downloaded automatically on first run via the Kornia model zoo. No manual download required.

---

## 🖥️ Running LunaMatch

```powershell
# Start the Streamlit web application
streamlit run app_v3.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Live Demo

Try the deployed application: [LunaMatch V3 · Streamlit](https://lunamatch.streamlit.app/)

### CLI Verification Scripts

```powershell
# Run same-sensor OHRC experiment
.\.venv\Scripts\python.exe verify_experiments.py

# Run cross-sensor LoFTR multimodal experiment
.\.venv\Scripts\python.exe verify_loftr_multimodal.py

# Run the full test suite
.\.venv\Scripts\python.exe -m pytest -v
```

---

## 📂 Project Structure

```
lunamatch/
├── app_v3.py                  # Main Streamlit application & pipeline
├── loftr_matcher.py           # LoFTR deep learning inference module
├── requirements_v3.txt        # Python dependencies
├── verify_experiments.py      # OHRC same-sensor experiment script
├── verify_loftr_multimodal.py # LoFTR cross-sensor experiment script
└── tests/
    ├── test_app_v3_routing.py                      # Sensor routing tests
    ├── test_app_v3_architecture_fields.py          # Architecture field tests
    ├── test_app_v3_ui_presentation.py              # UI component tests
    ├── test_app_v3_parse_metadata_xml.py           # XML parsing tests
    ├── test_app_v3_chandrayaan2_regression.py      # Chandrayaan-2 regression
    ├── test_app_v3_chandrayaan2_real_pds4_regression.py  # PDS4 regression
    └── test_loftr_and_magsac.py                   # LoFTR + MAGSAC++ tests
```

---

## 🧪 Test Suite

```
============================= 35 passed in 3.41s ==============================
```

All 35 automated tests pass, covering:
- Sensor routing logic (same vs cross-sensor)
- LoFTR preprocessing (multiples-of-8 resize)
- MAGSAC++ and RANSAC geometric verification
- Sub-pixel refinement
- Image visualization with differing dimensions
- PDS4 XML metadata parsing
- UI component presence

---

## 🧩 Pipeline Stages (app_v3.py)

| Stage | Description |
|---|---|
| 01 | Load & validate source/reference images + metadata |
| 02 | CLAHE contrast enhancement |
| 03 | Multi-scale Gaussian pyramid construction |
| 04 | Gradient + structural representations |
| 05 | **Sensor classification** → route to SIFT or LoFTR |
| 06 | **SIFT** (same-sensor) or **LoFTR** (cross-sensor) matching |
| 07 | Confidence filtering + spatial footprint gating |
| 08 | Match fusion and spatial grid balancing |
| 09 | **MAGSAC++** geometric model estimation (+ RANSAC baseline) |
| 10 | Spatial distribution quality scoring |
| 11 | **Sub-pixel corner refinement** (`cv2.cornerSubPix`) |
| 12 | Homography warp image registration |
| 13 | Evaluation metrics: RMSE, inlier ratio, overlap, runtime |

---

## ⚙️ UI Controls

| Control | Description |
|---|---|
| Geometric Verification Method | `MAGSAC++` (recommended) or `RANSAC` |
| LoFTR Confidence Threshold | Slider 0.1–0.9 (default 0.35) for cross-sensor matching |
| Geometric Model | `Homography` or `Affine` |
| Experimental Comparison | Run both verifiers and compare results |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.63+ | Web UI |
| `opencv-python` | 4.10+ | SIFT, MAGSAC++, cv2.cornerSubPix |
| `torch` | 2.14+ | LoFTR deep learning inference |
| `kornia` | 0.8.3+ | LoFTR transformer model |
| `numpy` | 2.5+ | Array operations |
| `rasterio` | 1.5+ | GeoTIFF reading |
| `pillow` | 12+ | Image I/O |
| `pandas` | 3.0+ | Metrics display |
| `pytest` | 9.1+ | Test suite |

---

## 🔬 Scientific Notes

> **Prototype correspondence validation.** Reported residual errors (RMSE values) are image-space pixel residuals measured on the matched image pair without an independent ground-truth DEM or ground control points. These are indicators of internal geometric consistency, not certified absolute geolocation accuracy.

For a production-grade system, add:
- DEM/sensor-geometry based terrain correction
- Independent ground-control / checkpoint validation
- Full-resolution georeferenced GeoTIFF export
- Validated learned cross-modal matcher trained on lunar imagery

---

## 👥 Team Akatsuki

Built with ❤️ for advancing lunar remote sensing and planetary image analysis.

---

<div align="center">

*"To the Moon, and beyond."*

⭐ **Star this repo if you found it useful!** ⭐

</div>
