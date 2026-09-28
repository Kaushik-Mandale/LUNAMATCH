<div align="center">

<a href="https://lunamatch.streamlit.app" target="_blank"><img src="https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"/></a>
<img src="https://img.shields.io/badge/Team-Akatsuki-red?style=for-the-badge&logo=rocket&logoColor=white"/>
<img src="https://img.shields.io/badge/Deep%20Learning-LoFTR-purple?style=for-the-badge&logo=pytorch&logoColor=white"/>

# 🌙 LunaMatch — Lunar Image Correspondence Engine

### *Multi-modal · Sun-angle Invariant · Scale Invariant*
### *Chandrayaan-2 Optical Image Correspondence — OHRC · TMC-2 · IIRS*

</div>

---

## Overview

**LunaMatch** is a research-grade image correspondence pipeline by **Team Akatsuki** for Chandrayaan-2 lunar optical imagery (same-sensor and cross-sensor OHRC ↔ LROC/TMC/IIRS).

**Live demo:** [lunamatch.streamlit.app](https://lunamatch.streamlit.app)

Core path: **ROI-gated LoFTR** (scale-aware) → multi-pass MAGSAC++ → sub-pixel refinement. Module `guided_match.py` implements the register-then-rematch cascade for denser coverage.

## Run locally

```bash
git clone https://github.com/Kaushik-Mandale/LUNAMATCH.git
cd LUNAMATCH
pip install -r requirements.txt
streamlit run app_v3.py
```

## Deploy

Streamlit Community Cloud from this GitHub repo (`app_v3.py`). After pushes, use **Manage app → Reboot** if the UI sticks on “in the oven”.
