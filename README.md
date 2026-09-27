---
title: LunaMatch
emoji: 🌙
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Cross-sensor lunar image correspondence (OHRC ↔ LROC) with ROI-LoFTR
tags:
  - streamlit
  - computer-vision
  - remote-sensing
  - lunar
  - loftr
---

## 🤗 Hugging Face Spaces (more RAM for LoFTR)

Streamlit Cloud ≈1–2.7 GB. Free HF **CPU Basic** ≈ **16 GB** — better for larger ROI / multi-tile LoFTR.

### Deploy (one-time)
1. Open [huggingface.co/new-space](https://huggingface.co/new-space)
2. **SDK = Docker**, hardware **CPU basic** (free), public
3. Push this repo into the Space:
```bash
git clone https://huggingface.co/spaces/<YOUR_HF_USER>/LUNAMATCH
cd LUNAMATCH
git remote add github https://github.com/Kaushik-Mandale/LUNAMATCH.git
git fetch github main
git checkout -B main github/main
git push -u origin main --force
```
4. Build uses `Dockerfile` → `streamlit run app_v3.py` on port **7860**
5. Optional variables: `LUNAMATCH_HF_SPACE=1`, `LUNAMATCH_ROI_LOFTR=1`

Streamlit Cloud demo: https://lunamatch.streamlit.app

---

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

Core path: **ROI-gated LoFTR** (scale-aware) → multi-pass MAGSAC++ → sub-pixel refinement. Module `guided_match.py` implements the register-then-rematch cascade for denser coverage.

See full pipeline stages and tests in the repository history and `app_v3.py` bootstrap.
