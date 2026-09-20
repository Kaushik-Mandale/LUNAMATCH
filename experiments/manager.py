"""
Experiment Manager for LunaMatch V3 (SIH26166).

Saves and organizes experiment records under experiments/<PAIR_ID>/:
- metadata.json (source and reference metadata, overlap analysis)
- config.json (pipeline hyperparameters, pyramid levels, matcher, verifier)
- metrics.json (computed reprojection RMSE, inliers, spatial coverage, timing)
"""
import io
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np


class ExperimentManager:
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_experiment(
        self,
        pair_id: str,
        source_meta: Dict[str, Any],
        reference_meta: Dict[str, Any],
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        source_preview: Optional[np.ndarray] = None,
        reference_preview: Optional[np.ndarray] = None,
        registered_img: Optional[np.ndarray] = None,
        correspondence_img: Optional[np.ndarray] = None,
    ) -> Path:
        """Save an experiment run to disk under experiments/<pair_id>/."""
        target_dir = self.base_dir / pair_id
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Metadata
        meta_payload = {
            "pair_id": pair_id,
            "source_metadata": source_meta,
            "reference_metadata": reference_meta,
        }
        (target_dir / "metadata.json").write_text(json.dumps(meta_payload, indent=2, default=str), encoding="utf-8")

        # 2. Config
        (target_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

        # 3. Metrics
        (target_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

        # 4. Optional Image Previews
        if source_preview is not None and source_preview.size > 0:
            cv2.imwrite(str(target_dir / "source_preview.png"), cv2.cvtColor(source_preview, cv2.COLOR_RGB2BGR) if source_preview.ndim == 3 else source_preview)
        if reference_preview is not None and reference_preview.size > 0:
            cv2.imwrite(str(target_dir / "reference_preview.png"), cv2.cvtColor(reference_preview, cv2.COLOR_RGB2BGR) if reference_preview.ndim == 3 else reference_preview)
        if registered_img is not None and registered_img.size > 0:
            cv2.imwrite(str(target_dir / "registration.png"), registered_img)
        if correspondence_img is not None and correspondence_img.size > 0:
            cv2.imwrite(str(target_dir / "matches_verified.png"), cv2.cvtColor(correspondence_img, cv2.COLOR_RGB2BGR) if correspondence_img.ndim == 3 else correspondence_img)

        return target_dir

    def list_experiments(self) -> list[str]:
        """List all saved experiment directory names."""
        if not self.base_dir.exists():
            return []
        return [p.name for p in self.base_dir.iterdir() if p.is_dir()]

    def load_experiment(self, pair_id: str) -> Optional[Dict[str, Any]]:
        """Load saved experiment JSON files."""
        target_dir = self.base_dir / pair_id
        if not target_dir.exists():
            return None

        result = {}
        if (target_dir / "metadata.json").exists():
            result["metadata"] = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
        if (target_dir / "config.json").exists():
            result["config"] = json.loads((target_dir / "config.json").read_text(encoding="utf-8"))
        if (target_dir / "metrics.json").exists():
            result["metrics"] = json.loads((target_dir / "metrics.json").read_text(encoding="utf-8"))
        return result
