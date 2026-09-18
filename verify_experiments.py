"""
Verification script for LunaMatch V3:
1. TEST A: OHRC ↔ OHRC (Same sensor → SIFT + MAGSAC++)
2. TEST B: OHRC ↔ TMC-2 (Cross sensor → LoFTR + MAGSAC++)
3. TEST C: Verifier Comparison: MAGSAC++ vs RANSAC baseline
"""
import io
import time
from pathlib import Path
from PIL import Image
import numpy as np
import app_v3

class FakeUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def run_experiment(name: str, src_path: Path, ref_path: Path, src_sensor: str, ref_sensor: str, verifier: str = "MAGSAC++"):
    print(f"\n==================================================")
    print(f"EXPERIMENT: {name}")
    print(f"Source Sensor: {src_sensor} | Reference Sensor: {ref_sensor}")
    print(f"Requested Verifier: {verifier}")
    print(f"Source Image: {src_path.name}")
    print(f"Reference Image: {ref_path.name}")
    print(f"==================================================")

    src_file = FakeUpload(src_path.name, src_path.read_bytes())
    ref_file = FakeUpload(ref_path.name, ref_path.read_bytes())

    t0 = time.perf_counter()
    try:
        res = app_v3.run_pipeline(
            source_file=src_file,
            reference_file=ref_file,
            source_sensor=src_sensor,
            reference_sensor=ref_sensor,
            max_side=1024,
            geometric_verifier=verifier,
            loftr_confidence_threshold=0.25,
            model="Homography",
        )
        elapsed = time.perf_counter() - t0
        report = res["report"]
        ms = report["matching_strategy"]

        print(f"STATUS: SUCCESS")
        print(f"Matcher Executed: {ms.get('matcher')}")
        print(f"Sensor Relation: {ms.get('sensor_relation')}")
        print(f"Verifier Executed: {ms.get('geometric_verifier')}")
        print(f"Primary Verifier: {ms.get('primary_verifier')}")
        print(f"Fallback Used: {ms.get('fallback_used')}")
        print(f"Total Fused Matches: {report.get('total_fused_correspondences')}")
        print(f"Training Inliers: {report.get('training_inliers')}")
        print(f"Training Inlier Ratio: {report.get('training_inlier_ratio', 0.0) * 100:.2f}%")
        print(f"Fit RMSE (working px): {report.get('fit_rmse_working_px')}")
        print(f"Holdout RMSE (working px): {report.get('holdout_rmse_working_px')}")
        print(f"Reference Overlap: {report.get('reference_overlap_fraction', 0.0) * 100:.2f}%")
        print(f"Subpixel Refinement Applied: {report.get('subpixel_refinement_applied')}")
        print(f"Processing Time: {elapsed:.2f}s")
        return res
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"STATUS: FAILED ({e})")
        print(f"Elapsed Time: {elapsed:.2f}s")
        return None


if __name__ == "__main__":
    ohrc1 = Path(r"D:\SIH\CHANDRAYAN2\OHRC\ch2_ohr_ncp_20241115T1326321339_d_img_d18\browse\calibrated\20241115\ch2_ohr_ncp_20241115T1326321339_b_brw_d18.png")
    ohrc2 = Path(r"D:\SIH\CHANDRAYAN2\OHRC\ch2_ohr_ncp_20241115T1525004388_d_img_d18\browse\calibrated\20241115\ch2_ohr_ncp_20241115T1525004388_b_brw_d18.png")
    tmc2 = Path(r"D:\SIH\CHANDRAYAN2\TMC2\ch2_tmc_ndn_20230612T2218425665_d_oth_n18\browse\derived\20230612\ch2_tmc_ndn_20230612T2218425665_b_bot_n18.png")

    print(f"OHRC 1 exists: {ohrc1.exists()}")
    print(f"OHRC 2 exists: {ohrc2.exists()}")
    print(f"TMC2 exists: {tmc2.exists()}")

    # 1. TEST A: Same sensor OHRC <-> OHRC (SIFT + MAGSAC++)
    if ohrc1.exists() and ohrc2.exists():
        res_a = run_experiment("TEST A: Same Sensor (OHRC <-> OHRC)", ohrc1, ohrc2, "OHRC", "OHRC", verifier="MAGSAC++")

    # 2. TEST B: Cross sensor OHRC <-> TMC-2 (LoFTR + MAGSAC++)
    if ohrc1.exists() and tmc2.exists():
        res_b = run_experiment("TEST B: Cross Sensor (OHRC <-> TMC)", ohrc1, tmc2, "OHRC", "TMC", verifier="MAGSAC++")

        # 3. TEST C: Comparison experiment: LoFTR + RANSAC baseline
        res_c = run_experiment("TEST C: Cross Sensor Baseline (OHRC <-> TMC via RANSAC)", ohrc1, tmc2, "OHRC", "TMC", verifier="RANSAC")
