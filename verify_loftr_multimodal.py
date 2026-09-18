"""
Verify LoFTR + MAGSAC++ on multimodal lunar terrain with scale/perspective/illumination shifts.
Demonstrates the full pipeline with genuine LoFTR inference -> confidence filtering -> MAGSAC++ -> spatial distribution -> sub-pixel refinement -> registration.
"""
import time
from pathlib import Path
import cv2
import numpy as np
import app_v3

class FakeUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_loftr_multimodal_pipeline():
    ohrc_path = Path(r"D:\SIH\CHANDRAYAN2\OHRC\ch2_ohr_ncp_20241115T1326321339_d_img_d18\browse\calibrated\20241115\ch2_ohr_ncp_20241115T1326321339_b_brw_d18.png")
    assert ohrc_path.exists()

    # Read original lunar image
    base = cv2.imread(str(ohrc_path), cv2.IMREAD_GRAYSCALE)
    h, w = base.shape[:2]
    cy, cx = h // 2, w // 2

    # Create source crop (simulating OHRC high-res view of craters)
    crop_src = base[cy - 250 : cy + 250, cx - 250 : cx + 250].copy()

    # Create reference crop (simulating TMC-2 view: expanded FOV, non-linear radiometric response, rotation, affine scale)
    crop_ref = base[cy - 280 : cy + 280, cx - 280 : cx + 280].copy()
    # Apply non-linear sensor response (gamma curve simulating different optical band / TMC vs OHRC response)
    crop_ref = np.power(crop_ref / 255.0, 1.3) * 255.0
    # Add slight rotation & scale
    M = cv2.getRotationMatrix2D((crop_ref.shape[1] / 2.0, crop_ref.shape[0] / 2.0), 3.0, 0.96)
    crop_ref = cv2.warpAffine(crop_ref, M, (crop_ref.shape[1], crop_ref.shape[0]))
    crop_ref = np.clip(crop_ref, 0, 255).astype(np.uint8)

    _, enc_src = cv2.imencode(".png", crop_src)
    _, enc_ref = cv2.imencode(".png", crop_ref)

    src_file = FakeUpload("ohrc_southpole_crop.png", enc_src.tobytes())
    ref_file = FakeUpload("tmc2_multimodal_crop.png", enc_ref.tobytes())

    print("\n==================================================")
    print("RUNNING MULTIMODAL EXPERIMENT: OHRC -> TMC-2")
    print("Matcher: LoFTR (Cross-sensor)")
    print("Verifier: MAGSAC++")
    print("==================================================")

    t0 = time.perf_counter()
    res = app_v3.run_pipeline(
        source_file=src_file,
        reference_file=ref_file,
        source_sensor="OHRC",
        reference_sensor="TMC",
        max_side=600,
        geometric_verifier="MAGSAC++",
        loftr_confidence_threshold=0.30,
        model="Homography",
    )
    elapsed = time.perf_counter() - t0

    report = res["report"]
    ms = report["matching_strategy"]

    print(f"PIPELINE STATUS: COMPLETED")
    print(f"Matcher: {ms.get('matcher')}")
    print(f"Sensor Relation: {ms.get('sensor_relation')}")
    print(f"LoFTR Execution Device: {ms.get('loftr_metrics', {}).get('device')}")
    print(f"LoFTR Raw Matches: {ms.get('loftr_metrics', {}).get('raw_matches')}")
    print(f"LoFTR Confidence-Filtered Matches: {ms.get('loftr_metrics', {}).get('filtered_matches')}")
    print(f"Geometric Verifier: {ms.get('geometric_verifier')}")
    print(f"Fallback Used: {ms.get('fallback_used')}")
    print(f"Total Fused Correspondences: {report.get('total_fused_correspondences')}")
    print(f"Verified Inliers: {report.get('training_inliers')}")
    print(f"Inlier Ratio: {report.get('training_inlier_ratio', 0.0) * 100:.2f}%")
    print(f"Fit RMSE: {report.get('fit_rmse_working_px'):.4f} px")
    print(f"Holdout RMSE: {report.get('holdout_rmse_working_px'):.4f} px")
    print(f"Sub-pixel Refinement Applied: {report.get('subpixel_refinement_applied')}")
    print(f"Registered Image Shape: {res['registered'].shape}")
    print(f"Elapsed Time: {elapsed:.2f}s")


if __name__ == "__main__":
    test_loftr_multimodal_pipeline()
