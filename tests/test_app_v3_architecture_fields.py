from pathlib import Path


def test_classical_pipeline_metrics_architecture_terms_exist_in_v3_app():
    text = Path("app_v3.py").read_text(encoding="utf-8")

    required_terms = [
        "RMSE",
        "Inlier Ratio",
        "Match Count",
        "Inlier Count",
        "Spatial Coverage",
        "Processing Time",
        "SIFT",
        "RANSAC",
        "Sub-pixel refinement",
        "Image Registration",
    ]

    missing = [term for term in required_terms if term not in text]
    assert not missing, f"Missing required architecture terms: {missing}"
