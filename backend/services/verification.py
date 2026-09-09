from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from detectors.compression import analyze_compression
from detectors.frequency import analyze_frequency
from detectors.metadata import analyze_metadata
from detectors.ml_model import analyze_ml_model
from detectors.pixel_stats import analyze_pixel_anomalies
from schemas.verification import CheckResult, VerificationChecks, VerificationResponse

MORPHED_THRESHOLD = 0.65
ORIGINAL_THRESHOLD = 0.35

CHECK_WEIGHTS: dict[str, float] = {
    "metadata": 0.10,
    "pixel_anomalies": 0.15,
    "frequency_analysis": 0.20,
    "compression_regions": 0.20,
    "ml_model": 0.35,
}


def _run_check(name: str, func, *args) -> CheckResult:
    score, flags, details = func(*args)
    model = details.get("model") if name == "ml_model" else None
    return CheckResult(score=score, flags=flags, details=details, model=model)


def verify_image(raw_bytes: bytes, image: Image.Image) -> VerificationResponse:
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            "metadata": executor.submit(_run_check, "metadata", analyze_metadata, image, raw_bytes),
            "pixel_anomalies": executor.submit(_run_check, "pixel_anomalies", analyze_pixel_anomalies, image),
            "frequency_analysis": executor.submit(_run_check, "frequency_analysis", analyze_frequency, image),
            "compression_regions": executor.submit(_run_check, "compression_regions", analyze_compression, image),
            "ml_model": executor.submit(_run_check, "ml_model", analyze_ml_model, image),
        }
        results = {name: future.result() for name, future in futures.items()}

    active_scores: list[float] = []
    active_weights: list[float] = []

    for name, result in results.items():
        if result.score is not None:
            active_scores.append(result.score)
            active_weights.append(CHECK_WEIGHTS[name])

    if active_weights:
        total_weight = sum(active_weights)
        morph_score = sum(s * w for s, w in zip(active_scores, active_weights)) / total_weight
    else:
        morph_score = 0.5

    if len(active_scores) >= 2:
        confidence = float(max(0.0, min(1.0, 1.0 - np.std(active_scores))))
    elif len(active_scores) == 1:
        confidence = 0.5
    else:
        confidence = 0.0

    if morph_score >= MORPHED_THRESHOLD:
        verdict = "likely_morphed"
    elif morph_score <= ORIGINAL_THRESHOLD:
        verdict = "likely_original"
    else:
        verdict = "inconclusive"

    checks = VerificationChecks(
        metadata=results["metadata"],
        pixel_anomalies=results["pixel_anomalies"],
        frequency_analysis=results["frequency_analysis"],
        compression_regions=results["compression_regions"],
        ml_model=results["ml_model"],
    )

    return VerificationResponse(
        verdict=verdict,
        confidence=round(confidence, 4),
        morph_score=round(float(morph_score), 4),
        checks=checks,
    )
