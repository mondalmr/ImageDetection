from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


def _block_entropy(block: np.ndarray) -> float:
    hist, _ = np.histogram(block.flatten(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def analyze_pixel_anomalies(image: Image.Image) -> tuple[float, list[str], dict[str, Any]]:
    flags: list[str] = []
    gray = np.array(image.convert("L"))
    block_size = 32
    height, width = gray.shape

    noise_values: list[float] = []
    entropy_values: list[float] = []
    std_values: list[float] = []
    anomalous_blocks = 0
    total_blocks = 0

    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            block = gray[y : y + block_size, x : x + block_size]
            laplacian = cv2.Laplacian(block, cv2.CV_64F)
            noise_values.append(float(np.var(laplacian)))
            entropy_values.append(_block_entropy(block))
            std_values.append(float(np.std(block)))
            total_blocks += 1

    if total_blocks == 0:
        return 0.0, flags, {"total_blocks": 0}

    noise_arr = np.array(noise_values)
    entropy_arr = np.array(entropy_values)
    std_arr = np.array(std_values)

    noise_median = float(np.median(noise_arr))
    noise_std = float(np.std(noise_arr)) or 1.0
    entropy_median = float(np.median(entropy_arr))
    entropy_std = float(np.std(entropy_arr)) or 1.0
    std_median = float(np.median(std_arr))
    std_std = float(np.std(std_arr)) or 1.0

    for noise, entropy, std_val in zip(noise_arr, entropy_arr, std_arr):
        noise_z = abs(noise - noise_median) / noise_std
        entropy_z = abs(entropy - entropy_median) / entropy_std
        std_z = abs(std_val - std_median) / std_std
        if noise_z > 2.0 or entropy_z > 2.0 or std_z > 2.0:
            anomalous_blocks += 1

    anomalous_fraction = anomalous_blocks / total_blocks
    noise_cv = float(np.std(noise_arr) / (np.mean(noise_arr) + 1e-6))

    if anomalous_fraction > 0.15:
        flags.append("high_anomalous_block_fraction")
    if noise_cv > 0.8:
        flags.append("inconsistent_noise_distribution")

    score = min(1.0, 0.6 * anomalous_fraction + 0.4 * min(1.0, noise_cv))

    details = {
        "total_blocks": total_blocks,
        "anomalous_blocks": anomalous_blocks,
        "anomalous_fraction": round(anomalous_fraction, 4),
        "noise_coefficient_of_variation": round(noise_cv, 4),
        "median_noise_variance": round(noise_median, 4),
    }

    return score, flags, details
