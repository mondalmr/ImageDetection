from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


def _compute_ela_map(image: Image.Image, quality: int = 90) -> tuple[np.ndarray, bool]:
    converted = False
    work_image = image.convert("RGB")

    if image.format not in ("JPEG", "JPG"):
        converted = True

    buffer = BytesIO()
    work_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")

    original_arr = np.array(work_image, dtype=np.float32)
    resaved_arr = np.array(resaved, dtype=np.float32)
    ela_map = np.abs(original_arr - resaved_arr).mean(axis=2)
    return ela_map, converted


def analyze_compression(image: Image.Image) -> tuple[float, list[str], dict[str, Any]]:
    flags: list[str] = []
    ela_map, converted_for_ela = _compute_ela_map(image)

    grid_rows, grid_cols = 4, 4
    height, width = ela_map.shape
    cell_h = max(1, height // grid_rows)
    cell_w = max(1, width // grid_cols)

    cell_means: list[list[float]] = []
    for row in range(grid_rows):
        row_means: list[float] = []
        for col in range(grid_cols):
            y0, x0 = row * cell_h, col * cell_w
            y1 = min(height, y0 + cell_h)
            x1 = min(width, x0 + cell_w)
            row_means.append(float(ela_map[y0:y1, x0:x1].mean()))
        cell_means.append(row_means)

    cell_arr = np.array(cell_means)
    max_adjacent_ratio = 1.0
    adjacent_ratios: list[float] = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            current = cell_arr[row, col]
            neighbors = []
            if row > 0:
                neighbors.append(cell_arr[row - 1, col])
            if col > 0:
                neighbors.append(cell_arr[row, col - 1])
            for neighbor in neighbors:
                ratio = max(current, neighbor) / (min(current, neighbor) + 1e-6)
                adjacent_ratios.append(ratio)
                max_adjacent_ratio = max(max_adjacent_ratio, ratio)

    mean_ela = float(cell_arr.mean())
    std_ela = float(cell_arr.std())
    cv_ela = std_ela / (mean_ela + 1e-6)

    if max_adjacent_ratio > 2.5:
        flags.append("regional_compression_mismatch")
    if cv_ela > 0.6:
        flags.append("high_ela_variation_across_regions")

    ratio_score = min(1.0, max(0.0, (max_adjacent_ratio - 1.5) / 3.0))
    cv_score = min(1.0, max(0.0, (cv_ela - 0.3) / 0.7))
    score = min(1.0, 0.6 * ratio_score + 0.4 * cv_score)

    details = {
        "converted_for_ela": converted_for_ela,
        "grid_size": [grid_rows, grid_cols],
        "max_adjacent_ela_ratio": round(max_adjacent_ratio, 4),
        "ela_coefficient_of_variation": round(cv_ela, 4),
        "cell_means": [[round(v, 4) for v in row] for row in cell_means],
    }

    return score, flags, details
