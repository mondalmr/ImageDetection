from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pywt
from PIL import Image


def _fft_anomaly_score(gray: np.ndarray) -> float:
    f_transform = np.fft.fft2(gray.astype(np.float32))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    magnitude = magnitude / (magnitude.max() + 1e-6)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_radius = min(cx, cy)

    inner_mask = radius <= max_radius * 0.3
    outer_mask = (radius > max_radius * 0.3) & (radius <= max_radius * 0.9)

    inner_energy = float(magnitude[inner_mask].mean()) if inner_mask.any() else 0.0
    outer_energy = float(magnitude[outer_mask].mean()) if outer_mask.any() else 0.0
    ratio = outer_energy / (inner_energy + 1e-6)

    return min(1.0, max(0.0, (ratio - 0.5) / 2.0))


def _wavelet_boundary_score(gray: np.ndarray) -> float:
    coeffs = pywt.wavedec2(gray.astype(np.float32), "db4", level=2)
    detail = coeffs[1][2]
    h, w = detail.shape
    grid = 8
    cell_h = max(1, h // grid)
    cell_w = max(1, w // grid)

    boundary_energies: list[float] = []
    interior_energies: list[float] = []

    for row in range(grid):
        for col in range(grid):
            y0, x0 = row * cell_h, col * cell_w
            y1 = min(h, y0 + cell_h)
            x1 = min(w, x0 + cell_w)
            cell = np.abs(detail[y0:y1, x0:x1])
            energy = float(np.mean(cell))
            is_boundary = row in (0, grid - 1) or col in (0, grid - 1)
            if is_boundary:
                boundary_energies.append(energy)
            else:
                interior_energies.append(energy)

    if not boundary_energies or not interior_energies:
        return 0.0

    boundary_mean = float(np.mean(boundary_energies))
    interior_mean = float(np.mean(interior_energies))
    ratio = boundary_mean / (interior_mean + 1e-6)
    return min(1.0, max(0.0, (ratio - 1.0) / 1.5))


def _edge_discontinuity_score(gray: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)

    edges = cv2.Canny(gray, 50, 150)
    edge_gradients = gradient[edges > 0]
    if edge_gradients.size == 0:
        return 0.0

    gradient_std = float(np.std(edge_gradients))
    gradient_mean = float(np.mean(edge_gradients))
    cv = gradient_std / (gradient_mean + 1e-6)
    return min(1.0, max(0.0, (cv - 0.5) / 1.5))


def analyze_frequency(image: Image.Image) -> tuple[float, list[str], dict[str, Any]]:
    flags: list[str] = []
    gray = np.array(image.convert("L"))

    if gray.shape[0] < 64 or gray.shape[1] < 64:
        resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(gray, (min(512, gray.shape[1]), min(512, gray.shape[0])), interpolation=cv2.INTER_AREA)

    fft_score = _fft_anomaly_score(resized)
    wavelet_score = _wavelet_boundary_score(resized)
    edge_score = _edge_discontinuity_score(resized)

    if fft_score > 0.5:
        flags.append("unnatural_fft_periodicity")
    if wavelet_score > 0.5:
        flags.append("wavelet_boundary_anomaly")
    if edge_score > 0.5:
        flags.append("edge_blending_discontinuity")

    score = min(1.0, 0.35 * fft_score + 0.35 * wavelet_score + 0.30 * edge_score)

    details = {
        "fft_anomaly_score": round(fft_score, 4),
        "wavelet_boundary_score": round(wavelet_score, 4),
        "edge_discontinuity_score": round(edge_score, 4),
    }

    return score, flags, details
