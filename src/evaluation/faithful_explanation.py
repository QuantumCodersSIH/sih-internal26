"""
SignalScope Bonus A — Faithful, deterministic explanation utilities.

This module does not infer semantic artifacts. It derives explanation text
only from the existing model-produced Grad-CAM heatmap and final prediction.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class LocalizationResult:
    heatmap: np.ndarray
    mean_activation: float
    max_activation: float
    p90_activation: float
    top10_mean: float
    concentration_ratio: float
    active_fraction: float
    bbox: tuple[int, int, int, int] | None
    bbox_area_ratio: float
    localized: bool
    diffuse: bool


def _to_numpy_heatmap(heatmap: Any) -> np.ndarray:
    """Convert predictor Grad-CAM output to a finite 2-D float array."""
    if hasattr(heatmap, "detach"):
        heatmap = heatmap.detach().cpu().numpy()

    arr = np.asarray(heatmap, dtype=np.float32)

    while arr.ndim > 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])

    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D heatmap, got shape {arr.shape}.")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)

    low = float(arr.min())
    high = float(arr.max())
    if high > low:
        arr = (arr - low) / (high - low)
    else:
        arr.fill(0.0)

    return arr


def _largest_component(mask: np.ndarray) -> tuple[tuple[int, int, int, int] | None, int]:
    """Return largest 4-connected component bounding box and pixel count."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_bbox = None
    best_size = 0

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            queue = deque([(y, x)])
            visited[y, x] = True

            min_x = max_x = x
            min_y = max_y = y
            size = 0

            while queue:
                cy, cx = queue.popleft()
                size += 1

                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ):
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if size > best_size:
                best_size = size
                best_bbox = (min_x, min_y, max_x + 1, max_y + 1)

    return best_bbox, best_size


def analyze_localization(
    heatmap: Any,
    *,
    percentile: float = 90.0,
    min_active_fraction: float = 0.01,
    max_active_fraction_for_localized: float = 0.35,
    max_bbox_area_ratio_for_localized: float = 0.65,
) -> LocalizationResult:
    """
    Convert existing Grad-CAM output into measurable localization signals.

    A bounding box is rejected when the activation is too diffuse.
    """
    arr = _to_numpy_heatmap(heatmap)
    h, w = arr.shape
    total = float(h * w)

    mean_activation = float(arr.mean())
    max_activation = float(arr.max())
    p90_activation = float(np.percentile(arr, percentile))

    flat = np.sort(arr.reshape(-1))
    top_n = max(1, int(np.ceil(0.10 * flat.size)))
    top10_mean = float(flat[-top_n:].mean())
    concentration_ratio = (
        top10_mean / max(mean_activation, 1e-8)
        if mean_activation > 0
        else 0.0
    )

    # Avoid a zero-valued percentile threshold turning the whole image active.
    threshold = max(p90_activation, 0.35 * max_activation)
    mask = (arr >= threshold) & (arr > 0.0)
    active_fraction = float(mask.mean())

    bbox, component_pixels = _largest_component(mask)
    bbox_area_ratio = 1.0
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        bbox_area_ratio = float((x2 - x1) * (y2 - y1) / total)

    localized = (
        component_pixels >= max(3, int(0.01 * total))
        and active_fraction >= min_active_fraction
        and active_fraction <= max_active_fraction_for_localized
        and bbox_area_ratio <= max_bbox_area_ratio_for_localized
        and concentration_ratio >= 1.15
    )

    diffuse = not localized
    if diffuse:
        bbox = None

    return LocalizationResult(
        heatmap=arr,
        mean_activation=mean_activation,
        max_activation=max_activation,
        p90_activation=p90_activation,
        top10_mean=top10_mean,
        concentration_ratio=concentration_ratio,
        active_fraction=active_fraction,
        bbox=bbox,
        bbox_area_ratio=bbox_area_ratio,
        localized=localized,
        diffuse=diffuse,
    )


def uncertainty_text(ai_probability: float, threshold: float = 0.56) -> str:
    """Deterministic uncertainty language tied to the production score."""
    p = float(ai_probability)
    distance = abs(p - threshold)

    if distance <= 0.05:
        return (
            "Uncertain — the model's score is close to the decision threshold. "
            "Treat this as model-based forensic evidence, not proof."
        )

    if p >= 0.85:
        return (
            "The model strongly favors AI-generated imagery, but this is a "
            "likelihood assessment rather than proof of synthetic origin."
        )

    if p >= threshold:
        return (
            "The model leans toward AI-generated imagery, but the evidence is "
            "not absolute and should be interpreted with other forensic signals."
        )

    if p <= 0.15:
        return (
            "The model strongly favors real imagery, but a low AI likelihood is "
            "not proof of authenticity."
        )

    return (
        "The model leans toward real imagery, but the evidence is not decisive "
        "and should be interpreted with other forensic signals."
    )


def build_grounded_evidence(
    *,
    ai_probability: float,
    predicted_class: int,
    localization: LocalizationResult,
) -> dict[str, Any]:
    """Build deterministic evidence statements from measurable signals only."""
    p = float(ai_probability)
    class_name = "AI-generated" if int(predicted_class) == 1 else "REAL"
    evidence: list[str] = []

    if localization.localized:
        evidence.append(
            "Strong model activation is concentrated in a localized region rather than being distributed uniformly."
        )
        evidence.append(
            f"The highest-activation region covers about {localization.active_fraction:.1%} of the heatmap area."
        )
        evidence.append(
            f"The top 10% of activation is {localization.concentration_ratio:.1f}× stronger than the overall heatmap mean."
        )
    else:
        evidence.append(
            "The model evidence is distributed across the image rather than concentrated in a single region."
        )
        evidence.append(
            f"Activation concentration is limited (top-10% / mean ratio: {localization.concentration_ratio:.1f}×)."
        )

    if p >= 0.85 or p <= 0.15:
        evidence.append(
            f"The fused model score strongly favors the {class_name} decision (AI likelihood {p:.1%})."
        )
    else:
        evidence.append(
            f"The fused model score favors the {class_name} decision (AI likelihood {p:.1%}), but is not extreme."
        )

    evidence.append(
        "The visual map is a spatial Grad-CAM explanation of the final fused classifier; "
        "the model also uses a frequency branch, but no unsupported frequency contribution is claimed."
    )

    return {
        "verdict_class": class_name,
        "evidence": evidence,
        "uncertainty": uncertainty_text(p),
        "metrics": {
            "heatmap_mean": localization.mean_activation,
            "heatmap_max": localization.max_activation,
            "heatmap_p90": localization.p90_activation,
            "active_fraction": localization.active_fraction,
            "concentration_ratio": localization.concentration_ratio,
            "bbox_area_ratio": localization.bbox_area_ratio,
        },
        "bbox": localization.bbox,
        "localized": localization.localized,
        "diffuse": localization.diffuse,
    }


def make_heatmap_image(
    heatmap: Any,
    *,
    size: tuple[int, int] | None = None,
) -> Image.Image:
    """Create a deterministic pseudo-color heatmap from Grad-CAM values."""
    arr = _to_numpy_heatmap(heatmap)

    if size is not None:
        base = Image.fromarray(
            np.uint8(np.clip(arr * 255.0, 0, 255)),
            mode="L",
        )
        base = base.resize(size, Image.Resampling.BILINEAR)
        arr = np.asarray(base, dtype=np.float32) / 255.0

    red = np.clip(arr * 255.0, 0, 255)
    green = np.clip(np.sqrt(arr) * 220.0, 0, 255)
    blue = np.clip((1.0 - arr) * 80.0, 0, 255)
    rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)

    return Image.fromarray(rgb, mode="RGB")


def draw_evidence_region(
    image: Image.Image,
    bbox: tuple[int, int, int, int] | None,
    *,
    line_width: int = 4,
) -> Image.Image:
    """Draw the genuine activation bounding box when localization passed."""
    out = image.copy().convert("RGB")

    if bbox is None:
        return out

    draw = ImageDraw.Draw(out)
    draw.rectangle(bbox, outline=(220, 30, 30), width=line_width)
    return out
