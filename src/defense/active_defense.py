"""
SignalScope Active Defence Analysis

Purpose:
    Evaluate how easily the existing SignalScope detector is changed by
    common post-processing attacks WITHOUT retraining and WITHOUT changing
    model weights or the production threshold.

The analysis uses the fixed production settings:
    temperature = 0.95
    threshold   = 0.56

It reports:
    - original prediction
    - attacked prediction
    - prediction flips
    - AI probability change
    - per-attack flip rate
    - per-class flip rate

This is a diagnostic / bonus-module analysis, not threshold tuning.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms

from src.models.forensic_model import SignalScopeForensicModel


IMAGE_SIZE = 32
TEMPERATURE = 0.95
THRESHOLD = 0.56


@dataclass(frozen=True)
class Attack:
    name: str
    transform: Callable[[Image.Image], Image.Image]


def jpeg_attack(quality: int) -> Callable[[Image.Image], Image.Image]:
    def apply(image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    return apply


def resize_roundtrip(target_size: int) -> Callable[[Image.Image], Image.Image]:
    def apply(image: Image.Image) -> Image.Image:
        small = image.resize(
            (target_size, target_size),
            Image.Resampling.BILINEAR,
        )
        return small.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )

    return apply


def add_noise(std: float) -> Callable[[Image.Image], Image.Image]:
    def apply(image: Image.Image) -> Image.Image:
        arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
        noise = np.random.default_rng(42).normal(
            0.0,
            std,
            size=arr.shape,
        )
        out = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray(
            (out * 255.0).round().astype(np.uint8),
            mode="RGB",
        )

    return apply


ATTACKS = [
    Attack("jpeg_70", jpeg_attack(70)),
    Attack("jpeg_50", jpeg_attack(50)),
    Attack(
        "resize_16_roundtrip",
        resize_roundtrip(16),
    ),
    Attack(
        "resize_8_roundtrip",
        resize_roundtrip(8),
    ),
    Attack(
        "gaussian_blur",
        lambda image: image.filter(
            ImageFilter.GaussianBlur(radius=0.8)
        ),
    ),
    Attack(
        "contrast_low",
        lambda image: ImageEnhance.Contrast(image).enhance(0.75),
    ),
    Attack(
        "contrast_high",
        lambda image: ImageEnhance.Contrast(image).enhance(1.25),
    ),
    Attack(
        "brightness_low",
        lambda image: ImageEnhance.Brightness(image).enhance(0.80),
    ),
    Attack(
        "brightness_high",
        lambda image: ImageEnhance.Brightness(image).enhance(1.20),
    ),
    Attack(
        "gaussian_noise",
        add_noise(0.03),
    ),
]


def collect_images(directory: Path, label: int, limit: int) -> list[tuple[Path, int]]:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    if limit > 0:
        paths = paths[:limit]

    return [(path, label) for path in paths]


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = SignalScopeForensicModel().to(device)
    checkpoint_data = torch.load(
        checkpoint,
        map_location=device,
    )
    state = checkpoint_data.get(
        "model_state_dict",
        checkpoint_data.get("state_dict", checkpoint_data),
    )
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
) -> tuple[float, int, float]:
    tensor = preprocess(image).to(device)
    logit = model(tensor)
    calibrated_logit = logit / TEMPERATURE
    probability = torch.sigmoid(calibrated_logit).item()
    predicted = int(probability >= THRESHOLD)
    decision_strength = (
        probability if predicted == 1 else 1.0 - probability
    )
    return probability, predicted, decision_strength


def evaluate(
    model: torch.nn.Module,
    samples: list[tuple[Path, int]],
    device: torch.device,
) -> list[dict]:
    rows: list[dict] = []

    for path, true_label in samples:
        try:
            with Image.open(path) as image:
                original = image.convert("RGB").copy()

            base_probability, base_prediction, base_strength = predict(
                model,
                original,
                device,
            )

            for attack in ATTACKS:
                attacked = attack.transform(original.copy())
                attack_probability, attack_prediction, attack_strength = predict(
                    model,
                    attacked,
                    device,
                )

                rows.append(
                    {
                        "image": str(path),
                        "true_class": "AI-generated"
                        if true_label == 1
                        else "REAL",
                        "attack": attack.name,
                        "original_probability": base_probability,
                        "attacked_probability": attack_probability,
                        "probability_delta": (
                            attack_probability - base_probability
                        ),
                        "original_prediction": base_prediction,
                        "attacked_prediction": attack_prediction,
                        "prediction_flipped": (
                            int(attack_prediction != base_prediction)
                        ),
                        "original_strength": base_strength,
                        "attacked_strength": attack_strength,
                        "strength_delta": (
                            attack_strength - base_strength
                        ),
                    }
                )
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    return rows


def summarise(rows: list[dict]) -> list[dict]:
    by_attack: dict[str, list[dict]] = {}

    for row in rows:
        by_attack.setdefault(row["attack"], []).append(row)

    summary: list[dict] = []

    for attack, attack_rows in by_attack.items():
        flips = [r["prediction_flipped"] for r in attack_rows]
        prob_deltas = [abs(r["probability_delta"]) for r in attack_rows]

        real_rows = [
            r for r in attack_rows
            if r["true_class"] == "REAL"
        ]
        fake_rows = [
            r for r in attack_rows
            if r["true_class"] == "AI-generated"
        ]

        summary.append(
            {
                "attack": attack,
                "samples": len(attack_rows),
                "flip_rate": float(np.mean(flips)),
                "mean_abs_probability_change": float(
                    np.mean(prob_deltas)
                ),
                "real_flip_rate": (
                    float(np.mean([
                        r["prediction_flipped"]
                        for r in real_rows
                    ]))
                    if real_rows
                    else 0.0
                ),
                "ai_flip_rate": (
                    float(np.mean([
                        r["prediction_flipped"]
                        for r in fake_rows
                    ]))
                    if fake_rows
                    else 0.0
                ),
            }
        )

    return sorted(
        summary,
        key=lambda row: (
            row["flip_rate"],
            row["mean_abs_probability_change"],
        ),
        reverse=True,
    )


def write_outputs(
    rows: list[dict],
    summary: list[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    details_csv = output_dir / "attack_results.csv"
    with details_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys() if rows else [],
        )
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary_json = output_dir / "attack_summary.json"
    summary_json.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# SignalScope Active Defence Analysis",
        "",
        "Production model weights and threshold were not changed.",
        "",
        f"- Temperature: `{TEMPERATURE}`",
        f"- Threshold: `{THRESHOLD}`",
        f"- Samples per class: controlled by `--limit`",
        "",
        "## Attack Summary",
        "",
        "| Attack | Samples | Flip rate | Mean |Δ AI probability| | REAL flip | AI flip |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in summary:
        lines.append(
            "| "
            f"{row['attack']} | "
            f"{row['samples']} | "
            f"{row['flip_rate'] * 100:.2f}% | "
            f"{row['mean_abs_probability_change']:.4f} | "
            f"{row['real_flip_rate'] * 100:.2f}% | "
            f"{row['ai_flip_rate'] * 100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A prediction flip means the attack changed the model's binary "
            "decision at the fixed production threshold. Higher flip rates "
            "indicate greater sensitivity to that degradation.",
            "",
            "This module is an analysis tool, not a new training stage. "
            "The attack set is not used to retune the threshold or model weights.",
            "",
            "## Defensive Guidance",
            "",
            "When a high flip rate is observed, the system should surface "
            "a robustness warning and recommend human review rather than "
            "presenting the prediction as definitive.",
            "",
        ]
    )

    (output_dir / "active_defense_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SignalScope active-defence / post-processing analysis."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/best_model.pt"),
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fake-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum images per class. Use 0 for all images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/active_defense"),
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("SignalScope Active Defence Analysis")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Threshold: {THRESHOLD}")
    print()

    real_samples = collect_images(
        args.real_dir,
        label=0,
        limit=args.limit,
    )
    fake_samples = collect_images(
        args.fake_dir,
        label=1,
        limit=args.limit,
    )

    samples = real_samples + fake_samples

    if not samples:
        raise RuntimeError("No evaluation images were found.")

    print(f"REAL samples: {len(real_samples)}")
    print(f"FAKE samples: {len(fake_samples)}")
    print(f"Total samples: {len(samples)}")
    print()

    model = load_model(
        args.checkpoint,
        device,
    )

    rows = evaluate(
        model,
        samples,
        device,
    )

    summary = summarise(rows)
    write_outputs(
        rows,
        summary,
        args.output_dir,
    )

    print()
    print("Attack summary:")
    for row in summary:
        print(
            f"{row['attack']:24s} "
            f"flip={row['flip_rate'] * 100:6.2f}% "
            f"| REAL={row['real_flip_rate'] * 100:6.2f}% "
            f"| AI={row['ai_flip_rate'] * 100:6.2f}%"
        )

    print()
    print(f"Results: {args.output_dir}")
    print("No retraining or threshold tuning was performed.")


if __name__ == "__main__":
    main()
