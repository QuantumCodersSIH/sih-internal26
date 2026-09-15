"""
SignalScope branch-contribution analysis.

Purpose:
    Analyze the existing trained model WITHOUT retraining and WITHOUT changing
    model weights. For known REAL false positives and known AI false negatives,
    compare:
      1) full model probability
      2) spatial-branch ablation probability
      3) frequency-branch ablation probability

The ablation scores answer: "How much does each branch contribute to the
final AI decision?"

Run from the SignalScope project root.

Example:
    python scripts/analyze_branch_contribution.py ^
      --real-csv "C:\path\TEMP_TEST_predictions(2).csv" ^
      --real-dir "C:\path\REAL" ^
      --fake-csv "C:\path\TEMP_TEST_predictions2.csv.xlsx" ^
      --fake-dir "C:\path\FAKE" ^
      --checkpoint "models\checkpoints\best_model.pt"

If the ZIPs have not been extracted, extract them first. This script does
NOT add any evaluation images to data/train.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.forensic_model import SignalScopeForensicModel
from src.data.augmentations import get_val_transforms


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return ckpt


def normalize_rel_path(value: str) -> str:
    value = str(value).strip().replace("\\", "/")
    if value.startswith("TEMP_TEST/"):
        value = value[len("TEMP_TEST/"):]
    return value


def find_image(root: Path, csv_value: str) -> Path | None:
    rel = normalize_rel_path(csv_value)
    direct = root / rel
    if direct.exists():
        return direct

    name = Path(rel).name
    candidates = list(root.rglob(name))
    if candidates:
        return candidates[0]

    # Some ZIPs contain a TEMP_TEST directory.
    candidates = list(root.rglob(Path(rel).name))
    return candidates[0] if candidates else None


def maybe_extract(zip_path: Path, destination: Path) -> Path:
    if destination.exists() and any(destination.rglob("*.jpg")):
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(destination)
    return destination


def make_model() -> torch.nn.Module:
    # This matches the current SignalScope architecture.
    return SignalScopeForensicModel()


def branch_scores(model, x: torch.Tensor):
    """
    Returns probabilities for:
      full
      spatial ablated (frequency only)
      frequency ablated (spatial only)

    IMPORTANT:
    These are diagnostic ablations, not probabilities from independently
    trained models. A branch score can be negative/positive in logit space;
    sigmoid converts the diagnostic logit to a 0..1 scale.
    """
    spatial = model.spatial(x)
    frequency = model.frequency(x)

    full_features = torch.cat([spatial, frequency], dim=1)
    spatial_only = torch.cat([spatial, torch.zeros_like(frequency)], dim=1)
    frequency_only = torch.cat([torch.zeros_like(spatial), frequency], dim=1)

    full_logit = model.classifier(full_features).squeeze(-1)
    spatial_logit = model.classifier(spatial_only).squeeze(-1)
    frequency_logit = model.classifier(frequency_only).squeeze(-1)

    return (
        full_logit,
        spatial_logit,
        frequency_logit,
    )


def analyze(csv_path: Path, image_root: Path, label: str, model, transform, device, threshold: float):
    df = pd.read_csv(csv_path) if csv_path.suffix.lower() == ".csv" else pd.read_excel(csv_path)

    rows = []
    for _, row in df.iterrows():
        pred = int(row["predicted_class"])
        is_error = (label == "REAL" and pred == 1) or (label == "FAKE" and pred == 0)
        if not is_error:
            continue

        path = find_image(image_root, row["image"])
        if path is None:
            continue

        try:
            with Image.open(path) as im:
                image = im.convert("RGB")
            x = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                full_logit, spatial_logit, frequency_logit = branch_scores(model, x)
                full_p = float(sigmoid(full_logit).item())
                spatial_p = float(sigmoid(spatial_logit).item())
                frequency_p = float(sigmoid(frequency_logit).item())

            rows.append({
                "image": str(row["image"]),
                "true_label": label,
                "full_ai_probability": full_p,
                "spatial_only_ai_probability": spatial_p,
                "frequency_only_ai_probability": frequency_p,
                "full_minus_spatial_only": full_p - spatial_p,
                "full_minus_frequency_only": full_p - frequency_p,
                "branch_disagreement": abs(spatial_p - frequency_p),
                "spatial_supports_ai": int(spatial_p >= threshold),
                "frequency_supports_ai": int(frequency_p >= threshold),
                "path": str(path),
            })
        except Exception as exc:
            print(f"[WARN] Failed: {path} -> {exc}")

    return pd.DataFrame(rows)


def summary(df: pd.DataFrame, label: str, threshold: float):
    if df.empty:
        return {"label": label, "count": 0}

    return {
        "label": label,
        "count": int(len(df)),
        "full_probability_mean": float(df.full_ai_probability.mean()),
        "full_probability_median": float(df.full_ai_probability.median()),
        "spatial_only_mean": float(df.spatial_only_ai_probability.mean()),
        "spatial_only_median": float(df.spatial_only_ai_probability.median()),
        "frequency_only_mean": float(df.frequency_only_ai_probability.mean()),
        "frequency_only_median": float(df.frequency_only_ai_probability.median()),
        "mean_branch_disagreement": float(df.branch_disagreement.mean()),
        "both_branches_ai_pct": float(
            100 * ((df.spatial_supports_ai == 1) & (df.frequency_supports_ai == 1)).mean()
        ),
        "spatial_only_ai_pct": float(100 * (df.spatial_supports_ai == 1).mean()),
        "frequency_only_ai_pct": float(100 * (df.frequency_supports_ai == 1).mean()),
        "threshold": threshold,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-csv", type=Path, required=True)
    ap.add_argument("--real-dir", type=Path, required=True)
    ap.add_argument("--fake-csv", type=Path, required=True)
    ap.add_argument("--fake-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/best_model.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/branch_analysis"))
    ap.add_argument("--threshold", type=float, default=0.56)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = make_model()
    load_checkpoint(model, args.checkpoint, device)
    transform = get_val_transforms()

    real_df = analyze(
        args.real_csv, args.real_dir, "REAL",
        model, transform, device, args.threshold
    )
    fake_df = analyze(
        args.fake_csv, args.fake_dir, "FAKE",
        model, transform, device, args.threshold
    )

    all_df = pd.concat([real_df, fake_df], ignore_index=True)
    all_df.to_csv(args.out_dir / "branch_contributions.csv", index=False)

    summaries = [
        summary(real_df, "REAL_false_positives", args.threshold),
        summary(fake_df, "FAKE_false_negatives", args.threshold),
    ]

    # Additional counts useful for deciding the next architecture change.
    if not all_df.empty:
        summaries.append({
            "label": "combined",
            "count": int(len(all_df)),
            "mean_branch_disagreement": float(all_df.branch_disagreement.mean()),
            "median_branch_disagreement": float(all_df.branch_disagreement.median()),
        })

    with open(args.out_dir / "branch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print("\nSignalScope Branch Contribution Analysis")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold:  {args.threshold}")
    print(f"REAL false positives analyzed: {len(real_df)}")
    print(f"FAKE false negatives analyzed: {len(fake_df)}")

    for s in summaries[:2]:
        print(f"\n{s['label']}")
        if s["count"] == 0:
            continue
        print(f"  Full probability median:      {s['full_probability_median']:.4f}")
        print(f"  Spatial-only median:          {s['spatial_only_median']:.4f}")
        print(f"  Frequency-only median:        {s['frequency_only_median']:.4f}")
        print(f"  Mean branch disagreement:     {s['mean_branch_disagreement']:.4f}")
        print(f"  Spatial-only says AI:         {s['spatial_only_ai_pct']:.1f}%")
        print(f"  Frequency-only says AI:       {s['frequency_only_ai_pct']:.1f}%")
        print(f"  Both branches say AI:         {s['both_branches_ai_pct']:.1f}%")

    print("\nSaved:")
    print(args.out_dir / "branch_contributions.csv")
    print(args.out_dir / "branch_summary.json")


if __name__ == "__main__":
    main()
