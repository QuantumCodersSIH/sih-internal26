from pathlib import Path
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import SignalScopeDataset
from src.data.augmentations import get_val_transforms
from src.models.forensic_model import SignalScopeForensicModel


CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "best_model.pt"
VAL_CSV = PROJECT_ROOT / "data" / "splits" / "val.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "metrics" / "calibration.json"

BATCH_SIZE = 128
TEMPERATURE = 0.95
FPR_LIMIT = 0.05


def collect_logits():
    dataset = SignalScopeDataset(
        VAL_CSV,
        transform=get_val_transforms(),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cpu")

    model = SignalScopeForensicModel().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logits = []
    labels = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)

            batch_logits = model(images).view(-1)

            logits.extend(batch_logits.cpu().numpy())
            labels.extend(targets.numpy())

    return np.asarray(logits), np.asarray(labels)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def brier_score(labels, probabilities):
    return float(np.mean((probabilities - labels) ** 2))


def calculate_metrics(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int32)

    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))

    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total else 0.0

    precision_real = (
        tn / (tn + fn)
        if (tn + fn)
        else 0.0
    )

    recall_real = (
        tn / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    f1_real = (
        2 * precision_real * recall_real /
        (precision_real + recall_real)
        if (precision_real + recall_real)
        else 0.0
    )

    precision_fake = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall_fake = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    f1_fake = (
        2 * precision_fake * recall_fake /
        (precision_fake + recall_fake)
        if (precision_fake + recall_fake)
        else 0.0
    )

    macro_f1 = (f1_real + f1_fake) / 2

    fpr = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    tpr = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

def calculate_roc_auc(labels, probabilities):
    order = np.argsort(probabilities)
    sorted_labels = labels[order]

    positives = np.sum(sorted_labels == 1)
    negatives = np.sum(sorted_labels == 0)

    if positives == 0 or negatives == 0:
        return 0.0

    rank_sum = np.sum(
        np.where(sorted_labels == 1)[0] + 1
    )

    auc = (
        rank_sum
        - positives * (positives + 1) / 2
    ) / (positives * negatives)

    return float(auc)

def main():
    print("=" * 70)
    print("SignalScope calibrated threshold selection")
    print("=" * 70)

    print(f"Validation file: {VAL_CSV}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Temperature: {TEMPERATURE}")

    logits, labels = collect_logits()

    raw_probabilities = sigmoid(logits)
    calibrated_probabilities = sigmoid(logits / TEMPERATURE)
    roc_auc = calculate_roc_auc(
    labels,
    calibrated_probabilities,
)
    raw_brier = brier_score(
    labels,
    raw_probabilities,
)

    calibrated_brier = brier_score(
        labels,
        calibrated_probabilities,
    )

    results = []

    for threshold in np.arange(0.10, 0.901, 0.01):
        result = calculate_metrics(
        labels,
        calibrated_probabilities,
        threshold,
    )

        results.append(result)

    # Best macro-F1 while respecting FPR <= 5%.
    eligible = [
        result
        for result in results
        if result["fpr"] <= FPR_LIMIT
    ]

    if eligible:
        best = max(
            eligible,
            key=lambda result: (
                result["macro_f1"],
                result["accuracy"],
            ),
        )
    else:
        best = max(
            results,
            key=lambda result: result["macro_f1"],
        )

    print("\nCalibrated validation results")
    print("-" * 70)
    print(f"ROC-AUC:       {roc_auc:.6f}")
    print(f"Raw Brier:      {raw_brier:.6f}")
    print(f"Calibrated Brier: {calibrated_brier:.6f}")
    print(f"Threshold:     {best['threshold']:.2f}")
    print(f"Accuracy:      {best['accuracy']:.6f}")
    print(f"Macro-F1:      {best['macro_f1']:.6f}")
    print(f"FPR:           {best['fpr']:.6f}")
    print(f"TPR:           {best['tpr']:.6f}")

    print("\nConfusion matrix")
    print(
        f"TN={best['tn']}  FP={best['fp']}  "
        f"FN={best['fn']}  TP={best['tp']}"
    )

    output = {
        "checkpoint": str(
            CHECKPOINT_PATH.relative_to(PROJECT_ROOT)
        ),
        "validation_csv": str(
            VAL_CSV.relative_to(PROJECT_ROOT)
        ),
        "temperature": TEMPERATURE,
        "roc_auc": float(roc_auc),
        "threshold_selection": {
            "method": "macro_f1_with_fpr_at_most_0.05",
            "probability_type": "temperature_scaled",
            "threshold": best["threshold"],
        },
        "metrics": best,
        "num_validation_samples": int(len(labels)),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(f"\nSaved: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()