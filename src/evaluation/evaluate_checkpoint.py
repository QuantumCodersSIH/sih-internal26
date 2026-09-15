import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt



from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augmentations import get_val_transforms
from src.data.dataset import SignalScopeDataset
from src.models.forensic_model import SignalScopeForensicModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "best_model.pt"
)

VAL_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "val.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

PLOTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "plots"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BATCH_SIZE = 64

# Selected from validation threshold analysis.
DECISION_THRESHOLD = 0.56


def load_model(device):

    model = SignalScopeForensicModel().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint ROC-AUC: "
        f"{checkpoint['best_auc']:.4f}"
    )

    return model


@torch.no_grad()
def get_predictions(
    model,
    loader,
    device,
):

    all_labels = []
    all_probabilities = []

    for images, labels in tqdm(
        loader,
        desc="Evaluating",
    ):

        images = images.to(device)

        logits = model(images)

        probabilities = torch.sigmoid(
            logits
        )

        all_labels.extend(
            labels.numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

    return (
        np.asarray(all_labels).astype(int),
        np.asarray(all_probabilities),
    )


def calculate_confusion_matrix(labels, predictions):

    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))

    return tn, fp, fn, tp


def calculate_metrics(
    labels,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = calculate_confusion_matrix(
        labels,
        predictions,
    )

    total = tn + fp + fn + tp

    accuracy = (
        (tn + tp) / total
        if total > 0
        else 0.0
    )

    # F1 for REAL class.
    precision_real = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else 0.0
    )

    recall_real = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    f1_real = (
        2 * precision_real * recall_real
        / (precision_real + recall_real)
        if (precision_real + recall_real) > 0
        else 0.0
    )

    # F1 for AI-generated class.
    precision_fake = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall_fake = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1_fake = (
        2 * precision_fake * recall_fake
        / (precision_fake + recall_fake)
        if (precision_fake + recall_fake) > 0
        else 0.0
    )

    macro_f1 = (
        f1_real + f1_fake
    ) / 2

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    tpr = (
        tp / (tp + fn)
        if (tp + fn) > 0
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


def calculate_roc_curve(labels, probabilities):

    thresholds = np.unique(
        probabilities
    )

    thresholds = np.concatenate(
        [
            [np.inf],
            thresholds[::-1],
        ]
    )

    fpr_values = []
    tpr_values = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = calculate_confusion_matrix(
            labels,
            predictions,
        )

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        tpr = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        fpr_values.append(fpr)
        tpr_values.append(tpr)

    fpr_values = np.asarray(fpr_values)
    tpr_values = np.asarray(tpr_values)

    # Sort by false-positive rate.
    order = np.argsort(fpr_values)

    fpr_values = fpr_values[order]
    tpr_values = tpr_values[order]

    auc = np.trapezoid(
        tpr_values,
        fpr_values,
    )

    return (
        fpr_values,
        tpr_values,
        float(auc),
    )


def find_best_threshold(
    labels,
    probabilities,
):

    thresholds = np.arange(
        0.10,
        0.91,
        0.01,
    )

    results = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            labels,
            probabilities,
            threshold,
        )

        results.append(metrics)

    best_f1 = max(
        results,
        key=lambda x: x["macro_f1"],
    )

    best_accuracy = max(
        results,
        key=lambda x: x["accuracy"],
    )

    candidates = [
        result
        for result in results
        if result["fpr"] <= 0.05
    ]

    best_low_fpr = None

    if candidates:

        best_low_fpr = max(
            candidates,
            key=lambda x: x["macro_f1"],
        )

    return (
        best_f1,
        best_accuracy,
        best_low_fpr,
    )
def save_roc_curve(
    fpr,
    tpr,
    auc,
):

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"SignalScope (AUC = {auc:.4f})",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Validation Set")
    plt.legend()
    plt.grid(alpha=0.3)

    output_path = (
        PLOTS_DIR
        / "roc_curve.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"ROC curve saved to: {output_path}"
    )

def save_confusion_matrix(
    labels,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = calculate_confusion_matrix(
    labels,
    predictions,
)

    cm = np.array([
        [tn, fp],
        [fn, tp],
    ])

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title(
        f"Confusion Matrix — Threshold {threshold:.2f}"
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.xticks(
        [0, 1],
        ["Real", "AI-generated"],
    )

    plt.yticks(
        [0, 1],
        ["Real", "AI-generated"],
    )

    for row in range(2):
        for col in range(2):

            plt.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                fontsize=14,
            )

    plt.colorbar()

    output_path = (
        PLOTS_DIR
        / "confusion_matrix.png"
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()

    print(
        f"Confusion matrix saved to: "
        f"{output_path}"
    )


def main():

    print()
    print("=" * 70)
    print("SignalScope Checkpoint Evaluation")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Validation:", VAL_CSV)

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

    model = load_model(device)

    labels, probabilities = get_predictions(
        model,
        loader,
        device,
    )

    roc_fpr, roc_tpr, auc = calculate_roc_curve(
    labels,
    probabilities,
    )

    print()
    print("=" * 70)
    print("ROC-AUC")
    print("=" * 70)

    print(
        f"Validation ROC-AUC: {auc:.4f}"
    )

    # Save ROC curve.
    save_roc_curve(
        roc_fpr,
        roc_tpr,
        auc,
    )

    print()
    print("=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    threshold_results = []

    for threshold in [
        0.30,
        0.40,
        0.45,
        0.50,
        0.55,
        0.56,
        0.60,
        0.70,
    ]:

        metrics = calculate_metrics(
            labels,
            probabilities,
            threshold,
        )

        threshold_results.append(
            metrics
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"Accuracy {metrics['accuracy']:.4f} | "
            f"Macro-F1 {metrics['macro_f1']:.4f} | "
            f"FPR {metrics['fpr']:.4f} | "
            f"TPR {metrics['tpr']:.4f}"
        )

    (
        best_f1,
        best_accuracy,
        best_low_fpr,
    ) = find_best_threshold(
        labels,
        probabilities,
    )

    print()
    print("=" * 70)
    print("BEST THRESHOLDS")
    print("=" * 70)

    print(
        "Best Macro-F1:",
        best_f1,
    )

    print(
        "Best Accuracy:",
        best_accuracy,
    )

    if best_low_fpr is not None:

        print(
            "Best Macro-F1 with FPR <= 5%:",
            best_low_fpr,
        )

    selected_metrics = calculate_metrics(
        labels,
        probabilities,
        DECISION_THRESHOLD,
    )

    print()
    print("=" * 70)
    print(
        f"CONFUSION MATRIX @ "
        f"{DECISION_THRESHOLD:.2f}"
    )
    print("=" * 70)

    print(
        np.array(
            [
                [
                    selected_metrics["tn"],
                    selected_metrics["fp"],
                ],
                [
                    selected_metrics["fn"],
                    selected_metrics["tp"],
                ],
            ]
        )
    )

    # Save confusion matrix plot.
    save_confusion_matrix(
        labels,
        probabilities,
        DECISION_THRESHOLD,
    )

    results = {
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "validation_samples": int(
            len(labels)
        ),
        "roc_auc": float(auc),
        "decision_threshold": DECISION_THRESHOLD,
        "thresholds": threshold_results,
        "best_macro_f1": best_f1,
        "best_accuracy": best_accuracy,
        "best_low_fpr": best_low_fpr,
        "default_threshold": selected_metrics,
    }

    output_path = (
        RESULTS_DIR
        / "checkpoint_evaluation.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print()
    print(
        f"Results saved to: "
        f"{output_path}"
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()