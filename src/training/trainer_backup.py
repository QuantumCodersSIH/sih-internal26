from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augmentations import (
    get_train_transforms,
    get_val_transforms,
)
from src.data.dataset import SignalScopeDataset
from src.models.forensic_model import SignalScopeForensicModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BATCH_SIZE = 64
NUM_WORKERS = 0

EPOCHS = 6

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

SEED = 42


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def set_seed(seed=42):

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

def create_dataloaders():

    train_dataset = SignalScopeDataset(
        PROJECT_ROOT / "data" / "splits" / "train.csv",
        transform=get_train_transforms(),
    )

    val_dataset = SignalScopeDataset(
        PROJECT_ROOT / "data" / "splits" / "val.csv",
        transform=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return train_loader, val_loader


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    all_labels = []
    all_probabilities = []

    for images, labels in tqdm(
        loader,
        desc="Validation",
        leave=False,
    ):

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        probabilities = torch.sigmoid(logits)

        total_loss += (
            loss.item() * images.size(0)
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

    labels = np.asarray(
        all_labels
    )

    probabilities = np.asarray(
        all_probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    average_loss = (
        total_loss / len(loader.dataset)
    )

    auc = roc_auc_score(
        labels,
        probabilities,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
    )

    cm = confusion_matrix(
        labels,
        predictions,
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)

    return {
        "loss": average_loss,
        "auc": auc,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "fpr": fpr,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print("SignalScope Training")
    print("=" * 70)

    print()
    print("Device:", device)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)

    train_loader, val_loader = (
        create_dataloaders()
    )

    print()
    print(
        f"Training images: "
        f"{len(train_loader.dataset):,}"
    )

    print(
        f"Validation images: "
        f"{len(val_loader.dataset):,}"
    )

    model = SignalScopeForensicModel().to(
        device
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    best_auc = -1.0

    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
        )

        for images, labels in progress:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            fused_logits, spatial_logits, frequency_logits = model(
    images,
    return_aux=True,
)

            fusion_loss = criterion(
                fused_logits,
                labels,
            )

            spatial_loss = criterion(
                spatial_logits,
                labels,
            )

            frequency_loss = criterion(
                frequency_logits,
                labels,
            )

            loss = (
                fusion_loss
                + 0.25 * spatial_loss
                + 0.25 * frequency_loss
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        metrics = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step()

        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"  Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"  Val Loss   : "
            f"{metrics['loss']:.4f}"
        )

        print(
            f"  ROC-AUC    : "
            f"{metrics['auc']:.4f}"
        )

        print(
            f"  Accuracy   : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"  Macro-F1   : "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"  FPR        : "
            f"{metrics['fpr']:.4f}"
        )

        print(
            "  Confusion Matrix:"
        )

        print(
            metrics["confusion_matrix"]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": metrics["loss"],
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "fpr": metrics["fpr"],
            }
        )

        # Save best model based on ROC-AUC.
        if metrics["auc"] > best_auc:

            best_auc = metrics["auc"]

            checkpoint_path = (
                CHECKPOINT_DIR
                / "best_model.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "scheduler_state_dict":
                        scheduler.state_dict(),
                    "best_auc": best_auc,
                    "config": {
                        "batch_size": BATCH_SIZE,
                        "learning_rate":
                            LEARNING_RATE,
                        "weight_decay":
                            WEIGHT_DECAY,
                    },
                },
                checkpoint_path,
            )

            print()
            print(
                f"  ★ Best model saved "
                f"(AUC={best_auc:.4f})"
            )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation ROC-AUC: "
        f"{best_auc:.4f}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_DIR / 'best_model.pt'}"
    )


if __name__ == "__main__":
    train()