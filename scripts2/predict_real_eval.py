from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentations import get_val_transforms
from src.models.forensic_model import SignalScopeForensicModel


CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "best_model.pt"
IMAGE_DIR = PROJECT_ROOT / "TEMP_TEST" / "REAL"
OUTPUT = PROJECT_ROOT / "TEMP_TEST_predictions_REAL.csv"

THRESHOLD = 0.56
TEMPERATURE = 0.95


class EvalDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        with Image.open(path) as image:
            image = image.convert("RGB")

        image = self.transform(image)
        return image, path.name


def load_model(device):
    model = SignalScopeForensicModel().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = sorted(IMAGE_DIR.rglob("*.jpg"))

    if len(image_paths) != 10000:
        raise RuntimeError(
            f"Expected 10,000 REAL images, found {len(image_paths)}"
        )

    transform = get_val_transforms()

    dataset = EvalDataset(image_paths, transform)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    model = load_model(device)

    rows = []

    with torch.no_grad():
        for images, names in loader:
            images = images.to(device)

            logits = model(images).squeeze(-1)

            calibrated_logits = logits / TEMPERATURE
            probabilities = torch.sigmoid(calibrated_logits)

            probabilities = probabilities.cpu().numpy()

            predictions = (probabilities >= THRESHOLD).astype(int)

            for name, probability, prediction in zip(
                names,
                probabilities,
                predictions,
            ):
                probability = float(probability)
                prediction = int(prediction)

                decision_strength = (
                    probability
                    if prediction == 1
                    else 1.0 - probability
                )

                verdict = (
                    "Likely AI-generated"
                    if prediction == 1
                    else "Likely real"
                )

                rows.append(
                    {
                        "image": f"TEMP_TEST\\{name}",
                        "ai_probability": probability,
                        "decision_strength": decision_strength,
                        "predicted_class": prediction,
                        "verdict": verdict,
                    }
                )

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT, index=False)

    print()
    print("SignalScope REAL Evaluation Predictions")
    print("---------------------------------------")
    print("Checkpoint:", CHECKPOINT)
    print("Images:", len(df))
    print("Threshold:", THRESHOLD)
    print("Temperature:", TEMPERATURE)
    print()
    print("Predicted classes:")
    print(df["predicted_class"].value_counts().sort_index())
    print()
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()