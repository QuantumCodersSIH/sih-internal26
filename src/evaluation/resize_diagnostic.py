from pathlib import Path

import torch
from PIL import Image

from src.inference.predictor import SignalScopePredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "train"
    / "FAKE"
    / "1000 (10).jpg"
)

THRESHOLD = 0.56


def resize_then_back(image, size):
    enlarged = image.resize(
        (size, size),
        Image.Resampling.BILINEAR,
    )

    restored = enlarged.resize(
        (32, 32),
        Image.Resampling.BILINEAR,
    )

    return restored


def predict(predictor, image):
    tensor = predictor.transform(image)
    tensor = tensor.unsqueeze(0).to(predictor.device)

    with torch.no_grad():
        logit = predictor.model(tensor)
        probability = torch.sigmoid(logit).item()

    prediction = (
        "AI-generated"
        if probability >= THRESHOLD
        else "Real"
    )

    return probability, prediction


def main():
    print("=" * 70)
    print("SignalScope Resize Sensitivity Diagnostic")
    print("=" * 70)

    predictor = SignalScopePredictor()

    original = Image.open(IMAGE_PATH).convert("RGB")

    tests = {
        "Original 32x32": original,
        "32 -> 64 -> 32": resize_then_back(original, 64),
        "32 -> 16 -> 32": resize_then_back(original, 16),
        "32 -> 128 -> 32": resize_then_back(original, 128),
        "32 -> 256 -> 32": resize_then_back(original, 256),
    }

    original_probability = None

    print()
    print(f"Image: {IMAGE_PATH.name}")
    print("Expected: FAKE")
    print()

    for name, image in tests.items():

        probability, prediction = predict(
            predictor,
            image,
        )

        if original_probability is None:
            original_probability = probability

        delta = probability - original_probability

        print(
            f"{name:<22} "
            f"AI probability: {probability:.4f}  "
            f"Prediction: {prediction:<13} "
            f"Δ: {delta:+.4f}"
        )

    print()
    print("=" * 70)
    print("RESIZE DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()