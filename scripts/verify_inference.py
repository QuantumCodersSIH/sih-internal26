from pathlib import Path

import torch
from PIL import Image

from src.data.augmentations import get_val_transforms
from src.models.forensic_model import SignalScopeForensicModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "best_model.pt"
)


def predict_image(model, image_path, transform):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        logit = model(image_tensor)
        probability = torch.sigmoid(logit).item()

    return logit.item(), probability


def main():

    print("=" * 60)
    print("SignalScope Inference Verification")
    print("=" * 60)

    # --------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------

    print("\nCheckpoint:")
    print(CHECKPOINT_PATH)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("Checkpoint exists: PASS")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    model = SignalScopeForensicModel()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("\nModel loaded: PASS")

    # --------------------------------------------------
    # Parameter count
    # --------------------------------------------------

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Model parameters: {parameter_count:,}"
    )

    # --------------------------------------------------
    # Find test images ONLY from train
    # --------------------------------------------------

    real_dir = (
        PROJECT_ROOT
        / "data"
        / "train"
        / "REAL"
    )

    fake_dir = (
        PROJECT_ROOT
        / "data"
        / "train"
        / "FAKE"
    )

    real_images = sorted(
        real_dir.glob("*.jpg")
    )

    fake_images = sorted(
        fake_dir.glob("*.jpg")
    )

    if not real_images:
        raise RuntimeError(
            "No REAL training images found."
        )

    if not fake_images:
        raise RuntimeError(
            "No FAKE training images found."
        )

    real_image = real_images[0]
    fake_image = fake_images[0]

    # --------------------------------------------------
    # Transform
    # --------------------------------------------------

    transform = get_val_transforms()

    # --------------------------------------------------
    # REAL
    # --------------------------------------------------

    real_logit, real_probability = predict_image(
        model,
        real_image,
        transform,
    )

    print("\n" + "-" * 60)
    print("REAL training image")
    print("-" * 60)

    print(
        f"Image: {real_image.name}"
    )

    print(
        f"Raw logit: {real_logit:.6f}"
    )

    print(
        f"AI probability: {real_probability:.6%}"
    )

    print(
        "Prediction:",
        "Likely AI-generated"
        if real_probability >= 0.56
        else "Likely real",
    )

    # --------------------------------------------------
    # FAKE
    # --------------------------------------------------

    fake_logit, fake_probability = predict_image(
        model,
        fake_image,
        transform,
    )

    print("\n" + "-" * 60)
    print("FAKE training image")
    print("-" * 60)

    print(
        f"Image: {fake_image.name}"
    )

    print(
        f"Raw logit: {fake_logit:.6f}"
    )

    print(
        f"AI probability: {fake_probability:.6%}"
    )

    print(
        "Prediction:",
        "Likely AI-generated"
        if fake_probability >= 0.56
        else "Likely real",
    )

    # --------------------------------------------------
    # Final verification
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    if real_probability != fake_probability:
        print(
            "PASS: Different images produce different predictions."
        )
    else:
        print(
            "WARNING: Predictions are identical."
        )

    print(
        "\nNo data/test images were used."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()