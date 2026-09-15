from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.data.augmentations import get_val_transforms
from src.inference.predictor import SignalScopePredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

THRESHOLD = 0.56


def jpeg_compress(image, quality):
    output = Path("temp_robustness.jpg")

    image.save(
        output,
        format="JPEG",
        quality=quality,
    )

    compressed = Image.open(output).convert("RGB")

    output.unlink()

    return compressed


def resize_image(image, size):
    return image.resize(
        (size, size),
        Image.Resampling.BILINEAR,
    )


def center_crop_resize(image):
    width, height = image.size

    crop_size = min(width, height)

    left = (width - crop_size) // 2
    top = (height - crop_size) // 2

    cropped = image.crop(
        (
            left,
            top,
            left + crop_size,
            top + crop_size,
        )
    )

    return cropped.resize(
        (32, 32),
        Image.Resampling.BILINEAR,
    )


def predict_image(predictor, image):
    image_tensor = predictor.transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(
        predictor.device
    )

    with torch.no_grad():
        logit = predictor.model(image_tensor)
        probability = torch.sigmoid(logit).item()

    prediction = (
        "AI-generated"
        if probability >= THRESHOLD
        else "Real"
    )

    return probability, prediction


def test_image(predictor, image_path, expected_label):
    original = Image.open(image_path).convert("RGB")

    transformations = {
        "Original": original,

        "JPEG quality 90": jpeg_compress(
            original,
            90,
        ),

        "JPEG quality 70": jpeg_compress(
            original,
            70,
        ),

        "JPEG quality 50": jpeg_compress(
            original,
            50,
        ),

        "Resize 64px": resize_image(
            original,
            64,
        ),

        "Resize 16px": resize_image(
            original,
            16,
        ),

        "Center crop": center_crop_resize(
            original,
        ),
    }

    print()
    print("=" * 70)
    print(f"Image: {image_path.name}")
    print(f"Expected: {expected_label}")
    print("=" * 70)

    original_probability = None

    for name, image in transformations.items():

        probability, prediction = predict_image(
            predictor,
            image,
        )

        if original_probability is None:
            original_probability = probability

        change = (
            probability - original_probability
        )

        print(
            f"{name:<20} "
            f"AI probability: {probability:.4f}  "
            f"Prediction: {prediction}  "
            f"Δ: {change:+.4f}"
        )


def main():

    print("=" * 70)
    print("SignalScope Robustness Test")
    print("=" * 70)

    predictor = SignalScopePredictor()

    real_image = (
        PROJECT_ROOT
        / "data"
        / "train"
        / "REAL"
        / "0000 (10).jpg"
    )

    fake_image = (
        PROJECT_ROOT
        / "data"
        / "train"
        / "FAKE"
        / "1000 (10).jpg"
    )

    test_image(
        predictor,
        real_image,
        "REAL",
    )

    test_image(
        predictor,
        fake_image,
        "FAKE",
    )

    print()
    print("=" * 70)
    print("ROBUSTNESS TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()