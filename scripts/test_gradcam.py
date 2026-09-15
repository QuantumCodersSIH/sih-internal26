from pathlib import Path

from src.evaluation.gradcam import explain_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data" / "train"

OUTPUT_DIR = PROJECT_ROOT / "results" / "plots" / "gradcam"


def find_first_image(folder):
    for image_path in folder.rglob("*.jpg"):
        return image_path

    raise FileNotFoundError(
        f"No JPG images found in {folder}"
    )


def main():
    print("=" * 70)
    print("SignalScope Grad-CAM Test")
    print("=" * 70)

    # Use a REAL image from training data only.
    real_image = find_first_image(
        IMAGE_DIR / "REAL"
    )

    # Use a FAKE image from training data only.
    fake_image = find_first_image(
        IMAGE_DIR / "FAKE"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tests = [
        ("REAL", real_image),
        ("FAKE", fake_image),
    ]

    for label, image_path in tests:

        output_path = (
            OUTPUT_DIR
            / f"gradcam_{label.lower()}.jpg"
        )

        print(f"\nInput: {image_path}")
        print(f"Expected label: {label}")

        probability, heatmap = explain_image(
            image_path,
            output_path,
        )

        prediction = (
            "AI-generated"
            if probability >= 0.56
            else "Real"
        )

        print(
            f"AI probability: {probability:.4f}"
        )

        print(
            f"Prediction: {prediction}"
        )

        print(
            f"Heatmap range: "
            f"{heatmap.min():.4f} - "
            f"{heatmap.max():.4f}"
        )

        print(
            f"Heatmap saved: {output_path}"
        )

    print("\n" + "=" * 70)
    print("GRAD-CAM TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()