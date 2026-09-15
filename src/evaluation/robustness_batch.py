from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.inference.predictor import SignalScopePredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "data" / "splits" / "val.csv"

THRESHOLD = 0.56

NUM_REAL = 10
NUM_FAKE = 10


def jpeg_compress(image, quality):
    temp_path = PROJECT_ROOT / "_temp_robustness.jpg"

    image.save(
        temp_path,
        format="JPEG",
        quality=quality,
    )

    compressed = Image.open(temp_path).convert("RGB").copy()

    temp_path.unlink(missing_ok=True)

    return compressed


def resize_then_back(image, size):
    resized = image.resize(
        (size, size),
        Image.Resampling.BILINEAR,
    )

    restored = resized.resize(
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

    predicted_label = (
        1 if probability >= THRESHOLD else 0
    )

    return probability, predicted_label


def main():

    print("=" * 70)
    print("SignalScope Batch Robustness Evaluation")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)

    real_samples = df[df["label"] == 0].head(NUM_REAL)
    fake_samples = df[df["label"] == 1].head(NUM_FAKE)

    samples = pd.concat(
        [real_samples, fake_samples]
    )

    predictor = SignalScopePredictor()

    transformations = [
        "Original",
        "JPEG 70",
        "JPEG 50",
        "32->64->32",
        "32->16->32",
    ]

    total = len(samples)

    correct_counts = {
        name: 0
        for name in transformations
    }

    probability_changes = {
        name: []
        for name in transformations
        if name != "Original"
    }

    for _, row in samples.iterrows():

        image_path = PROJECT_ROOT / row["path"]
        expected_label = int(row["label"])

        original = Image.open(
            image_path
        ).convert("RGB")

        transformed_images = {
            "Original": original,

            "JPEG 70": jpeg_compress(
                original,
                70,
            ),

            "JPEG 50": jpeg_compress(
                original,
                50,
            ),

            "32->64->32": resize_then_back(
                original,
                64,
            ),

            "32->16->32": resize_then_back(
                original,
                16,
            ),
        }

        original_probability = None

        for name, image in transformed_images.items():

            probability, predicted_label = predict(
                predictor,
                image,
            )

            if name == "Original":
                original_probability = probability

            else:
                probability_changes[name].append(
                    abs(
                        probability
                        - original_probability
                    )
                )

            if predicted_label == expected_label:
                correct_counts[name] += 1

    print()
    print("-" * 70)
    print("ROBUSTNESS RESULTS")
    print("-" * 70)

    for name in transformations:

        accuracy = (
            correct_counts[name] / total
        )

        print(
            f"{name:<18} "
            f"Accuracy: {accuracy:.2%} "
            f"({correct_counts[name]}/{total})"
        )

    print()
    print("-" * 70)
    print("AVERAGE ABSOLUTE PROBABILITY CHANGE")
    print("-" * 70)

    for name in probability_changes:

        values = probability_changes[name]

        average_change = sum(values) / len(values)

        print(
            f"{name:<18} "
            f"Average Δ: {average_change:.4f}"
        )

    print()
    print("-" * 70)
    print("PER-CLASS RESULTS")
    print("-" * 70)

    for label, label_name in [
        (0, "REAL"),
        (1, "FAKE"),
    ]:

        class_df = samples[
            samples["label"] == label
        ]

        print()
        print(
            f"{label_name} samples: "
            f"{len(class_df)}"
        )

        for name in transformations:

            correct = 0

            for _, row in class_df.iterrows():

                image_path = (
                    PROJECT_ROOT / row["path"]
                )

                original = Image.open(
                    image_path
                ).convert("RGB")

                if name == "Original":
                    image = original

                elif name == "JPEG 70":
                    image = jpeg_compress(
                        original,
                        70,
                    )

                elif name == "JPEG 50":
                    image = jpeg_compress(
                        original,
                        50,
                    )

                elif name == "32->64->32":
                    image = resize_then_back(
                        original,
                        64,
                    )

                else:
                    image = resize_then_back(
                        original,
                        16,
                    )

                _, prediction = predict(
                    predictor,
                    image,
                )

                if prediction == label:
                    correct += 1

            accuracy = correct / len(class_df)

            print(
                f"  {name:<16} "
                f"{accuracy:.2%}"
            )

    print()
    print("=" * 70)
    print("BATCH ROBUSTNESS TEST COMPLETE")
    print("=" * 70)
    print(f"Total samples: {total}")
    print("Data source: validation split only")
    print("Held-out test set: NOT USED")
    print("=" * 70)


if __name__ == "__main__":
    main()