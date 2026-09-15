from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.models.forensic_model import SignalScopeForensicModel
from src.data.augmentations import get_val_transforms
from src.evaluation.gradcam import GradCAM, make_overlay


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "splits" / "val.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "best_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "results" / "plots" / "gradcam_batch"

NUM_REAL = 5
NUM_FAKE = 5
THRESHOLD = 0.56


def main():
    device = torch.device("cpu")

    print("=" * 70)
    print("SignalScope Batch Grad-CAM Validation")
    print("=" * 70)

    # Load validation split only
    df = pd.read_csv(CSV_PATH)

    real_samples = df[df["label"] == 0].head(NUM_REAL)
    fake_samples = df[df["label"] == 1].head(NUM_FAKE)

    samples = pd.concat([real_samples, fake_samples])

    # Load model
    model = SignalScopeForensicModel().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = get_val_transforms()
    cam = GradCAM(model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    correct = 0

    for index, row in samples.iterrows():
        image_path = PROJECT_ROOT / row["path"]
        expected_label = int(row["label"])

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(image_tensor)
            probability = torch.sigmoid(logit).item()

        predicted_label = 1 if probability >= THRESHOLD else 0

        # Explain the predicted class
        heatmap = cam.generate(
            image_tensor,
            target_class=predicted_label,
        )

        overlay = make_overlay(image, heatmap)

        expected_name = "FAKE" if expected_label == 1 else "REAL"
        predicted_name = "AI-generated" if predicted_label == 1 else "Real"

        is_correct = expected_label == predicted_label

        if is_correct:
            correct += 1

        output_name = (
            f"{'correct' if is_correct else 'wrong'}_"
            f"{expected_name.lower()}_"
            f"{index}.jpg"
        )

        output_path = OUTPUT_DIR / output_name
        overlay.save(output_path, quality=95)

        print()
        print(f"Image: {image_path.name}")
        print(f"Expected: {expected_name}")
        print(f"AI probability: {probability:.4f}")
        print(f"Prediction: {predicted_name}")
        print(f"Correct: {'YES' if is_correct else 'NO'}")
        print(
            f"Heatmap range: "
            f"{heatmap.min():.4f} - {heatmap.max():.4f}"
        )
        print(f"Saved: {output_path}")

    print()
    print("=" * 70)
    print("BATCH GRAD-CAM TEST COMPLETE")
    print("=" * 70)
    print(f"Samples tested: {len(samples)}")
    print(f"Correct predictions: {correct}/{len(samples)}")
    print(f"Accuracy: {correct / len(samples):.2%}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()