from pathlib import Path

import torch
from PIL import Image

from src.data.augmentations import get_val_transforms
from src.evaluation.gradcam import GradCAM
from src.evaluation.explanation import generate_explanation
from src.models.forensic_model import SignalScopeForensicModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "best_model.pt"

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "train"
    / "FAKE"
    / "1000 (10).jpg"
)

THRESHOLD = 0.56


def main():
    device = torch.device("cpu")

    model = SignalScopeForensicModel().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(IMAGE_PATH).convert("RGB")

    transform = get_val_transforms()
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(image_tensor)
        probability = torch.sigmoid(logit).item()

    predicted_class = 1 if probability >= THRESHOLD else 0

    cam = GradCAM(model)
    heatmap = cam.generate(
        image_tensor,
        target_class=predicted_class,
    )

    explanation = generate_explanation(
        ai_probability=probability,
        heatmap=heatmap,
        threshold=THRESHOLD,
    )

    print("=" * 70)
    print("SignalScope Forensic Explanation")
    print("=" * 70)
    print()
    print(f"Verdict: {explanation.verdict}")
    print(f"AI probability: {probability:.4f}")
    print(f"Confidence: {explanation.confidence:.2%}")
    print()
    print("Evidence:")
    print(explanation.evidence)
    print()
    print("Uncertainty:")
    print(explanation.uncertainty)
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()