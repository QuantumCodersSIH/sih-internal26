from pathlib import Path

import torch
from PIL import Image

from src.data.augmentations import get_val_transforms
from src.evaluation.explanation import generate_explanation
from src.evaluation.gradcam import GradCAM
from src.models.forensic_model import SignalScopeForensicModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "best_model.pt"
)

DEFAULT_THRESHOLD = 0.56
CALIBRATION_TEMPERATURE = 0.95


class SignalScopePredictor:
    def __init__(
        self,
        checkpoint_path=CHECKPOINT_PATH,
        threshold=DEFAULT_THRESHOLD,
        device=None,
    ):
        self.device = torch.device(
            device if device is not None else "cpu"
        )

        self.threshold = threshold

        self.model = SignalScopeForensicModel().to(self.device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.transform = get_val_transforms()

        self.gradcam = GradCAM(self.model)

    def predict(self, image_path):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Model prediction
        with torch.no_grad():
            logit = self.model(image_tensor)
            calibrated_logit = logit / CALIBRATION_TEMPERATURE
            ai_probability = torch.sigmoid(calibrated_logit).item()
        # Classification
        predicted_class = (
            1
            if ai_probability >= self.threshold
            else 0
        )

        # Grad-CAM for predicted class
        heatmap = self.gradcam.generate(
            image_tensor,
            target_class=predicted_class,
        )

        # Human-readable explanation
        explanation = generate_explanation(
            ai_probability=ai_probability,
            heatmap=heatmap,
            threshold=self.threshold,
        )

        return {
            "verdict": explanation.verdict,
            "ai_probability": ai_probability,
            "confidence": explanation.confidence,
            "evidence": explanation.evidence,
            "uncertainty": explanation.uncertainty,
            "predicted_class": predicted_class,
            "heatmap": heatmap,
        }