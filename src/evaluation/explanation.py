from dataclasses import dataclass


@dataclass
class ForensicExplanation:
    verdict: str
    confidence: float
    evidence: str
    uncertainty: str


def generate_explanation(
    ai_probability: float,
    heatmap,
    threshold: float = 0.56,
) -> ForensicExplanation:

    probability = float(ai_probability)

    # Decision strength reflects how strongly the model score favors
# the predicted class. It is not a calibrated probability.
    confidence = (
        probability if probability >= threshold
        else 1.0 - probability
    )

    heatmap_strength = float(heatmap.mean())

    if probability >= threshold:
        verdict = "Likely AI-generated"

        if heatmap_strength >= 0.20:
            evidence = (
                "The model's strongest evidence is concentrated in "
                "localized visual regions. These regions contain the "
                "patterns the detector associated with synthetic imagery."
            )
        else:
            evidence = (
                "The detector found evidence associated with synthetic "
                "imagery, although the visual evidence is relatively "
                "diffuse across the image."
            )

        if probability >= 0.90:
            uncertainty = (
                "The model has high confidence in this likelihood "
                "assessment, but the result is not proof of synthetic origin."
            )
        elif probability >= 0.70:
            uncertainty = (
                "The model has moderate-to-high confidence. "
                "Treat this as a likelihood assessment rather than proof."
            )
        else:
            uncertainty = (
                "The prediction is close enough to the decision boundary "
                "that additional verification is recommended."
            )

    else:
        verdict = "Likely real"

        if heatmap_strength >= 0.20:
            evidence = (
                "The model's strongest evidence is concentrated in "
                "localized visual regions that support the real-image "
                "classification."
            )
        else:
            evidence = (
                "The model does not find strong evidence associated with "
                "synthetic imagery across the image."
            )

        if probability <= 0.10:
            uncertainty = (
                "The model has high confidence in this likelihood "
                "assessment, but this does not establish authenticity."
            )
        elif probability <= 0.30:
            uncertainty = (
                "The model has moderate-to-high confidence that the image "
                "is not synthetic according to its learned patterns."
            )
        else:
            uncertainty = (
                "The prediction is relatively close to the decision "
                "boundary, so additional verification is recommended."
            )

    return ForensicExplanation(
        verdict=verdict,
        confidence=confidence,
        evidence=evidence,
        uncertainty=uncertainty,
    )