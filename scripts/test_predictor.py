from pathlib import Path

from src.inference.predictor import SignalScopePredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REAL_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "train"
    / "REAL"
    / "0000 (10).jpg"
)

FAKE_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "train"
    / "FAKE"
    / "1000 (10).jpg"
)


def print_result(name, image_path, result):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Image: {image_path.name}")
    print(f"Verdict: {result['verdict']}")
    print(
        f"AI probability: "
        f"{result['ai_probability']:.4f}"
    )
    print(
        f"Confidence: "
        f"{result['confidence']:.2%}"
    )
    print()
    print("Evidence:")
    print(result["evidence"])
    print()
    print("Uncertainty:")
    print(result["uncertainty"])
    print()
    print(
        "Heatmap range: "
        f"{result['heatmap'].min():.4f} - "
        f"{result['heatmap'].max():.4f}"
    )


def main():
    print("=" * 70)
    print("SignalScope Integrated Predictor Test")
    print("=" * 70)

    predictor = SignalScopePredictor()

    real_result = predictor.predict(REAL_IMAGE)

    fake_result = predictor.predict(FAKE_IMAGE)

    print_result(
        "REAL IMAGE RESULT",
        REAL_IMAGE,
        real_result,
    )

    print_result(
        "FAKE IMAGE RESULT",
        FAKE_IMAGE,
        fake_result,
    )

    print()
    print("=" * 70)
    print("INTEGRATED PREDICTOR TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()