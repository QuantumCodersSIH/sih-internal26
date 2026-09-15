from pathlib import Path
import argparse
import json
import sys

# Make project root importable when running:
# python scripts/judge_predict.py <image>
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import SignalScopePredictor


def main():
    parser = argparse.ArgumentParser(
        description="SignalScope official-style prediction interface"
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to the image to classify"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.56,
        help="Decision threshold for AI-generated classification"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        raise SystemExit(1)

    predictor = SignalScopePredictor(
        threshold=args.threshold
    )

    result = predictor.predict(image_path)

    output = {
        "image": str(image_path),
        "verdict": result["verdict"],
        "ai_probability": float(result["ai_probability"]),
        "decision_strength": float(result["confidence"]),
        "predicted_class": int(result["predicted_class"]),
        "threshold": float(args.threshold),
    }

    print(json.dumps(output, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(output, file, indent=2)

        print(f"\nSaved prediction to: {output_path}")


if __name__ == "__main__":
    main()