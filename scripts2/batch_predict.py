from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import SignalScopePredictor


TEST_DIR = PROJECT_ROOT / "TEMP_TEST"
OUTPUT_FILE = PROJECT_ROOT / "TEMP_TEST_predictions.csv"


def main():
    if not TEST_DIR.exists():
        print(f"ERROR: Folder not found: {TEST_DIR}")
        return

    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    images = sorted(
        p for p in TEST_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not images:
        print("No images found.")
        return

    print(f"Found {len(images)} images.")
    print("Loading model...")

    predictor = SignalScopePredictor()

    results = []

    for index, image_path in enumerate(images, start=1):
        try:
            result = predictor.predict(image_path)

            results.append({
                "image": str(image_path.relative_to(PROJECT_ROOT)),
                "ai_probability": result["ai_probability"],
                "decision_strength": result["confidence"],
                "predicted_class": result["predicted_class"],
                "verdict": result["verdict"],
            })

            print(
                f"[{index}/{len(images)}] "
                f"{image_path.name} -> "
                f"{result['verdict']} "
                f"({result['ai_probability']:.4f})"
            )

        except Exception as error:
            print(f"[ERROR] {image_path}: {error}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "ai_probability",
                "decision_strength",
                "predicted_class",
                "verdict",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Completed: {len(results)}/{len(images)}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()