from pathlib import Path
from collections import Counter
import hashlib
import json

from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# IMPORTANT:
# Only TRAIN is used.
# The organizer's TEST set is intentionally untouched.
DATA_DIR = PROJECT_ROOT / "data" / "train"
REAL_DIR = DATA_DIR / "REAL"
FAKE_DIR = DATA_DIR / "FAKE"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = RESULTS_DIR / "dataset_audit.json"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_image_files(directory: Path):
    if not directory.exists():
        return []

    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )


def calculate_md5(path: Path):
    md5 = hashlib.md5()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            md5.update(chunk)

    return md5.hexdigest()


def inspect_image(path: Path):
    try:
        # First verify the file.
        with Image.open(path) as image:
            image.verify()

        # Reopen because verify() invalidates the image object.
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            format_name = image.format

        return {
            "valid": True,
            "width": width,
            "height": height,
            "mode": mode,
            "format": format_name,
        }

    except Exception as error:
        return {
            "valid": False,
            "error": str(error),
        }


def analyze_class(class_name: str, directory: Path):

    files = get_image_files(directory)

    print()
    print("=" * 70)
    print(f"Analyzing: {class_name}")
    print("=" * 70)
    print(f"Directory : {directory}")
    print(f"Images    : {len(files):,}")

    dimension_counter = Counter()
    mode_counter = Counter()
    format_counter = Counter()

    corrupted_files = []
    hashes = {}

    for path in tqdm(
        files,
        desc=f"{class_name} audit",
    ):

        info = inspect_image(path)

        if not info["valid"]:

            corrupted_files.append(
                {
                    "file": str(path.relative_to(PROJECT_ROOT)),
                    "error": info["error"],
                }
            )

            continue

        dimensions = (
            info["width"],
            info["height"],
        )

        dimension_counter[dimensions] += 1
        mode_counter[info["mode"]] += 1
        format_counter[info["format"]] += 1

        file_hash = calculate_md5(path)

        if file_hash not in hashes:
            hashes[file_hash] = []

        hashes[file_hash].append(
            str(path.relative_to(PROJECT_ROOT))
        )

    duplicate_groups = {
        file_hash: files
        for file_hash, files in hashes.items()
        if len(files) > 1
    }

    print()
    print(f"Valid images        : {len(files) - len(corrupted_files):,}")
    print(f"Corrupted images    : {len(corrupted_files):,}")

    print()
    print("Image dimensions:")

    for dimensions, count in dimension_counter.most_common():
        print(
            f"  {dimensions[0]}x{dimensions[1]} : {count:,}"
        )

    print()
    print("Color modes:")

    for mode, count in mode_counter.most_common():
        print(
            f"  {mode} : {count:,}"
        )

    print()
    print("Formats:")

    for format_name, count in format_counter.most_common():
        print(
            f"  {format_name} : {count:,}"
        )

    print()
    print(
        f"Exact duplicate groups: "
        f"{len(duplicate_groups):,}"
    )

    return {
        "class": class_name,
        "directory": str(directory),
        "total_files": len(files),
        "valid_files": len(files) - len(corrupted_files),
        "corrupted_files": corrupted_files,
        "dimensions": {
            f"{width}x{height}": count
            for (width, height), count
            in dimension_counter.items()
        },
        "color_modes": dict(mode_counter),
        "formats": dict(format_counter),
        "exact_duplicate_groups": duplicate_groups,
    }


def main():

    print()
    print("=" * 70)
    print("SignalScope Dataset Audit")
    print("=" * 70)
    print()

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset      : {DATA_DIR}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_DIR}"
        )

    if not REAL_DIR.exists():
        raise FileNotFoundError(
            f"REAL directory not found: {REAL_DIR}"
        )

    if not FAKE_DIR.exists():
        raise FileNotFoundError(
            f"FAKE directory not found: {FAKE_DIR}"
        )

    real_report = analyze_class(
        "REAL",
        REAL_DIR,
    )

    fake_report = analyze_class(
        "FAKE",
        FAKE_DIR,
    )

    real_count = real_report["total_files"]
    fake_count = fake_report["total_files"]

    total_count = real_count + fake_count

    if total_count > 0:

        real_percentage = (
            real_count / total_count
        ) * 100

        fake_percentage = (
            fake_count / total_count
        ) * 100

    else:

        real_percentage = 0
        fake_percentage = 0

    report = {

        "dataset": "SignalScope training dataset",

        "data_directory": str(DATA_DIR),

        "test_directory_used": False,

        "classes": {
            "REAL": real_report,
            "FAKE": fake_report,
        },

        "summary": {

            "total_images": total_count,

            "real_images": real_count,

            "fake_images": fake_count,

            "real_percentage": real_percentage,

            "fake_percentage": fake_percentage,
        },
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print()
    print("=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)

    print(
        f"REAL images : {real_count:,}"
    )

    print(
        f"FAKE images : {fake_count:,}"
    )

    print(
        f"TOTAL       : {total_count:,}"
    )

    print()

    print(
        f"REAL ratio  : {real_percentage:.2f}%"
    )

    print(
        f"FAKE ratio  : {fake_percentage:.2f}%"
    )

    print()

    print("Audit report saved to:")

    print(REPORT_PATH)

    print()
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print("Only data/train was analyzed.")
    print("The test directory was NOT used.")

    print("=" * 70)


if __name__ == "__main__":
    main()