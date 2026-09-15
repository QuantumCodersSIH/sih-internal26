from pathlib import Path
import hashlib
import csv
import random

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "train"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

REAL_DIR = DATA_DIR / "REAL"
FAKE_DIR = DATA_DIR / "FAKE"

TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"

SEED = 42
VAL_RATIO = 0.20


def calculate_md5(path: Path) -> str:
    """Return the MD5 hash of an image file."""

    md5 = hashlib.md5()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            md5.update(chunk)

    return md5.hexdigest()


def get_images(directory: Path):
    """Return all supported image files."""

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
    )


def collect_records():

    records = []

    for label, directory in [
        (0, REAL_DIR),
        (1, FAKE_DIR),
    ]:

        print(f"Scanning {directory} ...")

        for path in get_images(directory):

            # Verify the image is readable.
            with Image.open(path) as image:
                image.verify()

            file_hash = calculate_md5(path)

            records.append(
                {
                    "path": str(
                        path.relative_to(PROJECT_ROOT)
                    ),
                    "label": label,
                    "hash": file_hash,
                }
            )

    return records


def create_groups(records):

    groups = {}

    for record in records:

        group_id = (
            record["label"],
            record["hash"],
        )

        if group_id not in groups:
            groups[group_id] = []

        groups[group_id].append(record)

    return list(groups.values())


def split_groups(groups):

    random.seed(SEED)

    # Shuffle groups, NOT individual images.
    random.shuffle(groups)

    total_images = sum(
        len(group)
        for group in groups
    )

    target_val_images = int(
        total_images * VAL_RATIO
    )

    train_groups = []
    val_groups = []

    val_images = 0

    for group in groups:

        # Keep an entire duplicate group together.
        if (
            val_images < target_val_images
            and val_images + len(group)
            <= target_val_images
        ):

            val_groups.append(group)
            val_images += len(group)

        else:

            train_groups.append(group)

    return train_groups, val_groups


def flatten(groups):

    records = []

    for group in groups:
        records.extend(group)

    return records


def write_csv(path: Path, records):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "path",
                "label",
                "hash",
            ],
        )

        writer.writeheader()

        writer.writerows(records)


def verify_split(train_records, val_records):

    train_hashes = {
        (record["label"], record["hash"])
        for record in train_records
    }

    val_hashes = {
        (record["label"], record["hash"])
        for record in val_records
    }

    overlap = train_hashes.intersection(
        val_hashes
    )

    if overlap:

        raise RuntimeError(
            f"DATA LEAKAGE DETECTED: "
            f"{len(overlap)} duplicate groups "
            f"appear in both train and validation."
        )

    print()
    print("=" * 70)
    print("LEAKAGE CHECK")
    print("=" * 70)
    print("Train/validation duplicate overlap : 0")
    print("Status                              : PASS")


def print_summary(train_records, val_records):

    train_real = sum(
        record["label"] == 0
        for record in train_records
    )

    train_fake = sum(
        record["label"] == 1
        for record in train_records
    )

    val_real = sum(
        record["label"] == 0
        for record in val_records
    )

    val_fake = sum(
        record["label"] == 1
        for record in val_records
    )

    print()
    print("=" * 70)
    print("FINAL SPLIT")
    print("=" * 70)

    print()
    print("TRAIN")
    print(f"  REAL : {train_real:,}")
    print(f"  FAKE : {train_fake:,}")
    print(f"  TOTAL: {len(train_records):,}")

    print()
    print("VALIDATION")
    print(f"  REAL : {val_real:,}")
    print(f"  FAKE : {val_fake:,}")
    print(f"  TOTAL: {len(val_records):,}")

    print()
    total_records = len(train_records) + len(val_records)

    print(
        f"Train ratio: "
        f"{len(train_records) / total_records:.2%}"
    )

    print(
        f"Validation ratio: "
        f"{len(val_records) / total_records:.2%}"
    )


def main():

    print()
    print("=" * 70)
    print("SignalScope Leakage-Safe Dataset Split")
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print("Only data/train is being used.")
    print("Organizer test data is NOT accessed.")

    records = collect_records()

    print()
    print(f"Total images collected: {len(records):,}")

    groups = create_groups(records)

    print(
        f"Unique image groups: {len(groups):,}"
    )

    duplicate_groups = [
        group
        for group in groups
        if len(group) > 1
    ]

    duplicate_images = sum(
        len(group)
        for group in duplicate_groups
    )

    print(
        f"Duplicate groups: "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"Images belonging to duplicate groups: "
        f"{duplicate_images:,}"
    )

    train_groups, val_groups = split_groups(
        groups
    )

    train_records = flatten(
        train_groups
    )

    val_records = flatten(
        val_groups
    )

    verify_split(
        train_records,
        val_records,
    )

    print_summary(
        train_records,
        val_records,
    )

    write_csv(
        TRAIN_CSV,
        train_records,
    )

    write_csv(
        VAL_CSV,
        val_records,
    )

    print()
    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(TRAIN_CSV)
    print(VAL_CSV)


if __name__ == "__main__":
    main()