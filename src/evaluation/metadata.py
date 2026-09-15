from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS


def extract_exif(image_path):
    """
    Extract readable EXIF metadata from an image.

    Returns a dictionary of metadata fields.
    Missing EXIF data is represented by an empty dictionary.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as image:
        exif_data = image.getexif()

        metadata = {}

        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, str(tag_id))

            try:
                metadata[tag_name] = str(value)
            except Exception:
                metadata[tag_name] = "<unreadable>"

        return metadata