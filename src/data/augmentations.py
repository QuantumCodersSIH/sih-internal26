import io

import torch
from PIL import Image
from torchvision import transforms


IMAGE_SIZE = 32


class RandomJPEGCompression:
    """
    Apply mild random JPEG compression.

    This simulates common real-world recompression while
    preserving most of the forensic information in the image.
    """

    def __init__(self, quality_range=(70, 95), p=0.25):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, image):
        if torch.rand(1).item() > self.p:
            return image

        quality = int(
            torch.randint(
                self.quality_range[0],
                self.quality_range[1] + 1,
                (1,),
            ).item()
        )

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=quality,
        )

        buffer.seek(0)

        compressed = Image.open(buffer).convert("RGB")

        return compressed


def get_train_transforms():
    """
    Training augmentations.

    Designed to improve robustness against:
    - small geometric changes
    - JPEG recompression
    - mild blur
    - sensor/compression noise
    - small color variations

    We deliberately avoid aggressive transformations because
    forensic artifacts are important for this task.
    """

    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomApply(
            [
                transforms.RandomResizedCrop(
                    size=IMAGE_SIZE,
                    scale=(0.90, 1.0),
                    ratio=(0.95, 1.05),
                )
            ],
            p=0.25,
        ),

        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.08,
                    contrast=0.08,
                    saturation=0.08,
                    hue=0.02,
                )
            ],
            p=0.30,
        ),

        transforms.RandomApply(
            [
                transforms.GaussianBlur(
                    kernel_size=3,
                    sigma=(0.1, 1.0),
                )
            ],
            p=0.15,
        ),

        RandomJPEGCompression(
            quality_range=(70, 95),
            p=0.25,
        ),

        transforms.ToTensor(),

        transforms.RandomApply(
            [
                transforms.Lambda(
                    lambda x: torch.clamp(
                        x + torch.randn_like(x) * 0.01,
                        0.0,
                        1.0,
                    )
                )
            ],
            p=0.15,
        ),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize(
    (IMAGE_SIZE, IMAGE_SIZE),
    interpolation=transforms.InterpolationMode.BILINEAR,
),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomApply(
            [
                transforms.RandomResizedCrop(
                    size=IMAGE_SIZE,
                    scale=(0.90, 1.0),
                    ratio=(0.95, 1.05),
                )
            ],
            p=0.25,
        ),

        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.08,
                    contrast=0.08,
                    saturation=0.08,
                    hue=0.02,
                )
            ],
            p=0.30,
        ),

        transforms.RandomApply(
            [
                transforms.GaussianBlur(
                    kernel_size=3,
                    sigma=(0.1, 1.0),
                )
            ],
            p=0.15,
        ),

        RandomJPEGCompression(
            quality_range=(70, 95),
            p=0.25,
        ),

        transforms.ToTensor(),

        transforms.RandomApply(
            [
                transforms.Lambda(
                    lambda x: torch.clamp(
                        x + torch.randn_like(x) * 0.01,
                        0.0,
                        1.0,
                    )
                )
            ],
            p=0.15,
        ),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])
def get_val_transforms():
    """
    Deterministic validation preprocessing.

    Every image is resized to 32x32.
    No random augmentation is used.
    """

    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])