import torch
from torchvision import transforms


IMAGE_SIZE = 32


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

    No random augmentation is used here.
    """

    return transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])