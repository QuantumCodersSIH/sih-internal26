from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SignalScopeDataset(Dataset):
    """
    Dataset for SignalScope real-vs-synthetic image classification.

    CSV format:
        path,label,hash
    """

    def __init__(
        self,
        csv_file,
        transform=None,
    ):
        self.csv_file = Path(csv_file)
        self.transform = transform

        self.data = pd.read_csv(self.csv_file)

        if "path" not in self.data.columns:
            raise ValueError("CSV must contain a 'path' column.")

        if "label" not in self.data.columns:
            raise ValueError("CSV must contain a 'label' column.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = PROJECT_ROOT / row["path"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            raise RuntimeError(
                f"Failed to load image: {image_path}"
            ) from error

        label = torch.tensor(
            float(row["label"]),
            dtype=torch.float32,
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, label