import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        if (
            stride != 1
            or in_channels != out_channels
        ):
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + identity

        x = F.relu(x, inplace=True)

        return x


class SpatialBranch(nn.Module):
    """
    Learns spatial forensic features such as:
    - local texture
    - edges
    - pixel relationships
    - structural inconsistencies
    """

    def __init__(self):

        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.block1 = ResidualBlock(
            32,
            32,
        )

        self.block2 = ResidualBlock(
            32,
            64,
            stride=2,
        )

        self.block3 = ResidualBlock(
            64,
            64,
        )

        self.block4 = ResidualBlock(
            64,
            128,
            stride=2,
        )

        self.block5 = ResidualBlock(
            128,
            128,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        x = self.stem(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.pool(x)

        return torch.flatten(x, 1)


class FrequencyBranch(nn.Module):
    """
    Extracts frequency-domain forensic features.

    The FFT magnitude captures spectral patterns that can
    differ between natural images and generated images.
    """

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):

        # Convert RGB to grayscale.
        grayscale = (
            0.299 * x[:, 0:1]
            + 0.587 * x[:, 1:2]
            + 0.114 * x[:, 2:3]
        )

        # 2D Fourier transform.
        fft = torch.fft.fft2(
            grayscale,
            norm="ortho",
        )

        # Shift low frequencies to the center.
        fft = torch.fft.fftshift(fft)

        # Magnitude spectrum.
        magnitude = torch.abs(fft)

        # Log scaling improves numerical stability.
        magnitude = torch.log1p(magnitude)

        # Normalize each image independently.
        mean = magnitude.mean(
            dim=(-2, -1),
            keepdim=True,
        )

        std = magnitude.std(
            dim=(-2, -1),
            keepdim=True,
        ).clamp_min(1e-6)

        magnitude = (
            magnitude - mean
        ) / std

        x = self.network(magnitude)

        return torch.flatten(x, 1)


class SignalScopeForensicModel(nn.Module):
    """
    SignalScope multimodal forensic detector.

    Output:
        logits representing P(AI-generated)
    """

    def __init__(self):

        super().__init__()

        self.spatial = SpatialBranch()

        self.frequency = FrequencyBranch()

        self.classifier = nn.Sequential(

            nn.Linear(
                128 + 128,
                128,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.30),

            nn.Linear(
                128,
                1,
            ),
        )
        self.spatial_head = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.20),
        nn.Linear(64, 1),
    )

        self.frequency_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, x, return_aux=False):
        spatial = self.spatial(x)
        frequency = self.frequency(x)

        features = torch.cat([spatial, frequency], dim=1)

        fused_logit = self.classifier(features).squeeze(-1)

        if not return_aux:
            return fused_logit

        spatial_logit = self.spatial_head(spatial).squeeze(-1)
        frequency_logit = self.frequency_head(frequency).squeeze(-1)

        return (
            fused_logit,
            spatial_logit,
            frequency_logit,
        )