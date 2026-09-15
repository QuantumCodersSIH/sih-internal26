from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GradCAM:
    """
    Grad-CAM for the spatial CNN branch.

    We use an earlier convolutional block so that the heatmap
    retains more spatial detail at the native 32x32 resolution.
    """

    def __init__(self, model, target_layer=None):
        self.model = model

        # block3 operates at a higher spatial resolution than block5.
        if target_layer is None:
            target_layer = model.spatial.block3.conv2

        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, target_class=1):
        """
        Generate a normalized Grad-CAM heatmap.

        target_class:
            1 -> AI-generated evidence
            0 -> real-image evidence
        """

        self.model.zero_grad(set_to_none=True)

        output = self.model(image_tensor)

        # Model output is a logit.
        if target_class == 1:
            score = output.sum()
        else:
            score = (-output).sum()

        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM activations/gradients were not captured.")

        activations = self.activations
        gradients = self.gradients

        # Global-average-pool gradients over spatial dimensions.
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps.
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # Only positive evidence.
        cam = F.relu(cam)

        # Resize to original model input resolution: 32x32.
        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()

        # Normalize.
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return np.clip(cam, 0.0, 1.0)

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def _make_heatmap_rgb(heatmap):
    """
    Convert scalar heatmap into an RGB forensic heatmap.
    """

    heatmap = np.clip(heatmap, 0.0, 1.0)

    # Smooth blue -> cyan -> yellow -> red style mapping.
    red = np.clip(2.0 * heatmap, 0.0, 1.0)
    green = np.clip(2.0 * (1.0 - np.abs(heatmap - 0.5)), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - heatmap), 0.0, 1.0)

    return np.stack([red, green, blue], axis=-1)


def make_heatmap_image(heatmap, size=None):
    """
    Create a standalone heatmap showing only the strongest
    model activation regions.
    """

    heatmap = np.asarray(heatmap).astype(np.float32)
    heatmap = np.clip(heatmap, 0.0, 1.0)

    # Keep only the strongest 25% of activations.
    threshold = np.percentile(heatmap, 75)

    focused = np.zeros_like(heatmap)

    mask = heatmap >= threshold

    focused[mask] = heatmap[mask]

    # Re-normalize the selected evidence.
    if mask.any():
        minimum = focused[mask].min()
        maximum = focused[mask].max()

        if maximum - minimum > 1e-8:
            focused[mask] = (
                focused[mask] - minimum
            ) / (maximum - minimum)

    heatmap_rgb = _make_heatmap_rgb(focused)

    # Make non-evidence areas neutral instead of cyan.
    background = np.ones_like(heatmap_rgb) * 0.92

    mask_rgb = mask[..., None]

    result = np.where(
        mask_rgb,
        heatmap_rgb,
        background,
    )

    image = Image.fromarray(
        (result * 255).astype(np.uint8)
    )

    if size is not None:
        image = image.resize(
            size,
            Image.Resampling.BILINEAR
        )

    return image

def make_overlay(original_image, heatmap, alpha=0.45):
    """
    Overlay a Grad-CAM heatmap on the original image.

    The heatmap is resized automatically to the original image
    dimensions, so images of any resolution can be visualized.
    """

    import numpy as np
    from PIL import Image

    # Make sure original image is RGB
    original_image = original_image.convert("RGB")

    # Convert original image to numpy array
    original_array = np.asarray(original_image).astype(np.float32) / 255.0

    # Convert heatmap to numpy array
    heatmap_array = np.asarray(heatmap).astype(np.float32)

    # Remove unnecessary channel dimension if present
    if heatmap_array.ndim == 3:
        heatmap_array = np.squeeze(heatmap_array)

    # Ensure heatmap is in [0, 1]
    heatmap_array = np.clip(heatmap_array, 0.0, 1.0)

    # Resize heatmap to EXACTLY match original image size
    heatmap_pil = Image.fromarray(
        np.uint8(heatmap_array * 255.0)
    )

    heatmap_pil = heatmap_pil.resize(
        original_image.size,
        Image.Resampling.BILINEAR,
    )

    heatmap_array = np.asarray(heatmap_pil).astype(np.float32) / 255.0

    # Create RGB heatmap using a simple red intensity map
    heatmap_rgb = np.zeros(
        (heatmap_array.shape[0], heatmap_array.shape[1], 3),
        dtype=np.float32,
    )

    heatmap_rgb[:, :, 0] = heatmap_array
    heatmap_rgb[:, :, 1] = heatmap_array * 0.25

    # Blend original image and heatmap
    overlay = (
        original_array * (1.0 - alpha)
        + heatmap_rgb * alpha
    )

    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)

def make_masked_overlay(original_image, heatmap):
    """
    Alternative visualization that only highlights
    the strongest model evidence.
    """

    original = original_image.convert("RGB")

    original_array = (
        np.asarray(original).astype(np.float32) / 255.0
    )

    heatmap = np.asarray(heatmap).astype(np.float32)
    heatmap = np.clip(heatmap, 0.0, 1.0)

    threshold = np.percentile(heatmap, 75)

    mask = heatmap >= threshold

    heatmap_rgb = _make_heatmap_rgb(heatmap)

    result = original_array.copy()

    result[mask] = (
        0.45 * original_array[mask]
        + 0.55 * heatmap_rgb[mask]
    )

    result = np.clip(
        result * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(result)
def _make_heatmap_rgb(heatmap):
    heatmap = np.clip(heatmap, 0.0, 1.0)

    red = np.clip(2.0 * heatmap, 0.0, 1.0)
    green = np.clip(
        2.0 * (1.0 - np.abs(heatmap - 0.5)),
        0.0,
        1.0,
    )
    blue = np.clip(2.0 * (1.0 - heatmap), 0.0, 1.0)

    return np.stack([red, green, blue], axis=-1)


def make_heatmap_image(heatmap, size=None):
    """
    Create a standalone RGB heatmap image.
    """

    heatmap_rgb = _make_heatmap_rgb(heatmap)

    image = Image.fromarray(
        (heatmap_rgb * 255).astype(np.uint8)
    )

    if size is not None:
        image = image.resize(
            size,
            Image.Resampling.BILINEAR
        )

    return image