"""
SignalScope prediction interface.

This is the single function everything else in the project calls.
Right now it returns a PLACEHOLDER result so the UI can be built and
tested independently, before the real model is trained.

Once the real model is ready:
1. Save your trained weights to model/signalscope_model.h5
2. Save your fitted temperature-scaling value to model/temperature.npy
3. Replace the body of predict() below with the real loading + inference
   code (see MODEL_TODO.md in this folder for the exact plan).

The function signature (input/output shape) below should NOT change,
so nothing calling this function needs to change when the real model
is plugged in.
"""

import random


def predict(image_path: str) -> dict:
    """
    Args:
        image_path: path to an image file on disk.

    Returns:
        A dict with exactly these keys:
            "label": "real" or "fake"
            "confidence": float between 0.0 and 1.0
    """
    # --- PLACEHOLDER LOGIC — replace this whole block with the real model ---
    label = random.choice(["real", "fake"])
    confidence = round(random.uniform(0.55, 0.98), 2)
    # -------------------------------------------------------------------

    return {"label": label, "confidence": confidence}


if __name__ == "__main__":
    # Quick manual test: python predict.py path/to/image.jpg
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        result = predict(sys.argv[1])
        print(result)
