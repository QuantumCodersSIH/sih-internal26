# Model TODO

This folder currently has a PLACEHOLDER predict() function so the rest
of the app can be built and tested without waiting on the trained model.

## What needs to happen here later (owner: whoever trains the model)

- [ ] Train the CNN + frequency-fusion model (see the build guide)
- [ ] Save trained weights as `signalscope_model.h5` in this folder
- [ ] Save the fitted temperature-scaling value as `temperature.npy` in this folder
- [ ] Replace the placeholder logic inside `predict.py` with real
      preprocessing + inference + calibration, keeping the same
      function signature: `predict(image_path: str) -> dict`
- [ ] Test predict() directly (`python predict.py some_image.jpg`)
      on at least 15-20 real images before telling the team it's ready

## What the rest of the team can build in the meantime

- Anything importing `from model.predict import predict` already works
  today against the placeholder — no changes needed on their end once
  the real model is dropped in, since the input/output shape stays
  identical.
