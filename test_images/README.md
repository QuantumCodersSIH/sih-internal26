# Test Images

Put the curated outside test-image set here (see TEAM_TASKS.md item 3).

Naming convention:
- `real_01.jpg`, `real_02.jpg`, ... — genuine photos, not from our training set
- `fake_01.jpg`, `fake_02.jpg`, ... — AI-generated images, ideally from a
  generator not used in training (e.g. Midjourney/DALL-E, not just the
  Stable Diffusion images we trained on)

Purpose: these images are used to manually stress-test predict() and
check generalization before submission — they are NOT training data and
should never be added to the training/validation folders.
