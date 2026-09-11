# Team Tasks — Things That Don't Need to Wait on the Model

These can all be worked on right now, in parallel with dataset setup and
model training, since none of them depend on the real model being ready.

## 1. Polish the Streamlit UI (`app/app.py`)
It currently works end-to-end against a placeholder prediction — style it,
improve the layout/spacing, add a clearer loading state, and make the
real/fake result display more visually clear (e.g. a colored badge or a
simple confidence bar). Do not change the shape of what `predict()`
returns — the app should keep working unchanged once the real model is
plugged in.

## 2. Fill in the model report template (`report/model_report_template.md`)
Everything except the actual numbers can be written now: task description,
architecture overview (pixel branch + frequency branch + fusion,
per the plan), dataset sources and licenses, and a first draft of the
limitations section. Leave the metrics table (AUC, F1, confusion matrix)
blank — those get filled in once training is done.

## 3. NEW — Curate a diverse outside test-image set
Collect 20–30 images that were NOT part of our training dataset: a mix of
genuine real photos (take some yourself, or use ones you know the source
of) and AI-generated images (from a different generator than what we
trained on if possible — e.g. Midjourney or DALL-E output, not just
Stable Diffusion). Save them in the `test_images/` folder using the
naming convention `real_01.jpg`, `real_02.jpg`, `fake_01.jpg`, etc.
This set is what we'll use to stress-test the model's generalization
and the predict() interface's robustness before submission — see
`test_images/README.md` for details.

## 4. NEW — Draft the demo video script/storyboard
Write a rough script for the required 3–5 minute demo video: what order
to show things in (intro → upload a real image → upload a fake image →
explain the metrics briefly → mention limitations honestly), roughly how
long each part should take, and what should be visible on screen at each
point. Having this ready means we're not improvising the video the night
before the deadline.
