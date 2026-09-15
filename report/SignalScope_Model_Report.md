# SignalScope — Model Report


## 1. Task

SignalScope is a media-forensics system for detecting whether a general image is likely real or AI-generated.

The system is designed for general synthetic imagery such as scenes, objects, artwork, and product images. It provides a likelihood assessment rather than an accusation or proof of synthetic origin.


---

## 2. Dataset

The original training source contains 100,000 images:

- REAL: 50,000
- FAKE: 50,000
- Image format: JPG
- Image size: 32 × 32 RGB

The original dataset was divided into:

- Training: 80,000 images
- Validation: 20,000 images

An exact-duplicate-aware split was used to prevent identical images from appearing in both training and validation sets.

Duplicate leakage check:

- Duplicate groups detected: 920
- Images belonging to duplicate groups: 1,840
- Duplicate groups overlapping train/validation: 0

For the final training run, four additional datasets were sampled and added to the original training split.

Additional images:

- Data Set 1: 5,000 REAL + 5,000 FAKE
- Data Set 2: 5,000 REAL + 5,000 FAKE
- Data Set 3: 5,000 REAL + 5,000 FAKE
- Data Set 4: 5,000 REAL + 5,000 FAKE

Final training set:

- Training: 120,000 images
- Validation: 20,000 images

The additional datasets were used only for training. The validation split remained separate.

The organizer test set was not used during training, validation, threshold selection, calibration, or robustness experiments.


---

## 3. Preprocessing and Augmentation

All input images are resized to the model input resolution of 32 × 32.

Training augmentation includes:

- Random horizontal flip
- Mild random resized crop
- Mild color jitter
- Gaussian blur
- JPEG recompression
- Small additive image noise

Validation images use deterministic preprocessing without training augmentation.


---

## 4. Model Architecture

SignalScope uses a dual-branch forensic architecture.


### Spatial branch

A residual convolutional neural network learns spatial evidence such as:

- Local texture
- Edges
- Spatial patterns
- Fine image artifacts

The spatial branch produces a 128-dimensional representation.


### Frequency branch

The RGB image is converted to grayscale and transformed using a 2D FFT.

The frequency representation uses:

1. FFT transformation
2. Frequency shifting
3. Magnitude calculation
4. Log scaling
5. Per-image normalization

A convolutional network then learns frequency-domain artifacts.

The frequency branch produces a 128-dimensional representation.


### Fusion

The spatial and frequency features are concatenated:

- Spatial features: 128
- Frequency features: 128
- Combined representation: 256

The classifier uses:

```text
256 → 128 → 1
```

The final output is a logit representing the AI-generated likelihood after sigmoid conversion.


### Auxiliary supervision

The model also contains separate classification heads for the spatial and frequency branches.

Training uses:

```text
Total Loss =
Fusion Loss
+ 0.25 × Spatial Loss
+ 0.25 × Frequency Loss
```

This encourages both branches to learn useful forensic representations while the fused classifier provides the final prediction.


---

## 5. Training Configuration

- Epochs: 6
- Batch size: 64
- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 0.0001
- Learning-rate scheduler: CosineAnnealingLR
- Random seed: 42
- Loss: BCEWithLogitsLoss

The final checkpoint was selected using validation ROC-AUC.


---

## 6. Validation Results

Final validation performance:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.9918 |
| Accuracy | 95.59% |
| Macro-F1 | 95.59% |
| False Positive Rate | 5.18% |

Confusion matrix:

```text
                 Predicted
                 REAL    AI
Actual REAL      9434    515
Actual FAKE       367   9684
```

Best checkpoint:

```text
models/checkpoints/best_model.pt
```


---

## 7. External Generalization Evaluation

A separate 20,000-image evaluation set was used to measure generalization.

The evaluation set contains:

- REAL: 10,000
- FAKE: 10,000

This set was not used for training or threshold tuning.

Final results:

| Metric | Result |
|---|---:|
| Overall Accuracy | 95.405% |
| REAL correctly classified | 9,478 |
| REAL false positives | 522 |
| REAL False Positive Rate | 5.22% |
| FAKE correctly classified | 9,603 |
| FAKE false negatives | 397 |
| FAKE Recall / TPR | 96.03% |
| Macro-F1 | ~95.4% |

Confusion matrix:

```text
                 Predicted
                 REAL    AI
Actual REAL      9478    522
Actual FAKE       397   9603
```


---

## 8. Threshold and Calibration

The deployed inference configuration uses:

- Temperature: 0.95
- Decision threshold: 0.56

The inference pipeline applies temperature scaling to the model logit before sigmoid conversion.

The threshold was selected using validation data and was not tuned using the separate external evaluation set.

The displayed confidence value is treated as decision strength rather than a guarantee of authenticity.


---

## 9. Explainability

SignalScope includes a Grad-CAM-style explanation pipeline.

For each prediction, the system can produce:

- Heatmap
- Heatmap overlay
- Highlighted image regions
- Human-readable evidence
- Uncertainty wording

The explanation is intended to show regions associated with the model's decision. It is not treated as causal proof that a particular artifact caused the classification.


---

## 10. Robustness

A small robustness diagnostic was performed using a validation subset containing 10 REAL and 10 FAKE images.

Tested transformations include:

- Original
- JPEG quality 70
- JPEG quality 50
- 32 → 64 → 32 resizing
- 32 → 16 → 32 resizing

Observed accuracy:

| Condition | Accuracy |
|---|---:|
| Original | 100% |
| JPEG 70 | 100% |
| JPEG 50 | 95% |
| 32 → 64 → 32 | 90% |
| 32 → 16 → 32 | 90% |

This is a small diagnostic rather than a comprehensive robustness benchmark.


---

## 11. Metadata

SignalScope extracts EXIF metadata when available.

Metadata is treated only as supporting evidence.

Important limitations:

- Missing EXIF data is not proof of AI generation.
- Present metadata is not proof of authenticity.
- C2PA / Content Credentials are not implemented in the current version.


---

## 12. Interface

SignalScope includes a Streamlit interface for interactive image analysis.

The interface provides:

- Image upload
- Real / AI-generated prediction
- AI probability
- Decision strength
- Explanation text
- Grad-CAM visualization
- Metadata information


---

## 13. Limitations

The main limitations are:

- New image generators may produce artifacts not represented in the training data.
- The 32 × 32 input resolution limits access to very fine forensic details.
- Strong compression, resizing, screenshots, and editing can alter forensic evidence.
- Grad-CAM visualizations are not causal explanations.
- Model probabilities should not be interpreted as definitive proof of authenticity.
- The 20,000-image external evaluation demonstrates generalization but does not represent every possible generator or image-processing pipeline.


---

## 14. Final Performance Summary

```text
Training images:          120,000
Validation images:         20,000

Validation ROC-AUC:         0.9918
Validation Accuracy:       95.59%
Validation Macro-F1:       95.59%

External evaluation:       20,000
External Accuracy:         95.405%
External Macro-F1:         ~95.4%
External REAL FPR:          5.22%
External AI TPR:           96.03%
```

SignalScope therefore combines spatial and frequency-domain forensic evidence with explainability, robustness testing, metadata inspection, and a deployable interface while maintaining responsible likelihood-based presentation.
