# SignalScope

**Telling Real From Synthetic in the Age of Generative Media**

SignalScope is a computer-vision forensic screening tool that classifies a single image as **Likely Real** or **Likely AI-generated** and explains the decision using model-derived visual evidence. The system is designed for responsible forensic triage: it reports likelihoods rather than claiming definitive authenticity or authorship.

> **Hackathon focus:** strong real-vs-AI classification, generalization, faithful explanation, robustness/failure analysis, metadata inspection, and a reproducible Streamlit demo.

---

## 1. What SignalScope Builds

### Core task — implemented

- Single-image real-vs-AI-generated classification
- AI likelihood / decision strength
- Calibrated production inference
- Held-out validation metrics
- Separate external generalization evaluation
- Streamlit prediction interface

### Bonus modules

| Module | Status | What is included |
|---|---|---|
| **A — Faithful Explanation** | **Implemented** | Class-specific Grad-CAM-style evidence, heatmap, overlay, localization diagnostics, deterministic grounded explanation, uncertainty wording |
| **B — Generator Attribution** | Not implemented | No unsupported generator-family prediction is claimed |
| **C — Robustness to Degradation** | **Implemented as diagnostic** | JPEG, resizing, blur, contrast, brightness, and noise attack/flip analysis |
| **D — Provenance & Metadata** | **Partial** | EXIF extraction and responsible interpretation; C2PA/Content Credentials not implemented |
| **E — Multimodal Image + Text** | Not implemented | No caption/claim classifier is claimed |
| **F — Real-Time / Deployable** | **Implemented** | Interactive Streamlit application with drag-and-drop upload workflow |
| **G — Active Defence Analysis** | **Implemented as diagnostic** | Fixed-model post-processing attack study with failure analysis and human-review guidance |

The hackathon brief requires the core classifier, held-out evaluation, confusion matrix, and a reproducible prediction interface. It also explicitly scores faithful explanation, robustness, deployment, and honest failure analysis as optional bonus areas. 

---

## 2. Quick Start

The goal is that a judge can reproduce one prediction in under approximately 10 minutes.

### Requirements

- Python 3.10+ recommended
- Windows / Linux / macOS
- CPU is supported; GPU is optional
- The repository must contain the trained checkpoint at:

```text
models/checkpoints/best_model.pt
```

### Install

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the app

From the repository root:

```bash
python -m streamlit run app.py
```

Open the local Streamlit URL shown in the terminal, normally:

```text
http://localhost:8501
```

### Make a prediction

1. Upload a JPG, JPEG, PNG, or WebP image.
2. SignalScope displays:
   - verdict
   - AI probability
   - decision strength
   - model-evidence explanation
   - original image
   - artifact heatmap
   - evidence overlay
   - localization diagnostics
   - EXIF metadata when available
   - global Active Defence / robustness results
3. The same uploaded image can be tested repeatedly without changing the model.

### Important

The production classifier is **not retrained at inference time**.

---

## 3. Dataset and Data Provenance

### Core dataset

The core training/evaluation data is the **dataset provided for the SignalScope hackathon**.

The original project split contains:

- 100,000 original images
- 50,000 REAL
- 50,000 AI-generated / FAKE
- 80,000 training images
- 20,000 validation images
- 32 × 32 RGB input resolution

### Expanded training set

For the final expanded training run, the project added **40,000 additional images** sampled from four additional datasets supplied in the local `archive (2)` collection:

- 5,000 REAL + 5,000 AI-generated from each of four datasets
- 40,000 additional training images total
- Final training set: **120,000 images**
- Validation set remained separate at **20,000 images**

The added public datasets should be listed with their **exact original dataset names, source URLs, and licenses** before public submission. Do not invent a license if the archive documentation does not explicitly provide one.

### Data separation

The final workflow keeps the validation set and the separate external evaluation set outside training and threshold selection.

The external 20,000-image evaluation described below was **not used for training, threshold selection, or calibration**.

### Dataset/licence checklist before GitHub submission

- [ ] Record the exact name of every added public dataset
- [ ] Record its original source URL
- [ ] Record its license
- [ ] Confirm redistribution is allowed
- [ ] Keep the hackathon-provided dataset identified separately from added public data

The hackathon rules explicitly require citing public datasets and libraries in the README.

---

## 4. Model Architecture

SignalScope uses a **dual-branch spatial-frequency image forensic detector**.

```text
                     Input Image
                          │
                ┌─────────┴─────────┐
                │                   │
         Spatial branch       Frequency branch
                │                   │
        Residual CNN          RGB → grayscale
                │              FFT2 → shift
          128-d feature        magnitude/log
                │              normalization
                │             frequency CNN
                │              128-d feature
                └─────────┬─────────┘
                          │
                    Concatenate
                       256-d
                          │
                    Fusion MLP
                    256 → 128 → 1
                          │
                    AI logit
                          │
               temperature = 0.95
                          │
                    AI probability
                          │
               threshold = 0.56
                          │
              Likely Real / Likely AI
```

### Spatial branch

A lightweight CNN operates on the image in the spatial domain and extracts local visual representations.

### Frequency branch

The image is converted to grayscale and transformed using a 2-D FFT. The frequency magnitude is log-scaled and normalized before being processed by a CNN.

### Fusion

The spatial and frequency representations are concatenated:

```text
128 spatial + 128 frequency = 256
```

and passed through the final fusion classifier.

### Auxiliary training heads

The model also contains spatial and frequency auxiliary heads. They are used during training to encourage useful branch representations, but the deployed verdict comes from the **fused classifier**.

### Training

Final training configuration:

- Training set: 120,000 images
- Validation set: 20,000 images
- Seed: 42
- Batch size: 64
- Epochs: 6
- Optimizer: AdamW
- Learning rate: 1e-3
- Weight decay: 1e-4
- Scheduler: cosine annealing
- Loss: binary cross-entropy with auxiliary spatial/frequency losses
- Best checkpoint selected by validation ROC-AUC

Training/validation preprocessing uses lightweight forensic-preserving augmentation including horizontal flips, small crops/resizing, mild color changes, blur, JPEG compression, and small Gaussian noise.

---

## 5. Calibration and Production Decision Rule

Production inference uses fixed calibration settings:

```text
Temperature = 0.95
Decision threshold = 0.56
```

The deployed probability calculation is:

```text
calibrated_logit = logit / 0.95
AI probability = sigmoid(calibrated_logit)
```

Decision rule:

```text
AI probability >= 0.56  →  Likely AI-generated
AI probability < 0.56   →  Likely real
```

The displayed **decision strength** is the strength of the binary decision, not a guarantee of authenticity.

The threshold was not selected using the separate 20,000-image external evaluation set.

---

## 6. Verified Evaluation Results

### Local validation set

Final model:

| Metric | Result |
|---|---:|
| Images | 20,000 |
| ROC-AUC | **0.9918** |
| Accuracy | **95.59%** |
| Macro-F1 | **95.59%** |
| REAL FPR | **5.18%** |

Confusion matrix:

```text
                  Predicted
                  REAL      AI
Actual REAL      9434     515
Actual AI        367      9684
```

### External 20,000-image generalization evaluation

A separate 20,000-image set was kept outside training and production threshold/calibration selection.

```text
10,000 REAL
10,000 AI
20,000 total
```

Results:

| Metric | Result |
|---|---:|
| Accuracy | **95.405%** |
| Macro-F1 | **~95.4%** |
| REAL FPR | **5.22%** |
| AI TPR | **96.03%** |

Confusion matrix:

```text
                  Predicted
                  REAL      AI
Actual REAL      9478     522
Actual AI        397      9603
```

### Important unseen-generator reporting note

The hackathon submission contract specifically asks for:

- overall held-out AUC
- **unseen-generator-split AUC**
- macro-F1
- confusion matrix

The project currently has a verified local validation ROC-AUC of **0.9918** and a separate 20,000-image external generalization accuracy/F1 evaluation, but the **official organizer unseen-generator-split AUC is not available in the verified project results used to build this README**.

Therefore, this README deliberately does **not** invent or relabel another metric as the official unseen-generator AUC.

> **Before final submission, replace the field below with the organizer's official held-out unseen-generator AUC if/when the result is provided:**
>
> `Official unseen-generator-split ROC-AUC: TBD`

Do not claim the separate external 20,000-image diagnostic is the official unseen-generator test unless the organizers confirm that it is.

---

## 7. Confusion Matrix Summary

For the verified 20,000-image validation set:

```text
                    Predicted REAL    Predicted AI
Actual REAL              9434             515
Actual AI                 367            9684
```

This corresponds to:

- True negatives: 9,434
- False positives: 515
- False negatives: 367
- True positives: 9,684

The separate 20,000-image external evaluation is documented above and is intentionally kept separate from the validation results.

---

## 8. Bonus A — Faithful Explanation

SignalScope's explanation system is designed around the actual trained model output rather than generic image descriptions.

For each prediction it provides:

- class-specific Grad-CAM-style activation
- original image
- artifact heatmap
- evidence overlay
- localized evidence region when activation is genuinely concentrated
- heatmap mean
- maximum activation
- high-activation area
- concentration ratio
- deterministic evidence text
- uncertainty wording

The explanation is derived from the existing trained model's actual gradients/activations and final prediction. It does **not** call an external LLM or invent semantic artifacts.

The system intentionally avoids claims such as:

```text
"The fingers are incorrect."
"The reflection is physically impossible."
"The lighting proves this is AI."
```

unless such a semantic conclusion is directly supported by a measurable model signal.

The UI instead uses cautious evidence language such as:

```text
Strong activation is concentrated in localized regions.
The highlighted regions contributed strongly to the decision.
The strongest evidence is localized rather than uniformly distributed.
```

This follows the hackathon's requirement that Bonus A be judged on **faithfulness and usefulness, not fluent wording**.

---

## 9. Robustness and Active Defence

### Active Defence diagnostic

A fixed 200-image diagnostic was run using:

- 100 REAL images
- 100 AI-generated images
- temperature 0.95
- threshold 0.56
- no retraining
- no threshold retuning

Prediction-flip results:

| Attack | Overall flip | REAL flip | AI flip |
|---|---:|---:|---:|
| Resize 8 roundtrip | **34.0%** | 21.0% | **47.0%** |
| Resize 16 roundtrip | 16.5% | 9.0% | 24.0% |
| Gaussian blur | 6.0% | 5.0% | 7.0% |
| JPEG 50 | 6.0% | 6.0% | 6.0% |
| Contrast high | 5.5% | 9.0% | 2.0% |
| Contrast low | 5.0% | 2.0% | 8.0% |
| Gaussian noise | 4.0% | 2.0% | 6.0% |
| Brightness low | 4.0% | 5.0% | 3.0% |
| JPEG 70 | 2.0% | 3.0% | 1.0% |
| Brightness high | 1.5% | 0.0% | 3.0% |

### Main failure mode

The detector is most sensitive to **aggressive resizing**, especially for AI images.

SignalScope therefore recommends additional human review for heavily resized or strongly degraded images instead of presenting the result as definitive.

---

## 10. Metadata / Provenance

SignalScope extracts EXIF metadata when it exists.

Metadata is treated as **supporting forensic information only**.

Important:

- Missing EXIF is not proof of AI generation.
- Present EXIF is not proof of authenticity.
- EXIF can be stripped or modified.
- C2PA / Content Credentials are **not implemented** in the current version.

---

## 11. Deployment

SignalScope is packaged as a Streamlit application.

The current application supports:

- image upload
- real-vs-AI classification
- AI probability
- decision strength
- faithful explanation
- Grad-CAM visualization
- localization diagnostics
- EXIF inspection
- global robustness / Active Defence results
- responsible-use messaging

The application is intended for rapid forensic screening, not autonomous authenticity adjudication.

---

## 12. Known Limitations

### Small input resolution

The trained detector operates at **32 × 32** input resolution. This keeps inference lightweight but can discard fine forensic details.

### Generator drift

New image generators may create artifact patterns that were not represented in training.

### Post-processing sensitivity

Strong resizing, screenshots, compression, filtering, and other transformations can modify the forensic evidence. Aggressive resizing is a demonstrated weakness.

### Explanation limitations

Grad-CAM-style maps show regions associated with the model's decision. They are **not causal proof** that a particular human-interpretable artifact caused the classification.

### Probability interpretation

The displayed AI probability is model-derived and should not be interpreted as certainty.

### External evaluation scope

The 20,000-image external evaluation demonstrates generalization but does not establish performance against every generator, image source, or processing pipeline.

### Active Defence scope

The Active Defence experiment uses a controlled 200-image diagnostic sample. It is a failure-analysis tool, not a comprehensive adversarial benchmark.

---

## 13. Reproducibility Checklist

A fresh judge environment should be able to verify:

- [ ] Python environment created
- [ ] `requirements.txt` installed
- [ ] checkpoint present
- [ ] Streamlit app starts
- [ ] new image can be uploaded
- [ ] prediction is returned
- [ ] AI probability is shown
- [ ] Grad-CAM / faithful explanation renders
- [ ] metadata section renders
- [ ] Active Defence section renders
- [ ] no training is required to run inference

For code/evaluation changes, keep the production settings unchanged unless a documented experiment is intended:

```text
temperature = 0.95
threshold   = 0.56
```

---

## 14. Repository Structure

Recommended repository layout:

```text
SignalScope/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── checkpoints/
│       └── best_model.pt
├── src/
│   ├── models/
│   ├── inference/
│   ├── evaluation/
│   ├── training/
│   ├── data/
│   └── defense/
├── data/
│   ├── splits/
│   └── ...
├── results/
│   └── active_defense/
│       ├── attack_results.csv
│       ├── attack_summary.json
│       └── active_defense_report.md
└── report/
    └── model_report.md
```

Do not commit large/private datasets or large binary checkpoints when the submission rules provide a release/link mechanism for them; follow the hackathon's repository-size and data-distribution requirements.

---

## 15. Demo Video and Deployment Links

### Demo video

**3–5 minute demo:**  
[Demo Video](https://drive.google.com/drive/folders/1yxJgMDmgMh1C6y68mGFRaxkY1Lho5nOu?usp=sharing)

The demo should show:

1. SignalScope starting from a clean environment.
2. A new image being uploaded.
3. Real / AI-generated prediction.
4. AI probability and decision strength.
5. Faithful explanation.
6. Heatmap + localized evidence.
7. Metadata section.
8. Active Defence / robustness results.
9. Responsible-use limitation.

### Deployed application

`[ADD DEPLOYED APP URL IF AVAILABLE]`

### GitHub Repository

[GitHub Repository](https://github.com/QuantumCodersSIH/sih-internal26)

If there is no public deployment, leave this as:

```text
Public deployment: Not deployed; reproducible locally with Streamlit.
```

---

## 16. Originality and Third-Party Components

SignalScope's detector and explanation workflow were developed specifically for this hackathon project.

Open-source libraries are used for standard engineering functionality such as:

- PyTorch
- torchvision
- NumPy
- Pillow
- Streamlit
- related Python utilities

Any additional third-party model, dataset, code, notebook, or implementation should be explicitly cited here before public submission.

**Important:** do not claim an external dataset license or third-party implementation without verifying its original documentation.

---

## 17. Responsible Use

SignalScope is a **forensic screening and decision-support system**.

A prediction should not be treated as definitive proof of:

- authenticity
- synthetic origin
- authorship
- intent
- provenance

For high-stakes decisions, combine the model with source verification, provenance information, contextual investigation, and human review.

The interface therefore uses language such as **“Likely real”** and **“Likely AI-generated”** rather than certainty claims.

---

## 18. Submission Readiness

Before submitting the repository, verify these six required README items from the hackathon contract:

1. **Core + bonus modules:** documented above.
2. **Setup/run:** reproducible with the commands above.
3. **Datasets + licences:** exact added dataset names, URLs, and licences still need to be filled from the archive documentation.
4. **Metrics:** validation and external results are documented; the official unseen-generator AUC must be inserted when available.
5. **Architecture + robustness/calibration + limitations:** documented above.
6. **Demo video + deployment:** links must be inserted before submission.

The hackathon brief states that reproducibility is a scored gate and that unseen-generator AUC is the primary tie-break. Do not submit invented values for missing official metrics.

---

## Citation / References

- SIH-2026 Internal Hackathon — **SignalScope: Telling Real From Synthetic in the Age of Generative Media**
- PyTorch / torchvision
- Streamlit
- NumPy
- Pillow

See the repository's code and model report for implementation-specific details and experiment outputs.
