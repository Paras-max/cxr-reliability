# PROJECT MASTER CHEAT SHEET / HANDBOOK
# Reliability-Aware Multi-Agent System for Chest X-Ray Pneumonia Classification

> **RESEARCH PROTOTYPE. Not a clinical diagnostic system. Not validated for clinical use.**
> All thresholds marked as PROVISIONAL are not calibrated. All calibration marked INCOMPLETE is not done.
> Source of truth: actual repository code (inspected September 2026).

---

## 🔴 PROJECT STATUS SNAPSHOT (as of inspection date)

| Phase | Component | Status |
|---|---|---|
| P3 | Dataset + Splits | ✅ COMPLETE |
| P4 | Base Model + Preprocessing | ✅ COMPLETE |
| P4.5 | Probability Calibration (code) | ✅ IMPLEMENTED (code exists) |
| P4.5 | Calibration RUN (inference on val set) | ❌ NOT RUN — needs ~2 hours CPU |
| P5 | OOD Reference Fitting | ⚠️ DEV MODE ONLY (2000/78299 train images) |
| P5 | Full OOD Fit (78k images) | ❌ NOT RUN — estimated 10+ hours CPU |
| P6 | Uncertainty Agent | ✅ COMPLETE (provisional thresholds) |
| P7 | Quality Agent | ✅ COMPLETE (provisional thresholds) |
| P8 | Decision Agent | ✅ COMPLETE |
| P9 | Repair Agent | ✅ COMPLETE |
| P10 | Verification Agent | ✅ COMPLETE (provisional threshold) |
| P11 | Pipeline Orchestrator | ✅ COMPLETE |
| P12 | Streamlit Dashboard | ✅ COMPLETE (running) |
| P13 | Final Calibration + Evaluation | ❌ NOT STARTED |

---

# PART 1 — PROJECT OVERVIEW

## Simple Explanation

Imagine a hospital has an AI system that looks at chest X-rays and tries to detect pneumonia (a lung infection). The AI gives an answer — but how do you know if you should trust that answer?

Maybe the X-ray was blurry. Maybe the image is from a very different type of scanner than the AI was trained on. Maybe the AI is genuinely unsure. In all these cases, just blindly trusting the AI output is dangerous.

This project builds a **reliability checking system** around an existing pneumonia AI. Instead of just giving you a Yes/No answer, it checks:
1. Is the image good quality?
2. Does this image look like something the AI has seen before?
3. Is the AI confident about its answer?
4. If not, can we fix the image and try again?
5. Should a human doctor review this instead?

## Technical Explanation

**Project Title:** Reliability-Aware Multi-Agent System for Chest X-Ray Pneumonia Classification

**One-line definition:** A multi-agent pipeline that wraps a pretrained DenseNet-121 pneumonia classifier with reliability checks (image quality, out-of-distribution detection, uncertainty quantification) and routes each X-ray to an action: ACCEPT, REPAIR, ESCALATE, or REJECT — rather than blindly trusting model confidence.

**Problem Statement:** Pretrained medical AI models can produce unreliable predictions when inputs are degraded, from a different data distribution, or genuinely ambiguous. Standard classification pipelines do not detect or handle these failure modes.

**Research Question:** *Can a structured, auditable, multi-agent system reliably determine when an AI pneumonia classifier should be trusted, when image repair can improve reliability, and when human review is required?*

**Project Objective:** Study system-level reliability assessment for medical image classification — the research object is not raw accuracy but whether the system correctly knows when to trust its own predictions.

**Why Ordinary Classification is Insufficient:**
- High accuracy on training distribution ≠ reliable on real clinical images
- Models do not automatically flag when they are uncertain
- Models do not detect when input images are corrupted or from different scanners
- Models do not refuse to answer when they should not

**Why a Multi-Agent Architecture:**
- Separation of concerns: each agent has a single, testable responsibility
- Explicit routing rules: transparent, auditable, not a black box
- Safety by design: ambiguous cases fail toward human review, never silent accept
- Maintainable: each agent can be updated independently

**Why This Is a Research Prototype (Not a Clinical System):**
- Not validated on clinical patient data
- Thresholds are provisional / heuristic starting points
- No regulatory approval (FDA, CE)
- NIH ChestX-ray14 labels are text-mined, not radiologist-confirmed for every image
- Calibration is incomplete (as of inspection)

**Intended Users:** AI/ML researchers, final-year engineering students, academic research groups studying medical AI reliability.

---

# PART 2 — COMPLETE SYSTEM ARCHITECTURE

## Conceptual Flow

```
Raw Image (PNG/JPG/JPEG)
        ↓
┌──────────────────────────────────────────┐
│  INPUT VALIDATION (Orchestrator)          │
│  - SHA256 hash, UUID, non-destructive     │
└──────────────────────────────────────────┘
        ↓
  [PARALLEL Initial Screening]
  ┌─────────────────┐    ┌─────────────────────────────────┐
  │  Quality Agent  │    │  Base Model Agent (DenseNet-121) │
  │  - Blur         │    │  - Pneumonia score (index 8)     │
  │  - Noise        │    │  - 1024-dim feature vector       │
  │  - Exposure     │    └─────────────────────────────────┘
  └─────────────────┘           ↓
                       ┌─────────────────┐
                       │   OOD Agent     │
                       │  Mahalanobis    │
                       │  distance       │
                       └─────────────────┘
                               ↓
                       ┌─────────────────────┐
                       │  Uncertainty Agent  │
                       │  confidence, entropy│
                       └─────────────────────┘
        ↓
┌──────────────────────────────────────────┐
│  Decision Agent (Rule Table)              │
│  R1: SEVERE OOD   → REJECT               │
│  R2: BORDERLINE   → ESCALATE             │
│  R3: BAD QUALITY  → REPAIR               │
│  R4: HIGH uncer.  → ESCALATE             │
│  R6: LOW uncer.   → ACCEPT               │
│  R7: DEFAULT      → ESCALATE             │
└──────────────────────────────────────────┘
        ↓                    ↓                  ↓            ↓
   ACCEPT/ESCALATE        REJECT             REPAIR        ESCALATE
   Release or flag     Human review       Repair Agent      Human
   for human review    No prediction         ↓              review
                                    Verification Agent
                                    (before vs after)
                                         ↓           ↓
                                    VERIFIED      ESCALATE
                                    Accept      Human review
```

## Agent Summary Table

| Agent | Purpose | Input | Output | Algorithm | Thresholds | Status | Files |
|---|---|---|---|---|---|---|---|
| Quality Agent | Detect blur/noise/exposure | Raw image | QualityResult | Laplacian, SNR, mean intensity | blur<100, SNR<15dB, mean<20 or >235 | ✅ Implemented | `agents/quality.py`, `quality/evaluator.py` |
| Base Model Agent | Run DenseNet-121 | 224×224 tensor | Pneumonia score [0-1], 1024-dim features | DenseNet-121 forward pass | None | ✅ Implemented | `models/base_model.py` |
| OOD Agent | Detect distribution shift | 1024-dim features | Mahalanobis distance, OOD level | Mahalanobis distance (Cholesky) | borderline=56.3, severe=68.4 (DEV PROVISIONAL) | ⚠️ Dev mode only | `agents/ood.py`, `ood/detector.py`, `ood/mahalanobis.py` |
| Uncertainty Agent | Quantify model confidence | Pneumonia score | confidence, entropy, level | Binary entropy + confidence | conf≥0.85→LOW, conf≤0.60→HIGH | ✅ Implemented (provisional) | `agents/uncertainty.py`, `uncertainty/estimator.py` |
| Decision Agent | Route to action | Quality+OOD+Uncertainty results | Action (ACCEPT/REPAIR/ESCALATE/REJECT) | Rule table (7 rules, precedence-ordered) | Configurable borderline_margin | ✅ Implemented | `agents/decision/rules.py` |
| Repair Agent | Fix image quality | Raw image + QualityResult | Repaired image + RepairResult | CLAHE, NL-Means, Unsharp masking | Provisional repair bounds (all null in config) | ✅ Implemented | `agents/repair.py`, `repair/pipeline.py` |
| Verification Agent | Confirm repair improved reliability | Before/after signal bundles | VerificationResult (VERIFIED/ESCALATE/REJECT) | Delta confidence + quality/OOD checks | min_confidence_gain=0.15 (PROVISIONAL) | ✅ Implemented | `agents/verification.py` |

---

# PART 3 — DATASET

## What is NIH ChestX-ray14?

A large public chest X-ray dataset released by the US National Institutes of Health. It contains frontal chest X-ray images with 14 radiological findings labelled via natural language processing (NLP) on radiology reports — **not** direct radiologist annotation.

## Dataset Facts (Actual from code)

| Property | Value |
|---|---|
| Total images | ~112,120 |
| Conditions labelled | 14 pathologies (multi-label) |
| Pneumonia label | Derived from text mining (NLP) |
| Image format | PNG |
| Image resolution | Various (resized to 224×224 for model) |
| Total patients | ~30,805 unique patient IDs |

## Actual Split Sizes (from manifests, confirmed by code run)

| Split | Images | Pneumonia+ | Patients |
|---|---|---|---|
| Train | 78,299 | 1,019 (1.30%) | 21,563 |
| Validation | 17,097 | 192 (1.12%) | 4,621 |
| Test | 16,724 | 220 (1.32%) | 4,621 |

**Total: ~112,120 images. Heavily imbalanced: ~1.1–1.3% pneumonia positive.**

## Patient-Level Split (Critical)

**Image-level split:** Each image is randomly assigned to train/val/test.
**Problem:** One patient can have multiple images → if patient A's image 1 is in train and image 2 is in test, the model sees the same patient during training and evaluation = **data leakage**.

**Patient-level split:** All images from one patient go to the same split only. This prevents leakage and gives a more realistic measure of generalization to unseen patients.

**Why this matters for medical imaging:** In the clinic, you evaluate on new patients, not new images from known patients. Patient-level splits simulate this correctly.

## Manifest Files

- `data/processed/train.csv` — Training images
- `data/processed/validation.csv` — Validation images (used for calibration, threshold selection)
- `data/processed/test.csv` — Test images (MUST NOT be touched until final evaluation)

**Columns:** `image_id, image_path, patient_id, finding_labels, label, label_name, age, gender, view_position`

**Binary label:** `label=1` → Pneumonia present, `label=0` → No pneumonia

## Data Leakage Policy
- Calibration uses ONLY validation.csv
- OOD fitting uses ONLY train.csv
- test.csv MUST NOT be used for fitting or threshold selection
- test.csv is reserved for final evaluation only

## Data Cleaning Policy
**Excluded (dropped):** Missing image path, duplicate image IDs, missing labels.
**Flagged but KEPT:** Suspicious age (e.g., age=414 — known NIH quirk), missing age, invalid gender, unusual view position.

---

# PART 4 — DATA PREPROCESSING

## TorchXRayVision Preprocessing Pipeline

The **exact, verified** preprocessing pipeline (`models/preprocessing.py`):

```
1. Load image from disk using Pillow (PIL)
2. Convert to grayscale (mode "L") — CXRs are single-channel
3. Resize to 224×224 using LANCZOS resampling
4. Convert to float32 numpy array in [0.0, 255.0]
5. Normalize to TXV HU-like range:
       pixel_txv = (pixel_uint8 / 255.0) × 2048.0 − 1024.0
   Maps: 0   → −1024.0
         255 → +1024.0
6. Shape tensor as (1, 1, 224, 224)
7. Return torch.Tensor float32
```

**Key formula:** `pixel_txv = pixel_uint8 × (2048/255) − 1024`

**Do NOT use ImageNet normalization (mean/std subtraction).** TXV has its own range.

## Functions

| Function | File | Purpose |
|---|---|---|
| `load_image_for_txv(path)` | `models/preprocessing.py` | Load from disk → tensor |
| `prepare_image_for_txv(image)` | `models/preprocessing.py` | Any format → tensor (no double-normalization) |
| `prepare_image_for_quality(image)` | `quality/evaluator.py` | Any format → 2D float64 [0-255] for quality metrics |

## Critical Distinction

| Representation | Range | Purpose |
|---|---|---|
| Raw pixel | 0–255 uint8 | On disk, original image |
| Quality image | 0–255 float64 2D | Used by Quality Agent (NOT model-normalized) |
| TXV tensor | −1024 to +1024 float32 (1,1,224,224) | Model input |
| Model output | 0–1 (sigmoid) | Pneumonia raw score |

The Quality Agent uses the **original pixel scale**, NOT the TXV-normalized tensor.

---

# PART 5 — BASE MODEL

## What Is DenseNet-121?

**DenseNet** = Densely Connected Convolutional Network. Instead of residual (skip) connections between consecutive layers, each layer connects to **all subsequent layers**. This prevents vanishing gradients and encourages feature reuse.

**DenseNet-121** has 121 layers (4 Dense Blocks + 3 Transition Layers).

## TorchXRayVision

An open-source library (`torchxrayvision`) providing pretrained DenseNet models specifically for chest X-rays. The `densenet121-res224-nih` variant was trained on NIH ChestX-ray14 on 14 pathology classes.

## Model Details

| Property | Value |
|---|---|
| Model ID | `densenet121-res224-nih` |
| Input | (1, 1, 224, 224) float32 tensor, range [−1024, 1024] |
| Output | 18 sigmoid probabilities (NOT raw logits) |
| Target output | **Index 8 = Pneumonia** (verified 2026-09-22) |
| Feature layer | `features.norm5` (final BN before GAP) |
| Feature dimension | **1024** |
| Training dataset | NIH ChestX-ray14 |
| Weights | Pretrained, downloaded from TXV (~28 MB) |

## Full 18-Label Mapping

```
[0]  Atelectasis        [9]  Pleural_Thickening
[1]  Consolidation      [10] Cardiomegaly
[2]  Infiltration       [11] Nodule
[3]  Pneumothorax       [12] Mass
[4]  Edema              [13] Hernia
[5]  Emphysema          [14] (empty)
[6]  Fibrosis           [15] (empty)
[7]  Effusion           [16] (empty)
[8]  **PNEUMONIA**      [17] (empty)
```

## What the Base Model Does NOT Provide

- ❌ Raw pre-sigmoid logits (TXV applies sigmoid internally)
- ❌ Calibrated clinical probabilities
- ❌ Multi-class uncertainty
- ❌ Any claim of diagnostic correctness

## Terminology Rules (from code)

| Term | Meaning |
|---|---|
| `pneumonia_raw_score` | sigmoid probability at index 8, range [0,1] |
| `raw_probability` | same as above |
| `pneumonia_probability` | same as above (field name in contract) |
| `pneumonia_logit` | same value stored here too (TXV does not expose true logits) |
| **Calibrated probability** | Does NOT exist yet — Phase 13 planned |

## Feature Extraction

The `FeatureExtractor` context manager attaches a **forward hook** to `features.norm5`. During the same forward pass that produces output probabilities:
1. Hook captures (batch, 1024, 7, 7) activation from norm5
2. Applies ReLU
3. Global average pool → (batch, 1024)
4. Result: 1024-dim feature vector used by OOD Agent

**Zero extra inference overhead** — one forward pass feeds both outputs.

---

# PART 6 — MODEL CALIBRATION

## What Is Calibration?

A model is **well-calibrated** if when it says "80% confident," it is correct 80% of the time. The raw sigmoid output of a model trained for multi-label classification is NOT automatically calibrated for a specific binary task.

**Example:** If raw score = 0.7 but the model is only right 50% of the time at that score, then 0.7 is misleading.

## Why Calibration Matters

- Enables trust in the predicted probability
- Required before any clinical risk stratification
- ECE (Expected Calibration Error) measures how far predictions deviate from true frequencies

## Methods Implemented (in `calibration/probability_calibration.py`)

| Method | Description | Status |
|---|---|---|
| **Platt Scaling** | Logistic regression σ(a·x + b) on raw scores | ✅ Code implemented |
| **Isotonic Regression** | Non-parametric monotone mapping | ✅ Code implemented |
| **ECE** | Expected Calibration Error (15 equal-width bins) | ✅ Code implemented |
| **Brier Score** | Mean squared error of probabilities | ✅ Code implemented |
| **F1-optimal threshold** | argmax F1 over 200 threshold candidates | ✅ Code implemented |
| **Youden's J** | argmax (Sensitivity + Specificity − 1) | ✅ Code implemented |
| **Balanced Accuracy** | argmax balanced accuracy | ✅ Code implemented |

## Calibration Script: `scripts/calibrate_pneumonia.py`

**Parts:**
1. Run Base Model on ALL 17,097 validation images → save to `outputs/calibration/validation_predictions.csv`
2. Threshold analysis (200 candidates, 3 strategies)
3. Probability calibration (Platt + Isotonic)
4. Save `calibration_results.json`
5. Print summary

**Data Leakage Policy:** ONLY `validation.csv` is used. `test.csv` is NEVER touched.

## Current Calibration Status

| Item | Status |
|---|---|
| Calibration code (Platt, Isotonic, ECE, Brier, F1-optimal) | ✅ Fully implemented |
| `validation_predictions.csv` | ❌ NOT generated — inference not run |
| `platt_calibrator.joblib` | ❌ NOT saved |
| `isotonic_calibrator.joblib` | ❌ NOT saved |
| `calibration_results.json` | ❌ NOT saved |
| Dashboard using calibrated probability | ❌ NOT done — displays raw score as "Pneumonia Model Score" |

## Runtime Estimate

~400 ms/image on CPU × 17,097 images = **~1.9 hours** for full validation inference.

## How to Run Calibration (when ready)

```bash
# Full calibration on all 17,097 validation images:
python scripts/calibrate_pneumonia.py

# Force re-inference even if predictions file exists:
python scripts/calibrate_pneumonia.py --force-recompute

# Use Youden's J threshold strategy:
python scripts/calibrate_pneumonia.py --strategy youden
```

**DO NOT run this yet unless you can dedicate ~2 hours of CPU time.**

---

# PART 7 — QUALITY AGENT

## Purpose

Detect image-quality degradations that could make downstream model predictions unreliable. The Quality Agent inspects the **original pixel-scale image** (NOT the TXV-normalized tensor).

## Blur Assessment

**Method:** Laplacian variance

```python
laplacian_variance = cv2.Laplacian(img, cv2.CV_64F).var()
blur_pct = 1.0 - min(laplacian_variance / reference_max_variance, 1.0)
is_blurred = laplacian_variance < blur_laplacian_var_min
```

| Threshold | Value | Status |
|---|---|---|
| `blur_laplacian_var_min` | 100.0 | PROVISIONAL (PRD starting point) |
| `reference_max_variance` | 500.0 | PROVISIONAL |

**Interpretation:** The Laplacian operator approximates the second derivative of the image. Sharp images have high spatial frequency content → high Laplacian variance. Blurry images → low variance.

## Noise Assessment

**Method:** Residual from median filter → MAD-based SNR

```python
filtered = cv2.medianBlur(img_uint8, ksize=5)
residual = img.astype(float) - filtered.astype(float)
mad = np.median(np.abs(residual - np.median(residual)))
signal = np.mean(img)
snr_db = 20 * log10(signal / (1.4826 * mad + epsilon))
is_noisy = snr_db < snr_db_min
```

| Threshold | Value | Status |
|---|---|---|
| `snr_db_min` | 15.0 dB | PROVISIONAL (PRD starting point) |

**1.4826** is the consistency factor to make MAD a consistent estimator of standard deviation for Gaussian noise. **SNR = signal-to-noise ratio in decibels.**

## Exposure Assessment

**Method:** Mean intensity + dark/bright pixel fractions

```python
mean_intensity = img.mean()
is_underexposed = mean_intensity < exposure_mean_min
is_overexposed = mean_intensity > exposure_mean_max
```

| Threshold | Value | Status |
|---|---|---|
| `exposure_mean_min` | 20.0 | PROVISIONAL |
| `exposure_mean_max` | 235.0 | PROVISIONAL |

## Overall Quality Decision

```
If any component is POOR (blur, noise, exposure):
    overall = POOR

Elif any metric is within borderline_margin_pct (15%) of threshold:
    overall = DEGRADED

Else:
    overall = GOOD
```

**QualityLevel:** `GOOD`, `DEGRADED`, `POOR`

## Why These Metrics for Chest X-rays?

- **Blur:** Motion artifact (patient breathing), defocusing, or low scanner quality reduces diagnostic detail
- **Noise:** Quantum noise from low X-ray dose; affects edge detection and subtle opacities
- **Exposure:** Under/overexposed X-rays lose detail in dark or bright regions (lungs vs. mediastinum)

**Important:** These metrics detect technical quality, NOT diagnostic quality. A technically good image may still be diagnostically abnormal.

---

# PART 8 — OOD DETECTION

## What Is OOD?

**In-distribution (ID):** Images that look like what the model was trained on.
**Out-of-distribution (OOD):** Images from a different source, scanner, patient population, or preprocessing that the model has never seen similar data to.

**Distribution shift** occurs when test data comes from a different statistical distribution than training data.

**Why OOD matters for medical AI:** A model trained on NIH data from standard PA chest X-rays may produce unreliable predictions on:
- Lateral views
- Data from different hospitals/scanners
- Paediatric patients vs adult training data
- Different image processing pipelines

## Implementation

### Feature Extraction
The 1024-dim feature vector from `features.norm5` (GAP-pooled) is extracted for every image via the forward hook.

### Reference Distribution Fitting
Fitted on **training data only** using a multivariate Gaussian:
```
μ = mean of all training feature vectors (shape: 1024)
Σ = sample covariance + λI  (Tikhonov regularization)
L = Cholesky factor of Σ_reg (L·Lᵀ = Σ_reg)
```

### Mahalanobis Distance

**Formula:**
```
d(x) = √[(x − μ)ᵀ · Σ⁻¹ · (x − μ)]
```

**Numerical Implementation (avoids explicit matrix inversion):**
```
1. Solve triangular system: L·z = (x − μ)   [forward substitution]
2. d(x) = √(zᵀz)
```

This is exactly equivalent to the Mahalanobis distance but numerically stable. Uses `scipy.linalg.solve_triangular`.

### OOD Level Mapping

| Distance | Level |
|---|---|
| d ≤ borderline_threshold | IN_DISTRIBUTION |
| borderline < d ≤ severe_threshold | BORDERLINE |
| d > severe_threshold | SEVERE |

### Current OOD Thresholds (PROVISIONAL — DEV MODE ONLY)

From `artifacts/ood_dev/metadata.json`:
| Threshold | Value | Basis |
|---|---|---|
| `mahalanobis_borderline` | 56.32 | 99th percentile of 500 val images |
| `mahalanobis_severe` | 68.44 | Interpolated between 99th and 99.9th percentile |

**⚠️ CRITICAL: These thresholds are PROVISIONAL DEVELOPMENT defaults fitted on only 2000/78299 training images and 500/17097 validation images. They MUST NOT be used for final research evaluation.**

### Energy Score

The classic energy score (Liu et al., 2020) requires raw pre-sigmoid logits:
```
E(x) = −T · log Σᵢ exp(logitᵢ / T)
```

**TXV applies sigmoid internally and does NOT expose raw logits.**

**What is implemented instead:**
- `energy_score_from_logits()`: True energy formula — only usable with real logits
- `energy_score_from_probs()`: **Approximation** using inverse-sigmoid (logit transform) on probabilities
- Energy score is **DISABLED by default** (`energy_enabled: false`)
- Mahalanobis is the **primary OOD detector**

### OOD Status Summary

| Item | Status |
|---|---|
| Mahalanobis implementation | ✅ Complete |
| Energy score (approx.) | ✅ Implemented, disabled by default |
| Dev OOD artifacts (2000 train images) | ✅ Available at `artifacts/ood_dev/` |
| Full OOD artifacts (78k train images) | ❌ NOT fitted |
| OOD AUROC evaluation | ❌ NOT done (requires true OOD dataset) |

### Commands

```bash
# Development OOD fit (2000 training + 500 val images):
python scripts/fit_ood_reference.py --dev --dev-train-samples 2000 --dev-val-samples 500

# Full OOD fit (all 78k training images) — takes ~10 hours on CPU:
python scripts/fit_ood_reference.py

# Run dashboard with dev OOD:
$env:CXR_OOD_STATS_DIR = "artifacts/ood_dev"
streamlit run src/cxr_reliability/dashboard/app.py
```

---

# PART 9 — UNCERTAINTY AGENT

## Concepts

**Confidence:** How far the model's score is from the decision boundary (0.5).
- If score = 0.9 → model strongly predicts Pneumonia (high confidence)
- If score = 0.5 → model is maximally uncertain (equal probability for both classes)

**confidence = max(p, 1−p)**

This maps any score in [0,1] to a value in [0.5, 1.0] where 0.5 = maximum uncertainty, 1.0 = maximum confidence.

## Formulas

**Binary entropy:**
```
H(p) = −p·log₂(p) − (1−p)·log₂(1−p)
```
Maximum H = 1 bit (at p=0.5), Minimum H = 0 (at p=0 or p=1).

**Normalized entropy:**
```
H_norm(p) = H(p) / log₂(2) = H(p)     [since log₂(2)=1]
           = −p·log₂(p) − (1−p)·log₂(1−p)
```
Normalized entropy is in [0, 1].

## Uncertainty Level Classification (Two-Level System)

| Condition | Level |
|---|---|
| conf ≥ 0.85 **AND** norm_entropy ≤ 0.25 | LOW |
| otherwise | HIGH |

**LOW requires high confidence AND low normalized entropy. All other samples are HIGH uncertainty.**

## Important Clarification: What This IS and IS NOT

**IS:** Binary entropy derived from the model's single sigmoid score for Pneumonia. This reflects the model output ambiguity.

**IS NOT:**
- Predictive entropy from multiple forward passes (MC dropout / deep ensembles)
- Epistemic vs. aleatoric uncertainty decomposition
- Calibrated uncertainty estimate
- Clinical confidence in diagnosis

The system uses **binary entropy from a single deterministic forward pass** as a **proxy** for uncertainty.

## Implementation Files

- `agents/uncertainty.py` — UncertaintyAgent wrapper
- `uncertainty/estimator.py` — UncertaintyEstimator (levels, reasoning)
- `uncertainty/confidence.py` — `compute_binary_confidence(p) = max(p, 1-p)`
- `uncertainty/entropy.py` — `compute_binary_entropy()`, `compute_normalized_entropy()`

---

# PART 10 — DECISION AGENT

## Complete Rule Table

```
PRECEDENCE ORDER (rules checked in order, first match wins):

R0. Input validation fails        → ESCALATE  (safe default)
R1. OOD level = SEVERE            → REJECT    (no prediction released)
R2. OOD = BORDERLINE              → ESCALATE or REJECT (configurable)
    OR any signal near threshold
R3. Quality = POOR or DEGRADED    → REPAIR    (only if OOD = IN_DISTRIBUTION)
    AND OOD = IN_DISTRIBUTION
R4. Quality = GOOD, OOD = ID      → ESCALATE
    AND uncertainty = HIGH
R6. Quality = GOOD, OOD = ID      → ACCEPT
    AND uncertainty = LOW
R7. Any other combination         → ESCALATE  (safe default)
```

## Actions Explained

| Action | Meaning | Prediction Released? | Human Review? |
|---|---|---|---|
| ACCEPT | All signals good, trust model | Yes | No |
| REPAIR | Quality problem detected, try to fix image | No (yet) | No |
| ESCALATE | Ambiguous or uncertain — flag for review | No | Yes |
| REJECT | Severe OOD — do not trust prediction at all | No | Yes |

## Two-Level Uncertainty Rationale (LOW / HIGH)

Uncertainty is evaluated as a 2-level categorization:
- **LOW:** confidence ≥ 0.85 AND normalized entropy ≤ 0.25 (proceeds to ACCEPT when Quality=GOOD, OOD=IN_DISTRIBUTION).
- **HIGH:** everything else, including borderline and intermediate confidence samples (routes to ESCALATE for human review). No active MEDIUM state exists.

## Why REPAIR → Verification → Still Possibly ESCALATE

REPAIR means "try to fix the image quality and re-evaluate." After repair:
- Fresh inference runs on the repaired image
- Verification Agent checks if confidence improved by ≥ 0.15 (provisional)
- If yes → ACCEPT (released as "Accepted After Repair")
- If no → ESCALATE (repair did not achieve sufficient improvement)

Repair never automatically produces ACCEPT.

## Rule IDs (used in audit records)

```
R0_INVALID_INPUT_ESCALATE
R1_SEVERE_OOD_REJECT
R2_BORDERLINE_SIGNAL
R3_QUALITY_REPAIR
R4_HIGH_UNCERTAINTY_ESCALATE
R6_GOOD_IN_LOW_ACCEPT
R7_SAFE_DEFAULT_ESCALATE
```

---

# PART 11 — REPAIR AGENT

## Overview

The Repair Agent applies **non-destructive, deterministic image quality corrections** when directed by the Decision Agent. It does NOT:
- Modify the original image tensor (always clones first)
- Make diagnostic claims
- Guarantee improvement
- Invent anatomy or hallucinate structures

## Repair Methods

### 1. Exposure Correction: CLAHE

**CLAHE** = Contrast Limited Adaptive Histogram Equalization

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
repaired = clahe.apply(img_uint8)
```

**Why:** Regular histogram equalization can over-enhance uniform regions. CLAHE clips the histogram at a configurable limit to prevent noise amplification. Tile-based: applies locally (8×8 tiles), preserving local contrast.

**Parameters (provisional):** `clip_limit=2.0`, `tile_grid_size=(8,8)`

### 2. Noise Reduction: Non-Local Means

```python
repaired = cv2.fastNlMeansDenoising(img, h=3.0,
    templateWindowSize=7, searchWindowSize=21)
```

**Why:** NL-Means denoises by averaging similar patches across the entire image — preserves structural edges (lung borders) better than Gaussian blur. Parameter `h=3.0` controls strength.

**Parameters (provisional):** `h=3.0`, `template=7×7`, `search=21×21`

### 3. Blur Correction: Unsharp Masking

```python
gaussian = cv2.GaussianBlur(img, (0,0), sigmaX=1.0)
repaired = cv2.addWeighted(img, 1 + 0.5, gaussian, -0.5, 0)
```

**Why:** Conservative sharpening. Adds back a fraction (0.5) of the high-frequency component (original minus blurred). Avoids aggressive ring artifacts.

**Parameters (provisional):** `radius=1.0`, `amount=0.5`

## Repair Execution Order (deterministic)

```
1. EXPOSURE (CLAHE first — improves signal for subsequent steps)
2. NOISE (NL-means — on exposure-corrected image)
3. BLUR (Unsharp mask — last, to not sharpen noise)
```

## Repair vs Diagnostic Improvement

**Image enhancement** ≠ **diagnostic improvement**

- Repair can improve pixel statistics (blur metric, SNR)
- It does NOT prove the repaired image is more diagnostically correct
- Verification Agent checks if *model confidence* improved, not clinical correctness
- `image_changed=True` means numerical pixel change only

---

# PART 12 — VERIFICATION AGENT

## Purpose

Confirm that image repair actually improved reliability signals before releasing the prediction. If repair does not help enough, escalate to human review.

## What Is Compared (Before vs After Repair)

| Signal | Before | After | Delta |
|---|---|---|---|
| Confidence | conf_before | conf_after | delta_conf = conf_after − conf_before |
| Quality level | q_level_before | q_level_after | quality improved? |
| OOD distance | ood_dist_before | ood_dist_after | delta_ood |
| OOD level | ood_level_before | ood_level_after | degraded? |
| Entropy | ent_before | ent_after | delta_entropy |

## Verification Rules (Evaluated in Order)

1. If repair was not applied → FAIL → ESCALATE/REJECT
2. If confidence data missing → FAIL
3. If post-repair OOD = SEVERE → FAIL
4. If OOD worsened (ID → BORDERLINE or SEVERE) → FAIL
5. If quality worsened (GOOD → POOR) or SNR dropped > 5 dB → FAIL
6. If delta_confidence < min_confidence_gain (0.15) → FAIL
7. All conditions pass → VERIFIED → release prediction

## The 0.15 Confidence Gain Threshold

**Source:** PRD FR-7 example (e.g., ≥ 0.15)

**Status: PROVISIONAL DEVELOPMENT DEFAULT**

The `VerificationResult` contract always includes `threshold_is_provisional=True`. This value was not calibrated on data; it is an engineering starting point.

## Verification Outcomes

| Outcome | Next Step | Prediction Released? |
|---|---|---|
| VERIFIED | RELEASE | Yes ("Accepted After Repair") |
| ESCALATE | Human review | No |
| REJECT | Human review (no escalations left) | No |
| NOT_APPLICABLE | N/A | N/A (only when action ≠ REPAIR) |

---

# PART 13 — ORCHESTRATOR / PIPELINE

## Implementation: `ReliabilityPipeline` (`pipeline/orchestrator.py`)

## Complete Execution Flow

```python
def predict(image, input_id=None):
    1. INPUT VALIDATION
       - Non-destructive clone of input image
       - Compute SHA256 hash of input
       - Generate UUID for this inference run

    2. PREPROCESSING
       - prepare_image_for_txv(image) → model tensor
       - (existing base model preprocessing, no duplication)

    3. INITIAL SCREENING (sequential):
       a. quality_result  = quality_agent.run(image)
       b. model_forward   = base_model.run(tensor)
       c. ood_result      = ood_agent.run(features=model_forward.features)
       d. uncertainty_res = uncertainty_agent.run(score=model_forward.result.pneumonia_probability)

    4. INITIAL DECISION:
       decision = decision_agent.decide(quality, ood, uncertainty, state=DecisionState())

    5. ACTION ROUTING:
       if action == ACCEPT:
           → PipelineState.ACCEPTED
           → Release prediction

       elif action == ESCALATE:
           → PipelineState.ESCALATED
           → needs_human_review = True
           → Withhold prediction

       elif action == REJECT:
           → PipelineState.REJECTED
           → needs_human_review = True
           → Withhold prediction

       elif action == REPAIR:
           if repair_attempts >= max_repair_attempts:
               → PipelineState.ESCALATED (loop limit exceeded)
           else:
               repaired_img, repair_result = repair_agent.run(image, quality, decision)

               if not repair_result.repair_applied:
                   → PipelineState.ESCALATED (repair skipped/refused)
               else:
                   # Fresh after-repair inference:
                   quality_after  = quality_agent.run(repaired_img)
                   tensor_after   = prepare_image_for_txv(repaired_img)
                   model_after    = base_model.run(tensor_after)
                   ood_after      = ood_agent.run(features=model_after.features)
                   uncert_after   = uncertainty_agent.run(model_after.result.pneumonia_probability)

                   verif = verification_agent.run(before=..., after=...)

                   if verif.next_step == RELEASE:
                       → PipelineState.ACCEPTED_AFTER_REPAIR
                       → Release prediction
                   else:
                       → PipelineState.ESCALATED or REJECTED

    6. AUDIT LOGGING (safe — exceptions never block result)

    7. RETURN PipelineResult
```

## PipelineState Enum

```python
class PipelineState(str, Enum):
    ACCEPTED           = "ACCEPTED"
    ACCEPTED_AFTER_REPAIR = "ACCEPTED_AFTER_REPAIR"
    ESCALATED          = "ESCALATED"
    REJECTED           = "REJECTED"
    ERROR              = "ERROR"
```

## Safety Principles (from code)

- **Never silently ACCEPT on error:** Any exception during agent execution → safe ESCALATE
- **Non-destructive image handling:** Original image is cloned before any modification
- **Pre/post signal separation:** Before-repair signals never reused as after-repair signals
- **max_repair_attempts:** Default = 1 (configurable); prevents infinite repair loops
- **Zero clinical claims:** All reasoning strings explicitly disclaim research prototype status

## PipelineResult Contract

```python
@dataclass
class PipelineResult:
    pipeline_state: PipelineState
    final_action: Action
    needs_human_review: bool
    prediction: PredictionSummary | None     # None if not ACCEPTED
    quality: QualityResult
    base_model: BaseModelResult
    ood: OODResult
    uncertainty: UncertaintyResult
    decision: DecisionResult
    repair: RepairResult | None
    verification: VerificationResult | None
    repaired_image_png: bytes | None         # PNG bytes of repaired image (optional)
    input_sha256: str
    inference_id: str
    total_latency_ms: float
    disclaimer: str
    audit_record: dict | None
```

---

# PART 14 — DASHBOARD

## Launch Command

```bash
# From project root:
streamlit run src/cxr_reliability/dashboard/app.py

# Or with dev OOD stats:
$env:CXR_OOD_STATS_DIR = "artifacts/ood_dev"
streamlit run src/cxr_reliability/dashboard/app.py

# Or using launcher script:
bash scripts/run_dashboard.sh
```

**URL:** http://localhost:8501 (default Streamlit port)

## Dashboard Architecture

```
Streamlit UI  →  ReliabilityPipeline.predict(image)  →  All 7 agents
```

Zero duplication of agent logic in the UI.

## User Workflow

1. Upload chest X-ray (PNG, JPG, JPEG) via file uploader
2. Preview image with metadata (dimensions, color mode)
3. Click **"Analyze X-Ray"** button
4. View results across multiple tabs:
   - **System Assessment** — Final state, action, reliability label
   - **Human Review Banner** — Prominent warning when needs_human_review=True
   - **Before/After Repair** — Side-by-side comparison when repair was applied
   - **Base Model & Uncertainty tab**
   - **Quality Screening tab**
   - **OOD Shift Detection tab**
   - **Decision Routing tab**
   - **Audit/Latency Trace tab**

## Dashboard Files

| File | Purpose |
|---|---|
| `dashboard/app.py` | Main entry point, pipeline initialization, Streamlit page |
| `dashboard/components.py` | All render_*() functions for tabs and banners |

## Caching Policy

- Model/pipeline initialization: `@st.cache_resource` (loaded once per session)
- Per-image inference results: NOT globally cached (each analysis runs fresh)

## Important Disclaimers Displayed

- Output is "Pneumonia Model Score" — NOT a calibrated clinical probability
- "Research prototype — not intended for medical diagnosis"
- All thresholds are provisional
- OOD thresholds are dev-only when using ood_dev artifacts

## Why the UI Should NOT Call Output a Clinical Diagnosis

The model was trained on text-mined NIH labels (not direct radiologist annotation), is not calibrated, uses provisional thresholds, and has not been validated in any clinical setting. Calling the output a diagnosis would be factually incorrect and potentially misleading.

---

# PART 15 — PROJECT FILE STRUCTURE

```
cxr-reliability/
├── configs/
│   ├── thresholds/
│   │   └── v0_prd_defaults.yaml         ← Starting thresholds (PROVISIONAL, all null except verification)
│   └── pipeline_config.yaml             ← Execution config (borderline action, loop limits)
│
├── data/
│   └── processed/
│       ├── train.csv                    ← 78,299 images (21,563 patients)
│       ├── validation.csv               ← 17,097 images (4,621 patients)
│       └── test.csv                     ← 16,724 images (4,621 patients)
│
├── dataset/                             ← NIH images (images_001/ … images_012/)
│   └── images_NNN/images/              ← PNG files
│
├── artifacts/
│   ├── ood_dev/                         ← DEV OOD artifacts (2000 train, 500 val)
│   │   ├── reference_stats.npz          ← Mean, covariance, Cholesky factor
│   │   └── metadata.json               ← Thresholds, provenance (borderline=56.32, severe=68.44)
│   └── ood/                             ← Full OOD artifacts (NOT YET FITTED)
│
├── outputs/
│   └── calibration/                     ← Calibration results (NOT YET GENERATED)
│       ├── validation_predictions.csv
│       ├── calibration_results.json
│       ├── platt_calibrator.joblib
│       ├── isotonic_calibrator.joblib
│       ├── threshold_results.csv
│       ├── calibration_curve.png
│       └── threshold_analysis.png
│
├── scripts/
│   ├── calibrate_pneumonia.py           ← Full calibration workflow (Parts 1-5)
│   ├── calibrate_thresholds.py          ← NOT IMPLEMENTED (raises NotImplementedError)
│   ├── fit_ood_reference.py             ← OOD reference fitting (full + dev mode)
│   ├── evaluate_ood.py                  ← OOD evaluation script
│   ├── test_base_model.py               ← Base model smoke test
│   └── run_dashboard.sh                 ← Dashboard launcher
│
├── src/cxr_reliability/
│   ├── agents/
│   │   ├── base.py                      ← AgentBase abstract class
│   │   ├── quality.py                   ← QualityAgent (Phase 7)
│   │   ├── ood.py                       ← OODAgent (Phase 5)
│   │   ├── uncertainty.py               ← UncertaintyAgent (Phase 6)
│   │   ├── repair.py                    ← RepairAgent (Phase 9)
│   │   ├── verification.py              ← VerificationAgent (Phase 10)
│   │   └── decision/
│   │       ├── rules.py                 ← RuleTableDecisionAgent (Phase 8)
│   │       └── interface.py             ← Protocol definition
│   │
│   ├── calibration/
│   │   ├── probability_calibration.py   ← ProbabilityCalibrator (Platt + Isotonic)
│   │   ├── threshold_calibration.py     ← ThresholdAnalyzer (F1, Youden, BalAcc)
│   │   ├── confidence_cal.py            ← Convenience wrappers
│   │   ├── ood_fit.py                   ← OOD stats fitting functions
│   │   ├── quality_roc.py               ← (minimal, skeleton)
│   │   ├── registry.py                  ← freeze_thresholds() — NOT IMPLEMENTED
│   │   ├── repair_bounds.py             ← (skeleton)
│   │   └── verify_margin.py             ← (skeleton)
│   │
│   ├── config/
│   │   ├── settings.py                  ← get_settings(), paths, env vars
│   │   ├── thresholds.py                ← Threshold dataclasses (Pydantic)
│   │   └── pipeline_config.py           ← PipelineConfig, ExecutionConfig, LoopLimits
│   │
│   ├── contracts/
│   │   ├── common.py                    ← AgentName, DISCLAIMER
│   │   ├── base_model.py                ← BaseModelResult
│   │   ├── quality.py                   ← QualityResult, QualityLevel, QualityFlags
│   │   ├── ood.py                       ← OODResult, OODLevel
│   │   ├── uncertainty.py               ← UncertaintyResult, UncertaintyLevel
│   │   ├── decision.py                  ← DecisionResult, Action, DecisionState
│   │   ├── repair.py                    ← RepairResult, RepairStep
│   │   ├── verification.py              ← VerificationResult, SignalBundle
│   │   └── pipeline.py                  ← PipelineResult, PipelineState, PredictionSummary
│   │
│   ├── dashboard/
│   │   ├── app.py                       ← Streamlit entry point
│   │   └── components.py                ← All render_*() UI functions
│   │
│   ├── data/
│   │   ├── preprocessing.py             ← clean_metadata(), exclusion report
│   │   └── splits.py                    ← Patient-level split generation
│   │
│   ├── models/
│   │   ├── base_model.py                ← BaseModelAgent (Phase 4)
│   │   ├── preprocessing.py             ← TXV image preprocessing
│   │   ├── feature_hook.py              ← FeatureExtractor context manager
│   │   ├── txv_loader.py                ← TXV model loading, weight download
│   │   └── model_factory.py             ← get_model() factory function
│   │
│   ├── ood/
│   │   ├── detector.py                  ← OODDetector high-level interface
│   │   ├── mahalanobis.py               ← Mahalanobis distance (Cholesky)
│   │   ├── energy.py                    ← Energy score (true + pseudo-logit approx.)
│   │   └── statistics.py                ← fit_reference_stats(), save/load
│   │
│   ├── pipeline/
│   │   └── orchestrator.py              ← ReliabilityPipeline (Phase 11)
│   │
│   ├── quality/
│   │   ├── evaluator.py                 ← QualityEvaluator, prepare_image_for_quality()
│   │   ├── blur.py                      ← evaluate_blur()
│   │   ├── noise.py                     ← evaluate_noise()
│   │   └── exposure.py                  ← evaluate_exposure()
│   │
│   ├── repair/
│   │   ├── pipeline.py                  ← apply_sequential_repairs(), RepairConfig
│   │   ├── exposure.py                  ← repair_exposure() — CLAHE
│   │   ├── noise.py                     ← repair_noise() — NL-Means
│   │   └── blur.py                      ← repair_blur() — Unsharp masking
│   │
│   ├── uncertainty/
│   │   ├── estimator.py                 ← UncertaintyEstimator
│   │   ├── confidence.py                ← compute_binary_confidence()
│   │   └── entropy.py                   ← compute_binary_entropy(), normalized
│   │
│   ├── audit/                           ← JSONL audit logger
│   ├── evaluation/                      ← Metrics (partial stubs)
│   ├── api/                             ← FastAPI (stub — raises NotImplementedError)
│   └── reporting/                       ← Template report writer (stub)
│
├── tests/
│   ├── unit/                            ← 20 test files
│   │   ├── test_quality_agent.py
│   │   ├── test_base_model.py
│   │   ├── test_ood_agent.py
│   │   ├── test_ood_detector.py
│   │   ├── test_ood_dev_reference.py
│   │   ├── test_mahalanobis.py
│   │   ├── test_uncertainty_agent.py
│   │   ├── test_decision_agent.py
│   │   ├── test_repair_agent.py
│   │   ├── test_verification_agent.py
│   │   ├── test_orchestrator.py
│   │   ├── test_dashboard.py
│   │   ├── test_calibration.py
│   │   ├── test_contracts.py
│   │   ├── test_config.py
│   │   ├── test_audit_log.py
│   │   ├── test_base_model.py
│   │   ├── test_corruptions.py
│   │   ├── test_data_splits.py
│   │   └── test_evaluation_metrics.py
│   ├── integration/
│   └── regression/
│
└── docs/
    ├── architecture.md
    ├── calibration_log.md
    ├── decisions.md
    ├── limitations.md
    ├── module_contracts.md
    ├── pneumonia_calibration.md
    ├── quality_agent.md
    └── uncertainty_agent.md
```

---

# PART 16 — TESTING

## Test Suite Overview

| File | Tests | What It Validates |
|---|---|---|
| `test_quality_agent.py` | ~25 | Blur, noise, exposure detection, QualityLevel |
| `test_base_model.py` | ~20 | Forward pass, feature extraction, preprocessing |
| `test_ood_agent.py` | ~20 | OOD agent interface, is_stats_available property |
| `test_ood_detector.py` | ~25 | OODDetector fit/score/level mapping |
| `test_ood_dev_reference.py` | ~8 | Dev OOD mode, metadata, threshold checks |
| `test_mahalanobis.py` | ~20 | Mahalanobis distance, Cholesky, batch |
| `test_uncertainty_agent.py` | ~25 | Entropy, confidence, levels, edge cases |
| `test_decision_agent.py` | ~60+ | All 7 rules, edge cases, invalid inputs |
| `test_repair_agent.py` | ~30 | CLAHE, NL-means, unsharp, non-destructive |
| `test_verification_agent.py` | ~40+ | All verification conditions, bundles |
| `test_orchestrator.py` | ~20 | Full pipeline, action routing, repair loop |
| `test_dashboard.py` | ~25 | Dashboard render functions, component tests |
| `test_calibration.py` | ~20 | ProbabilityCalibrator, ThresholdAnalyzer, ECE |
| `test_contracts.py` | ~10 | Pydantic contract validation |
| `test_config.py` | ~5 | Settings loading |
| `test_audit_log.py` | ~5 | Audit logger |
| `test_corruptions.py` | ~3 | Image corruption library |
| `test_data_splits.py` | ~3 | Patient-level split validation |
| `test_evaluation_metrics.py` | ~5 | Metrics stubs |

**Last known status:** 423 tests passed (all unit tests, prior to inspection date)

## Test Commands

```bash
# Run ALL tests:
python -m pytest tests/

# Run with verbose output:
python -m pytest tests/ -v

# Run specific module:
python -m pytest tests/unit/test_decision_agent.py -v

# Run with markers (skip slow tests):
python -m pytest tests/ -m "not slow"

# Run with coverage:
python -m pytest tests/ --cov=src/cxr_reliability

# Quick smoke run:
python -m pytest tests/unit/test_config.py tests/unit/test_contracts.py -v
```

---

# PART 17 — CURRENT STATUS TABLE

| Component | Status | Evidence | Remaining Work |
|---|---|---|---|
| **Dataset (splits)** | ✅ Complete | train(78k), val(17k), test(17k) CSVs exist | None |
| **Preprocessing** | ✅ Complete | `models/preprocessing.py` tested | None |
| **Base Model Agent** | ✅ Complete | Forward pass, features, index 8 verified | None |
| **Quality Agent** | ✅ Complete (provisional) | All blur/noise/exposure metrics implemented | Threshold calibration |
| **OOD (code)** | ✅ Complete | Mahalanobis, Cholesky, energy, detector | None |
| **OOD (dev artifacts)** | ⚠️ Provisional | 2000/78299 train images; thresholds dev-only | Run full fitting (78k images) |
| **OOD (full artifacts)** | ❌ Missing | Not yet run | ~10 hours CPU job |
| **Uncertainty Agent** | ✅ Complete (provisional) | Binary entropy + confidence, levels | Threshold calibration after P4.5 |
| **Decision Agent** | ✅ Complete | 7 rules, all tested | None (rule frozen) |
| **Repair Agent** | ✅ Complete | CLAHE, NL-means, unsharp, non-destructive | Calibrate repair bounds |
| **Verification Agent** | ✅ Complete | Before/after comparison, min_conf_gain=0.15 | Calibrate confidence gain threshold |
| **Orchestrator** | ✅ Complete | Full 7-agent pipeline, repair loop | None |
| **Dashboard** | ✅ Complete | Streamlit running | None |
| **Calibration (code)** | ✅ Complete | Platt, Isotonic, ECE, Brier, F1-optimal | Run on validation set (~2h CPU) |
| **Calibration (run)** | ❌ Not done | No validation_predictions.csv | Run calibrate_pneumonia.py |
| **Final Evaluation** | ❌ Not started | test.csv untouched | After calibration complete |

---

# PART 18 — COMMANDS CHEAT SHEET

## Environment

```bash
# Activate Python virtual environment (Windows):
.venv\Scripts\activate

# Install dependencies:
pip install -r requirements-dev.txt
pip install -e .

# Check environment:
python -c "import torchxrayvision; import torch; print(torchxrayvision.__version__)"
```

## Testing

```bash
# Run all tests:
python -m pytest tests/ -v

# Run specific test file:
python -m pytest tests/unit/test_decision_agent.py -v

# Run fast tests only (skip model loading):
python -m pytest tests/unit/test_calibration.py tests/unit/test_contracts.py -v

# Run with coverage:
python -m pytest tests/ --cov=src/cxr_reliability --cov-report=term-missing
```

## Dashboard

```bash
# Run dashboard (requires dev OOD artifacts):
$env:CXR_OOD_STATS_DIR = "artifacts/ood_dev"
streamlit run src/cxr_reliability/dashboard/app.py

# Without env variable (auto-discovers artifacts/ood_dev):
streamlit run src/cxr_reliability/dashboard/app.py
```

## OOD Fitting

```bash
# Development OOD (2000 train + 500 val images, ~5-10 minutes on CPU):
python scripts/fit_ood_reference.py --dev --dev-train-samples 2000 --dev-val-samples 500

# Smoke test (10 images, validates script runs):
python scripts/fit_ood_reference.py --max-samples 10 --smoke-test

# Full OOD reference fitting (78k images, ~10 hours on CPU):
python scripts/fit_ood_reference.py

# Full OOD fitting on GPU (faster):
python scripts/fit_ood_reference.py --device cuda
```

## Calibration

```bash
# Run probability calibration on validation set (~2 hours on CPU):
python scripts/calibrate_pneumonia.py

# Use Youden's J threshold instead of F1-optimal:
python scripts/calibrate_pneumonia.py --strategy youden

# Force re-run even if predictions CSV exists:
python scripts/calibrate_pneumonia.py --force-recompute
```

## Base Model Inspection

```bash
# Quick base model smoke test:
python scripts/test_base_model.py

# Inspect model output labels:
python -c "import torchxrayvision as xrv; m = xrv.models.DenseNet(weights='densenet121-res224-nih'); print(list(m.pathologies))"
```

## Dataset Inspection

```bash
# Check split sizes:
python -c "import pandas as pd; [print(n, len(pd.read_csv(f'data/processed/{n}.csv'))) for n in ['train','validation','test']]"
```

---

# PART 19 — TROUBLESHOOTING

| Problem | Cause | How to Check | Fix |
|---|---|---|---|
| `FileNotFoundError: artifacts/ood_dev/reference_stats.npz` | OOD dev artifacts not present | `ls artifacts/ood_dev/` | Run `python scripts/fit_ood_reference.py --dev` |
| Dashboard shows "OOD stats not found" | Wrong OOD path or env var | Check `CXR_OOD_STATS_DIR` env var | `$env:CXR_OOD_STATS_DIR = "artifacts/ood_dev"` |
| All predictions are ESCALATE | OOD thresholds too tight or uncertainty thresholds wrong | Check metadata.json thresholds vs actual distances | Verify dev OOD thresholds or use full OOD fit |
| `AttributeError: 'method' object has no attribute ...` | `is_stats_available` called as method not property | Check `agents/ood.py` for `@property` decorator | Fixed in current code — reinstall or check |
| `RuntimeError: Model not loaded` | `load_model()` not called before `run()` | Check pipeline initialization | Call `base_model.load_model()` first |
| Slow inference (~400ms/image) | CPU-only inference | Check `device` setting | Use GPU if available: `device="cuda"` |
| `ValueError: Feature dimension mismatch` | OOD stats fitted with different model/layer | Check metadata.json `feature_dim` | Refit OOD with correct model |
| Calibration takes too long | 17,097 images at ~400ms each = ~2 hours | Check image count | Use --max-samples for subset test first |
| `ModuleNotFoundError: cxr_reliability` | Package not installed in editable mode | `python -c "import cxr_reliability"` | `pip install -e .` |
| Test failures on `test_ood_dev_reference.py` | Dev OOD artifacts missing or stale | Run the test file | Run `python scripts/fit_ood_reference.py --dev` first |
| `data/processed/` vs `data/manifests/` confusion | Script uses wrong path | Check `settings.data_dir` in get_settings() | Scripts default to `data/processed/` |
| Streamlit port conflict | Another Streamlit instance running | `netstat -an | grep 8501` | Kill existing instance or use `--server.port 8502` |

---

# PART 20 — RESEARCH EVALUATION

## When to Do Final Evaluation

Only after:
1. Full OOD fitting complete (78k training images)
2. Calibration run complete (17k validation images)
3. Optimal thresholds selected from validation set
4. All thresholds frozen in a versioned YAML

**NEVER touch test.csv during fitting or threshold selection.**

## Metrics to Report

### Classification Performance
- **F1 score** (primary — dataset is imbalanced ~1.1%)
- **Precision** (how often Pneumonia prediction is correct)
- **Recall / Sensitivity** (fraction of true Pneumonia detected)
- **Specificity** (fraction of true negatives correctly identified)
- **ROC-AUC** (threshold-independent ranking ability)
- **PR-AUC** (more informative than ROC for imbalanced datasets)
- **Confusion matrix** (TP, FP, TN, FN)

### Why Accuracy Alone Is Insufficient
The dataset is ~1.1% Pneumonia positive. A classifier that always predicts "No Pneumonia" achieves ~98.9% accuracy but 0% recall. Accuracy is misleading on imbalanced datasets.

### Calibration Metrics
- **ECE** (Expected Calibration Error) — before and after calibration
- **Brier Score** — before and after calibration
- Reliability diagram (calibration curve)

### OOD Evaluation
- **OOD AUROC** — requires a true OOD dataset (NOT NIH validation images)
- False-positive OOD rate on in-distribution images
- Note: NIH validation = in-distribution; cannot compute OOD AUROC from it

### Reliability/Pipeline Metrics
- Per-action breakdown: accuracy/error for ACCEPTED, ACCEPTED_AFTER_REPAIR
- Repair recovery rate (fraction of REPAIR → VERIFIED)
- Escalation rate (fraction escalated to human review)
- Average latency per image (ms)

### Train/Validation/Test Separation
```
train.csv     → Fit OOD reference statistics ONLY
validation.csv → Calibrate thresholds, select operating points, calibrate probabilities
test.csv      → Final evaluation ONLY (report final numbers)
```

---

# PART 21 — SAFETY AND LIMITATIONS

## Critical Limitations

1. **Research Prototype Only:** Not validated for clinical use, no regulatory approval.
2. **Not a Medical Device:** This software must not be used for actual patient diagnosis.
3. **Uncalibrated Probabilities:** The Pneumonia score is NOT a calibrated clinical probability.
4. **Provisional Thresholds:** All OOD thresholds, uncertainty thresholds, repair bounds, and verification margin are provisional starting points, not calibrated values.
5. **Dev OOD Mode:** Current OOD reference uses only 2000/78299 training images — thresholds will change significantly with full fitting.
6. **NIH Label Quality:** Labels derived from NLP on radiology reports — not direct radiologist annotation. Some labels may be incorrect.
7. **Single Model:** Only one pretrained model (DenseNet-121, NIH). No ensemble, no second opinion.
8. **Uncertainty Simplification:** Binary entropy from a single forward pass ≠ true epistemic uncertainty.
9. **Repair Not Validated:** Image enhancement does not prove clinical diagnostic improvement.
10. **Population Bias:** NIH ChestX-ray14 is predominantly adult US patients — may not generalize to other populations, ages, or equipment.
11. **Distribution Shift:** The OOD detector was fitted on NIH data. It may incorrectly classify valid clinical images from different scanners as OOD.
12. **Rare Valid Cases:** High Mahalanobis distance does not always mean OOD — rare but valid presentations may be flagged.
13. **No Longitudinal Validation:** System has not been tested on temporal cohort data.
14. **CPU-only by default:** Dashboard runs on local CPU; no performance optimization for real-time clinical use.

---

# PART 22 — BASIC CONCEPTS FOR VIVA

## AI, ML, Deep Learning

| Concept | Simple | Technical | In This Project |
|---|---|---|---|
| **AI** | Machines that appear intelligent | Systems making decisions from data/rules | The entire reliability system |
| **ML** | Machines learning from examples | Optimization of parameterized functions on training data | DenseNet-121 was ML-trained |
| **Deep Learning** | Many-layered neural networks | Hierarchical feature learning via gradient descent | DenseNet-121 is deep learning |
| **CNN** | Networks that detect visual patterns | Convolutional layers extract spatial features via learned filters | DenseNet-121 is a CNN |
| **DenseNet** | CNN where each layer connects to all future layers | Dense connectivity prevents vanishing gradients, encourages feature reuse | Base model architecture |
| **Transfer Learning** | Reuse a model trained on one task for another | Fine-tune or use pretrained weights for a related domain | We use NIH-pretrained DenseNet as-is |
| **Classification** | Assigning a category to an input | f: X → {class₁, ..., classₙ} | Pneumonia vs. Non-Pneumonia |
| **Binary classification** | Two-class classification | Sigmoid output ∈ [0,1] with threshold | Our task |
| **Multi-label classification** | Each input can have multiple labels | Sigmoid per label independently | How DenseNet-121 was trained (14 conditions) |
| **Sigmoid** | Squashing function to [0,1] | σ(x) = 1/(1+e⁻ˣ) | Applied to DenseNet output |
| **Probability** | Likelihood of an event | Real number in [0,1] | Raw Pneumonia score (not calibrated) |
| **Calibration** | Making probabilities match real frequencies | Post-hoc scaling (Platt, Isotonic) | Phase 4.5 — code complete, not yet run |
| **Confidence** | How certain the model is | max(p, 1-p) | Uncertainty Agent computation |
| **Uncertainty** | Model doubt | Entropy H(p) = -p log p - (1-p) log(1-p) | Binary entropy from sigmoid score |
| **Entropy** | Measure of information/disorder | H = -Σ p(x) log p(x) | Binary entropy (max=1 at p=0.5) |
| **OOD** | Image doesn't look like training data | Distribution shift: P_test ≠ P_train | OOD Agent detects this |
| **Mahalanobis distance** | Distance accounting for correlations | d = √[(x-μ)ᵀ Σ⁻¹ (x-μ)] | Primary OOD score |
| **Covariance** | How features vary together | Σ_{ij} = E[(Xᵢ-μᵢ)(Xⱼ-μⱼ)] | Reference distribution shape |
| **Cholesky decomposition** | Matrix factorization Σ = LLᵀ | Numerically stable way to avoid explicit matrix inversion | Used in Mahalanobis computation |
| **CLAHE** | Adaptive contrast enhancement | Contrast Limited Adaptive Histogram Equalization | Exposure repair method |
| **Denoising** | Removing noise from image | NL-means: average similar patches | Noise repair method |
| **Unsharp masking** | Sharpening via high-freq amplification | img_sharp = img + amount × (img - gaussian(img)) | Blur repair method |
| **Agent** | Autonomous decision-making unit | Software module with defined input/output/responsibility | Each of the 7 agents |
| **Multi-agent system** | Multiple collaborating agents | Decentralized, specialized components | 7 agents orchestrated by pipeline |
| **Orchestrator** | Coordinator of agents | Deterministic controller managing execution flow | ReliabilityPipeline |
| **Human-in-the-loop** | Human review for uncertain cases | Escalation mechanism for ambiguous AI decisions | ESCALATE/REJECT → human review |
| **Reliability** | Knowing when to trust predictions | Combination of quality, OOD, uncertainty checks | The core research problem |
| **Auditability** | Traceable, explainable decisions | Every agent returns reasoning string; audit JSONL logs | Requirement of the system |

---

# PART 23 — ADVANCED TECHNICAL CONCEPTS

## Sigmoid

```
σ(x) = 1 / (1 + e^(−x))
```
Maps any real number to (0,1). TXV applies sigmoid internally to each of the 18 outputs.

## Binary Entropy

```
H(p) = −p·log₂(p) − (1−p)·log₂(1−p)
```
- p = Pneumonia raw score ∈ [0,1]
- Maximum at p=0.5: H(0.5) = 1.0 bit (maximum uncertainty)
- Minimum at p=0 or p=1: H = 0 bit (maximum certainty)

## Normalized Entropy

Since we use log₂ and binary classification, log₂(2)=1, so H_norm = H for binary case.

```
H_norm(p) = H(p) / log₂(number of classes) = H(p) / log₂(2) = H(p)
```

Always in [0, 1].

## Mahalanobis Distance

```
d(x) = √[(x − μ)ᵀ · Σ_reg⁻¹ · (x − μ)]
```

Where:
- x = 1024-dim feature vector of query image
- μ = mean vector of training features (1024-dim)
- Σ_reg = regularized covariance matrix (1024×1024)
- λ = regularization constant (λI added to Σ)

**Implementation via Cholesky:**
```
Σ_reg = L · Lᵀ  (Cholesky factorization)
Solve: L·z = (x − μ)    (forward substitution, O(D²))
d(x) = √(zᵀz) = ||z||₂
```

**Why Cholesky:**
- Explicit inversion of 1024×1024 matrix is numerically unstable (condition number issues)
- Cholesky solve is O(D²) vs inversion O(D³)
- `scipy.linalg.solve_triangular` is highly optimized

## Covariance Regularization

```
Σ_reg = Σ + λI
```

The raw sample covariance of 1024-dim features may be rank-deficient (if N < D = 1024). Adding λI (Tikhonov regularization) ensures positive definiteness, enabling Cholesky decomposition.

Default: λ = 1e−5

## Laplacian Variance (Blur)

```
∇²f = ∂²f/∂x² + ∂²f/∂y²
Laplacian filter kernel (3×3):
  [0,  1, 0]
  [1, -4, 1]
  [0,  1, 0]

blur_metric = Var(∇²f)
```

High variance = sharp image (many edges with large second derivatives).
Low variance = blurry image (smooth, few edges).

## SNR (Signal-to-Noise Ratio)

```
Residual = image − median_filter(image)
MAD = median(|residual − median(residual)|)
σ_noise = 1.4826 × MAD   (robust noise estimate)
SNR_dB = 20 × log₁₀(mean(image) / (σ_noise + ε))
```

Higher SNR (dB) = cleaner image. Threshold: SNR < 15 dB → flagged as noisy.

## Confidence

```
confidence(p) = max(p, 1−p)
```
Maps [0,1] → [0.5, 1.0]. Score near 0.5 → confidence=0.5 (maximum uncertainty). Score near 0 or 1 → confidence near 1.0.

## Calibration Metrics

**ECE (Expected Calibration Error):**
```
ECE = Σ_b (|B_b| / n) × |acc(B_b) − conf(B_b)|
```
Where B_b = samples in bin b, acc = fraction of true positives, conf = mean predicted probability.

**Brier Score:**
```
BS = (1/n) × Σᵢ (pᵢ − yᵢ)²
```
Mean squared error between predicted probability and binary label. Perfect = 0, worst = 1.

---

# PART 24 — END-TO-END EXAMPLES

## Case 1: Ideal Case — ACCEPT

**Image:** Standard PA chest X-ray, good quality, from NIH scanner

| Agent | Result |
|---|---|
| Quality Agent | Laplacian=350, SNR=28dB, mean=115 → GOOD |
| Base Model | Pneumonia score = 0.03 (very low) |
| OOD Agent | Mahalanobis = 28.5 (< 56.3 borderline) → IN_DISTRIBUTION |
| Uncertainty Agent | confidence = max(0.03, 0.97) = 0.97, entropy = 0.20 → LOW |
| Decision Agent | Rule R6: GOOD + ID + LOW → ACCEPT |
| Final State | ACCEPTED. Prediction released: Non-Pneumonia (raw score=0.03) |

## Case 2: Poor Quality Image — REPAIR → ACCEPT

**Image:** Blurry chest X-ray (motion artifact)

| Agent | Result |
|---|---|
| Quality Agent | Laplacian=45 (< 100) → BLUR detected → POOR |
| Base Model | Pneumonia score = 0.55 |
| OOD Agent | Mahalanobis = 31.2 → IN_DISTRIBUTION |
| Uncertainty Agent | confidence = 0.55, entropy = 0.99 → HIGH |
| Decision Agent | Rule R3: POOR + ID → REPAIR |
| Repair Agent | Unsharp masking applied → repaired image |
| **Fresh inference on repaired image:** | |
| Quality After | Laplacian=155 → GOOD |
| OOD After | Mahalanobis = 30.1 → IN_DISTRIBUTION |
| Uncertainty After | Pneumonia score = 0.91 → confidence = 0.91, entropy = 0.42 → MEDIUM → still uncertain |
| Verification | delta_confidence = 0.91 − 0.55 = 0.36 ≥ 0.15 → VERIFIED |
| Final State | ACCEPTED_AFTER_REPAIR. Prediction released. |

## Case 3: Good Quality + High Uncertainty — ESCALATE

**Image:** Clear X-ray but model is ambiguous

| Agent | Result |
|---|---|
| Quality Agent | GOOD |
| Base Model | Pneumonia score = 0.52 |
| OOD Agent | IN_DISTRIBUTION |
| Uncertainty Agent | confidence = 0.52, entropy ≈ 1.0 → HIGH |
| Decision Agent | Rule R4: GOOD + ID + HIGH → ESCALATE |
| Final State | ESCALATED. needs_human_review=True. No prediction released. |

## Case 4: Severe OOD — REJECT

**Image:** Image from very different scanner (e.g., lateral view or paediatric scan)

| Agent | Result |
|---|---|
| Quality Agent | GOOD (image is technically fine) |
| Base Model | Pneumonia score = 0.72, features are very unusual |
| OOD Agent | Mahalanobis = 91.2 (> 68.4 severe threshold) → SEVERE |
| Decision Agent | Rule R1: SEVERE OOD → REJECT (regardless of quality or uncertainty) |
| Final State | REJECTED. needs_human_review=True. No prediction released. |

## Case 5: Repair Does Not Help — ESCALATE

**Image:** Moderately blurry, but sharpening doesn't improve model confidence

| Agent | Result |
|---|---|
| Quality Agent | POOR (blur) |
| OOD Agent | IN_DISTRIBUTION |
| Decision Agent | Rule R3: POOR + ID → REPAIR |
| Repair Agent | Unsharp masking applied |
| Quality After | Slightly improved (Laplacian: 65 → 105) |
| Uncertainty After | Confidence: 0.58 → 0.63 (delta = 0.05 < 0.15 threshold) |
| Verification | delta_confidence = 0.05 < 0.15 → FAIL |
| Final State | ESCALATED. needs_human_review=True. Repair insufficient. |

---

# PART 25 — VIVA QUESTIONS AND ANSWERS

## Basic Questions

**Q1: What is this project about in one sentence?**
A: A multi-agent system that checks whether a chest X-ray AI's prediction can be trusted, by evaluating image quality, distribution shift, and model uncertainty before releasing or escalating the result.

**Q2: Why not just use the DenseNet model directly?**
A: Raw model outputs are uncalibrated, do not indicate when the image is degraded or OOD, and never refuse to answer even when they should. The reliability system adds these safety checks.

**Q3: What is the difference between accuracy and reliability?**
A: Accuracy measures how often the model is correct on average. Reliability measures whether the model knows when to be trusted — when to answer and when to escalate.

**Q4: What is the dataset used?**
A: NIH ChestX-ray14, containing ~112,120 chest X-ray images with text-mined labels for 14 conditions. Train: 78,299 images, Validation: 17,097, Test: 16,724. ~1.1% Pneumonia positive (severely imbalanced).

**Q5: What is the Pneumonia output index in DenseNet-121 NIH?**
A: **Index 8** (verified by live model probe; labels: Atelectasis[0]...Effusion[7], Pneumonia[8], etc.).

**Q6: What preprocessing is applied?**
A: Load → grayscale → resize 224×224 (LANCZOS) → normalize to [−1024, +1024] via `pixel_txv = pixel × (2048/255) − 1024`.

**Q7: What is the output of the DenseNet model?**
A: 18 sigmoid probabilities in [0,1]. NOT raw logits. The model applies sigmoid internally.

## Intermediate Questions

**Q8: Why patient-level splitting?**
A: One patient has multiple X-rays. If image 1 is in train and image 2 is in test, the model has "seen" the patient during training → data leakage. Patient-level split ensures all images from one patient stay in the same split.

**Q9: What is OOD and why does it matter?**
A: Out-of-distribution means the test image comes from a different statistical distribution than training data. The model may produce unreliable predictions on OOD images because it extrapolates beyond its training experience.

**Q10: Why Mahalanobis distance for OOD?**
A: It measures how far a feature vector is from the training distribution center, scaled by the distribution's spread and correlations (covariance). A high Mahalanobis distance = the image's features are unusual relative to training data.

**Q11: Why is covariance regularization needed?**
A: The 1024-dim covariance matrix may be rank-deficient (if training N < 1024 in dev mode). Adding λI ensures the matrix is positive definite and Cholesky-decomposable.

**Q12: Why use Cholesky instead of explicit matrix inversion?**
A: Explicit 1024×1024 inversion amplifies numerical errors in near-singular matrices. Cholesky solve via `solve_triangular` is numerically stable and avoids O(D³) inversion.

**Q13: What is binary entropy and why use it for uncertainty?**
A: Binary entropy H(p) = −p·log₂(p) − (1-p)·log₂(1-p) measures how ambiguous the score p is. It is maximum (1 bit) at p=0.5 (equal uncertainty) and zero at p=0 or p=1 (certainty). It is used as a proxy for model uncertainty from a single sigmoid output.

**Q14: What is confidence in this project?**
A: `confidence = max(p, 1−p)`. Ranges from 0.5 (maximum uncertainty) to 1.0 (maximum confidence). It measures how far the score is from the decision boundary.

**Q15: What are the uncertainty thresholds?**
A: Two-level system: LOW if conf≥0.85 AND norm_entropy≤0.25; HIGH otherwise.

**Q16: What does the Decision Agent do?**
A: It applies a deterministic, precedence-ordered rule table to Quality + OOD + Uncertainty signals and produces one of 4 actions: ACCEPT, REPAIR, ESCALATE, REJECT.

**Q17: How is uncertainty routed?**
A: GOOD + IN_DISTRIBUTION + LOW → ACCEPT (R6). GOOD + IN_DISTRIBUTION + HIGH → ESCALATE (R4). Everything else falls through to appropriate repair, reject, or safe default escalate rules. No MEDIUM state exists.

## Advanced Questions

**Q18: What is the energy score and why is it approximate here?**
A: The energy score (Liu et al., 2020) = −T·log Σ exp(logitᵢ/T) requires pre-sigmoid logits. TXV applies sigmoid internally, so we use pseudo-logits via inverse sigmoid (logit(p) = log(p/(1−p))). This is explicitly labeled an approximation. Energy is disabled by default.

**Q19: Why is calibration necessary?**
A: The DenseNet was trained for 14-class multi-label classification, not for our specific binary Pneumonia task. Its raw sigmoid scores may not correspond to true empirical frequencies. Platt scaling (logistic regression) and isotonic regression correct this post-hoc.

**Q20: What is ECE?**
A: Expected Calibration Error = Σ_b (n_b/n) × |acc_b − conf_b|. Measures the weighted average gap between predicted confidence and actual fraction of positives per bin. Perfect calibration → ECE=0.

**Q21: What is the F1-optimal threshold?**
A: The classification threshold on [0,1] that maximizes F1 score on the validation set. Used because the dataset is imbalanced (~1.1% positive), so accuracy-maximizing thresholds would predict all-negative.

**Q22: What does the Verification Agent check?**
A: It compares before/after repair signals. It fails (→ ESCALATE) if: repair not applied, confidence data missing, OOD worsened to SEVERE, OOD degraded (ID→BORDERLINE), quality worsened significantly, OR delta_confidence < 0.15 (provisional).

**Q23: What is non-destructive image handling?**
A: The Repair Agent clones the input image before applying any operation. The original tensor/array is never modified. After repair, both the original and repaired images coexist independently.

**Q24: What is CLAHE and why use it for exposure repair?**
A: Contrast Limited Adaptive Histogram Equalization. Regular histogram equalization applies globally, which can over-enhance uniform regions. CLAHE works locally (8×8 tiles) and clips histogram at clip_limit=2.0 to prevent noise amplification. Suitable for X-rays because lung and mediastinum regions have very different intensity ranges.

**Q25: What is NL-means denoising and why use it?**
A: Non-Local Means denoising averages similar patches from across the entire image (not just local neighborhood). Preserves structural edges (lung borders, rib edges) better than Gaussian blur. Important for X-rays where subtle opacity patterns matter.

## Research-Level Questions

**Q26: Why is this a multi-agent system rather than a single end-to-end model?**
A: Separation of concerns → each agent has a single testable responsibility. Transparent rules → each decision is explainable and auditable. Independent calibration → each component can be calibrated and updated without retraining the whole system. Safety by design → explicit safety rules for escalation.

**Q27: What is the key research contribution?**
A: A systematic study of whether combining quality screening, OOD detection, and uncertainty quantification into a multi-agent system can reliably determine when a pneumonia classifier should be trusted, when repair can help, and when human review is required.

**Q28: What are the main limitations of the current OOD detection?**
A: (1) Dev mode: only 2000/78299 training images used; thresholds will change. (2) NIH validation images are in-distribution — cannot compute OOD AUROC from them. (3) No true OOD dataset evaluated. (4) Feature distribution uses Gaussian assumption (may not hold for 1024-dim features). (5) Rare but valid presentations may be flagged as OOD.

**Q29: Why does the project use binary entropy from a single forward pass rather than predictive entropy from Monte Carlo dropout?**
A: Implementation constraint and clarity. MC dropout requires multiple stochastic forward passes and TXV models are not set up with dropout enabled during inference. Binary entropy from a single pass is simple, deterministic, and clearly documented. The module docstring explicitly states what it IS and IS NOT.

**Q30: Can the repair ever make the image worse?**
A: Yes. Unsharp masking can introduce ringing artifacts. NL-means with high `h` can remove fine diagnostic detail. CLAHE can amplify noise in some regions. The Verification Agent checks for quality/OOD degradation and will fail verification (→ ESCALATE) if the repaired image is worse than the original.

## Implementation-Specific Questions

**Q31: Where is the Pneumonia index defined?**
A: `PNEUMONIA_OUTPUT_INDEX = 8` in `models/base_model.py`. Verified by live model probe (2026-09-22). Also dynamically read from `model.pathologies` list during `load_model()`.

**Q32: What is the feature layer used for OOD?**
A: `features.norm5` — the final batch normalization layer before global average pooling in DenseNet-121. After ReLU + adaptive avg pool → 1024-dim vector.

**Q33: How is the repair execution order determined?**
A: Hard-coded in `REPAIR_EXECUTION_ORDER = (EXPOSURE, NOISE, BLUR)`. Deterministic: exposure first improves signal for noise reduction; noise reduction before sharpening prevents amplifying noise.

**Q34: What happens if the model fails mid-pipeline?**
A: Any exception in any agent is caught by the orchestrator. The pipeline returns `PipelineState.ERROR` with `needs_human_review=True`. Predictions are never released on error.

**Q35: How does the dashboard discover OOD artifacts?**
A: Priority order: (1) `CXR_OOD_STATS_DIR` environment variable, (2) `artifacts/ood_dev/`, (3) `artifacts/ood/`, (4) `outputs/ood/` (legacy). The UI shows which mode (dev/full) is active and displays appropriate warnings.

## Safety Questions

**Q36: Why does REJECT withhold the prediction?**
A: A SEVERE OOD image means the model is evaluating something very different from its training distribution. Its output is unreliable. Releasing a prediction from a model that has never seen similar data would be misleading.

**Q37: What is the role of human review?**
A: When the pipeline is not confident enough to release a prediction (ESCALATE or REJECT), it flags the case for a human expert (radiologist). This is the safety net that prevents missed diagnoses and false diagnoses from being acted upon automatically.

**Q38: Why is the project not a clinical system?**
A: Not validated on real clinical data, no regulatory approval, uncalibrated thresholds, NIH labels are NLP-derived (not radiologist-confirmed), single model with no ensemble, no audit by clinical experts, no prospective validation study.

**Q39: What does the disclaimer string say?**
A: `DISCLAIMER` constant from `contracts/common.py` is included in every `PipelineResult`: "Research prototype only. Not intended for clinical diagnosis. Not validated for clinical use."

**Q40: What happens if max_repair_attempts is exceeded?**
A: The pipeline escalates immediately without running the repair → `PipelineState.ESCALATED` with reasoning "Repair attempt limit exceeded." Prevents infinite loops.

---

# PART 26 — PRESENTATION EXPLANATIONS

## 1-Minute (30-second talk + context)

"We built a reliability-checking wrapper around a pretrained chest X-ray AI. Instead of blindly trusting AI predictions, our system checks image quality, detects unusual inputs, and quantifies model uncertainty before deciding whether to accept the prediction, attempt repair, escalate to human review, or reject the case entirely. The system is transparent: every decision is explained and logged."

## 3-Minute

"Pneumonia is a serious condition diagnosed from chest X-rays. Pretrained AI models like DenseNet-121 can detect it automatically, but raw AI predictions are unreliable — the model doesn't know when an image is blurry, from a different scanner, or when the model itself is unsure.

We built a multi-agent reliability system around this model. It has seven specialized agents: Quality checks image sharpness, noise, and exposure. The Base Model extracts a 1024-dimensional feature vector along with the pneumonia score. The OOD Agent computes the Mahalanobis distance to detect distribution shift. The Uncertainty Agent computes binary entropy. The Decision Agent applies a deterministic rule table — severe OOD → reject, quality problems → repair, high uncertainty → escalate.

If repair is recommended, the Repair Agent applies CLAHE, non-local means denoising, or unsharp masking. After repair, fresh inference runs and the Verification Agent checks if reliability improved. The whole system produces one of four actions: ACCEPT, REPAIR→VERIFY, ESCALATE, or REJECT — never silently accepting uncertain predictions.

The research question is whether this structured reliability assessment correctly identifies trustworthy, improvable, and untrustworthy cases."

## 5-Minute

Add: 
- Dataset details (NIH ChestX-ray14, 112k images, 1.1% Pneumonia positive, patient-level splits)
- Mahalanobis distance mathematics
- Why calibration matters (raw sigmoid ≠ calibrated probability)
- Current status (OOD in dev mode, calibration not yet run)
- Provisional vs calibrated thresholds

## 10-Minute

Add:
- Full architecture walkthrough with code examples
- PipelineResult structure
- All 7 rules of Decision Agent with rationale
- Verification conditions in detail
- Current test suite (423 tests, 20 test files)
- Calibration ECE and Brier score explanation
- Research evaluation plan (what will be reported)
- Limitations and future work

---

# PART 27 — EXPLAIN LIKE I'M A BEGINNER

## Level 1: School-Level

Imagine you have an AI program that looks at lung X-rays and says "This person has pneumonia" or "This person doesn't."

But the AI can be wrong sometimes. And worse — it can be confidently wrong. Like a student who guesses on every question with full confidence.

Our project adds a "check before trusting" system around the AI. It asks:
1. Is the X-ray photo clear? (Quality check)
2. Does this X-ray look similar to the ones the AI was trained on? (OOD check)
3. Is the AI sure about its answer? (Uncertainty check)
4. If not, can we improve the photo and try again? (Repair)
5. If we still can't be sure, should a doctor check it instead? (Escalate)

## Level 2: Engineering Student

The project wraps a pretrained DenseNet-121 (from TorchXRayVision, trained on NIH ChestX-ray14) with a 7-agent reliability pipeline:

- **Quality Agent:** Laplacian variance (blur), SNR from MAD (noise), mean intensity (exposure)
- **Base Model Agent:** 18-class sigmoid outputs, Pneumonia at index 8, 1024-dim features from `features.norm5`
- **OOD Agent:** Mahalanobis distance with Cholesky-based numerics on training reference distribution
- **Uncertainty Agent:** Binary entropy + confidence from sigmoid Pneumonia score
- **Decision Agent:** 7-rule precedence table → ACCEPT / REPAIR / ESCALATE / REJECT
- **Repair Agent:** CLAHE + NL-means + unsharp masking (sequential, non-destructive)
- **Verification Agent:** Before/after delta analysis, min_confidence_gain=0.15 threshold

The Orchestrator (Phase 11 `ReliabilityPipeline`) coordinates all 7 agents and the repair loop.

## Level 3: Technical Implementation

[See Parts 1-14 above — full technical details with code references]

## Level 4: Research-Level

The system investigates whether a structured, auditable multi-agent framework can be more reliable than raw model confidence for medical image AI. The research questions are:
- Does quality screening correctly reduce false positives in reliable predictions?
- Does Mahalanobis OOD detection correctly identify distribution-shifted inputs?
- Does the binary entropy uncertainty proxy correlate with actual prediction correctness?
- Does image repair + verification create a meaningful "reliability recovery" pathway?
- What fraction of cases require human review under the current threshold settings?

These questions require calibrated thresholds (Phase 13) and final evaluation on test.csv to answer definitively.

---

# PART 28 — ONE-PAGE QUICK CHEAT SHEET

```
PROJECT: Reliability-Aware Multi-Agent System for CXR Pneumonia Classification
PURPOSE: Determine when to trust, repair, escalate, or reject AI predictions

DATASET:
  NIH ChestX-ray14 | Train:78,299 | Val:17,097 | Test:16,724 | ~1.1% Pneumonia
  Patient-level split | CSVs: data/processed/{train,validation,test}.csv

MODEL:
  densenet121-res224-nih (TorchXRayVision)
  Input: (1,1,224,224) float32, range [-1024,1024]
  Output: 18 sigmoid scores | Pneumonia = index 8
  Features: 1024-dim from features.norm5 (for OOD)

SEVEN AGENTS:
  1. Quality Agent    → Blur(Laplacian<100), Noise(SNR<15dB), Exposure(20-235) → GOOD/DEGRADED/POOR
  2. Base Model Agent → Pneumonia score [0-1] + 1024-dim features
  3. OOD Agent        → Mahalanobis distance → IN_DIST/BORDERLINE/SEVERE
  4. Uncertainty Agent→ confidence=max(p,1-p), entropy=H(p) → LOW / HIGH
  5. Decision Agent   → rule table → ACCEPT/REPAIR/ESCALATE/REJECT
  6. Repair Agent     → CLAHE + NL-means + Unsharp mask (sequential, non-destructive)
  7. Verification Agent→ delta_confidence ≥ 0.15 → VERIFIED or ESCALATE

DECISION RULES (precedence order):
  R1: Severe OOD → REJECT
  R2: Borderline → ESCALATE
  R3: Poor/Degraded quality + ID → REPAIR
  R4: HIGH uncertainty → ESCALATE
  R6: GOOD + ID + LOW → ACCEPT
  R7: Default → ESCALATE

KEY FORMULAS:
  Preprocessing: pixel_txv = pixel × (2048/255) − 1024
  Confidence: conf = max(p, 1−p)
  Binary entropy: H = −p·log₂(p) − (1−p)·log₂(1−p)
  Mahalanobis: d = √(zᵀz), where L·z = (x−μ), Σ=LLᵀ

KEY THRESHOLDS (PROVISIONAL):
  Quality: blur_var<100, SNR<15dB, mean<20 or >235
  OOD (DEV): borderline=56.32, severe=68.44 (2000 train images only)
  Uncertainty: HIGH if conf≤0.60 or entropy≥0.60; LOW if conf≥0.85 and entropy≤0.25
  Verification: min_confidence_gain = 0.15

PIPELINE FLOW:
  Image → Preprocessing → Quality + BaseModel → OOD + Uncertainty → Decision
  → ACCEPT (release) | ESCALATE/REJECT (human review)
  → REPAIR → Fresh inference → Verification → ACCEPT or ESCALATE

COMMANDS:
  Dashboard: streamlit run src/cxr_reliability/dashboard/app.py
  Tests: python -m pytest tests/ -v
  Dev OOD: python scripts/fit_ood_reference.py --dev
  Calibration: python scripts/calibrate_pneumonia.py

CURRENT STATUS:
  ✅ All 7 agents implemented | ✅ Pipeline complete | ✅ Dashboard running
  ⚠️ OOD: dev mode (2000/78299 images)
  ❌ Calibration not run | ❌ Final evaluation pending

MAJOR LIMITATIONS:
  Research prototype only | Not clinically validated | Provisional thresholds
  OOD thresholds are development-only | Calibration incomplete
```

---

# PART 29 — TOP 30 THINGS TO REMEMBER

1. **Pneumonia is at index 8** in DenseNet-121 NIH output (18 labels total). Verified by live probe.

2. **TXV normalization:** `pixel_txv = pixel × (2048/255) − 1024` (range: −1024 to +1024). NOT ImageNet normalization.

3. **The model outputs sigmoid probabilities, NOT raw logits.** TXV applies sigmoid internally.

4. **Feature layer is `features.norm5`**, producing 1024-dim vector after ReLU + global average pool.

5. **Mahalanobis is computed via Cholesky** (L·z = (x−μ), d = ||z||). NOT explicit matrix inversion.

6. **λ=1e−5 regularization** added to covariance for positive definiteness (Tikhonov).

7. **OOD reference is fitted on TRAINING DATA ONLY.** Thresholds selected from VALIDATION DATA. TEST DATA never touched.

8. **Current OOD is DEV MODE** — only 2000/78299 training images. Thresholds are PROVISIONAL.

9. **Uncertainty = binary entropy from sigmoid score**, NOT predictive entropy from MC dropout.

10. **confidence = max(p, 1−p)**, ranges from 0.5 (max uncertainty) to 1.0 (max confidence).

11. **Decision rule precedence:** SEVERE OOD → REJECT first, always.

12. **Two-level uncertainty:** LOW if conf≥0.85 & norm_entropy≤0.25; HIGH otherwise. No MEDIUM state.

13. **REPAIR never automatically means ACCEPT.** Requires Verification Agent to confirm improvement.

14. **Verification threshold (min_confidence_gain=0.15) is PROVISIONAL** — `threshold_is_provisional=True` in every VerificationResult.

15. **Repair execution order is deterministic:** Exposure → Noise → Blur (CLAHE → NL-means → Unsharp).

16. **Repair is non-destructive:** Original image is always cloned. Never modified in-place.

17. **Dataset: 78,299 train | 17,097 val | 16,724 test.** ~1.1% Pneumonia (severely imbalanced).

18. **Patient-level splitting:** All images from one patient in the same split. Prevents patient-level data leakage.

19. **Calibration code exists (Platt + Isotonic + ECE + Brier + F1-optimal) but has NOT been run.** No validation_predictions.csv yet.

20. **Dashboard displays "Pneumonia Model Score"** — explicitly NOT a calibrated clinical probability.

21. **All thresholds in v0_prd_defaults.yaml are either PRD starting points or NULL.** Nothing is empirically calibrated except verification.min_confidence_gain=0.15 (example from PRD).

22. **Energy score is disabled by default and approximate.** Uses pseudo-logit (inverse sigmoid) when enabled. NOT canonical energy score.

23. **Quality Agent uses original pixel scale [0-255]**, NOT the TXV-normalized tensor.

24. **Pipeline uses max_repair_attempts=1 by default.** Prevents infinite repair loops.

25. **Any exception in any agent → PipelineState.ERROR → needs_human_review=True.** Never silent failure.

26. **The system's research question is reliability, not accuracy.** It studies whether the pipeline correctly identifies trustworthy vs. untrustworthy predictions.

27. **Test.csv MUST NOT be touched until final evaluation.** Every threshold and calibration must be frozen first.

28. **Age=414 is a known NIH data quirk** (suspicious age). Flagged but kept (not excluded).

29. **The system is a RESEARCH PROTOTYPE** — no regulatory approval, no clinical validation, not for medical diagnosis.

30. **All decisions are auditable:** Every agent returns a reasoning string, every inference writes an audit JSONL record, every rule has a stable ID (R0–R7).

---

*Generated from actual repository code inspection. Last updated: 2026-09-23.*
*Source of truth: cxr-reliability repository at `C:\Users\PARAS\Desktop\AI SEM 5 B1 G5\PROJECT\cxr-reliability\cxr-reliability`*
