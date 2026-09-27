# Pneumonia Base Model Calibration Documentation (Phase 4.5)

## 1. Executive Summary & Purpose

The **Base Model** integrated in Phase 4 uses TorchXRayVision's pretrained DenseNet-121 (`densenet121-res224-nih`).
This model was trained multi-label across 14 chest X-ray findings on the NIH dataset. Its output layer yields raw sigmoid values in [0, 1] for each condition (with Pneumonia at output index 8).

In our project, we evaluate a single **binary task**:
- `1` = Pneumonia
- `0` = Non-Pneumonia

Because the model was trained under multi-label binary cross-entropy loss over 14 targets, its raw sigmoid scores do **not** directly represent calibrated binary probabilities for Pneumonia versus all Non-Pneumonia images in our validation split. Furthermore, using a default threshold of 0.5 can lead to sub-optimal precision and recall on an imbalanced dataset where Pneumonia prevalence is approximately 1.1%.

Phase 4.5 addresses these issues by:
1. Systematically evaluating classification performance across 200 candidate threshold values (0.01 ... 0.99) on the validation split.
2. Deriving threshold candidates using three principled strategies: **F1-optimal**, **Youden's J**, and **Balanced Accuracy**.
3. Selecting a `project_validation_threshold` (default: F1-optimal) to serve as the positive-class decision operating point.
4. Fitting post-hoc probability calibrators (**Platt scaling** and **Isotonic regression**) to map raw scores to calibrated probabilities without altering prediction rank ordering.

---

## 2. Multi-label vs Binary Task Mapping Rationale

Why is calibration necessary when the model outputs sigmoid scores between 0 and 1?

1. **Class Imbalance & Multi-label Loss Structure**: Multi-label models predict target probabilities independently per label. The background rate of Pneumonia in NIH Data_Entry_2017 is very low (~1.3%). Consequently, uncalibrated raw scores tend to cluster at low numerical ranges (e.g. 0.05 ... 0.40), while true positives may hover around 0.40 ... 0.60.
2. **Threshold Sensitivity**: A standard 0.50 threshold assumes balanced class distributions and symmetric misclassification costs. In chest radiography screening, precision and recall must be carefully traded off based on empirical validation performance.
3. **Calibrated Probabilities for Downstream Agents**: Downstream uncertainty, OOD, and decision agents require true probability estimates (where a score of 0.70 actually corresponds to a 70% empirical positive rate in validation samples).

---

## 3. Threshold Analysis Methodology & Selection Strategies

We evaluate candidate threshold values τ ∈ [0.01, 0.99] in 200 uniform steps.

For each threshold τ, we compute:
- True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
- Precision = TP / (TP + FP)
- Recall / Sensitivity = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- F1 Score = 2 · P · R / (P + R)
- Balanced Accuracy = (Sensitivity + Specificity) / 2
- Youden's J Index = Sensitivity + Specificity - 1

### Strategies Evaluated

1. **`f1_optimal` (Default Selected Strategy)**:
   - Chooses τ* = argmax_τ F1(τ).
   - *Rationale*: Optimal for highly imbalanced binary classification (~1.1% positive rate). Maximises the harmonic mean of precision and recall without being inflated by true negative counts.

2. **`youden` (Youden's J Index)**:
   - Chooses τ* = argmax_τ (Sensitivity(τ) + Specificity(τ) - 1).
   - Equal weight to sensitivity and specificity.

3. **`balanced_accuracy`**:
   - Chooses τ* = argmax_τ BalancedAccuracy(τ).

The chosen threshold is formally designated as `project_validation_threshold`.

---

## 4. Post-Hoc Probability Calibration

We implement two post-hoc calibration methods on validation predictions:

### 4.1 Platt Scaling (Primary Method)
- Fits a univariate logistic regression model on the raw scores: P(y=1 | s) = σ(a·s + b).
- Preserves monotonicity completely.
- Robust against overfitting when positive sample counts are modest (N_pos ≈ 190).

### 4.2 Isotonic Regression (Secondary Comparison)
- Fits a non-parametric, monotonic step function P(y=1 | s) = m(s).
- Flexible, but can overfit step boundaries on small positive subsets.

### 4.3 Calibration Metrics
- **Expected Calibration Error (ECE)**: Computed over 15 equal-width confidence bins:
  ECE = Σ_b (|B_b| / N) · |acc(B_b) - conf(B_b)|
- **Brier Score**: Mean squared error between binary labels and predicted probabilities:
  Brier = (1/N) · Σ (y_i - p_hat_i)²

---

## 5. Data Leakage Isolation Policy

> **CRITICAL**: All threshold evaluation and probability calibrator fitting is performed
> **EXCLUSIVELY on `data/processed/validation.csv`**.

| Split | Role | Used for calibration? |
|---|---|---|
| `train.csv` | Model was pretrained on NIH (not our train split) | Reference only — **never fit** |
| `validation.csv` | Calibration + threshold fitting | **YES — all fitting here** |
| `test.csv` | Final system evaluation | **NEVER touched until calibration is frozen** |

---

## 6. Artifacts & Generated Deliverables

Running `python scripts/calibrate_pneumonia.py` outputs all calibration artifacts to `outputs/calibration/`:

| File | Description |
|---|---|
| `validation_predictions.csv` | Full predictions (17,097 rows) with columns: image_id, patient_id, ground_truth, raw_pneumonia_score |
| `threshold_results.csv` | Full table of 200 evaluated thresholds and their metrics |
| `threshold_analysis.png` | 5-panel visualization (Precision, Recall, F1, Specificity, Balanced Accuracy vs Threshold) |
| `calibration_curve.png` | Reliability diagram & Brier/ECE comparison chart (Raw vs Platt vs Isotonic) |
| `calibration_results.json` | Complete numerical summary: statistics, metrics, candidates, selected threshold |
| `platt_calibrator.joblib` | Serialized Platt scaling model for inference reuse |
| `isotonic_calibrator.joblib` | Serialized Isotonic regression model |

---

## 7. Disclaimers & Limitations

1. **Not a Clinical Threshold**: The parameter `project_validation_threshold` is an empirical decision boundary derived solely for system engineering and evaluation on the NIH validation split. It has **not** been validated in a prospective clinical trial.

2. **Uncalibrated vs Calibrated Terminology**: Raw sigmoid outputs from `BaseModelAgent` must be explicitly referred to as `raw_pneumonia_score`. Transformed values from Platt scaling are `calibrated_probability`. These terms must NOT be used interchangeably.

3. **Small Positive Class**: With ~192 Pneumonia cases in the validation set (1.12%), calibration metrics (especially ECE at the high-probability bins) should be interpreted with caution. Isotonic regression in particular may overfit step boundaries with this sample size.

4. **Pretrained Model Limitations**: The densenet121-res224-nih model was trained on text-mined NIH labels which are known to contain noise. The model's Pneumonia output has not been validated against radiologist consensus.
