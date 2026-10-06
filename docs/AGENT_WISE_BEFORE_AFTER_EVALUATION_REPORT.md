# Agent-Wise Before vs After Evaluation Report
## Reliability-Aware Multi-Agent System for Chest X-Ray Pneumonia Classification

**Evaluation Date:** 2026-10-01 22:43:52  
**Evaluation Script:** `scripts/run_before_after_evaluation.py` (v2.0)  
**Configuration & Thresholds:** `configs/thresholds/v0_prd_defaults.yaml` (Frozen Baseline)  
**Model Architecture:** TorchXRayVision DenseNet-121 (`densenet121-res224-nih`)  
**Operating Decision Threshold:** `0.522161`  
**Verification Safety Margin:** `-0.01` (Non-degradation guard)  

---

### Executive Summary

This report establishes a rigorous, scientifically grounded **Before vs After Evaluation Framework** for the Multi-Agent Chest X-Ray Pneumonia Classification System. 

Rather than treating the multi-agent architecture as an opaque monolith, each agent is independently evaluated with mathematically appropriate metrics:
1. **No Fabricated Metrics:** Standard classification metrics (Accuracy, F1, ROC-AUC) are computed **strictly** where a valid clinical ground truth exists (Base Model, Released Diagnostic Predictions, Paired Pre/Post Repair Cohorts). Non-diagnostic agents (Quality, OOD, Decision, Verification) are evaluated via routing adherence, transition rates, and safety filter properties.
2. **Population Distinction:** The standalone Base Model is evaluated across the complete NIH test set ($N = 16,724$), whereas the Reliability-Aware System selectively releases high-confidence, verified predictions ($N = 7,349$, $43.94\%$ coverage) and diverts ambiguous cases to human review ($N = 9,375$, $56.06\%$).
3. **Paired Repair Fidelity:** Pre-repair and post-repair diagnostic metrics are calculated on the **exact same image cohort** ($N_{before} = N_{after}$), preventing sample selection bias.
4. **Natural vs Synthetic Grounding:** NIH natural repairs consist entirely of blur correction (`unsharp_mask`). Noise reduction (NLMeans) and exposure equalization (CLAHE) are benchmarked using controlled synthetic corruptions with explicit labeling.

---

### 1. Overall System Comparison

| Metric | Standalone Base Model | Reliability-Aware System (Released) | Delta | Population / Nature |
| :--- | :--- | :--- | :--- | :--- |
| **Cohort Size ($N$)** | **16,724** | **7,349** | -9,375 | Full Test Set vs Released High-Trust Subset |
| **Pneumonia Cases ($N_{pos}$)** | 220 | 84 | -136 | Natural NIH minority class (Prevalence: 1.32%) |
| **Non-Pneumonia ($N_{neg}$)** | 16,504 | 7,265 | -9,239 | True Negative population |
| **Accuracy** | **90.94%** | **92.31%** | **+1.37%** | Valid diagnostic metric |
| **Specificity** | 91.86% | 93.15% | +1.29% | Valid diagnostic metric |
| **Precision** | 3.52% | 3.30% | -0.22% | Low due to extreme class imbalance |
| **Recall / Sensitivity** | 22.27% | 20.24% | -2.03% | Selective release withholds uncertain positives |
| **NPV (Negative Predictive Value)** | 98.88% | 99.02% | +0.13% | Valid diagnostic metric |
| **F1 Score** | 0.0608 | 0.0568 | -0.0040 | Valid diagnostic metric |
| **Prediction Coverage** | 100.0% | **43.94%** | -56.06% | Selective prediction release rate |
| **Human Review Rate** | 0.0% | **56.06%** | +56.06% | Safely withheld from automated release |

---

### 2. Dataset and Evaluation Population

- **Dataset:** National Institutes of Health (NIH) ChestX-ray14 Benchmark
- **Split:** Patient-isolated frozen test partition (`data/processed/test.csv`)
- **Total Test Images ($N$):** 16,724
- **Ground Truth Distribution:**
  - Pneumonia (Positive): 220 cases (1.315%)
  - Non-Pneumonia (Negative): 16,504 cases (98.685%)
- **Patient Isolation:** No patient IDs overlap between train, development, and test splits.

---

### 3. Evaluation Methodology

- **Operating Threshold:** `0.522161` calibrated on development split to optimize clinical sensitivity and specificity balance.
- **Selective Classification Protocol:** Predictions are released if and only if:
  1. Quality is acceptable (initially GOOD or successfully repaired to GOOD with verification approval).
  2. Image is IN-DISTRIBUTION in representation space (Mahalanobis distance below borderline boundary).
  3. Prediction uncertainty is validated as LOW or passes verification non-degradation guard.
- **Metric Validity Rule:** No F1, ROC-AUC, or Accuracy is manufactured for intermediate agents.

---

### 4. Standalone Base Model Evaluation

Evaluated directly on the raw DenseNet-121 feature representation and sigmoid outputs across all 16,724 test cases without the reliability layer.

| Metric | Standalone Base Model Value | Sample Size ($N$) | Definition & Clinical Meaning |
| :--- | :--- | :--- | :--- |
| **True Positives (TP)** | 49 | 16,724 | Correctly detected pneumonia cases |
| **True Negatives (TN)** | 15,160 | 16,724 | Correctly identified non-pneumonia images |
| **False Positives (FP)** | 1,344 | 16,724 | Healthy images falsely flagged as pneumonia |
| **False Negatives (FN)** | 171 | 16,724 | Pneumonia cases missed by the model |
| **Accuracy** | **90.94%** | 16,724 | (TP + TN) / Total |
| **Precision** | **3.52%** | 16,724 | TP / (TP + FP) |
| **Recall / Sensitivity** | **22.27%** | 16,724 | TP / (TP + FN) |
| **Specificity** | **91.86%** | 16,724 | TN / (TN + FP) |
| **Negative Predictive Value** | **98.88%** | 16,724 | TN / (TN + FN) |
| **F1 Score** | **0.0608** | 16,724 | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.7016** | 16,724 | Area under ROC curve across continuous scores |
| **PR-AUC** | **0.0281** | 16,724 | Area under Precision-Recall curve |

---

### 5. Quality Agent Evaluation

The Quality Agent assesses input chest X-rays for blur (Laplacian variance), noise (SNR dB), and exposure anomalies (mean intensity and histogram saturation).

- **Initial Image Distribution ($N = 16,724$):**
  - **GOOD:** 3,608 (21.57%)
  - **DEGRADED:** 940 (5.62%)
  - **POOR:** 12,176 (72.81%)
- **Quality-Triggered Repairs Applied:** 12,082 (72.24% of test set)
- **Classification Metrics:** N/A — not a valid metric for this agent (no natural NIH image quality ground truth).

#### Quality 3x3 Transition Matrix (After Repair)

| Before Repair | After: GOOD | After: DEGRADED | After: POOR | Total Repaired |
| :--- | :--- | :--- | :--- | :--- |
| **POOR** | **7,766** | 1,692 | 2,624 | 12,082 |
| **DEGRADED** | 0 | 0 | 0 | 0 |
| **GOOD** | 0 | 0 | 0 | 0 |

#### Quality Recovery Rates:
- **Quality Improvement Rate:** **78.28%** (9,458 images improved)
- **Quality Unchanged Rate:** **21.72%** (2,624 images)
- **Quality Worsening Rate:** **0.00%** (0 images worsened)
- **Poor-to-Good Recovery Rate:** **64.28%** (7,766 / 12,082)
- **Poor-to-Acceptable Recovery Rate:** **78.28%**

---

### 6. Blur Analysis (Before vs After Repair)

In the natural NIH dataset, **100% of quality-triggered repairs were blur-related**, repaired using deterministic Unsharp Masking (`unsharp_mask`).

- **Natural Blur Repairs Attempted:** 12,082
- **Continuous Laplacian Variance Metrics (Targeted Paired Cohort, $N = 22$):**
  - **Mean Variance Before:** 60.95
  - **Mean Variance After:** 118.63
  - **Median Variance Before:** 61.24
  - **Median Variance After:** 128.06
  - **Mean Laplacian Delta:** +57.67
  - **Percentage Improved:** 100.0%
  - **Percentage Unchanged:** 0.0%
  - **Percentage Worsened:** 0.0%

---

### 7. Noise Analysis (Controlled Synthetic Benchmark)

- **Natural NIH Noise Repairs:** **0 / N/A** (Natural NIH images exhibited SNR above the corruption trigger threshold).
- **Controlled Synthetic Noise Benchmark ($N = 3$ images):**
  - Corruption: Additive zero-mean Gaussian noise ($\sigma = 0.35$)
  - Repair Method: Non-Local Means Denoising (`repair_noise`, h=3.0, patch=7, window=21)
  - **Mean SNR Original:** 36.67 dB
  - **Mean SNR Corrupted:** 18.92 dB
  - **Mean SNR Repaired:** 18.92 dB
  - **Mean SNR Recovery Delta:** +-0.00 dB
  - **Percentage SNR Improved:** 0.0%
- *Disclaimer: Synthetic controlled benchmark — not natural NIH test-set results.*

---

### 8. Exposure Analysis (Controlled Synthetic Benchmark)

- **Natural NIH Exposure Repairs:** **0 / N/A** (Exposure anomalies were not the primary trigger in natural NIH).
- **Controlled Synthetic Exposure Benchmark ($N = 3$ images):**
  - Corruption: Exposure shift ($\Delta = -0.3$)
  - Repair Method: Contrast Limited Adaptive Histogram Equalization (`repair_exposure`, CLAHE clip=2.0)
  - **Mean Intensity Original:** 127.02
  - **Mean Intensity Corrupted:** 99.39
  - **Mean Intensity Repaired:** 116.30
  - **Histogram Standard Deviation Repaired:** 58.88
- *Disclaimer: Synthetic controlled benchmark — not natural NIH test-set results.*

---

### 9. OOD Agent Evaluation

The OOD Agent monitors representation-space Mahalanobis distance relative to the NIH training distribution fitted across DenseNet feature activations.

- **Total Test Images ($N$):** 16,724
- **OOD Categorization:**
  - **IN_DISTRIBUTION:** 16,537 (98.88%)
  - **BORDERLINE:** 86 (0.51%)
  - **SEVERE:** 101 (0.60%)
- **Mahalanobis Distance Statistics:**
  - Mean: 27.35 | Median: 26.61 | Std: 4.61
  - Min: 17.77 | Max: 96.42
  - 95th Percentile: 35.54 | 99th Percentile: 42.65
- **Routing Enforcement:**
  - 100% of SEVERE cases (101) routed to **REJECT** -> Human Review.
  - 100% of BORDERLINE cases (86) routed to **ESCALATE** -> Human Review.
- **Classification Metrics:** Standard OOD classification accuracy/F1 is not reported because the in-distribution test set does not provide an external ground-truth OOD label.

---

### 10. Uncertainty Agent Evaluation

Evaluates normalized binary entropy and calibrated confidence across two levels: **LOW** and **HIGH**.

- **Total Test Images ($N$):** 16,724
  - **LOW Uncertainty:** 2,653 (15.86%)
  - **HIGH Uncertainty:** 14,071 (84.14%)
- **Subgroup Analysis:**
  - **LOW Group:** 1,642 released, 1,011 withheld.
  - **HIGH Group:** 5,707 released (post-repair verified), 8,364 withheld.
- *Clinical Note: HIGH uncertainty does NOT imply the model is wrong; rather, it indicates proximity to decision boundaries or feature ambiguity.*

---

### 11. Decision Agent Evaluation

Evaluates routing policies governing image disposition.

| Action | Count | Percentage | Downstream Disposition |
| :--- | :--- | :--- | :--- |
| **ACCEPT** | 7,349 | 43.94% | Released directly to automated reporting |
| **REPAIR** | 0 | 0.00% | Passed to Repair Agent -> Verification Agent |
| **ESCALATE** | 9,274 | 55.45% | Withheld for expert human radiologist review |
| **REJECT** | 101 | 0.60% | Rejected & withheld due to severe OOD anomaly |

- **Final Disposition:**
  - Released Predictions: **7,349 (43.94%)**
  - Withheld Predictions: **9,375 (56.06%)**
- **Classification Metrics:** N/A — not a valid metric for this agent (policy gating contract).

---

### 12. Repair Agent Evaluation

- **Repair Attempts:** 12,176
- **Repairs Applied:** 12,082 (72.24%)
- **Repairs Skipped:** 94
- **Repairs Refused:** 0
- **Quality Improvement Rate:** **78.28%**
- **Quality Unchanged Rate:** **21.72%**
- **Quality Worsening Rate:** **0.00%**
- **Breakdown by Defect Type:**
  - Blur (`unsharp_mask`): 12,082 attempted, 12,082 applied, 9,458 improved.
  - Noise (`nl_means`): 0 natural NIH (evaluated in synthetic benchmark).
  - Exposure (`clahe`): 0 natural NIH (evaluated in synthetic benchmark).

---

### 13. Verification Agent Evaluation

Evaluates safety gatekeeping on repaired images prior to clinical release.

- **Cases Entering Verification:** 12,082
- **Verified (Released):** **6,848 (56.68%)**
- **Escalated (Withheld):** **5,234 (43.32%)**
- **Rejected:** 0
- **Deterministic Verification Failure Reasons ($N = 5,234$ Escalations):**
  1. **Quality Unresolved (POOR -> POOR):** 2,624 cases (50.1%)
  2. **Partial Improvement (POOR -> DEGRADED):** 1,692 cases (32.3%)
  3. **Confidence Degradation Guard Exceeded ($\Delta conf < -0.01$):** 787 cases (15.0%)
- **Confidence Delta Distribution:**
  - Mean: -0.002435 | Median: -0.000042
  - Min: -0.066912 | Max: 0.035933

---

### 14. Paired Before vs After Repair Model Performance

Evaluated strictly on the **exact same image cohort** ($N_{before} = N_{after} = 22$) undergoing blur repair:

| Metric | Before Repair | After Repair | Delta | Sample Size ($N$) |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 59.09% | 59.09% | +0.00% | 22 |
| **Precision** | 75.00% | 75.00% | +0.00% | 22 |
| **Recall / Sensitivity** | 27.27% | 27.27% | +0.00% | 22 |
| **Specificity** | 90.91% | 90.91% | +0.00% | 22 |
| **F1 Score** | 0.4000 | 0.4000 | +0.0000 | 22 |
| **Negative Predictive Value** | 55.56% | 55.56% | +0.00% | 22 |

#### Paired Model Signal Changes:
- **Raw Score Mean Delta:** +0.0017
- **Confidence Mean Delta:** -0.0014
- **Prediction Flips:** 0 total flips (0 pos -> neg, 0 neg -> pos).
- **Prediction Stability Rate:** **100.0%**

---

### 15. Safety Filtering & Error Analysis

The Reliability System withholds ambiguous and corrupted cases from automated release, routing them to human clinical review.

| Error Category | Standalone Base Model Errors | Released System Errors | Errors Safely Withheld | Error Reduction % |
| :--- | :--- | :--- | :--- | :--- |
| **False Positives (FP)** | **1,344** | **498** | **846** | **62.9%** |
| **False Negatives (FN)** | **171** | **67** | **104** | **60.8%** |
| **Total Diagnostic Errors** | **1,515** | **565** | **950** | **62.7%** |

- **Key Takeaway:** The multi-agent reliability layer successfully withheld **846 False Positives (62.9%)** and **104 False Negatives (60.8%)** that would otherwise have been erroneous automated diagnoses.

---

### 16. Confusion Matrices

```
1. Standalone Base Model (Full Test Set, N = 16,724):
                     Predicted Negative    Predicted Positive
True Non-Pneumonia        TN = 15,160           FP = 1,344
True Pneumonia            FN = 171              TP = 49

2. Selective Reliability-Aware System (Released Subset, N = 7,349):
                     Predicted Negative    Predicted Positive
True Non-Pneumonia        TN = 6,767            FP = 498
True Pneumonia            FN = 67               TP = 17

3. Paired Repaired Cohort — Before Repair (N = 22):
                     Predicted Negative    Predicted Positive
True Non-Pneumonia        TN = 10              FP = 1
True Pneumonia            FN = 8              TP = 3

4. Paired Repaired Cohort — After Repair (N = 22):
                     Predicted Negative    Predicted Positive
True Non-Pneumonia        TN = 10              FP = 1
True Pneumonia            FN = 8              TP = 3
```

---

### 17. Visualizations Generated

All visualization artifacts are generated at 300 DPI and stored under `outputs/before_after/`:
- `outputs/before_after/base_model_confusion_matrix.png`
- `outputs/before_after/reliability_system_confusion_matrix.png`
- `outputs/before_after/repair_paired_confusion_matrices.png`
- `outputs/before_after/base_vs_reliability_metrics.png`
- `outputs/before_after/quality_transition_matrix.png`
- `outputs/before_after/decision_routing.png`
- `outputs/before_after/confidence_delta_distribution.png`

---

### 18. Sample-Size Caveats & Statistical Notes

1. **Extreme Imbalance:** In the NIH test set, Pneumonia accounts for only 220 out of 16,724 images (1.315% positive prevalence). Consequently, precision and F1 scores are mathematically constrained by the base rate and should not be compared directly with balanced datasets.
2. **Selective Coverage:** Released system metrics ($N = 7,349$) represent a filtered population and cannot be directly compared to the full test set without noting the 56.06% human review rate.
3. **Paired Repair Sample Size:** The targeted paired benchmark was conducted on $N = 22$ images with balanced positive/negative representation to verify continuous signal changes.

---

### 19. Scientific Limitations

1. **Research Prototype Notice:** This system is an academic research prototype and is **not clinically certified** or cleared for diagnostic use.
2. **Deterministic Sequence:** Repairs are applied in a fixed order (Exposure -> Noise -> Blur), which is an engineering implementation rather than a clinically validated processing pipeline.
3. **Synthetic Grounding:** While blur occurs naturally in NIH, noise and exposure repairs rely on synthetic corruptions due to lack of defect annotations in the public dataset.

---

### 20. Reproducibility Information

- **Git Commit:** Current HEAD
- **Random Seeds:** Pipeline execution: 42; Bootstrap/corruptions: 42
- **Operating Threshold:** `0.522161`
- **Verification Threshold:** `-0.01`
- **Test Set Source:** `data/processed/test.csv` ($N = 16,724$)
- **Full Evaluation Artifacts:** `outputs/evaluation_full_test.csv` & `outputs/before_after_evaluation_summary.json`

---

### 21. Final Findings

1. **Base Model Accuracy:** Pretrained DenseNet-121 achieves 90.94% accuracy with 22.27% recall and 91.86% specificity at threshold 0.5222.
2. **Image Quality Recovery:** Unsharp masking successfully recovers 78.28% of poor-quality images, with 64.28% transitioning from Poor to Good.
3. **Verification Gatekeeping:** The Verification Agent effectively rejects/escalates 43.32% of repairs due to unresolved quality defects or confidence degradation, ensuring only verified improvements are released.
4. **Error Containment:** The reliability architecture withholds 62.9% of False Positives and 60.8% of False Negatives from automated release, elevating released accuracy to **92.31%**.
