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
- **Repair Agent Accounting:**
  - **Repair Agent Entries:** 12,176 (72.81% of test set)
  - **Repairs Applied:** 12,082 (72.24% of test set, 99.23% of entries)
  - **Repairs Skipped:** 94 (0.56% of test set, 0.77% of entries)
  - **Repairs Refused:** 0 (0.00%)
  *(Note: Not all 12,176 candidate entries are classified as repairs applied; 94 images were skipped where repair was not warranted).*
- **Classification Metrics:** N/A — not a valid metric for this agent (no natural NIH image quality ground truth).

#### Quality 3x3 Transition Matrix (After Repair)

| Before Repair | After: GOOD | After: DEGRADED | After: POOR | Total Repaired |
| :--- | :--- | :--- | :--- | :--- |
| **POOR** | **7,766** | 1,692 | 2,624 | 12,082 |
| **DEGRADED** | 0 | 0 | 0 | 0 |
| **GOOD** | 0 | 0 | 0 | 0 |

#### Quality Recovery Rates:
- **Quality Improvement Rate:** **78.28%** (9,458 images improved out of 12,082 applied repairs)
- **Quality Unchanged Rate:** **21.72%** (2,624 images)
- **Quality Worsening Rate:** **0.00%** (0 images worsened)
- **Poor-to-Good Recovery Rate:** **64.28%** (7,766 / 12,082)
- **Poor-to-Acceptable Recovery Rate:** **78.28%**

---

### 6. Blur Analysis (Before vs After Repair)

In the natural NIH dataset, **100% of quality-triggered repairs were blur-related**, repaired using deterministic Unsharp Masking (`unsharp_mask`).

- **Natural Blur Repairs Attempted:** 12,082 applied (out of 12,176 Repair Agent entries)
- **Continuous Laplacian Variance Metrics (Targeted Paired Cohort, $N = 22$):**
  *(Note: The N=22 cohort represents the repaired subset of a targeted 30-image paired benchmark; 8 candidate images were initially GOOD quality and therefore did not undergo repair).*
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
  - Corruption: Additive zero-mean Gaussian noise ($\sigma_{\text{noise}} = \text{severity} \times 50.0 = 0.35 \times 50 = 17.5$)
  - Repair Method: Non-Local Means Denoising (`repair_noise`, $h=3.0$, patch=7, window=21)
  - **Mean SNR Original:** 36.67 dB
  - **Mean SNR Corrupted:** 18.92 dB
  - **Mean SNR Repaired:** 18.92 dB
  - **Mean SNR Recovery Delta:** +0.00 dB (exact floating delta: $-5.24 \times 10^{-7}\text{ dB}$)
  - **Percentage SNR Improved:** 0.0%
- **Audit Verification of the 0.00 dB Recovery Delta:**
  1. **SNR Implementation:** Uses spatial median residual filtering with robust Median Absolute Deviation (MAD, $\sigma = \text{MAD} / 0.6745$) and $20 \log_{10}(\mu_{\text{signal}} / \sigma_{\text{noise}})$. The SNR formulation is mathematically correct and consistent.
  2. **Image Normalization & Range:** Grayscale uint8 array formatted strictly within $[0, 255]$.
  3. **Corrupted Image Input:** Gaussian noise standard deviation is $\sigma = 17.5$ on $[0, 255]$.
  4. **Repaired Image Output & Actual NLMeans Parameters:** `repair_noise` applies OpenCV `cv2.fastNlMeansDenoising` with filter strength $h = 3.0$, `templateWindowSize = 7`, and `searchWindowSize = 21`.
  5. **Root Cause Analysis:** In OpenCV NL-Means, the parameter $h$ regulates filtering strength and must be comparable to the noise standard deviation ($h \approx 10\text{--}15$ for $\sigma = 17.5$). With $h = 3.0$, the patch similarity exponential weights $\exp(-d / h^2)$ decay to zero for all differing patches. As a result, `cv2.fastNlMeansDenoising` modifies zero pixels (mean absolute pixel difference between corrupted and repaired array is identically $0.0000$).
  6. **Audit Conclusion:** The 0.00 dB improvement is a **genuine algorithmic outcome of parameter under-scaling** (conservative $h=3.0$ setting versus $\sigma=17.5$ synthetic noise) rather than an evaluation script bug or measurement failure. The result is reported exactly as measured without artificial modification.
- *Disclaimer: Synthetic controlled benchmark — not natural NIH test-set results.*

---

### 8. Exposure Analysis (Controlled Synthetic Benchmark)

- **Natural NIH Exposure Repairs:** **0 / N/A** (Exposure anomalies were not the primary trigger in natural NIH).
- **Controlled Synthetic Exposure Benchmark ($N = 3$ images):**
  - Corruption: Exposure shift ($\Delta = -0.3$, negative offset of $-30.0$ intensity units)
  - Repair Method: Contrast Limited Adaptive Histogram Equalization (`repair_exposure`, CLAHE clip=2.0, tile grid=$8 \times 8$)
  - **Mean Intensity Original:** 127.02
  - **Mean Intensity Corrupted:** 99.39
  - **Mean Intensity Repaired:** 116.30
  - **Distance of Repaired Mean from Original:** 10.72 intensity units ($|116.30 - 127.02|$; recovered delta from corrupted $= +16.91$)
  - **Dark-Pixel Fraction ($< 10$ intensity):**
    - Original: 7.07% (0.0707)
    - Corrupted: 10.42% (0.1042)
    - Repaired: 10.21% (0.1021)
  - **Bright-Pixel Fraction ($> 245$ intensity):**
    - Original: 0.03% (0.0003)
    - Corrupted: 0.00% (0.0000)
    - Repaired: 0.00% (0.0000)
  - **Histogram Standard Deviation:**
    - Original: 58.31
    - Corrupted: 54.01
    - Repaired: 58.88
- **Scientific Exposure Evaluation Caveat:**
  - *Do not claim successful exposure restoration based only on mean intensity.* While CLAHE successfully shifted the global mean intensity toward baseline ($99.39 \rightarrow 116.30$, closing within 10.72 units of original) and restored overall histogram spread ($\text{std} = 58.88$), the dark-pixel fraction remained essentially at the corrupted level ($10.21\%$ repaired vs $10.42\%$ corrupted, compared to $7.07\%$ original). CLAHE redistributes local histogram contrast across tiles rather than performing true inverse radiometric exposure compensation.
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

Evaluates routing policies governing image disposition. To ensure internal consistency across pipeline stages, initial Decision Agent routing actions are clearly separated from intermediate repair/verification outcomes and final released/withheld dispositions.

#### A. Initial Decision Agent Routing Actions

| Initial Action | Count | Percentage | Description / Downstream Destination |
| :--- | :--- | :--- | :--- |
| **ACCEPT** | 501 | 3.00% | Direct automated release (good quality, in-distribution, low uncertainty) |
| **REPAIR** | 13,001 | 77.74% | Routed to Repair Agent for image restoration |
| **ESCALATE** | 3,121 | 18.66% | Withheld for expert human radiologist review (borderline OOD, high uncertainty, etc.) |
| **REJECT** | 101 | 0.60% | Severe OOD representation anomaly; strictly withheld |
| **Total** | **16,724** | **100.00%** | Full patient-isolated NIH test cohort |

#### B. Repair & Verification Outcomes (Downstream Pipeline Stages)
- **Initial Repair Candidates Routed:** 13,001
- **Repair Agent Entries:** 12,176
  - **Repairs Applied:** 12,082 (72.24% of total test set, 99.23% of entries)
  - **Repairs Skipped:** 94 (0.56% of total test set, 0.77% of entries)
  - **Repairs Refused:** 0 (0.00%)
- **Verification Gatekeeping (Entering Verification: 12,082):**
  - **Verified for Release:** 6,848 (56.68% of entering verification)
  - **Escalated (Withheld):** 5,234 (43.32% of entering verification)

#### C. Final System Disposition

| Final Disposition Category | Count | Percentage | Population Accounting & Source |
| :--- | :--- | :--- | :--- |
| **Direct ACCEPT** | 501 | 3.00% | Initial ACCEPT without requiring repair |
| **Verified Repair Releases** | 6,848 | 40.95% | Post-repair verified and approved for release |
| **Total Released** | **7,349** | **43.94%** | Direct ACCEPT (501) + Verified Repair (6,848) |
| **Final Withheld (Human Review)** | **9,375** | **56.06%** | Total cases withheld from automated release |
| **Total Test Images** | **16,724** | **100.00%** | Released (7,349) + Withheld (9,375) |

- **Exact Population Accounting for Withheld Cases ($N = 9,375$):**
  - Initial ESCALATE: 3,121
  - Initial REJECT: 101
  - Verification Post-Repair Escalations: 5,234
  - Non-repaired / Skipped / Diverted Repair Candidates: 919 (13,001 initial repair minus 12,082 entering verification)
  - Mathematical Reconciliation: $3,121 + 101 + 5,234 + 919 = 9,375$ cases ($100.0\%$ consistency)
- **Classification Metrics:** N/A — not a valid metric for this agent (policy gating contract).

---

### 12. Repair Agent Evaluation

- **Repair Agent Entries:** 12,176
- **Repairs Applied:** 12,082 (72.24% of test set, 99.23% of entries)
- **Repairs Skipped:** 94 (0.56% of test set, 0.77% of entries)
- **Repairs Refused:** 0 (0.00%)
*(Clarification: Not all 12,176 cases are reported as repairs applied; 94 images were evaluated but skipped without modifying pixel data).*
- **Quality Improvement Rate:** **78.28%** (9,458 improved / 12,082 applied)
- **Quality Unchanged Rate:** **21.72%** (2,624 unchanged / 12,082 applied)
- **Quality Worsening Rate:** **0.00%** (0 worsened / 12,082 applied)
- **Breakdown by Defect Type:**
  - Blur (`unsharp_mask`): 12,176 entries (12,082 applied, 94 skipped, 0 refused; 9,458 improved).
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

> **Cohort Specification:**  
> The N=22 cohort is the repaired subset of a targeted 30-image paired benchmark. Eight candidate images were initially GOOD quality and therefore did not undergo repair. The N=22 cohort is not the full 12,082-repair population. The N=22 cohort is suitable for paired feasibility/signal analysis (verifying logit and prediction stability under sharpening) but is not a statistically powered population-level diagnostic evaluation.

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

### 15. Real Image-Level Classification Transition Analysis

To provide undeniable evidence of real classification behavior, predictions and ground-truth labels are evaluated at the individual image level across the frozen NIH test evaluation dataset ($N = 16,724$).

#### A. Image-Level Evaluation Schema
For each image in `outputs/evaluation_full_test.csv`:
- `image_id`: Unique image filename
- `ground_truth`: Clinical label ($0 = \text{Non-Pneumonia}$, $1 = \text{Pneumonia}$)
- `baseline_prediction`: Standalone Base Model prediction $(\text{raw\_model\_score} \ge 0.522161)$
- `baseline_correctness`: $\text{baseline\_prediction} == \text{ground\_truth}$
- `proposed_final_prediction`: Automated prediction when released (`prediction_positive`); otherwise marked `WITHHELD`
- `proposed_correctness`: Evaluated on released subset against ground truth
- `final_action`: Pipeline terminal action (`ACCEPT`, `ESCALATE`, `REJECT`)
- `released / withheld status`: `RELEASED` ($N = 7,349$) or `WITHHELD` ($N = 9,375$)

#### B. Real Image-Level 6-Way Transition Analysis ($N = 16,724$)

| Transition Category | Image-Level Transition Path | Count ($N$) | Percentage of Total ($N = 16,724$) | Clinical & Operational Meaning |
| :--- | :--- | :--- | :--- | :--- |
| **1. Baseline Correct → After Correct** | Correct $\rightarrow$ Released & Correct | **6,784** | **40.56%** | Correct baseline predictions successfully released to automated reporting |
| **2. Baseline Correct → After Incorrect** | Correct $\rightarrow$ Released & Incorrect | **0** | **0.00%** | Zero new diagnostic errors introduced into released predictions |
| **3. Baseline Correct → Withheld** | Correct $\rightarrow$ Safely Withheld | **8,425** | **50.38%** | Correct predictions withheld from automated release (routed to human radiologist review) |
| **4. Baseline Incorrect → After Correct** | Incorrect $\rightarrow$ Released & Correct | **0** | **0.00%** | Zero baseline diagnostic errors converted into correct released predictions |
| **5. Baseline Incorrect → After Incorrect** | Incorrect $\rightarrow$ Released & Incorrect | **565** | **3.38%** | Baseline diagnostic errors remaining in released automated output (498 FP, 67 FN) |
| **6. Baseline Incorrect → Withheld** | Incorrect $\rightarrow$ Safely Withheld | **950** | **5.68%** | Standalone baseline diagnostic errors safely withheld from automated release (846 FP, 104 FN) |
| **Total Cohort Accounting** | **Sum of all 6 transitions** | **16,724** | **100.00%** | **Complete mathematical reconciliation** |

#### C. Summary Classification Validation Counts

| Metric / Count Category | Exact Count | Reference Cohort | Clinical Interpretation |
| :--- | :--- | :--- | :--- |
| **Errors corrected** | **0** | Full Test ($N = 16,724$) | No baseline error was converted to a correct diagnosis by automated processing |
| **Errors remaining** | **565** | Released ($N = 7,349$) | Residual automated errors (498 False Positives, 67 False Negatives) |
| **Errors withheld** | **950** | Withheld ($N = 9,375$) | 846 False Positives (62.9%) and 104 False Negatives (60.8%) safely routed to human review |
| **Correct predictions retained** | **6,784** | Released ($N = 7,349$) | 6,767 True Negatives and 17 True Positives released with high confidence |
| **Correct predictions unnecessarily withheld** | **8,425** | Withheld ($N = 9,375$) | Cost of caution: 8,393 True Negatives and 32 True Positives diverted to human review |
| **New errors introduced** | **0** | Released ($N = 7,349$) | Zero correct baseline predictions corrupted into released errors |
| **Prediction stability on released cohort** | **100.0%** | Released ($N = 7,349$) | Zero prediction flips between base score threshold and released prediction |

> **Explicit Architectural Limitation Statement:**  
> A true image-level Before/After *repair* classification comparison across all 12,082 repairs cannot be reconstructed from the frozen `outputs/evaluation_full_test.csv` because the full-test evaluation pipeline logged only the final representation score (`raw_model_score`) rather than recording side-by-side pre-repair and post-repair scores for every candidate image.  
> A matched image-level pre/post repair diagnostic comparison exists *strictly* in the targeted Paired Repair Benchmark cohort ($N = 22$), where both pre-repair and post-repair forward passes were explicitly logged side-by-side on the exact same images. That cohort demonstrated 0 prediction flips, 0 errors corrected, and 0 new errors introduced (100.0% stability).

---

### 16. Agent-Wise Before vs After Effectiveness

| Agent | Before (Input / Baseline) | After (System Output) | Real Evidence | Result | Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Model** | Standalone DenseNet-121 on full test set ($N = 16,724$) | Diagnostic predictions released by system ($N = 7,349$) | Acc: 90.94% $\rightarrow$ 92.31% (+1.37%)<br>Spec: 91.86% $\rightarrow$ 93.15% (+1.29%)<br>Sens: 22.27% $\rightarrow$ 20.24% (-2.03%)<br>F1: 0.0608 $\rightarrow$ 0.0568 (-0.0040) | Selective release improves accuracy and specificity on released subset | Improvement stems entirely from selective routing/filtering, NOT from altered model weights |
| **Quality Agent** | Raw test images: 3,608 Good, 940 Degraded, 12,176 Poor ($N = 16,724$) | Post-repair: 7,766 Good, 1,692 Degraded, 2,624 Poor ($N = 12,082$) | 78.28% improved (9,458/12,082)<br>21.72% unchanged (2,624/12,082)<br>0.00% worsened (0/12,082)<br>64.28% Poor $\rightarrow$ Good | High recovery of Laplacian sharpness via deterministic unsharp masking | Natural NIH failures are 100% blur; lacks natural ground-truth quality labels |
| **OOD Agent** | Unmonitored representation space (all 16,724 treated equally) | Mahalanobis distance gating: 16,537 ID, 86 Borderline, 101 Severe | 100% of Severe (101) $\rightarrow$ REJECT<br>100% of Borderline (86) $\rightarrow$ ESCALATE<br>Routing adherence: 100.0% | Zero severe representation anomalies allowed into automated release | Natural NIH lacks external OOD labels; true OOD AUROC requires separate OOD benchmark |
| **Uncertainty Agent** | Unstratified continuous scores (overall error rate: 9.06%) | Stratified cohorts:<br>LOW: $N = 2,653$<br>HIGH: $N = 14,071$ | LOW error rate: 0.45% (Acc: 99.55%)<br>HIGH error rate: 10.68% (Acc: 89.32%)<br>All 1,344 baseline FP in HIGH cohort | Uncertainty level directly corresponds to observed classification reliability | 84.14% of cases fall into HIGH uncertainty; withholding all would collapse coverage |
| **Decision Agent** | Monolithic release (100% automated release, 1,515 errors) | 4-way routing policy:<br>ACCEPT: 501<br>REPAIR: 13,001<br>ESCALATE: 3,121<br>REJECT: 101 | 950 baseline errors withheld (846 FP, 104 FN)<br>8,425 baseline correct withheld<br>Released: 7,349 (43.94%) | Shields automated output from 62.7% of standalone baseline errors | Substantial human review burden (56.06% withheld), withholding 55.4% of correct diagnoses |
| **Repair Agent** | Blurred radiographs (Laplacian mean: 60.95 on Paired $N = 22$) | Sharpened radiographs (Laplacian mean: 118.63) | Mean Laplacian $\Delta = +57.67$<br>Mean score $\Delta = +0.0017$<br>Prediction flips: 0 (100% stability)<br>Acc: 59.09% $\rightarrow$ 59.09% | Restores physical sharpness while strictly preserving model logit stability | No measurable diagnostic improvement demonstrated with current benchmark |
| **Verification Agent** | 12,082 repaired images entering without post-repair safety check | Verified: 6,848 (56.68%)<br>Escalated: 5,234 (43.32%)<br>Rejected: 0 (0.00%) | Escalation causes: 2,624 Poor, 1,692 Degraded, 787 Conf Guard<br>382 baseline errors withheld<br>4,852 correct withheld | Safely intercepts incomplete repairs and confidence degradation | Uses surrogate quality and confidence guards; withholds 4,852 correct cases alongside 382 errors |

#### Deep-Dive Agent Breakdown

##### 1. Base Model (Ground-Truth Classification Performance)
Evaluated across all 16,724 patient-isolated test radiographs:
- **Accuracy:** 90.94% (15,209 / 16,724)
- **Precision:** 3.52% (49 / 1,393)
- **Recall / Sensitivity:** 22.27% (49 / 220)
- **Specificity:** 91.86% (15,160 / 16,504)
- **F1 Score:** 0.0608
- **Negative Predictive Value (NPV):** 98.88% (15,160 / 15,331)
- **ROC-AUC:** 0.7016
- **PR-AUC:** 0.0281
*Clinical Note: Standalone precision and F1 are constrained by the 1.32% base prevalence in the natural NIH population.*

##### 2. Quality Agent (Image Quality Recovery)
Evaluates spatial frequency blur, high-frequency noise, and radiometric exposure without fabricating classification metrics:
- **Quality Improvement Rate:** **78.28%** (9,458 improved / 12,082 repairs applied)
- **Quality Unchanged Rate:** **21.72%** (2,624 unchanged / 12,082 repairs applied)
- **Quality Worsening Rate:** **0.00%** (0 worsened / 12,082 repairs applied)
- **Poor-to-Good Transition Rate:** **64.28%** (7,766 / 12,082)
- **Poor-to-Degraded Transition Rate:** **14.00%** (1,692 / 12,082)
- **Poor-to-Poor (Unresolved) Rate:** **21.72%** (2,624 / 12,082)
- **Laplacian Variance Delta (Paired $N = 22$):** Mean $60.95 \rightarrow 118.63$ ($\Delta = +57.67$, 100% improved)

##### 3. OOD Agent (Representation Distance & Routing Adherence)
Monitors Mahalanobis distance fitted on DenseNet penultimate activations:
- **IN_DISTRIBUTION:** 16,537 images (98.88%)
- **BORDERLINE:** 86 images (0.51%) — 100% routed to ESCALATE (0 released)
- **SEVERE:** 101 images (0.60%) — 100% routed to REJECT (0 released)
- **Routing Adherence:** **100.0%** (187 / 187 non-ID images safely withheld)
*Limitation: True OOD detection accuracy/AUROC cannot be determined on the in-distribution NIH test set; evaluation requires an external out-of-distribution benchmark dataset.*

##### 4. Uncertainty Agent (Empirical Reliability by Uncertainty Stratification)
Compares actual ground-truth classifications across uncertainty levels to determine whether uncertainty corresponds to observed reliability:
- **LOW Uncertainty Cohort ($N = 2,653$, $15.86\%$):**
  - **Accuracy:** **99.55%** (2,641 correct / 2,653)
  - **Error Rate:** **0.45%** (12 errors: 0 False Positives, 12 False Negatives)
  - **Released Subset ($N = 1,642$):** Accuracy **99.57%**, 0 False Positives, 7 False Negatives
- **HIGH Uncertainty Cohort ($N = 14,071$, $84.14\%$):**
  - **Accuracy:** **89.32%** (12,568 correct / 14,071)
  - **Error Rate:** **10.68%** (1,503 errors: 1,344 False Positives, 159 False Negatives)
  - **Released Subset ($N = 5,707$ post-repair verified):** Accuracy **90.22%**, 498 False Positives, 60 False Negatives
- **Empirical Validation:** The Uncertainty Agent successfully segregates low-risk from high-risk diagnostic regimes. The error rate in the HIGH uncertainty group ($10.68\%$) is **23.7 times higher** than in the LOW uncertainty group ($0.45\%$), and 100% of standalone False Positives (1,344) reside in the HIGH uncertainty group.

##### 5. Decision Agent (Routing Policy & Error Segregation)
Evaluates initial policy routing before downstream repair and verification:
- **Initial Action Counts:** ACCEPT = 501, REPAIR = 13,001, ESCALATE = 3,121, REJECT = 101
- **Baseline Error Routing:** Out of 1,515 standalone errors, 1,511 ($99.74\%$) were initially diverted away from direct release (only 4 False Negatives in 501 direct ACCEPT).
- **Final Error Containment:** Downstream pipeline safely withheld **950 baseline errors** (846 False Positives, 104 False Negatives).
- **Withheld Correct Predictions:** Diverted **8,425 correct baseline predictions** to human review.

##### 6. Repair Agent (Paired Feasibility Cohort, $N = 22$)
Evaluates paired ground-truth correctness before and after unsharp masking on the exact same images:
- **Quality Improvement:** Mean Laplacian variance $+57.67$ (100% improved)
- **Confidence Delta:** Mean $\Delta = -0.0014$, Median $\Delta = +0.00006$
- **Raw Score Delta:** Mean $\Delta = +0.0017$, Median $\Delta = +0.00022$
- **Prediction Flips:** **0 total flips** (0 positive $\rightarrow$ negative, 0 negative $\rightarrow$ positive)
- **Diagnostic Correctness Before vs After:**
  - Accuracy: 59.09% (13/22) $\rightarrow$ 59.09% (13/22) ($\Delta = 0.00\%$)
  - Precision: 75.00% (3/4) $\rightarrow$ 75.00% (3/4) ($\Delta = 0.00\%$)
  - Sensitivity: 27.27% (3/11) $\rightarrow$ 27.27% (3/11) ($\Delta = 0.00\%$)
  - Specificity: 90.91% (10/11) $\rightarrow$ 90.91% (10/11) ($\Delta = 0.00\%$)
  - F1 Score: 0.4000 $\rightarrow$ 0.4000 ($\Delta = 0.0000$)
- **Empirical Finding:** Unsharp masking restores physical image sharpness without destabilizing model predictions, but **does not alter diagnostic accuracy on this $N=22$ cohort**.

##### 7. Verification Agent (Gatekeeping & Degradation Prevention)
Evaluates safety gatekeeping rules on 12,082 entering repairs:
- **Verified for Release:** 6,848 (56.68%)
- **Escalated (Withheld):** 5,234 (43.32%)
- **Escalation Breakdown:**
  - Quality unresolved (Poor $\rightarrow$ Poor): 2,624 cases (50.1%)
  - Quality partial recovery (Poor $\rightarrow$ Degraded): 1,692 cases (32.3%)
  - Confidence degradation guard exceeded ($\Delta conf < -0.01$): 787 cases (15.0%)
  - Secondary/other escalations: 131 cases (2.5%)
- **Baseline Errors Withheld by Verification:** **382 errors** (prevented from entering released pool)
- **Baseline Correct Predictions Withheld:** **4,852 correct cases**
- **New Errors Prevented from Release:** 0 new errors were introduced by repair, but verification successfully blocked 382 baseline errors from automated release.

---

### 17. Confusion Matrices

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

### 18. Visualizations Generated

All visualization artifacts are generated at 300 DPI and stored under `outputs/before_after/`:
- `outputs/before_after/base_model_confusion_matrix.png`
- `outputs/before_after/reliability_system_confusion_matrix.png`
- `outputs/before_after/repair_paired_confusion_matrices.png`
- `outputs/before_after/base_vs_reliability_metrics.png`
- `outputs/before_after/quality_transition_matrix.png`
- `outputs/before_after/decision_routing.png`
- `outputs/before_after/confidence_delta_distribution.png`

---

### 19. Sample-Size Caveats & Statistical Notes

1. **Extreme Imbalance:** In the NIH test set, Pneumonia accounts for only 220 out of 16,724 images (1.315% positive prevalence). Consequently, precision and F1 scores are mathematically constrained by the base rate and should not be compared directly with balanced datasets.
2. **Selective Coverage:** Released system metrics ($N = 7,349$) represent a filtered population and cannot be directly compared to the full test set without noting the 56.06% human review rate.
3. **Paired Repair Sample Size:** The N=22 cohort is the repaired subset of a targeted 30-image paired benchmark. Eight candidate images were initially GOOD quality and therefore did not undergo repair. The N=22 cohort is not the full 12,082-repair population. The N=22 cohort is suitable for paired feasibility/signal analysis (verifying logit and prediction stability under sharpening) but is not a statistically powered population-level diagnostic evaluation.

---

### 20. Scientific Limitations

1. **Research Prototype Notice:** This system is an academic research prototype and is **not clinically certified** or cleared for diagnostic use.
2. **Deterministic Sequence:** Repairs are applied in a fixed order (Exposure -> Noise -> Blur), which is an engineering implementation rather than a clinically validated processing pipeline.
3. **Synthetic Grounding:** While blur occurs naturally in NIH, noise and exposure repairs rely on synthetic corruptions due to lack of defect annotations in the public dataset.

---

### 21. Reproducibility Information

- **Git Commit:** Current HEAD
- **Random Seeds:** Pipeline execution: 42; Bootstrap/corruptions: 42
- **Operating Threshold:** `0.522161`
- **Verification Threshold:** `-0.01`
- **Test Set Source:** `data/processed/test.csv` ($N = 16,724$)
- **Full Evaluation Artifacts:** `outputs/evaluation_full_test.csv` & `outputs/before_after_evaluation_summary.json`

---

### 22. Final Findings

To maintain clinical and mathematical validity, findings are strictly separated by architectural component and operational role:

#### A. Base-Model Diagnostic Performance (Full Test Population, N = 16,724)
- Pretrained TorchXRayVision DenseNet-121 achieves an overall accuracy of **90.94%**, specificity of **91.86%**, and sensitivity (recall) of **22.27%** at the calibrated operating threshold of `0.522161`.
- Because of severe class imbalance in the NIH test set (positive prevalence: 1.32%, 220 pneumonia cases), standalone precision is **3.52%** and F1 is **0.0608**, with continuous ROC-AUC of **0.7016** and PR-AUC of **0.0281**.

#### B. Quality-Agent & Repair Image-Quality Behavior
- **Quality Assessment:** Evaluated across all 16,724 test images, the Quality Agent categorized 72.81% (12,176) as POOR, 5.62% (940) as DEGRADED, and 21.57% (3,608) as GOOD. In natural NIH images, quality failures were 100% blur-dominated.
- **Repair Application:** Out of 12,176 Repair Agent entries, 12,082 repairs were applied via unsharp masking, 94 were skipped, and 0 were refused.
- **Restoration Efficacy:** Deterministic unsharp masking improved Laplacian variance in **78.28%** (9,458) of repaired images, with **64.28%** (7,766) recovering from POOR to GOOD quality and 0.00% worsening.
- **Controlled Synthetic Benchmarks:** CLAHE shifted global mean intensity (+16.91 units) on synthetic underexposure, while NL-Means at conservative $h=3.0$ produced a 0.00 dB SNR delta against $\sigma=17.5$ noise due to filter strength under-scaling.

#### C. Out-of-Distribution (OOD) Routing
- Mahalanobis distance profiling in DenseNet feature activation space flagged **101 SEVERE** outliers (0.60%) and **86 BORDERLINE** cases (0.51%).
- Safety policy gating was 100% compliant: 100% of SEVERE cases were routed to **REJECT** (withheld from release), and 100% of BORDERLINE cases were routed to **ESCALATE** (withheld for human review).

#### D. Uncertainty Routing
- The Uncertainty Agent classified 2,653 images (15.86%) as LOW uncertainty and 14,071 (84.14%) as HIGH uncertainty.
- In the released LOW-uncertainty cohort ($N = 1,642$), automated diagnostic accuracy reached **99.57%** with 0 false positives, demonstrating effective isolation of high-trust diagnostic regimes.

#### E. Selective-Release Behavior & Error Containment
- Overall, the multi-agent system released **7,349 predictions** (43.94% coverage) and safely withheld **9,375 cases** (56.06% human review rate).
- Released prediction accuracy increased to **92.31%** (+1.37% over standalone base model), and specificity increased to **93.15%** (+1.29%).
- Of the errors observed in the standalone full-test predictions, **846 false-positive cases (62.9%)** and **104 false-negative cases (60.8%)** were not present in the automatically released subset because those cases were withheld from automated release.

#### F. Paired Repair Model-Performance Benchmark (Targeted Cohort, N = 22)
- Evaluated on the exact same paired cohort before and after blur repair ($N_{before} = N_{after} = 22$), diagnostic metrics remained identical (Accuracy: 59.09%, Precision: 75.00%, Recall: 27.27%, Specificity: 90.91%, F1: 0.4000) with 0 prediction flips (100% prediction stability).
- Continuous signals exhibited minimal mean score change (+0.0017) and confidence change (-0.0014) despite substantial image sharpening ($\Delta\text{Laplacian} = +57.67$).
- **Diagnostic Conclusion:** Repair acts as an image-quality restoration step that maintains model stability without destabilizing predictions. Based on this $N=22$ feasibility cohort, **repair cannot be claimed to improve pneumonia diagnostic performance**. Expanding to larger cohorts (e.g., $N=200$) is required for powered clinical evaluation.

---

### 23. What Actually Improved? (Evidence Synthesis)

| System Component / Dimension | Before Condition | After Condition | Measured Change | Evidence Status |
| :--- | :--- | :--- | :--- | :--- |
| **Released Prediction Accuracy** | 90.94% (Full Test, $N=16,724$) | 92.31% (Released, $N=7,349$) | +1.37% accuracy, +1.29% specificity | **✓ Demonstrated improvement** |
| **Diagnostic Error Containment** | 1,515 baseline errors (Full Test) | 565 released errors | 950 errors withheld (846 FP, 104 FN) | **✓ Demonstrated improvement** |
| **Low-Uncertainty Reliability Isolation** | Unstratified error rate: 9.06% | LOW Uncertainty error rate: 0.45% | Error rate reduced by 20.1x (99.55% accuracy) | **✓ Demonstrated improvement** |
| **Image Sharpness (Laplacian Variance)** | 60.95 (Paired Cohort, $N=22$) | 118.63 (Paired Cohort, $N=22$) | +57.67 Laplacian variance (+94.6%) | **✓ Demonstrated improvement** |
| **Quality State Transitions (Poor $\rightarrow$ Good)** | 12,176 Poor-quality test images | 7,766 recovered to Good quality | 64.28% Poor $\rightarrow$ Good, 0.00% worsening | **✓ Demonstrated improvement** |
| **Prediction Stability Under Repair** | Pre-repair predictions ($N=22$) | Post-repair predictions ($N=22$) | 0 prediction flips out of 22 (100% stability) | **≈ No measurable change / preserved** |
| **Base Model Weights & Architecture** | DenseNet-121 (AUROC: 0.7016) | DenseNet-121 (AUROC: 0.7016) | Frozen pretrained model unchanged | **≈ No measurable change / preserved** |
| **Repair Diagnostic Accuracy Gain (Paired $N=22$)** | 59.09% accuracy (13/22 correct) | 59.09% accuracy (13/22 correct) | 0.00% accuracy delta, 0 errors corrected | **≈ No measurable change / preserved** |
| **Synthetic Noise Denoising (NLMeans $h=3.0$)** | 18.92 dB (Corrupted) | 18.92 dB (Repaired) | +0.00 dB SNR delta (parameter under-scaled) | **≈ No measurable change / preserved** |
| **Synthetic Exposure Radiometric Restoration** | 99.39 mean intensity (Corrupted) | 116.30 mean intensity (Repaired) | Mean +16.91 units; dark pixel frac 10.21% vs 10.42% | **⚠ Insufficient evidence** |
| **True External OOD Detection AUROC** | Full NIH test set | Mahalanobis gating | 100% routing compliance on internal NIH | **⚠ Insufficient evidence** |
| **Population-Level Paired Repair Diagnostic Impact** | Full 12,082 repairs | Full 12,082 repairs | Pre-repair scores not logged in frozen CSV | **⚠ Insufficient evidence** |
| **Automated Diagnostic Coverage** | 100.0% Automated | 43.94% Automated | -56.06% coverage (8,425 correct cases withheld) | **✗ Worsened** |
| **Positive Class Recall / Sensitivity** | 22.27% (49/220 Pneumonia) | 20.24% (17/84 Released Pneumonia) | -2.03% sensitivity (136 positives withheld) | **✗ Worsened** |

> **Faculty Validation Conclusion:**  
> **"No measurable improvement demonstrated with the current benchmark"** in intrinsic classifier weights, repair-induced diagnostic accuracy on paired radiograph cohorts, or NL-Means denoising under the current $h=3.0$ configuration.  
> Real demonstrated improvements are strictly confined to **selective prediction release (error containment via risk-based withholding)** and **deterministic image quality sharpening (Laplacian variance recovery without model destabilization)**. All claims of clinical efficacy are bounded by the 56.06% human review rate and the 1.32% positive prevalence of the NIH benchmark.
