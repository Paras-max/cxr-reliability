"""
Report text content definitions.
Part 4: Text sections 17 to 26 and Faculty Defense Q&A (Questions 1 to 30).
"""

from __future__ import annotations

SECTION_17_TITLE = "Section 17 — Overall Before vs After Comprehensive Comparison"
SECTION_17_BODY = """
The table below presents a rigorous, multidimensional comparison between the conventional baseline pipeline (unconditional DenseNet-121) and the proposed reliability-aware multi-agent architecture.

CRITICAL STATISTICAL DISTINCTION:
Baseline accuracy (90.9412%) is evaluated on the ENTIRE uncurated test cohort (N=16,724). Released subset accuracy (92.3119%) is evaluated exclusively on the SELECTIVELY RELEASED cohort (N=7,349, 43.94% coverage). These two figures represent fundamentally different populations. The multi-agent system does not claim to make DenseNet-121 more accurate; rather, it purifies the released cohort by withholding high-risk, ambiguous, and corrupted inputs.

COMPREHENSIVE BEFORE vs AFTER MATRIX (17 CORE DIMENSIONS):

1. Architecture:
   - Before: Single-stage monolithic feedforward CNN (DenseNet-121).
   - After: 7-Agent supervisory architecture with quality, distribution, calibration, decision, repair, and verification agents.
   - Evidence: End-to-end multi-agent pipeline implementation.
   - Interpretation: Transforms black-box inference into modular, inspectable clinical workflow.

2. Image Quality Handling:
   - Before: Zero inspection. All images processed regardless of optical degradation.
   - After: Automated screening across Laplacian variance (blur), SNR (noise), and mean intensity (exposure).
   - Evidence: 12,176 poor-quality images identified out of 16,724 test cases.
   - Interpretation: Prevents garbage-in, garbage-out failure modes.

3. Blur Degradation:
   - Before: Blurred images processed unconditionally, risking spurious feature activations.
   - After: Flagged when Laplacian variance < 100.0; targeted unsharp masking applied; verified post-repair.
   - Evidence: Blur tuning sweep (100% variance gain) and paired blur cohort (variance 58.31 -> 116.02).
   - Interpretation: Restores anatomical edge sharpness prior to diagnostic reliance.

4. Noise Degradation:
   - Before: High sensor noise passed directly, potentially simulating alveolar infiltrates.
   - After: Flagged when SNR < 15 dB; Non-Local Means denoising applied conditionally.
   - Evidence: NLMeans tuning sweep identified h=7.0 to h=10.0 as effective range (+0.57 to +6.60 dB SNR gain).
   - Interpretation: Suppresses stochastic noise while monitoring for over-smoothing hazards.

5. Exposure Abnormalities:
   - Before: Extreme over/underexposed radiographs forced through CNN.
   - After: Flagged outside [20, 235] range; CLAHE contrast redistribution applied conditionally.
   - Evidence: CLAHE tuning sweep recovered intensity from 115.42 to 123.95 toward middle gray.
   - Interpretation: Prevents dynamic range clipping in critical retrocardiac and apical zones.

6. Disease Classification:
   - Before: Standalone DenseNet-121 pneumonia score at threshold 0.522161.
   - After: Standalone DenseNet-121 pneumonia score at threshold 0.522161 (identical frozen weights).
   - Evidence: Zero modifications made to model parameters or architecture.
   - Interpretation: Diagnostic classification engine remains completely unchanged.

7. Probability Calibration:
   - Before: Raw uncalibrated network sigmoid scores (ECE ~ 0.051).
   - After: Post-hoc Platt Scaling calibrated probabilities (ECE ~ 0.015).
   - Evidence: Calibration fit on validation cohort with logistic parameters A and B.
   - Interpretation: Probabilities reflect true empirical clinical event frequencies.

8. Uncertainty Representation:
   - Before: Unrepresented. Model outputs treated with equal clinical weight.
   - After: Explicit confidence (max(p, 1-p)) and normalized Shannon entropy (H_norm).
   - Evidence: Low uncertainty cohort achieved 99.57% accuracy vs 90.22% for high uncertainty.
   - Interpretation: Provides an actionable signal for automated triage.

9. Out-of-Distribution Screening:
   - Before: Unmonitored. Severe demographic/device outliers receive forced predictions.
   - After: 1024-D Mahalanobis distance via Cholesky decomposition; 3-tier stratification (ID, Borderline, Severe).
   - Evidence: 101 severe OOD cases (100%) rejected; 86 borderline cases (100%) escalated.
   - Interpretation: Prevents silent failures on distributional anomalies.

10. Decision Routing:
    - Before: Binary forced classification on 100% of inputs.
    - After: Deterministic, auditable rule hierarchy (ACCEPT, REPAIR, ESCALATE, REJECT).
    - Evidence: Initial routing: 501 Accept, 13,001 Repair, 3,121 Escalate, 101 Reject.
    - Interpretation: Replaces unconditional release with transparent clinical triage.

11. Targeted Image Repair:
    - Before: Absent. No restoration mechanisms exist.
    - After: Targeted CLAHE, NLMeans, and Unsharp Masking applied conditionally.
    - Evidence: Evaluated on 12,082 repairs across full test cohort.
    - Interpretation: Salvages repairable images without human radiologist intervention.

12. Post-Repair Verification:
    - Before: Absent.
    - After: 5 mandatory safety gates (execution, label stability, OOD ID, quality recovery, non-degradation).
    - Evidence: 6,848 repairs verified and released; 5,234 escalated due to gate failures.
    - Interpretation: Ensures repair failure does not compromise patient safety.

13. Human Clinical Review:
    - Before: Absent. System acts as an unassisted autonomous predictor.
    - After: Formal escalation pathway for 9,375 cases (56.06% of test cohort).
    - Evidence: Deterministic triage routing directly into radiologist review queues.
    - Interpretation: Establishes a realistic human-in-the-loop clinical partnership.

14. Automated Coverage / Selective Release:
    - Before: 100.0% coverage (16,724 / 16,724 images released).
    - After: 43.94% coverage (7,349 / 16,724 images released; 9,375 withheld).
    - Evidence: Full test evaluation results.
    - Interpretation: High-coverage automated release reserved strictly for high-reliability cases.

15. Total Error Containment:
    - Before: 1,515 baseline diagnostic errors released into clinical workflow.
    - After: 565 errors released; 950 errors safely withheld from automatic release.
    - Evidence: 846 false positives and 104 false negatives contained (62.71% overall containment).
    - Interpretation: Dramatically reduces erroneous automated diagnostic reports.

16. Image-Quality Recovery:
    - Before: 0% recovery (unmodified inputs).
    - After: 78.28% objective quality improvement rate across 12,082 evaluated repairs.
    - Evidence: 7,766 Poor -> Good transitions and 1,692 Poor -> Degraded transitions; 0.0% worsening.
    - Interpretation: Demonstrates strong physical signal restoration capability.

17. Diagnostic Correctness from Repair:
    - Before: N/A.
    - After: 0 diagnostic errors corrected in authoritative N=35 paired diagnostic benchmark (100% stability).
    - Evidence: 17 Corr -> Corr, 18 Inc -> Inc, 0 Inc -> Corr, 0 Corr -> Inc.
    - Interpretation: Empirically establishes that image repair restores quality, not frozen CNN accuracy.
"""

SECTION_18_TITLE = "Section 18 — What Did Each Agent Actually Improve? (Detailed Narrative)"
SECTION_18_BODY = """
Rather than presenting a superficial scorecard, this narrative documents the tangible, experimentally verified contribution of each individual agent within the supervisory framework:

1. Quality Agent:
   - What it improved: Objective technical visibility. Prior to this agent, the system was blind to optical defects. The Quality Agent successfully identified 12,176 poor-quality radiographs and provided the necessary diagnostic signal to trigger restorative algorithms.
   - What it did not improve: It does not classify pathology or improve model accuracy.

2. Base Model Agent (DenseNet-121):
   - What it improved: Maintained stable diagnostic classification across all tests. Provided the disease prediction and rich 1024-dimensional semantic embeddings.
   - What it did not improve: Its standalone error rate (1,515 errors on test set) remained identical because its weights were frozen.

3. Calibration & Uncertainty Agent:
   - What it improved: Probability trustworthiness and uncertainty stratification. Reduced ECE from ~0.051 to ~0.015 via Platt Scaling. Stratified the test cohort such that low-uncertainty predictions achieved 99.57% accuracy.
   - What it did not improve: Calibration did not improve discriminative ranking (ROC-AUC remained 0.7016). High uncertainty does not mean a case is definitely incorrect.

4. Out-of-Distribution (OOD) Agent:
   - What it improved: Distributional safety screening. Prevented 101 severe distributional outliers from releasing unverified predictions (100% rejection rate) and escalated 86 borderline cases.
   - What it did not improve: Does not determine pneumonia presence or absence.

5. Decision Agent:
   - What it improved: Deterministic governance. Translated disparate multi-agent signals into four auditable actions, eliminating ad-hoc thresholding.
   - What it did not improve: Does not generate predictions or modify images.

6. Image Repair Agent:
   - What it improved: Objective image metrics. Successfully improved image quality in 78.28% of evaluated cases (7,766 Poor -> Good transitions; mean blur variance +57.71).
   - What it did not improve: Did NOT improve diagnostic correctness for the frozen DenseNet-121 (0 errors corrected across the paired N=35 cohort).

7. Verification Agent:
   - What it improved: Failure containment. Acted as a definitive safety firewall, intercepting 5,234 post-repair radiographs that failed quality recovery or non-degradation gates and preventing automated release.
   - What it did not improve: Does not repair images itself; it evaluates and gates.

8. Selective Release & Human Review:
   - What it improved: System-level clinical safety. Contained 950 baseline diagnostic errors from automated release, elevating released cohort accuracy from 90.94% to 92.31%.
   - What it did not improve: Does not correct withheld cases autonomously; relies on human expert triage.
"""

SECTION_19_TITLE = "Section 19 — How Each Agent Was Evaluated (Task-Specific Methodology)"
SECTION_19_BODY = """
A central scientific tenet of this project is that each agent must be evaluated according to its actual technical function, rather than forcing an artificial "accuracy" metric onto non-diagnostic components:

"We evaluate each agent according to its actual function. Diagnostic components are evaluated against disease ground truth, while reliability components are evaluated using task-specific objective measures rather than forcing a generic accuracy metric."

EVALUATION METHODOLOGIES BY AGENT:

- Base Model Agent:
  - Evaluation Metric: Standalone diagnostic discrimination against radiologist ground truth.
  - Measures: TP, TN, FP, FN, Accuracy, Precision, Recall, Specificity, NPV, F1-Score, ROC-AUC, PR-AUC.

- Quality Agent:
  - Evaluation Metric: Objective physical signal metrics and quality state transitions.
  - Measures: Laplacian variance deltas, SNR gain (dB), mean intensity shifts, transition counts (Poor -> Good, Poor -> Degraded, Poor -> Poor), and percentage improved vs worsened.

- Calibration & Uncertainty Agent:
  - Evaluation Metric: Reliability diagrams, calibration error, and risk stratification.
  - Measures: Expected Calibration Error (ECE), Brier Score, normalized Shannon entropy distribution, and empirical accuracy stratification across Low vs High uncertainty cohorts.

- OOD Detection Agent:
  - Evaluation Metric: Distance distribution statistics and safety routing adherence.
  - Measures: Mahalanobis distance percentiles (95th borderline, 99th severe), percentage of severe outliers rejected (100%), and percentage of borderline cases escalated (100%).

- Decision Agent:
  - Evaluation Metric: Operational routing consistency and reconciliation audit.
  - Measures: 100% adherence to rule precedence table R1–R7; exact arithmetic reconciliation between initial routing and final operational disposition.

- Image Repair Agent:
  - Evaluation Metric: Paired objective quality changes and diagnostic state transitions.
  - Measures: Laplacian/SNR/intensity deltas, prediction stability percentage, 4-way transition matrix (Corr->Corr, Inc->Inc, Corr->Inc, Inc->Corr), and number of diagnostic errors corrected.

- Verification Agent:
  - Evaluation Metric: Gate audit outcomes, escalation behavior, and post-repair error containment.
  - Measures: Gate pass/fail rates across the 5 safety gates, post-repair escalation rate (43.32%), and verification of non-degradation bounds.
"""

SECTION_20_TITLE = "Section 20 — Real Project Demonstration Cases"
SECTION_20_BODY = """
To illustrate how the multi-agent system operates in practice, four canonical case studies from the project evaluation are presented:

CASE 1: PREDICTION PRESERVATION FOLLOWING TARGETED REPAIR (Image: 00016732_027.png)
- Ground Truth: Pneumonia Positive (1)
- Pre-Repair Status: Quality POOR, Raw Score = 0.523042 (Correct prediction, True Positive)
- Repair Applied: Unsharp Masking
- Post-Repair Status: Quality transitioned to GOOD, Raw Score = 0.523016 (Correct prediction, True Positive)
- Diagnostic Transition: Correct -> Correct (100% stability, score delta = -0.000026)
- Post-Repair Verification: Final Action = ESCALATE (triggered by gate review thresholds)
- What it demonstrates: Targeted repair successfully restored objective image quality from POOR to GOOD while maintaining complete model prediction stability.

CASE 2: MEASURABLE QUALITY RECOVERY WITH PERSISTENT DIAGNOSTIC ERROR (Image: 00022877_014.png)
- Ground Truth: Pneumonia Positive (1)
- Pre-Repair Status: Quality POOR, Raw Score = 0.507659 (Incorrect prediction, False Negative)
- Repair Applied: Fast Non-Local Means Denoising
- Post-Repair Status: Quality transitioned to DEGRADED, Laplacian Delta = -34.30, SNR Delta = +0.60 dB, Raw Score = 0.507672 (Incorrect prediction, False Negative)
- Diagnostic Transition: Incorrect -> Incorrect (score delta = +0.000013)
- Post-Repair Verification: Final Action = ESCALATE
- What it demonstrates: Denoising improved SNR (+0.60 dB), but the underlying diagnostic error persisted completely. This exemplifies why quality recovery cannot be assumed to equal diagnostic improvement.

CASE 3: VERIFICATION SAFETY GUARDRAIL INTERCEPTION (Image: 00015646_014.png)
- Ground Truth: Pneumonia Positive (1)
- Pre-Repair Status: Quality POOR, Raw Score = 0.506344 (Incorrect prediction, False Negative)
- Repair Applied: Unsharp Masking
- Post-Repair Status: Quality transitioned to GOOD, Raw Score = 0.506506 (Incorrect prediction, False Negative)
- Diagnostic Transition: Incorrect -> Incorrect
- Post-Repair Verification: Final Action = ESCALATE TO RADIOLOGIST REVIEW
- What it demonstrates: The system prevented automated release of an incorrect prediction even though image sharpness reached GOOD, protecting the clinical workflow.

CASE 4: DELIBERATELY BLURRED STRESS TEST (Synthetic Catastrophic Blur)
- Ground Truth: Clinical Evaluation Image
- Pre-Repair Status: Quality POOR, Laplacian Variance = 1.70, Raw Score = 0.5041, Uncertainty HIGH, OOD IN_DISTRIBUTION (27.98), Decision = REPAIR
- Repair Applied: Unsharp Masking (radius=1.0, amount=0.5)
- Post-Repair Status: Array modified = TRUE, Laplacian Variance = 1.72 (Delta = +0.016), SNR = 59.2 dB, Score = 0.5041, Quality = POOR
- Post-Repair Verification: Status = ESCALATE (Gate 4 Quality Recovery FAILED; POOR -> POOR)
- What it demonstrates: Repair does not force acceptance. When physical degradation is beyond restorative capacity, Verification halts the process and escalates to human review.
"""

SECTION_21_TITLE = "Section 21 — What the Project Proves (Empirically Supported Claims)"
SECTION_21_BODY = """
The empirical findings of this project rigorously substantiate thirteen core scientific claims:

1. A multi-agent supervisory architecture can be constructed around a frozen, pretrained chest X-ray classifier without modifying its architecture or weights.
2. Technical image quality (sharpness, noise, exposure) can be objectively quantified prior to clinical reliance using lightweight, deterministic signal processing algorithms.
3. Distributional shifts and severe outliers can be monitored in the 1024-dimensional semantic feature space using Mahalanobis distance solved stably via Cholesky decomposition.
4. Prediction uncertainty can be mathematically formulated via probability calibration, confidence bounds, and normalized binary Shannon entropy.
5. Multi-agent reliability evidence can drive deterministic, auditable rule-based triage decisions (ACCEPT, REPAIR, ESCALATE, REJECT).
6. Targeted image processing (CLAHE, NLMeans, Unsharp Masking) can be conditionally executed based on specific identified defects.
7. Post-repair radiographs can undergo a complete fresh inference pass across quality, disease prediction, OOD, and uncertainty agents.
8. A multi-gate Verification Agent can audit post-repair outcomes and prevent automated release when recovery criteria are unmet.
9. Selective release can contain 950 baseline diagnostic errors (62.71% containment rate), elevating released cohort accuracy from 90.94% to 92.31%.
10. A structured human review pathway provides a safe clinical fallback for ambiguous, corrupted, or out-of-distribution radiographs.
11. Targeted repair can achieve significant objective image quality improvements (78.28% recovery rate; +57.71 Laplacian variance gain for blur; up to +19.91 dB SNR gain for noise).
12. Targeted repair did NOT demonstrate diagnostic accuracy improvement for the frozen DenseNet-121 in the authoritative N=35 paired benchmark (0 errors corrected, 100% stability).
13. Catastrophic degradation (such as severe blur) correctly triggers targeted repair, but because repair cannot reconstruct destroyed high frequencies, the Verification Agent reliably intercepts the failure and escalates to human review.
"""

SECTION_22_TITLE = "Section 22 — What the Project Does NOT Prove (Strict Negative Boundaries)"
SECTION_22_BODY = """
To uphold rigorous scientific integrity, the project explicitly acknowledges eight boundaries that are NOT claimed or proven:

1. It does NOT prove clinical diagnostic superiority: The system does not claim that DenseNet-121 outperforms experienced radiologists or other commercial CAD algorithms.
2. It does NOT prove that image repair improves pneumonia classification accuracy: Across all paired benchmarks and parameter sweeps, repair corrected zero diagnostic errors for the frozen model.
3. It does NOT prove formal out-of-distribution AUROC: Without an external, multi-modal benchmark of non-chest images, formal OOD discriminative curves cannot be reported on NIH ChestX-ray14.
4. It does NOT prove that synthetic noise and exposure models perfectly capture all real-world radiograph artifacts: Synthetic Gaussian noise and linear brightness shifts are controlled approximations of complex radiographic hardware physics.
5. It does NOT prove clinical safety for deployment: This software is an academic research prototype and has not undergone formal clinical safety or regulatory certification.
6. It does NOT replace radiologists: The system is designed explicitly as a supervisory decision-support tool that relies on human radiologists for 56.06% of cases.
7. It does NOT demonstrate state-of-the-art diagnostic discrimination: Standalone baseline ROC-AUC on the full NIH test cohort is 0.7016, reflecting the known challenges of label noise in ChestX-ray14.
8. It is NOT a commercially certified medical device: It remains strictly an experimental research prototype.
"""

SECTION_23_TITLE = "Section 23 — Limitations and Future Research Directions"
SECTION_23_BODY = """
A transparent appraisal of project limitations identifies several crucial avenues for future investigation:

1. Research Prototype Status: The architecture is implemented in Python and PyTorch for academic evaluation and lacks FDA 510(k) or CE mark certification for clinical deployment.
2. NIH ChestX-ray14 Label Noise: Ground-truth disease labels in ChestX-ray14 were extracted via automated natural language processing (NLP) of radiology reports, which introduces known label noise (~10-15% error rate in NLP extraction).
3. Extreme Class Imbalance: With pneumonia prevalence at exactly 1.3155% (220 positives out of 16,724 test cases), precision (PPV) is inherently suppressed into low single digits (3.52% baseline, 3.30% released), meaning that even small false positive shifts heavily impact F1-scores.
4. Lack of External Labeled OOD Benchmarks: OOD thresholds were calibrated empirically using in-distribution percentiles (95th and 99th) rather than validated against external non-chest radiographs (e.g., pediatric foreign bodies, CT slices, or artifacts).
5. Synthetic Degradation Benchmarks: While the natural blur cohort was extracted directly from clinical data, the noise and exposure tuning studies relied on synthetic degradations due to the absence of paired "clean vs corrupted" raw radiographs from the same patient exposure.
6. Sample Size of Paired Diagnostic Benchmark: The authoritative paired diagnostic validation cohort comprised N=35 repaired cases. While statistically sufficient to demonstrate 100% stability and zero flips, larger paired cohorts across multiple pathology classes would provide broader statistical power.
7. Frozen Model Representation Gap: Because DenseNet-121 was frozen, its convolutional kernels could not adapt to subtle distribution shifts introduced by post-hoc image filters. Joint fine-tuning of restorative modules and classifier backbones represents a promising future direction.
8. Need for Prospective Clinical Trials: Real-world clinical utility must ultimately be evaluated through prospective multi-reader, multi-case (MRMC) trials with practicing thoracic radiologists measuring reading time, diagnostic accuracy, and user trust.
"""

SECTION_24_TITLE = "Section 24 — Consolidated Method Selection RATIONALE Matrix"
SECTION_24_BODY = """
The table below consolidates the technical justification, problem alignment, measurement target, and primary limitation for every algorithm selected in the multi-agent system:

1. Quality Agent — Sharpness:
   - Method: Laplacian Variance (Var(del^2 I))
   - Why Selected: O(N) convolution; highly sensitive to edge loss; no neural network overhead.
   - What it Measures: High-frequency spatial second derivative (anatomical sharpness).
   - Main Limitation: Invariant to blur directionality; sensitive to sensor noise spikes.
   - Project Findings: 100% monotonic variance recovery across blur sweeps; successfully guided unsharp masking.

2. Quality Agent — Noise:
   - Method: Signal-to-Noise Ratio (SNR in dB)
   - Why Selected: Direct measurement of signal dominance over background noise floor.
   - What it Measures: Ratio of mean signal power to estimated noise variance.
   - Main Limitation: Homogeneous background patch estimation can be noisy in crowded lung fields.
   - Project Findings: Baseline test mean SNR = 39.8 dB; correctly flagged low-SNR corrupted images.

3. Quality Agent — Exposure:
   - Method: Mean Intensity & Histogram Analysis
   - Why Selected: Evaluates global radiometric range and detector saturation.
   - What it Measures: Global brightness mean and variance across [0, 255].
   - Main Limitation: Cannot detect localized regional underexposure in isolated lung zones.
   - Project Findings: Successfully screened out extreme exposure outliers (<20 or >235).

4. Base Model Agent:
   - Method: TorchXRayVision DenseNet-121 (densenet121-res224-nih)
   - Why Selected: Dense feature reuse; domain-specific chest radiograph pretraining; extracts 1024-D GAP features.
   - What it Measures: Continuous pneumonia disease score and semantic feature representation.
   - Main Limitation: Susceptible to unconditional inference failure on degraded or outlier images.
   - Project Findings: Achieved 0.7016 ROC-AUC on full test cohort; 1024-D features enabled effective OOD screening.

5. Calibration Agent:
   - Method: Post-hoc Platt Scaling (Logistic Calibration)
   - Why Selected: Preserves ROC ranking; monotonic; computationally lightweight; minimizes ECE.
   - What it Measures: Calibrated posterior probability p_cal in [0, 1].
   - Main Limitation: Assumes sigmoidal miscalibration; does not improve discrimination.
   - Project Findings: Reduced ECE from ~0.051 to ~0.015 on validation cohort.

6. Uncertainty Agent:
   - Method: Confidence (max(p, 1-p)) & Normalized Binary Shannon Entropy
   - Why Selected: Quantifies predictive dispersion; mathematically bounded in [0, 1].
   - What it Measures: Epistemic boundary closeness and predictive certainty.
   - Main Limitation: Does not distinguish data noise (aleatoric) from model ignorance (epistemic).
   - Project Findings: Low-uncertainty cohort achieved 99.57% accuracy vs 90.22% for high-uncertainty cohort.

7. OOD Detection Agent:
   - Method: Mahalanobis Distance in 1024-D Feature Space
   - Why Selected: Accounts for feature variance and covariance across correlated latent channels.
   - What it Measures: Statistical distance from in-distribution training manifold.
   - Main Limitation: Requires inverted covariance matrix; assumes unimodal Gaussian feature distribution.
   - Project Findings: Successfully rejected 100% of severe outliers (101/101) and escalated 100% of borderline cases (86/86).

8. Numerical OOD Solver:
   - Method: Cholesky Decomposition (Sigma = L * L^T) with Tikhonov Regularization
   - Why Selected: Numerically stable; avoids explicit matrix inversion of 1024x1024 covariance matrix.
   - What it Measures: Solves L * y = (x - mu), reducing distance computation to ||y||_2.
   - Main Limitation: Fails if covariance matrix is not strictly positive definite (requires epsilon regularization).
   - Project Findings: Guaranteed stable distance calculation across all 16,724 test cases with zero numerical exceptions.

9. Decision Agent:
   - Method: Deterministic Rule-Based Arbiter (Hierarchical Rules R1–R7)
   - Why Selected: 100% auditable; mathematically deterministic; zero black-box risk in safety arbitration.
   - What it Measures: Maps multi-agent evidence bundle to ACCEPT, REPAIR, ESCALATE, or REJECT.
   - Main Limitation: Static rules do not dynamically adapt without policy re-specification.
   - Project Findings: Perfectly reconciled 16,724 initial decisions to final operational dispositions.

10. Image Repair — Blur:
    - Method: Unsharp Masking (radius=1.0, amount=0.5)
    - Why Selected: Lightweight O(N) edge boosting; restores high-frequency gradients without ring artifacts.
    - What it Measures: Enhances local spatial gradients across anatomical borders.
    - Main Limitation: Cannot hallucinate destroyed optical frequencies (as proven by severe blur stress test).
    - Project Findings: 100% variance improvement across sweeps (+57.71 variance gain); 100% prediction stability.

11. Image Repair — Noise:
    - Method: Fast Non-Local Means Denoising (template=7, search=21)
    - Why Selected: Exploits anatomical patch redundancy; attenuates noise while preserving edges.
    - What it Measures: Weighted non-local patch averaging.
    - Main Limitation: High h induces over-smoothing and can erase pulmonary infiltrates.
    - Project Findings: h=3.0 yielded 0 dB gain; h=10.0 gave +6.6 dB gain but flipped 1 correct case to incorrect.

12. Image Repair — Exposure:
    - Method: CLAHE (clipLimit=2.0, tileGridSize=(8, 8))
    - Why Selected: Restores local contrast without global dynamic range blowout.
    - What it Measures: Local adaptive histogram redistribution.
    - Main Limitation: Non-physical remapping; causes probability drift in frozen CNN.
    - Project Findings: Re-centered mean intensity from 115.42 to 123.95; caused probability drift of -0.052.

13. Verification Agent:
    - Method: 5-Gate Multi-Criteria Safety Firewall
    - Why Selected: Enforces defense-in-depth; prevents unverified releases from reaching clinical workflow.
    - What it Measures: Execution, label stability, OOD status, quality recovery, and confidence/SNR non-degradation.
    - Main Limitation: Conservative gating withholds cases that might have been diagnostically benign.
    - Project Findings: Verified 6,848 repairs; safely escalated 5,234 unrecovered cases; blocked severe blur stress test.
"""

SECTION_25_TITLE = "Section 25 — Faculty-Friendly Verbal Presentation Pitch (3-Minute Executive Summary)"
SECTION_25_BODY = """
This concise verbal presentation is tailored for oral defense before academic faculty and review committees:

"Respected faculty members,

Before our project, clinical chest X-ray AI operated as an unconditional prediction engine:
A radiograph entered DenseNet-121, and the model directly produced a pneumonia prediction regardless of whether the image was severely blurred, heavily corrupted by sensor noise, or an extreme out-of-distribution outlier. On the full NIH ChestX-ray14 test set of 16,724 images, this direct approach produced 1,515 baseline diagnostic errors.

Our project introduces a reliability-aware supervisory architecture around that frozen classifier. We do not replace DenseNet-121, and we do not alter its diagnostic weights. Instead, we wrap it in seven specialized agents:

1. The Quality Agent checks whether the radiograph is technically usable using Laplacian sharpness, SNR, and intensity analysis.
2. The Base Model produces the disease prediction and extracts a 1024-dimensional semantic feature representation.
3. The Calibration and Uncertainty Agent calibrates raw scores via Platt Scaling and calculates predictive confidence and Shannon entropy.
4. The OOD Agent monitors distribution shifts using Mahalanobis distance solved stably via Cholesky decomposition.
5. The Decision Agent applies a deterministic rule table to decide whether to accept, repair, escalate, or reject.
6. When an image has repairable technical defects, the Repair Agent applies targeted restoration — CLAHE for exposure, Non-Local Means for noise, or Unsharp Masking for blur.
7. Crucially, we then run fresh inference, and the Verification Agent enforces five safety gates to verify that the repair actually succeeded before authorizing release. If evidence is insufficient, the case is safely escalated to human radiologist review.

Our experimental results demonstrate two vital conclusions:
First, the system achieves massive error containment: by selectively releasing 43.94% of cases and withholding 56.06%, exactly 950 baseline errors were prevented from being automatically released, lifting released accuracy from 90.94% to 92.31%.
Second, and most importantly for scientific honesty: we do NOT claim that image repair improves DenseNet's diagnostic accuracy. In our authoritative paired experiment of 35 repaired cases, diagnostic correctness did not improve — prediction stability was 100%, with zero errors corrected. Furthermore, in our deliberately blurred stress test, the system proved that repair does not force acceptance: because severe blur could not be recovered, the Verification Agent correctly escalated the image to human review.

In summary, our contribution shifts medical AI from 'prediction-only AI' to 'reliability-aware AI' — a system that knows when to predict, when to repair, when to verify, and when to withhold."
"""

SECTION_26_TITLE = "Section 26 — Final Conclusion: From Prediction-Only AI to Reliability-Aware AI"
SECTION_26_BODY = """
The foundational contribution of this research project is not simply attempting to make a deep neural network slightly more accurate on a static benchmark. It is fundamentally transforming the operational paradigm of clinical medical imaging:

BEFORE THE PROJECT:
    Input Radiograph → Deep Neural Network → Forced Unconditional Prediction
    (Blind to quality defects, blind to distribution shifts, blind to uncertainty, releasing 1,515 errors)

AFTER THE PROJECT:
    Input Radiograph → Quality Assessment → DenseNet Classification + 1024-D Feature Extraction → Probability Calibration & Uncertainty Estimation → OOD Distributional Screening → Deterministic Decision Arbitration → Conditional Targeted Image Repair → Fresh Post-Repair Re-Inference → Multi-Gate Verification → Selective Release OR Safe Human Review

By establishing rigorous supervisory governance, the system transitions medical artificial intelligence:
    "FROM PREDICTION-ONLY AI TO RELIABILITY-AWARE AI."

The system operates as an intelligent clinical partner that knows when to PREDICT, when to REPAIR, when to VERIFY, when to ESCALATE, and when to WITHHOLD.
"""

# FACULTY DEFENSE Q&A — 30 DEFINITIVE TECHNICAL QUESTIONS
SECTION_QA_TITLE = "Comprehensive Faculty Defense Q&A: 30 Definitive Technical Questions"
SECTION_QA_BODY = """
QUESTION 1: What was the project before this work?
ANSWER: Before this project, the system was a conventional monolithic deep learning pipeline: an input chest radiograph was fed directly into TorchXRayVision DenseNet-121, producing an unconditional binary pneumonia prediction regardless of severe image quality degradation, out-of-distribution demographic anomalies, or high model uncertainty.

QUESTION 2: What did you add to the system?
ANSWER: We introduced an active multi-agent supervisory reliability layer comprising six specialized agents around the frozen DenseNet-121 model: Quality Agent, Calibration and Uncertainty Agent, OOD Detection Agent, Decision Agent, Image Repair Agent, and Verification Agent, alongside a selective release and human review triage architecture.

QUESTION 3: What is the actual scientific and engineering contribution?
ANSWER: The primary contribution is not retraining or replacing DenseNet-121, but creating an evidence-governed supervisory framework that transforms unconditional 'prediction-only AI' into 'reliability-aware AI' capable of containing baseline diagnostic errors, screening distribution shifts, performing targeted restoration, and enforcing multi-gate verification.

QUESTION 4: Why was DenseNet-121 selected as the Base Model?
ANSWER: DenseNet-121 features dense layer connectivity that encourages feature reuse and mitigates vanishing gradients. We utilized TorchXRayVision's 'densenet121-res224-nih' checkpoint because it is pretrained on vast multi-institutional chest radiographs, capturing domain-specific pulmonary features while providing a 1024-dimensional semantic embedding for OOD detection.

QUESTION 5: Why did you choose Laplacian Variance for blur detection?
ANSWER: The discrete Laplacian operator computes the spatial second derivative of intensity, quantifying high-frequency edge sharpness (rib borders, cardiac margins). Blur attenuates high spatial frequencies, causing variance to drop toward zero. It is an O(N) convolution requiring zero additional neural network inference.

QUESTION 6: Why did you choose Signal-to-Noise Ratio (SNR) for noise detection?
ANSWER: Sensor noise competes directly with anatomical lung markings, potentially simulating pulmonary infiltrates. Measuring SNR in decibels (10 * log10(mu^2 / sigma_noise^2)) directly quantifies signal dominance over the background noise floor.

QUESTION 7: Why did you choose Mean Intensity and Histogram Statistics for exposure?
ANSWER: Underexposure obliterates lung parenchyma into black noise (<20), while overexposure causes sensor burnout and saturation washouts (>235). Mean intensity and histogram variance provide computationally lightweight, deterministic bounds on global dynamic range.

QUESTION 8: Why did you select Platt Scaling for probability calibration?
ANSWER: Deep networks trained with cross-entropy produce overconfident, uncalibrated sigmoid scores. Platt Scaling fits a logistic transformation (p_cal = 1 / (1 + exp(-(Az + B)))) on validation logits, preserving the ROC curve and class ranking while minimizing Expected Calibration Error (ECE from ~0.051 to ~0.015).

QUESTION 9: How is prediction confidence defined and why?
ANSWER: Confidence is defined as max(p_cal, 1 - p_cal) bounded in [0.5, 1.0]. A score of 0.50 represents maximal predictive ambiguity (random guess), whereas scores near 0.0 or 1.0 reflect strong directional certainty.

QUESTION 10: Why did you incorporate Normalized Binary Shannon Entropy?
ANSWER: Normalized entropy H_norm(p) = [-p ln(p) - (1-p) ln(1-p)] / ln(2) quantifies the dispersion of the probability distribution. It provides a non-linear, mathematically bounded [0, 1] metric that penalizes boundary-adjacent predictions far more aggressively than linear distance.

QUESTION 11: Why did you select Mahalanobis Distance for OOD detection?
ANSWER: Euclidean distance assumes independent, isotropic feature variances. In deep neural feature spaces, activations across the 1024 channels are heavily correlated. Mahalanobis distance scales distances by the covariance matrix (Sigma), measuring true statistical divergence from the in-distribution manifold.

QUESTION 12: Why did you utilize Cholesky Decomposition for the Mahalanobis solver?
ANSWER: Explicit inversion of a 1024x1024 covariance matrix is numerically unstable and computationally expensive. Decomposing Sigma into lower-triangular L * L^T (with Tikhonov regularization) allows solving L * y = (x - mu) via forward substitution, reducing distance calculation to the Euclidean norm ||y||_2.

QUESTION 13: Why is the Decision Agent rule-based rather than a machine learning classifier?
ANSWER: In clinical AI safety, supervisory arbitration must be deterministic, auditable, and transparent. A rule table (R1–R7) eliminates the risk of covariate shift within the decision policy itself and guarantees explicit safety invariants (e.g., severe outliers are always rejected).

QUESTION 14: Why was CLAHE chosen for exposure repair and what is its limitation?
ANSWER: CLAHE redistributes local contrast across contextual grids while clipping histogram spikes, preventing noise amplification in dark soft tissues. Its limitation is that it is an optical spatial remapping, not an inverse physical model of X-ray photon attenuation.

QUESTION 15: Why was Non-Local Means chosen for denoising and what is its limitation?
ANSWER: NLMeans exploits non-local anatomical self-similarity by computing weighted patch averages, preserving sharp parenchymal borders that simple Gaussian smoothing destroys. Its limitation is computational cost and the risk of over-smoothing delicate interstitial markings at high filtering strengths (h >= 10.0).

QUESTION 16: Why was Unsharp Masking chosen for blur repair and what is its limitation?
ANSWER: Unsharp masking boosts local high-frequency gradients by subtracting a blurred Gaussian mask. It is computationally lightweight and effective for moderate motion blur. Its limitation is that it cannot reconstruct optical information that has been completely destroyed.

QUESTION 17: Why is a post-repair Verification Agent strictly necessary?
ANSWER: A restored radiograph may improve an objective image metric (such as Laplacian variance or SNR) while simultaneously introducing subtle pixel distortions that destabilize the classifier. Verification enforces defense-in-depth, ensuring that repaired images satisfy five safety gates before automated release.

QUESTION 18: What did each agent actually improve?
ANSWER: Quality improved technical defect visibility; Base Model provided stable disease classification and feature embeddings; Calibration improved probability reliability (ECE 0.051 -> 0.015); Uncertainty enabled 99.57% accuracy risk stratification; OOD intercepted 100% of severe outliers; Decision provided auditable triage; Repair improved image metrics in 78.28% of cases; Verification blocked 5,234 failed repairs; and Selective Release contained 950 baseline errors.

QUESTION 19: How did you evaluate each agent without forcing a generic accuracy metric?
ANSWER: Diagnostic components (Base Model) were evaluated against radiologist ground truth (ROC-AUC, F1, sensitivity). Reliability components were evaluated with task-specific measures: Quality by variance/SNR deltas; Calibration by ECE and Brier score; Uncertainty by entropy distribution and group accuracy; OOD by routing adherence; Decision by reconciliation consistency; Repair by paired quality gains and stability; and Verification by gate failure rates.

QUESTION 20: Did image repair improve diagnostic classification accuracy?
ANSWER: No. In our authoritative paired experiment of 35 repaired cases, diagnostic correctness remained unchanged (0 errors corrected, 0 new errors, 100% stability). Repair restores physical image quality, not the diagnostic accuracy of the frozen classifier.

QUESTION 21: What happened in the authoritative final paired repair experiment (N=35)?
ANSWER: Across 35 repaired cases (17 positive, 18 negative), 17 cases transitioned Correct -> Correct, 18 transitioned Incorrect -> Incorrect, and zero cases flipped classification labels. Accuracy before and after repair was exactly 48.5714% (Delta = 0.0%). While quality improved in 85.71% of cases, diagnostic errors corrected was exactly zero.

QUESTION 22: What occurred during the deliberately blurred image stress test?
ANSWER: A severely blurred radiograph (Laplacian variance 1.70 vs 100 threshold) triggered repair via Unsharp Masking. The repair modified the array, but variance only moved to 1.72 (Delta +0.016), remaining POOR. The Verification Agent evaluated the repaired image, failed Gate 4 (Quality Recovery), and escalated the case to human review (POOR -> POOR).

QUESTION 23: Why did the severe blur stress test case escalate instead of being accepted?
ANSWER: The artificial blur destroyed high spatial frequencies beyond the restorative capacity of local unsharp masking. Because Laplacian variance remained at 1.72 (far below the 100.0 threshold), Gate 4 failed, proving that repair does not equal automatic acceptance.

QUESTION 24: How many cases were released automatically versus withheld?
ANSWER: Out of 16,724 test cases, exactly 7,349 cases (43.94%) were automatically released (501 direct releases + 6,848 verified repair releases). Exactly 9,375 cases (56.06%) were safely withheld for radiologist review.

QUESTION 25: Why is withholding 56.06% of cases considered a safety mechanism rather than a failure?
ANSWER: In clinical radiology, forcing automated predictions on low-confidence, corrupted, or out-of-distribution radiographs compromises patient safety. Withholding is an intentional triage mechanism that reserves automated release strictly for verified, high-certainty cases.

QUESTION 26: How did the system contain baseline diagnostic errors?
ANSWER: The standalone baseline generated 1,515 diagnostic errors (1,344 FP + 171 FN). Among the 7,349 released cases, only 565 errors occurred. Exactly 950 baseline errors (846 FP + 104 FN) were contained because the corresponding cases were safely withheld from automated release (62.71% containment rate).

QUESTION 27: What are the primary limitations of the project?
ANSWER: It is an academic research prototype; NIH labels contain NLP-derived label noise; pneumonia prevalence is heavily imbalanced (1.32%); synthetic degradation models approximate complex physical artifacts; and the system has not been validated in a prospective clinical trial.

QUESTION 28: What does the project actually prove?
ANSWER: It proves that active reliability monitoring can be wrapped around a frozen medical classifier, that quality and distribution shifts can be objectively screened, that selective release contains 62.71% of baseline errors, that image repair recovers image quality (78.28%), that repair does NOT improve frozen model accuracy, and that multi-gate verification reliably intercepts failed repairs.

QUESTION 29: What does the project explicitly NOT prove?
ANSWER: It does not prove clinical superiority, does not prove repair improves pneumonia diagnosis, does not prove formal OOD AUROC on external non-chest datasets, does not replace human radiologists, and does not constitute a certified medical device.

QUESTION 30: Why is this multi-agent architecture superior to simply using DenseNet-121 directly?
ANSWER: Standalone DenseNet-121 blindly releases 1,515 errors into clinical workflows on corrupted, ambiguous, and outlier radiographs. The multi-agent architecture identifies defects, salvages repairable cases, verifies stability, and withholds 950 errors for expert review, providing a transparent, auditable, and safety-governed AI partnership.
"""

print("Section definitions 17-26 and Q&A loaded.")
