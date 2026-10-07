"""
Report text content definitions.
Part 2: Text sections 4 to 8.
"""

from __future__ import annotations

SECTION_4_TITLE = "Section 4 — Agent 1: Quality Agent (Signal Processing Rationale & Results)"
SECTION_4_BODY = """
The Quality Agent screens radiographs prior to classifier reliance, identifying technical defects that distort anatomical boundaries or induce artificial feature representations.

MATHEMATICAL METHODS & SELECTION RATIONALE:
1. Laplacian Variance (Sharpness / Blur Detection):
   - Discrete 2D Laplacian operator: L(I) = (d^2 I / dx^2) + (d^2 I / dy^2), convolved via a standard 3x3 discrete kernel.
   - Variance Var(L(I)) quantifies the second spatial derivative across the image. High variance corresponds to sharp anatomical borders (ribs, cardiac margins, diaphragm). Blur smooths high spatial frequencies, collapsing Laplacian variance toward zero.
   - Why selected: Extremely fast O(N) convolution requiring zero additional neural network inference; deterministic and invariant to global brightness shifts.
   - Operating threshold: 100.0 (variance < 100 flags blur; reference sharp variance = 500.0).

2. Signal-to-Noise Ratio (SNR):
   - Measured as 10 * log10(mu^2 / sigma_noise^2) in dB, where background noise variance is estimated across homogeneous sub-regions.
   - Why selected: High acquisition noise introduces high-frequency grain that can simulate pulmonary infiltrates. SNR directly quantifies signal dominance over noise floor.
   - Operating threshold: 15.0 dB (SNR < 15 dB flags severe noise).

3. Mean Intensity & Histogram Distribution (Exposure Analysis):
   - Computed as the global mean pixel intensity and standard deviation across [0, 255].
   - Why selected: Underexposure (< 20) obliterates lung parenchyma into black noise; overexposure (> 235) causes sensor saturation / burnout where subtle opacities are washed out.
   - Operating threshold: [20.0, 235.0].

EXPERIMENTAL RESULTS ON FULL TEST SET (N=16,724):
- GOOD Quality: 3,608 radiographs (21.57%)
- DEGRADED Quality: 940 radiographs (5.62%)
- POOR Quality: 12,176 radiographs (72.81%)

POST-REPAIR QUALITY RECOVERY (N=12,082 repairs evaluated):
- POOR → GOOD (Full Recovery): 7,766 radiographs (64.28%)
- POOR → DEGRADED (Partial Recovery): 1,692 radiographs (14.00%)
- POOR → POOR (Unresolved Defect): 2,624 radiographs (21.72%)
- Overall Quality Improvement Rate: 78.28% (9,458 / 12,082 recovered)
- Quality Worsening Rate: 0.0% (zero images degraded post-repair)

METHODOLOGICAL DISTINCTION NOTE:
These metrics represent objective image-quality recovery and transition rates. They are NOT termed "Quality Accuracy", as natural NIH radiographs do not possess subjective radiologist quality labels.
"""

SECTION_5_TITLE = "Section 5 — Agent 2: DenseNet-121 Base Model (Diagnostic Engine & Feature Extraction)"
SECTION_5_BODY = """
The diagnostic engine is TorchXRayVision's DenseNet-121 architecture (densenet121-res224-nih), trained on multi-institutional chest radiography data.

WHY DENSENET-121 WAS SELECTED:
1. Architectural Suitability: Dense connectivity (each layer receives feature maps from all preceding layers via concatenation) encourages extensive feature reuse and mitigates vanishing gradients across deep medical representations.
2. Domain Pretraining: Unlike ImageNet-pretrained networks whose early kernels respond to consumer photograph textures, TorchXRayVision weights are pretrained on vast radiograph archives, capturing pulmonary opacities, consolidations, and lung field geometries.
3. Dual-Purpose Utility: DenseNet-121 acts simultaneously as:
   - The disease classification engine (outputting the continuous scalar pneumonia probability score).
   - The feature extraction backbone: extracting a 1024-dimensional semantic embedding from the final Global Average Pooling (GAP) layer prior to the linear classification head.

OPERATING CONSTRAINTS:
The Base Model is intentionally frozen throughout this project. It does not update its weights during inference, nor is it retrained. It provides the definitive diagnostic assessment; the surrounding multi-agent pipeline assesses whether that assessment should be trusted and released.
"""

SECTION_6_TITLE = "Section 6 — Agent 3: Probability Calibration and Uncertainty Estimation"
SECTION_6_BODY = """
PART A — PROBABILITY CALIBRATION (PLATT SCALING):
Raw sigmoid outputs from deep neural networks are notoriously uncalibrated; a network outputting 0.85 may only be correct 60% of the time due to overparameterized cross-entropy minimization.
- Method Selected: Post-hoc Platt Scaling (logistic calibration: p_cal = 1 / (1 + exp(-(A * z + B)))), parameterized by scalar slope A and intercept B fitted on validation data.
- Why selected: Monotonic transformation that preserves the receiver operating characteristic (ROC) curve and class ranking while minimizing Expected Calibration Error (ECE).
- Results: Reduced ECE from ~0.051 to ~0.015, ensuring that released probabilities reflect empirical positive frequencies.
- Critical Distinction: Calibration refines probability alignment; it does NOT alter AUC or ranking discrimination.

PART B — PREDICTION CONFIDENCE:
Defined as max(p_cal, 1 - p_cal) in the range [0.5, 1.0]. A probability of 0.51 reflects minimal confidence (near-random guess), whereas probabilities of 0.02 or 0.98 indicate high confidence.

PART C — NORMALIZED BINARY SHANNON ENTROPY:
H_norm(p) = [-p * ln(p) - (1-p) * ln(1-p)] / ln(2), strictly bounded in [0.0, 1.0]. Entropy measures predictive dispersion; p=0.5 yields maximum entropy (1.0), while p=0 or p=1 yields zero entropy.

CATEGORICAL UNCERTAINTY RULE (TWO-LEVEL SPECIFICATION):
A prediction is designated LOW uncertainty if and only if:
    Confidence >= 0.85  AND  Normalized Entropy <= 0.25
All other predictions are designated HIGH uncertainty.

FULL TEST SET STRATIFICATION RESULTS (N=16,724):
- LOW Uncertainty Cohort: 2,653 images (15.86%)
  - Released subset: 1,642 images
  - Released Accuracy: 99.57% (1,635 TN, 0 TP, 0 FP, 7 FN)
- HIGH Uncertainty Cohort: 14,071 images (84.14%)
  - Released subset: 5,707 images
  - Released Accuracy: 90.22% (5,132 TN, 17 TP, 498 FP, 60 FN)

EMPIRICAL INTERPRETATION:
Low uncertainty strongly correlates with high diagnostic precision and safety (+9.35% accuracy gap over high uncertainty). Crucially, HIGH uncertainty does NOT imply the prediction is incorrect; rather, it identifies radiographs near the boundary requiring human radiologist confirmation.
"""

SECTION_7_TITLE = "Section 7 — Agent 4: Out-of-Distribution (OOD) Detection"
SECTION_7_BODY = """
Deep classifiers produce erratic, overconfident predictions when evaluated on inputs that violate training distribution geometry. The OOD Agent monitors distribution conformity in the 1024-dimensional DenseNet feature space.

MATHEMATICAL METHOD: MAHALANOBIS DISTANCE & CHOLESKY DECOMPOSITION:
D_M(x) = sqrt( (x - mu)^T * Sigma^{-1} * (x - mu) )
where mu is the empirical mean feature vector (1024-D) and Sigma is the covariance matrix computed across the in-distribution training cohort.

WHY MAHALANOBIS WAS SELECTED:
Euclidean distance treats all feature axes as independent and isotropic. In medical feature spaces, activations are heavily correlated. Mahalanobis distance scales distances by directional feature variance and covariance, measuring true statistical divergence from the data manifold.

WHY CHOLESKY DECOMPOSITION WAS SELECTED:
Direct inversion of a 1024x1024 covariance matrix is numerically unstable and computationally expensive. Using Cholesky decomposition (Sigma = L * L^T with Tikhonov regularization epsilon = 1e-4 * trace(Sigma) * I) allows solving the lower-triangular system L * y = (x - mu) via forward substitution, reducing computation to ||y||_2.

EXPERIMENTAL RESULTS ON FULL TEST SET (N=16,724):
- IN_DISTRIBUTION: 16,537 radiographs (98.88%)
- BORDERLINE: 86 radiographs (0.51%) — Distance within [35.54, 42.65] (95th to 99th percentile)
- SEVERE: 101 radiographs (0.60%) — Distance > 42.65 (> 99th percentile)
- Mahalanobis Statistics: Mean = 27.35, Median = 26.61, Min = 17.77, Max = 96.42.

SAFETY ENFORCEMENT AUDIT:
- 100% of SEVERE OOD cases (101 / 101) were intercepted and REJECTED.
- 100% of BORDERLINE cases (86 / 86) were intercepted and ESCALATED.
- Zero outlier images were permitted to release automated predictions.

LIMITATION:
This is distribution containment, not OOD classification accuracy. Without external non-chest distractors (e.g., CT or natural images), formal external AUROC cannot be claimed.
"""

SECTION_8_TITLE = "Section 8 — Agent 5: Decision Agent (Rule-Based Arbitration & Reconciliation)"
SECTION_8_BODY = """
The Decision Agent arbitrates multi-agent evidence into deterministic operational actions.

WHY A RULE-BASED ARBITER WAS SELECTED:
In high-stakes clinical AI, safety arbitration must not be delegated to a secondary "black-box" machine learning classifier. A rule table provides:
- Mathematical determinism and 100% auditability.
- Explicit safety invariants (e.g., severe outliers never produce release).
- Zero risk of covariate shift within the arbitration policy itself.

RULE PRECEDENCE HIERARCHY (PRD FR-5 & ARCHITECTURE SPECIFICATION):
1. Rule R1 (Severe Outlier): If OOD == SEVERE → Action.REJECT.
2. Rule R2 (Borderline Signal): If OOD == BORDERLINE or near threshold → Action.ESCALATE.
3. Rule R3 (Repair Opportunity): If Quality in (POOR, DEGRADED) and OOD == IN_DISTRIBUTION → Action.REPAIR.
4. Rule R4 (High Uncertainty Gating): If Quality == GOOD, OOD == IN_DISTRIBUTION, Uncertainty == HIGH → Action.ESCALATE.
5. Rule R6 (Optimal Release): If Quality == GOOD, OOD == IN_DISTRIBUTION, Uncertainty == LOW → Action.ACCEPT.
6. Rule R7 (Fail-Safe Default): Any unhandled or ambiguous combination → Action.ESCALATE.

INITIAL ROUTING vs FINAL DISPOSITION:
Initial routing classifies incoming images; final disposition represents the eventual fate after repair and verification:
- Initial Decision Routing (N=16,724):
  - Direct ACCEPT: 501 (3.00%)
  - Routed to REPAIR: 13,001 (77.74%)
  - Direct ESCALATE: 3,121 (18.66%)
  - Direct REJECT: 101 (0.60%)
  - Reconciled Total: 501 + 13,001 + 3,121 + 101 = 16,724.
- Final Disposition Outcome:
  - Total Released: 7,349 (43.94%) = 501 (Direct Accept) + 6,848 (Verified Repair Releases).
  - Total Withheld: 9,375 (56.06%) = 3,121 (Direct Escalate) + 5,234 (Repair Verification Escalations) + 919 (Unrepairable/Bounds Refused) + 101 (Rejections).
  - Reconciled Total: 7,349 + 9,375 = 16,724.
"""

print("Section definitions 4-8 loaded.")
