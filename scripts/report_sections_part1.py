"""
Report text content definitions and docx builders.
Part 1: Text sections 1 to 14.
"""

from __future__ import annotations

# Text sections stored as modular dictionaries and formatted generators
SECTION_1_TITLE = "Section 1 — Executive Summary: The Core Paradigm Shift"
SECTION_1_BODY = """
The central objective of this research project is addressing a critical, foundational vulnerability in conventional clinical deep learning: the paradigm of unconditional inference. In conventional computer-aided detection (CAD) systems, deep convolutional neural networks operate under a closed-world assumption, taking an arbitrary input radiograph and producing a diagnostic disease prediction regardless of severe image quality degradation, out-of-distribution demographic or pathological shifts, or high epistemic model uncertainty.

BEFORE THE PROJECT (Conventional Base Model Pipeline):
The baseline system represents the standard clinical AI workflow:
    Input Chest Radiograph → TorchXRayVision DenseNet-121 → Unconditional Pneumonia Classification

Under this architecture, the model forces a diagnostic prediction on 100% of incoming cases. On the full test cohort of 16,724 radiographs from the NIH ChestX-ray14 dataset, this direct approach generates 1,515 baseline diagnostic errors (1,344 false positives and 171 false negatives). Radiographs suffering from severe motion blur, extreme sensor noise, truncation, or foreign hardware are treated identically to pristine, calibrated clinical radiographs.

AFTER THE PROJECT (Reliability-Aware Multi-Agent Supervisory Architecture):
The proposed project establishes an active multi-agent reliability architecture around the frozen DenseNet-121 diagnostic engine:
    Input Chest Radiograph → Quality Assessment → DenseNet Forward Pass & Feature Extraction → Probability Calibration & Uncertainty Estimation → Out-of-Distribution Screening → Rule-Based Decision Arbitration → Conditional Targeted Image Repair → Fresh Post-Repair Inference Pass → Multi-Gate Verification → Selective Release OR Triage to Radiologist Review

The core conceptual shift of the project is summarized as:
    BEFORE: "What is the model's prediction for this image?"
    AFTER: "Does this radiograph meet the requisite quality, distribution, and confidence conditions for its prediction to be safely released?"

The multi-agent system does not replace DenseNet-121 with a new classifier, nor does it alter the underlying diagnostic weights. Instead, it introduces an active governance framework that determines whether an automated prediction should be directly released, conditionally repaired and verified, or withheld for human expert consultation.
"""

SECTION_2_TITLE = "Section 2 — Complete System Architecture and Multi-Agent Dataflow"
SECTION_2_BODY = """
The reliability framework coordinates seven specialized agents into a deterministic, auditable pipeline. It is critical to emphasize that these agents are not seven independent pneumonia classifiers; only the Base Model performs disease diagnosis. The remaining six agents act as specialized supervisory guardians evaluating image integrity, feature distribution, probability calibration, risk arbitration, reversible restoration, and safety gating.

1. Quality Agent: Assesses physical and acquisition signal integrity (Laplacian variance for blur, SNR for noise, mean intensity/histogram for exposure) prior to clinical reliance.
2. Base Model Agent (DenseNet-121): Frozen diagnostic engine pretrained on multi-institutional chest radiographs. Produces the binary pneumonia classification score and extracts a 1024-dimensional semantic feature representation from the final global average pooling layer.
3. Calibration & Uncertainty Agent: Calibrates raw network scores via Platt Scaling and estimates prediction trustworthiness via max-probability confidence and normalized binary Shannon entropy.
4. Out-of-Distribution (OOD) Agent: Evaluates whether the radiograph's 1024-dimensional feature vector conforms to the reference in-distribution manifold using Mahalanobis distance solved via Cholesky decomposition.
5. Decision Agent: A deterministic, explainable rule engine implementing PRD FR-5 precedence logic to arbitrate among four operational actions: ACCEPT, REPAIR, ESCALATE, or REJECT.
6. Image Repair Agent: Executes targeted, non-destructive image processing algorithms (CLAHE for exposure, Non-Local Means for noise, Unsharp Masking for blur) in a fixed sequence when repair is indicated.
7. Verification Agent: Audits post-repair radiographs by enforcing five mandatory safety gates across quality recovery, non-degradation of confidence, preservation of OOD status, and diagnostic label stability before authorizing automated release.

END-TO-END DATAFLOW PIPELINE:
Stage 1: Input Radiograph Ingestion → Validated 2D grayscale uint8 array standardized non-destructively.
Stage 2: Initial Screening → Quality Agent evaluates signal metrics. Concurrently, DenseNet-121 computes raw pneumonia score and extracts 1024-D feature vector. OOD Agent calculates Mahalanobis distance. Uncertainty Agent computes calibrated confidence and normalized entropy.
Stage 3: Decision Arbitration → Decision Agent evaluates the composite signal bundle against hierarchical rules R1–R7. If ACCEPT, prediction is released directly. If ESCALATE/REJECT, prediction is withheld and flagged for radiologist review. If REPAIR, workflow advances to Stage 4.
Stage 4: Targeted Image Repair → Repair Agent applies deterministic filtering targeting identified defect flags. Original image array is preserved untouched.
Stage 5: Fresh Post-Repair Inference → The repaired radiograph undergoes a complete fresh inference pass across Quality, DenseNet-121, OOD, and Uncertainty agents. Cached pre-repair signals are strictly never reused.
Stage 6: Multi-Gate Verification → Verification Agent evaluates before-vs-after signal deltas. If all five gates pass, prediction is released as ACCEPTED_AFTER_REPAIR. If any gate fails, the case safely escalates to human review.
"""

SECTION_3_TITLE = "Section 3 — Dataset and Experimental Baseline"
SECTION_3_BODY = """
The system was evaluated on the NIH ChestX-ray14 dataset, comprising 112,120 frontal chest radiographs from 30,805 unique patients.

DATASET SPLIT INTEGRITY & CLASS IMBALANCE:
To prevent data leakage, dataset splitting was performed strictly at the patient level:
- Training Cohort: 78,484 images across 21,563 patients.
- Validation Cohort: 16,912 images across 4,621 patients.
- Final Frozen Test Cohort: 16,724 images across 4,621 patients.

Class imbalance reflects clinical real-world epidemiology: within the 16,724 test cohort, exactly 220 images (1.3155%) are positive for pneumonia, while 16,504 images (98.6845%) are negative. This profound imbalance dictates that positive predictive value (PPV / precision) is mathematically bounded in low single digits, while negative predictive value (NPV) remains exceptionally high (>98.8%).

STANDALONE BASELINE PERFORMANCE (FROZEN DENSENET-121):
Evaluating the standalone DenseNet-121 across the entire uncurated test cohort (N=16,724) at the calibrated F1-optimal operating threshold of 0.522161 establishes the baseline benchmark:
- Total Cases: 16,724 (100% coverage, 0% withheld)
- True Positives (TP): 49
- True Negatives (TN): 15,160
- False Positives (FP): 1,344
- False Negatives (FN): 171
- Accuracy: 90.9412%
- Precision (PPV): 3.5176%
- Recall (Sensitivity): 22.2727%
- Specificity (TNR): 91.8565%
- Negative Predictive Value (NPV): 98.8846%
- F1-Score: 0.060756
- Area Under ROC Curve (AUROC): 0.701619
- Area Under Precision-Recall Curve (AUPRC): 0.028085
- Total Baseline Diagnostic Errors: 1,515 (1,344 FP + 171 FN)
"""

print("Section definitions 1-3 loaded.")
