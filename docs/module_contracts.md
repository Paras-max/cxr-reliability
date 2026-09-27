# Module contracts

Generated from the docstring metadata of each module. Each entry states responsibility, input, output, dependencies and the implementation phase. Scripts and tests are described in their own files. Phases refer to `docs/architecture.md`.


## Configuration

### `src/cxr_reliability/config/__init__.py`

**Configuration package.** (phase P0)

- **Responsibility:** Typed access to environment settings, pipeline wiring and versioned thresholds.
- **Input:** YAML files under configs/ and CXR_* environment variables.
- **Output:** Validated pydantic models (Settings, PipelineConfig, Thresholds).
- **Dependencies:** pydantic, pydantic-settings, pyyaml

### `src/cxr_reliability/config/settings.py`

**Environment-driven runtime settings.** (phase P0)

- **Responsibility:** Read CXR_* environment variables / .env into one validated object: device, seed, directories, active thresholds version, API and tracking settings.
- **Input:** .env file and process environment.
- **Output:** Settings (pydantic BaseSettings); get_settings() returns a cached instance.
- **Dependencies:** pydantic-settings

### `src/cxr_reliability/config/thresholds.py`

**Versioned threshold schema and loader.** (phase P0)

- **Responsibility:** Define every threshold the PRD mentions (quality, OOD, uncertainty, decision, repair, verification). Fields the PRD does not quantify are None until calibrated, so the system can list exactly which values are still unresolved.
- **Input:** A YAML file from configs/thresholds/.
- **Output:** Thresholds model; Thresholds.unresolved() lists dotted names of None fields.
- **Dependencies:** pydantic, pyyaml

### `src/cxr_reliability/config/pipeline_config.py`

**Pipeline wiring configuration schema and loader.** (phase P0)

- **Responsibility:** Model identifiers, loop limits, execution flags and audit settings that are not thresholds. Kept separate so thresholds can be recalibrated without touching wiring.
- **Input:** configs/pipeline.yaml.
- **Output:** PipelineConfig model.
- **Dependencies:** pydantic, pyyaml


## Contracts

### `src/cxr_reliability/contracts/__init__.py`

**Agent input/output contracts.** (phase P0)

- **Responsibility:** Single source of truth for every schema exchanged between agents, the pipeline, the audit log and the API. Contains data definitions only, no behavior.
- **Input:** n/a
- **Output:** pydantic models and enums.
- **Dependencies:** pydantic

### `src/cxr_reliability/contracts/common.py`

**Shared contract primitives.** (phase P0)

- **Responsibility:** Base envelope returned by every agent (score + label + human-readable reasoning), agent names, and the research-prototype disclaimer. Enforces PRD section 4 Explainability: no agent result may have an empty reasoning string.
- **Input:** n/a
- **Output:** AgentResult, AgentName, StrictModel, DISCLAIMER.
- **Dependencies:** pydantic

### `src/cxr_reliability/contracts/quality.py`

**Quality Agent contract (PRD FR-1).** (phase P0)

- **Responsibility:** Schema for blur / noise / exposure measurements, flags and the overall verdict.
- **Input:** n/a
- **Output:** QualityResult, QualityFlags, QualityLevel, DefectType.
- **Dependencies:** contracts.common

### `src/cxr_reliability/contracts/base_model.py`

**Base Model contract (PRD FR-3).** (phase P0)

- **Responsibility:** Schema for the pneumonia classifier output. Feature vectors are passed in memory (models.base_model.ModelForward), not serialized into audit records.
- **Input:** n/a
- **Output:** BaseModelResult.
- **Dependencies:** contracts.common

### `src/cxr_reliability/contracts/uncertainty.py`

**Uncertainty Agent contract (PRD FR-4).** (phase P0)

- **Responsibility:** Schema for confidence and predictive entropy computed from the base model output.
- **Input:** n/a
- **Output:** UncertaintyResult, ConfidenceLevel.
- **Dependencies:** contracts.common

### `src/cxr_reliability/contracts/ood.py`

**OOD Agent contract (PRD FR-2).** (phase P0)

- **Responsibility:** Schema for Mahalanobis distance, energy score and the OOD verdict.
- **Input:** n/a
- **Output:** OODResult, OODLevel.
- **Dependencies:** contracts.common

### `src/cxr_reliability/contracts/decision.py`

**Decision Agent contract (PRD FR-5).** (phase P0)

- **Responsibility:** Schema for the action decision, the rule that fired, the signals that drove it and the margins to each threshold (needed for auditability and the borderline safety rule).
- **Input:** n/a
- **Output:** DecisionResult, DecisionState, Action.
- **Dependencies:** contracts.common

### `src/cxr_reliability/contracts/repair.py`

**Repair Agent contract (PRD FR-6).** (phase P0)

- **Responsibility:** Schema describing which reversible fixes were applied. The repaired pixel array is held in memory, never in the contract; every repaired output is tagged repaired=True.
- **Input:** n/a
- **Output:** RepairResult, RepairStep.
- **Dependencies:** contracts.common, contracts.quality

### `src/cxr_reliability/contracts/verification.py`

**Verification Agent contract (PRD FR-7).** (phase P0)

- **Responsibility:** Schema for the before/after comparison and the release / escalate / reject verdict.
- **Input:** n/a
- **Output:** VerificationResult, SignalBundle, NextStep.
- **Dependencies:** contracts.common

### `src/cxr_reliability/contracts/pipeline.py`

**End-to-end pipeline output contract (PRD FR-8).** (phase P0)

- **Responsibility:** Schema for what the system returns per image: a prediction with a reliability label, or a needs-human-review flag, plus the full agent trace. Enforces that the needs_human_review flag and the reliability label can never disagree.
- **Input:** n/a
- **Output:** PipelineOutput, PredictionSummary, ReliabilityLabel.
- **Dependencies:** contracts.* (all agent contracts)

### `src/cxr_reliability/contracts/audit.py`

**Audit record contract (PRD section 4, Auditability).** (phase P0)

- **Responsibility:** Schema for one per-inference log entry: signals, action taken, verification deltas, and provenance (thresholds version, model ids, weight hashes, input hash).
- **Input:** n/a
- **Output:** AuditRecord, ExecutionPath.
- **Dependencies:** contracts.pipeline


## Dataset utilities

### `src/cxr_reliability/data/__init__.py`

**Dataset utilities package.** (phase P1)

- **Responsibility:** Loading, splitting, manifesting, transforming and synthetically corrupting datasets.
- **Input:** n/a
- **Output:** n/a
- **Dependencies:** pandas, numpy, torch (inside submodules)

### `src/cxr_reliability/data/nih.py`

**NIH ChestX-ray14 loader (PRD section 6).** (phase P1)

- **Responsibility:** Parse Data_Entry_2017.csv, derive the binary Pneumonia label, expose a torch Dataset over the image folders, and read the official train/val and test list files.
- **Input:** Path to Data_Entry_2017.csv and the images_001..012 folders; official split list files
- **Output:** metadata DataFrame (filename, labels, patient_id, age, gender, view); binary label Series (Pneumonia vs rest); torch Dataset yielding (image, label, meta)
- **Dependencies:** pandas, numpy, torch, pillow, data.transforms

### `src/cxr_reliability/data/external_ood.py`

**Natural OOD datasets: CheXpert and PadChest (PRD section 6).** (phase P1)

- **Responsibility:** Load frontal-view samples from CheXpert and PadChest for the natural OOD test of the NIH-only fast model. Optional far-OOD (non-chest) set is a proposed addition.
- **Input:** Local dataset folders and their metadata files (access-approved downloads).
- **Output:** Datasets yielding (image, source_name).
- **Dependencies:** pandas, torch, pillow, data.transforms

### `src/cxr_reliability/data/splits.py`

**Patient-level dataset splits.** (phase P1)

- **Responsibility:** Build ID-fit, ID-val (calibration) and test splits with zero patient overlap, and provide the overlap check used by the data tests.
- **Input:** NIH metadata DataFrame (must contain patient_id), seed, split fractions.
- **Output:** SplitManifest (dict of split name to filename list) and overlap assertions.
- **Dependencies:** pandas, numpy, scikit-learn (GroupShuffleSplit), data.manifests

### `src/cxr_reliability/data/manifests.py`

**Split manifest I/O with content hashes.** (phase P1)

- **Responsibility:** Write and read split manifests and compute a stable content hash so every result can reference the exact data split it used (PRD section 4, Reproducibility).
- **Input:** SplitManifest, output path.
- **Output:** Manifest CSV/JSON files and their SHA-256.
- **Dependencies:** hashlib, json, pandas

### `src/cxr_reliability/data/transforms.py`

**Image preprocessing.** (phase P1)

- **Responsibility:** Produce the TorchXRayVision-normalized model input for each model, and the standardized grayscale view used by the Quality and Repair agents.
- **Input:** Raw image array (H x W, uint8 or float) and target size.
- **Output:** Model-ready tensor/array; standardized grayscale uint8 array.
- **Dependencies:** numpy, torchxrayvision, opencv-python-headless

### `src/cxr_reliability/data/corruptions.py`

**Synthetic corruption library with known severity.** (phase P3)

- **Responsibility:** Generate seeded blur, noise, exposure and combined degradations at graded severities. Provides the ground truth needed to calibrate Quality thresholds (ROC), Repair bounds and the Verification margin, since NIH has no natural defect labels.
- **Input:** Clean image array, defect type, severity, seed.
- **Output:** Corrupted image array plus a record of the exact parameters applied.
- **Dependencies:** numpy, opencv-python-headless, scikit-image

### `src/cxr_reliability/data/rare_valid.py`

**Rare-but-valid case selection (PRD section 7).** (phase P1)

- **Responsibility:** Define and select in-distribution cases with rare label combinations or unusual strata, so the OOD false-positive rate can be reported separately from true distribution shift.
- **Input:** NIH metadata DataFrame, selection criteria (approved by the project owner).
- **Output:** List of filenames forming the rare-but-valid evaluation set.
- **Dependencies:** pandas


## Base Model and Escalation Model (FR-3)

### `src/cxr_reliability/models/__init__.py`

**Model package.** (phase P2)

- **Responsibility:** Loading pretrained TorchXRayVision models, extracting mid-layer features, and the two model wrappers (fast Base Model and Escalation model).
- **Input:** n/a
- **Output:** n/a
- **Dependencies:** torch, torchxrayvision (inside submodules)

### `src/cxr_reliability/models/txv_loader.py`

**TorchXRayVision weight loading.** (phase P2)

- **Responsibility:** Load a pretrained model by id, return it in eval mode on the requested device, and verify the weight file hash. Pinned to the versions recorded in the lockfile.
- **Input:** model_id (e.g. densenet121-res224-nih), weights directory, device.
- **Output:** torch.nn.Module in eval mode; weights SHA-256 string.
- **Dependencies:** torch, torchxrayvision

### `src/cxr_reliability/models/feature_hook.py`

**Mid-layer feature extraction.** (phase P2)

- **Responsibility:** Attach a forward hook to a chosen layer and return a pooled feature vector in the same forward pass that produces the logits (one pass feeds Uncertainty and OOD).
- **Input:** Model, layer name.
- **Output:** Context manager that yields a pooled feature vector per batch.
- **Dependencies:** torch

### `src/cxr_reliability/models/base_model.py`

**Base Model agent (PRD FR-3): fast pneumonia classifier.** (phase P2)

- **Responsibility:** Run one forward pass of densenet121-res224-nih, return the Pneumonia output, the audit-only outputs of all pathologies, and the mid-layer feature vector for the OOD Agent. No fine-tuning: the pretrained model is used as-is.
- **Input:** Preprocessed image tensor from data.transforms.
- **Output:** ModelForward: BaseModelResult contract + raw logits + feature vector (in memory only).
- **Dependencies:** torch, models.txv_loader, models.feature_hook, contracts.base_model

### `src/cxr_reliability/models/escalation.py`

**Escalation model (PRD FR-5 'Escalate to larger model').** (phase P2)

- **Responsibility:** Run resnet50-res512-all on the same image when the Decision Agent escalates. Same output contract as the Base Model. Note: this model was trained on NIH + CheXpert + PadChest + MIMIC, so it must not be used as the reference for the natural OOD test.
- **Input:** Original or repaired image at 512 resolution.
- **Output:** ModelForward (same shape as the fast model's output).
- **Dependencies:** torch, models.txv_loader, models.base_model


## Agents (FR-1, FR-2, FR-4, FR-5, FR-6, FR-7)

### `src/cxr_reliability/agents/__init__.py`

**Agent package (Quality, OOD, Uncertainty, Decision, Repair, Verification).** (phase P3-P6)

- **Responsibility:** Hold the six non-model agents. The seventh agent, Base Model, lives in the models package because it wraps a neural network. Agents are deterministic except the optional learned Decision Agent (v2).
- **Input:** n/a
- **Output:** n/a
- **Dependencies:** contracts

### `src/cxr_reliability/agents/base.py`

**Common agent base class.** (phase P0)

- **Responsibility:** Give every agent a stable name and version and a place to attach thresholds. Each concrete agent defines its own typed run() because inputs differ per agent.
- **Input:** n/a
- **Output:** AgentBase abstract class.
- **Dependencies:** contracts.common

### `src/cxr_reliability/agents/quality.py`

**Quality Agent (PRD FR-1).** (phase P3)

- **Responsibility:** Detect blur (Laplacian variance), noise (SNR in dB) and exposure problems (mean intensity and histogram spread) before they contaminate the diagnosis. Also compute blur_pct = 100 * (1 - min(laplacian_variance / reference_max_variance, 1)). The noise residual definition is not in the PRD and must be fixed and documented in Phase 3.
- **Input:** Standardized grayscale image (numpy array).
- **Output:** QualityResult with per-defect flags, overall good/poor, repairable, near_threshold, and a human-readable reasoning string.
- **Dependencies:** numpy, opencv-python-headless, scikit-image, config.thresholds.QualityThresholds, contracts.quality

### `src/cxr_reliability/agents/ood.py`

**OOD Agent (PRD FR-2).** (phase P4)

- **Responsibility:** Primary: Mahalanobis distance of the base model's mid-layer feature vector from the in-distribution mean and covariance. Cross-check: energy score from logits. Returns in-distribution / borderline / severe. Fitted statistics come from calibration; this class only scores.
- **Input:** Feature vector and logits from ModelForward; fitted statistics artifact (mean, covariance)
- **Output:** OODResult with distance, energy, percentile vs in-distribution, level, detector agreement.
- **Dependencies:** numpy, scipy, config.thresholds.OODThresholds, calibration.ood_fit (stats artifact), contracts.ood

### `src/cxr_reliability/agents/uncertainty.py`

**Uncertainty Agent (PRD FR-4).** (phase P4)

- **Responsibility:** Compute confidence and predictive entropy as pure post-processing of the Pneumonia output (no separate model). Multi-label model: binary form, confidence = max(p, 1-p).
- **Input:** Pneumonia logit (or probability) from BaseModelResult.
- **Output:** UncertaintyResult with confidence, binary entropy, high/borderline/low level.
- **Dependencies:** numpy, config.thresholds.UncertaintyThresholds, contracts.uncertainty

### `src/cxr_reliability/agents/decision/__init__.py`

**Decision Agent package (PRD FR-5).** (phase P6)

- **Responsibility:** Combine quality, OOD and confidence signals into Accept / Repair / Escalate / Reject. v1 is a fixed, explainable rule table; v2 (stretch) is a small learned weighting.
- **Input:** n/a
- **Output:** n/a
- **Dependencies:** agents.decision.interface

### `src/cxr_reliability/agents/decision/interface.py`

**Decision Agent interface.** (phase P6)

- **Responsibility:** Define the single method both v1 and v2 implement, so they can be swapped and ablated without touching the orchestrator.
- **Input:** QualityResult; OODResult; UncertaintyResult; DecisionState (attempt counters)
- **Output:** DecisionResult with action, rule id, driving signals and threshold margins.
- **Dependencies:** contracts.*

### `src/cxr_reliability/agents/decision/rules.py`

**Decision Agent v1: rule table.** (phase P6)

- **Responsibility:** Deterministic precedence-ordered rules. Any signal within borderline_margin of its threshold fails toward Escalate/Reject, never silent Accept (PRD section 4). Every decision reports the rule id and margins. Precedence order is documented in docs/architecture.md and needs owner sign-off.
- **Input:** QualityResult, OODResult, UncertaintyResult; DecisionState; DecisionThresholds; PipelineConfig.execution.borderline_action
- **Output:** DecisionResult.
- **Dependencies:** config.thresholds, config.pipeline_config, contracts.decision

### `src/cxr_reliability/agents/decision/learned.py`

**Decision Agent v2 (stretch): learned weighting.** (phase P7b (stretch))

- **Responsibility:** Small scikit-learn model (logistic regression or shallow MLP) over the three scalar signals, trained to trade off accuracy, cost and safety. Not to be claimed as a contribution unless an ablation shows it beats v1 (PRD section 11).
- **Input:** Scalar signals per sample (quality score, OOD score, confidence); correctness labels; cost model (owner-defined)
- **Output:** Trained model artifact; DecisionResult at inference.
- **Dependencies:** scikit-learn, agents.decision.interface

### `src/cxr_reliability/agents/repair.py`

**Repair Agent (PRD FR-6).** (phase P5)

- **Responsibility:** Apply targeted, reversible fixes only: unsharp masking or Wiener deconvolution (blur), non-local means (noise), CLAHE (exposure). Never invents anatomy. Every output is tagged repaired=True and is never trusted like a clean original. Refuses defects outside the calibrated repair bounds.
- **Input:** Standardized grayscale image; QualityResult (which defects, how severe)
- **Output:** Repaired image array (in memory) and RepairResult describing each step.
- **Dependencies:** opencv-python-headless, scikit-image, config.thresholds.RepairThresholds, contracts.repair

### `src/cxr_reliability/agents/verification.py`

**Verification Agent (PRD FR-7).** (phase P6)

- **Responsibility:** Confirm that a repair or escalation actually helped by comparing signals before and after. Accept only if confidence improves by at least the calibrated margin (PRD starting value 0.15); otherwise escalate further or reject. Also reports OOD delta and label flips so a marginal or suspicious change is not silently trusted.
- **Input:** SignalBundle before; SignalBundle after; VerificationThresholds
- **Output:** VerificationResult with verified flag, deltas and next step (release / escalate / reject).
- **Dependencies:** config.thresholds.VerificationThresholds, contracts.verification


## Pipeline Orchestrator (FR-8)

### `src/cxr_reliability/pipeline/__init__.py`

**Pipeline orchestration package (PRD FR-8).** (phase P6)

- **Responsibility:** Coordinate the seven agents end to end. Hand-rolled orchestrator per PRD section 5.
- **Input:** n/a
- **Output:** n/a
- **Dependencies:** agents, models, audit

### `src/cxr_reliability/pipeline/orchestrator.py`

**Pipeline Orchestrator (PRD FR-8).** (phase P6)

- **Responsibility:** Run Quality in parallel with the shared base-model forward pass, then Uncertainty and OOD, then Decision. On Repair or Escalate, run the action, re-score, then Verification before release. Enforce loop limits so every path terminates, tag repaired/escalated outputs, assemble PipelineOutput, and hand the record to the audit logger.
- **Input:** Image array; optional input_id; PipelineConfig and Thresholds; the seven agent instances
- **Output:** PipelineOutput (prediction + reliability label, or needs-human-review flag) and one audit record.
- **Dependencies:** all agents, models.base_model/escalation, audit.audit_log, pipeline.budget, contracts.pipeline

### `src/cxr_reliability/pipeline/budget.py`

**Latency and compute-cost accounting.** (phase P6)

- **Responsibility:** Time each stage and label the path (fast vs escalated) so the compute-cost metric in PRD section 7 can be computed. The budget value itself is set by the project owner after baseline measurement.
- **Input:** Stage names and callables / timing contexts.
- **Output:** Per-stage latency dict (ms) and total.
- **Dependencies:** time (stdlib)


## Logging / Auditing

### `src/cxr_reliability/audit/__init__.py`

**Logging and auditing package (PRD section 4).** (phase P6)

- **Responsibility:** Per-inference audit records and provenance hashing.
- **Input:** n/a
- **Output:** n/a
- **Dependencies:** contracts.audit

### `src/cxr_reliability/audit/audit_log.py`

**Per-inference audit logger.** (phase P6)

- **Responsibility:** Append one AuditRecord per inference to a JSONL file (action taken, driving signals, verification deltas, latency, provenance) and read records back for analysis and the dashboard. Writes must be atomic per line and never silently dropped.
- **Input:** AuditRecord.
- **Output:** JSONL file under logs/audit; iterator over stored records.
- **Dependencies:** contracts.audit, json, pathlib

### `src/cxr_reliability/audit/provenance.py`

**Provenance hashing.** (phase P6)

- **Responsibility:** Hash input images, config files and weight files so every audit record and result ties back to exact inputs and versions.
- **Input:** Bytes, array, or file path.
- **Output:** SHA-256 hex strings; software version dict.
- **Dependencies:** hashlib


## Calibration utilities

### `src/cxr_reliability/calibration/__init__.py`

**Calibration utilities package.** (phase P3-P5)

- **Responsibility:** Every threshold that is a starting point in the PRD is calibrated here on held-out data, never assumed. Outputs a new versioned thresholds file.
- **Input:** n/a
- **Output:** n/a
- **Dependencies:** data, models, agents, evaluation.metrics

### `src/cxr_reliability/calibration/quality_roc.py`

**Quality threshold calibration.** (phase P3)

- **Responsibility:** ROC analysis of blur, noise and exposure statistics against synthetic corruptions of known severity, plus the reference_max_laplacian_var percentile on clean images.
- **Input:** ID-val images and the corruption library.
- **Output:** Calibrated QualityThresholds values, ROC curves and AUROC per defect.
- **Dependencies:** scikit-learn, numpy, data.corruptions, agents.quality

### `src/cxr_reliability/calibration/ood_fit.py`

**OOD statistics fitting and threshold calibration.** (phase P4)

- **Responsibility:** Fit the in-distribution feature mean and (shrinkage) covariance on the ID-fit split, save them as an artifact, then set the borderline / severe distance and energy thresholds from percentiles on ID-val and check them against the natural OOD set.
- **Input:** Feature vectors from the fast model on ID-fit and ID-val; CheXpert / PadChest features.
- **Output:** Statistics artifact (npz) and calibrated OODThresholds values.
- **Dependencies:** numpy, scipy, scikit-learn (Ledoit-Wolf), models.base_model

### `src/cxr_reliability/calibration/confidence_cal.py`

**Confidence calibration and cut selection.** (phase P4)

- **Responsibility:** Fit a temperature (or Platt) scaling on the Pneumonia logit for the ECE comparison, and choose high / low confidence cuts and the positive-class operating point from risk-coverage analysis on ID-val.
- **Input:** Pneumonia logits and labels on ID-val.
- **Output:** Temperature value and UncertaintyThresholds values.
- **Dependencies:** scikit-learn, scipy, numpy

### `src/cxr_reliability/calibration/repair_bounds.py`

**Repair reversibility bounds (PRD section 11).** (phase P5)

- **Responsibility:** Measure recovery versus corruption severity to decide how much blur / noise / exposure error is repairable versus straight to Reject.
- **Input:** Corrupted-but-labeled images and the Repair Agent.
- **Output:** Calibrated RepairThresholds values and recovery-vs-severity curves.
- **Dependencies:** data.corruptions, agents.repair, evaluation.repair_recovery

### `src/cxr_reliability/calibration/verify_margin.py`

**Verification margin calibration.** (phase P5)

- **Responsibility:** Sweep the confidence-gain margin on corrupted labeled images and choose the value that balances accepting genuine improvements against accepting noisy ones.
- **Input:** Before/after signal bundles with correctness labels.
- **Output:** Calibrated VerificationThresholds values.
- **Dependencies:** numpy, contracts.verification

### `src/cxr_reliability/calibration/registry.py`

**Threshold registry: freeze and version.** (phase P3-P5)

- **Responsibility:** Combine calibrated pieces into one Thresholds object, write it as configs/thresholds/v1_calibrated_<hash>.yaml with provenance, and refuse edits to a frozen file. Frozen thresholds are required before any test-set evaluation.
- **Input:** Calibrated threshold sections and calibration run metadata.
- **Output:** Versioned thresholds YAML, its hash, and an entry in docs/calibration_log.md.
- **Dependencies:** config.thresholds, hashlib, pyyaml


## Evaluation

### `src/cxr_reliability/evaluation/__init__.py`

**Evaluation package (PRD section 7).** (phase P7)

- **Responsibility:** System-level reliability metrics, baselines and ablations.
- **Input:** n/a
- **Output:** n/a
- **Dependencies:** numpy, pandas, scikit-learn

### `src/cxr_reliability/evaluation/metrics.py`

**Core classification and selective-prediction metrics.** (phase P2)

- **Responsibility:** AUROC, AUPRC, sensitivity at fixed specificity, risk-coverage curves and patient-level bootstrap confidence intervals.
- **Input:** Labels, scores, patient ids.
- **Output:** Metric values with confidence intervals.
- **Dependencies:** scikit-learn, numpy

### `src/cxr_reliability/evaluation/calibration_metrics.py`

**Calibration error metrics.** (phase P4)

- **Responsibility:** ECE and reliability-diagram data for raw, temperature-scaled and Accept-bucket confidences, reported separately so selection and calibration effects are not mixed.
- **Input:** Probabilities and labels (optionally filtered by action).
- **Output:** ECE values and binned reliability data.
- **Dependencies:** numpy

### `src/cxr_reliability/evaluation/action_breakdown.py`

**Per-action accuracy breakdown (PRD section 7).** (phase P7)

- **Responsibility:** Accuracy, error rate, sensitivity and specificity within each action bucket (Accept / Repair / Escalate / Reject) with counts, and the share of hard cases moved out of Accept, so easy-subset accuracy alone is never reported.
- **Input:** Per-image PipelineOutput or audit records plus ground-truth labels.
- **Output:** Table by action with counts and confidence intervals.
- **Dependencies:** pandas, evaluation.metrics

### `src/cxr_reliability/evaluation/ood_eval.py`

**OOD evaluation (PRD section 7).** (phase P4)

- **Responsibility:** AUROC of Mahalanobis, energy and combined scores on NIH versus CheXpert/PadChest, and the false-positive OOD rate on rare-but-valid cases, reported separately.
- **Input:** Scores on in-distribution, natural OOD and rare-but-valid sets.
- **Output:** AUROC per detector; false-positive rate at the calibrated threshold.
- **Dependencies:** scikit-learn, evaluation.metrics

### `src/cxr_reliability/evaluation/repair_recovery.py`

**Repair recovery measurement (PRD section 7).** (phase P5)

- **Responsibility:** recovery_pct = (variance_after_repair - variance_blurred) / (variance_original - variance_blurred) x 100 on deliberately corrupted images, plus SSIM / PSNR versus the original and whether the base-model prediction recovers, because Laplacian variance alone can be inflated by amplified noise. Failure cases must be reported.
- **Input:** Original, corrupted and repaired images with known labels.
- **Output:** Recovery percentages and curves by severity.
- **Dependencies:** scikit-image, numpy, agents.repair

### `src/cxr_reliability/evaluation/cost.py`

**Compute cost evaluation.** (phase P7)

- **Responsibility:** Median and p95 latency, fast path versus escalated path, CPU and GPU.
- **Input:** Audit records with per-stage latency.
- **Output:** Cost table.
- **Dependencies:** pandas, pipeline.budget

### `src/cxr_reliability/evaluation/baselines.py`

**Baselines.** (phase P7)

- **Responsibility:** B0 raw model always answers; B1 confidence-threshold reject only; B2 always escalate. The full system (B3, and B4 with Decision v2) is compared against these.
- **Input:** Model outputs on the evaluation split.
- **Output:** Per-baseline prediction/abstention arrays.
- **Dependencies:** models, evaluation.metrics

### `src/cxr_reliability/evaluation/ablations.py`

**Ablations.** (phase P7 / P7b)

- **Responsibility:** Remove one agent at a time (quality, OOD, uncertainty, verification) and compare Decision v1 with v2. Required before any claim about the learned Decision Agent.
- **Input:** Evaluation records under each configuration.
- **Output:** Ablation table.
- **Dependencies:** evaluation.action_breakdown

### `src/cxr_reliability/evaluation/tracking.py`

**Experiment tracking wrapper.** (phase P0)

- **Responsibility:** Thin MLflow wrapper so calibration and evaluation runs log parameters, thresholds hash, manifest hash and metrics uniformly.
- **Input:** Run name, params, metrics, artifact paths.
- **Output:** MLflow run.
- **Dependencies:** mlflow, config.settings

### `src/cxr_reliability/evaluation/report.py`

**Evaluation report assembly.** (phase P7)

- **Responsibility:** Compile metrics, figures and tables into outputs/ for the dashboard and docs. Must include failure cases and the checked pre-registered targets.
- **Input:** Metric tables and figures.
- **Output:** Report files under outputs/evaluation.
- **Dependencies:** matplotlib, seaborn, pandas


## FastAPI backend

### `src/cxr_reliability/api/__init__.py`

**FastAPI backend package (PRD section 5).** (phase P8)

- **Responsibility:** Lightweight REST layer to submit an image and receive a prediction, reliability label and reasoning trace. Research demo only.
- **Input:** n/a
- **Output:** n/a
- **Dependencies:** fastapi, pipeline

### `src/cxr_reliability/api/schemas.py`

**API request/response schemas.** (phase P8)

- **Responsibility:** Wire-level response models. The prediction payload reuses PipelineOutput so the API cannot drift from the pipeline contract. Every response carries the disclaimer.
- **Input:** n/a
- **Output:** PredictResponse, HealthResponse, ConfigResponse, ErrorResponse.
- **Dependencies:** pydantic, contracts.pipeline

### `src/cxr_reliability/api/dependencies.py`

**API dependency wiring.** (phase P8)

- **Responsibility:** Build the pipeline once at startup (models loaded once, not per request) and expose it to routes. Central place for upload size and content-type validation.
- **Input:** Settings, thresholds and pipeline configs.
- **Output:** A ready ReliabilityPipeline; validated image arrays.
- **Dependencies:** config, pipeline.orchestrator, fastapi

### `src/cxr_reliability/api/routes.py`

**API routes.** (phase P8)

- **Responsibility:** POST /v1/predict, GET /v1/health, GET /v1/config, GET /v1/audit/{id}. Uploads are not persisted by default. Docs state the API is for public research datasets only.
- **Input:** Multipart image upload; audit id.
- **Output:** PredictResponse / HealthResponse / ConfigResponse / AuditRecord.
- **Dependencies:** fastapi, api.schemas, api.dependencies, audit.audit_log

### `src/cxr_reliability/api/main.py`

**FastAPI application factory.** (phase P8)

- **Responsibility:** Create the FastAPI app, register routes, load the pipeline at startup. Run with uvicorn cxr_reliability.api.main:create_app --factory.
- **Input:** Environment settings.
- **Output:** FastAPI application.
- **Dependencies:** fastapi, api.routes, api.dependencies, config.settings


## Streamlit dashboard

### `src/cxr_reliability/dashboard/__init__.py`

**Streamlit dashboard package.** (phase P8)

- **Responsibility:** Demo and analysis UI: single-image agent trace, evaluation results, calibration viewer, audit explorer.
- **Input:** n/a
- **Output:** n/a
- **Dependencies:** streamlit

### `src/cxr_reliability/dashboard/components.py`

**Shared dashboard components.** (phase P8)

- **Responsibility:** Persistent research-prototype banner (required on every page), agent-trace rendering and common plots.
- **Input:** PipelineOutput or evaluation tables.
- **Output:** Streamlit widgets.
- **Dependencies:** streamlit, matplotlib, contracts.common.DISCLAIMER

### `src/cxr_reliability/dashboard/app.py`

**Streamlit entry point.** (phase P8)

- **Responsibility:** Landing page and navigation. Run with streamlit run src/cxr_reliability/dashboard/app.py.
- **Input:** n/a
- **Output:** Streamlit app.
- **Dependencies:** streamlit, dashboard.components

### `src/cxr_reliability/dashboard/pages/1_Single_Image.py`

**Single-image demo page.** (phase P8)

- **Responsibility:** Upload a public research image and show each agent's score, label and reasoning, the action taken, and original-versus-repaired comparison.
- **Input:** Uploaded image.
- **Output:** Agent trace and reliability label.
- **Dependencies:** api or pipeline

### `src/cxr_reliability/dashboard/pages/2_Evaluation.py`

**Evaluation results page.** (phase P8)

- **Responsibility:** Per-action accuracy table, OOD ROC curves, risk-coverage, reliability diagram, cost.
- **Input:** Files under outputs/evaluation.
- **Output:** Tables and figures.
- **Dependencies:** evaluation.report outputs

### `src/cxr_reliability/dashboard/pages/3_Calibration.py`

**Threshold calibration viewer.** (phase P8)

- **Responsibility:** Show calibrated thresholds versus PRD starting points, ROC curves, and the list of still-unresolved thresholds.
- **Input:** Thresholds YAML files and calibration outputs.
- **Output:** Tables and figures.
- **Dependencies:** config.thresholds

### `src/cxr_reliability/dashboard/pages/4_Audit_Explorer.py`

**Audit log explorer.** (phase P8)

- **Responsibility:** Filter and inspect per-inference audit records: signals, action, verification deltas.
- **Input:** logs/audit JSONL.
- **Output:** Filterable table and record detail view.
- **Dependencies:** audit.audit_log


## Reporting (optional)

### `src/cxr_reliability/reporting/__init__.py`

**Optional report generation package.** (phase Optional)

- **Responsibility:** Turn structured pipeline output into human-readable text. Downstream of the Decision Agent only. No LLM is used in the diagnostic path.
- **Input:** n/a
- **Output:** n/a
- **Dependencies:** contracts.pipeline

### `src/cxr_reliability/reporting/template_report.py`

**Template-based report writer.** (phase Optional)

- **Responsibility:** Render PipelineOutput into a readable summary from fixed templates. An LLM adapter is intentionally not scaffolded: the PRD allows one only as an optional report writer, and it needs separate owner approval.
- **Input:** PipelineOutput.
- **Output:** Plain-text or markdown summary containing the disclaimer.
- **Dependencies:** contracts.pipeline
