# 🏛️ SYSTEM BLOCK DIAGRAM
## Reliability-Aware Multi-Agent System for Chest X-Ray Pneumonia Classification

> **Purpose:** Structural architectural block diagram, component decomposition, hardware/software stack, and module interactions.

---

## 1. Complete System Architecture Block Diagram (Mermaid)

```mermaid
graph TB
    %% User and Presentation Layer
    subgraph UI_Layer ["1. Presentation & Interaction Layer"]
        CLI["CLI Tool / Evaluation Scripts<br/>(scripts/run_pipeline.py)"]
        StreamlitApp["Streamlit Dashboard<br/>(src/cxr_reliability/dashboard/app.py)"]
        BatchEval["Batch Benchmark Evaluator<br/>(16,724 Test Runner)"]
    end

    %% Pipeline Orchestration Layer
    subgraph Orchestration_Layer ["2. Orchestration & State Management Layer"]
        Orchestrator["Pipeline Orchestrator<br/>(orchestrator.py)"]
        ConfigManager["Config & Threshold Manager<br/>(pipeline.yaml, thresholds.yaml)"]
        StateTracker["PipelineResult & PipelineOutput<br/>(Pydantic Data Schemas)"]
    end

    %% Multi-Agent Operational Layer
    subgraph Agent_Layer ["3. Multi-Agent Reliability Subsystems"]
        Agent_Quality["Quality Agent<br/>(evaluator.py)<br/>• Blur (Laplacian)<br/>• Noise (MAD SNR)<br/>• Exposure"]
        Agent_BaseModel["Base Model Agent<br/>(DenseNet-121)<br/>• 1024-D GAP Features<br/>• Raw Pneumonia Sigmoid"]
        Agent_OOD["OOD Agent<br/>(mahalanobis.py)<br/>• Cholesky Solver<br/>• D_M Distance"]
        Agent_Uncertainty["Uncertainty Agent<br/>(estimator.py)<br/>• Platt Calibration<br/>• Normalized Binary Entropy"]
        Agent_Decision["Decision Agent<br/>(rules.py)<br/>• Rule Matrix R0-R7<br/>• Priority Action Selection"]
        Agent_Repair["Repair Agent<br/>(repair/pipeline.py)<br/>• CLAHE Filter<br/>• NL-Means Denoising<br/>• Unsharp Masking"]
        Agent_Verification["Verification Agent<br/>(verification.py)<br/>• 5 Safety Gates<br/>• Anti-Hallucination Guard"]
    end

    %% Scientific & Pretrained Artifacts Layer
    subgraph Storage_Layer ["4. Artifacts & Reference Data Layer"]
        OOD_Stats[("OOD Reference Stats<br/>(reference_stats.npz)<br/>1024-D Mean & Covariance<br/>from 78,299 Train Images")]
        PretrainedModel[("Pretrained Weights<br/>densenet121-res224-nih<br/>TorchXRayVision")]
        CalibModel[("Platt Scaling Parameters<br/>(calibration_results.json)<br/>Logistic Calibrator")]
    end

    %% Clinical Output Gate
    subgraph Output_Layer ["5. Decision & Clinical Routing Layer"]
        Output_Release["✅ ACCEPTED<br/>Automated Diagnostic Release<br/>(High Confidence, Clear, Familiar)"]
        Output_Review["⚠️ WITHHELD<br/>Escalate for Human Radiologist Review<br/>(Uncertain / Borderline / Degraded)"]
        Output_Reject["🛑 REJECTED<br/>Block Corrupted / Severe OOD Image<br/>(Wrong Anatomy, Heavy Artifacts)"]
    end

    %% Layer Interconnections
    UI_Layer --> Orchestrator
    ConfigManager --> Orchestrator
    Orchestrator <--> StateTracker

    Orchestrator --> Agent_Quality
    Orchestrator --> Agent_BaseModel
    Orchestrator --> Agent_OOD
    Orchestrator --> Agent_Uncertainty
    Orchestrator --> Agent_Decision
    Orchestrator --> Agent_Repair
    Orchestrator --> Agent_Verification

    PretrainedModel -.-> Agent_BaseModel
    OOD_Stats -.-> Agent_OOD
    CalibModel -.-> Agent_Uncertainty

    Agent_Decision --> Output_Release
    Agent_Decision --> Output_Review
    Agent_Decision --> Output_Reject
    Agent_Verification --> Output_Release
    Agent_Verification --> Output_Review

    %% Styling
    classDef uiStyle fill:#e8f4fd,stroke:#0d6efd,stroke-width:2px;
    classDef orchStyle fill:#f3e8fd,stroke:#6f42c1,stroke-width:2px;
    classDef agentStyle fill:#e6f9f0,stroke:#20c997,stroke-width:2px;
    classDef storeStyle fill:#fff3cd,stroke:#ffc107,stroke-width:2px;
    classDef outStyle fill:#fdf0ed,stroke:#fd7e14,stroke-width:2px;

    class CLI,StreamlitApp,BatchEval uiStyle;
    class Orchestrator,ConfigManager,StateTracker orchStyle;
    class Agent_Quality,Agent_BaseModel,Agent_OOD,Agent_Uncertainty,Agent_Decision,Agent_Repair,Agent_Verification agentStyle;
    class OOD_Stats,PretrainedModel,CalibModel storeStyle;
    class Output_Release,Output_Review,Output_Reject outStyle;
```

---

## 2. Text / ASCII Block Diagram

```text
========================================================================================
                          PRESENTATION & CLIENT LAYER
      [ Streamlit Dashboard ]   [ CLI Scripts ]   [ Pytest Test Suite (448 Tests) ]
===========================================┬============================================
                                           │ Request / Image File
                                           ▼
========================================================================================
                         PIPELINE ORCHESTRATOR LAYER
      • Central State Coordinator (`ReliabilityPipeline`)
      • Configuration & Threshold Injector (`pipeline.yaml`, `thresholds.yaml`)
      • Immutable Pydantic Schemas (`PipelineResult`, `PipelineOutput`)
===========================================┬============================================
                                           │
    ┌────────────────┬─────────────────────┼─────────────────────┬──────────────────┐
    ▼                ▼                     ▼                     ▼                  ▼
┌──────────────┐ ┌───────────────┐ ┌───────────────┐ ┌────────────────┐ ┌───────────────┐
│ QUALITY      │ │ BASE MODEL    │ │ UNCERTAINTY   │ │ OOD DETECTOR   │ │ DECISION      │
│ EVALUATOR    │ │ (DenseNet-121)│ │ ESTIMATOR     │ │ (Mahalanobis)  │ │ RULE ENGINE   │
├──────────────┤ ├───────────────┤ ├───────────────┤ ├────────────────┤ ├───────────────┤
│ • Laplacian  │ │ • 1024-D GAP  │ │ • Platt Calib │ │ • 78k Dataset  │ │ • Rules R0-R7 │
│   Blur Var   │ │   Features    │ │ • Conf = max  │ │   Statistics   │ │ • Priority    │
│ • MAD SNR    │ │ • Pneumonia   │ │ • Binary      │ │ • Regularized  │ │   Resolution  │
│ • Exposure   │ │   Sigmoid     │ │   Entropy     │ │   Cholesky     │ │ • Action:     │
│   Bounds     │ │   Head (Idx 8)│ │ • Thresholds  │ │ • D_M Metric   │ │   Acc/Esc/Rej │
└──────────────┘ └───────────────┘ └───────────────┘ └────────────────┘ └───────┬───────┘
                                                                                │
                                           ┌────────────────────────────────────┘
                                           │ IF Action == REPAIR
                                           ▼
========================================================================================
                       REPAIR & VERIFICATION SAFETY LOOP
┌──────────────────────────────────────────────────┐ ┌──────────────────────────────────┐
│ REPAIR AGENT                                     │ │ VERIFICATION AGENT               │
├──────────────────────────────────────────────────┤ ├──────────────────────────────────┤
│ 1. CLAHE (Local Contrast & Exposure Normalizer)  │ │ Gate 1: Repair was applied       │
│ 2. NL-Means (High-Fidelity Edge-Safe Denoising)  │ │ Gate 2: No label flip (p <=> 0.5)│
│ 3. Unsharp Masking (Frequency Edge Sharpening)   │ │ Gate 3: OOD remains IN-DIST      │
│                                                  │ │ Gate 4: Quality improves to GOOD │
│ Output: Non-destructive in-memory image tensor   │ │ Gate 5: Delta Conf & SNR guards  │
└──────────────────────────────────────────────────┘ └─────────────────┬────────────────┘
                                                                       │
=======================================================================╪================
                         CLINICAL ROUTING OUTPUTS                      │
                                                                       ▼
  [ ✅ RELEASE PREDICTION ]             [ ⚠️ WITHHOLD: ESCALATE ]     [ 🛑 REJECT ]
  • Direct Accept (R6)                  • Uncertain Model (R4)         • Severe OOD (R1)
  • Verified Repaired (Gate 1-5 Passed) • Borderline OOD (R2)          • Corrupted (R0)
  • Acc: 92.31%, NPV: 99.02%            • Unverified Repair (Gate Fail)
========================================================================================
```

---

## 3. Layer-by-Layer Architectural Breakdown

### Layer 1: Presentation & Client Interface
- **Streamlit Web UI (`app.py`):** Interactive frontend allowing radiologists/evaluators to upload an X-ray, inspect real-time agent dials, view before/after repair image overlays, and inspect gate audits.
- **Batch Evaluation Harness:** Evaluates massive datasets (e.g., 16,724 images across 8 hours) with zero memory leaks.
- **Pytest Suite:** 448 unit/integration tests confirming deterministic behavior and schema invariants.

### Layer 2: Orchestration & State Management
- **Pipeline Orchestrator:** Implements the Facade and Chain of Responsibility patterns, executing agents in sequence and passing state through `PipelineContext`.
- **Pydantic Data Contracts:** Guarantees strict type-safety and non-negotiable invariant:
  $$\text{prediction is None} \iff \text{needs\_human\_review is True}$$

### Layer 3: Multi-Agent Specialist Layer
- **Decoupled Responsibilities:** Each agent performs exactly one mathematical or domain check.
- **No Shared State:** Agents operate on pure data passed via input arguments, preventing race conditions or cross-contamination.

### Layer 4: Storage & Reference Artifacts
- **`reference_stats.npz` (20 MB):** Precomputed mean vector $\mu \in \mathbb{R}^{1024}$ and regularized covariance matrix $\Sigma \in \mathbb{R}^{1024 \times 1024}$ computed across 78,299 patient-strict training chest X-rays.
- **`densenet121-res224-nih`:** Pretrained weights from the TorchXRayVision library.
- **`calibration_results.json`:** Platt scaling parameters trained on 17,097 validation cases.

### Layer 5: Clinical Routing Layer
- **Three Safe Exits:**
  1. **RELEASED:** Safe for automated review (Acc = 92.31%, Specificity = 93.15%, NPV = 99.02%).
  2. **ESCALATE:** Prediction withheld; routed to human radiologist workstation with full audit trace.
  3. **REJECT:** Prediction withheld; notification sent to technician that image is invalid or severely out-of-distribution.

---

## 4. Software Stack & Dependencies

| Component | Library / Framework | Version / Role |
|---|---|---|
| Deep Learning Core | `torch`, `torchvision` | Model loading, tensor operations |
| Medical Imaging | `torchxrayvision` | Pretrained DenseNet-121 weights |
| Image Processing | `opencv-python` (`cv2`) | Laplacian, CLAHE, NL-Means, Unsharp Mask |
| Scientific Computing | `numpy`, `scipy` | Cholesky factorization, MAD estimator |
| Calibration | `scikit-learn` | Platt Logistic Regression calibrator |
| Data Validation | `pydantic` | Pipeline state models & invariant enforcement |
| Presentation UI | `streamlit` | Interactive multi-agent dashboard |
| Configuration | `pyyaml` | Centralized pipeline & threshold parameters |
| Testing | `pytest` | 448 automated unit and integration tests |

---

## 5. Viva Talking Points on Architecture
- **Why Multi-Agent over Monolithic Model?**  
  A monolithic end-to-end model is a black box. If it gives a wrong prediction, you cannot pinpoint why. In our multi-agent architecture, failures are transparent: we know if it was rejected due to blur (Quality Agent), alien anatomy (OOD Agent), or model hesitation (Uncertainty Agent).
- **Extensibility:**  
  New agents (e.g., Patient Demographics Agent, DICOM Metadata Agent) can be plugged in without refactoring existing classifier code.
- **Safety by Construction:**  
  The verification loop creates an anti-hallucination boundary: even if an image is repaired, the prediction is never released unless verified against all 5 safety gates.
