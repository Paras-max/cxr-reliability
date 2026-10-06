# PROJECT MASTER CHEATSHEET — Reliability-Aware CXR System
> **RESEARCH PROTOTYPE ONLY. Not a clinical diagnostic tool. Not FDA/CE approved.**

---

## ⚡ QUICK REVISION (READ THIS FIRST)

### What is this project?
> "Instead of blindly trusting an AI reading chest X-rays, our system checks if the image is clear, familiar, and certain — fixes bad images if possible — and calls a human doctor whenever in doubt."

### The 7 Agents (Simple Version)
| # | Agent | Plain English Job | Method |
|---|---|---|---|
| 1 | **Quality Agent** | Is the photo blurry/noisy/dark? | Laplacian variance, SNR, exposure mean |
| 2 | **Base Model** | Read the X-ray for Pneumonia | DenseNet-121, Index 8 output |
| 3 | **OOD Agent** | Is this a real chest X-ray? | Mahalanobis distance (1024-D features) |
| 4 | **Uncertainty Agent** | Is the AI guessing or confident? | Binary entropy + confidence = max(p, 1-p) |
| 5 | **Decision Agent** | Traffic Police — what to do next? | Precedence rule table (R0–R7) |
| 6 | **Repair Agent** | Fix the bad image | CLAHE → NL-Means → Unsharp Masking |
| 7 | **Verification Agent** | Did fixing it actually help safely? | 5 strict gates — quality, OOD, no label flip |

### The 4 Actions
- **ACCEPT** — Clean, familiar, confident → Release prediction
- **REPAIR** — Fixable defect → Clean & re-check
- **ESCALATE** — Uncertain/repair failed → Withhold, call doctor
- **REJECT** — Totally wrong image → Block immediately

### Top 5 Numbers to Remember
| Fact | Value |
|---|---|
| Total dataset images | **112,120** (30,805 patients, 0 overlap) |
| Automated vs Human Review | **44% released, 56% sent to doctors** |
| Negative Predictive Value | **99.02%** |
| Classification Accuracy (released) | **92.31%** |
| Average inference time (CPU) | **1.70 seconds/image** |

### How We Tested It Works (6 Points)
1. **Safety:** NPV = 99.02% — when AI says "No Pneumonia", it's right 99% of the time
2. **Defect Detection:** Synthetically blurred/noisy images — Quality Agent caught **100%**
3. **OOD:** On 17,097 real X-rays → only 1% false alarm; on fake/wrong images → **100% rejected**
4. **Calibration:** Platt scaling dropped ECE from **31.15% → 0.004%**
5. **Safe Repair:** 12,082 repairs — 78.3% improved, 0% worsened, **0% diagnosis flipped**
6. **Code:** **448 tests passed**, 16,724 images ran 8 hours — **0 crashes**

### 1-Minute Viva Speech
> *"Respected Sir/Madam, standard medical AI models fail silently — they make confident wrong guesses when given blurry or unfamiliar images. Our project wraps a DenseNet-121 pneumonia classifier with 7 reliability agents. We check image quality, detect out-of-distribution inputs using Mahalanobis distance, and measure predictive entropy. If an image is degraded we fix it, then strictly verify the fix didn't change the diagnosis. Anything uncertain is withheld for human review. On 16,724 test images, we safely automated 44% with 92.3% accuracy and 99.02% NPV, while escalating all 56% of doubtful cases to doctors."*

---

## 📌 CORE NUMBERS AT A GLANCE

| Parameter | Value |
|---|---|
| Dataset | NIH ChestX-ray14 — 112,120 frontal CXRs, 30,805 patients |
| Train / Val / Test | 78,299 / 17,097 / 16,724 images (patient-strict, seed 42) |
| Pneumonia prevalence | 1.28% (1,431 positive cases) |
| Base model | `densenet121-res224-nih` (TorchXRayVision) |
| Feature vector | 1024-D from Global Average Pooling layer |
| OOD thresholds | Borderline: **42.19** \| Severe: **45.31** (99th percentile on val set) |
| OOD regularizer λ | 10⁻⁵ (diagonal covariance) |
| Uncertainty: LOW | Confidence ≥ 0.85 AND Normalized Entropy ≤ 0.25 |
| Calibration ECE | Raw: 31.15% → Platt: 0.004% |
| Calibration Brier | Raw: 0.1494 → Platt: 0.0111 |
| Decision threshold | **0.522161** (F1-optimal on validation) |
| CLAHE settings | clip_limit=2.0, tile_grid=(8×8) |
| NL-Means settings | h=3.0, template=7, search=21 |
| Verification guard | Δconfidence ≥ −0.01, ΔSNR ≥ −5 dB, quality→GOOD, no label flip |
| Test coverage | 7,349 released (43.94%) \| 9,375 withheld (56.06%) |
| Accuracy / Spec / NPV | 92.31% / 93.15% / 99.02% (on released) |
| Latency | Mean 1.70s \| Median 1.32s \| P95 4.63s |
| Test suite | **448 passed, 0 failed** |

---

## 🔬 TECHNICAL DETAILS

### Quality Agent — 3 Checks
| Check | Formula | Threshold | Defect |
|---|---|---|---|
| **Blur** | Var(Laplacian(I)) | < 100 → POOR | Low edge variance = blurry |
| **Noise** | SNR = 20·log₁₀(μ/σ_noise) | < 15 dB → POOR | σ_noise via MAD estimator |
| **Exposure** | Mean pixel intensity | < 20 or > 235 → POOR | Dark or washed out |

> **GOOD** = all pass \| **DEGRADED** = partial issues \| **POOR** = any threshold breached

### OOD Agent — Mahalanobis Distance
```
D_M(z) = sqrt( (z - μ)ᵀ · Σ_reg⁻¹ · (z - μ) )

where:
  z     = 1024-D feature vector of test image
  μ     = mean of 78,299 training images
  Σ_reg = covariance + 10⁻⁵ · I  (regularized)
  
Solved efficiently with Cholesky:  L·y = (z - μ),  D_M = ||y||₂
```
| D_M Value | OOD Level | Action |
|---|---|---|
| < 42.19 | IN_DISTRIBUTION | Allowed to ACCEPT/REPAIR |
| 42.19–45.31 | BORDERLINE | → ESCALATE |
| ≥ 45.31 | SEVERE | → REJECT |

### Uncertainty Agent — Binary Entropy
```
confidence(p) = max(p, 1-p)          ∈ [0.5, 1.0]
H(p)          = -p·ln(p) - (1-p)·ln(1-p)   [nats]
H_norm(p)     = H(p) / ln(2)         ∈ [0.0, 1.0]

LOW if:  confidence ≥ 0.85  AND  H_norm ≤ 0.25
HIGH if: everything else
```

> Why binary entropy (not softmax)? TorchXRayVision uses **independent sigmoid heads** for 18 diseases — multi-label, not multi-class.

### Decision Agent — Rule Table (Precedence Order)
| Rule | Condition | Action |
|---|---|---|
| R0 | Invalid/corrupt input | ESCALATE |
| R1 | OOD = SEVERE | **REJECT** |
| R2 | OOD = BORDERLINE | ESCALATE |
| R3 | Quality = POOR or DEGRADED + In-Distribution | **REPAIR** |
| R4 | Quality = GOOD + In-Distribution + Uncertainty = HIGH | ESCALATE |
| R6 | Quality = GOOD + In-Distribution + Uncertainty = LOW | **ACCEPT** |
| R7 | Anything else (safe default) | ESCALATE |

### Repair Agent — Sequential Pipeline
```
Defective Image → CLAHE (exposure) → NL-Means (noise) → Unsharp Mask (blur)
```
- Non-destructive: original image on disk is never modified
- Execution order is fixed regardless of which defects are present

### Verification Agent — 5 Gates (ALL must pass)
1. **Repair was actually applied** (not skipped)
2. **No label flip** — prediction must not cross 0.5
3. **OOD stayed IN_DISTRIBUTION**
4. **Quality resolved to GOOD** (POOR→DEGRADED still fails)
5. **Confidence non-degradation** Δconf ≥ −0.01 AND ΔSNR ≥ −5 dB

> **VERIFIED** → `accepted_after_repair` \| **FAILED** → `needs_human_review`

### Probability Calibration
| Method | ECE | Brier |
|---|---|---|
| Raw DenseNet output | 31.15% | 0.1494 |
| **Platt Scaling (selected)** | **0.004%** | **0.0111** |
| Isotonic Regression | 0.000% | 0.0110 |

> Platt was chosen over Isotonic to avoid overfitting on rare positives (only 192 positive cases in val set).

---

## 📊 TEST RESULTS SUMMARY

### Full Test Set — N = 16,724 images
| Action | Count | % | Released? |
|---|---|---|---|
| ACCEPT (direct) | 501 | 3.0% | ✅ Yes |
| ACCEPT (after repair) | 6,848 | 41.0% | ✅ Yes |
| **TOTAL RELEASED** | **7,349** | **43.94%** | ✅ |
| ESCALATE | 9,274 | 55.5% | ❌ No |
| REJECT | 101 | 0.6% | ❌ No |
| **TOTAL WITHHELD** | **9,375** | **56.06%** | ❌ |

### Classification on Released Cases
| Metric | Value |
|---|---|
| True Positives | 17 |
| False Positives | 498 |
| True Negatives | 6,767 |
| False Negatives | 67 |
| **Accuracy** | **92.31%** |
| **Specificity** | **93.15%** |
| **NPV** | **99.02%** |
| Sensitivity | 20.24% |
| F1 | 0.057 |

> Low F1/sensitivity is expected at 1.32% prevalence. The system's value is as a **safety filter** — 99% NPV means safe to trust negatives.

### Repair Performance — N = 12,082 applied repairs
| Metric | Result |
|---|---|
| Quality improved | 78.28% (9,458 images) |
| Quality unchanged | 21.72% (2,624 images) |
| Quality worsened | **0.00%** |
| Verified & released | 6,848 (56.7%) |
| Escalated (failed gates) | 5,234 (43.3%) |
| Mean Δconfidence | −0.0024 (range: −0.067 to +0.036) |

---

## 🗂️ FILE STRUCTURE (Key Files Only)

```
cxr-reliability/
├── configs/
│   ├── pipeline.yaml                  # max_repair_attempts=1, borderline_action=escalate
│   └── thresholds/v0_prd_defaults.yaml # PRD baseline thresholds
├── artifacts/ood/
│   ├── metadata.json                  # borderline=42.19, severe=45.31
│   └── reference_stats.npz            # 1024-D mean + covariance (20 MB, 78k images)
├── outputs/
│   ├── calibration/calibration_results.json   # ECE, Brier, Platt/Isotonic
│   ├── evaluation_full_test_summary.json      # 16,724 test results
│   └── dataset_split_report.json             # Split counts, overlap check
├── src/cxr_reliability/
│   ├── agents/decision/rules.py        # R0-R7 rule table
│   ├── agents/verification.py          # 5-gate verification logic
│   ├── pipeline/orchestrator.py        # Main pipeline coordinator
│   ├── ood/mahalanobis.py              # Cholesky distance calculation
│   ├── uncertainty/estimator.py        # Entropy + confidence
│   ├── quality/evaluator.py            # Blur, noise, exposure
│   ├── repair/pipeline.py              # CLAHE → NL-Means → Unsharp
│   └── dashboard/app.py                # Streamlit UI entry point
└── tests/                              # 448 tests — 0 failures
```

---

## ❓ VIVA Q&A (Quick Answers)

**Q: What is the main goal?**  
Determine WHEN to trust an AI prediction — not just what the prediction is.

**Q: Why multi-agent instead of one model?**  
Each agent has one job, is independently testable, and failures are transparent (not a black box).

**Q: Why patient-level splitting?**  
Same patient's multiple X-rays in both train and test = model memorizes the patient, not the disease (data leakage).

**Q: Why Mahalanobis distance for OOD?**  
Unlike Euclidean distance, it accounts for feature correlations and different scales across 1024 dimensions.

**Q: Why binary entropy instead of softmax entropy?**  
TorchXRayVision uses independent sigmoid heads (multi-label) — not a single softmax (multi-class).

**Q: Why was the PRD verification rule (+0.15 confidence gain) changed?**  
Empirically, the max observed confidence gain from image repair was only +0.036. Requiring +0.15 would reject 100% of valid repairs.

**Q: Why Platt scaling over Isotonic regression?**  
Platt is smooth and parametric — prevents overfitting on very few positive cases (only 192 in val set).

**Q: Why is specificity (93%) better than sensitivity (20%)?**  
Because pneumonia is only 1.3% prevalent — false positives swamp rare true positives. The system is an exclusionary safety filter: trust its negatives, not its positives.

**Q: What happens if repair makes the image worse?**  
Verification Gate 4 fails → ESCALATE → prediction withheld → human review. Quality can never worsen on a released prediction (0.00% worsening rate).

**Q: What guarantees the system never silently accepts a bad image?**  
Three architectural guarantees: (1) Rule R7 safe-default escalates anything unhandled; (2) `needs_human_review` must equal `True` whenever `prediction` is `None` (Pydantic invariant); (3) Any exception in any agent also forces `NEEDS_HUMAN_REVIEW`.

---

## ⚠️ LIMITATIONS (Tell This Honestly)
1. **Labels from NLP mining** — not direct radiologist annotation (~10% error rate)
2. **1.28% prevalence** — low precision/sensitivity; high NPV
3. **Only NIH X-rays in OOD reference** — may over-reject images from different hospitals/scanners
4. **Repair can't recover lost information** — severely blurred/dark images are irreparable
5. **CPU only** — 1.7s average; repair cases can take up to 4.6s per image

---

## 🚀 RUN COMMANDS

```powershell
# Start Streamlit dashboard
cd "C:\Users\PARAS\Desktop\AI SEM 5 B1 G5\PROJECT\cxr-reliability\cxr-reliability"
streamlit run src\cxr_reliability\dashboard\app.py

# Run tests
pytest tests/ -v

# Run on a single image (Python)
from cxr_reliability.pipeline.orchestrator import ReliabilityPipeline
result = pipeline.run("path/to/image.png")
print(result.output.reliability_label)
```
