# Reliability-Aware Multi-Agent System for Chest X-Ray Pneumonia Classification

> **Research prototype. Not a clinical diagnostic system. Not validated for clinical use.**
> Use public research datasets only. Never process real patient data with this project.

This project wraps a pretrained pneumonia classifier with reliability checks (image quality,
out-of-distribution detection, uncertainty) and chooses an action for each input:
**Accept / Repair / Escalate / Reject**. The object of study is *whether the system should
trust a prediction*, not raw classification accuracy.

The source of truth is `docs/PRD_Reliability-Aware_CXR_System.pdf`.

## Status

**Scaffold only.** Configuration and contract schemas are real and tested. Every agent,
data utility, calibration routine, evaluation function, API route and dashboard page is a
stub that raises `NotImplementedError` and names its implementation phase. Nothing has been
calibrated: `configs/thresholds/v0_prd_defaults.yaml` contains the PRD starting points only,
and thresholds the PRD does not quantify are `null`.

## Architecture

Seven agents (PRD FR-1 to FR-7): Quality, OOD, Base Model, Uncertainty, Decision, Repair,
Verification. Only the Base Model is a neural network. See `docs/architecture.md`.

## Layout

```
configs/        pipeline wiring, versioned thresholds, experiment configs
data/           raw / interim / processed / manifests (only manifests tracked)
models/         pretrained-weight cache and fitted OOD statistics (gitignored)
outputs/        calibration and evaluation artifacts (gitignored)
logs/           per-inference audit records (gitignored)
notebooks/      exploration only, no pipeline logic
scripts/        download, manifest building, OOD fitting, calibration, evaluation, launchers
src/cxr_reliability/
  config/       settings, thresholds schema, pipeline config
  contracts/    pydantic schemas for every agent input/output, pipeline output, audit record
  data/         NIH loader, CheXpert/PadChest OOD sets, splits, manifests, transforms,
                corruption library, rare-but-valid selection
  models/       TorchXRayVision loader, feature hook, Base Model, Escalation model
  agents/       quality, ood, uncertainty, repair, verification, decision/ (v1 rules, v2 learned)
  pipeline/     orchestrator, latency accounting
  audit/        per-inference JSONL logger, provenance hashing
  calibration/  quality ROC, OOD fit, confidence, repair bounds, verification margin, registry
  evaluation/   metrics, per-action breakdown, OOD eval, repair recovery, cost, baselines,
                ablations, tracking, report
  api/          FastAPI backend
  dashboard/    Streamlit app and pages
  reporting/    optional template-based report writer (no LLM)
tests/          unit/, integration/, regression/, fixtures/
docs/           PRD, architecture, module contracts, decisions, calibration log, limitations, model card
```

Every module's docstring states its **responsibility, input, output, dependencies** and
implementation phase; `docs/module_contracts.md` collects them in one place.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
cp .env.example .env
pytest                      # config and contract tests run; specification tests are skipped
```

Docker: `docker compose up --build` (API on :8000, dashboard on :8501). Both start but the
endpoints raise `NotImplementedError` until Phase 8.

## Dashboard (Phase 12 — Research Prototype Interface)

The project includes an interactive research-prototype dashboard built with **Streamlit**, serving as an auditable visualization interface over the Phase 11 `ReliabilityPipeline`.

### Research Prototype Disclaimer
> **CRITICAL**: This dashboard is a **research prototype only**. It is not clinically validated and is not intended for medical diagnosis. Outputs are presented as experimental model scores, distribution shift categories, and reliability assessments—never clinical diagnoses or confirmed pathology statements.

### Prerequisites & Dependencies
Streamlit is specified in `requirements.txt`:
```bash
pip install streamlit>=1.32
```

### Launching the Dashboard
From the project root:
```bash
streamlit run src/cxr_reliability/dashboard/app.py
```
Or use the launcher script:
```bash
bash scripts/run_dashboard.sh
```

### User Workflow
1. **Upload Radiograph**: Upload a chest X-ray image (`PNG`, `JPG`, or `JPEG`). The image is processed locally and never transmitted to external services.
2. **Review Preview**: Inspect the image dimensions and color mode before analysis.
3. **Execute Analysis**: Click the primary **Analyze X-Ray** button. This invokes `ReliabilityPipeline.predict(image)` as the single source of truth.
4. **Inspect Reliability Assessment**:
   - **System Assessment**: Final pipeline state (`ACCEPT`, `VERIFIED`, `ESCALATE`, `REJECT`, `ERROR`), final action, reliability label, and human review requirement.
   - **Human Review Banner**: Prominently displayed whenever `needs_human_review == True`, with the exact decision rationale.
   - **Before / After Comparison**: When non-destructive repair is applied, the dashboard renders the unaltered original image and the repaired image side-by-side.
   - **Detailed Agent Signals**: Detailed tabs for Base Model & Uncertainty, Quality Screening, OOD Shift Detection, Decision Routing, and Audit/Latency Trace.

### Current Limitations
- **Uncalibrated Model Scores**: The Base Model output is a raw sigmoid probability from DenseNet-121 and has not undergone isotonic or temperature calibration (deferred to Phase 13).
- **Provisional Thresholds**: OOD and Verification thresholds are starting defaults (`v0_prd_defaults.yaml`); empirical OOD fitting across external cohorts is scheduled for Phase 13.
- **Local Inference**: Processing runs on the local CPU/GPU device without distributed scaling.

## Principles

- Thresholds are calibrated on held-out data, versioned, and frozen before test evaluation.
- Every agent returns a score, a label and a human-readable reasoning string.
- Every inference writes an audit record.
- Ambiguous signals fail toward Escalate/Reject, never silent Accept.
- Repaired outputs are always tagged and never trusted like clean originals.
- No general-purpose LLM in the diagnostic path.

## Open questions

See `docs/decisions.md` for PRD ambiguities that need an owner decision before the relevant
phase begins.
