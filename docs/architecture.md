# Architecture

Source of truth: `docs/PRD_Reliability-Aware_CXR_System.pdf`. Items marked PROPOSED are design
choices made while scaffolding, not PRD requirements, and need owner sign-off.

## Data flow (PRD FR-8, adapted)

```
image -> preprocess
   |-> Quality Agent (CPU, parallel)
   |-> Base Model: densenet121-res224-nih (one forward pass)
          |- Pneumonia logit -> Uncertainty Agent
          |- mid-layer features + logits -> OOD Agent
                     |
         Quality + OOD + Uncertainty -> Decision Agent
   Accept | Repair -> Repair Agent | Escalate -> resnet50-res512-all | Reject
                     |
        re-score -> Verification Agent
        verified -> release (tagged)   not verified -> escalate once, then reject
                     |
   audit record (JSONL) -> prediction + reliability label OR "needs human review"
```

PRD FR-8 says Quality and OOD run in parallel before the base model, but the primary OOD
method uses the base model's own features. The scaffold therefore runs Quality in parallel
with the shared base-model forward pass, and OOD after it.

## Decision precedence (PROPOSED for Decision v1)

1. Severe OOD -> Reject.
2. Any signal within `borderline_margin` of its threshold, or OOD borderline -> Escalate
   (or Reject if `execution.borderline_action: reject`). PRD section 4 safety default.
3. Poor quality: repairable -> Repair; beyond bounds -> Reject. OOD is re-scored after repair.
4. Good quality, in-distribution, low confidence -> Escalate.
5. Good quality, in-distribution, high confidence -> Accept.
6. After Repair/Escalate: verified -> release with tag; not verified -> escalate once, then Reject.

Loop limits (PROPOSED): at most 1 repair and 1 escalation per input.

## Development phases

| Phase | Scope | Depends on |
|---|---|---|
| P0 | Repo, config, threshold registry, contracts (scaffolded here) | none |
| P1 | NIH manifests, patient-level splits, CheXpert/PadChest loaders, rare-valid set | P0 |
| P2 | Base model wrapper, feature hook, baseline metrics | P1 |
| P3 | Corruption library, Quality Agent, quality ROC calibration | P1 |
| P4 | Uncertainty and OOD Agents, calibration, OOD evaluation | P2 |
| P5 | Repair Agent, recovery measurement, repair bounds, verification margin | P3 |
| P6 | Decision v1, Verification, orchestrator, audit log | P3, P4, P5 |
| P7 | Full evaluation, baselines, ablations, cost | P6 |
| P7b | Decision v2 (stretch) and ablation vs v1 | P7 |
| P8 | FastAPI, Streamlit, Docker | P6 |
| P9 | Docs, model card, limitations | P7 |

## Layering

`contracts` and `config` depend on nothing internal. `data`, `models`, `agents` depend on
`contracts`/`config`. `pipeline` depends on agents and models. `audit`, `calibration`,
`evaluation` depend on the layers below. `api` and `dashboard` depend on `pipeline`/`audit`
and never contain decision logic.
