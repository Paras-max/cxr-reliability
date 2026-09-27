# Model card (skeleton)

Research prototype. Not a clinical diagnostic system.

| Field | Value |
|---|---|
| Fast model | `densenet121-res224-nih` (TorchXRayVision), used as-is |
| Escalation model | `resnet50-res512-all` (TorchXRayVision), used as-is |
| Weight SHA-256 | pending (Phase 2) |
| TorchXRayVision version | pending (lockfile) |
| Training data of pretrained models | See TorchXRayVision documentation: NIH only (fast); NIH + CheXpert + PadChest + MIMIC (escalation) |
| Intended use | Studying reliability signals around a pneumonia classifier |
| Out of scope | Clinical use, patient triage, any diagnostic claim |
| Evaluation data | pending |
| Known limitations | See `limitations.md` |
