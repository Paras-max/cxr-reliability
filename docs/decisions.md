# Open decisions and PRD issues

Items found while reading the PRD. None are resolved by this scaffold.

| # | Issue | Scaffold treatment | Owner decision needed |
|---|---|---|---|
| 1 | FR-8 runs OOD before the base model, but OOD uses base-model features | One shared forward pass; OOD after it | Confirm |
| 2 | Models are multi-label; "softmax confidence" does not apply | Binary confidence and entropy on the Pneumonia output | Confirm |
| 3 | Decision table has undefined cells (e.g. poor quality + borderline OOD) | Precedence order in `architecture.md` | Approve or amend |
| 4 | Pneumonia is a rare, text-mined label; accuracy is misleading | Add AUROC, AUPRC, sensitivity at fixed specificity, risk-coverage | Confirm |
| 5 | PRD split sizes may not match the official NIH lists | Verify against official list files in Phase 1 | none |
| 6 | No natural labels for blur/noise/exposure | Synthetic corruption library provides ground truth | Confirm |
| 7 | Laplacian-variance recovery can be inflated by amplified noise | Also report SSIM/PSNR and prediction recovery | Confirm |
| 8 | `resnet50-res512-all` saw CheXpert/PadChest, so it cannot be the natural-OOD reference | OOD evaluated for the NIH-only fast model | Confirm |
| 9 | CheXpert/PadChest may need access approval | Check terms before download | Obtain access |
| 10 | OOD AUROC "target" and latency budget have no numbers | Set after baseline measurement, before test evaluation | Set numbers |
| 11 | No binary operating point for the positive label | `uncertainty.positive_class_threshold` stays null until calibrated | Choose criterion |
| 12 | Definition of "noise residual" for SNR is not given | Fixed and documented in Phase 3 | Confirm |
| 13 | Rare-but-valid definition is not given | Proposed in Phase 1 for approval | Approve |
