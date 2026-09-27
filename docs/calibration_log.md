# Calibration log

One entry per calibration run. PRD section 9: all thresholds are starting points requiring
per-dataset calibration. Nothing below has been calibrated yet.

| Threshold | PRD starting value | Calibrated value | Method | Data split (manifest hash) | Thresholds file | Date |
|---|---|---|---|---|---|---|
| quality.blur_laplacian_var_min | 100 | pending | ROC on synthetic blur vs clean | pending | pending | |
| quality.snr_db_min | 15 | pending | ROC on synthetic noise | pending | pending | |
| quality.exposure_mean_min / max | 20 / 235 | pending | ROC on synthetic exposure shifts | pending | pending | |
| quality.reference_max_laplacian_var | not given | pending | Percentile on clean images | pending | pending | |
| ood.mahalanobis_borderline | 95th percentile | pending | Percentile on ID-val | pending | pending | |
| ood.mahalanobis_severe | not given | pending | Higher percentile, checked on natural OOD | pending | pending | |
| ood.energy_* | not given | pending | Percentile on ID-val | pending | pending | |
| uncertainty.confidence_high_min / low_max | not given | pending | Risk-coverage on ID-val | pending | pending | |
| uncertainty.positive_class_threshold | not given | pending | Operating-point criterion (owner) | pending | pending | |
| decision.borderline_margin | not given | pending | Sensitivity analysis | pending | pending | |
| repair.* bounds | not given | pending | Recovery vs severity | pending | pending | |
| verification.min_confidence_gain | 0.15 | pending | Margin sweep on corrupted labeled images | pending | pending | |
