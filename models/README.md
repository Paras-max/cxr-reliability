# models/

Local cache for pretrained weights (TorchXRayVision downloads) and fitted statistics.
Weights are gitignored. Record each weight file's SHA-256 in `docs/model_card.md` and in
the threshold/calibration artifacts so results can be tied to exact checkpoints.

Expected contents once implemented
- TorchXRayVision cache for `densenet121-res224-nih` and `resnet50-res512-all`
- OOD statistics (feature mean, covariance) fitted on the in-distribution fit split
