# Limitations (PRD section 9, plus scaffold notes)

1. Repair is a proxy for image-quality improvement, not proof of diagnostic accuracy.
   Mitigated by reversible-only fixes, tagging, and measured recovery rates.
2. All thresholds are starting points needing per-dataset calibration via ROC analysis.
3. OOD detectors can conflate rare-but-real presentations with true distribution shift.
   Both failure modes are reported separately.
4. No component is clinically validated or regulatory-approved. This is a research
   prototype and must say so wherever it is shown.
5. General-purpose LLMs are not in the diagnostic path.

Additional points to state in any write-up
- NIH labels are text-mined from reports, so ground truth is noisy.
- CheXpert/PadChest test site/scanner shift, not wrong-body-part shift.
- The fast model's pretrained weights may have seen images used here as held-out.
- Verified confidence gain is not the same as correctness; correctness is evaluated on
  corrupted images with known labels.
