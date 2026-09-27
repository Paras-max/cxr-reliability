# Quality Agent — Phase 7 Documentation

## 1. Purpose of the Quality Agent
The Quality Agent (PRD FR-1) assesses chest radiograph quality prior to downstream multi-agent decision routing. It examines three primary physical dimensions of image fidelity:
1. **Blur**: Loss of spatial detail and high-frequency anatomical edges.
2. **Noise**: Elevated high-frequency grain, detector quantum mottle, or electronic noise.
3. **Exposure**: Photometric underexposure (excessive darkness) or overexposure (bright saturation/clipping).

The Quality Agent is a reliability filter:
* It does **not** diagnose pathology or predict pneumonia.
* It does **not** alter the original image data.
* It flags technical defects that degrade neural network reliability.

---

## 2. Blur Metric (Laplacian Variance)
Image sharpness is quantified using the variance of the 2D discrete Laplacian operator:
$$\text{laplacian\_variance} = \text{Var}(\nabla^2 I)$$

* **Discrete Kernel**: Standard $3 \times 3$ Laplacian kernel with spatial reflection padding.
* **Blur Percentage** (PRD FR-1 formula):
  $$\text{blur\_pct} = 100 \times \left(1 - \min\left(\frac{\text{laplacian\_variance}}{\text{reference\_max\_variance}}, 1.0\right)\right)$$
  Where `reference_max_variance` (default: 500.0) represents a reference sharp radiograph.
* **Threshold**: Minimum acceptable variance `blur_laplacian_var_min = 100.0`. Inputs below 100.0 are flagged as blurred.

---

## 3. Noise Metric (High-Frequency Residual SNR)
Image noise is evaluated using an edge-preserving spatial residual and Median Absolute Deviation (MAD):
1. **Edge-Preserving Residual**:
   $$R = I - \text{median\_filter}(I, 3 \times 3)$$
2. **Robust Noise Dispersion**:
   $$\sigma_{\text{noise}} = \frac{\text{median}(|R - \text{median}(R)|)}{0.6745}$$
   The robust MAD isolates true noise without penalizing sharp structural anatomical boundaries.
3. **Signal-to-Noise Ratio (SNR in dB)**:
   $$\text{SNR}_{\text{dB}} = 20 \log_{10}\left(\frac{\max(\mu_{\text{signal}}, \epsilon)}{\sigma_{\text{noise}} + \epsilon}\right)$$
* **Numerical Safeguard**: When $\sigma_{\text{noise}} \to 0$, $\text{SNR}_{\text{dB}}$ is cleanly capped at $+100.0\text{ dB}$ to avoid infinite or NaN values.
* **Threshold**: Minimum acceptable SNR `snr_db_min = 15.0 dB`. Inputs below 15.0 dB are flagged as noisy.

---

## 4. Exposure Metrics
Photometric distribution statistics are evaluated on standardized $[0, 255]$ pixel intensities:
* **Mean Intensity**: $\mu_I = \frac{1}{N} \sum I_i$
* **Histogram Standard Deviation**: $\sigma_I = \text{std}(I)$ (measuring dynamic range spread)
* **Saturation Fractions**: Dark saturation ($\text{fraction} < 10.0$) and bright saturation ($\text{fraction} > 245.0$).
* **Thresholds** (PRD v0 defaults):
  * Underexposure: $\mu_I < 20.0$ (too dark)
  * Overexposure: $\mu_I > 235.0$ (washed out)

---

## 5. Overall Quality Decision Rule
The Quality Agent combines individual metrics into an overall status (`GOOD`, `DEGRADED`, `POOR`):
1. **POOR Quality**:
   * Flagged if **any** critical dimension fails its threshold:
     * $\text{laplacian\_variance} < 100.0$ (blur), OR
     * $\text{SNR}_{\text{dB}} < 15.0\text{ dB}$ (noise), OR
     * $\mu_I < 20.0$ or $\mu_I > 235.0$ (exposure failure).
2. **DEGRADED Quality**:
   * Flagged if all dimensions pass, but one or more metrics lie within the near-threshold borderline margin (e.g. $\pm 15\%$ of threshold).
3. **GOOD Quality**:
   * All metrics satisfy configured thresholds with acceptable margins.

---

## 6. Development Thresholds
Initial development defaults from PRD v0:
* `blur_laplacian_var_min`: `100.0`
* `reference_max_variance`: `500.0`
* `snr_db_min`: `15.0 dB`
* `exposure_mean_min`: `20.0`
* `exposure_mean_max`: `235.0`
* `borderline_margin_pct`: `0.15` (15%)

> [!IMPORTANT]
> These thresholds are **provisional starting points**. They have not been clinically or dataset calibrated.

---

## 7. Limitations & Why Thresholds Require Future Calibration
* **Sensor and Hardware Variations**: Different X-ray machines, digital detectors, and beam energies produce differing natural noise and contrast profiles. Universal uncalibrated cutoffs cannot capture these variations without ROC analysis.
* **Deferred Calibration (Phase 3/4.5)**: Per-dataset ROC calibration on in-distribution validation data is required before production deployment.
* **Resolution Independence**: Laplacian variance scales with image resolution; inputs should be evaluated at consistent radiograph resolutions.

---

## 8. Difference Between Image Quality and Clinical Diagnosis
* **Image Quality**: Evaluates whether the digital artifact has sufficient physical fidelity (sharpness, noise margin, dynamic range) to be interpreted by a model.
* **Clinical Diagnosis**: Evaluates whether pathology (e.g. pneumonia) is present in the patient.
* **Non-Equivalence**:
  * Good technical quality does **not** imply that pneumonia is absent or present.
  * Poor technical quality does **not** mean the patient does not have pneumonia; it means the automated system cannot reliably confirm the diagnosis without review or repair.
