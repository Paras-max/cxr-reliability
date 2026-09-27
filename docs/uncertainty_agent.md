# Uncertainty Agent — Phase 6 Documentation

## 1. Purpose of the Uncertainty Agent
The Uncertainty Agent (PRD FR-4) analyzes prediction ambiguity for the target pathology (Pneumonia) directly from the Base Model's output. It computes continuous measures of confidence and predictive entropy to classify the model's prediction into deterministic uncertainty levels (`LOW`, `MEDIUM`, `HIGH`). This signal feeds the future Decision Agent to help route cases (e.g., to Accept, Escalate, or Reject).

The Uncertainty Agent is purely analytical:
- It does **not** run neural network inference or modify inputs.
- It does **not** access the patient dataset or images.
- It does **not** perform out-of-distribution (OOD) detection.
- It does **not** make clinical diagnoses or final pipeline decisions.

---

## 2. Confidence Definition
For a continuous probability score $p \in [0.0, 1.0]$ representing the Pneumonia output:
$$\text{confidence}(p) = \max(p, 1 - p)$$

* Range: $[0.5, 1.0]$.
* At $p = 0.5$, $\text{confidence} = 0.5$ (maximal ambiguity between Pneumonia and Non-Pneumonia).
* At $p = 0.0$ or $p = 1.0$, $\text{confidence} = 1.0$ (strong model conviction in non-pneumonia or pneumonia respectively).

---

## 3. Predictive Entropy Definition
Binary predictive entropy measures the expected information or unpredictability of the binary distribution:
$$H(p) = -p \ln(p) - (1 - p) \ln(1 - p) \quad [\text{in nats}]$$

* **Numerical safeguard**: $p$ is clamped to $[\epsilon, 1 - \epsilon]$ with default $\epsilon = 10^{-8}$ to prevent $\ln(0)$ and $-\infty/\text{NaN}$.
* **Boundary values**: At $p = 0.0$ and $p = 1.0$, $H(p) = 0.0$.
* **Peak**: At $p = 0.5$, $H(0.5) = \ln(2) \approx 0.693147\text{ nats}$.
* **Symmetry**: $H(p) = H(1 - p)$.

---

## 4. Normalized Entropy
To provide an interpretable metric invariant to logarithm base, normalized entropy is computed as:
$$H_{\text{norm}}(p) = \frac{H(p)}{\ln(2)} \in [0.0, 1.0]$$

* $H_{\text{norm}}(0.0) = H_{\text{norm}}(1.0) = 0.0$
* $H_{\text{norm}}(0.5) = 1.0$

---

## 5. Uncertainty-Level Decision Rule
The agent categorizes uncertainty deterministically using a transparent rule combining confidence and normalized entropy:

1. **HIGH Uncertainty**:
   $$\text{confidence} \le \text{confidence\_low\_threshold} \quad \text{OR} \quad H_{\text{norm}} \ge \text{entropy\_high\_threshold}$$
2. **LOW Uncertainty**:
   $$\text{confidence} \ge \text{confidence\_high\_threshold} \quad \text{AND} \quad H_{\text{norm}} \le \text{entropy\_low\_threshold}$$
3. **MEDIUM Uncertainty**:
   All intermediate combinations (moderate confidence and intermediate predictive entropy).

---

## 6. Development Thresholds
The initial development thresholds are configured as follows:
* `confidence_high_threshold`: `0.85`
* `confidence_low_threshold`: `0.60`
* `entropy_low_threshold`: `0.25`
* `entropy_high_threshold`: `0.60`
* `epsilon`: `1e-8`

> [!IMPORTANT]
> These thresholds are **provisional development defaults**. They are not empirically or clinically calibrated.

---

## 7. Limitations Caused by Deferred Calibration
TorchXRayVision's DenseNet-121 model applies sigmoid activations internally across 18 multi-label outputs and does not expose raw un-activated logits. Furthermore, post-hoc probability calibration (Phase 4.5 via Platt scaling or Isotonic regression) is deferred.

Therefore:
* The raw model score $p$ is an **uncalibrated sigmoid probability**.
* The derived confidence $\max(p, 1 - p)$ must **not** be interpreted as a calibrated probability of diagnostic correctness.
* Post-hoc calibration (evaluating Expected Calibration Error / Brier score) will be used in downstream phases to refine these thresholds.

---

## 8. Difference Between Uncertainty and OOD
* **Uncertainty Agent (Phase 6)**: Measures **prediction ambiguity** (in-model epistemic/aleatoric uncertainty) given the model's classification score. An image may be completely in-distribution (e.g., standard frontal chest X-ray), but have visual features that make pneumonia versus normal borderline ($p \approx 0.5$), resulting in high uncertainty.
* **OOD Agent (Phase 5)**: Measures **feature representation distance** ($\chi^2$ / Mahalanobis distance) from the in-distribution training manifold. An input can have an extreme model score ($p \approx 0.99$ or $0.01$) yet be completely out-of-distribution (e.g., lateral projection, pediatric scan, non-chest radiograph), which the OOD Agent detects independently.

---

## 9. Difference Between Model Uncertainty and Clinical Uncertainty
* **Model Uncertainty**: Reflects the internal ambiguity of the neural network's mathematical output based on its learned parameters and training data representation.
* **Clinical Uncertainty**: Reflects diagnostic ambiguity in patient health status, clinical history, presentation, and differential diagnoses.
* **Safety Mandate**: The Uncertainty Agent **does not make clinical diagnoses**, does not predict patient outcomes, and does not claim that low uncertainty implies clinical correctness or high uncertainty implies clinical error.
