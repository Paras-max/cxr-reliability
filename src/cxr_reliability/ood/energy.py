"""Energy score — optional secondary OOD signal — Phase 5.

IMPORTANT — Logit Availability Notice
--------------------------------------
TorchXRayVision's densenet121-res224-nih model applies sigmoid activation
internally before returning outputs. The model does NOT expose raw logits
through its standard forward() interface.

Consequence for Energy Score:
    The classic energy score (Liu et al., 2020, "Energy-based OOD Detection"):
        E(x) = -T · log Σᵢ exp(logitᵢ / T)
    requires raw logits. Because TXV provides sigmoid probabilities, NOT
    logits, the canonical energy formula cannot be correctly applied here.

What we implement instead:
    1. If genuine logits are provided (e.g., from a future model that exposes
       pre-sigmoid outputs), use the true energy formula.
    2. If only probabilities are available, compute a pseudo-energy via
       inverse sigmoid ("logit" transformation) as an approximation:
           logit(p) = log(p / (1 - p))
       This is clearly labeled as an approximation and NOT equivalent to
       the true energy score from raw logits.
    3. Energy score is DISABLED by default (energy_enabled: false in config).
       It must be explicitly enabled.

The energy score is a SECONDARY cross-check signal only. Mahalanobis
distance is the PRIMARY OOD detector in this system.

Scientific Constraints:
    - We do NOT fabricate logits.
    - We clearly document the approximation.
    - We do NOT claim pseudo-energy = true energy score.
    - We do NOT claim energy score improves over Mahalanobis for this model.

Reference:
    Liu, W. et al. (2020). "Energy-based Out-of-distribution Detection."
    NeurIPS 2020. https://arxiv.org/abs/2010.03759

Implementation phase: P5
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Small constant to avoid log(0) in pseudo-logit computation
_EPS = 1e-6

# Whether to log a warning about the pseudo-logit approximation once only
_pseudo_logit_warned = False


def energy_score_from_logits(
    logits: np.ndarray | torch.Tensor,
    temperature: float = 1.0,
) -> float:
    """
    Compute the true energy score from raw (pre-sigmoid) logits.

    Formula:  E(x) = -T · log Σᵢ exp(logitᵢ / T)

    A lower energy score indicates in-distribution;
    a higher (less negative) score suggests out-of-distribution.

    Parameters
    ----------
    logits      : 1D array/tensor of raw logits (pre-activation, NOT probabilities)
    temperature : T > 0, scales the energy. Default: 1.0.

    Returns
    -------
    float — energy score (negative value; closer to 0 = more OOD)
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    if isinstance(logits, torch.Tensor):
        arr = logits.detach().cpu().numpy()
    else:
        arr = np.asarray(logits)

    arr = arr.flatten().astype(np.float64)

    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "logits contain NaN/Inf. Cannot compute energy score."
        )

    # Numerically stable via log-sum-exp:  log Σᵢ exp(aᵢ) = max(a) + log Σᵢ exp(aᵢ - max(a))
    scaled = arr / temperature
    max_val = np.max(scaled)
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
    energy = -temperature * log_sum_exp

    return float(energy)


def energy_score_from_probs(
    probs: np.ndarray | torch.Tensor,
    temperature: float = 1.0,
) -> float:
    """
    Compute an APPROXIMATE energy score from sigmoid probabilities.

    IMPORTANT: This is an approximation, not the canonical energy score.
    The TorchXRayVision model applies sigmoid internally and does NOT expose
    raw logits. To approximate, we apply the inverse sigmoid (logit function)
    to recover pseudo-logits:

        pseudo_logit(p) = log(p / (1 - p))    for p ∈ (0, 1)

    The result is then passed to the standard energy formula. This is NOT
    equivalent to computing energy from true pre-sigmoid logits and should
    NOT be interpreted as the canonical energy-based OOD score.

    Parameters
    ----------
    probs       : 1D array/tensor of sigmoid probabilities in (0, 1)
    temperature : T > 0

    Returns
    -------
    float — approximate pseudo-energy score
    """
    global _pseudo_logit_warned
    if not _pseudo_logit_warned:
        logger.warning(
            "Energy score is being computed from probabilities via inverse-sigmoid "
            "(pseudo-logit). This is an approximation because TorchXRayVision "
            "applies sigmoid internally and does not expose raw logits. "
            "The resulting value is NOT the canonical energy score. "
            "Use energy_score_from_logits() when genuine logits are available. "
            "(This warning is shown once per process.)"
        )
        _pseudo_logit_warned = True

    if isinstance(probs, torch.Tensor):
        arr = probs.detach().cpu().numpy()
    else:
        arr = np.asarray(probs)

    arr = arr.flatten().astype(np.float64)

    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "probs contain NaN/Inf. Cannot compute approximate energy score."
        )

    # Clip to (eps, 1-eps) to avoid log(0) or log(inf)
    clipped = np.clip(arr, _EPS, 1.0 - _EPS)

    # Inverse sigmoid: logit(p) = log(p / (1-p))
    pseudo_logits = np.log(clipped / (1.0 - clipped))

    return energy_score_from_logits(pseudo_logits, temperature=temperature)


def energy_score_safe(
    probs_or_logits: np.ndarray | torch.Tensor | None,
    temperature: float = 1.0,
    is_logits: bool = False,
) -> float | None:
    """
    Safely compute energy score, returning None on failure.

    Parameters
    ----------
    probs_or_logits : model outputs (probabilities or logits); None → returns None
    temperature     : energy temperature T
    is_logits       : if True, treat input as raw logits (use canonical formula);
                      if False, treat as probabilities (use pseudo-logit approximation)

    Returns
    -------
    float or None — energy score (None if computation fails or input is None)
    """
    if probs_or_logits is None:
        return None

    try:
        if is_logits:
            return energy_score_from_logits(probs_or_logits, temperature=temperature)
        else:
            return energy_score_from_probs(probs_or_logits, temperature=temperature)
    except Exception as exc:
        logger.warning(
            "Energy score computation failed (returning None): %s", exc
        )
        return None
