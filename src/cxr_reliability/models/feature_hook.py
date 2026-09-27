"""Mid-layer feature extraction via forward hooks (Phase 4).

Responsibility:
    Attach a forward hook to a chosen DenseNet layer and return a pooled
    1024-dimensional feature vector in the SAME forward pass that produces
    the output probabilities (one pass feeds both the model output and the
    future OOD Agent — no extra inference overhead).

FEATURE LAYER CHOICE (documented):
    Layer: model.features.norm5
    Why:
        - This is the final batch-norm layer BEFORE the global average pool
          and the linear classifier in DenseNet-121.
        - Its output shape is (batch, 1024, H', W') where H' and W' depend
          on the input resolution (7×7 for 224-px input).
        - After adaptive average pooling → (batch, 1024): this is the
          standard "penultimate feature representation" used across the
          literature for OOD detection (Mahalanobis distance, energy score).
        - The 1024-dim vector captures the full learned representation of
          the DenseNet and is directly adjacent to the decision boundary,
          making it maximally informative for OOD scoring.
    Dimension: 1024
    Used by: Future OOD Agent (Phase 5) for Mahalanobis distance scoring.

Dependencies:
    torch

Implementation phase: P4
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# The layer name to hook — verified against the actual TXV DenseNet architecture
# (features children: conv0 norm0 relu0 pool0 denseblock1..4 transition1..3 norm5)
FEATURE_LAYER_NAME = "features.norm5"
FEATURE_DIM = 1024  # output dimension after global avg pool


class FeatureExtractor:
    """
    Context manager that attaches a forward hook to `features.norm5` and
    captures the globally-pooled 1024-dim feature vector during the forward pass.

    Usage
    -----
        extractor = FeatureExtractor(model)
        with extractor:
            output = model(image_tensor)
            features = extractor.pooled_features()  # shape: (batch, 1024)

    The hook is automatically removed on __exit__ (even if an exception occurs),
    so there is no risk of accumulating duplicate hooks across calls.
    """

    def __init__(self, model: nn.Module, layer_name: str = FEATURE_LAYER_NAME) -> None:
        self.model = model
        self.layer_name = layer_name
        self._hook_handle: Any = None
        self._captured: torch.Tensor | None = None

    def __enter__(self) -> "FeatureExtractor":
        target = self._get_layer(self.layer_name)
        self._hook_handle = target.register_forward_hook(self._hook_fn)
        self._captured = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _hook_fn(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        """
        Called automatically during the forward pass.
        Applies ReLU (norm5 output must go through ReLU before pooling,
        matching DenseNet's forward() logic) then global average pools
        to produce the 1024-dim feature vector.
        """
        activated = F.relu(output, inplace=False)
        # Adaptive average pool: (batch, 1024, H, W) → (batch, 1024, 1, 1)
        pooled = F.adaptive_avg_pool2d(activated, (1, 1))
        # Flatten to (batch, 1024)
        self._captured = pooled.view(pooled.size(0), -1).detach()

    def pooled_features(self) -> torch.Tensor:
        """
        Return the captured feature vector of shape (batch, 1024).

        Raises RuntimeError if called outside the context manager or before
        a forward pass has occurred.
        """
        if self._captured is None:
            raise RuntimeError(
                "No features captured. Ensure a forward pass has been run "
                "inside the FeatureExtractor context manager."
            )
        return self._captured

    def _get_layer(self, layer_name: str) -> nn.Module:
        """
        Navigate nested module by dot-separated layer name.
        e.g. 'features.norm5' → model.features.norm5
        """
        module = self.model
        for part in layer_name.split("."):
            try:
                module = getattr(module, part)
            except AttributeError:
                available = [n for n, _ in self.model.named_modules()]
                raise ValueError(
                    f"Layer '{layer_name}' not found in model. "
                    f"Available top-level modules: "
                    f"{[n for n, _ in self.model.named_children()]}"
                ) from None
        return module
