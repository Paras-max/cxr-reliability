"""API dependency wiring.

Responsibility:
    Build the pipeline once at startup (models loaded once, not per request) and expose it
    to routes. Central place for upload size and content-type validation.

Input:
    Settings, thresholds and pipeline configs.

Output:
    A ready ReliabilityPipeline; validated image arrays.

Dependencies:
    config, pipeline.orchestrator, fastapi

Implementation phase: P8
"""

from __future__ import annotations


def build_pipeline():
    raise NotImplementedError("Phase 8")


def decode_and_validate_upload(data: bytes, content_type: str, max_bytes: int):
    raise NotImplementedError("Phase 8")
