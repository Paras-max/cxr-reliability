"""Configuration package.

Responsibility:
    Typed access to environment settings, pipeline wiring and versioned thresholds.

Input:
    YAML files under configs/ and CXR_* environment variables.

Output:
    Validated pydantic models (Settings, PipelineConfig, Thresholds).

Dependencies:
    pydantic, pydantic-settings, pyyaml

Implementation phase: P0
"""
