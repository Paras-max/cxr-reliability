"""FastAPI application factory.

Responsibility:
    Create the FastAPI app, register routes, load the pipeline at startup. Run with
    uvicorn cxr_reliability.api.main:create_app --factory.

Input:
    Environment settings.

Output:
    FastAPI application.

Dependencies:
    fastapi, api.routes, api.dependencies, config.settings

Implementation phase: P8
"""

from __future__ import annotations


def create_app():
    raise NotImplementedError("Phase 8")
