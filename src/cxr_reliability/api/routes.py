"""API routes.

Responsibility:
    POST /v1/predict, GET /v1/health, GET /v1/config, GET /v1/audit/{id}. Uploads are not
    persisted by default. Docs state the API is for public research datasets only.

Input:
    Multipart image upload; audit id.

Output:
    PredictResponse / HealthResponse / ConfigResponse / AuditRecord.

Dependencies:
    fastapi, api.schemas, api.dependencies, audit.audit_log

Implementation phase: P8
"""

from __future__ import annotations


def build_router():
    raise NotImplementedError("Phase 8")
