#!/usr/bin/env bash
set -euo pipefail
exec uvicorn cxr_reliability.api.main:create_app --factory \
  --host "${CXR_API_HOST:-0.0.0.0}" --port "${CXR_API_PORT:-8000}"
