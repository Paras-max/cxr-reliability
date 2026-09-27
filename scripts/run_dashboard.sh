#!/usr/bin/env bash
set -euo pipefail
exec streamlit run src/cxr_reliability/dashboard/app.py
