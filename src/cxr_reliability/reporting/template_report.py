"""Template-based report writer.

Responsibility:
    Render PipelineOutput into a readable summary from fixed templates. An LLM adapter is
    intentionally not scaffolded: the PRD allows one only as an optional report writer,
    and it needs separate owner approval.

Input:
    PipelineOutput.

Output:
    Plain-text or markdown summary containing the disclaimer.

Dependencies:
    contracts.pipeline

Implementation phase: Optional
"""

from __future__ import annotations

from cxr_reliability.contracts.pipeline import PipelineOutput


def render_report(output: PipelineOutput) -> str:
    raise NotImplementedError("Optional phase")
