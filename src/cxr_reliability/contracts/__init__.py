"""Agent input/output contracts.

Responsibility:
    Single source of truth for every schema exchanged between agents, the pipeline, the
    audit log and the API. Contains data definitions only, no behavior.

Input:
    n/a

Output:
    pydantic models and enums.

Dependencies:
    pydantic

Implementation phase: P0
"""
