"""Common agent base class.

Responsibility:
    Give every agent a stable name and version and a place to attach thresholds. Each
    concrete agent defines its own typed run() because inputs differ per agent.

Input:
    n/a

Output:
    AgentBase abstract class.

Dependencies:
    contracts.common

Implementation phase: P0
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from cxr_reliability.contracts.common import AgentName


class AgentBase(ABC):
    name: ClassVar[AgentName]
    version: ClassVar[str]
