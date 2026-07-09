"""Specialist subagents — each with its own system prompt and its own context."""

from __future__ import annotations

from .analyst import Analyst, AnalysisModel
from .base import Subagent
from .critic import Critic, CriticLoopResult, CriticOutput, run_critic_loop
from .researcher import ResearchFinding, ResearchFindings, Researcher
from .writer import Writer, WriterOutput

__all__ = [
    "Subagent",
    "Researcher",
    "ResearchFindings",
    "ResearchFinding",
    "Analyst",
    "AnalysisModel",
    "Critic",
    "CriticOutput",
    "CriticLoopResult",
    "run_critic_loop",
    "Writer",
    "WriterOutput",
]
