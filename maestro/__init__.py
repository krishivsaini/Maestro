"""Maestro — a supervisor-orchestrated multi-agent system.

A supervisor agent decomposes a goal into subtasks at runtime and delegates each
to a role-specialized subagent (Researcher, Analyst, Critic, Writer), each with
its own context. Independent subtasks run in bounded parallel; a Critic subagent
can reject and return work until it passes; subagent failures recover visibly;
and every run is replayable node-by-node.
"""

__version__ = "0.1.0"
