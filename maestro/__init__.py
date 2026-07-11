"""Maestro — a supervisor-orchestrated multi-agent system.

A supervisor agent decomposes a goal into subtasks at runtime and delegates each
to a role-specialized subagent (Researcher, Analyst, Critic, Writer), each with
its own context. Independent subtasks run in bounded parallel; a Critic subagent
can reject and return work until it passes; subagent failures recover visibly;
and every run is replayable node-by-node.
"""

import os

# faiss-cpu bundles its own libomp on macOS; when a second OpenMP runtime is pulled
# in (e.g. during the threaded research fan-out) the process aborts with "OMP: Error
# #15 ... libomp.dylib already initialized". This is the runtime's own documented
# escape hatch, set before faiss is imported so the duplicate load is tolerated.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
