"""The LangGraph StateGraph — the supervisor pattern made explicit (§9, §15).

Shape::

    START -> supervisor(plan) -> research(bounded-parallel) --.
                                                              |
                        (failed researchers)  recover <--+    |  (all ok)
                                                |        |    |
                                                +--------+    |
                        (recovery exhausted)  degrade         |
                                                |             |
                                                v             v
                                              analyze <-------'
                                                  |
                                                  v
                             write <--(PASS / ceiling)-- critique
                               |                           |
                              END                (REJECT & under ceiling) --> analyze

- ``supervisor`` performs dynamic decomposition (§10).
- ``research`` runs the researcher subtasks with bounded parallelism (§11).
- ``recover``/``degrade`` implement visible recovery (§15): a failed researcher is
  retried (bounded by ``MAX_RECOVERY_ATTEMPTS``); if recovery is exhausted the
  subtask is degraded and the run proceeds with partial evidence, flagged in the
  output. Every recovery decision is a logged trace transition.
- ``critique`` -> ``analyze`` is the critic loop, bounded by ``MAX_CRITIC_ITERS``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

from langgraph.graph import END, START, StateGraph

from .agents import Analyst, Critic, Researcher, Writer
from .config import Settings, get_settings
from .logging_config import get_logger
from .loop_control import repeated_delegation, step_ceiling_reached, stuck_in_critic_loop
from .memory import distill_findings
from .scheduler import run_schedule
from .state import (
    CriticDecision,
    EventType,
    MaestroState,
    Role,
    RunStatus,
    Subtask,
    SubtaskStatus,
    TraceEvent,
    new_state,
)
from .supervisor import Planner, build_llm_planner, decompose

log = get_logger("graph")


@dataclass
class MaestroAgents:
    researcher: object  # anything with .run(subtask, tools=, corpus=) -> (Subtask, list[Evidence])
    analyst: Analyst
    critic: Critic
    writer: Writer
    settings: Settings
    planner: Optional[Planner] = None  # None -> LLM planner built on demand
    tools: object = None  # ToolRegistry; None -> DEFAULT_REGISTRY
    corpus: Optional[list] = None  # local retrieval corpus for offline research
    memory: object = None  # LongTermMemory; None -> long-term memory disabled


def build_default_agents(settings: Optional[Settings] = None) -> MaestroAgents:
    cfg = settings or get_settings()
    return MaestroAgents(
        researcher=Researcher(settings=cfg),
        analyst=Analyst(settings=cfg),
        critic=Critic(settings=cfg),
        writer=Writer(settings=cfg),
        settings=cfg,
        planner=build_llm_planner(cfg),
    )


def _ev(event_type: EventType, agent: str, summary: str, **kw) -> TraceEvent:
    return TraceEvent(event_type=event_type, agent=agent, summary=summary, **kw)


def _first_by_role(subtasks: list[Subtask], role: Role) -> Subtask:
    for s in subtasks:
        if s.role == role:
            return s
    return Subtask(id=f"auto_{role.value}", description=f"{role.value} task", role=role)


def _researchers(subtasks: list[Subtask]) -> list[Subtask]:
    return [s for s in subtasks if s.role == Role.researcher]


def build_graph(
    agents: Optional[MaestroAgents] = None,
    *,
    settings: Optional[Settings] = None,
):
    """Compile and return the Maestro StateGraph."""
    ag = agents or build_default_agents(settings)
    cfg = ag.settings

    def _run_researchers(researchers: list[Subtask], label: str):
        """Run a set of researcher subtasks bounded-parallel; collect evidence + events."""
        collected: list = []
        events: list[TraceEvent] = []
        lock = threading.Lock()

        def worker(st: Subtask) -> Subtask:
            updated, evidence = ag.researcher.run(st, tools=ag.tools, corpus=ag.corpus)
            with lock:
                collected.extend(evidence)
            return updated

        def on_event(phase: str, st: Subtask) -> None:  # single-threaded (scheduler loop)
            if phase == "dispatched":
                events.append(_ev(EventType.subtask_dispatched, "researcher",
                                  f"{label} {st.id}", subtask_id=st.id))
            else:
                events.append(_ev(EventType.subagent_result, "researcher",
                                  f"{st.id} -> {st.status.value}", subtask_id=st.id))

        updated, report = run_schedule(researchers, worker, cap=cfg.max_parallel, on_event=on_event)
        return updated, collected, events, report

    # --- nodes ---
    def supervisor_node(state: MaestroState) -> dict:
        if state.get("subtasks"):
            return {}
        subs = decompose(state["goal"], planner=ag.planner, settings=cfg)
        events = [_ev(EventType.plan_produced, "supervisor",
                      f"planned {len(subs)} subtasks: {[s.id for s in subs]}")]
        update = {
            "subtasks": subs,
            "step_count": state.get("step_count", 0) + 1,
            "status": RunStatus.running.value,
        }
        # long-term memory recall (§14): inform this run with prior findings
        if ag.memory is not None:
            hits = ag.memory.query(state["thread_id"], state["goal"], k=3)
            if hits:
                update["memory_hits"] = hits
                events.append(_ev(EventType.memory_recall, "supervisor",
                                  f"recalled {len(hits)} prior finding(s) from long-term memory"))
        update["trace"] = events
        return update

    def research_node(state: MaestroState) -> dict:
        researchers = _researchers(state["subtasks"])
        if not researchers:
            return {"step_count": state.get("step_count", 0) + 1}
        updated, evidence, events, report = _run_researchers(researchers, "dispatched")
        log.info("research: %d subtasks, max_concurrency=%d, %d evidence",
                 len(updated), report.max_concurrency, len(evidence))
        return {
            "subtasks": updated,
            "evidence": evidence,
            "trace": events,
            "step_count": state.get("step_count", 0) + 1,
        }

    def recover_node(state: MaestroState) -> dict:
        failed = [s for s in _researchers(state["subtasks"]) if s.status == SubtaskStatus.failed]
        attempt = state.get("recovery_attempts", 0) + 1
        decision = "retry" if attempt == 1 else "re-delegate"
        retry_subs = [s.model_copy(update={"status": SubtaskStatus.pending}) for s in failed]
        updated, evidence, events, _ = _run_researchers(retry_subs, f"recover({decision})")
        rec = _ev(EventType.recovery, "supervisor",
                  f"recovery attempt {attempt} ({decision}) on {[s.id for s in failed]}",
                  recovery_decision=decision)
        log.info("recovery attempt %d (%s) on %s", attempt, decision, [s.id for s in failed])
        return {
            "subtasks": updated,
            "evidence": evidence,
            "recovery_attempts": attempt,
            "trace": [rec] + events,
            "step_count": state.get("step_count", 0) + 1,
        }

    def degrade_node(state: MaestroState) -> dict:
        failed = [s for s in _researchers(state["subtasks"]) if s.status == SubtaskStatus.failed]
        degraded = [s.model_copy(update={"status": SubtaskStatus.degraded}) for s in failed]
        events = [
            _ev(EventType.degraded, "supervisor",
                f"degraded {s.id} after exhausting recovery; proceeding on partial evidence",
                subtask_id=s.id)
            for s in failed
        ]
        return {
            "subtasks": degraded,
            "trace": events,
            "step_count": state.get("step_count", 0) + 1,
        }

    def recover_router(state: MaestroState) -> str:
        failed = [s for s in _researchers(state["subtasks"]) if s.status == SubtaskStatus.failed]
        if not failed:
            return "analyze"
        # global safety nets: step ceiling or a delegation loop -> stop retrying
        if step_ceiling_reached(state, cfg) or repeated_delegation(state.get("trace", []), cfg.loop_detect_threshold):
            return "degrade"
        if state.get("recovery_attempts", 0) < cfg.max_recovery_attempts:
            return "recover"
        return "degrade"

    def analyze_node(state: MaestroState) -> dict:
        analyst_st = _first_by_role(state["subtasks"], Role.analyst)
        verdicts = state.get("critic_verdicts", [])
        feedback = verdicts[-1].feedback if (verdicts and verdicts[-1].decision == CriticDecision.rejected) else None
        revision = state.get("critic_iterations", 0)
        updated, draft = ag.analyst.run(
            analyst_st, state["goal"], state.get("evidence", []),
            feedback=feedback, revision=revision, memory_hits=state.get("memory_hits", []),
        )
        ev = _ev(EventType.subagent_result, "analyst",
                 f"analysis draft rev {revision}", subtask_id=analyst_st.id)
        return {
            "analysis": draft,
            "subtasks": [updated],
            "trace": [ev],
            "step_count": state.get("step_count", 0) + 1,
        }

    def critique_node(state: MaestroState) -> dict:
        critic_st = _first_by_role(state["subtasks"], Role.critic)
        iteration = state.get("critic_iterations", 0) + 1
        updated, verdict = ag.critic.run(
            critic_st, state["goal"], state["analysis"], state.get("evidence", []), iteration=iteration
        )
        etype = EventType.critic_pass if verdict.decision == CriticDecision.passed else EventType.critic_reject
        ev = _ev(etype, "critic", f"review {iteration}: {verdict.decision.value}",
                 subtask_id=critic_st.id, critic_verdict=verdict.decision.value)
        return {
            "critic_verdicts": [verdict],
            "critic_iterations": iteration,
            "subtasks": [updated],
            "trace": [ev],
            "step_count": state.get("step_count", 0) + 1,
        }

    def critic_router(state: MaestroState) -> str:
        last = state["critic_verdicts"][-1]
        if last.decision == CriticDecision.passed:
            return "write"
        if state.get("critic_iterations", 0) >= cfg.max_critic_iters:
            return "write"  # ceiling reached -> degrade gracefully
        # global safety nets: step ceiling or a stalled (no-progress) critic loop
        if step_ceiling_reached(state, cfg) or stuck_in_critic_loop(state):
            return "write"
        return "revise"

    def write_node(state: MaestroState) -> dict:
        writer_st = _first_by_role(state["subtasks"], Role.writer)
        verdicts = state.get("critic_verdicts", [])
        critic_passed = bool(verdicts) and verdicts[-1].decision == CriticDecision.passed
        degraded_subs = [s for s in state["subtasks"] if s.status == SubtaskStatus.degraded]

        # A hard HALT (forced termination by a loop/cost control) is distinct from a
        # graceful DEGRADE (critic ceiling / partial evidence).
        halt_reason = None
        if step_ceiling_reached(state, cfg):
            halt_reason = f"step ceiling (MAX_STEPS={cfg.max_steps}) reached"
        elif stuck_in_critic_loop(state) and state.get("critic_iterations", 0) < cfg.max_critic_iters:
            halt_reason = "loop detector: critic gave identical feedback repeatedly without progress"

        validated = critic_passed and halt_reason is None
        updated, answer = ag.writer.run(
            writer_st, state["goal"], state["analysis"], state.get("evidence", []), validated=validated
        )
        notes = []
        if degraded_subs:
            notes.append(f"{len(degraded_subs)} research subtask(s) degraded; brief rests on partial evidence.")
        if halt_reason:
            notes.append(f"Run halted: {halt_reason}.")
        if notes:
            joined = " ".join(notes)
            answer = answer.model_copy(update={"notes": f"{answer.notes + ' ' if answer.notes else ''}{joined}".strip()})

        if halt_reason:
            status = RunStatus.halted.value
        elif critic_passed and not degraded_subs:
            status = RunStatus.completed.value
        else:
            status = RunStatus.degraded.value

        events = [_ev(EventType.subagent_result, "writer", "final brief composed", subtask_id=writer_st.id)]
        if halt_reason:
            events.append(_ev(EventType.halted, "supervisor", halt_reason))
        elif not critic_passed:
            events.append(_ev(EventType.degraded, "supervisor", "critic ceiling reached; not fully validated"))

        # long-term memory write (§14): persist distilled findings for later turns
        if ag.memory is not None:
            findings = distill_findings(state.get("analysis"), answer)
            for finding in findings:
                ag.memory.add(state["thread_id"], finding)
            if findings:
                events.append(_ev(EventType.memory_write, "supervisor",
                                  f"stored {len(findings)} finding(s) to long-term memory"))

        events.append(_ev(EventType.completed, "supervisor", f"run completed ({status})"))
        return {
            "final_output": answer,
            "status": status,
            "subtasks": [updated],
            "trace": events,
            "step_count": state.get("step_count", 0) + 1,
        }

    # --- wire the graph ---
    sg = StateGraph(MaestroState)
    sg.add_node("supervisor", supervisor_node)
    sg.add_node("research", research_node)
    sg.add_node("recover", recover_node)
    sg.add_node("degrade", degrade_node)
    sg.add_node("analyze", analyze_node)
    sg.add_node("critique", critique_node)
    sg.add_node("write", write_node)

    sg.add_edge(START, "supervisor")
    sg.add_edge("supervisor", "research")
    _recover_map = {"recover": "recover", "degrade": "degrade", "analyze": "analyze"}
    sg.add_conditional_edges("research", recover_router, _recover_map)
    sg.add_conditional_edges("recover", recover_router, _recover_map)
    sg.add_edge("degrade", "analyze")
    sg.add_edge("analyze", "critique")
    sg.add_conditional_edges("critique", critic_router, {"revise": "analyze", "write": "write"})
    sg.add_edge("write", END)
    return sg.compile()


def run_goal(
    goal: str,
    *,
    agents: Optional[MaestroAgents] = None,
    settings: Optional[Settings] = None,
    thread_id: Optional[str] = None,
    recursion_limit: int = 50,
) -> MaestroState:
    """Convenience: build the graph and run one goal to completion."""
    graph = build_graph(agents=agents, settings=settings)
    return graph.invoke(new_state(goal, thread_id=thread_id), {"recursion_limit": recursion_limit})
