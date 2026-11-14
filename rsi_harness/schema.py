"""
Core data structures for Dynamic Agentic Harnesses & Recursive Self-Improvement (RSI).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class TopologyType(str, Enum):
    """Execution graph topologies compiled dynamically per task."""
    DIRECT_REACT = "direct_react"                  # Fast single-loop tool execution
    PLAN_EXECUTE_VERIFY = "plan_execute_verify"    # Decompose -> Execute -> Deterministic Gate
    MULTI_CRITIC_DEBATE = "multi_critic_debate"    # Generator + Adversarial Critic + Synthesizer
    DIVIDE_AND_CONQUER = "divide_and_conquer"      # Map-Reduce chunking for context-heavy tasks


class ContextStrategy(str, Enum):
    """Context window management strategies to mitigate context length limitations."""
    FULL_HISTORY = "full_history"                  # Standard append-only messages
    SLIDING_SCRATCHPAD = "sliding_scratchpad"      # Compresses intermediate steps into structured state
    HIERARCHICAL_MAP_REDUCE = "hierarchical_map_reduce"  # Isolates sub-contexts per chunk


@dataclass
class PlaybookRule:
    """
    Agentic Context Engineering (ACE) structured delta rule.
    Prevents context collapse and brevity bias by storing granular, versioned heuristics.
    """
    rule_id: str
    domain_tags: List[str]
    when_condition: str
    directive: str
    anti_pattern: str
    confidence: float = 0.8
    times_applied: int = 0
    times_succeeded: int = 0

    def to_prompt_line(self) -> str:
        return (
            f"[{self.rule_id} | conf={self.confidence:.2f}] "
            f"WHEN {self.when_condition} -> DO: {self.directive} "
            f"(AVOID: {self.anti_pattern})"
        )


@dataclass
class SynthesizedSkill:
    """
    Darwin Gödel Machine (DGM) / LATM style self-authored Python tool/skill.
    Offloads brittle LLM token reasoning into deterministic code execution.
    """
    name: str
    description: str
    domain_tags: List[str]
    source_code: str
    entrypoint_name: str
    unit_test_code: str
    verified: bool = False
    verification_error: Optional[str] = None
    usage_count: int = 0
    callable_fn: Optional[Callable[..., Any]] = field(default=None, repr=False)


@dataclass
class HarnessConfig:
    """
    The dynamic harness configuration compiled Just-In-Time (JIT) for a task,
    and mutated across recursive generations via Self-Harness RSI.
    """
    task_category: str
    topology: TopologyType
    context_strategy: ContextStrategy
    max_context_tokens: int
    active_skills: List[str] = field(default_factory=list)
    active_rule_ids: List[str] = field(default_factory=list)
    require_deterministic_verification: bool = False
    enable_jit_tool_synthesis: bool = False
    pre_execution_checks: List[str] = field(default_factory=list)
    post_execution_guardrails: List[str] = field(default_factory=list)
    max_intra_task_retries: int = 2
    generation: int = 1


@dataclass
class StepEvent:
    """Single step in an agent's execution trace."""
    step_index: int
    stage: str
    actor: str
    input_summary: str
    output_summary: str
    tool_used: Optional[str] = None
    tokens_used: int = 0
    error: Optional[str] = None


@dataclass
class ExecutionTrace:
    """Full telemetry captured during a task run for Self-Harness Weakness Mining."""
    task_id: str
    task_prompt: str
    task_category: str
    harness_snapshot: HarnessConfig
    steps: List[StepEvent] = field(default_factory=list)
    final_output: Any = None
    passed_verification: bool = False
    verification_feedback: str = ""
    peak_context_tokens: int = 0
    failure_mode: Optional[str] = None


@dataclass
class HarnessMutationProposal:
    """
    A proposed recursive self-improvement mutation produced by the Outer-Loop Reflector/Curator.
    Covers ACE Playbook rules, DGM Skill synthesis, and Self-Harness topology/guardrail edits.
    """
    proposal_id: str
    diagnosed_weakness: str
    root_cause: str
    target_category: str
    new_topology: Optional[TopologyType] = None
    new_context_strategy: Optional[ContextStrategy] = None
    enable_jit_tool_synthesis: Optional[bool] = None
    enable_deterministic_verification: Optional[bool] = None
    added_guardrail: Optional[str] = None
    new_playbook_rule: Optional[PlaybookRule] = None
    new_skill: Optional[SynthesizedSkill] = None
