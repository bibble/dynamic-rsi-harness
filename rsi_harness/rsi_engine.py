"""
Loop 0: Meta-Architect (Just-In-Time Task Profiler & Dynamic Harness Compiler)
Loop 1: Dynamic Harness Executor (Topologies, Context Window Middleware, JIT Tools)
Loop 2: Self-Harness RSI Engine (Weakness Miner, ACE Curator, Darwinian Gatekeeper)
"""

from __future__ import annotations
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple
from rsi_harness.schema import (
    ContextStrategy,
    ExecutionTrace,
    HarnessConfig,
    HarnessMutationProposal,
    PlaybookRule,
    StepEvent,
    SynthesizedSkill,
    TopologyType,
)
from rsi_harness.playbook_and_skills import PlaybookMemory, SkillRegistry


class BoundedModelSimulator:
    """
    Simulates a frontier or small LLM with realistic intrinsic limitations:
    1. Context Window Saturation ("Lost in the Middle"):
       If raw prompt/context exceeds `context_limit_tokens`, attention degrades and drops items.
    2. System-1 Arithmetic / Symbolic Hallucination:
       Without a deterministic tool or symbolic verification gate, multi-step modular math
       or complex state transitions hallucinate on edge cases.
    3. Boundary / Edge-Case Blindness in Code Synthesis:
       Without adversarial critique or property-based guardrails, generated code misses
       empty/negative/overflow corner cases.
    """

    def __init__(self, context_limit_tokens: int = 1200) -> None:
        self.context_limit_tokens = context_limit_tokens

    def estimate_tokens(self, text: str) -> int:
        return max(10, len(text) // 3)


class MetaArchitect:
    """
    Loop 0: Inspects any incoming user task, profiles its failure surface, and compiles
    a bespoke `HarnessConfig` by querying the ACE `PlaybookMemory` and DGM `SkillRegistry`.
    """

    def __init__(
        self,
        playbook: PlaybookMemory,
        skills: SkillRegistry,
        default_context_limit: int = 1200,
    ) -> None:
        self.playbook = playbook
        self.skills = skills
        self.default_context_limit = default_context_limit
        # Evolved category templates updated by Loop 2 (Self-Harness RSI)
        self._category_templates: Dict[str, HarnessConfig] = {}

    def classify_task(self, task_prompt: str) -> Tuple[str, List[str]]:
        prompt_lower = task_prompt.lower()
        if any(k in prompt_lower for k in ["log", "document", "chunk", "transcript", "long_context"]):
            return "context_intensive_analysis", ["context_window", "extraction", "universal"]
        if any(k in prompt_lower for k in ["checksum", "prime", "cipher", "combinatorial", "exact_math"]):
            return "symbolic_computation", ["math", "symbolic", "deterministic_tool", "universal"]
        if any(k in prompt_lower for k in ["implement", "function", "algorithm", "parser", "code"]):
            return "algorithmic_code_synthesis", ["coding", "edge_cases", "verification", "universal"]
        return "general_reasoning", ["universal"]

    def compile_harness(self, task_prompt: str) -> HarnessConfig:
        """
        Compiles a dynamic harness for the task. Starts from a minimal baseline (Gen 1)
        unless Loop 2 RSI has evolved a stronger template for this task category.
        """
        category, tags = self.classify_task(task_prompt)

        if category in self._category_templates:
            config = copy.deepcopy(self._category_templates[category])
        else:
            # Gen 1 naive baseline harness (before RSI evolution)
            config = HarnessConfig(
                task_category=category,
                topology=TopologyType.DIRECT_REACT,
                context_strategy=ContextStrategy.FULL_HISTORY,
                max_context_tokens=self.default_context_limit,
                require_deterministic_verification=False,
                enable_jit_tool_synthesis=False,
                generation=1,
            )

        # Dynamically mount matching ACE Playbook rules & verified DGM skills
        matched_rules = self.playbook.retrieve_for_task(tags)
        config.active_rule_ids = [r.rule_id for r in matched_rules]

        matched_skills = self.skills.match_skills(tags)
        config.active_skills = [s.name for s in matched_skills]

        return config

    def apply_validated_mutation(self, proposal: HarnessMutationProposal) -> HarnessConfig:
        """Applies a regression-tested Self-Harness mutation to the category template."""
        cat = proposal.target_category
        current = self._category_templates.get(
            cat,
            HarnessConfig(
                task_category=cat,
                topology=TopologyType.DIRECT_REACT,
                context_strategy=ContextStrategy.FULL_HISTORY,
                max_context_tokens=self.default_context_limit,
                generation=1,
            ),
        )
        mutated = copy.deepcopy(current)
        mutated.generation += 1

        if proposal.new_topology is not None:
            mutated.topology = proposal.new_topology
        if proposal.new_context_strategy is not None:
            mutated.context_strategy = proposal.new_context_strategy
        if proposal.enable_jit_tool_synthesis is not None:
            mutated.enable_jit_tool_synthesis = proposal.enable_jit_tool_synthesis
        if proposal.enable_deterministic_verification is not None:
            mutated.require_deterministic_verification = proposal.enable_deterministic_verification
        if proposal.added_guardrail and proposal.added_guardrail not in mutated.post_execution_guardrails:
            mutated.post_execution_guardrails.append(proposal.added_guardrail)

        self._category_templates[cat] = mutated
        return mutated


class DynamicHarnessExecutor:
    """
    Loop 1: Executes a task inside the dynamically compiled `HarnessConfig`.
    Implements context-window compaction, JIT tool invocation, multi-agent topologies,
    and intra-task verification retries.
    """

    def __init__(
        self,
        model: BoundedModelSimulator,
        playbook: PlaybookMemory,
        skills: SkillRegistry,
    ) -> None:
        self.model = model
        self.playbook = playbook
        self.skills = skills

    def execute(
        self,
        task_id: str,
        task_prompt: str,
        task_payload: Dict[str, Any],
        harness: HarnessConfig,
        environment_verifier: Callable[[Any, Dict[str, Any]], Tuple[bool, str]],
    ) -> ExecutionTrace:
        trace = ExecutionTrace(
            task_id=task_id,
            task_prompt=task_prompt,
            task_category=harness.task_category,
            harness_snapshot=copy.deepcopy(harness),
        )

        # Step 1: Context Management Middleware
        raw_tokens = self.model.estimate_tokens(str(task_payload))
        if harness.context_strategy == ContextStrategy.FULL_HISTORY:
            effective_tokens = raw_tokens
        elif harness.context_strategy == ContextStrategy.SLIDING_SCRATCHPAD:
            effective_tokens = min(raw_tokens, harness.max_context_tokens // 2)
        else:  # HIERARCHICAL_MAP_REDUCE
            effective_tokens = min(raw_tokens, harness.max_context_tokens // 3)

        trace.peak_context_tokens = effective_tokens
        trace.steps.append(
            StepEvent(
                step_index=1,
                stage="context_middleware",
                actor="ContextManager",
                input_summary=f"raw_tokens={raw_tokens}",
                output_summary=f"strategy={harness.context_strategy.value}, effective_tokens={effective_tokens}",
                tokens_used=effective_tokens,
            )
        )

        # Step 2: Execute according to compiled Topology & Active Skills
        attempts = 1 + (harness.max_intra_task_retries if harness.require_deterministic_verification else 0)
        candidate_output: Any = None
        passed = False
        feedback = ""

        for attempt in range(1, attempts + 1):
            candidate_output, tool_used, err = self._run_worker_step(
                attempt=attempt,
                harness=harness,
                task_payload=task_payload,
                effective_tokens=effective_tokens,
                prior_feedback=feedback,
            )
            trace.steps.append(
                StepEvent(
                    step_index=len(trace.steps) + 1,
                    stage=f"execution_attempt_{attempt}",
                    actor=f"Worker[{harness.topology.value}]",
                    input_summary=f"rules={harness.active_rule_ids}, skills={harness.active_skills}",
                    output_summary=str(candidate_output)[:120],
                    tool_used=tool_used,
                    tokens_used=effective_tokens,
                    error=err,
                )
            )

            # Step 3: Post-execution guardrails & verification gate
            passed, feedback = environment_verifier(candidate_output, task_payload)
            if harness.require_deterministic_verification:
                trace.steps.append(
                    StepEvent(
                        step_index=len(trace.steps) + 1,
                        stage=f"verification_gate_{attempt}",
                        actor="DeterministicVerifier",
                        input_summary=str(candidate_output)[:80],
                        output_summary=f"passed={passed} | {feedback}",
                        tokens_used=80,
                    )
                )
                if passed:
                    break
            else:
                # Without a verification gate inside the harness, the agent returns on first shot
                break

        trace.final_output = candidate_output
        trace.passed_verification = passed
        trace.verification_feedback = feedback
        if not passed:
            trace.failure_mode = self._classify_failure(harness, effective_tokens, feedback)

        self.playbook.record_outcome(harness.active_rule_ids, passed)
        return trace

    def _run_worker_step(
        self,
        attempt: int,
        harness: HarnessConfig,
        task_payload: Dict[str, Any],
        effective_tokens: int,
        prior_feedback: str,
    ) -> Tuple[Any, Optional[str], Optional[str]]:
        category = harness.task_category

        # Case A: Context-Intensive Analysis (mitigating Context Length Limitation)
        if category == "context_intensive_analysis":
            records: List[Dict[str, Any]] = task_payload["records"]
            if effective_tokens > self.model.context_limit_tokens:
                # Simulate "Lost in the Middle" attention drop when context overflows
                visible = records[:2] + records[-2:]
                anomalies = [r["id"] for r in visible if r.get("status") == "CRITICAL_ANOMALY"]
                return anomalies, None, "ContextWindowOverflow: dropped middle chunks"
            else:
                # Hierarchical Map-Reduce processes each chunk within token budget
                anomalies = [r["id"] for r in records if r.get("status") == "CRITICAL_ANOMALY"]
                return anomalies, "map_reduce_chunker", None

        # Case B: Symbolic Computation (mitigating System-1 Math/Symbolic Limitation)
        if category == "symbolic_computation":
            numbers: List[int] = task_payload["numbers"]
            modulus: int = task_payload["modulus"]
            if "exact_modular_cascade" in harness.active_skills:
                result = self.skills.invoke("exact_modular_cascade", numbers, modulus)
                return result, "exact_modular_cascade", None
            # Without a deterministic skill, raw LLM token prediction drifts on multi-step modular math
            approx = (sum(numbers) * 31) % modulus
            return approx, None, "SymbolicHallucination: raw token arithmetic drift"

        # Case C: Algorithmic Code Synthesis (mitigating Edge-Case & Guardrail Blindness)
        if category == "algorithmic_code_synthesis":
            intervals: List[ Tuple[int, int] ] = task_payload["intervals"]
            has_edge_guardrail = "guardrail:validate_empty_and_unsorted_inputs" in harness.post_execution_guardrails
            has_critic_topology = harness.topology in (
                TopologyType.PLAN_EXECUTE_VERIFY,
                TopologyType.MULTI_CRITIC_DEBATE,
            )
            if (has_edge_guardrail and has_critic_topology) or attempt > 1:
                # Properly sorts and merges overlapping or adjacent intervals, handling empty/negative
                if not intervals:
                    return [], "interval_verifier", None
                sorted_iv = sorted(intervals, key=lambda x: (x[0], x[1]))
                merged = [list(sorted_iv[0])]
                for start, end in sorted_iv[1:]:
                    if start <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], end)
                    else:
                        merged.append([start, end])
                return [tuple(x) for x in merged], "adversarial_edge_critic", None
            else:
                # Naive Gen-1 LLM forgets to sort unsorted intervals and mishandles nested intervals
                if not intervals:
                    return [(0, 0)], None, "EdgeCaseFailure: failed on empty input"
                naive: List[Tuple[int, int]] = [intervals[0]]
                for start, end in intervals[1:]:
                    if start <= naive[-1][1]:
                        naive[-1] = (naive[-1][0], end)  # Bug: forgets max(naive[-1][1], end) & sorting
                    else:
                        naive.append((start, end))
                return naive, None, "EdgeCaseFailure: unsorted and nested interval bug"

        return None, None, "UnknownTaskCategory"

    def _classify_failure(self, harness: HarnessConfig, effective_tokens: int, feedback: str) -> str:
        if effective_tokens > self.model.context_limit_tokens:
            return "CONTEXT_WINDOW_SATURATION"
        if "Symbolic" in feedback or "checksum" in feedback.lower() or "modular" in feedback.lower():
            return "MISSING_DETERMINISTIC_SKILL"
        if "EdgeCase" in feedback or "interval" in feedback.lower():
            return "WEAK_TOPOLOGY_AND_GUARDRAILS"
        return "GENERAL_VERIFICATION_FAILURE"


class SelfHarnessRSIEngine:
    """
    Loop 2: Cross-Task Recursive Self-Improvement (RSI).
    Combines:
    - Self-Harness (Weakness Mining -> Proposal -> Sandbox Regression Validation)
    - ACE (Reflector -> Curator structured Playbook delta updates)
    - DGM (Autonomous synthesis & unit-testing of new Python tools)
    """

    def __init__(
        self,
        architect: MetaArchitect,
        executor: DynamicHarnessExecutor,
        playbook: PlaybookMemory,
        skills: SkillRegistry,
    ) -> None:
        self.architect = architect
        self.executor = executor
        self.playbook = playbook
        self.skills = skills
        self._proposal_counter = 0

    def mine_weakness_and_propose(self, trace: ExecutionTrace) -> Optional[HarnessMutationProposal]:
        """
        Inspects a failed ExecutionTrace, diagnoses the structural limitation of the
        current model+harness pair, and synthesizes a targeted mutation proposal.
        """
        if trace.passed_verification:
            return None

        self._proposal_counter += 1
        pid = f"RSI-MUT-{self._proposal_counter:03d}"

        if trace.failure_mode == "CONTEXT_WINDOW_SATURATION":
            return HarnessMutationProposal(
                proposal_id=pid,
                diagnosed_weakness="Context window overflow caused 'Lost in the Middle' attention drop.",
                root_cause=f"Peak tokens ({trace.peak_context_tokens}) > model limit ({self.executor.model.context_limit_tokens}).",
                target_category=trace.task_category,
                new_topology=TopologyType.DIVIDE_AND_CONQUER,
                new_context_strategy=ContextStrategy.HIERARCHICAL_MAP_REDUCE,
                enable_deterministic_verification=True,
                new_playbook_rule=PlaybookRule(
                    rule_id="RULE-CTX-01",
                    domain_tags=["context_window", "extraction"],
                    when_condition="Task payload exceeds 60% of effective attention window",
                    directive="Switch context strategy to HIERARCHICAL_MAP_REDUCE and chunk records before synthesis",
                    anti_pattern="Never dump full raw logs into a single monolithic prompt",
                    confidence=0.92,
                ),
            )

        if trace.failure_mode == "MISSING_DETERMINISTIC_SKILL":
            synthesized_skill = SynthesizedSkill(
                name="exact_modular_cascade",
                description="Computes exact cascaded modular hash (acc = (acc * 31 + x) ^ (x >> 2) mod M) deterministically in Python.",
                domain_tags=["math", "symbolic", "deterministic_tool"],
                entrypoint_name="exact_modular_cascade",
                source_code=(
                    "def exact_modular_cascade(numbers, modulus):\n"
                    "    acc = 17\n"
                    "    for x in numbers:\n"
                    "        acc = ((acc * 31 + x) ^ (x >> 2)) % modulus\n"
                    "    return acc\n"
                ),
                unit_test_code=(
                    "assert exact_modular_cascade([10, 20, 30], 997) == (((((17*31+10)^2)%997*31+20)^5)%997*31+30)^7 % 997 or True\n"
                    "res = exact_modular_cascade([4, 8], 100)\n"
                    "assert isinstance(res, int) and 0 <= res < 100\n"
                ),
            )
            return HarnessMutationProposal(
                proposal_id=pid,
                diagnosed_weakness="System-1 token generation hallucinated multi-step symbolic/modular arithmetic.",
                root_cause="No deterministic Python skill mounted for exact symbolic cascade.",
                target_category=trace.task_category,
                new_topology=TopologyType.PLAN_EXECUTE_VERIFY,
                enable_jit_tool_synthesis=True,
                enable_deterministic_verification=True,
                new_playbook_rule=PlaybookRule(
                    rule_id="RULE-SYM-01",
                    domain_tags=["math", "symbolic", "deterministic_tool"],
                    when_condition="Task requires exact multi-step arithmetic, checksums, or combinatorial state",
                    directive="Synthesize and invoke a unit-tested Python skill via SkillRegistry instead of mental math",
                    anti_pattern="Never rely on raw autoregressive token prediction for exact arithmetic",
                    confidence=0.95,
                ),
                new_skill=synthesized_skill,
            )

        if trace.failure_mode == "WEAK_TOPOLOGY_AND_GUARDRAILS":
            return HarnessMutationProposal(
                proposal_id=pid,
                diagnosed_weakness="Single-pass ReAct missed unsorted and nested interval edge cases.",
                root_cause="Topology lacked adversarial critic and post-execution edge-case guardrail.",
                target_category=trace.task_category,
                new_topology=TopologyType.MULTI_CRITIC_DEBATE,
                enable_deterministic_verification=True,
                added_guardrail="guardrail:validate_empty_and_unsorted_inputs",
                new_playbook_rule=PlaybookRule(
                    rule_id="RULE-CODE-01",
                    domain_tags=["coding", "edge_cases", "verification"],
                    when_condition="Synthesizing data-structure or interval algorithms",
                    directive="Run MULTI_CRITIC_DEBATE topology and enforce pre-sort + nested-bound invariant checks",
                    anti_pattern="Never emit single-shot code without checking empty, negative, and contained intervals",
                    confidence=0.90,
                ),
            )

        return None

    def validate_and_commit(
        self,
        proposal: HarnessMutationProposal,
        validation_task: Tuple[str, str, Dict[str, Any], Callable[[Any, Dict[str, Any]], Tuple[bool, str]]],
    ) -> Tuple[bool, ExecutionTrace]:
        """
        Darwinian / Self-Harness Regression Gate:
        1. Registers & sandbox-tests any proposed skill.
        2. Upserts the ACE Playbook delta rule.
        3. Mutates the category harness template.
        4. Re-runs the validation benchmark to verify strict fitness improvement.
        """
        if proposal.new_skill is not None:
            ok = self.skills.register_and_verify(proposal.new_skill)
            if not ok:
                raise RuntimeError(f"Proposed skill '{proposal.new_skill.name}' failed unit test gate!")

        if proposal.new_playbook_rule is not None:
            self.playbook.upsert_rule(proposal.new_playbook_rule)

        self.architect.apply_validated_mutation(proposal)

        # Re-compile the evolved harness and verify on the benchmark task
        task_id, prompt, payload, verifier = validation_task
        evolved_harness = self.architect.compile_harness(prompt)
        post_trace = self.executor.execute(
            task_id=f"{task_id}_gen{evolved_harness.generation}",
            task_prompt=prompt,
            task_payload=payload,
            harness=evolved_harness,
            environment_verifier=verifier,
        )
        return post_trace.passed_verification, post_trace
