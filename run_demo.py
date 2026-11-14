"""
End-to-End Prototype Demonstration of a Universal Dynamic Agentic Harness
with Recursive Self-Improvement (RSI).

Demonstrates how a bounded model (with strict context-length limits, arithmetic/symbolic
drift, and single-shot coding blindness) autonomously evolves its own harness across
three distinct task families:
  1. Long-Context Log Anomaly Extraction (Context Window Saturation -> Map-Reduce Chunking)
  2. Multi-Step Symbolic Modular Cascade (Token Math Hallucination -> DGM Tool Synthesis)
  3. Algorithmic Interval Merging with Edge Cases (Single-Shot Blindness -> Multi-Critic + Guardrails)
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Tuple
from rsi_harness.playbook_and_skills import PlaybookMemory, SkillRegistry
from rsi_harness.rsi_engine import (
    BoundedModelSimulator,
    DynamicHarnessExecutor,
    MetaArchitect,
    SelfHarnessRSIEngine,
)


# ---------------------------------------------------------------------------
# Benchmark Tasks & Ground-Truth Verifiers
# ---------------------------------------------------------------------------

def build_benchmark_suite() -> List[Tuple[str, str, Dict[str, Any], Callable[[Any, Dict[str, Any]], Tuple[bool, str]]]]:
    # Task 1: Context-Intensive Log Analysis (exceeds bounded model's 1200-token window)
    records = []
    expected_anomalies = []
    for i in range(1, 45):
        is_middle_anomaly = i in (12, 23, 34)
        status = "CRITICAL_ANOMALY" if is_middle_anomaly else "OK"
        if is_middle_anomaly:
            expected_anomalies.append(f"LOG-{i:03d}")
        records.append({
            "id": f"LOG-{i:03d}",
            "status": status,
            "message": f"Subsystem telemetry heartbeat frame {i} with extended diagnostic payload " * 3,
        })

    def verify_logs(output: Any, payload: Dict[str, Any]) -> Tuple[bool, str]:
        if output == expected_anomalies:
            return True, f"All {len(expected_anomalies)} middle-context anomalies recovered."
        return False, f"Context drop: expected {expected_anomalies}, got {output}."

    # Task 2: Symbolic Cascade Checksum (triggers System-1 token math drift without a tool)
    numbers = [142, 891, 3045, 77, 6192, 401, 9999, 2718, 31415]
    modulus = 10007
    acc = 17
    for x in numbers:
        acc = ((acc * 31 + x) ^ (x >> 2)) % modulus
    expected_checksum = acc

    def verify_symbolic(output: Any, payload: Dict[str, Any]) -> Tuple[bool, str]:
        if output == expected_checksum:
            return True, f"Exact symbolic checksum verified ({expected_checksum})."
        return False, f"Symbolic drift: expected checksum {expected_checksum}, got {output}."

    # Task 3: Algorithmic Code Synthesis (unsorted + nested intervals edge cases)
    intervals = [(8, 12), (1, 10), (2, 6), (15, 18), (14, 15)]
    expected_merged = [(1, 12), (14, 18)]

    def verify_intervals(output: Any, payload: Dict[str, Any]) -> Tuple[bool, str]:
        if output == expected_merged:
            return True, f"All unsorted & nested intervals merged accurately: {output}."
        return False, f"EdgeCase bug on unsorted/nested intervals: expected {expected_merged}, got {output}."

    return [
        (
            "TASK-01-LONG-CONTEXT",
            "Analyze long_context telemetry log transcript and extract all CRITICAL_ANOMALY IDs.",
            {"records": records},
            verify_logs,
        ),
        (
            "TASK-02-SYMBOLIC-MATH",
            "Compute exact_math modular cascade checksum over the sequence.",
            {"numbers": numbers, "modulus": modulus},
            verify_symbolic,
        ),
        (
            "TASK-03-ALGO-CODE",
            "Implement interval merge algorithm handling unsorted and nested intervals.",
            {"intervals": intervals},
            verify_intervals,
        ),
    ]


def run_rsi_demo() -> None:
    print("=" * 88)
    print(" DYNAMIC AGENTIC HARNESS & RECURSIVE SELF-IMPROVEMENT (RSI) PROTOTYPE")
    print("=" * 88)

    playbook = PlaybookMemory()
    skills = SkillRegistry()
    model = BoundedModelSimulator(context_limit_tokens=1200)
    architect = MetaArchitect(playbook=playbook, skills=skills, default_context_limit=1200)
    executor = DynamicHarnessExecutor(model=model, playbook=playbook, skills=skills)
    rsi_engine = SelfHarnessRSIEngine(
        architect=architect,
        executor=executor,
        playbook=playbook,
        skills=skills,
    )

    suite = build_benchmark_suite()

    for task_id, prompt, payload, verifier in suite:
        print(f"\n{'-' * 88}")
        print(f"[USER TASK] {task_id}: {prompt}")
        print(f"{'-' * 88}")

        # 1. Loop 0: Compile Gen-1 Baseline Harness
        gen1_harness = architect.compile_harness(prompt)
        print(
            f"[Loop 0 | JIT Architect] Compiled Gen-{gen1_harness.generation} Harness -> "
            f"Topology={gen1_harness.topology.value}, "
            f"Context={gen1_harness.context_strategy.value}, "
            f"Skills={gen1_harness.active_skills}"
        )

        # 2. Loop 1: Execute Task under Gen-1 Harness
        gen1_trace = executor.execute(task_id, prompt, payload, gen1_harness, verifier)
        status_str = "PASS" if gen1_trace.passed_verification else "FAIL"
        print(
            f"[Loop 1 | Execution]     Outcome={status_str} | "
            f"PeakTokens={gen1_trace.peak_context_tokens}/{model.context_limit_tokens} | "
            f"Feedback: {gen1_trace.verification_feedback}"
        )

        # 3. Loop 2: Recursive Self-Improvement (Weakness Mining -> Proposal -> Validation Gate)
        if not gen1_trace.passed_verification:
            proposal = rsi_engine.mine_weakness_and_propose(gen1_trace)
            assert proposal is not None, "Expected RSI Engine to propose a harness mutation!"
            print(f"[Loop 2 | WeaknessMiner] Diagnosed: {proposal.diagnosed_weakness}")
            print(f"                         Root Cause: {proposal.root_cause}")
            if proposal.new_skill:
                print(f"                         DGM Skill Synthesized: '{proposal.new_skill.name}' (with unit tests)")
            if proposal.new_playbook_rule:
                print(f"                         ACE Playbook Delta: {proposal.new_playbook_rule.to_prompt_line()}")

            passed_gate, gen2_trace = rsi_engine.validate_and_commit(
                proposal, (task_id, prompt, payload, verifier)
            )
            evolved = gen2_trace.harness_snapshot
            print(
                f"[Loop 2 | RegressionGate] Committed Gen-{evolved.generation} Harness -> "
                f"Topology={evolved.topology.value}, "
                f"Context={evolved.context_strategy.value}, "
                f"Skills={evolved.active_skills}, "
                f"Guardrails={evolved.post_execution_guardrails}"
            )
            print(
                f"[Loop 1 | Re-Execution]  Outcome={'PASS' if passed_gate else 'FAIL'} | "
                f"PeakTokens={gen2_trace.peak_context_tokens}/{model.context_limit_tokens} | "
                f"Feedback: {gen2_trace.verification_feedback}"
            )
            assert passed_gate, f"Task {task_id} should pass after RSI harness evolution!"

    print(f"\n{'=' * 88}")
    print(" FINAL EVOLVED SYSTEM STATE (ACE PLAYBOOK + DGM SKILL REGISTRY)")
    print(f"{'=' * 88}")
    print(f"Verified Self-Authored Skills ({len(skills.all_skills)}):")
    for sk in skills.all_skills:
        print(f"  - {sk.name}: {sk.description} (verified={sk.verified}, uses={sk.usage_count})")

    print(f"\nACE Structured Playbook Rules ({len(playbook.all_rules)}):")
    for rule in playbook.all_rules:
        print(f"  - {rule.to_prompt_line()}")
    print("=" * 88)


if __name__ == "__main__":
    run_rsi_demo()
