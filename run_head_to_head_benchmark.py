"""
Head-to-Head Benchmark Runner:
  [A] Bare `qwen3-coder-next:latest` (Single-Shot Pass@1)
  [B] Dynamic RSI Harness (`qwen3-coder-next:latest` + Multi-Critic + ACE Playbook + Sandbox Introspection)

Also stress-tests the "Runtime Already Correct, Task Still Fails -> ACE Playbook Evolution" loop:
  - We inject subtle adversarial spec requirements into the test suite so we can observe:
    1. Bare Model Pass@1 score & exact failure modes.
    2. Dynamic RSI Harness Loop 0 (JIT Rule Retrieval via `nomic-embed-text:latest`),
       Loop 1 (Multi-Agent Planner + Adversarial Edge-Case Critic + Generator), and
       Loop 2 (Introspective Trace Reflection -> ACE PlaybookRule Curation -> Zero-Shot Transfer).
"""

from __future__ import annotations
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rsi_harness.coding_benchmark import CodingBenchmarkTask, build_hard_coding_benchmark
from rsi_harness.llm_client import (
    LLMConfig,
    OpenAICompatibleClient,
    extract_code_block,
    extract_json_object,
)
from rsi_harness.playbook_and_skills import PlaybookMemory, SkillRegistry
from rsi_harness.schema import PlaybookRule, SynthesizedSkill


def evaluate_task_in_sandbox(
    skills: SkillRegistry,
    task: CodingBenchmarkTask,
    source_code: str,
) -> Tuple[bool, int, int, Optional[str]]:
    """
    Runs the task's assertion suite statement-by-statement against `source_code`
    to compute:
      - `all_passed`: bool
      - `assertions_passed`: int
      - `total_assertions`: int
      - `first_error`: detailed pytest-style LHS vs RHS diagnostic string
    """
    namespace: Dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(source_code, namespace)
    except Exception as exc:
        return False, 0, 1, f"SyntaxOrImportError: {type(exc).__name__}: {exc}"

    if task.entrypoint not in namespace:
        return False, 0, 1, f"MissingEntrypoint: `{task.entrypoint}` not defined."

    # Split top-level test blocks (single-line asserts or try/except blocks)
    blocks: List[str] = []
    current_block: List[str] = []
    for line in task.unit_tests.splitlines():
        if not line.strip():
            continue
        if (line.startswith(" ") or line.startswith("\t") or line.startswith("except")) and current_block:
            current_block.append(line)
        else:
            if current_block:
                blocks.append("\n".join(current_block))
            current_block = [line]
    if current_block:
        blocks.append("\n".join(current_block))

    passed_count = 0
    first_error: Optional[str] = None

    for idx, block in enumerate(blocks, start=1):
        try:
            exec(block, namespace)
            passed_count += 1
        except Exception as exc:
            if first_error is None:
                detail = str(exc)
                stripped = block.strip()
                if isinstance(exc, AssertionError) and stripped.startswith("assert ") and "==" in stripped:
                    expr_body = stripped[len("assert ") :].strip()
                    parts = expr_body.split("==", 1)
                    if len(parts) == 2:
                        try:
                            lhs_val = eval(parts[0].strip(), namespace)
                            rhs_val = eval(parts[1].strip(), namespace)
                            detail = f"LHS `{parts[0].strip()}` -> {lhs_val!r}, expected RHS -> {rhs_val!r}"
                        except Exception:
                            pass
                first_error = f"Test #{idx} failed (`{stripped.splitlines()[0]}`): {type(exc).__name__}: {detail}"

    all_passed = passed_count == len(blocks)
    return all_passed, passed_count, len(blocks), first_error


def run_bare_model_baseline(
    client: OpenAICompatibleClient,
    task: CodingBenchmarkTask,
    skills: SkillRegistry,
) -> Dict[str, Any]:
    """
    Mode A: Bare `qwen3-coder-next:latest` (Single-Shot Pass@1, no critic, no playbook, no sandbox retry).
    """
    resp = client.chat(
        messages=[{"role": "user", "content": task.prompt}],
        temperature=0.1,
        max_tokens=client.config.max_tokens,
    )
    code = extract_code_block(resp.content)
    all_passed, passed_cnt, total_cnt, err = evaluate_task_in_sandbox(skills, task, code)
    return {
        "task_id": task.task_id,
        "all_passed": all_passed,
        "passed_cnt": passed_cnt,
        "total_cnt": total_cnt,
        "tokens": resp.total_tokens,
        "latency_ms": resp.latency_ms,
        "error": err,
        "code": code,
    }


def run_dynamic_rsi_harness(
    client: OpenAICompatibleClient,
    task: CodingBenchmarkTask,
    playbook: PlaybookMemory,
    skills: SkillRegistry,
    max_attempts: int = 2,
) -> Dict[str, Any]:
    """
    Mode B: Dynamic RSI Harness (`qwen3-coder-next:latest` + Multi-Critic + ACE Playbook + Sandbox Gate).
    1. Loop 0 (JIT Harness Compilation):
       Queries `PlaybookMemory` using `nomic-embed-text:latest` + `domain_tags` to inject learned rules.
    2. Loop 1 (Multi-Agent Execution):
       - Agent 1 (Adversarial Spec & Edge-Case Critic): Analyzes the task + active ACE rules to produce
         an explicit Edge-Case & Invariant Checklist.
       - Agent 2 (Code Synthesizer): Implements the Python solution guided by the Critic's checklist
         and ACE Playbook rules.
       - Deterministic Sandbox Gate: Runs unit tests with LHS/RHS assertion introspection.
    3. Loop 2 (ACE Reflector & Curator):
       If Attempt $k$ fails the sandbox gate, the Reflector inspects the exact LHS vs RHS discrepancy,
       synthesizes a new `PlaybookRule` (`WHEN -> DO -> AVOID`), upserts it into `PlaybookMemory`,
       and repairs the implementation on Attempt $k+1$.
    """
    total_tokens = 0
    total_latency_ms = 0.0

    # Loop 0: Retrieve active ACE Playbook rules via `nomic-embed-text:latest`
    active_rules = playbook.retrieve_for_task(
        domain_tags=task.domain_tags,
        query_text=f"{task.title}: {task.prompt[:300]}",
        top_k=4,
    )
    rules_block = (
        "\n".join(f"  - {r.to_prompt_line()}" for r in active_rules)
        if active_rules
        else "  (No prior playbook rules mounted yet)"
    )

    # Loop 1A: Multi-Agent Step 1 — Adversarial Spec Critic & Architect
    critic_prompt = (
        "You are the Adversarial Spec Critic in a Multi-Agent Coding Harness.\n"
        f"Task: {task.title}\n"
        f"Specification:\n{task.prompt}\n\n"
        f"Active ACE Playbook Rules:\n{rules_block}\n\n"
        "List the top 4 subtle edge cases, precedence/state traps, or rounding/tie-breaker invariants "
        "that a developer must get right. Be concise (bullet points only)."
    )
    critic_resp = client.chat(
        messages=[{"role": "user", "content": critic_prompt}],
        temperature=0.1,
        max_tokens=max(1024, client.config.max_tokens // 2),
    )
    total_tokens += critic_resp.total_tokens
    total_latency_ms += critic_resp.latency_ms
    critic_checklist = critic_resp.content.strip()

    # Loop 1B: Multi-Agent Step 2 — Code Generator + Introspective Sandbox Loop
    gen_prompt = (
        f"{task.prompt}\n\n"
        f"=== ACTIVE ACE PLAYBOOK RULES ===\n{rules_block}\n\n"
        f"=== ADVERSARIAL CRITIC CHECKLIST ===\n{critic_checklist}\n\n"
        f"Write ONLY the complete Python implementation for `{task.entrypoint}` inside a ```python``` block."
    )
    messages: List[Dict[str, str]] = [{"role": "user", "content": gen_prompt}]

    first_shot_passed = False
    final_passed = False
    passed_cnt = 0
    total_cnt = 0
    last_error: Optional[str] = None
    synthesized_rules: List[PlaybookRule] = []
    attempts_used = 0

    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        gen_resp = client.chat(
            messages=messages,
            temperature=0.1,
            max_tokens=client.config.max_tokens,
        )
        total_tokens += gen_resp.total_tokens
        total_latency_ms += gen_resp.latency_ms

        code = extract_code_block(gen_resp.content)
        all_passed, passed_cnt, total_cnt, last_error = evaluate_task_in_sandbox(skills, task, code)

        if attempt == 1:
            first_shot_passed = all_passed

        if all_passed:
            final_passed = True
            playbook.record_outcome([r.rule_id for r in active_rules], True)
            break

        # Loop 2: Runtime was right, but task failed -> ACE Reflector extracts a PlaybookRule + repairs!
        # Self-Harness Two-Block Format: separate ```json``` rule from ```python``` code so dense models
        # like `deepseek-r1:32b` never corrupt multi-line Python strings inside JSON escaping!
        reflect_prompt = (
            "You are the ACE Reflector & Code Repair Agent.\n"
            f"Your implementation of `{task.entrypoint}` failed the sandbox verification gate with:\n"
            f"  {last_error}\n\n"
            "First, write a 3-line Rule summary:\n"
            "RULE_WHEN: <trigger condition>\n"
            "RULE_DO: <exact fix directive>\n"
            "RULE_AVOID: <mistake to avoid>\n\n"
            f"Second, provide the complete repaired Python implementation of `{task.entrypoint}` inside a ```python``` block."
        )
        messages.append({"role": "assistant", "content": gen_resp.content})
        messages.append({"role": "user", "content": reflect_prompt})

        repair_resp = client.chat(
            messages=messages,
            temperature=0.1,
            max_tokens=client.config.max_tokens,
            json_mode=False,
        )
        total_tokens += repair_resp.total_tokens
        total_latency_ms += repair_resp.latency_ms

        raw_text = repair_resp.content
        when_m = next((line.split(":", 1)[1].strip() for line in raw_text.splitlines() if line.strip().startswith("RULE_WHEN:")), f"Implementing {task.category}")
        do_m = next((line.split(":", 1)[1].strip() for line in raw_text.splitlines() if line.strip().startswith("RULE_DO:")), "Enforce strict spec invariants")
        avoid_m = next((line.split(":", 1)[1].strip() for line in raw_text.splitlines() if line.strip().startswith("RULE_AVOID:")), "Avoid naive assumptions")
        candidate_rule = PlaybookRule(
            rule_id=f"RULE-{task.task_id}-A{attempt}",
            domain_tags=task.domain_tags,
            when_condition=when_m,
            directive=do_m,
            anti_pattern=avoid_m,
            confidence=0.92,
        )
        repaired_code = extract_code_block(raw_text)

        all_passed, passed_cnt, total_cnt, last_error = evaluate_task_in_sandbox(skills, task, repaired_code)
        if all_passed:
            # Darwinian Playbook Gate: ONLY commit the candidate PlaybookRule when the repair passes verification!
            if candidate_rule is not None:
                playbook.upsert_rule(candidate_rule)
                synthesized_rules.append(candidate_rule)
            final_passed = True
            attempts_used = attempt + 1
            playbook.record_outcome([r.rule_id for r in active_rules] + [r.rule_id for r in synthesized_rules], True)
            break
        else:
            messages.append({"role": "assistant", "content": repair_resp.content})
            messages.append(
                {
                    "role": "user",
                    "content": f"Still failing sandbox gate: {last_error}\nOutput ONLY the fixed ```python``` code.",
                }
            )

    return {
        "task_id": task.task_id,
        "active_rules_mounted": [r.rule_id for r in active_rules],
        "first_shot_passed": first_shot_passed,
        "final_passed": final_passed,
        "passed_cnt": passed_cnt,
        "total_cnt": total_cnt,
        "attempts_used": attempts_used,
        "tokens": total_tokens,
        "latency_ms": total_latency_ms,
        "error": last_error,
        "new_rules": synthesized_rules,
    }


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3-coder-next:latest"
    print("=" * 94)
    print(f" HEAD-TO-HEAD BENCHMARK: BARE {model_name} vs. DYNAMIC RSI HARNESS")
    print("=" * 94)

    config = LLMConfig.lan_ollama(
        model=model_name,
        embedding_model="nomic-embed-text:latest",
        keep_alive="24h",
    )
    if "deepseek" in model_name.lower() or "qwen3.5" in model_name.lower():
        config.max_tokens = 2048
        config.timeout_sec = 900.0
    client = OpenAICompatibleClient(config)
    client.warm_up_ollama("24h")

    playbook = PlaybookMemory(llm_client=client)
    skills = SkillRegistry(llm_client=client)

    # Also add a tricky hidden edge-case task (BENCH-05) where operator precedence & banker's rounding
    # plus implicit whitespace/unary minus trap bare models unless guided by Critic + RSI!
    tasks = build_hard_coding_benchmark()
    # Add an extra adversarial assertion to BENCH-01 and BENCH-02: chained `@` with negative numbers & whitespace
    # and chained unary minus `-(2 + 3)` where naive tokenizers break!
    tasks[0].unit_tests += (
        "assert eval_fin_dsl('-(2 + 3) * 4', {}) == '-20.00'\n"
        "assert eval_fin_dsl('10 - 3 - 2', {}) == '5.00'\n"
    )
    tasks[1].unit_tests += (
        "assert eval_risk_formula('-(1 + 2) * 3', {}) == '-9.000'\n"
        "assert eval_risk_formula('10 - 3 - 2', {}) == '5.000'\n"
    )
    # Add an extra adversarial assertion to BENCH-03: COMMIT when outer tx has expired key & DELETE inside nested tx
    tasks[2].unit_tests += (
        "assert run_kv_ops([('SET', 'a', 1, 0, 10), ('BEGIN',), ('DELETE', 'a', 5), ('COMMIT',), ('GET', 'a', 6), ('COUNT', 6)]) == [True, None, 0]\n"
        "assert run_kv_ops([('SET', 'a', 1, 0, 5), ('BEGIN',), ('DELETE', 'a', 6), ('ROLLBACK',), ('COUNT', 6)]) == [False, 0]\n"
    )
    # Add an extra adversarial assertion to BENCH-04: Diamond dependency with 3-step cycle reachable only from a branch
    tasks[3].unit_tests += (
        "try:\n    schedule_dag_waves({'R': 10, 'X': 5, 'Y': 5, 'Z': 5}, [('R', 'X'), ('X', 'Y'), ('Y', 'Z'), ('Z', 'X')], 2)\n    assert False, 'Expected ValueError on downstream cycle'\nexcept ValueError:\n    pass\n"
    )

    bare_results = []
    rsi_results = []

    for task in tasks:
        print(f"\n{'-' * 94}")
        print(f"[TASK] {task.task_id}: {task.title}")
        print(f"{'-' * 94}")

        # 1. Run Bare Model
        b_res = run_bare_model_baseline(client, task, skills)
        bare_results.append(b_res)
        b_status = "PASS" if b_res["all_passed"] else f"FAIL ({b_res['passed_cnt']}/{b_res['total_cnt']})"
        print(
            f"  [A] BARE {model_name} (Pass@1): {b_status:<14} | "
            f"Tokens={b_res['tokens']:<5} | Latency={b_res['latency_ms']/1000:.1f}s"
        )
        if b_res["error"]:
            print(f"      -> Bare Failure Mode: {b_res['error']}")

        # 2. Run Dynamic RSI Harness
        r_res = run_dynamic_rsi_harness(client, task, playbook, skills)
        rsi_results.append(r_res)
        r_status = "PASS" if r_res["final_passed"] else f"FAIL ({r_res['passed_cnt']}/{r_res['total_cnt']})"
        print(
            f"  [B] DYNAMIC RSI HARNESS:            {r_status:<14} | "
            f"Attempts={r_res['attempts_used']} (1st-shot={'PASS' if r_res['first_shot_passed'] else 'FAIL'}) | "
            f"MountedRules={r_res['active_rules_mounted']} | "
            f"Tokens={r_res['tokens']:<5} | Latency={r_res['latency_ms']/1000:.1f}s"
        )
        if r_res["new_rules"]:
            for nr in r_res["new_rules"]:
                print(f"      -> ACE Playbook Rule Learned: {nr.to_prompt_line()}")
        if r_res["error"] and not r_res["final_passed"]:
            print(f"      -> Remaining Error: {r_res['error']}")

    # Summary Table
    bare_tasks_passed = sum(1 for r in bare_results if r["all_passed"])
    rsi_first_shot = sum(1 for r in rsi_results if r["first_shot_passed"])
    rsi_final_passed = sum(1 for r in rsi_results if r["final_passed"])
    bare_assertions = sum(r["passed_cnt"] for r in bare_results)
    rsi_assertions = sum(r["passed_cnt"] for r in rsi_results)
    total_assertions = sum(r["total_cnt"] for r in bare_results)

    print(f"\n{'=' * 94}")
    print(" FINAL BENCHMARK SCORE COMPARISON")
    print(f"{'=' * 94}")
    print(f"  Bare `{model_name}` (Pass@1):             {bare_tasks_passed}/{len(tasks)} Tasks ({bare_tasks_passed/len(tasks)*100:.1f}%) | {bare_assertions}/{total_assertions} Assertions ({bare_assertions/total_assertions*100:.1f}%)")
    print(f"  Dynamic RSI Harness (1st-Shot w/ Critic + Playbook): {rsi_first_shot}/{len(tasks)} Tasks ({rsi_first_shot/len(tasks)*100:.1f}%)")
    print(f"  Dynamic RSI Harness (With Intra-Task RSI Loop):      {rsi_final_passed}/{len(tasks)} Tasks ({rsi_final_passed/len(tasks)*100:.1f}%) | {rsi_assertions}/{total_assertions} Assertions ({rsi_assertions/total_assertions*100:.1f}%)")
    print("=" * 94)
