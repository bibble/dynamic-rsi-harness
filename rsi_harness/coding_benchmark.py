"""
Hard Coding & Multi-Agent Workflow Benchmark:
  Bare `qwen3-coder-next:latest` (Pass@1) vs. Dynamic RSI Harness (`qwen3-coder-next:latest` + ACE + DGM + Multi-Critic)

Evaluates 4 tricky software-engineering & algorithmic tasks with adversarial edge-case test suites
(plus a Held-Out Transfer Task to test zero-shot transfer of learned ACE Playbook Rules):

1. BENCH-01 (Train) -> Held-Out Transfer to BENCH-02 (Test):
   Custom Spec DSL & Temporal Expression Evaluator with Non-Standard Operator Precedence
   & Financially Exact Half-Even Rounding.
2. BENCH-03:
   Transactional In-Memory KV Store with Nested `BEGIN` / `ROLLBACK` / `COMMIT` & TTL Expiration.
3. BENCH-04:
   DAG Task Scheduler with Diamond Dependency Deduplication, Cycle Detection, and Priority Tie-Breaking.

For each task we run:
  [A] BARE MODEL (Single-shot Pass@1): Raw `qwen3-coder-next:latest` prompt -> code -> test suite.
  [B] DYNAMIC RSI HARNESS:
      - Loop 0: JIT Harness Compilation (mounts relevant ACE Playbook rules via `nomic-embed-text:latest`
                and configures Multi-Critic / Deterministic Verification).
      - Loop 1: Multi-Agent Execution (Planner/Critic -> Generator -> Pytest-style Introspective Sandbox).
      - Loop 2: When an attempt fails, the ACE Reflector extracts a surgical `PlaybookRule`
                (`WHEN -> DO -> AVOID`) from the exact failing assertion LHS/RHS trace, repairs the code,
                and persists the rule in `PlaybookMemory` so subsequent tasks benefit zero-shot!
"""

from __future__ import annotations
from dataclasses import dataclass
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rsi_harness.llm_client import (
    LLMConfig,
    OpenAICompatibleClient,
    extract_code_block,
    extract_json_object,
)
from rsi_harness.playbook_and_skills import PlaybookMemory, SkillRegistry
from rsi_harness.schema import PlaybookRule, SynthesizedSkill


@dataclass
class CodingBenchmarkTask:
    task_id: str
    title: str
    category: str
    domain_tags: List[str]
    entrypoint: str
    prompt: str
    unit_tests: str
    is_transfer_target: bool = False


def build_hard_coding_benchmark() -> List[CodingBenchmarkTask]:
    return [
        # -------------------------------------------------------------------------
        # Task 1: Non-Standard Precedence & Exact Decimal Rounding (Trap: Python eval() / float)
        # -------------------------------------------------------------------------
        CodingBenchmarkTask(
            task_id="BENCH-01-CUSTOM-DSL",
            title="Custom Financial Pipe DSL (`eval_fin_dsl`)",
            category="dsl_compiler_and_numerics",
            domain_tags=["dsl", "parser", "precedence", "decimal", "rounding"],
            entrypoint="eval_fin_dsl",
            prompt=(
                "Write a Python function `eval_fin_dsl(expr: str, vars_dict: dict[str, str]) -> str`.\n"
                "The DSL evaluates arithmetic expressions over decimal numbers and variable names:\n"
                "1. Numbers are decimal strings (e.g., `'10.25'`, `'-3.5'`) and variables in `vars_dict` map to decimal strings.\n"
                "2. Supported binary operators are `+`, `-`, `*`, and `@` (where `a @ b` means `(a + b) / 2`).\n"
                "3. CRITICAL OPERATOR PRECEDENCE RULE: Unlike standard math, in this DSL:\n"
                "   - `+` and `-` have the HIGHEST precedence (evaluated first, left-to-right).\n"
                "   - `*` and `@` have the LOWEST precedence (evaluated second, left-to-right).\n"
                "   - Parentheses `(...)` override precedence as usual.\n"
                "4. All intermediate arithmetic MUST be exact (use `decimal.Decimal`, never `float`), and the final\n"
                "   returned string MUST be formatted to exactly 2 decimal places using `ROUND_HALF_EVEN` (banker's rounding),\n"
                "   e.g. `'12.34'`. If a variable is missing from `vars_dict`, raise `KeyError`.\n"
                "Return ONLY valid Python code defining `eval_fin_dsl` and any helper functions."
            ),
            unit_tests=(
                "assert eval_fin_dsl('2 + 3 * 4', {}) == '20.00'\n"
                "assert eval_fin_dsl('10 - 2 @ 4', {}) == '6.00'\n"
                "assert eval_fin_dsl('2 * (3 + 4) - 1', {}) == '12.00'\n"
                "assert eval_fin_dsl('x + y * z', {'x': '1.005', 'y': '0.000', 'z': '2.5'}) == '2.51'\n"
                "assert eval_fin_dsl('0.1 + 0.2 - 0.3', {}) == '0.00'\n"
                "assert eval_fin_dsl('1.005 * 1', {}) == '1.00'\n"
                "assert eval_fin_dsl('1.015 * 1', {}) == '1.02'\n"
            ),
        ),
        # -------------------------------------------------------------------------
        # Task 2 (Held-Out Transfer Task): Shares the Inverted-Precedence & Decimal Trap
        # -------------------------------------------------------------------------
        CodingBenchmarkTask(
            task_id="BENCH-02-TRANSFER-DSL",
            title="Risk-Weighted Matrix Formula Evaluator (`eval_risk_formula`)",
            category="dsl_compiler_and_numerics",
            domain_tags=["dsl", "parser", "precedence", "decimal", "rounding"],
            entrypoint="eval_risk_formula",
            is_transfer_target=True,
            prompt=(
                "Write a Python function `eval_risk_formula(formula: str, env: dict[str, str]) -> str`.\n"
                "Rules:\n"
                "1. Operands are decimal strings or variable identifiers looked up in `env`.\n"
                "2. Binary operators:\n"
                "   - HIGHEST precedence (left-to-right): `+` and `-`\n"
                "   - LOWEST precedence (left-to-right): `*` and `~` (where `a ~ b` is `(a * b) / (a + b)`, or `Decimal('0')` if `a + b == 0`).\n"
                "   - Parentheses `(...)` override precedence.\n"
                "3. Use exact `decimal.Decimal` arithmetic throughout (never `float`) and format the final output to\n"
                "   exactly 3 decimal places (`'0.000'`) using `ROUND_HALF_EVEN`.\n"
                "Return ONLY valid Python code defining `eval_risk_formula`."
            ),
            unit_tests=(
                "assert eval_risk_formula('1 + 2 * 3 + 4', {}) == '21.000'\n"
                "assert eval_risk_formula('6 ~ 3 + 3', {}) == '3.000'\n"
                "assert eval_risk_formula('5 + -5 ~ 10', {}) == '0.000'\n"
                "assert eval_risk_formula('a + b * c', {'a': '1.0005', 'b': '0.000', 'c': '1'}) == '1.000'\n"
                "assert eval_risk_formula('a + b * c', {'a': '1.0015', 'b': '0.000', 'c': '1'}) == '1.002'\n"
            ),
        ),
        # -------------------------------------------------------------------------
        # Task 3: Nested Transactions + TTL Tombstone Trap
        # -------------------------------------------------------------------------
        CodingBenchmarkTask(
            task_id="BENCH-03-TX-KV-STORE",
            title="Nested Transactional KV Store with TTL (`run_kv_ops`)",
            category="stateful_systems_engineering",
            domain_tags=["transactions", "kv_store", "ttl", "state_machine"],
            entrypoint="run_kv_ops",
            prompt=(
                "Write a Python function `run_kv_ops(ops: list[tuple]) -> list` that simulates an in-memory key-value store\n"
                "supporting nested transactions and per-key Time-To-Live (TTL) timestamps.\n"
                "Supported tuples in `ops`:\n"
                "  - `('SET', key, val, now_ts, ttl_sec)`: sets `key` to `val`. If `ttl_sec` is not `None`, the key expires\n"
                "    strictly AT or AFTER `now_ts + ttl_sec` (i.e., alive when `query_ts < now_ts + ttl_sec`, expired when `query_ts >= now_ts + ttl_sec`).\n"
                "    Setting a key overwrites any prior TTL on that key.\n"
                "  - `('GET', key, query_ts)`: appends the current value of `key` at `query_ts` (or `None` if missing/deleted/expired) to the output list.\n"
                "  - `('DELETE', key, now_ts)`: removes `key`. Appends `True` to output list if the key existed and was NOT expired at `now_ts`, else `False`.\n"
                "  - `('COUNT', query_ts)`: appends the count of currently alive (non-deleted, non-expired at `query_ts`) keys to the output list.\n"
                "  - `('BEGIN',)` : opens a new nested transaction scope.\n"
                "  - `('COMMIT',)` : merges the innermost open transaction into its parent scope (or base store). If no transaction is open, appends `'NO_TX'`.\n"
                "  - `('ROLLBACK',)` : discards all changes (`SET` and `DELETE`, including TTL updates) made in the innermost open transaction. If no transaction is open, appends `'NO_TX'`.\n"
                "Return the output list produced by `GET`, `DELETE`, `COUNT`, and failed `COMMIT`/`ROLLBACK`."
            ),
            unit_tests=(
                "assert run_kv_ops([('SET', 'a', 10, 100, 50), ('GET', 'a', 149), ('GET', 'a', 150)]) == [10, None]\n"
                "assert run_kv_ops([('SET', 'a', 1, 10, 20), ('BEGIN',), ('SET', 'a', 2, 15, None), ('GET', 'a', 50), ('ROLLBACK',), ('GET', 'a', 25), ('GET', 'a', 30)]) == [2, 1, None]\n"
                "assert run_kv_ops([('SET', 'x', 5, 0, 10), ('DELETE', 'x', 10), ('SET', 'y', 7, 0, 10), ('DELETE', 'y', 9)]) == [False, True]\n"
                "assert run_kv_ops([('SET', 'k', 'v1', 0, None), ('BEGIN',), ('DELETE', 'k', 1), ('COUNT', 2), ('BEGIN',), ('SET', 'k', 'v2', 2, 5), ('COUNT', 4), ('ROLLBACK',), ('COUNT', 4), ('ROLLBACK',), ('GET', 'k', 10)]) == [True, 0, 1, 0, 'v1']\n"
                "assert run_kv_ops([('ROLLBACK',), ('COMMIT',)]) == ['NO_TX', 'NO_TX']\n"
            ),
        ),
        # -------------------------------------------------------------------------
        # Task 4: DAG Critical-Path & Priority Topological Wave Scheduler
        # -------------------------------------------------------------------------
        CodingBenchmarkTask(
            task_id="BENCH-04-DAG-SCHEDULER",
            title="Bounded-Concurrency Priority DAG Wave Scheduler (`schedule_dag_waves`)",
            category="graph_algorithms",
            domain_tags=["dag", "scheduler", "topological_sort", "cycle_detection"],
            entrypoint="schedule_dag_waves",
            prompt=(
                "Write a Python function `schedule_dag_waves(tasks: dict[str, int], deps: list[tuple[str, str]], max_workers: int) -> list[list[str]]`.\n"
                "Specification:\n"
                "1. `tasks` maps `task_id -> priority` (higher integer = higher priority).\n"
                "2. `deps` is a list of `(u, v)` meaning task `u` MUST complete in a strictly earlier wave before task `v` can start.\n"
                "   - Note: `deps` may contain duplicate edges `(u, v)`! Deduplicate them so in-degrees are not double-counted.\n"
                "   - If `deps` references a task ID not in `tasks`, raise `ValueError('Unknown task')`.\n"
                "   - Self-loops `(u, u)` or cycles of any length must raise `ValueError('Cycle detected')`.\n"
                "3. Execution happens in discrete waves. In each wave, among all currently ready tasks (in-degree == 0), select at most\n"
                "   `max_workers` tasks ordered by:\n"
                "   - PRIMARY: highest `priority` first (`-priority`)\n"
                "   - TIE-BREAKER: lexicographically smallest `task_id` first (`task_id` ascending)\n"
                "4. All tasks selected in a wave run in parallel and complete at the end of that wave, unlocking their dependents for the NEXT wave.\n"
                "   The task IDs within each returned wave list must be ordered by the selection order above.\n"
                "Return the list of waves (`list[list[str]]`). If `tasks` is empty, return `[]`."
            ),
            unit_tests=(
                "assert schedule_dag_waves({}, [], 2) == []\n"
                "assert schedule_dag_waves({'A': 1, 'B': 5, 'C': 5}, [('A', 'B'), ('A', 'B'), ('A', 'C')], 1) == [['A'], ['B'], ['C']]\n"
                "assert schedule_dag_waves({'T1': 10, 'T2': 20, 'T3': 20, 'T4': 5}, [('T2', 'T4'), ('T3', 'T4')], 2) == [['T2', 'T3'], ['T1', 'T4']]\n"
                "try:\n    schedule_dag_waves({'A': 1, 'B': 2}, [('A', 'B'), ('B', 'A')], 2)\n    assert False, 'Expected ValueError on cycle'\nexcept ValueError:\n    pass\n"
                "try:\n    schedule_dag_waves({'A': 1}, [('A', 'Z')], 2)\n    assert False, 'Expected ValueError on unknown task'\nexcept ValueError:\n    pass\n"
            ),
        ),
    ]
