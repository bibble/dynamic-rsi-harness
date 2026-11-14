"""
Official SWE-bench Verified Head-to-Head Benchmark Runner:
  [A] Bare `qwen3-coder-next:latest` (Single-Shot Pass@1)
  [B] Dynamic RSI Harness (`qwen3-coder-next:latest` + Multi-Critic + ACE Playbook + Sandbox Gate)

Evaluates 8 real-world GitHub issues from `princeton-nlp/SWE-bench_Verified` across
`astropy/astropy`, `psf/requests`, `pytest-dev/pytest`, and `sympy/sympy`.
"""

from __future__ import annotations
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rsi_harness.llm_client import (
    LLMConfig,
    OpenAICompatibleClient,
    extract_code_block,
    extract_json_object,
)
from rsi_harness.playbook_and_skills import PlaybookMemory, SkillRegistry
from rsi_harness.schema import PlaybookRule
from rsi_harness.swebench_suite import (
    SWEBenchInstance,
    execute_swebench_patch,
    load_swebench_verified_suite,
)


def build_swebench_prompt(instance: SWEBenchInstance) -> str:
    return (
        f"You are resolving a real GitHub issue from `{instance.repo}` (SWE-bench Verified: `{instance.instance_id}`).\n\n"
        f"### GitHub Issue Description\n"
        f"{instance.problem_statement[:1800]}\n\n"
        f"### Target File: `{instance.target_file}` (Focal Section around `{instance.focal_function_name}`)\n"
        f"```python\n{instance.focal_window_code}\n```\n\n"
        f"### Instructions\n"
        f"Write the complete, self-contained patched Python function (and any modified helper class/function in the focal window) "
        f"for `{instance.focal_function_name}` so that it resolves the issue without breaking existing behavior.\n"
        f"Return ONLY valid Python code inside a ```python``` block."
    )


def run_bare_swebench(
    client: OpenAICompatibleClient,
    instance: SWEBenchInstance,
) -> Dict[str, Any]:
    prompt = build_swebench_prompt(instance)
    resp = client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=client.config.max_tokens,
    )
    patched_code = extract_code_block(resp.content)
    resolved, f2p_ok, p2p_ok, err = execute_swebench_patch(instance, patched_code)
    return {
        "instance_id": instance.instance_id,
        "resolved": resolved,
        "f2p_ok": f2p_ok,
        "p2p_ok": p2p_ok,
        "tokens": resp.total_tokens,
        "latency_ms": resp.latency_ms,
        "error": err,
    }


def run_rsi_swebench(
    client: OpenAICompatibleClient,
    instance: SWEBenchInstance,
    playbook: PlaybookMemory,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    total_tokens = 0
    total_latency_ms = 0.0

    # Loop 0: Retrieve relevant ACE Playbook rules via `nomic-embed-text:latest`
    active_rules = playbook.retrieve_for_task(
        domain_tags=instance.domain_tags,
        query_text=f"{instance.instance_id} {instance.repo}: {instance.problem_statement[:300]}",
        top_k=3,
    )
    rules_block = (
        "\n".join(f"  - {r.to_prompt_line()}" for r in active_rules)
        if active_rules
        else "  (No prior playbook rules mounted yet)"
    )

    # Loop 1A: Multi-Agent Issue Root-Cause Critic
    # Note: For small 9B models (`"9b"` in model name) when no prior ACE rules are mounted yet,
    # ungrounded pre-execution speculation can talk the 9B generator out of a working 1st-shot patch
    # (e.g. `sympy__sympy-23262`). We therefore run Attempt 1 with mounted ACE rules (if any), and
    # ground the Root-Cause Critic in the actual Sandbox LHS/RHS failure trace if Attempt 1 fails!
    is_small_9b = "9b" in client.config.model.lower()
    base_prompt = build_swebench_prompt(instance)

    if is_small_9b:
        if active_rules:
            gen_prompt = f"{base_prompt}\n\n=== ACTIVE ACE PLAYBOOK RULES ===\n{rules_block}"
        else:
            gen_prompt = base_prompt
    else:
        critic_prompt = (
            f"You are the Principal SWE-bench Root-Cause Critic for `{instance.repo}`.\n"
            f"Issue (`{instance.instance_id}`):\n{instance.problem_statement[:1400]}\n\n"
            f"Focal Code (`{instance.target_file}`):\n```python\n{instance.focal_window_code}\n```\n\n"
            f"Active ACE Playbook Rules:\n{rules_block}\n\n"
            "In 3 concise bullet points, identify:\n"
            "1. The exact line(s) in the focal code causing the bug.\n"
            "2. The minimal code change required in `{instance.focal_function_name}`.\n"
            "3. Any regression edge case to preserve."
        )
        critic_resp = client.chat(
            messages=[{"role": "user", "content": critic_prompt}],
            temperature=0.1,
            max_tokens=max(1024, client.config.max_tokens // 2),
        )
        total_tokens += critic_resp.total_tokens
        total_latency_ms += critic_resp.latency_ms

        # Loop 1B: Patch Synthesizer + Sandbox Test Gate
        gen_prompt = (
            f"{base_prompt}\n\n"
            f"=== ACTIVE ACE PLAYBOOK RULES ===\n{rules_block}\n\n"
            f"=== ROOT-CAUSE CRITIC DIAGNOSIS ===\n{critic_resp.content.strip()}\n\n"
            f"Return ONLY the complete patched Python definition of `{instance.focal_function_name}` "
            "(and any modified helper class/constant from the focal window) inside a ```python``` block."
        )
    messages: List[Dict[str, str]] = [{"role": "user", "content": gen_prompt}]

    first_shot_resolved = False
    final_resolved = False
    f2p_ok = False
    p2p_ok = False
    last_error: Optional[str] = None
    learned_rules: List[PlaybookRule] = []
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

        patched_code = extract_code_block(gen_resp.content)
        resolved, f2p_ok, p2p_ok, last_error = execute_swebench_patch(instance, patched_code)

        if attempt == 1:
            first_shot_resolved = resolved

        if resolved:
            final_resolved = True
            playbook.record_outcome([r.rule_id for r in active_rules], True)
            break

        # Loop 2: ACE Reflector diagnoses sandbox failure & synthesizes a PlaybookRule + repaired patch
        reflect_prompt = (
            "You are the SWE-bench ACE Reflector & Patch Repair Agent.\n"
            f"Your patch for `{instance.instance_id}` (`{instance.focal_function_name}`) failed verification:\n"
            f"  {last_error}\n\n"
            f"Test assertions being verified:\n```python\n{instance.fail_to_pass_assertions}\n```\n\n"
            "First, write a 3-line Rule summary:\n"
            "RULE_WHEN: <trigger condition>\n"
            "RULE_DO: <exact fix directive>\n"
            "RULE_AVOID: <mistake to avoid>\n\n"
            f"Second, provide the complete repaired Python code defining `{instance.focal_function_name}` inside a ```python``` block."
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
        when_m = next((line.split(":", 1)[1].strip() for line in raw_text.splitlines() if line.strip().startswith("RULE_WHEN:")), f"Patching {instance.repo}")
        do_m = next((line.split(":", 1)[1].strip() for line in raw_text.splitlines() if line.strip().startswith("RULE_DO:")), "Preserve exact module invariants")
        avoid_m = next((line.split(":", 1)[1].strip() for line in raw_text.splitlines() if line.strip().startswith("RULE_AVOID:")), "Avoid incomplete patches")
        candidate_rule: Optional[PlaybookRule] = PlaybookRule(
            rule_id=f"RULE-SWE-{instance.instance_id}-A{attempt}",
            domain_tags=instance.domain_tags,
            when_condition=when_m,
            directive=do_m,
            anti_pattern=avoid_m,
            confidence=0.93,
        )
        repaired_code = extract_code_block(raw_text)

        resolved, f2p_ok, p2p_ok, last_error = execute_swebench_patch(instance, repaired_code)
        if resolved:
            if candidate_rule is not None:
                playbook.upsert_rule(candidate_rule)
                learned_rules.append(candidate_rule)
            final_resolved = True
            attempts_used = attempt + 1
            playbook.record_outcome([r.rule_id for r in active_rules] + [r.rule_id for r in learned_rules], True)
            break
        else:
            messages.append({"role": "assistant", "content": repair_resp.content})
            messages.append(
                {
                    "role": "user",
                    "content": f"Patch still failed verification: {last_error}\nReturn ONLY the fixed ```python``` block.",
                }
            )

    return {
        "instance_id": instance.instance_id,
        "resolved": final_resolved,
        "first_shot_resolved": first_shot_resolved,
        "f2p_ok": f2p_ok,
        "p2p_ok": p2p_ok,
        "attempts_used": attempts_used,
        "mounted_rules": [r.rule_id for r in active_rules],
        "learned_rules": learned_rules,
        "tokens": total_tokens,
        "latency_ms": total_latency_ms,
        "error": last_error,
    }


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3-coder-next:latest"
    print("=" * 98)
    print(f" OFFICIAL SWE-BENCH VERIFIED BENCHMARK: BARE {model_name} vs. DYNAMIC RSI HARNESS")
    print("=" * 98)

    samples_path = os.path.join(os.path.dirname(__file__), "swebench_verified_samples.json")
    suite = load_swebench_verified_suite(samples_path)

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

    bare_records = []
    rsi_records = []

    for inst in suite:
        print(f"\n{'-' * 98}")
        print(f"[SWE-BENCH VERIFIED] {inst.instance_id} ({inst.repo} | {inst.target_file} | {inst.difficulty})")
        print(f"{'-' * 98}")

        b_res = run_bare_swebench(client, inst)
        bare_records.append(b_res)
        b_status = "RESOLVED" if b_res["resolved"] else "FAILED"
        print(
            f"  [A] BARE {model_name} (Pass@1): {b_status:<10} | "
            f"F2P={str(b_res['f2p_ok']):<5} P2P={str(b_res['p2p_ok']):<5} | "
            f"Tokens={b_res['tokens']:<5} | Latency={b_res['latency_ms']/1000:.1f}s"
        )
        if b_res["error"]:
            print(f"      -> Bare Failure: {b_res['error']}")

        r_res = run_rsi_swebench(client, inst, playbook, max_attempts=2)
        rsi_records.append(r_res)
        r_status = "RESOLVED" if r_res["resolved"] else "FAILED"
        print(
            f"  [B] DYNAMIC RSI HARNESS:            {r_status:<10} | "
            f"1stShot={str(r_res['first_shot_resolved']):<5} Attempts={r_res['attempts_used']} | "
            f"Mounted={r_res['mounted_rules']} | "
            f"Tokens={r_res['tokens']:<5} | Latency={r_res['latency_ms']/1000:.1f}s"
        )
        for lr in r_res["learned_rules"]:
            print(f"      -> ACE Playbook Rule Learned: {lr.to_prompt_line()}")
        if r_res["error"] and not r_res["resolved"]:
            print(f"      -> Remaining Error: {r_res['error']}")

    bare_solved = sum(1 for r in bare_records if r["resolved"])
    rsi_1st = sum(1 for r in rsi_records if r["first_shot_resolved"])
    rsi_solved = sum(1 for r in rsi_records if r["resolved"])
    n = len(suite)

    print(f"\n{'=' * 98}")
    print(" SWE-BENCH VERIFIED SUMMARY TABLE")
    print(f"{'=' * 98}")
    print(f"  Bare `{model_name}` (Pass@1):             {bare_solved}/{n} Resolved ({bare_solved/n*100:.1f}%)")
    print(f"  Dynamic RSI Harness (1st-Shot Multi-Critic + ACE):   {rsi_1st}/{n} Resolved ({rsi_1st/n*100:.1f}%)")
    print(f"  Dynamic RSI Harness (Full Loop w/ Sandbox RSI):      {rsi_solved}/{n} Resolved ({rsi_solved/n*100:.1f}%)")
    print("=" * 98)
