"""
Live End-to-End RSI Harness Test against a Local/LAN Ollama Server (`RSI_OLLAMA_HOST`).
Uses:
  - Model: `qwen3-coder-next:latest`
  - Embeddings: `nomic-embed-text:latest`

Demonstrates:
1. Connectivity & 24h GPU warmup via `OpenAICompatibleClient`.
2. Semantic embedding retrieval via `nomic-embed-text:latest`.
3. Live Recursive Self-Improvement (RSI) loop where `qwen3-coder-next:latest`:
   - Synthesizes a Python skill (`SynthesizedSkill`) + unit tests and an ACE `PlaybookRule`.
   - When the LLM's mental-math assertion in its own unit test fails the sandbox gate
     (proving why LLMs cannot do raw token arithmetic!), the RSI loop feeds the exact sandbox
     `AssertionError` trace back to `qwen3-coder-next:latest` to self-repair its test suite
     using structural/symbolic invariants, passes the gate, and solves the task!
"""

from __future__ import annotations
import sys
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


def run_live_lan_test() -> None:
    print("=" * 88)
    print(" LIVE LAN LLM SERVER + DYNAMIC RSI HARNESS INTEGRATION TEST")
    print("=" * 88)

    # 1. Initialize dedicated LLM connector for Local/LAN Ollama (keeping flexibility for online models)
    config = LLMConfig.lan_ollama(
        model="qwen3-coder-next:latest",
        embedding_model="nomic-embed-text:latest",
        keep_alive="24h",
    )
    client = OpenAICompatibleClient(config)

    # 2. Verify connectivity & list available models
    models = client.list_models()
    print(f"[Step 1 | Connectivity] Connected to {config.base_url}")
    print(f"                        Available models: {models}")

    # 3. Pin `qwen3-coder-next:latest` in GPU memory for 24h
    warmup_status = client.warm_up_ollama(keep_alive="24h")
    print(
        f"[Step 2 | GPU Warmup]   Pinned '{config.model}' with keep_alive='24h' -> "
        f"done={warmup_status.get('done')}, reason={warmup_status.get('done_reason')}"
    )

    # 4. Initialize ACE Playbook + DGM SkillRegistry backed by `nomic-embed-text:latest`
    playbook = PlaybookMemory(llm_client=client)
    skills = SkillRegistry(llm_client=client)

    # Test semantic embeddings via `nomic-embed-text:latest`
    seed_rule = PlaybookRule(
        rule_id="RULE-SEMANTIC-01",
        domain_tags=["symbolic", "checksum"],
        when_condition="Computing custom bitwise rolling hashes or modular cascades",
        directive="Synthesize a deterministic Python function and validate with symbolic expressions rather than mental arithmetic constants",
        anti_pattern="Do not estimate bitwise XOR or modular arithmetic in raw tokens—even inside unit test expected values",
        confidence=0.93,
    )
    playbook.upsert_rule(seed_rule)
    retrieved = playbook.retrieve_for_task(
        domain_tags=[],
        query_text="How should I calculate an exact bitwise XOR modular hash over integers?",
        top_k=1,
    )
    print(
        f"[Step 3 | Embeddings]   Semantic retrieval via '{config.embedding_model}' matched: "
        f"{retrieved[0].rule_id} (conf={retrieved[0].confidence:.2f})"
    )

    # 5. Live DGM + ACE Self-Improvement Cycle powered by `qwen3-coder-next:latest`
    print(f"\n{'-' * 88}")
    print(f"[Step 4 | Live RSI Tool Synthesis & Self-Repair via {config.model}]")
    print(f"{'-' * 88}")

    base_prompt = (
        "You are the Meta-RSI Engine (Darwin Gödel Machine + Agentic Context Engineering).\n"
        "A bounded worker agent failed to compute the exact function `rolling_xor_checksum(nums, mod)` "
        "in raw tokens.\n"
        "Specification for `rolling_xor_checksum(nums: list[int], mod: int) -> int`:\n"
        "  - Start with `acc = 42`.\n"
        "  - For each integer `x` in `nums` (in order): `acc = ((acc * 37 + x) ^ (x >> 1)) % mod`.\n"
        "  - Return `acc`.\n\n"
        "Return ONLY a valid JSON object with these exact keys:\n"
        "{\n"
        '  "skill_name": "rolling_xor_checksum",\n'
        '  "description": "One-sentence description",\n'
        '  "source_code": "Full Python def rolling_xor_checksum(nums, mod): ...",\n'
        '  "unit_test_code": "Python assert statements testing rolling_xor_checksum",\n'
        '  "rule_when": "When task requires rolling bitwise modular checksums",\n'
        '  "rule_directive": "What the harness should do",\n'
        '  "rule_anti_pattern": "What to avoid"\n'
        "}"
    )

    messages = [{"role": "user", "content": base_prompt}]
    verified = False
    synthesized_skill = None
    parsed = {}

    for attempt in range(1, 4):
        resp = client.chat(
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            json_mode=True,
        )
        print(
            f"[Attempt {attempt} | LLM Call]  Model={resp.model} | "
            f"Tokens={resp.total_tokens} | Latency={resp.latency_ms:.0f}ms"
        )
        parsed = extract_json_object(resp.content)
        synthesized_skill = SynthesizedSkill(
            name=parsed["skill_name"],
            description=parsed["description"],
            domain_tags=["math", "symbolic", "checksum"],
            source_code=extract_code_block(parsed["source_code"]),
            entrypoint_name="rolling_xor_checksum",
            unit_test_code=extract_code_block(parsed["unit_test_code"]),
        )
        verified = skills.register_and_verify(synthesized_skill)
        if verified:
            print(f"[Attempt {attempt} | Sandbox]   Skill '{synthesized_skill.name}' PASSED unit tests!")
            break
        else:
            print(
                f"[Attempt {attempt} | Sandbox]   Skill '{synthesized_skill.name}' FAILED sandbox gate -> "
                f"{synthesized_skill.verification_error}"
            )
            print(f"                        Failed unit_test_code was:\n{synthesized_skill.unit_test_code}")
            # Feed the sandbox execution error + retrieved ACE rule back to the LLM for self-repair
            repair_feedback = (
                f"Sandbox verification failed with exact diagnostic:\n"
                f"  {synthesized_skill.verification_error}\n"
                f"Your `source_code` was:\n{synthesized_skill.source_code}\n\n"
                f"Apply ACE Playbook Rule: {retrieved[0].to_prompt_line()}\n"
                "Fix the failing assertion in `unit_test_code` using the exact LHS value evaluated by the "
                "Python sandbox (or remove/replace mental-math constant guesses with exact values) "
                "and return the corrected JSON object."
            )
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": repair_feedback})

    assert verified and synthesized_skill is not None, "Skill failed to verify after RSI self-repair!"

    # Invoke the live LLM-authored skill on a benchmark vector
    test_nums = [142, 891, 3045, 77, 6192]
    test_mod = 10007
    output_val = skills.invoke("rolling_xor_checksum", test_nums, test_mod)

    # Ground-truth check
    expected = 42
    for x in test_nums:
        expected = ((expected * 37 + x) ^ (x >> 1)) % test_mod
    assert output_val == expected, f"Expected {expected}, got {output_val}"
    print(f"[Live Skill Execution]  rolling_xor_checksum({test_nums}, {test_mod}) -> {output_val} (Expected: {expected}) PASS!")

    # Upsert the LLM-authored ACE Playbook rule
    new_rule = PlaybookRule(
        rule_id="RULE-LIVE-LLM-02",
        domain_tags=["math", "symbolic", "checksum"],
        when_condition=parsed["rule_when"],
        directive=parsed["rule_directive"],
        anti_pattern=parsed["rule_anti_pattern"],
        confidence=0.95,
    )
    playbook.upsert_rule(new_rule)
    print(f"[ACE Playbook Delta]    {new_rule.to_prompt_line()}")
    print("=" * 88)


if __name__ == "__main__":
    run_live_lan_test()
