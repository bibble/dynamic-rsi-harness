"""
Agentic Context Engineering (ACE) Playbook & Darwin Gödel Machine (DGM) Skill Registry.

1. PlaybookMemory: Manages structured delta rules (Generation-Reflection-Curation)
   without brevity bias or context collapse. Supports hybrid tag + semantic embedding
   retrieval via `nomic-embed-text:latest` when an LLM client is attached.
2. SkillRegistry: Compiles, sandbox-tests, and persists self-authored Python tools
   so bounded LLMs can offload deterministic computation and parsing.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from rsi_harness.llm_client import BaseLLMClient, cosine_similarity
from rsi_harness.schema import PlaybookRule, SynthesizedSkill


class PlaybookMemory:
    """
    Implements Agentic Context Engineering (ACE).
    Instead of rewriting a monolithic system prompt (which loses historical nuance),
    ACE maintains a structured table of fine-grained rules indexed by domain tags
    and optional dense vector embeddings (`nomic-embed-text:latest`).
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None) -> None:
        self._rules: Dict[str, PlaybookRule] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._llm_client = llm_client

    def _compute_embedding(self, text: str) -> Optional[List[float]]:
        if self._llm_client is None or not getattr(self._llm_client.config, "embedding_model", ""):
            return None
        try:
            vecs = self._llm_client.embed([text])
            return vecs[0] if vecs else None
        except Exception:
            return None

    def upsert_rule(self, rule: PlaybookRule) -> PlaybookRule:
        """
        Curator operation: merges a delta rule or inserts a new one without erasing
        unrelated heuristics. Automatically indexes rule embedding if available.
        """
        if rule.rule_id in self._rules:
            existing = self._rules[rule.rule_id]
            existing.directive = rule.directive
            existing.anti_pattern = rule.anti_pattern
            existing.confidence = min(0.99, (existing.confidence + rule.confidence) / 2 + 0.05)
            for tag in rule.domain_tags:
                if tag not in existing.domain_tags:
                    existing.domain_tags.append(tag)
            target = existing
        else:
            self._rules[rule.rule_id] = rule
            target = rule

        vec = self._compute_embedding(target.to_prompt_line())
        if vec:
            self._embeddings[target.rule_id] = vec
        return target

    def retrieve_for_task(
        self,
        domain_tags: List[str],
        query_text: str = "",
        top_k: int = 5,
    ) -> List[PlaybookRule]:
        """
        Retrieves highest-relevance rules matching the task's domain profile, combining
        symbolic tag overlap, empirical confidence, and dense embedding similarity.
        """
        tag_set = set(domain_tags)
        query_vec = self._compute_embedding(query_text) if query_text else None

        scored: List[Tuple[float, PlaybookRule]] = []
        for rule in self._rules.values():
            tag_match = bool(tag_set.intersection(rule.domain_tags) or "universal" in rule.domain_tags)
            sim = 0.0
            if query_vec and rule.rule_id in self._embeddings:
                sim = cosine_similarity(query_vec, self._embeddings[rule.rule_id])
            # Require at least one matching domain tag OR very high semantic similarity (>= 0.78)
            # so repo-specific rules don't pollute unrelated domains!
            if tag_match or sim >= 0.78:
                score = (1.5 if tag_match else 0.0) + rule.confidence + (0.5 * sim)
                scored.append((score, rule))

        scored.sort(key=lambda item: (item[0], item[1].times_succeeded), reverse=True)
        return [r for _, r in scored[:top_k]]

    def record_outcome(self, rule_ids: List[str], success: bool) -> None:
        """Updates empirical fitness of applied rules based on verification outcome."""
        for rid in rule_ids:
            if rid in self._rules:
                rule = self._rules[rid]
                rule.times_applied += 1
                if success:
                    rule.times_succeeded += 1
                    rule.confidence = min(0.99, rule.confidence + 0.04)
                else:
                    rule.confidence = max(0.10, rule.confidence - 0.08)

    @property
    def all_rules(self) -> List[PlaybookRule]:
        return list(self._rules.values())


class SkillRegistry:
    """
    Implements DGM / LATM (LLMs as Tool Makers) dynamic skill compilation & verification.
    New tools proposed by the agent or Meta-RSI loop are compiled and unit-tested in an
    isolated namespace before being admitted to the registry.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None) -> None:
        self._skills: Dict[str, SynthesizedSkill] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._llm_client = llm_client

    def register_and_verify(self, skill: SynthesizedSkill) -> bool:
        """
        Compiles `skill.source_code`, runs `skill.unit_test_code` in an isolated scope,
        and registers the callable only if all assertions pass.
        """
        namespace: Dict[str, Any] = {"__builtins__": __builtins__}
        try:
            exec(skill.source_code, namespace)
            if skill.entrypoint_name not in namespace:
                skill.verified = False
                skill.verification_error = f"Entrypoint '{skill.entrypoint_name}' not found in source_code."
                return False
            # Run self-verification unit tests line-by-line or block to capture exact failing statement
            for line_no, line in enumerate(skill.unit_test_code.splitlines(), start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    exec(line, namespace)
                except Exception as line_exc:
                    detail = str(line_exc)
                    # Pytest-style assertion introspection: evaluate LHS and RHS of `assert A == B`
                    if isinstance(line_exc, AssertionError) and stripped.startswith("assert ") and "==" in stripped:
                        expr_body = stripped[len("assert "):].strip()
                        parts = expr_body.split("==", 1)
                        if len(parts) == 2:
                            try:
                                lhs_val = eval(parts[0].strip(), namespace)
                                rhs_val = eval(parts[1].strip(), namespace)
                                detail = (
                                    f"LHS `{parts[0].strip()}` evaluated to {lhs_val!r}, "
                                    f"but RHS `{parts[1].strip()}` evaluated to {rhs_val!r} "
                                    "(check operator precedence of `%` vs `^` or intermediate modulo)"
                                )
                            except Exception:
                                pass
                    skill.verified = False
                    skill.verification_error = (
                        f"{type(line_exc).__name__} on unit_test line {line_no} (`{stripped}`): {detail}"
                    )
                    return False
            skill.callable_fn = namespace[skill.entrypoint_name]
            skill.verified = True
            skill.verification_error = None
            self._skills[skill.name] = skill
            if self._llm_client and getattr(self._llm_client.config, "embedding_model", ""):
                try:
                    vecs = self._llm_client.embed([f"{skill.name}: {skill.description}"])
                    if vecs:
                        self._embeddings[skill.name] = vecs[0]
                except Exception:
                    pass
            return True
        except Exception as exc:
            skill.verified = False
            skill.verification_error = f"{type(exc).__name__}: {exc}"
            return False

    def get_skill(self, name: str) -> Optional[SynthesizedSkill]:
        return self._skills.get(name)

    def match_skills(self, domain_tags: List[str], query_text: str = "") -> List[SynthesizedSkill]:
        tag_set = set(domain_tags)
        query_vec: Optional[List[float]] = None
        if query_text and self._llm_client and getattr(self._llm_client.config, "embedding_model", ""):
            try:
                vecs = self._llm_client.embed([query_text])
                query_vec = vecs[0] if vecs else None
            except Exception:
                query_vec = None

        matched: List[SynthesizedSkill] = []
        for s in self._skills.values():
            if not s.verified:
                continue
            tag_hit = bool(tag_set.intersection(s.domain_tags))
            sim_hit = False
            if query_vec and s.name in self._embeddings:
                sim_hit = cosine_similarity(query_vec, self._embeddings[s.name]) >= 0.55
            if tag_hit or sim_hit:
                matched.append(s)
        return matched

    def invoke(self, name: str, *args: Any, **kwargs: Any) -> Any:
        skill = self._skills.get(name)
        if not skill or not skill.verified or not skill.callable_fn:
            raise ValueError(f"Skill '{name}' is not registered or verified.")
        skill.usage_count += 1
        return skill.callable_fn(*args, **kwargs)

    @property
    def all_skills(self) -> List[SynthesizedSkill]:
        return list(self._skills.values())
