"""
Dedicated, provider-agnostic LLM & Embedding Connector Module for the Dynamic RSI Harness.

Supports:
1. Local/LAN Ollama servers via OpenAI-compatible `/v1` API + native `/api/generate` warmup
   (default: `http://localhost:11434/v1`, model `qwen3-coder-next:latest`,
   embeddings `nomic-embed-text:latest`).
2. Online OpenAI-compatible endpoints (OpenAI, Gemini OpenAI-compat, vLLM, OpenRouter, DeepSeek, etc.).
3. Zero external dependency requirement: uses modern `openai.OpenAI` SDK if installed, and
   transparently falls back to Python standard-library `urllib.request` so it works in any environment.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class LLMConfig:
    """
    Configuration for connecting the RSI Harness to a LAN or Online LLM provider.
    Can be initialized directly, from environment variables, or via preset factories.
    """
    provider: str = "lan_ollama"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen3-coder-next:latest"
    embedding_model: str = "nomic-embed-text:latest"
    timeout_sec: float = 180.0
    temperature: float = 0.2
    max_tokens: int = 2048
    keep_alive: str = "24h"
    extra_headers: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Loads configuration from environment variables with LAN Ollama defaults."""
        return cls(
            provider=os.getenv("RSI_LLM_PROVIDER", "lan_ollama"),
            base_url=os.getenv("RSI_LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("RSI_LLM_API_KEY", "ollama"),
            model=os.getenv("RSI_LLM_MODEL", "qwen3-coder-next:latest"),
            embedding_model=os.getenv("RSI_EMBEDDING_MODEL", "nomic-embed-text:latest"),
            timeout_sec=float(os.getenv("RSI_LLM_TIMEOUT_SEC", "180.0")),
            temperature=float(os.getenv("RSI_LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("RSI_LLM_MAX_TOKENS", "2048")),
            keep_alive=os.getenv("RSI_OLLAMA_KEEP_ALIVE", "24h"),
        )

    @classmethod
    def lan_ollama(
        cls,
        host: Optional[str] = None,
        model: str = "qwen3-coder-next:latest",
        embedding_model: str = "nomic-embed-text:latest",
        keep_alive: str = "24h",
    ) -> "LLMConfig":
        """Preset for the Local/LAN Ollama server."""
        resolved_host = host or os.getenv("RSI_OLLAMA_HOST", "http://localhost:11434")
        base = resolved_host.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return cls(
            provider="lan_ollama",
            base_url=base,
            api_key="ollama",
            model=model,
            embedding_model=embedding_model,
            timeout_sec=180.0,
            keep_alive=keep_alive,
        )

    @classmethod
    def online_openai_compatible(
        cls,
        base_url: str,
        api_key: str,
        model: str,
        embedding_model: str = "",
        provider: str = "online_openai_compat",
        timeout_sec: float = 120.0,
    ) -> "LLMConfig":
        """
        Preset for switching to any online OpenAI-compatible model
        (e.g., OpenAI, Gemini `/v1beta/openai`, OpenRouter, Together, DeepSeek, Anthropic proxy).
        """
        return cls(
            provider=provider,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=model,
            embedding_model=embedding_model,
            timeout_sec=timeout_sec,
        )

    @property
    def native_ollama_root(self) -> str:
        """Returns the root URL without `/v1` for native Ollama API calls (`/api/generate`)."""
        if self.base_url.endswith("/v1"):
            return self.base_url[:-3]
        return self.base_url.rstrip("/")


@dataclass
class LLMResponse:
    """Normalized response returned by any LLMClient implementation."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None


class BaseLLMClient(Protocol):
    """Protocol implemented by all LLM connectors so the RSI harness remains model-agnostic."""
    config: LLMConfig

    def list_models(self) -> List[str]:
        ...

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        ...

    def embed(self, texts: List[str]) -> List[List[float]]:
        ...


class OpenAICompatibleClient:
    """
    Universal OpenAI-compatible client that connects to the Local/LAN Ollama server
    (`http://localhost:11434/v1` or `RSI_OLLAMA_HOST`) as well as any online OpenAI-compatible API.

    Automatically uses the `openai.OpenAI` v1+ SDK if available, or falls back to
    direct REST calls via `urllib.request` so it never breaks on older/minimal environments.
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._sdk_client: Any = None
        try:
            from openai import OpenAI  # type: ignore
            self._sdk_client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout_sec,
                default_headers=self.config.extra_headers or None,
            )
        except Exception:
            self._sdk_client = None

    def _http_request(
        self,
        url: str,
        method: str = "GET",
        payload: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        headers.update(self.config.extra_headers)
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        effective_timeout = timeout or self.config.timeout_sec
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} calling {url}: {err_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error calling {url}: {exc.reason}") from exc

    def warm_up_ollama(self, keep_alive: Optional[str] = None) -> Dict[str, Any]:
        """
        Pins the model (`qwen3-coder-next:latest`, 51 GB) in GPU memory on the LAN Ollama
        host using `/api/generate` with `keep_alive` (default: '24h').
        """
        url = f"{self.config.native_ollama_root}/api/generate"
        payload = {
            "model": self.config.model,
            "keep_alive": keep_alive or self.config.keep_alive,
        }
        return self._http_request(url, method="POST", payload=payload, timeout=self.config.timeout_sec)

    def list_models(self) -> List[str]:
        """Queries `/v1/models` and returns available model IDs."""
        if self._sdk_client is not None:
            try:
                page = self._sdk_client.models.list()
                return [m.id for m in page.data]
            except Exception:
                pass
        url = f"{self.config.base_url}/models"
        data = self._http_request(url, method="GET")
        return [item["id"] for item in data.get("data", []) if "id" in item]

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Executes a chat completion against the configured LAN or online model."""
        temp = self.config.temperature if temperature is None else temperature
        max_tok = self.config.max_tokens if max_tokens is None else max_tokens
        t0 = time.perf_counter()

        # Note: Reasoning models like `deepseek-r1` emit `<think>` before JSON, which breaks
        # Ollama's strict GBNF grammar if `response_format={"type": "json_object"}` is forced.
        model_lower = self.config.model.lower()
        is_reasoning_model = "deepseek-r1" in model_lower or "qwq" in model_lower
        is_qwen35_hybrid = "qwen3.5" in model_lower or "deepseek-v4" in model_lower

        if self.config.provider == "lan_ollama" and is_qwen35_hybrid:
            # Qwen3.5 / DeepSeek-V4-Pro ChatML `<think>\n\n</think>\n\n{prefix}` prefill + Tab-Indentation Adapter:
            # 1. Prevents 2048-token `<think>` runaway OR 1000-token prose preambles.
            # 2. Bypasses Ollama 0.34.2's `qwen35` BPE space-collapsing detokenizer bug (`"    "` -> `" "`)
            #    by enforcing TAB (`\t`) indentation for all Python blocks.
            url = f"{self.config.native_ollama_root}/api/generate"
            last_user = messages[-1].get("content", "") if messages else ""
            wants_python = "```python```" in last_user or "Python" in last_user
            is_reflector = "RULE_WHEN:" in last_user
            tab_instruction = (
                "\n\nCRITICAL PYTHON INDENTATION RULE: Because of the GGUF tokenizer, you MUST indent all Python "
                "code blocks using TAB characters (`\t`, 1 tab per indentation level, `\t\t` for 2 levels, etc.), "
                "NEVER spaces!\nExample:\n```python\ndef _tab_example(x):\n\tif x > 0:\n\t\treturn x\n\treturn 0\n```"
            )
            parts_q: List[str] = []
            for idx_m, m in enumerate(messages):
                role = m.get("role", "user")
                m_content = m.get("content", "")
                if (wants_python or is_reflector) and "bullet" not in last_user.lower():
                    # Convert any 4-space indentation in focal code snippets in the prompt to TABs (`\t`)
                    # so the model naturally mirrors `\t` indentation when patching existing functions!
                    m_content = re.sub(
                        r"(?m)^(    )+",
                        lambda match: "\t" * (len(match.group(0)) // 4),
                        m_content,
                    )
                if idx_m == len(messages) - 1 and role == "user" and (wants_python or is_reflector) and "bullet" not in last_user.lower():
                    m_content = m_content + tab_instruction
                parts_q.append(f"<|im_start|>{role}\n{m_content}<|im_end|>")

            if json_mode:
                prefix = "{\n"
                stop_seqs = ["\n}\n", "}\n\n", "<|im_end|>"]
            elif is_reflector:
                prefix = "RULE_WHEN: "
                stop_seqs = ["<|im_end|>"]
            elif wants_python and "bullet" not in last_user.lower():
                prefix = "```python\n"
                stop_seqs = ["\n```", "<|im_end|>"]
            else:
                prefix = ""
                stop_seqs = ["<|im_end|>"]

            raw_prompt = "\n".join(parts_q) + f"\n<|im_start|>assistant\n<think>\n\n</think>\n\n{prefix}"
            options_q: Dict[str, Any] = {
                "temperature": temp,
                "num_predict": max_tok,
                "num_ctx": 4096,
                "stop": stop_seqs,
            }
            payload_q: Dict[str, Any] = {
                "model": self.config.model,
                "prompt": raw_prompt,
                "raw": True,
                "stream": False,
                "keep_alive": self.config.keep_alive,
                "options": options_q,
            }
            raw_c = self._http_request(url, method="POST", payload=payload_q, timeout=self.config.timeout_sec)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            resp_body = raw_c.get("response", "") or ""
            if prefix.startswith("```python\n") and not resp_body.rstrip().endswith("```"):
                content = f"{prefix}{resp_body}\n```"
            elif prefix == "{\n" and not resp_body.rstrip().endswith("}"):
                content = "{\n" + resp_body + "\n}"
            else:
                content = prefix + resp_body
            p_tok = int(raw_c.get("prompt_eval_count", 0))
            c_tok = int(raw_c.get("eval_count", 0))
            return LLMResponse(
                content=content,
                model=raw_c.get("model", self.config.model),
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=p_tok + c_tok,
                latency_ms=latency_ms,
                raw_response=raw_c,
            )

        # When calling a LAN Ollama reasoning model while also calling `nomic-embed-text:latest`
        # on the same host, use `/api/chat` with `keep_alive="24h"` and bounded `num_ctx=4096`
        # so Ollama keeps BOTH models in VRAM simultaneously without 60s model-swap thrashing!
        if self.config.provider == "lan_ollama" and is_reasoning_model:
            # Official DeepSeek-R1 empty `<think>\n</think>\n` prefill via `/api/generate` (`raw: True`):
            # Prevents 2000-token runaway `<think>` monologues from exhausting `num_predict` and returning
            # empty `message.content`, while externalizing reasoning into our Multi-Critic & ACE Playbook!
            url = f"{self.config.native_ollama_root}/api/generate"
            parts: List[str] = []
            for m in messages:
                role = m.get("role", "user")
                if role == "system":
                    parts.append(m.get("content", ""))
                elif role == "user":
                    parts.append(f"<|User|>{m.get('content', '')}")
                elif role == "assistant":
                    parts.append(f"<|Assistant|><think>\n</think>\n{m.get('content', '')}")
            last_user = messages[-1].get("content", "") if messages else ""
            wants_python = "```python```" in last_user or "Python" in last_user
            is_reflector = "RULE_WHEN:" in last_user
            if json_mode:
                prefix = "{\n"
                stop_seqs = ["\n}\n", "}\n\n"]
            elif is_reflector:
                prefix = "RULE_WHEN: "
                stop_seqs = []
            elif wants_python and "bullet" not in last_user.lower():
                prefix = "```python\n"
                stop_seqs = ["\n```"]
            else:
                prefix = ""
                stop_seqs = []

            raw_prompt = "\n".join(parts) + f"<|Assistant|><think>\n</think>\n{prefix}"
            options_dict: Dict[str, Any] = {
                "temperature": temp,
                "num_predict": max_tok,
                "num_ctx": 4096,
            }
            if stop_seqs:
                options_dict["stop"] = stop_seqs

            payload_ollama: Dict[str, Any] = {
                "model": self.config.model,
                "prompt": raw_prompt,
                "raw": True,
                "stream": False,
                "keep_alive": self.config.keep_alive,
                "options": options_dict,
            }
            raw_o = self._http_request(url, method="POST", payload=payload_ollama, timeout=self.config.timeout_sec)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            resp_body = raw_o.get("response", "") or ""
            if prefix == "```python\n" and not resp_body.rstrip().endswith("```"):
                content = f"```python\n{resp_body}\n```"
            elif prefix == "{\n" and not resp_body.rstrip().endswith("}"):
                content = "{\n" + resp_body + "\n}"
            else:
                content = prefix + resp_body
            p_tok = int(raw_o.get("prompt_eval_count", 0))
            c_tok = int(raw_o.get("eval_count", 0))
            return LLMResponse(
                content=content,
                model=raw_o.get("model", self.config.model),
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=p_tok + c_tok,
                latency_ms=latency_ms,
                raw_response=raw_o,
            )

        if self._sdk_client is not None:
            kwargs: Dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": max_tok,
            }
            if json_mode and not is_reasoning_model:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self._sdk_client.chat.completions.create(**kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            content = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            p_tok = getattr(usage, "prompt_tokens", 0) or 0
            c_tok = getattr(usage, "completion_tokens", 0) or 0
            tot_tok = getattr(usage, "total_tokens", p_tok + c_tok) or (p_tok + c_tok)
            return LLMResponse(
                content=content,
                model=getattr(resp, "model", self.config.model),
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=tot_tok,
                latency_ms=latency_ms,
            )

        # Fallback HTTP client (`/v1/chat/completions`)
        url = f"{self.config.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok,
        }
        if json_mode and not is_reasoning_model:
            payload["response_format"] = {"type": "json_object"}

        raw = self._http_request(url, method="POST", payload=payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        choices = raw.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = raw.get("usage", {})
        p_tok = int(usage.get("prompt_tokens", 0))
        c_tok = int(usage.get("completion_tokens", 0))
        tot_tok = int(usage.get("total_tokens", p_tok + c_tok))
        return LLMResponse(
            content=content,
            model=raw.get("model", self.config.model),
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=tot_tok,
            latency_ms=latency_ms,
            raw_response=raw,
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings using `self.config.embedding_model`
        (e.g., `nomic-embed-text:latest` on the LAN Ollama server) for semantic
        ACE Playbook rule & DGM Skill retrieval.
        """
        if not self.config.embedding_model:
            raise ValueError("No embedding_model configured on LLMConfig.")

        if self._sdk_client is not None:
            resp = self._sdk_client.embeddings.create(
                model=self.config.embedding_model,
                input=texts,
            )
            return [item.embedding for item in resp.data]

        url = f"{self.config.base_url}/embeddings"
        payload = {
            "model": self.config.embedding_model,
            "input": texts,
        }
        raw = self._http_request(url, method="POST", payload=payload)
        return [item["embedding"] for item in raw.get("data", [])]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two embedding vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def strip_thinking_tags(text: str) -> str:
    """Strips `<think>...</think>` blocks emitted by reasoning models like DeepSeek-R1."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # If an unclosed <think> tag exists (e.g. token limit hit mid-thought), strip up to the last code fence or brace
    if "<think>" in cleaned.lower():
        idx = cleaned.lower().find("<think>")
        after = cleaned[idx:]
        fence_idx = after.find("```")
        if fence_idx != -1:
            cleaned = cleaned[:idx] + after[fence_idx:]
    return cleaned.strip()


def sanitize_tab_indented_python(code: str) -> str:
    """
    If Python code uses TAB (`\t`) indentation (e.g. from the `qwen35` GGUF Tab-Indentation Adapter),
    normalizes leading whitespace on each line by converting each leading `\t` to 4 spaces, stripping
    stray single spaces from Ollama 0.34.2's `qwen35` BPE detokenizer, and enforcing Python's colon-block
    indentation invariant (`expected an indented block` when a line after `:` had spaces collapsed).
    """
    lines = code.splitlines()
    has_tab_indent = any(line.startswith("\t") or line.startswith(" \t") for line in lines)
    if not has_tab_indent:
        return code
    cleaned_lines: List[str] = []
    prev_indent = 0
    prev_ended_with_colon = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        m = re.match(r"^([ \t]+)(.*)$", line)
        if m:
            ws, rest = m.group(1), m.group(2)
            tab_count = ws.count("\t")
            if tab_count > 0:
                indent_spaces = tab_count * 4
            else:
                if rest.startswith(("def ", "class ", "import ", "from ", "@")):
                    indent_spaces = 0
                elif prev_ended_with_colon:
                    indent_spaces = prev_indent + 4
                else:
                    indent_spaces = max(4, prev_indent)
        else:
            rest = stripped
            indent_spaces = (prev_indent + 4) if prev_ended_with_colon else 0

        if prev_ended_with_colon and not rest.startswith("#") and indent_spaces <= prev_indent:
            indent_spaces = prev_indent + 4

        cleaned_lines.append((" " * indent_spaces) + rest)
        if not rest.startswith("#"):
            prev_indent = indent_spaces
            code_part = rest.split("#", 1)[0].rstrip()
            prev_ended_with_colon = code_part.endswith(":")
    return "\n".join(cleaned_lines)


def extract_code_block(text: str, language: str = "python") -> str:
    """Extracts the first fenced code block from an LLM response (after stripping `<think>` tags)."""
    cleaned = strip_thinking_tags(text)
    pattern = rf"```(?:{language})?\s*\n(.*?)```"
    matches = re.findall(pattern, cleaned, re.DOTALL | re.IGNORECASE)
    if matches:
        raw_code = matches[-1].strip()
    else:
        # Handle unclosed code fence if truncated at max_tokens
        unclosed = re.search(rf"```(?:{language})?\s*\n(.*)$", cleaned, re.DOTALL | re.IGNORECASE)
        raw_code = unclosed.group(1).strip() if unclosed else cleaned.strip()
    if language.lower() == "python":
        return sanitize_tab_indented_python(raw_code)
    return raw_code


def extract_json_object(text: str) -> Dict[str, Any]:
    """Robustly extracts a JSON object from an LLM response (handling `<think>` tags, markdown fences, or preambles)."""
    cleaned = strip_thinking_tags(text)
    fenced = extract_code_block(cleaned, language="json")
    try:
        return json.loads(fenced)
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError(f"Could not parse JSON object from LLM response: {cleaned[:200]}")
