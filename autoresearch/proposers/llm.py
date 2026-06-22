"""LLM-driven proposer using the Anthropic SDK.

Design:
- Prompt is split into a stable prefix (system + project description + search
  space + top-K history snapshot) and a small variable suffix (current phase,
  baseline, request). The stable prefix uses Anthropic prompt caching (5-10x
  cost reduction on cache hits).
- A local diskcache also stores responses keyed by sha256 of (model, system,
  history, phase, request). Hits avoid hitting the API at all when the loop
  is resumed and the inputs are bit-identical.
- If the SDK is missing, the API key is unset, or any HTTP call fails, the
  proposer falls back to ``HeuristicProposer`` (if configured) or raises.
"""

from __future__ import annotations

import hashlib
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

import yaml

from autoresearch.proposers.heuristic import HeuristicProposer
from autoresearch.search_space import SPACES, validate_config


CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_cache")
DEFAULT_MODEL = "claude-sonnet-4-6"


SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an ML researcher tuning the pico-JEPA self-supervised video model.

    HYPOTHESIS UNDER TEST:
      An ensemble of small pico-JEPA classifiers trained on data partitions
      can outperform a single larger model trained on all the data.

    HARDWARE: Intel i7, 65GB RAM, GTX 1060 6GB. Memory is the binding constraint.

    YOUR TASK:
      Given the current best config and recent experiment history, propose
      ONE config delta for a specific phase. Output ONLY a JSON object with
      this exact shape:
        {
          "delta": { "<key>": <value>, ... },
          "reasoning": "<one or two sentences justifying the choice based on history>"
        }
      Do not output anything outside this JSON object.

    RULES:
      - Stay strictly within the declared search space.
      - Avoid configurations that recently OOM'd or failed.
      - Prefer mutations that are likely to help based on the history.
      - For pretrain: vit_embed_dim must be divisible by vit_num_heads.
      - Mutate 1-3 keys per delta unless a wider exploratory step is justified.
      - Keep "reasoning" under 200 characters; cite specific history ids/scores when relevant.
    """
)


def _format_search_space() -> str:
    blocks = []
    for phase, space in SPACES.items():
        lines = [f"## phase: {phase}"]
        for name, hp in space.items():
            if hp.values is not None:
                lines.append(f"  {name}: one of {hp.values}")
            else:
                kind = "log-uniform" if hp.log else "uniform"
                lines.append(f"  {name}: {kind} in [{hp.low}, {hp.high}]")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_history(history: List[Dict[str, Any]], k: int = 20) -> str:
    if not history:
        return "(no history yet)"
    rows = []
    for h in history[:k]:
        score = h.get("score")
        rows.append(
            f"- id={h.get('id')} phase={h.get('phase')} status={h.get('status')} "
            f"score={score} wallclock_s={h.get('wallclock_s')}"
        )
    return "\n".join(rows)


def _hash_request(model: str, system: str, history_blob: str, phase: str, request: str) -> str:
    h = hashlib.sha256()
    for chunk in (model, system, history_blob, phase, request):
        h.update(chunk.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


class LLMProposer:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 512,
        fallback: Optional[HeuristicProposer] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.fallback = fallback
        self.last_reasoning: Optional[str] = None
        self._cache = self._open_cache()
        self._client = self._build_client(api_key)

    @staticmethod
    def _open_cache():
        try:
            import diskcache  # type: ignore
            os.makedirs(CACHE_DIR, exist_ok=True)
            return diskcache.Cache(CACHE_DIR)
        except Exception:
            # diskcache is optional; fall back to a plain dict (non-persistent).
            return {}

    @staticmethod
    def _build_client(api_key: Optional[str]):
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError("anthropic SDK not installed; run `pip install anthropic`.") from e
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set; cannot use LLMProposer.")
        return Anthropic(api_key=key)

    def propose(
        self,
        phase: str,
        base_config: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            return self._propose_via_api(phase, base_config, history)
        except Exception as e:
            print(f"[LLMProposer] error: {e!r}; falling back to heuristic.")
            if self.fallback is None:
                self.fallback = HeuristicProposer()
            return self.fallback.propose(phase, base_config, history)

    def _propose_via_api(
        self,
        phase: str,
        base_config: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        space_blob = _format_search_space()
        history_blob = _format_history(history)
        request_user = textwrap.dedent(
            f"""\
            Current phase: {phase}
            Current best config (relevant subset):
            ```yaml
            {yaml.safe_dump({k: base_config.get(k) for k in SPACES[phase] if k in base_config}, sort_keys=True)}
            ```

            Recent experiments (newest first):
            {history_blob}

            Output JSON only, in the exact shape declared in the system prompt. Example:
            {{"delta": {{"learning_rate": 0.0003, "mask_ratio": 0.6}}, "reasoning": "history ids 11-13 with lr=2e-4 plateaued at probe=0.12; lowering lr to escape local minimum."}}
            """
        )

        prompt_hash = _hash_request(self.model, SYSTEM_PROMPT + space_blob, history_blob, phase, request_user)
        cached = self._cache_get(prompt_hash)
        if cached is not None:
            delta, reasoning = self._split_delta_reasoning(cached)
            self.last_reasoning = reasoning
            return self._coerce_delta(phase, delta)

        # Stable prefix (cacheable) + small variable suffix.
        system = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT + "\n\nSEARCH SPACE:\n" + space_blob,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        messages = [{"role": "user", "content": request_user}]
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        raw = self._extract_json(text)
        delta, reasoning = self._split_delta_reasoning(raw)
        self.last_reasoning = reasoning
        # Persist the full {delta, reasoning} envelope so cache hits still surface the reasoning.
        self._cache_put(prompt_hash, {"delta": delta, "reasoning": reasoning})
        return self._coerce_delta(phase, delta)

    @staticmethod
    def _split_delta_reasoning(payload: Any) -> tuple:
        """Accept either the new envelope {'delta': {...}, 'reasoning': '...'} or
        a bare delta dict (legacy cache format). Returns (delta_dict, reasoning_or_None).
        """
        if isinstance(payload, dict) and "delta" in payload and isinstance(payload["delta"], dict):
            return payload["delta"], payload.get("reasoning")
        if isinstance(payload, dict):
            return payload, None
        return {}, None

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip("`").strip()
        try:
            return json.loads(text)
        except Exception:
            # Try to grab the first {...} block.
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start : end + 1])
            raise

    @staticmethod
    def _coerce_delta(phase: str, delta: Dict[str, Any]) -> Dict[str, Any]:
        space = SPACES.get(phase, {})
        clean: Dict[str, Any] = {}
        for k, v in delta.items():
            if k not in space:
                continue
            hp = space[k]
            if hp.values is not None and v not in hp.values:
                continue
            if hp.values is None and not hp.in_range(v):
                continue
            clean[k] = v
        return clean

    def _cache_get(self, key: str):
        try:
            return self._cache[key]
        except Exception:
            return None

    def _cache_put(self, key: str, value: Dict[str, Any]) -> None:
        try:
            self._cache[key] = value
        except Exception:
            pass
