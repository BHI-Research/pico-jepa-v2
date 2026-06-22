"""Heuristic proposer: random search with anti-OOM rules and a soft memory of
recently-failed configurations.

This is the default proposer (no API needed) and the fallback when the LLM
proposer errors out. It is cheap and robust, but tends to plateau earlier
than a well-prompted LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from autoresearch.search_space import SPACES, estimated_oom


class HeuristicProposer:
    def __init__(self, seed: int = 1337, max_attempts: int = 50):
        self.rng = np.random.default_rng(seed)
        self.max_attempts = max_attempts

    def propose(
        self,
        phase: str,
        base_config: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        space = SPACES.get(phase)
        if not space:
            raise ValueError(f"No search space declared for phase {phase!r}.")

        # Collect recent failure signatures so we don't immediately propose
        # a similar config that just OOM'd.
        recent_oom_keys = self._recent_failure_keys(history, status_set={"oom", "failed_oom"})

        for _ in range(self.max_attempts):
            # Pick K hparams to mutate (K small to avoid jumping too far).
            keys = list(space.keys())
            self.rng.shuffle(keys)
            n_mutate = int(self.rng.integers(1, min(4, len(keys)) + 1))
            chosen = keys[:n_mutate]
            delta = {k: space[k].sample(self.rng) for k in chosen}

            # Overlay on base for OOM check + cross-field validity.
            candidate = {**base_config, **delta}
            if phase == "pretrain":
                # Reject obviously-OOM configurations and recent failures.
                if estimated_oom(candidate):
                    continue
                if self._signature(candidate) in recent_oom_keys:
                    continue
                # Heads must divide embed dim.
                embed = candidate.get("vit_embed_dim")
                heads = candidate.get("vit_num_heads")
                if embed and heads and embed % heads != 0:
                    # Coerce heads to a valid divisor by re-sampling once.
                    candidates = [h for h in space["vit_num_heads"].values if embed % h == 0]
                    if not candidates:
                        continue
                    delta["vit_num_heads"] = int(self.rng.choice(candidates))

            return delta

        # Fallback: return any one mutation.
        key = list(space.keys())[int(self.rng.integers(0, len(space)))]
        return {key: space[key].sample(self.rng)}

    def _recent_failure_keys(self, history: List[Dict[str, Any]], status_set: set) -> set:
        keys = set()
        for h in history[:20]:
            if h.get("status") in status_set:
                # Use config_hash as a coarse signature.
                if h.get("config_hash"):
                    keys.add(h["config_hash"])
        return keys

    def _signature(self, candidate: Dict[str, Any]) -> str:
        items = sorted(
            (k, v) for k, v in candidate.items()
            if k in {"vit_embed_dim", "vit_depth", "batch_size", "frames_per_clip"}
        )
        return repr(items)
