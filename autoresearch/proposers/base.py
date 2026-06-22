"""Proposer interface."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol


class Proposer(Protocol):
    """Generates a config delta for the next experiment in a given phase."""

    def propose(
        self,
        phase: str,
        base_config: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return a dict of (key, value) pairs to overlay on base_config.

        Args:
            phase: 'pretrain' | 'classify' | 'ensemble'.
            base_config: the current best-known config for the phase.
            history: recent experiment dicts from the ledger (newest first).

        Returns:
            A delta dict. The runner merges it onto base_config.
        """
        ...
