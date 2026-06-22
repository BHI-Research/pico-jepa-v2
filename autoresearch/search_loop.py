"""Main search loop CLI.

Usage:
    python -m autoresearch.search_loop \\
        --base-config configs/config.yaml \\
        --max-wallclock 6h --max-iters 100 \\
        --proposer heuristic

The loop:
1. Loads base config and prepares the deterministic triple split.
2. Picks a phase (warmup-then-bandit policy).
3. Asks the proposer for a config delta for that phase.
4. Runs it via Runner.
5. If it improves the ratchet, promotes artifacts and tags git.
6. Generates a markdown report for the iteration.
7. Repeats until the budget is exhausted or the search plateaus.

Resume is automatic: any 'running' rows in the ledger from a previous process
are marked 'interrupted' and the loop continues from the last best.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from autoresearch.budget import Budget, parse_wallclock
from autoresearch.ledger import Ledger
from autoresearch.prepare import Splits, file_hash as prepare_file_hash, make_splits
from autoresearch.proposers.heuristic import HeuristicProposer
from autoresearch.ratchet import is_improvement, promote_artifacts
from autoresearch.runner import Runner, RunnerConfig, encoder_arch_from_best, merge


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PHASES = ("pretrain", "classify", "ensemble")
WARMUP_PLAN: Tuple[str, ...] = (
    "pretrain", "classify", "pretrain", "classify",
    "ensemble", "pretrain", "classify", "ensemble",
)


@dataclass
class PhaseStats:
    n_attempts: int = 0
    n_improvements: int = 0
    total_wallclock_s: float = 0.0

    def expected_improvement(self) -> float:
        if self.n_attempts == 0:
            return 1.0  # exploration: assume promising until evidence
        return self.n_improvements / self.n_attempts

    def expected_wallclock_s(self) -> float:
        if self.n_attempts == 0:
            return 60.0  # arbitrary low default
        return self.total_wallclock_s / self.n_attempts


def pick_phase_warmup(
    iter_idx: int,
    has_pretrained_encoder: bool = False,
    skip_pretrain: bool = False,
) -> str:
    """Return the phase for warmup iteration ``iter_idx``.

    When a pretrain checkpoint is already promoted (manual pretrain done
    outside the loop, or recovered from a previous run), the warmup skips
    its pretrain slots. When ``skip_pretrain`` is True the substitution is
    unconditional — useful to lock pretrain out for an entire run.
    """
    phase = WARMUP_PLAN[iter_idx % len(WARMUP_PLAN)]
    if phase == "pretrain" and (skip_pretrain or has_pretrained_encoder):
        # Replace pretrain slots with classify.
        return "classify"
    return phase


def pick_phase_ucb(
    stats: Dict[str, PhaseStats], total_iters: int,
    force_classify: bool = False, skip_pretrain: bool = False,
) -> str:
    """Budgeted UCB1 with cost penalty.

    Score = (mean_improvement_rate / mean_wallclock_s) + sqrt(2 ln N / n_p).
    The classify phase is forced once after a pretrain ratchet to refresh
    downstream submodels. When ``skip_pretrain`` is True the pretrain phase
    is excluded from the candidate set entirely.
    """
    if force_classify:
        return "classify"
    candidate_phases = tuple(p for p in PHASES if not (skip_pretrain and p == "pretrain"))
    best_phase = candidate_phases[0]
    best_score = -math.inf
    log_total = math.log(max(total_iters, 1))
    for phase in candidate_phases:
        s = stats[phase]
        n_p = max(s.n_attempts, 1)
        exploit = s.expected_improvement() / max(s.expected_wallclock_s(), 1.0)
        explore = math.sqrt(2 * log_total / n_p)
        score = exploit + explore
        if score > best_score:
            best_score = score
            best_phase = phase
    return best_phase


def load_base_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def best_known_config(ledger: Ledger, phase: str, base: Dict[str, Any]) -> Dict[str, Any]:
    best = ledger.best_for_phase(phase)
    if best is None:
        return dict(base)
    cfg = yaml.safe_load(best["config_yaml"]) or {}
    return merge(base, cfg)


def sanitize_classify_config_for_promotion(
    submodel_config_path: str,
    base_config: Dict[str, Any],
    work_dir: str,
    output_name: str = "promoted_classify_config.yaml",
) -> str:
    """Strip per-submodel CSV paths from a classify config before promoting it.

    The classify adapter overwrites ``classify_csv_path`` in each submodel's
    YAML to point at that submodel's partitioned CSV (~237 videos for one
    submodel out of a 1467-video set). If we promote that YAML straight to
    ``configs/best/classify.yaml``, every downstream consumer
    (``_maybe_run_phase4``, ``run_phase4`` CLI, future classify iterations
    that inherit from best) ends up training on the tiny partition instead
    of the full classify set. We restore the path to whatever the base
    config declared and write a sanitized copy alongside the original.

    The ``output_name`` argument lets the caller invoke this function once
    per submodel without overwriting itself (used by the heterogeneous
    ensemble flow: one sanitized config per submodel preserves head_type,
    seed, lr, etc., which are needed at inference time).
    """
    if not os.path.exists(submodel_config_path):
        return submodel_config_path
    with open(submodel_config_path) as f:
        cfg = yaml.safe_load(f) or {}
    original_csv = base_config.get("classify_csv_path")
    original_dir = base_config.get("classify_video_dir", base_config.get("video_dir"))
    if original_csv:
        cfg["classify_csv_path"] = original_csv
    if original_dir:
        cfg["classify_video_dir"] = original_dir
    sanitized = os.path.join(work_dir, output_name)
    with open(sanitized, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    return sanitized


def load_per_submodel_configs(
    n_models: int, fallback_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Read ``configs/best/classify_{i}.yaml`` for i in 1..n_models.

    Used by the ensemble and Phase 4 loaders to reconstruct heterogeneous
    submodels (each with its own head_type/freeze/seed). When a per-submodel
    config file is missing (e.g., an older classify experiment that ran
    before the heterogeneous-promotion change), the loader falls back to
    ``configs/best/classify.yaml`` and then to ``fallback_config``.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    best_dir = os.path.join(project_root, "configs", "best")
    global_path = os.path.join(best_dir, "classify.yaml")
    global_cfg = {}
    if os.path.exists(global_path):
        with open(global_path) as f:
            global_cfg = yaml.safe_load(f) or {}

    configs: List[Dict[str, Any]] = []
    for i in range(1, n_models + 1):
        per_path = os.path.join(best_dir, f"classify_{i}.yaml")
        if os.path.exists(per_path):
            with open(per_path) as f:
                configs.append(yaml.safe_load(f) or {})
        elif global_cfg:
            configs.append(dict(global_cfg))
        else:
            configs.append(dict(fallback_config))
    return configs


def best_pretrain_encoder(ledger: Ledger) -> Optional[str]:
    """Return the path to the promoted pretrain encoder, or None.

    Sources, in priority order:
      1. ``models_best/pretrain.pth`` on disk — covers manual promotion
         outside the search loop (user runs ``app/train.py`` and ``cp``s
         the encoder).
      2. Ratchet table — covers in-loop promotions.

    The disk check comes first so a manually-promoted encoder works even
    when the ledger has no ratchet row yet.
    """
    candidate = os.path.join(PROJECT_ROOT, "models_best", "pretrain.pth")
    if os.path.exists(candidate):
        return candidate
    rec = ledger.get_ratchet("pretrain")
    if rec is None:
        return None
    return candidate if os.path.exists(candidate) else None


def append_proposer_log_jsonl(
    work_dir: str, iter_id: int, phase: str, proposer_name: str,
    delta: Dict[str, Any], reasoning: Optional[str],
) -> None:
    """Append the proposer decision to a JSONL audit log under artifacts/reports."""
    reports_dir = os.path.join(work_dir, "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, "proposer_log.jsonl")
    entry = {
        "ts": time.time(),
        "iter_id": iter_id,
        "phase": phase,
        "proposer": proposer_name,
        "delta": delta,
        "reasoning": reasoning,
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


def write_iteration_report(
    work_dir: str, iter_id: int, phase: str, result: Dict[str, Any], improved: bool,
    proposer_info: Optional[Dict[str, Any]] = None,
) -> None:
    reports_dir = os.path.join(work_dir, "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, f"it_{iter_id:04d}.md")
    lines = [
        f"# Iteration {iter_id} — phase: {phase}",
        f"- status: **{result.get('status')}**",
        f"- score: {result.get('score')}",
        f"- improved: **{improved}**",
        f"- wallclock_s: {result.get('wallclock_s')}",
        f"- experiment_id: {result.get('experiment_id')}",
    ]
    if proposer_info:
        lines.extend([
            "",
            "## Proposer decision",
            f"- proposer: **{proposer_info.get('proposer')}**",
            "- delta:",
            "```json",
            json.dumps(proposer_info.get("delta", {}), indent=2, sort_keys=True, default=str),
            "```",
        ])
        reasoning = proposer_info.get("reasoning")
        if reasoning:
            lines.extend([
                "- reasoning:",
                f"  > {reasoning}",
            ])
    lines.extend([
        "",
        "## Metrics",
        "```json",
        json.dumps(result.get("metrics", {}), indent=2, default=str),
        "```",
    ])
    with open(path, "w") as f:
        f.write("\n".join(lines))


def make_proposer(name: str, fallback_name: Optional[str] = None):
    if name == "heuristic":
        return HeuristicProposer()
    if name == "llm":
        try:
            from autoresearch.proposers.llm import LLMProposer
            primary = LLMProposer()
            if fallback_name == "heuristic":
                primary.fallback = HeuristicProposer()
            return primary
        except Exception as e:
            if fallback_name == "heuristic":
                print(f"[autoresearch] LLM proposer unavailable ({e}); falling back to heuristic.")
                return HeuristicProposer()
            raise
    raise ValueError(f"Unknown proposer {name!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Autoresearch search loop for pico-JEPA.")
    parser.add_argument("--base-config", default=os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    parser.add_argument("--max-wallclock", default="6h", help="e.g. 6h, 30m, 90s. Default 6h.")
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--proposer", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--fallback", choices=["heuristic", "none"], default="heuristic")
    parser.add_argument("--work-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"))
    parser.add_argument("--ledger-path", default=None)
    parser.add_argument("--pretrain-timeout", default="90m")
    parser.add_argument("--classify-timeout-per-submodel", default="20m")
    parser.add_argument("--probe-videos", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--skip-pretrain", action="store_true",
        help="Lock pretrain out of both warmup and bandit; only classify/ensemble "
             "are proposed. Requires a promoted models_best/pretrain.pth — the loop "
             "aborts at startup if none exists.",
    )
    args = parser.parse_args(argv)

    base_config = load_base_config(args.base_config)
    classify_csv_path = base_config.get(
        "classify_csv_path",
        os.path.join(base_config.get("classify_video_dir", base_config.get("video_dir", ".")),
                     base_config.get("csv_file_labeled", ""))
    )
    splits = make_splits(classify_csv_path)

    ledger = Ledger(args.ledger_path) if args.ledger_path else Ledger()
    interrupted = ledger.mark_orphan_running_as_interrupted()
    if interrupted:
        print(f"[autoresearch] Marked {interrupted} stale 'running' rows as 'interrupted'.")

    runner = Runner(
        ledger=ledger,
        runner_config=RunnerConfig(
            work_dir=os.path.join(args.work_dir, "iterations"),
            pretrain_timeout_s=parse_wallclock(args.pretrain_timeout),
            classify_timeout_s_per_submodel=parse_wallclock(args.classify_timeout_per_submodel),
            probe_videos=args.probe_videos,
        ),
    )

    budget = Budget(
        max_wallclock_s=parse_wallclock(args.max_wallclock),
        max_iters=args.max_iters,
        plateau_patience=args.plateau_patience,
    )

    fallback_name = None if args.fallback == "none" else args.fallback
    proposer = make_proposer(args.proposer, fallback_name=fallback_name)

    # Integrity check on prepare.py.
    prep_hash = prepare_file_hash()
    print(f"[autoresearch] prepare.py hash: {prep_hash}")
    print(f"[autoresearch] num_classes: {splits.num_classes}, train: {len(splits.train_indices)}, "
          f"val: {len(splits.val_indices)}, holdout: {len(splits.holdout_indices)}")

    # --skip-pretrain requires a promoted encoder; abort early if missing.
    if args.skip_pretrain:
        if best_pretrain_encoder(ledger) is None:
            print(
                "[autoresearch] ERROR: --skip-pretrain set but no promoted encoder "
                "found at models_best/pretrain.pth. Train one first with "
                "`python app/train.py --config_path configs/config.yaml`, then "
                "`cp pico_jepa_pretrained_encoder.pth models_best/pretrain.pth`.",
                file=sys.stderr,
            )
            return 2
        print("[autoresearch] --skip-pretrain active: only classify/ensemble will be proposed.")

    stats = {p: PhaseStats() for p in PHASES}
    force_classify_next = False
    iter_id = ledger.recent_history(limit=1)
    iter_id = (iter_id[0]["id"] if iter_id else 0) + 1

    while budget.can_continue():
        warmup_n = len(WARMUP_PLAN)
        has_encoder = best_pretrain_encoder(ledger) is not None
        if budget.iters_done < warmup_n:
            phase = pick_phase_warmup(
                budget.iters_done,
                has_pretrained_encoder=has_encoder,
                skip_pretrain=args.skip_pretrain,
            )
        else:
            phase = pick_phase_ucb(
                stats, total_iters=budget.iters_done,
                force_classify=force_classify_next,
                skip_pretrain=args.skip_pretrain,
            )
        force_classify_next = False

        # Skip phases that need prerequisites we don't have yet.
        # With --skip-pretrain the encoder is guaranteed to exist (validated
        # at startup), so the redirects never bottom out at "pretrain".
        if phase == "classify" and best_pretrain_encoder(ledger) is None:
            phase = "pretrain"
        if phase == "ensemble" and ledger.best_for_phase("classify") is None:
            phase = "classify" if best_pretrain_encoder(ledger) else "pretrain"
        if args.skip_pretrain and phase == "pretrain":
            # Should be unreachable given the startup guard, but defend
            # explicitly so a future refactor of the redirects above can't
            # silently violate --skip-pretrain.
            phase = "classify"

        base = best_known_config(ledger, phase, base_config)
        delta = proposer.propose(phase=phase, base_config=base, history=ledger.recent_history(limit=50))
        config = merge(base, delta)

        proposer_active = type(proposer).__name__
        print(f"\n[autoresearch] iter {iter_id} | phase={phase} | proposer={proposer_active} | proposing delta: {json.dumps(delta, sort_keys=True)}")
        reasoning = getattr(proposer, "last_reasoning", None)
        if reasoning:
            print(f"[LLMProposer reasoning] {reasoning}")
        append_proposer_log_jsonl(runner.cfg.work_dir, iter_id, phase, proposer_active, delta, reasoning)
        proposer_info = {"proposer": proposer_active, "delta": delta, "reasoning": reasoning}
        improved = False
        result: Dict[str, Any] = {}

        if phase == "pretrain":
            result = runner.run_pretrain(config=config, splits=splits, iter_id=iter_id)
            if result.get("status") == "completed":
                if is_improvement("pretrain", result["score"], result["wallclock_s"], ledger):
                    promote_artifacts(
                        phase="pretrain", experiment_id=result["experiment_id"],
                        config_path=os.path.join(runner.cfg.work_dir, f"it_{iter_id:04d}_pretrain", "config.yaml"),
                        artifact_paths=[result["encoder_path"]],
                        score=result["score"], ledger=ledger,
                    )
                    improved = True
                    force_classify_next = True

        elif phase == "classify":
            encoder = best_pretrain_encoder(ledger)
            result = runner.run_classify(config=config, splits=splits, encoder_path=encoder, iter_id=iter_id)
            if result.get("status") in ("completed", "partial") and result.get("submodels"):
                submodel_paths = [s["classifier_save_path"] for s in result["submodels"] if s["status"] == "completed"]
                if submodel_paths and is_improvement("classify", result["score"], result["wallclock_s"], ledger):
                    classify_work_dir = os.path.join(runner.cfg.work_dir, f"it_{iter_id:04d}_classify")
                    # Main promoted config = submodel 1 (legacy / fallback).
                    promoted_cfg_path = sanitize_classify_config_for_promotion(
                        submodel_config_path=os.path.join(classify_work_dir, "submodel_1_config.yaml"),
                        base_config=base_config,
                        work_dir=classify_work_dir,
                    )
                    # Per-submodel sanitized configs — needed when submodels are
                    # heterogeneous (different head_type/freeze_encoder).
                    per_submodel_configs = []
                    for i in range(1, len(submodel_paths) + 1):
                        cfg_path = sanitize_classify_config_for_promotion(
                            submodel_config_path=os.path.join(
                                classify_work_dir, f"submodel_{i}_config.yaml"
                            ),
                            base_config=base_config,
                            work_dir=classify_work_dir,
                            output_name=f"promoted_classify_config_{i}.yaml",
                        )
                        per_submodel_configs.append(cfg_path)
                    promote_artifacts(
                        phase="classify", experiment_id=result["experiment_id"],
                        config_path=promoted_cfg_path,
                        artifact_paths=submodel_paths,
                        score=result["score"], ledger=ledger,
                        per_artifact_configs=per_submodel_configs,
                    )
                    improved = True

        elif phase == "ensemble":
            best_classify = ledger.best_for_phase("classify")
            if best_classify is None:
                budget.record_iteration(False)
                continue
            classify_cfg = yaml.safe_load(best_classify["config_yaml"]) or {}
            n_models = int(classify_cfg.get("num_models", 4))
            submodel_paths = [
                os.path.join(PROJECT_ROOT, "models_best", f"classify_{i}.pth") for i in range(1, n_models + 1)
            ]
            available = [p for p in submodel_paths if os.path.exists(p)]
            if not available:
                budget.record_iteration(False)
                continue
            arch = encoder_arch_from_best(ledger, fallback_config=base_config)
            # Heterogeneous ensemble: read per-submodel configs (head_type,
            # freeze_encoder, seed, lr may all differ). Fallback to the global
            # promoted config if the per-submodel files don't exist.
            per_submodel_cfgs = load_per_submodel_configs(
                n_models=len(available), fallback_config=classify_cfg,
            )
            submodel_configs = [
                merge(merge(base_config, c), arch) for c in per_submodel_cfgs
            ]
            result = runner.run_ensemble(
                config=config, submodel_paths=available,
                submodel_configs=submodel_configs, splits=splits, iter_id=iter_id,
            )
            if result.get("status") == "completed" and is_improvement(
                "ensemble", result["score"], result["wallclock_s"], ledger
            ):
                ensemble_promote_dir = os.path.join(runner.cfg.work_dir, f"it_{iter_id:04d}_classify")
                ensemble_promote_cfg = sanitize_classify_config_for_promotion(
                    submodel_config_path=os.path.join(ensemble_promote_dir, "submodel_1_config.yaml"),
                    base_config=base_config,
                    work_dir=ensemble_promote_dir,
                )
                promote_artifacts(
                    phase="ensemble", experiment_id=result["experiment_id"],
                    config_path=ensemble_promote_cfg,
                    artifact_paths=[],  # no checkpoint for ensemble itself
                    score=result["score"], ledger=ledger,
                )
                improved = True
                # After ensemble improves, kick off Phase 4 evaluation.
                _maybe_run_phase4(
                    runner, ledger, splits, base_config, classify_cfg, available,
                    submodel_configs, config, iter_id,
                    ensemble_classifier=result.get("ensemble_classifier"),
                )

        # Update bandit stats and report.
        s = stats[phase]
        s.n_attempts += 1
        if improved:
            s.n_improvements += 1
        s.total_wallclock_s += float(result.get("wallclock_s") or 0.0)

        write_iteration_report(runner.cfg.work_dir, iter_id, phase, result, improved, proposer_info)
        budget.record_iteration(improved)
        iter_id += 1

        bs = budget.status()
        print(f"[autoresearch] iter done | improved={improved} | elapsed={bs['elapsed_s']:.0f}s | "
              f"remaining={bs['remaining_s']} | no_improve_streak={bs['consecutive_no_improvement']}/{bs['plateau_patience']}")

    print("\n[autoresearch] search loop terminated.")
    print(json.dumps(budget.status(), indent=2, default=str))
    return 0


def _maybe_run_phase4(
    runner: Runner, ledger: Ledger, splits: Splits, base_config: Dict[str, Any],
    classify_cfg: Dict[str, Any], submodel_paths: List[str],
    submodel_configs: List[Dict[str, Any]], ensemble_config: Dict[str, Any], iter_id: int,
    ensemble_classifier: Any = None,
) -> None:
    """Train a single 'general' classifier on the full training set and compare."""
    from autoresearch.adapters import classify as classify_adapter
    general_dir = os.path.join(runner.cfg.work_dir, f"it_{iter_id:04d}_general")
    os.makedirs(general_dir, exist_ok=True)
    encoder = best_pretrain_encoder(ledger)
    if encoder is None:
        return
    arch = encoder_arch_from_best(ledger, fallback_config=base_config)
    general_cfg = merge(merge(base_config, classify_cfg), arch)
    general_cfg["encoder_save_path"] = encoder
    general_cfg["num_models"] = 1
    general_cfg["partition_strategy"] = "random-subset"  # full training set, single model

    res = classify_adapter.run_classify(
        base_config=general_cfg,
        classify_csv_path=splits.csv_path,
        train_indices=splits.train_indices,
        val_indices=splits.val_indices,
        num_models=1, partition_strategy="random-subset",
        work_dir=general_dir, seed=1337,
        per_submodel_timeout_s=runner.cfg.classify_timeout_s_per_submodel,
    )
    if res["status"] == "failed" or not res["submodels"]:
        return
    general_path = res["submodels"][0]["classifier_save_path"]
    if not os.path.exists(general_path):
        return

    runner.run_phase4(
        ensemble_config=ensemble_config,
        ensemble_submodel_paths=submodel_paths,
        ensemble_submodel_configs=submodel_configs,
        general_model_path=general_path,
        general_config=general_cfg,
        splits=splits,
        iter_id=iter_id,
        ensemble_classifier=ensemble_classifier,
    )


if __name__ == "__main__":
    sys.exit(main())
