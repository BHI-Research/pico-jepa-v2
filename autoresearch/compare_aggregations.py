"""Compare all four ensemble aggregation methods in a single run.

Reuses the inferences (val and holdout) so the four methods cost only the
aggregation step (CPU, seconds). The general model is also evaluated once.
A markdown table is written to ``autoresearch/artifacts/reports/phase4_comparison.md``
and each result is persisted as its own ``phase4`` experiment in the ledger.

Usage:
    python -m autoresearch.compare_aggregations
    python -m autoresearch.compare_aggregations --reuse-general
    python -m autoresearch.compare_aggregations --num-models 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

from autoresearch.adapters import ensemble as ensemble_adapter
from autoresearch.adapters import infer as infer_adapter
from autoresearch.budget import parse_wallclock
from autoresearch.ledger import ExperimentRecord, Ledger
from autoresearch.prepare import Splits, fitness_gap, load_holdout_indices, make_splits
from autoresearch.run_phase4 import (
    BEST_MODELS_DIR,
    find_classify_config,
    find_per_submodel_configs,
    find_pretrain_encoder,
    find_submodel_paths,
    load_yaml,
    train_general_model,
)
from autoresearch.runner import (
    Runner,
    RunnerConfig,
    _config_hash,
    _config_yaml,
    _git_sha,
    encoder_arch_from_best,
    merge,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts", "reports")

AGGREGATIONS: List[Dict[str, Any]] = [
    {"name": "hard_vote",     "kwargs": {}},
    {"name": "soft_vote",     "kwargs": {"temperature": 1.0}},
    {"name": "weighted_vote", "kwargs": {"temperature": 1.0}},
    {"name": "stacking",      "kwargs": {"meta_learner": "logreg"}},
]


def _verdict(stats: Dict[str, float]) -> str:
    if stats["gap_ci_low"] > 0:
        return "SUPPORTED"
    if stats["gap"] > 0:
        return "INCONCLUSIVE"
    return "REJECTED"


def evaluate_one(
    name: str,
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    holdout_probs: np.ndarray,
    holdout_labels: np.ndarray,
    general_preds: np.ndarray,
    per_model_acc: List[float],
    kwargs: Dict[str, Any],
    bootstrap_seed: int = 1337,
) -> Dict[str, Any]:
    """Run a single aggregation method on already-cached probabilities."""
    started = time.perf_counter()
    if name == "hard_vote":
        ens_preds = ensemble_adapter.hard_vote(holdout_probs)
    elif name == "soft_vote":
        ens_preds = ensemble_adapter.soft_vote(
            holdout_probs, temperature=float(kwargs.get("temperature", 1.0))
        )
    elif name == "weighted_vote":
        ens_preds = ensemble_adapter.weighted_vote(
            holdout_probs,
            weights=per_model_acc,
            temperature=float(kwargs.get("temperature", 1.0)),
        )
    elif name == "stacking":
        clf = ensemble_adapter.stacking_fit(
            val_probs=val_probs, val_labels=val_labels,
            meta_learner=kwargs.get("meta_learner", "logreg"),
        )
        ens_preds = ensemble_adapter.stacking_predict(clf, holdout_probs)
    else:
        raise ValueError(f"Unknown aggregation {name!r}")

    stats = ensemble_adapter.bootstrap_gap_ci(
        ensemble_preds=ens_preds,
        general_preds=general_preds,
        labels=holdout_labels,
        seed=bootstrap_seed,
    )
    stats["gap"] = fitness_gap(stats["ensemble_top1"], stats["general_top1"])
    stats["aggregation"] = name
    stats["wallclock_s"] = time.perf_counter() - started
    stats["verdict"] = _verdict(stats)
    return stats


def write_markdown_table(
    results: List[Dict[str, Any]],
    n_models: int,
    holdout_size: int,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sorted_rows = sorted(results, key=lambda r: r["gap"], reverse=True)
    lines = [
        "# Phase 4 — comparación de agregaciones",
        "",
        f"- holdout_size: {holdout_size}",
        f"- n_models: {n_models}",
        f"- general_top1: {results[0]['general_top1']:.4f}",
        "",
        "| aggregation | ensemble_top1 | gap | CI low | CI high | verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in sorted_rows:
        lines.append(
            f"| `{r['aggregation']}` | {r['ensemble_top1']:.4f} | "
            f"{r['gap']:+.4f} | {r['gap_ci_low']:+.4f} | {r['gap_ci_high']:+.4f} | "
            f"**{r['verdict']}** |"
        )
    lines += ["", "## Detalles JSON", "```json",
              json.dumps(sorted_rows, indent=2, default=str), "```"]
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[compare] markdown report -> {output_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compare all four ensemble aggregations on the holdout.")
    p.add_argument("--base-config", default=os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    p.add_argument("--ledger-path", default=None)
    p.add_argument("--num-models", type=int, default=None)
    p.add_argument("--general-model", default=None)
    p.add_argument("--reuse-general", action="store_true")
    p.add_argument("--num-eval-clips", type=int, default=1,
                   help="V-JEPA-style multi-clip eval; 10 = average softmax over 10 clips per video.")
    p.add_argument("--classify-timeout-per-submodel", default="20m")
    p.add_argument("--report-path", default=os.path.join(REPORTS_DIR, "phase4_comparison.md"))
    p.add_argument("--bootstrap-seed", type=int, default=1337,
                   help="Seed for the bootstrap CI resampling. Vary this (e.g. 42, 7, 2024) "
                        "to confirm that SUPPORTED verdicts are robust to the bootstrap random draw.")
    args = p.parse_args(argv)

    ledger = Ledger(args.ledger_path) if args.ledger_path else Ledger()
    base_config = load_yaml(args.base_config)
    classify_cfg = find_classify_config(ledger)
    submodel_paths = find_submodel_paths(num_models_hint=args.num_models)
    if not submodel_paths:
        raise SystemExit("No submodels in models_best/. Run the search loop first.")
    encoder_path = find_pretrain_encoder()

    classify_csv_path = base_config.get(
        "classify_csv_path",
        os.path.join(
            base_config.get("classify_video_dir", base_config.get("video_dir", ".")),
            base_config.get("csv_file_labeled", ""),
        ),
    )
    splits = make_splits(classify_csv_path)
    print(f"[compare] num_classes={splits.num_classes} | train={len(splits.train_indices)} "
          f"| val={len(splits.val_indices)} | holdout={len(splits.holdout_indices)}")
    print(f"[compare] submodels: {len(submodel_paths)}")

    arch = encoder_arch_from_best(ledger, fallback_config=base_config)
    # Heterogeneous ensemble: read per-submodel configs from
    # configs/best/classify_{i}.yaml. Fallback to the global classify.yaml.
    per_submodel_cfgs = find_per_submodel_configs(
        n_models=len(submodel_paths), fallback_config=classify_cfg,
    )
    submodel_configs = []
    for cfg_i in per_submodel_cfgs:
        sm_cfg = merge(merge(base_config, cfg_i), arch)
        sm_cfg["encoder_save_path"] = encoder_path
        submodel_configs.append(sm_cfg)

    # General model: train once, reuse across the 4 aggregations.
    promoted_general = os.path.join(BEST_MODELS_DIR, "general.pth")
    if args.general_model and os.path.exists(args.general_model):
        general_path = args.general_model
        print(f"[compare] using --general-model {general_path}")
    elif args.reuse_general and os.path.exists(promoted_general):
        general_path = promoted_general
        print(f"[compare] reusing existing {general_path}")
    else:
        runner = Runner(
            ledger=ledger,
            runner_config=RunnerConfig(
                work_dir=os.path.join(os.path.dirname(__file__), "artifacts", "iterations"),
                classify_timeout_s_per_submodel=parse_wallclock(args.classify_timeout_per_submodel),
            ),
        )
        general_dir = os.path.join(runner.cfg.work_dir, "phase4_general")
        general_path = train_general_model(
            base_config=base_config, classify_cfg=classify_cfg, splits=splits,
            encoder_path=encoder_path, runner=runner, work_dir=general_dir,
        )

    # All submodels share classify_video_dir (it's an architectural-adjacent key
    # restored from base_config by encoder_arch_from_best). Take from the first.
    video_dir = submodel_configs[0].get(
        "classify_video_dir", submodel_configs[0].get("video_dir")
    )

    # --- Phase 4 evaluation: cache all inferences once. ---
    os.environ["AUTORESEARCH_PHASE"] = "4"
    try:
        # Holdout inference for all submodels.
        holdout_idx = load_holdout_indices(splits)
        holdout_paths, holdout_labels = infer_adapter.videos_from_csv(
            splits.csv_path, video_dir=video_dir, indices=holdout_idx
        )
        num_clips = max(1, int(args.num_eval_clips))
        print(f"[compare] inferring submodels on holdout ({len(holdout_paths)} videos, num_clips={num_clips})...")
        ens_out = infer_adapter.infer_with_models(
            model_paths=submodel_paths, configs=submodel_configs,
            num_classes=splits.num_classes, video_paths=holdout_paths, labels=holdout_labels,
            num_clips=num_clips,
        )

        # Val inference (only needed for stacking, but we do it once and keep it).
        val_paths, val_labels = infer_adapter.videos_from_csv(
            splits.csv_path, video_dir=video_dir, indices=splits.val_indices
        )
        print(f"[compare] inferring submodels on val ({len(val_paths)} videos, num_clips={num_clips})...")
        val_out = infer_adapter.infer_with_models(
            model_paths=submodel_paths, configs=submodel_configs,
            num_classes=splits.num_classes, video_paths=val_paths, labels=val_labels,
            num_clips=num_clips,
        )

        # General inference on holdout (same architecture as the submodels).
        general_cfg = merge(merge(base_config, classify_cfg), arch)
        general_cfg["encoder_save_path"] = encoder_path
        print("[compare] inferring general model on holdout...")
        gen_out = infer_adapter.infer_with_models(
            model_paths=[general_path], configs=[general_cfg],
            num_classes=splits.num_classes, video_paths=holdout_paths, labels=holdout_labels,
            num_clips=num_clips,
        )
        general_preds = gen_out["preds"][0]
        general_top1 = float((general_preds == np.asarray(holdout_labels)).mean())
        print(f"[compare] general_top1 on holdout: {general_top1:.4f}")

        results: List[Dict[str, Any]] = []
        for entry in AGGREGATIONS:
            name = entry["name"]
            print(f"[compare] evaluating {name}...")
            stats = evaluate_one(
                name=name,
                val_probs=val_out["probs"], val_labels=val_out["labels"],
                holdout_probs=ens_out["probs"], holdout_labels=np.asarray(holdout_labels),
                general_preds=general_preds, per_model_acc=ens_out["per_model_acc"] or [1.0] * len(submodel_paths),
                kwargs=entry["kwargs"],
                bootstrap_seed=args.bootstrap_seed,
            )
            stats["n_models"] = len(submodel_paths)
            stats["holdout_size"] = len(holdout_labels)
            results.append(stats)

            # Persist into the ledger as a phase4 experiment per aggregation.
            rec = ExperimentRecord(
                phase="phase4", status="running",
                config_hash=_config_hash({"agg": name, "n": len(submodel_paths)}),
                config_yaml=_config_yaml({"phase4": stats}),
                git_sha=_git_sha(),
            )
            exp_id = ledger.start_experiment(rec)
            ledger.finish_experiment(
                exp_id, status="completed", wallclock_s=stats["wallclock_s"],
                score=stats["gap"], metrics=stats,
            )
            print(f"[compare]   -> ensemble_top1={stats['ensemble_top1']:.4f} | "
                  f"gap={stats['gap']:+.4f} | CI=[{stats['gap_ci_low']:+.4f}, {stats['gap_ci_high']:+.4f}] | "
                  f"verdict={stats['verdict']}")
    finally:
        os.environ.pop("AUTORESEARCH_PHASE", None)

    write_markdown_table(
        results=results, n_models=len(submodel_paths),
        holdout_size=len(holdout_labels), output_path=args.report_path,
    )

    # Stdout summary table.
    print("\n" + "=" * 78)
    print(f"{'aggregation':<16}{'ens_top1':>10}{'gap':>10}{'CI_low':>10}{'CI_high':>10}{'verdict':>16}")
    print("-" * 78)
    for r in sorted(results, key=lambda x: x["gap"], reverse=True):
        print(
            f"{r['aggregation']:<16}"
            f"{r['ensemble_top1']:>10.4f}"
            f"{r['gap']:>+10.4f}"
            f"{r['gap_ci_low']:>+10.4f}"
            f"{r['gap_ci_high']:>+10.4f}"
            f"{r['verdict']:>16}"
        )
    print("=" * 78)
    best = max(results, key=lambda r: r["gap"])
    print(f"\nBest aggregation: {best['aggregation']} (gap={best['gap']:+.4f}, verdict={best['verdict']})")
    return 0 if best["gap_ci_low"] > 0 else (1 if best["gap"] <= 0 else 0)


if __name__ == "__main__":
    sys.exit(main())
