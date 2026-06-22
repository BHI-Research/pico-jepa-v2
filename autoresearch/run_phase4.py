"""Standalone CLI to run Phase 4 (hypothesis evaluation) with the artifacts
that are currently promoted in ``models_best/`` and ``configs/best/``.

Use this when the search loop terminated by plateau or budget without ever
triggering Phase 4 (which only runs after a Phase 3 ratchet improvement).

Usage:
    python -m autoresearch.run_phase4 \\
        --base-config configs/config.yaml \\
        --aggregation soft_vote
        # optionally --general-model models_best/general.pth to skip retraining

If ``--general-model`` is omitted, this command trains a single classifier on
the full train_indices using the best pretrain encoder, saves it to
``models_best/general.pth`` for reuse, then evaluates ensemble vs general on
the protected holdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import yaml

from autoresearch.adapters import classify as classify_adapter
from autoresearch.adapters import ensemble as ensemble_adapter
from autoresearch.adapters import infer as infer_adapter
from autoresearch.ledger import Ledger
from autoresearch.prepare import Splits, fitness_gap, load_holdout_indices, make_splits
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
BEST_CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs", "best")
BEST_MODELS_DIR = os.path.join(PROJECT_ROOT, "models_best")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def find_classify_config(ledger: Ledger) -> Dict[str, Any]:
    """Prefer ``configs/best/classify.yaml``; fall back to the ledger best."""
    promoted = os.path.join(BEST_CONFIGS_DIR, "classify.yaml")
    if os.path.exists(promoted):
        return load_yaml(promoted)
    rec = ledger.best_for_phase("classify", accept_partial=True)
    if rec is None:
        raise SystemExit(
            "No classify config found. Either configs/best/classify.yaml is missing "
            "and no classify experiment in the ledger has status completed/partial. "
            "Run the search loop first."
        )
    return yaml.safe_load(rec["config_yaml"]) or {}


def find_per_submodel_configs(
    n_models: int, fallback_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Read configs/best/classify_{i}.yaml; fall back to classify.yaml then fallback_config."""
    global_cfg = fallback_config
    promoted = os.path.join(BEST_CONFIGS_DIR, "classify.yaml")
    if os.path.exists(promoted):
        global_cfg = load_yaml(promoted)
    configs: List[Dict[str, Any]] = []
    for i in range(1, n_models + 1):
        per_path = os.path.join(BEST_CONFIGS_DIR, f"classify_{i}.yaml")
        if os.path.exists(per_path):
            configs.append(load_yaml(per_path))
        else:
            configs.append(dict(global_cfg))
    return configs


def find_submodel_paths(num_models_hint: Optional[int]) -> List[str]:
    """Return existing models_best/classify_*.pth paths in numeric order."""
    if not os.path.isdir(BEST_MODELS_DIR):
        return []
    candidates: List[str] = []
    for name in sorted(os.listdir(BEST_MODELS_DIR)):
        if name.startswith("classify_") and name.endswith(".pth"):
            candidates.append(os.path.join(BEST_MODELS_DIR, name))
    if num_models_hint is not None:
        candidates = candidates[:num_models_hint]
    return candidates


def find_pretrain_encoder() -> str:
    path = os.path.join(BEST_MODELS_DIR, "pretrain.pth")
    if not os.path.exists(path):
        raise SystemExit(
            f"Pretrain encoder not found at {path}. Run the search loop until "
            "at least one pretrain experiment is promoted."
        )
    return path


def train_general_model(
    base_config: Dict[str, Any],
    classify_cfg: Dict[str, Any],
    splits: Splits,
    encoder_path: str,
    runner: Runner,
    work_dir: str,
    ledger: Optional[Ledger] = None,
) -> str:
    """Train a single (general) classifier on the full train set and return its .pth path."""
    arch = encoder_arch_from_best(ledger, fallback_config=base_config)
    cfg = merge(merge(base_config, classify_cfg), arch)
    cfg["encoder_save_path"] = encoder_path
    cfg["num_models"] = 1
    cfg["partition_strategy"] = "random-subset"

    print(f"[run_phase4] Training general model (1 submodel, full train set) -> {work_dir}")
    res = classify_adapter.run_classify(
        base_config=cfg,
        classify_csv_path=splits.csv_path,
        train_indices=splits.train_indices,
        val_indices=splits.val_indices,
        num_models=1,
        partition_strategy="random-subset",
        work_dir=work_dir,
        seed=1337,
        per_submodel_timeout_s=runner.cfg.classify_timeout_s_per_submodel,
    )
    if not res["submodels"]:
        raise SystemExit("General-model training produced no submodels.")
    s = res["submodels"][0]
    if s["status"] not in ("completed", "partial"):
        raise SystemExit(f"General-model training failed: status={s['status']}.")
    src = s["classifier_save_path"]
    if not os.path.exists(src):
        raise SystemExit(f"General-model checkpoint missing at {src}.")
    target = os.path.join(BEST_MODELS_DIR, "general.pth")
    os.makedirs(BEST_MODELS_DIR, exist_ok=True)
    import shutil
    shutil.copy2(src, target)
    print(f"[run_phase4] General model saved to {target}")
    return target


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run Phase 4 (ensemble vs general) with current artifacts.")
    p.add_argument("--base-config", default=os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    p.add_argument("--ledger-path", default=None)
    p.add_argument("--aggregation", choices=["hard_vote", "soft_vote", "weighted_vote", "stacking"],
                   default="soft_vote")
    p.add_argument("--meta-learner", choices=["logreg", "svm"], default="logreg")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num-eval-clips", type=int, default=1,
                   help="V-JEPA-style multi-clip eval. 10 averages softmax over 10 uniformly-spaced clips; 1 = single center clip (default).")
    p.add_argument("--num-models", type=int, default=None,
                   help="Override the number of submodels picked from models_best/.")
    p.add_argument("--general-model", default=None,
                   help="Path to an existing general .pth to compare against. "
                        "If omitted, one is trained at runtime and saved to models_best/general.pth.")
    p.add_argument("--reuse-general", action="store_true",
                   help="If models_best/general.pth exists, reuse it instead of retraining.")
    p.add_argument("--classify-timeout-per-submodel", default="20m")
    args = p.parse_args(argv)

    ledger = Ledger(args.ledger_path) if args.ledger_path else Ledger()
    base_config = load_yaml(args.base_config)
    classify_cfg = find_classify_config(ledger)
    submodel_paths = find_submodel_paths(num_models_hint=args.num_models)
    if not submodel_paths:
        raise SystemExit(
            "No submodels found in models_best/classify_*.pth. Run the search "
            "loop until at least one classify experiment is promoted."
        )
    encoder_path = find_pretrain_encoder()

    classify_csv_path = base_config.get(
        "classify_csv_path",
        os.path.join(
            base_config.get("classify_video_dir", base_config.get("video_dir", ".")),
            base_config.get("csv_file_labeled", ""),
        ),
    )
    splits = make_splits(classify_csv_path)

    print(f"[run_phase4] num_classes={splits.num_classes} | train={len(splits.train_indices)} "
          f"| val={len(splits.val_indices)} | holdout={len(splits.holdout_indices)}")
    print(f"[run_phase4] submodels: {len(submodel_paths)} -> {submodel_paths}")
    print(f"[run_phase4] aggregation={args.aggregation}")

    # Build per-submodel configs (heterogeneous: head_type, freeze_encoder, etc.
    # can vary). FORCE the architecture to match the promoted pretrain checkpoint
    # -- otherwise load_state_dict mismatches.
    arch = encoder_arch_from_best(ledger, fallback_config=base_config)
    per_submodel_cfgs = find_per_submodel_configs(
        n_models=len(submodel_paths), fallback_config=classify_cfg,
    )
    submodel_configs = []
    for cfg_i in per_submodel_cfgs:
        sm_cfg = merge(merge(base_config, cfg_i), arch)
        sm_cfg["encoder_save_path"] = encoder_path
        submodel_configs.append(sm_cfg)

    # Pick or train the general model.
    promoted_general = os.path.join(BEST_MODELS_DIR, "general.pth")
    if args.general_model and os.path.exists(args.general_model):
        general_path = args.general_model
    elif args.reuse_general and os.path.exists(promoted_general):
        general_path = promoted_general
        print(f"[run_phase4] Reusing existing general model at {general_path}")
    else:
        from autoresearch.budget import parse_wallclock
        runner = Runner(
            ledger=ledger,
            runner_config=RunnerConfig(
                work_dir=os.path.join(os.path.dirname(__file__), "artifacts", "iterations"),
                classify_timeout_s_per_submodel=parse_wallclock(args.classify_timeout_per_submodel),
            ),
        )
        general_dir = os.path.join(runner.cfg.work_dir, "phase4_general")
        general_path = train_general_model(
            base_config=base_config,
            classify_cfg=classify_cfg,
            splits=splits,
            encoder_path=encoder_path,
            runner=runner,
            work_dir=general_dir,
            ledger=ledger,
        )

    # --- Phase 4 evaluation ---
    os.environ["AUTORESEARCH_PHASE"] = "4"
    try:
        holdout_idx = load_holdout_indices(splits)
        video_dir = submodel_configs[0].get("classify_video_dir", submodel_configs[0].get("video_dir"))
        paths, labels = infer_adapter.videos_from_csv(
            splits.csv_path, video_dir=video_dir, indices=holdout_idx
        )

        num_clips = max(1, int(args.num_eval_clips))
        # Ensemble inference on holdout.
        ens_out = infer_adapter.infer_with_models(
            model_paths=submodel_paths,
            configs=submodel_configs,
            num_classes=splits.num_classes,
            video_paths=paths,
            labels=labels,
            num_clips=num_clips,
        )

        # If stacking, fit on val first.
        ensemble_classifier = None
        if args.aggregation == "stacking":
            val_paths, val_labels = infer_adapter.videos_from_csv(
                splits.csv_path, video_dir=video_dir, indices=splits.val_indices
            )
            val_out = infer_adapter.infer_with_models(
                model_paths=submodel_paths, configs=submodel_configs,
                num_classes=splits.num_classes, video_paths=val_paths, labels=val_labels,
                num_clips=num_clips,
            )
            ensemble_classifier = ensemble_adapter.stacking_fit(
                val_probs=val_out["probs"], val_labels=val_out["labels"],
                meta_learner=args.meta_learner,
            )
            ens_preds = ensemble_adapter.stacking_predict(ensemble_classifier, ens_out["probs"])
        else:
            weights = ens_out["per_model_acc"] if args.aggregation == "weighted_vote" else None
            ens_preds = ensemble_adapter.aggregate(
                method=args.aggregation, probs=ens_out["probs"],
                weights=weights, temperature=args.temperature,
            )["preds"]

        # General inference on holdout (same arch as the submodels).
        general_cfg = merge(merge(base_config, classify_cfg), arch)
        general_cfg["encoder_save_path"] = encoder_path
        gen_out = infer_adapter.infer_with_models(
            model_paths=[general_path], configs=[general_cfg],
            num_classes=splits.num_classes, video_paths=paths, labels=labels,
            num_clips=num_clips,
        )
        gen_preds = gen_out["preds"][0]

        stats = ensemble_adapter.bootstrap_gap_ci(
            ensemble_preds=ens_preds, general_preds=gen_preds, labels=labels,
        )
        stats["gap"] = fitness_gap(stats["ensemble_top1"], stats["general_top1"])
        stats["aggregation"] = args.aggregation
        stats["n_models"] = len(submodel_paths)
        stats["holdout_size"] = len(labels)
        stats["num_eval_clips"] = num_clips
    finally:
        os.environ.pop("AUTORESEARCH_PHASE", None)

    # Persist into the ledger as a phase4 experiment.
    from autoresearch.ledger import ExperimentRecord
    rec = ExperimentRecord(
        phase="phase4", status="running",
        config_hash=_config_hash({"agg": args.aggregation, "n": len(submodel_paths)}),
        config_yaml=_config_yaml({"phase4": stats}),
        git_sha=_git_sha(),
    )
    exp_id = ledger.start_experiment(rec)
    ledger.finish_experiment(
        exp_id, status="completed", wallclock_s=0.0,
        score=stats["gap"], metrics=stats,
    )

    # Pretty-print verdict.
    print("\n" + "=" * 60)
    print("Phase 4 — Hypothesis evaluation")
    print("=" * 60)
    print(f"  aggregation:        {stats['aggregation']}")
    print(f"  n_models:           {stats['n_models']}")
    print(f"  holdout_size:       {stats['holdout_size']}")
    print(f"  ensemble_top1:      {stats['ensemble_top1']:.4f}")
    print(f"  general_top1:       {stats['general_top1']:.4f}")
    print(f"  gap:                {stats['gap']:+.4f}")
    print(f"  gap CI [2.5, 97.5]: [{stats['gap_ci_low']:+.4f}, {stats['gap_ci_high']:+.4f}]")
    if stats["gap_ci_low"] > 0:
        verdict = "SUPPORTED — ensemble is significantly better."
    elif stats["gap"] > 0:
        verdict = "INCONCLUSIVE — gap is positive but CI includes zero."
    else:
        verdict = "REJECTED — ensemble does not beat the general model on this holdout."
    print(f"  Hypothesis:         {verdict}")
    print(f"  ledger experiment_id: {exp_id}")
    print("=" * 60 + "\n")
    print(json.dumps(stats, indent=2, default=str))
    return 0 if stats["gap_ci_low"] > 0 else (1 if stats["gap"] <= 0 else 0)


if __name__ == "__main__":
    sys.exit(main())
