"""Per-iteration runner: takes a (phase, config) pair, executes the right
adapter, captures metrics into the ledger, and returns a normalized result.

The runner is the integration point between the search loop (which decides
what to try) and the adapters (which run real PyTorch code in a subprocess).
It also enforces validation against ``search_space`` before doing any work.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import yaml

from autoresearch.adapters import classify as classify_adapter
from autoresearch.adapters import ensemble as ensemble_adapter
from autoresearch.adapters import infer as infer_adapter
from autoresearch.adapters import pretrain as pretrain_adapter
from autoresearch.ledger import ExperimentRecord, Ledger
from autoresearch.metrics import jepa_probe
from autoresearch.prepare import Splits, fitness_gap
from autoresearch.search_space import validate_config


@dataclass
class RunnerConfig:
    work_dir: str
    pretrain_timeout_s: float = 90 * 60          # 90 min per pretrain experiment
    classify_timeout_s_per_submodel: float = 20 * 60  # 20 min per submodel
    ensemble_timeout_s: float = 5 * 60           # 5 min total (CPU-bound)
    probe_videos: int = 500
    probe_epochs: int = 1


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except Exception:
        return None


def _config_hash(config: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _config_yaml(config: Dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=True)


def merge(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(delta)
    return out


# Keys that define the *architecture* of the encoder. They are owned by the
# pretrain phase: classify and ensemble must inherit them from the promoted
# pretrain config so the classifier's encoder matches the saved checkpoint.
# If the proposer mutates any of these in classify, we silently override.
ENCODER_ARCH_KEYS: tuple = (
    "vit_embed_dim",
    "vit_depth",
    "vit_num_heads",
    "vit_mlp_ratio",
    "vit_patch_size_t",
    "vit_patch_size_h",
    "vit_patch_size_w",
    "frames_per_clip",
    "resize_height",
    "resize_width",
    "video_channels",
    "vit_dropout",
)


def encoder_arch_from_best(
    ledger,
    fallback_config: Dict[str, Any],
    project_root: str = None,
) -> Dict[str, Any]:
    """Return the architecture sub-dict from the promoted pretrain config.

    Priority: ``<project_root>/configs/best/pretrain.yaml`` > ledger's best
    pretrain row > fallback_config. ``project_root`` is configurable so tests
    can isolate themselves from the real on-disk artifacts.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    promoted = os.path.join(project_root, "configs", "best", "pretrain.yaml")
    cfg: Dict[str, Any] = {}
    if os.path.exists(promoted):
        with open(promoted) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        rec = ledger.best_for_phase("pretrain", accept_partial=True) if ledger else None
        if rec is not None:
            cfg = yaml.safe_load(rec["config_yaml"]) or {}
    if not cfg:
        cfg = fallback_config
    return {k: cfg[k] for k in ENCODER_ARCH_KEYS if k in cfg}


class Runner:
    def __init__(self, ledger: Ledger, runner_config: RunnerConfig):
        self.ledger = ledger
        self.cfg = runner_config

    # --- Phase 1 ---

    def run_pretrain(
        self,
        config: Dict[str, Any],
        splits: Splits,
        iter_id: int,
    ) -> Dict[str, Any]:
        ok, errors = validate_config("pretrain", config)
        config_hash = _config_hash(config)
        if not ok:
            return self._record_invalid("pretrain", config, config_hash, errors)
        if self.ledger.has_completed_config(config_hash, "pretrain"):
            return {"status": "skipped_duplicate", "config_hash": config_hash}

        rec = ExperimentRecord(
            phase="pretrain",
            status="running",
            config_hash=config_hash,
            config_yaml=_config_yaml(config),
            git_sha=_git_sha(),
        )
        exp_id = self.ledger.start_experiment(rec)

        work_dir = os.path.join(self.cfg.work_dir, f"it_{iter_id:04d}_pretrain")
        # Direct encoder save_path into the iteration work_dir so each run is isolated.
        config = dict(config)
        config["encoder_save_path"] = os.path.join(work_dir, "encoder.pth")

        result = pretrain_adapter.run_pretrain(
            config=config, work_dir=work_dir, timeout_s=self.cfg.pretrain_timeout_s
        )

        # Run the linear probe on the encoder (the actual fitness signal).
        probe = {"probe_top1": 0.0, "probe_wallclock_s": 0.0}
        if result["status"] == "completed" and os.path.exists(config["encoder_save_path"]):
            try:
                probe = jepa_probe.linear_probe(
                    encoder_path=config["encoder_save_path"],
                    config=config,
                    classify_csv_path=splits.csv_path,
                    classify_video_dir=config.get(
                        "classify_video_dir", config.get("video_dir")
                    ),
                    num_classes=splits.num_classes,
                    num_videos=self.cfg.probe_videos,
                    epochs=self.cfg.probe_epochs,
                )
            except Exception as e:
                probe = {"probe_top1": 0.0, "probe_error": str(e)}

        metrics = {**result["metrics"], **probe}
        score = float(probe.get("probe_top1", 0.0))
        wallclock_s = result.get("wallclock_s", 0.0)
        self.ledger.finish_experiment(
            exp_id, status=result["status"], wallclock_s=wallclock_s,
            score=score, metrics=metrics,
            notes=(result.get("stderr_tail") or "")[-512:],
        )
        if result["status"] == "completed":
            self.ledger.add_artifact(exp_id, "encoder", config["encoder_save_path"])
            self.ledger.add_artifact(exp_id, "config", os.path.join(work_dir, "config.yaml"))

        return {
            "status": result["status"],
            "experiment_id": exp_id,
            "config_hash": config_hash,
            "score": score,
            "metrics": metrics,
            "encoder_path": config["encoder_save_path"],
            "wallclock_s": wallclock_s,
        }

    # --- Phase 2 ---

    def run_classify(
        self,
        config: Dict[str, Any],
        splits: Splits,
        encoder_path: str,
        iter_id: int,
    ) -> Dict[str, Any]:
        ok, errors = validate_config("classify", config)
        config_hash = _config_hash(config)
        if not ok:
            return self._record_invalid("classify", config, config_hash, errors)
        if self.ledger.has_completed_config(config_hash, "classify"):
            return {"status": "skipped_duplicate", "config_hash": config_hash}

        rec = ExperimentRecord(
            phase="classify", status="running", config_hash=config_hash,
            config_yaml=_config_yaml(config), git_sha=_git_sha(),
        )
        exp_id = self.ledger.start_experiment(rec)

        work_dir = os.path.join(self.cfg.work_dir, f"it_{iter_id:04d}_classify")
        os.makedirs(work_dir, exist_ok=True)

        # Inject the encoder we want all submodels to start from, and force
        # the architecture keys to match the promoted pretrain config (the
        # checkpoint was saved with those dimensions; using the base config's
        # values would cause a silent shape mismatch / random encoder).
        cfg = dict(config)
        cfg["encoder_save_path"] = encoder_path
        cfg.update(encoder_arch_from_best(self.ledger, fallback_config=cfg))

        result = classify_adapter.run_classify(
            base_config=cfg,
            classify_csv_path=splits.csv_path,
            train_indices=splits.train_indices,
            val_indices=splits.val_indices,
            num_models=int(cfg.get("num_models", 4)),
            partition_strategy=cfg.get("partition_strategy", "disjoint"),
            work_dir=work_dir,
            seed=1337,
            per_submodel_timeout_s=self.cfg.classify_timeout_s_per_submodel,
        )

        score = float(result["mean_val_acc"])
        wallclock_s = result["wallclock_s"]
        self.ledger.finish_experiment(
            exp_id, status=result["status"], wallclock_s=wallclock_s,
            score=score, metrics={
                "mean_val_acc": result["mean_val_acc"],
                "num_models": int(cfg.get("num_models", 4)),
                "partition_strategy": cfg.get("partition_strategy", "disjoint"),
                "submodels": [
                    {
                        "status": s["status"],
                        "wallclock_s": s["wallclock_s"],
                        "metrics": s["metrics"],
                        "classifier_save_path": s["classifier_save_path"],
                    }
                    for s in result["submodels"]
                ],
            },
        )
        for i, s in enumerate(result["submodels"], start=1):
            if s["status"] == "completed":
                self.ledger.add_artifact(exp_id, f"submodel_{i}", s["classifier_save_path"])
        return {
            "status": result["status"],
            "experiment_id": exp_id,
            "config_hash": config_hash,
            "score": score,
            "submodels": result["submodels"],
            "wallclock_s": wallclock_s,
        }

    # --- Phase 3 ---

    def run_ensemble(
        self,
        config: Dict[str, Any],
        submodel_paths: Sequence[str],
        submodel_configs: Sequence[Dict[str, Any]],
        splits: Splits,
        iter_id: int,
    ) -> Dict[str, Any]:
        ok, errors = validate_config("ensemble", config)
        config_hash = _config_hash(config)
        if not ok:
            return self._record_invalid("ensemble", config, config_hash, errors)
        if self.ledger.has_completed_config(config_hash, "ensemble"):
            return {"status": "skipped_duplicate", "config_hash": config_hash}

        rec = ExperimentRecord(
            phase="ensemble", status="running", config_hash=config_hash,
            config_yaml=_config_yaml(config), git_sha=_git_sha(),
        )
        exp_id = self.ledger.start_experiment(rec)
        started = time.perf_counter()

        # Force submodel configs to share the promoted encoder architecture.
        arch = encoder_arch_from_best(self.ledger, fallback_config=submodel_configs[0])
        submodel_configs = [merge(c, arch) for c in submodel_configs]

        # Build val probabilities once (for stacking and weighted_vote).
        val_paths, val_labels = infer_adapter.videos_from_csv(
            splits.csv_path,
            video_dir=submodel_configs[0].get(
                "classify_video_dir", submodel_configs[0].get("video_dir")
            ),
            indices=splits.val_indices,
        )

        num_eval_clips = int(config.get("num_eval_clips", 1) or 1)
        infer_out = infer_adapter.infer_with_models(
            model_paths=list(submodel_paths),
            configs=list(submodel_configs),
            num_classes=splits.num_classes,
            video_paths=val_paths,
            labels=val_labels,
            num_clips=num_eval_clips,
        )

        method = config.get("aggregation", "soft_vote")
        weights = infer_out["per_model_acc"] if method == "weighted_vote" else None

        agg = ensemble_adapter.aggregate(
            method=method,
            probs=infer_out["probs"],
            weights=weights,
            temperature=float(config.get("temperature", 1.0)),
            val_probs=infer_out["probs"] if method == "stacking" else None,
            val_labels=infer_out["labels"] if method == "stacking" else None,
            meta_learner=config.get("meta_learner", "logreg"),
        )
        val_acc = ensemble_adapter.accuracy(agg["preds"], infer_out["labels"])

        wallclock_s = time.perf_counter() - started
        self.ledger.finish_experiment(
            exp_id, status="completed", wallclock_s=wallclock_s,
            score=float(val_acc),
            metrics={
                "ensemble_val_acc": val_acc,
                "per_model_acc": infer_out["per_model_acc"],
                "aggregation": method,
                "n_models": len(submodel_paths),
                "num_eval_clips": num_eval_clips,
            },
        )
        return {
            "status": "completed",
            "experiment_id": exp_id,
            "config_hash": config_hash,
            "score": float(val_acc),
            "ensemble_classifier": agg.get("classifier"),  # None unless stacking
            "wallclock_s": wallclock_s,
        }

    # --- Phase 4 (final hypothesis evaluation) ---

    def run_phase4(
        self,
        ensemble_config: Dict[str, Any],
        ensemble_submodel_paths: Sequence[str],
        ensemble_submodel_configs: Sequence[Dict[str, Any]],
        general_model_path: str,
        general_config: Dict[str, Any],
        splits: Splits,
        iter_id: int,
        ensemble_classifier: Any = None,
    ) -> Dict[str, Any]:
        os.environ["AUTORESEARCH_PHASE"] = "4"
        try:
            from autoresearch.prepare import load_holdout_indices
            holdout_idx = load_holdout_indices(splits)

            # Architecture must match the saved checkpoints (encoder + classifier
            # heads were trained with the promoted pretrain dims).
            arch = encoder_arch_from_best(self.ledger, fallback_config=general_config)
            general_config = merge(dict(general_config), arch)
            ensemble_submodel_configs = [merge(dict(c), arch) for c in ensemble_submodel_configs]

            video_dir = general_config.get(
                "classify_video_dir", general_config.get("video_dir")
            )
            paths, labels = infer_adapter.videos_from_csv(
                splits.csv_path, video_dir=video_dir, indices=holdout_idx
            )

            num_eval_clips = int(ensemble_config.get("num_eval_clips", 1) or 1)
            # Ensemble inference.
            ens_out = infer_adapter.infer_with_models(
                model_paths=list(ensemble_submodel_paths),
                configs=list(ensemble_submodel_configs),
                num_classes=splits.num_classes,
                video_paths=paths,
                labels=labels,
                num_clips=num_eval_clips,
            )
            method = ensemble_config.get("aggregation", "soft_vote")
            if method == "stacking":
                # If Phase 3 didn't already train a meta-learner (or it wasn't
                # propagated through the callback), fit one now on the val
                # split. The submodels haven't seen val during training so it
                # is a clean source for the stacker.
                if ensemble_classifier is None:
                    val_paths, val_labels = infer_adapter.videos_from_csv(
                        splits.csv_path, video_dir=video_dir, indices=splits.val_indices
                    )
                    val_out = infer_adapter.infer_with_models(
                        model_paths=list(ensemble_submodel_paths),
                        configs=list(ensemble_submodel_configs),
                        num_classes=splits.num_classes,
                        video_paths=val_paths, labels=val_labels,
                        num_clips=num_eval_clips,
                    )
                    ensemble_classifier = ensemble_adapter.stacking_fit(
                        val_probs=val_out["probs"], val_labels=val_out["labels"],
                        meta_learner=ensemble_config.get("meta_learner", "logreg"),
                    )
                ens_preds = ensemble_adapter.stacking_predict(ensemble_classifier, ens_out["probs"])
            else:
                weights = ens_out["per_model_acc"] if method == "weighted_vote" else None
                ens_preds = ensemble_adapter.aggregate(
                    method=method, probs=ens_out["probs"], weights=weights,
                    temperature=float(ensemble_config.get("temperature", 1.0)),
                )["preds"]

            # General model inference (use the same num_eval_clips for fair comparison).
            gen_out = infer_adapter.infer_with_models(
                model_paths=[general_model_path],
                configs=[general_config],
                num_classes=splits.num_classes,
                video_paths=paths,
                labels=labels,
                num_clips=num_eval_clips,
            )
            gen_preds = gen_out["preds"][0]

            stats = ensemble_adapter.bootstrap_gap_ci(
                ensemble_preds=ens_preds,
                general_preds=gen_preds,
                labels=labels,
            )
            stats["gap"] = fitness_gap(stats["ensemble_top1"], stats["general_top1"])
        finally:
            os.environ.pop("AUTORESEARCH_PHASE", None)

        rec = ExperimentRecord(
            phase="phase4", status="completed", config_hash=_config_hash(stats),
            config_yaml=_config_yaml({"phase4": stats}), git_sha=_git_sha(),
        )
        exp_id = self.ledger.start_experiment(rec)
        self.ledger.finish_experiment(
            exp_id, status="completed", wallclock_s=0.0,
            score=stats["gap"], metrics=stats,
        )
        return {"status": "completed", "experiment_id": exp_id, "metrics": stats}

    # --- helpers ---

    def _record_invalid(self, phase: str, config: Dict[str, Any], config_hash: str, errors: List[str]) -> Dict[str, Any]:
        rec = ExperimentRecord(
            phase=phase, status="invalid", config_hash=config_hash,
            config_yaml=_config_yaml(config), git_sha=_git_sha(),
            notes="validation_errors: " + "; ".join(errors),
        )
        exp_id = self.ledger.start_experiment(rec)
        self.ledger.finish_experiment(
            exp_id, status="invalid", wallclock_s=0.0, score=None,
            metrics={"validation_errors": errors},
        )
        return {"status": "invalid", "experiment_id": exp_id, "errors": errors}
