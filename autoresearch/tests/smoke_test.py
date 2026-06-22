"""Lightweight smoke test for the autoresearch system.

Verifies:
1. All modules import cleanly.
2. SQLite ledger can be initialized and round-trip an experiment.
3. Search-space sampler produces valid configs.
4. Heuristic proposer returns a non-empty delta.
5. ``prepare.make_splits`` produces a 60/20/20 stratified split when given a
   tiny synthetic CSV.
6. Holdout guard raises outside phase 4.
7. Ensemble aggregations agree on a contrived input.

Does NOT run any actual training (no GPU, no torchcodec, no real videos).
For end-to-end validation with real (tiny) training, use the
``DatasetPrueba`` from ``download_test_dataset.sh`` and run::

    python -m autoresearch.search_loop \\
        --base-config configs/config-test.yaml \\
        --max-iters 2 --max-wallclock 30m
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import List

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_imports() -> None:
    import autoresearch  # noqa: F401
    from autoresearch import budget, ledger, prepare, ratchet, runner, search_loop, search_space  # noqa: F401
    from autoresearch.adapters import classify, ensemble, infer, pretrain  # noqa: F401
    from autoresearch.metrics import jepa_probe  # noqa: F401
    from autoresearch.proposers import base, heuristic  # noqa: F401
    print("[smoke] imports: OK")


def test_ledger_roundtrip(tmpdir: str) -> None:
    from autoresearch.ledger import ExperimentRecord, Ledger
    path = os.path.join(tmpdir, "test_ledger.sqlite")
    ledger = Ledger(path)
    rec = ExperimentRecord(
        phase="pretrain", config_hash="abc123", config_yaml="key: 1",
    )
    exp_id = ledger.start_experiment(rec)
    ledger.finish_experiment(exp_id, status="completed", wallclock_s=12.3, score=0.5,
                             metrics={"final_loss": 0.1, "probe_top1": 0.5})
    rows = ledger.recent_history(limit=5)
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["score"] == 0.5
    ledger.set_ratchet("pretrain", exp_id, score=0.5)
    assert ledger.get_ratchet("pretrain")["current_score"] == 0.5
    print("[smoke] ledger: OK")


def test_search_space_sample() -> None:
    from autoresearch.search_space import SPACES, validate_config
    rng = np.random.default_rng(42)
    for phase, space in SPACES.items():
        cfg = {k: hp.sample(rng) for k, hp in space.items()}
        ok, errors = validate_config(phase, cfg)
        assert ok, f"sampled config for {phase} failed validation: {errors}"
    print("[smoke] search_space: OK")


def test_heuristic_proposer() -> None:
    from autoresearch.proposers.heuristic import HeuristicProposer
    base = {
        "vit_embed_dim": 192, "vit_depth": 8, "vit_num_heads": 8,
        "batch_size": 10, "frames_per_clip": 8, "mask_ratio": 0.75,
        "ema_decay": 0.996, "learning_rate": 2e-4, "weight_decay": 0.05,
        "predictor_lr_multiplier": 2.0, "predictor_depth": 2, "predictor_heads": 4,
        "vit_mlp_ratio": 3.0, "num_epochs": 4,
    }
    proposer = HeuristicProposer(seed=7)
    delta = proposer.propose("pretrain", base, history=[])
    assert delta and any(k in base for k in delta), "delta has no keys overlapping base"
    print(f"[smoke] heuristic proposer: OK (delta={list(delta.keys())})")


def test_prepare_splits(tmpdir: str) -> None:
    from autoresearch.prepare import HoldoutAccessError, load_holdout_indices, make_splits
    rng = np.random.default_rng(1337)
    rows = []
    for c in range(10):
        for j in range(20):
            rows.append({"video_name": f"class_{c}/v_{j}.mp4", "label": c})
    rng.shuffle(rows)
    csv_path = os.path.join(tmpdir, "classify.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    splits = make_splits(csv_path)
    n = len(rows)
    assert len(splits.train_indices) >= int(0.55 * n)
    assert len(splits.holdout_indices) >= int(0.15 * n)
    assert (
        len(set(splits.train_indices) & set(splits.val_indices)) == 0
    ), "train/val overlap"
    assert (
        len(set(splits.train_indices) & set(splits.holdout_indices)) == 0
    ), "train/holdout overlap"

    # Stratification: every class appears in every split.
    df = pd.read_csv(csv_path)
    for split_name, idx in [("train", splits.train_indices), ("val", splits.val_indices), ("holdout", splits.holdout_indices)]:
        labels_in_split = set(df.iloc[list(idx)]["label"].unique())
        assert labels_in_split == set(range(10)), f"{split_name} missing classes: {set(range(10)) - labels_in_split}"

    # Holdout guard.
    raised = False
    try:
        load_holdout_indices(splits)
    except HoldoutAccessError:
        raised = True
    assert raised, "holdout should raise outside phase 4"

    os.environ["AUTORESEARCH_PHASE"] = "4"
    try:
        idx = load_holdout_indices(splits)
        assert idx == splits.holdout_indices
    finally:
        os.environ.pop("AUTORESEARCH_PHASE", None)

    print(f"[smoke] prepare splits: OK (train={len(splits.train_indices)}, val={len(splits.val_indices)}, holdout={len(splits.holdout_indices)})")


def test_ensemble_aggregations() -> None:
    from autoresearch.adapters.ensemble import (
        accuracy, aggregate, bootstrap_gap_ci, hard_vote, soft_vote, weighted_vote,
    )
    # 3 models, 4 videos, 5 classes. Construct so true label = argmax of sum.
    np.random.seed(0)
    probs = np.random.dirichlet(alpha=np.ones(5), size=(3, 4)).astype(np.float32)
    labels = np.argmax(probs.mean(axis=0), axis=1)

    soft = soft_vote(probs)
    hard = hard_vote(probs)
    weighted = weighted_vote(probs, weights=[1.0, 1.0, 1.0])

    assert accuracy(soft, labels) >= 0.99, "soft vote should match by construction"
    assert soft.shape == (4,) and hard.shape == (4,) and weighted.shape == (4,)

    # Stacking with sklearn.
    val_probs = probs.copy()
    val_labels = labels
    out = aggregate("stacking", probs, val_probs=val_probs, val_labels=val_labels, meta_learner="logreg")
    assert out["preds"].shape == (4,)

    # Bootstrap CI on a contrived case where ensemble = labels and general is random.
    gen_preds = (labels + 1) % 5  # always wrong
    stats = bootstrap_gap_ci(soft, gen_preds, labels, n_boot=200, seed=0)
    assert stats["gap_mean"] > 0.5
    print("[smoke] ensemble: OK")


def test_budget_parsing() -> None:
    from autoresearch.budget import Budget, parse_wallclock
    assert parse_wallclock("1h") == 3600
    assert parse_wallclock("30m") == 1800
    assert parse_wallclock("90s") == 90
    b = Budget(max_wallclock_s=10, max_iters=2, plateau_patience=3)
    assert b.can_continue()
    b.record_iteration(False); b.record_iteration(False)
    assert not b.can_continue()  # max_iters hit
    print("[smoke] budget: OK")


def test_classifier_head_types() -> None:
    """Phase A: VideoClassifier supports linear, mlp_2layer, attentive heads."""
    import torch
    from models.video_classifier import VideoClassifier
    cfg = {
        "video_channels": 3, "frames_per_clip": 8, "resize_height": 224, "resize_width": 224,
        "vit_patch_size_t": 2, "vit_patch_size_h": 16, "vit_patch_size_w": 16,
        "vit_embed_dim": 96, "vit_depth": 2, "vit_num_heads": 4, "vit_mlp_ratio": 2.0,
        "vit_dropout": 0.0,
    }
    x = torch.randn(2, 3, 8, 224, 224)
    last_n = None
    for ht in ("linear", "mlp_2layer", "attentive"):
        cfg["head_type"] = ht
        m = VideoClassifier(cfg, num_classes=4, freeze_encoder=True)
        out = m(x)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert out.shape == (2, 4), f"{ht}: bad output shape {out.shape}"
        if last_n is not None:
            assert n > last_n, f"head_type={ht} should have more trainable params than the previous"
        last_n = n
    print("[smoke] classifier head types (linear/mlp_2layer/attentive): OK")


def test_search_space_phase_a_keys() -> None:
    """Phase A added head_type & num_eval_clips to the search space."""
    from autoresearch.search_space import CLASSIFY, ENSEMBLE, validate_config
    assert "head_type" in CLASSIFY and "head_attn_heads" in CLASSIFY
    assert "num_eval_clips" in ENSEMBLE
    ok, errs = validate_config("classify", {"head_type": "attentive"})
    assert ok, errs
    ok, errs = validate_config("ensemble", {"num_eval_clips": 10})
    assert ok, errs
    ok, errs = validate_config("classify", {"head_type": "nonsense"})
    assert not ok
    print("[smoke] search_space Phase A keys: OK")


def test_multiblock_masking() -> None:
    """Phase B: multiblock masks have uniform cardinality across batch."""
    import torch
    from utils.utils import generate_multiblock_masks, generate_spatiotemporal_masks
    H, W, T = 14, 14, 4
    HW = H * W
    K = int(0.75 * HW)
    for n_blocks in (1, 2, 3, 4):
        m = generate_multiblock_masks(
            num_patches_t=T, num_patches_h=H, num_patches_w=W,
            mask_ratio=0.75, num_blocks=n_blocks, device=torch.device("cpu"), batch_size=8,
        )
        assert m.shape == (8, T * HW), f"unexpected shape {m.shape}"
        per_sample = m.sum(dim=1)
        assert torch.all(per_sample == per_sample[0]), \
            f"non-uniform cardinality across batch for n_blocks={n_blocks}: {per_sample.tolist()}"
        assert per_sample[0].item() == K * T, \
            f"per-sample mask count {per_sample[0].item()} != expected {K * T}"
    # Single-block fallback delegates to the legacy function.
    m1 = generate_multiblock_masks(T, H, W, 0.75, 1, torch.device("cpu"), 4)
    m2 = generate_spatiotemporal_masks(T, H, W, 0.75, torch.device("cpu"), 4)
    assert m1.shape == m2.shape
    print("[smoke] multiblock masking: OK")


def test_diversify_submodel_config() -> None:
    """diversify_submodel_config produces deterministic, distinct, valid configs."""
    from autoresearch.adapters.classify import diversify_submodel_config
    from autoresearch.search_space import CLASSIFY

    base = {
        "freeze_encoder": True,
        "learning_rate_classifier": 1e-4,
        "num_epochs_classify": 4,
        "head_type": "attentive",
        "vit_embed_dim": 192,
        "num_classes": 30,
    }

    cfgs = [diversify_submodel_config(base, submodel_idx=i, master_seed=1337) for i in range(1, 5)]

    # Submodel 1 acts as a control — base preserved (apart from seed).
    assert cfgs[0]["learning_rate_classifier"] == base["learning_rate_classifier"]
    assert cfgs[0]["head_type"] == base["head_type"]
    assert cfgs[0]["freeze_encoder"] == base["freeze_encoder"]

    # Submodels 2-4 must each have a unique seed and diverge from base.
    seeds = {c["seed"] for c in cfgs}
    assert len(seeds) == 4, f"expected 4 unique seeds, got {len(seeds)}"

    # All lrs within the search space declared range [1e-5, 1e-3].
    lr_range = CLASSIFY["learning_rate_classifier"]
    for c in cfgs:
        lr = c["learning_rate_classifier"]
        assert lr_range.low <= lr <= lr_range.high, f"lr {lr} out of range"

    # head_type rotates through valid values and actually diverges (the whole
    # point: heterogeneous ensemble needs different architectures, not just
    # different seeds).
    valid_head_types = set(CLASSIFY["head_type"].values)
    for c in cfgs:
        assert c["head_type"] in valid_head_types, f"invalid head_type {c['head_type']}"
    # Submodels 2..4 should cover at least 2 distinct head_types (the rotation
    # cycles through (linear, mlp_2layer, attentive)).
    diversified_head_types = {c["head_type"] for c in cfgs[1:]}
    assert len(diversified_head_types) >= 2, (
        f"expected diversified head_types across submodels 2-4, got {diversified_head_types}"
    )

    # Determinism: same input gives same output.
    cfg_repeat = diversify_submodel_config(base, submodel_idx=3, master_seed=1337)
    assert cfg_repeat == cfgs[2], "non-deterministic output"
    print(f"[smoke] diversify_submodel_config (seeds={sorted(seeds)}): OK")


def test_heterogeneous_ensemble_loading(tmpdir: str) -> None:
    """3 VideoClassifier with different head_type each save+load their own
    state_dict without RuntimeError. Replicates the heterogeneous ensemble
    inference path: each submodel is reconstructed with its own arch before
    load_state_dict.
    """
    import torch
    from models.video_classifier import VideoClassifier

    base_cfg = {
        "video_channels": 3, "frames_per_clip": 8,
        "resize_height": 224, "resize_width": 224,
        "vit_patch_size_t": 2, "vit_patch_size_h": 16, "vit_patch_size_w": 16,
        "vit_embed_dim": 96, "vit_depth": 2, "vit_num_heads": 4, "vit_mlp_ratio": 2.0,
        "vit_dropout": 0.0,
    }

    saved = {}
    for head_type in ("linear", "mlp_2layer", "attentive"):
        cfg = {**base_cfg, "head_type": head_type, "head_attn_heads": 4, "head_mlp_ratio": 2.0}
        model = VideoClassifier(cfg, num_classes=4, freeze_encoder=True)
        path = os.path.join(tmpdir, f"clf_{head_type}.pth")
        torch.save(model.state_dict(), path)
        saved[head_type] = (path, cfg)

    # Now simulate the ensemble loader: each submodel reconstructed with its
    # config, then load_state_dict on the matching checkpoint.
    for head_type, (path, cfg) in saved.items():
        model = VideoClassifier(cfg, num_classes=4, freeze_encoder=True)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=True)  # would crash on mismatch

    # And confirm that mixing fails — wrong arch should raise. This protects
    # against future refactors that silently revert the per-submodel config flow.
    wrong_cfg = {**base_cfg, "head_type": "linear"}
    wrong_model = VideoClassifier(wrong_cfg, num_classes=4, freeze_encoder=True)
    mlp_state = torch.load(saved["mlp_2layer"][0], map_location="cpu")
    raised = False
    try:
        wrong_model.load_state_dict(mlp_state, strict=True)
    except RuntimeError:
        raised = True
    assert raised, "linear classifier should NOT accept mlp_2layer state_dict"
    print("[smoke] heterogeneous ensemble loading (linear/mlp_2layer/attentive): OK")


def test_classify_promotion_sanitization(tmpdir: str) -> None:
    """When the ratchet promotes a classify config, it must replace the
    submodel's partitioned CSV path with the original full-dataset path.
    """
    import yaml as _yaml
    from autoresearch.search_loop import sanitize_classify_config_for_promotion

    base_config = {
        "classify_csv_path": "/dataset/k700-2020/val/classify_subset.csv",  # full
        "classify_video_dir": "/dataset/k700-2020/val/",
        "vit_embed_dim": 192,
    }
    submodel_cfg = {
        "classify_csv_path": "/tmp/iters/it_X/csvs/submodel_1_train.csv",  # partitioned
        "classify_video_dir": "/dataset/k700-2020/val/",
        "num_models": 4,
        "head_type": "attentive",
    }
    submodel_path = os.path.join(tmpdir, "submodel_1_config.yaml")
    with open(submodel_path, "w") as f:
        _yaml.safe_dump(submodel_cfg, f)

    promoted = sanitize_classify_config_for_promotion(
        submodel_config_path=submodel_path, base_config=base_config, work_dir=tmpdir,
    )
    promoted_cfg = _yaml.safe_load(open(promoted))
    assert promoted_cfg["classify_csv_path"] == base_config["classify_csv_path"], (
        f"classify_csv_path not restored to base: {promoted_cfg['classify_csv_path']}"
    )
    assert promoted_cfg["num_models"] == 4, "non-CSV keys must be preserved"
    assert promoted_cfg["head_type"] == "attentive", "non-CSV keys must be preserved"
    print("[smoke] classify config sanitization on promotion: OK")


def test_encoder_arch_propagation(tmpdir: str) -> None:
    """Architecture keys (vit_embed_dim etc.) flow from pretrain config best
    into classify configs even when the base/classify config disagrees.
    Without this, load_state_dict gets a shape mismatch and the classifier
    silently trains on a random encoder.
    """
    from autoresearch.ledger import ExperimentRecord, Ledger
    from autoresearch.runner import ENCODER_ARCH_KEYS, encoder_arch_from_best
    import yaml as _yaml
    ledger = Ledger(os.path.join(tmpdir, "ledger.sqlite"))
    pretrain_yaml = _yaml.safe_dump({
        "vit_embed_dim": 256, "vit_depth": 8, "vit_num_heads": 8,
        "vit_mlp_ratio": 3.0, "vit_patch_size_t": 2, "vit_patch_size_h": 16,
        "vit_patch_size_w": 16, "frames_per_clip": 8, "resize_height": 224,
        "resize_width": 224, "video_channels": 3, "vit_dropout": 0.1,
    })
    rec = ExperimentRecord(phase="pretrain", config_hash="aaa", config_yaml=pretrain_yaml)
    exp_id = ledger.start_experiment(rec)
    ledger.finish_experiment(exp_id, status="completed", wallclock_s=1.0, score=0.5, metrics={})

    fallback = {"vit_embed_dim": 192, "vit_depth": 8, "vit_num_heads": 8}
    # Use tmpdir as project_root so we don't read the real configs/best/pretrain.yaml.
    arch = encoder_arch_from_best(ledger, fallback_config=fallback, project_root=tmpdir)
    assert arch["vit_embed_dim"] == 256, "should override base config from pretrain best"
    assert all(k in ENCODER_ARCH_KEYS for k in arch)
    print("[smoke] encoder arch propagation: OK")


def test_pretrain_forward_multiblock() -> None:
    """End-to-end: a tiny PicoJEPA with multiblock masking does a forward pass."""
    import torch
    from models.pico_jepa import PicoJEPA_Pretrain
    cfg = {
        "video_channels": 3, "frames_per_clip": 4,
        "resize_height": 64, "resize_width": 64,
        "vit_patch_size_t": 2, "vit_patch_size_h": 16, "vit_patch_size_w": 16,
        "vit_embed_dim": 64, "vit_depth": 2, "vit_num_heads": 4, "vit_mlp_ratio": 2.0,
        "vit_dropout": 0.0,
        "predictor_depth": 1, "predictor_heads": 4,
        "mask_ratio": 0.5,
        "ema_decay": 0.996, "ema_decay_start": 0.996, "ema_decay_end": 0.9999,
        "total_steps": 10,
        "masking_strategy": "multiblock", "num_mask_blocks": 2,
    }
    model = PicoJEPA_Pretrain(cfg)
    model.set_step(0, total_steps=10)
    videos = torch.randn(2, 3, 4, 64, 64)
    loss = model(videos)
    assert loss.dim() == 0, f"loss should be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()
    print(f"[smoke] pretrain forward (multiblock, B=2): loss={loss.item():.4f} OK")


def test_ema_schedule_and_lwd() -> None:
    """Phase B: EMA cosine schedule + layer-wise LR decay shape correctly."""
    import sys, os, types, importlib
    import torch
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from models.pico_jepa import PicoJEPA_Pretrain
    cfg = {
        "video_channels": 3, "frames_per_clip": 8,
        "resize_height": 224, "resize_width": 224,
        "vit_patch_size_t": 2, "vit_patch_size_h": 16, "vit_patch_size_w": 16,
        "vit_embed_dim": 96, "vit_depth": 2, "vit_num_heads": 4, "vit_mlp_ratio": 2.0,
        "vit_dropout": 0.0,
        "predictor_depth": 1, "predictor_heads": 4,
        "mask_ratio": 0.75,
        "ema_decay": 0.996, "ema_decay_start": 0.996, "ema_decay_end": 0.9999,
        "total_steps": 100,
        "masking_strategy": "multiblock", "num_mask_blocks": 2,
    }
    model = PicoJEPA_Pretrain(cfg)
    model.set_step(0, total_steps=100)
    m_start = model._current_ema_decay()
    model.set_step(50)
    m_mid = model._current_ema_decay()
    model.set_step(100)
    m_end = model._current_ema_decay()
    assert abs(m_start - 0.996) < 1e-6, f"start should be 0.996 got {m_start}"
    assert abs(m_end - 0.9999) < 1e-6, f"end should be 0.9999 got {m_end}"
    assert 0.996 < m_mid < 0.9999, f"mid should be between endpoints got {m_mid}"

    # Layer-wise LR decay sanity (import only — the helper lives in app/train.py).
    from app.train import build_pretrain_param_groups
    groups = build_pretrain_param_groups(
        model=model, base_lr=2e-4, predictor_lr_multiplier=2.0,
        weight_decay=0.05, layerwise_lr_decay=0.9,
    )
    encoder_lrs = [g["lr"] for g in groups if g["lr"] < 2e-4]
    encoder_lrs_full = [g["lr"] for g in groups if g["lr"] <= 2e-4]
    assert any(lr < 2e-4 for lr in encoder_lrs_full), "expected at least one decayed LR"
    print(f"[smoke] EMA schedule (0.996->{m_mid:.5f}->{m_end:.5f}) + LWD: OK")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_imports()
        test_ledger_roundtrip(tmp)
        test_search_space_sample()
        test_heuristic_proposer()
        test_prepare_splits(tmp)
        test_ensemble_aggregations()
        test_budget_parsing()
        test_classifier_head_types()
        test_search_space_phase_a_keys()
        test_multiblock_masking()
        test_pretrain_forward_multiblock()
        test_ema_schedule_and_lwd()
        test_encoder_arch_propagation(tmp)
        test_classify_promotion_sanitization(tmp)
        test_diversify_submodel_config()
        test_heterogeneous_ensemble_loading(tmp)
    print("\n[smoke] all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
