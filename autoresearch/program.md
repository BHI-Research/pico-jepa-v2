# Program (LOCKED — instructions for the LLM proposer)

This file is read by `proposers/llm.py` at the start of every iteration. It
defines the research goal, the metric, and the rules the proposer must
respect. Edits to this file are the user-facing way to redirect the search.
**The autoresearch system computes a SHA of this file at startup; if it
changes between resumes, the loop refuses to continue without explicit
confirmation.**

## Hypothesis under test

An ensemble of small **pico-JEPA** classifiers, trained on **partitions** of
the data, can outperform a **single larger** model trained on **all** the
data, on a held-out test set of K700 video clips.

## Official metric (LOCKED in `prepare.py`)

`gap = ensemble_top1_holdout − general_top1_holdout`

Positive gap supports the hypothesis. Bootstrap CI excluding zero is required
for a strong claim.

## What the proposer may change

Per phase, only the keys declared in `search_space.py` are accepted. Anything
else is ignored.

- **Phase 1 (pretrain)**: `learning_rate`, `predictor_lr_multiplier`, `mask_ratio`,
  `ema_decay`, `weight_decay`, `vit_embed_dim`, `vit_depth`, `vit_num_heads`,
  `predictor_depth`, `predictor_heads`, `vit_mlp_ratio`, `batch_size`,
  `num_epochs`, `frames_per_clip`,
  `masking_strategy` (`tubelet` | `multiblock`), `num_mask_blocks`,
  `ema_decay_end` (cosine schedule endpoint), `layerwise_lr_decay`.
- **Phase 2 (classify)**: `num_models`, `partition_strategy`, `freeze_encoder`,
  `learning_rate_classifier`, `learning_rate_encoder_finetune`,
  `num_epochs_classify`, `batch_size_classify`, `classify_weight_decay`,
  `head_type` (`linear` | `mlp_2layer` | `attentive`),
  `head_attn_heads`, `head_mlp_ratio`.
- **Phase 3 (ensemble)**: `aggregation`, `meta_learner`, `temperature`,
  `num_eval_clips` (V-JEPA-style multi-clip averaging at inference time).
- **Phase 4** is automatic: the loop runs it after every Phase 3 improvement.

## Hard constraints

1. `vit_embed_dim` MUST be divisible by `vit_num_heads`.
2. Stay within the declared ranges in `search_space.py`.
3. Do NOT propose configurations that are nearly identical to recent OOM
   failures — the runner penalizes them.
4. Mutate 1-3 keys per delta unless a clear case for a wider exploratory
   step exists (e.g., long plateau).

## Hardware envelope

Intel i7 / 65 GB RAM / GTX 1060 6 GB VRAM. The 1060 is the bottleneck.
Configurations above ~`embed_dim*depth*batch*frames > 1.6 × current_best`
are rejected by the heuristic in `search_space.estimated_oom`.

## Output format

**JSON only. No prose, no markdown fences.** Example:

```json
{"learning_rate": 0.0003, "mask_ratio": 0.6}
```

The proposer's response is parsed strictly. Any extra text is discarded.

## Strategy hints (optional)

- Pretrain is expensive on this GPU (~10× a classify step). Don't propose
  pretrain mutations if Phase 2 hasn't converged on the current encoder.
- `head_type="attentive"` historically gives the largest single-step gain
  over `linear` (~+10-17pp top-1 in V-JEPA). Worth trying first when classify
  scores plateau.
- `num_eval_clips=10` adds ~+2-4pp at 10× inference cost — fine for Phase 3/4.
- `masking_strategy="multiblock"` with `num_mask_blocks=2-4` regularizes
  pretrain on small datasets. Pair with `mask_ratio` 0.7-0.85.
- `ema_decay_end=0.9999` (with start=0.996) gives a gentle anneal that helps
  when batch size is small (1060 = batch ≤ 12).
- `layerwise_lr_decay=0.85-0.95` reduces overfitting in early encoder layers.
- Stacking generally beats voting once `num_models >= 4`. Worth trying after
  classify stabilizes.
- `partition_strategy="bagging"` plus weighted voting tends to be a robust
  baseline. `disjoint` is more aggressive but high variance.
- If Phase 4 gap is positive but its CI includes zero, propose more classify
  iters instead of more pretrain iters.
- If `pretrain` linear probe stays below 0.15 for many iters, the bottleneck
  is encoder capacity, not classify/ensemble. Prefer increasing `num_epochs`
  (16-32) over deeper search in classify.
- **Two thresholds gate the central hypothesis**. The ensemble only beats the general
  when BOTH hold; in either degraded regime the submodels converge to the
  general's functional space and `gap` collapses to ~0 (REJECTED):
  1. **Encoder capacity** `vit_embed_dim >= 192`. With `embed_dim=128` the
     stacking gap collapses to +0.00 (REJECTED). **Never propose `vit_embed_dim`
     below 192**, even if a smaller encoder gives a higher probe — higher probe
     does NOT translate into a positive Phase 4 gap.
  2. **Class breadth** (~30 classes, not ~10). A 10-class ablation (embed=192,
     holdout=200) gave `gap=-0.015` (REJECTED, conclusive). The ensemble edge
     scales with the number of classes: few classes leave no room for submodels
     to specialize on complementary class partitions. Do not assume results
     transfer across very different `num_classes`.
- `freeze_encoder=false` with `learning_rate_encoder_finetune ∈ [1e-6, 1e-5]`
  is the highest-leverage classify knob when the probe is below 0.20. Try it
  before exploring head_type variations.
- `num_eval_clips` is paid twice per successful iteration (Phase 3 val + Phase
  4 holdout). Picking 10 is fine; picking 30+ rarely helps and doubles cost.
- When ensemble accuracy collapses to the same value as one submodel (the
  symptom: `hard_vote == soft_vote` to the decimal), submodels lack diversity.
  Set ``diversify_submodels: true`` in the base config — each submodel then
  gets a distinct seed, learning rate (±30% log-jitter), head_type (rotation
  through linear/mlp_2layer/attentive), freeze_encoder toggle, and
  num_epochs_classify (±2). Without diversity the ensemble cannot beat a
  single model trained on the full data.
