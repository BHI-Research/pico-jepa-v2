# autoresearch

Automated hypothesis testing for **pico-JEPA**, inspired by
[karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Hypothesis

> An ensemble of small pico-JEPA models trained on data partitions can
> outperform a single larger model trained on all the data.

## What it does

Runs a budget-bounded search over the four pipeline phases — pretrain,
classify, ensemble, infer — proposing config changes, training real models,
and ratcheting the best results into git. After each Phase 3 improvement, it
automatically evaluates Phase 4: does the ensemble actually beat the general
model on the protected holdout?

The search space includes V-JEPA-inspired knobs out of the box:
- **Phase 1 (pretrain)**: multi-block tubelet masking, EMA cosine schedule
  (decay anneals from start to end), layer-wise LR decay over the encoder.
- **Phase 2 (classify)**: cross-attention "attentive probe" head (in addition
  to the legacy linear head), N-submodel ensembles with 4 partition strategies.
- **Phase 3 (ensemble)**: hard/soft/weighted vote, sklearn stacking, plus
  V-JEPA-style multi-clip averaging at inference (`num_eval_clips`).

Two proposers ship in the box:
- `--proposer heuristic` — random search with anti-OOM rules. No internet.
- `--proposer llm` — Anthropic Claude with prompt caching, reads
  `program.md` to bias toward high-leverage knobs. Falls back to heuristic
  on any failure when invoked with `--fallback heuristic`.

## Quick start

```bash
# 1. Smoke test (no GPU needed) — verifies the full module chain works.
python -m autoresearch.tests.smoke_test

# 2. Real run with a small budget (heuristic proposer, no API).
python -m autoresearch.search_loop \
    --base-config configs/config.yaml \
    --proposer heuristic \
    --max-wallclock 1h \
    --max-iters 5

# 3. Real run with the LLM proposer (Anthropic Claude).
export ANTHROPIC_API_KEY=...
python -m autoresearch.search_loop \
    --base-config configs/config.yaml \
    --proposer llm --fallback heuristic \
    --max-wallclock 8h --max-iters 100

# 4. Inspect the ledger.
python -m autoresearch.report --best
python -m autoresearch.report --hypothesis
python -m autoresearch.report --last 30 --phase classify

# 5. Run Phase 4 standalone with the currently promoted artifacts (skip the
#    search loop). Trains a "general" baseline once and compares vs ensemble.
python -m autoresearch.run_phase4 --aggregation soft_vote

# 6. Compare all four aggregation methods on the same holdout
#    (reuses inferences across methods — the four runs cost seconds total).
python -m autoresearch.compare_aggregations --reuse-general --num-eval-clips 10
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  search_loop.py            CLI: budget + bandit phase selection  │
│         │                                                        │
│         ├─→ proposers/{heuristic,llm}.py     config delta        │
│         │                                                        │
│         ├─→ runner.py    validate, materialize, time, ledger     │
│         │       │                                                │
│         │       ├─→ adapters/pretrain.py     subprocess train.py │
│         │       ├─→ adapters/classify.py     N submodels         │
│         │       ├─→ adapters/infer.py        batch inference     │
│         │       ├─→ adapters/ensemble.py     voting / stacking   │
│         │       └─→ metrics/jepa_probe.py    linear probe        │
│         │                                                        │
│         └─→ ratchet.py    promote artifacts, git tag             │
│                                                                  │
│  run_phase4.py             Standalone CLI: Phase 4 with current  │
│                            artifacts (no search loop required)   │
│  compare_aggregations.py   Bench all 4 aggregations on cached    │
│                            inferences; markdown report           │
│  report.py                 Terminal report CLI (--best,          │
│                            --hypothesis, --last N --phase X)     │
│                                                                  │
│  prepare.py (LOCKED)       triple split + official fitness       │
│  ledger.py                 SQLite: experiments / artifacts /     │
│                            llm_cache / ratchet                   │
│  budget.py                 wallclock + max-iters + plateau       │
│  search_space.py           per-phase ranges, OOM heuristics      │
└──────────────────────────────────────────────────────────────────┘
```

## Resume

Resume is automatic. Any `running` row left over from a killed process is
marked `interrupted` at startup; the loop re-reads the SQLite ledger and the
git tags `autoresearch/best/{phase}` to recover state. Configs whose
`config_hash` already appear as `completed` are skipped.
