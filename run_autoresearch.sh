#!/bin/bash
mkdir -p logs && TS=$(date +%Y%m%d_%H%M)
python -u -m autoresearch.search_loop --base-config configs/config.yaml --proposer llm --fallback heuristic --skip-pretrain --max-wallclock 4h --max-iters 20 --plateau-patience 8  --classify-timeout-per-submodel 30m 2>&1 | tee logs/search_loops_${TS}.log
