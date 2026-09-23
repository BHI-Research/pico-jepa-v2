# pico-JEPA v2: Self-Supervised Video Representation Learning and Classification

A minimal Joint-Embedding Predictive Architecture (JEPA) for self-supervised
learning from videos, followed by a supervised classification stage to evaluate
the learned representations. Video loading uses `torchcodec` for efficient CTHW
clip decoding.

**Main idea:** instead of one large model, use an *ensemble of ultra-lightweight
models*, each trained on a portion of the data. The central hypothesis is that a
set of pico-JEPA models can outperform a single, larger model trained on all the
data.

This is the v2 of the project: an automated research system (`autoresearch/`)
drives the full experiment end-to-end and evaluates the hypothesis on a
protected holdout.

## Publication

This work is presented at **CACIC 2026** — the XXXII Congreso Argentino de
Ciencias de la Computación, organized by [RedUNCI](https://redunci.info.unlp.edu.ar/)
and hosted by UTN Facultad Regional Concepción del Uruguay (Entre Ríos,
Argentina), [5–9 October 2026](https://www.frcu.utn.edu.ar/index.php/cacic-2026).
If you use this code, please cite:

```bibtex
@inproceedings{rostagno2026picov2,
  title     = {pico-JEPA v2: ¿Cuándo Superan Muchos Modelos Pequeños a Uno Grande?},
  author    = {Rostagno, Adrián and Iparraguirre, Javier and Briatore, Roberto and González, Agustín and Aggio, Santiago},
  booktitle = {XXXII Congreso Argentino de Ciencias de la Computación (CACIC)},
  year      = {2026},
  address   = {Concepción del Uruguay, Entre Ríos, Argentina},
  month     = {October}
}
```

The manuscript lives under [paper/](paper/) and the conference talk under
[slides/](slides/).

## Project Structure

- `app/` — core scripts: `train.py` (JEPA pre-training), `classify_videos.py`
  (supervised classifier per submodel), `infer_video.py` (single-video inference).
- `models/` — PyTorch definitions: `backbone.py`, `pico_jepa.py`, `video_classifier.py`.
- `dataset/` — `VideoDataset` for loading CTHW clips via torchcodec.
- `configs/config.yaml` — single source-of-truth config (encoder + pretrain + classify hyperparameters).
- `autoresearch/` — automated search system that runs the full experiment; see
  its own [README](autoresearch/README.md). It iterates four phases (pretrain →
  classify → ensemble → Phase-4 holdout eval) via `search_loop.py`, backed by an
  SQLite ledger, artifact promotion, budget control, and heuristic/LLM proposers.
- `prepare_pretrain_subset.py` / `prepare_classify_subset.py` — build the K700 subsets.
- `download_k700.sh` — dataset download helper.
- `run_autoresearch.sh` — convenience launcher for the search loop.
- `requirements.txt` — Python dependencies.

---

## Environment Setup (Ubuntu 24.04)

1. **Create and activate a Conda environment:**

    ```bash
    conda create -n pico-jepa-v2 python=3.12 -y
    conda activate pico-jepa-v2
    ```

2. **Install PyTorch** (pick one):

    ```bash
    # NVIDIA GPU (CUDA). For a specific CUDA version, see https://pytorch.org/get-started/locally/
    pip3 install torch torchvision torchaudio

    # CPU-only
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

3. **Install FFmpeg for `torchcodec`:**

    ```bash
    conda install ffmpeg=6 -c conda-forge -y
    ```

4. **Install `torchcodec`:**

    ```bash
    pip install torchcodec --no-cache-dir
    ```

    *(Building from source may require CMake, Ninja, a C++ compiler, and NASM —
    see the [torchcodec repo](https://github.com/pytorch/torchcodec).)*

5. **Install the remaining requirements:**

    ```bash
    pip install -r requirements.txt
    ```

---

## How to Run (Full K700 Dataset)

The experiment is driven end-to-end by the **autoresearch** search loop, which
proposes hyperparameters and iterates four phases: pre-training, classification
(ensemble submodels + a general baseline), ensemble aggregation, and a Phase-4
evaluation on a protected holdout.

**0. Download the dataset and build the subsets:**

```bash
sh k700_2020_downloader.sh
sh k700_2020_extractor.sh
python prepare_pretrain_subset.py --k700_dir /dataset/k700-2020/train \
    --num_clusters 10 --videos_per_class 300 --diversity_sample 400
python prepare_classify_subset.py        # builds the labeled classify CSV
```

**1. Self-supervised pre-training of the encoder (JEPA):**

```bash
python app/train.py --config_path configs/config.yaml \
    --output_json logs/pretrain.json
# Promote the encoder for the search loop:
mkdir -p models_best && cp pico_jepa_pretrained_encoder.pth models_best/pretrain.pth
```

**2. Run the search loop** (classify + ensemble + Phase 4 over the promoted
encoder; LLM proposer with a heuristic fallback):

```bash
sh run_autoresearch.sh
# or directly:
python -m autoresearch.search_loop --base-config configs/config.yaml \
    --proposer llm --fallback heuristic --skip-pretrain \
    --max-wallclock 4h --max-iters 20 --plateau-patience 8 \
    --classify-timeout-per-submodel 30m
```

**3. Inspect results / test the hypothesis on the holdout:**

```bash
python -m autoresearch.report --hypothesis          # gap table + verdict
python -m autoresearch.run_phase4 --reuse-general    # force Phase 4 with current artifacts
python -m autoresearch.compare_aggregations --num-models 5 --num-eval-clips 10
```

The official metric is `gap = ensemble_top1_holdout − general_top1_holdout`; the
hypothesis is **supported** when the bootstrap 95% CI excludes zero. The holdout
is loaded only inside `autoresearch/prepare.py` under a phase guard, so the
search never sees it.
