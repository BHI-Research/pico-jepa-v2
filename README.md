# pico-JEPA: Self-Supervised Video Representation Learning and Classification

This project implements a minimal version of a Joint-Embedding Predictive Architecture (JEPA) for self-supervised learning from videos, followed by a classification stage to evaluate the learned representations. It uses `torchcodec` for efficient video data loading and includes an ensemble method where multiple 'tiny' models vote on the final classification, often leading to improved accuracy and robustness.

Main Idea: Instead of a single giant model, we propose using an ensemble of ultra-lightweight models.

• An architecture designed to be a "base unit" for ensembles, small enough to enable collaborative architectures.

• Central Hypothesis: A set of pico-JEPA models, trained on portions of data, can outperform a single, larger model trained on all the data.


## Publication

This work is part of a publication. If you use this code, please cite our paper:

```bash
@inproceedings{rostagno2025pico,
  title = {pico-JEPA: Comprendiendo el Video con Modelos Ultra-Ligeros y la Sabiduría Colectiva},
  author = {Rostagno, Adrián and Iparraguirre, Javier and Friedrich, Guillermo and Aggio, Santiago and Briatore, Roberto and Tobio, Lucas and Coca, Diego},
  booktitle = {XXXI Congreso Argentino de Ciencias de la Computación (CACIC)},
  year = {2025},
  address = {Viedma, Argentina},
  month = {October},
  note = {6-10 de Octubre}
}
```

## Project Structure

- `app/`: Core training/inference scripts.
  - `train.py`: Self-supervised JEPA pre-training of the video encoder.
  - `classify_videos.py`: Supervised classifier on top of the pre-trained encoder (one submodel per call).
  - `infer_video.py`: Inference on a single video.
- `models/`: PyTorch model definitions (`backbone.py`, `pico_jepa.py`, `video_classifier.py`).
- `dataset/`: `VideoDataset` for data loading (torchcodec, CTHW clips).
- `configs/config.yaml`: Single source-of-truth config (encoder, pretrain and classify hyperparameters).
- `autoresearch/`: The automated search system that drives the full experiment (see [autoresearch/README.md](autoresearch/README.md)).
  - `search_loop.py`: Main CLI — iterates the four phases (pretrain → classify → ensemble → Phase 4 holdout eval).
  - `runner.py`, `ledger.py`, `ratchet.py`, `budget.py`, `prepare.py`: orchestration, SQLite ledger, artifact promotion, budget, protected splits.
  - `proposers/`: `heuristic.py` and `llm.py` (Anthropic) hyperparameter proposers.
  - `adapters/`: wrappers around pretrain/classify/infer/ensemble.
  - `run_phase4.py`, `compare_aggregations.py`, `report.py`: standalone CLIs for hypothesis eval, aggregation benchmark and terminal reports.
- `prepare_pretrain_subset.py` / `prepare_classify_subset.py`: build the K700 subsets (pretrain / labeled classify CSVs).
- `run_autoresearch.sh`: convenience launcher for the search loop.
- `download_k700.sh`, `download_test_dataset.sh`: dataset download helpers.
- `requirements.txt`: Python package dependencies.

---

## Environment Setup (Ubuntu 24.04)

These instructions will guide you through creating a Conda environment and installing the necessary packages.

1. **Create and Activate Conda Environment:**
    Open your terminal and run:

    ```bash
    # Create a new Conda environment named 'pico-jepa' with Python 3.12
    conda create -n pico-jepa python=3.12 -y

    # Activate the newly created environment
    conda activate pico-jepa
    ```

2. **Install PyTorch:**
    Choose one of the following commands based on your hardware:

    - **For systems with a compatible NVIDIA GPU (CUDA support):**

        ```bash
        pip3 install torch torchvision torchaudio
        ```

        *(Note: For specific CUDA versions, visit the [PyTorch official website](https://pytorch.org/get-started/locally/) for the correct command.)*

    - **For CPU-only systems (no NVIDIA GPU):**

        ```bash
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ```

3. **Install FFmpeg for `torchcodec`:**
    `torchcodec` relies on FFmpeg. Install a compatible version (e.g., version 6) from `conda-forge` into your environment.

    ```bash
    conda install ffmpeg=6 -c conda-forge -y
    ```

4. **Install `torchcodec`:**
    After FFmpeg is set up, install `torchcodec`.

    ```bash
    pip uninstall torchcodec -y # Uninstall previous attempts if any
    pip install torchcodec --no-cache-dir
    ```

    *(Note: Building `torchcodec` from source might require development tools like CMake, Ninja, a C++ compiler, and NASM. Consult the [official `torchcodec` GitHub](https://github.com/pytorch/torchcodec) if you encounter build issues.)*

5. **Install Remaining Requirements:**

    ```bash
    pip install -r requirements.txt
    ```
---

## How to Run (Full K700 Dataset)

The experiment is driven end-to-end by the **autoresearch** search loop, which
proposes hyperparameters and iterates four phases: pre-training, classification
(the ensemble submodels + a general baseline), ensemble aggregation, and a
Phase-4 evaluation on a protected holdout. See [EJEMPLO.md](EJEMPLO.md) for the
full walk-through; the minimal path is below.

**0. Download the dataset and build the subsets:**

```bash
sh download_k700.sh
python prepare_pretrain_subset.py --k700_dir /dataset/k700-2020/train \
    --num_clusters 10 --videos_per_class 300 --diversity_sample 400
python prepare_classify_subset.py        # builds the labeled classify CSV
```

**1. Self-supervised pre-training** of the encoder (JEPA):

```bash
python app/train.py --config_path configs/config.yaml \
    --output_json logs/pretrain.json
# Promote the encoder for the search loop:
mkdir -p models_best && cp pico_jepa_pretrained_encoder.pth models_best/pretrain.pth
```

**2. Run the search loop** (classify + ensemble + Phase 4 over the promoted
encoder). Uses the LLM proposer with a heuristic fallback:

```bash
sh run_autoresearch.sh
# or directly:
python -m autoresearch.search_loop --base-config configs/config.yaml \
    --proposer llm --fallback heuristic --skip-pretrain \
    --max-wallclock 4h --max-iters 20 --plateau-patience 8 \
    --classify-timeout-per-submodel 30m
```

**3. Inspect results / test the hypothesis** on the holdout:

```bash
python -m autoresearch.report --hypothesis          # gap table + verdict
python -m autoresearch.run_phase4 --reuse-general    # force Phase 4 with current artifacts
python -m autoresearch.compare_aggregations --num-models 5 --num-eval-clips 10
```

The official metric is `gap = ensemble_top1_holdout − general_top1_holdout`; the
hypothesis is **supported** when the bootstrap 95% CI excludes zero. The holdout
is loaded only inside `autoresearch/prepare.py` under a phase guard, so the
search never sees it.
