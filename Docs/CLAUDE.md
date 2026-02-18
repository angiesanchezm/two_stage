# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two-stage sign language generation framework: generates 3D skeleton keypoints from gloss (sign language text) inputs. Uses a Transformer encoder + BiLSTM decoder architecture with adversarial training via a pose discriminator. Targets ECCV publication.

## Commands

### Training
```bash
python __main__.py CVT ./Configs/Base.yaml
```

### Inference
```bash
python __main__.py --ckpt {checkpoint_path} CVT_test ./Configs/Base.yaml
```

### Install Dependencies
```bash
pip install -r requirements.txt
```
Note: `requirements.txt` is incomplete. Actual dependencies include PyTorch, torchtext, numpy, opencv-python, scipy, pyyaml, and tensorboard.

## Architecture

### Two-Stage Pipeline
1. **Stage 1 - Gloss Encoding:** Input gloss tokens → linear embedding → Transformer encoder (2 layers, 8 heads, 512-dim)
2. **Stage 2 - Skeleton Generation:** Temporal convolution layers → target Transformer encoder → BiLSTM decoder (2 layers, bidirectional) → linear output to 151-dim skeleton (150 joints + 1 counter)

### Key Module Layout
- `__main__.py` — Entry point, dispatches to train (`CVT`) or test (`CVT_test`)
- `CVT/CVT_training.py` — Training manager: training loop, validation, checkpointing, TensorBoard logging
- `CVT/CVT_prediction.py` — Inference/validation logic
- `CVT/Conv_model.py` — Two-stage model definition with temporal convolution layers
- `model.py` — Base encoder-decoder model (builds encoder, decoder, embeddings, output layer)
- `encoders.py` — Transformer encoder
- `decoders.py` + `BiLSTM.py` — BiLSTM decoder
- `discriminator_Data.py` — Convolutional pose discriminator for adversarial training
- `loss.py` — RegLoss (MSE/L1 with masking), HuberLoss, XentLoss
- `data_operate/dataset.py` — PyTorch dataset, batch collation, padding for variable-length sequences
- `data_operate/vocab_dataset.py` — Vocabulary handling
- `helpers.py` — Config loading, checkpoint management, DTW computation, logging, mask generation
- `plot_videos.py` — Skeleton-to-video rendering with OpenCV

### Configuration
All hyperparameters are in `Configs/Base.yaml`. Key settings:
- `model.trg_size`: skeleton joint count (150 for 50 joints × 3 coords)
- `training.validation_freq`: steps between validations (default 10000)
- `training.early_stopping_metric`: uses DTW (Dynamic Time Warping)
- `training.model_dir`: where checkpoints are saved

### Data Format
- `.gloss` files: one gloss sentence per line (whitespace-separated tokens)
- `.skels` files: space-separated floats, one sequence per line. Values are joint coordinates divided by 3 to normalize to [-1, 1]. Each frame = 150 floats (50 joints × 3D)
- `.files` files: one sequence identifier per line (matches line-by-line with .gloss and .skels)
- Data lives in `Data/german/{train,dev,test}.*`

### Data Conversion
`convert_data_sidd_twostage/convert_signjoey_to_twostage.py` converts SignJoey format (61 joints × 3D) to the two-stage format expected by this codebase.

### Evaluation
Primary metric is DTW (Dynamic Time Warping) distance between predicted and ground-truth skeleton sequences, implemented in `dtw.py` and `helpers.py`.
