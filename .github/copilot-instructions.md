# GitHub Copilot Instructions for JaxFM

This repository implements a JAX-based training pipeline for Multimodal Diffusion Transformers (mmDiT), utilizing Equinox, Optax, and Grain.

## 🏗 Architecture Overview

- **Core Framework**: JAX + Equinox (neural networks) + Optax (optimization).
- **Configuration**: Hydra & OmegaConf. Main config is `configs/config.yaml`.
- **Data Loading**: `grain` (Google's data loader). Data is stored in `array_record` format.
- **Checkpointing**: `orbax.checkpoint`.
- **Logging**: `wandb`.
- **Models**:
  - `models/mmDiT`: Main diffusion transformer architecture.
  - `models/vae`: Variational Autoencoder for latent space operations.
  - `t5gemma`: Text encoder (frozen) for conditioning.

## 🛠 Developer Workflows

- **Training**:
  - Stage 1: `python train_stage1.py`
  - Stage 2: `python train_stage2.py`
  - Use `scripts/*.slurm` for cluster execution.
- **Data Preparation**:
  - Data loading logic is in `data/data.py`.
  - Uses `grain.ArrayRecordDataSource` and `grain.DataLoader`.
- **Environment**:
  - JAX with CUDA support.
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false` is typically set in scripts.

## 📝 Coding Conventions

- **Type Hinting**: Use `jaxtyping` extensively for array shapes and types.
  - Example: `Float[Array, "batch seq_len dim"]`
- **JAX/Equinox Patterns**:
  - Use `eqx.Module` for model components.
  - Use `eqx.filter_jit` for JIT compilation.
  - Explicitly handle PRNG keys (`jax.random.PRNGKey`).
  - Use `jax.sharding.NamedSharding` and `jax.sharding.Mesh` for distributed training.
- **State Management**:
  - Training state is typically a tuple: `(model, opt_state)`.
  - EMA (Exponential Moving Average) is used for model weights (`model_ema`).

## 🔌 Integration & Dependencies

- **WandB**: Initialize in `train_stage*.py` using `wandb.init`.
- **Orbax**: Use `ocp.CheckpointManager` for saving/restoring.
  - Checkpoints include `state`, `model_ema`, and `dataset` (Grain iterator).
- **Hydra**: Configs are composed from `configs/` directory. Use `hydra.utils.instantiate` to create objects from config.

## ⚠️ Critical Details

- **Sharding**: Pay attention to `model_sharding` and `data_sharding` in training loops. The mesh typically uses axis name "data".
- **Text Encoding**: `encode_with_t5gemma_encoder` in `train_stage2.py` handles text conditioning.
- **Data Pipeline**: `ParseRecordStage1` and `ParseRecordStage2` in `data/data.py` define how raw records are processed.
