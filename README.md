# FlowTS

Domain-agnostic latent flow matching engine for multivariate time series generation, forecasting, and imputation. VAE tokenizer + OT-conditional flow matching in latent space, with pluggable denoising backbones (DiT, Mamba).

## Setup

```bash
uv venv
uv sync --extra dev
```

## Data

Synthetic multivariate Heston paths (SDE-simulated, ground-truth-controlled). A real-world benchmark (ETT/Electricity/Traffic/OHLCV) will be added as the primary real-data evaluation.

```bash
uv run python scripts/generate_synthetic.py   # -> data/raw/heston_{train,val,test}.npz
uv run python scripts/process_data.py         # -> data/processed/heston_{train,val,test}.npz (log-returns)
```
