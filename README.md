# FlowTS

Domain-agnostic latent flow matching engine for multivariate time series generation, forecasting, and imputation. VAE tokenizer + OT-conditional flow matching in latent space, with pluggable denoising backbones (DiT, Mamba).

## Status

- [x] Data pipeline: synthetic Heston SDE generator, train/val/test splits, log-return preprocessing, lazy-loading `TimeSeriesDataset`
- [ ] VAE (patchify + encode/decode, recon + KL training)
- [ ] DiT backbone + OT-CFM training loop
- [ ] torch.compile optimization study (AdaLN path)
- [ ] Mamba backbone
- [ ] Forecasting-as-inpainting adaptation
- [ ] LoRA adaptation
- [ ] Evaluation suite (accuracy, calibration, performance benchmarking)
- [ ] Failure analysis

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

## Architecture

*to-be-filled*

## Ablation results

*to-be-filled*

## Calibration

*to-be-filled*

## Failure analysis

*to-be-filled*