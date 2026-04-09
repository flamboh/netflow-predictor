# NetFlow Predictor

This project contains tabular, MLP, GRU, and curve-aware GRU baselines for predicting
next-window netflow targets from 5-minute traces.

Current modeling path:

- It reads the SQLite dataset in
  `data/2025-03-01-to-2026-03-31/netflow_window.sqlite`.
- By default it trains on `oh_ir1_gw` only.
- It derives next-window count, delta, and direction targets.
- It uses a reduced feature set:
  base traffic/count features, cyclical time features, compact spectrum
  summaries, compact structure summaries, and optional raw curve blocks.
- It supports `linear`, `xgboost`, `gru`, `mlp`, and `curve_gru` backends.
- For neural delta runs, the current default training setup uses `huber`
  loss with a standard target scale.

Run the baseline with `uv`:

```bash
uv run python -m src.main
```

Train on all routers:

```bash
uv run python -m src.main --train-router all
```

Run the GRU baseline:

```bash
uv run python -m src.main \
  --model-backend gru \
  --sequence-length 12 \
  --feature-blocks base,spectrum,structure
```

Run the fixed-window MLP backend:

```bash
uv run python -m src.main \
  --model-backend mlp \
  --sequence-length 36 \
  --feature-blocks base,spectrum,structure
```

Try raw curve blocks:

```bash
uv run python -m src.main \
  --model-backend gru \
  --sequence-length 36 \
  --feature-blocks base,spectrum_raw,structure_raw
```

Run the Conv1D curve encoder plus GRU backend:

```bash
uv run python -m src.main \
  --model-backend curve_gru \
  --sequence-length 36 \
  --feature-blocks base,spectrum_raw,structure_raw
```

Request a prediction for one held-out 5-minute trace:

```bash
uv run python -m src.main \
  --router cc_ir1_gw \
  --timestamp 1748459100
```
