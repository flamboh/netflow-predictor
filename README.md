# NetFlow Predictor

This project contains simplified tabular and GRU baselines for predicting
next-window netflow targets from 5-minute traces.

Current modeling path:

- It reads the SQLite dataset in `data/netflow_window.sqlite`.
- It derives next-window count, delta, and direction targets.
- It uses a reduced feature set:
  base traffic/count features, cyclical time features, compact spectrum
  summaries, and compact structure summaries.
- It supports `linear`, `xgboost`, and `gru` backends.

Run the baseline with `uv`:

```bash
uv run python -m src.main
```

Run the GRU baseline:

```bash
uv run python -m src.main \
  --model-backend gru \
  --sequence-length 12 \
  --feature-blocks base,spectrum,structure
```

Request a prediction for one held-out 5-minute trace:

```bash
uv run python -m src.main \
  --router cc_ir1_gw \
  --timestamp 1748459100
```
