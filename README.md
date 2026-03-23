# NetFlow Predictor

This project contains a simple PyTorch regression baseline for predicting
the total `bytes` seen in a 5-minute netflow trace.

The baseline intentionally stays small:

- It reads the SQLite dataset in `data/netflow_window.sqlite`.
- It predicts `netflow_stats.bytes`.
- It uses current trace counts, a few lag features, and small joined
  aggregates from `ip_stats` and `protocol_stats`.
- It uses one linear layer in PyTorch, so the model is easy to inspect.

Run the baseline with `uv`:

```bash
uv run python main.py
```

Request a prediction for one held-out 5-minute trace:

```bash
uv run python main.py \
  --router cc_ir1_gw \
  --timestamp 1748459100
```
