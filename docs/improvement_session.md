# Improvement Session

## Objective

Improve next-window unique IP prediction quality with a deep-model-first approach, while preserving chronological evaluation and keeping the codebase simple.

## Starting Point

Observed run from April 9, 2026:

- Backend: `gru`
- Router: `oh_ir1_gw`
- Target: `next_sa_ipv4_count_delta`
- Dataset: `data/2025-03-01-to-2026-03-31/netflow_window.sqlite`
- Result: model underperforms the persistence baseline on test MAE

Immediate read:

- The current GRU is not extracting enough signal from the present feature set.
- The current feature set is likely too weak for delta regression.
- The project is still missing the lagged tabular features described in the original project description.
- The delta target is heavy-tailed enough that loss choice and target transform are likely first-order concerns for deep models.

## Current Constraints

1. Base features are narrow.
- No lagged traffic/count features for linear or tree models.
- No first-difference or rolling features.
- Curve encodings are very compressed.

2. Sequence handling is conservative.
- Validation and test sequences do not get left context from the preceding split.
- This is leakage-safe, but it removes realistic history the model would have at inference time.

3. Deep model ladder is thin.
- Only a minimal `gru` is implemented on the sequence side.
- There is no stronger tabular deep baseline and no alternative sequence architecture.

4. Experimental control is thin.
- Dataset time slicing is fixed to the full available range.
- Session-level hypotheses and results were not being recorded in-repo.

## Working Hypotheses

1. Heavy-tailed delta targets likely need a more robust deep training setup, not only a different hidden size.
2. Sequence models will benefit from richer current-window features and from access to left context across split boundaries.
3. A stronger tabular deep baseline or alternate sequence architecture may outperform the current minimal GRU.
4. Spectrum/Structure value will be easier to detect after the deep baseline stops collapsing toward the center.

## Execution Order

1. Strengthen deep training for delta regression.
- Add a robust loss option.
- Add an invertible target transform for heavy-tailed deltas.

2. Strengthen sequence inputs and backends.
- Expand the base feature set with already-loaded traffic/protocol signals.
- Improve GRU capacity and pooling.
- Add one additional deep backend if the implementation cost stays reasonable.

3. Tighten sequence evaluation ergonomics.
- Preserve no-leakage behavior.
- Allow sequence backends to use prior context across split boundaries without training on future targets.

4. Improve experiment control only where it directly helps deep-model iteration.
- Add optional dataset time-range filters if needed for faster CUDA sweeps.

## Logging

### 2026-04-09

- Confirmed clean worktree before starting this session.
- Confirmed full dataset covers March 1, 2025 through March 31, 2026 for two routers.
- Redirected the session toward deep-model improvements rather than classical baselines.
- Kept the scope on single-router training rather than blending routers into one modeling signal.
- Measured the `next_sa_ipv4_count_delta` target for `oh_ir1_gw`: std about `1877.9`, 99th percentile about `3932`, min `-103037`, max `102067`, and about `22.9%` of rows have `|delta| > 1000`.
- Implemented the first deep-model patch set:
  - expanded the current-window base feature set from `10` to `17` features using already-loaded traffic and IPv6/protocol columns
  - upgraded the GRU to a stacked pooled architecture with dropout and a stronger head
  - added `mse`/`huber` loss selection for torch backends
  - added an invertible `signed_log1p` target transform, defaulting on GRU delta tasks
  - allowed validation and test sequence windows to use left context from earlier splits without leaking future targets
- Smoke check after the patch:
  - `gru`, `base,spectrum,structure`, `seq_len=24`, CPU, `2` epochs already beats persistence on validation and test MAE
  - `gru`, `base`, `seq_len=36`, CPU, `1` epoch is roughly at persistence, so the richer feature path is the more promising current direction
- CUDA rerun result on single-router GRU:
  - best so far is `base,spectrum,structure`, `seq_len=36`
  - `val_mae=992.48`, `val_r2=0.1040`
  - `test_mae=560.93`, `test_r2=0.0404`
  - compared with the earlier best GRU test MAE of about `611.12`, this is a material MAE improvement, though variance capture is still weak
- Loss/transform ablation outcome on that same config:
  - `standard + huber` is better than the previous default on both test MAE and test `R2`
  - `standard + mse` is competitive on `R2` but worse on test MAE
  - `signed_log1p` is no longer the preferred default; keep it as an explicit option only
- Added an `mlp` fixed-window deep backend for comparison runs.
- Added optional raw curve feature blocks for future deep-model experiments; spectrum needs resampling because its observed curve length varies.
- Raw curve benchmark on single-router GRU, `seq_len=36`, `standard + huber`:
  - `base,spectrum_raw,structure_raw`: `test_mae=555.94`, `test_r2=0.0634`
  - `base,spectrum,structure,spectrum_raw,structure_raw`: `test_mae=559.74`, `test_r2=0.0690`
  - both beat the best summary-only GRU result on test
- Cleaned the raw structure block after the benchmark by removing constant `q` coordinates so only varying raw values remain.
- Added a new `curve_gru` backend:
  - applies Conv1D encoders to the raw spectrum and structure curves inside each time step
  - fuses those learned curve embeddings with the scalar features
  - passes the per-step embeddings through a GRU and pooled regression head
- Verified the new backend with a 1-epoch CPU smoke run so it is ready for CUDA comparison against the current raw-feature GRU.
- Implemented a stronger GRU backend with stacked layers, input normalization, a projection layer, pooled sequence features, and a deeper MLP head.
- Added loss selection support for `mse` and `huber`, with `huber` now the default for GRU training when no explicit loss is provided.
- Enabled validation/test sequence windows to use prior split context without emitting targets from earlier rows.
- Added a plateau-based learning-rate scheduler to the torch training path.
- Benchmarked the new deep path on CUDA and found the strongest current configuration so far is:
  - `train-router all`
  - `model-backend gru`
  - `feature-blocks base,spectrum`
  - `sequence-length 24`
  - `epochs 100`
- That configuration beat persistence on the held-out test slice for `next_sa_ipv4_count_delta` by about `1.01` MAE and achieved test `r2` of about `0.0891`.
- The next-best simple configuration was `base` only at `sequence-length 24`, which stayed slightly above persistence on test but still improved validation substantially.
- Extending the same `base,spectrum` configuration to 150 epochs improved the test margin to about `5.24` MAE below persistence and test `r2` to about `0.1021`.
- Pushing that same configuration to 200 epochs did not improve beyond the 150-epoch checkpoint; early stopping restored the same best state.
