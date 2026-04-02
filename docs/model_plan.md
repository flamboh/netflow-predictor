# Model Plan

## Goal

Primary question:
- do Spectrum/Structure features add predictive value for next-window unique IP count tasks

Secondary question:
- what model class captures that value most cleanly

Interpretation:
- this is mainly an ablation / validation project
- not mainly a deep-learning project

## Data Scope

Current working slice:
- single 10-week window

Practical extension:
- can likely extract 1+ year of data later
- useful if current slice too small/noisy for stable comparisons

Implication:
- start with the 10-week window for fast iteration
- expand only if needed for robustness or generalization checks

## Recommended Target Order

Start simple:
1. direction of change in next-window unique source/destination IP count
2. binned magnitude of change
3. exact next-window unique count
4. exact next-window delta

Reason:
- easier targets will tell us sooner whether Spectrum/Structure carry signal
- exact-count regression is higher-noise; bad first proof target

## Model Ladder

Use same chronological splits, same metrics, same feature sets across all stages.

1. Naive baselines
- predict next count = current count
- predict next delta = 0
- simple moving-average baseline

2. Linear baselines
- ridge / lasso / elastic-net style regression
- logistic regression for direction-of-change task
- Poisson or negative-binomial variant if count target behavior suggests it

3. Tree baselines
- gradient-boosted trees
- random forest as a sanity check, not necessarily final

4. Sequence-aware classical models
- linear models with lag stacks
- tree models with lag stacks
- optional AR/ARIMA-style comparison if useful

5. Deep learning
- only after strong non-DL baselines
- only justified if simpler models plateau or if raw curve shape appears important
- candidates: LSTM/RNN, Transformer, S4

## Feature-Set Ablations

Run all model stages against the same ablation matrix:

1. Base features only
- traffic totals
- protocol counts
- lagged traffic features
- lagged unique IP counts
- time features

2. Base + Spectrum

3. Base + Structure

4. Base + Spectrum + Structure

Goal:
- isolate incremental value from Spectrum/Structure
- avoid confounding feature-set changes with model-class changes

## Spectrum/Structure Encoding Order

Start with simpler encodings first:
1. compact summary statistics
2. fixed sampled points from each curve
3. PCA or other low-dimensional projection of curve vectors
4. raw pointwise encodings
5. learned sequence encodings inside DL models

Reason:
- if simple encodings already help, evidence for usefulness is stronger
- if only a deep model on raw curves helps, interpretation is weaker

## Evaluation

Need:
- chronological train/validation/test splits
- no leakage across future windows
- compare by task-appropriate metrics

Suggested metrics:
- direction task: accuracy, F1, ROC-AUC
- regression/count task: MAE, RMSE, R^2
- incremental value: delta vs base-feature baseline

## Decision Rule

Prefer the simplest model that shows stable incremental benefit from Spectrum/Structure.

Deep learning becomes justified only if:
- data volume increases materially
- simpler models saturate
- raw Spectrum/Structure shape seems to matter beyond compact encodings
