# CrunchDAO Equity Market Neutral #2

Participation in the [CrunchDAO DataCrunch #2](https://hub.crunchdao.com/competitions/datacrunch-2) 
competition — predicting expected returns for the 3000 most liquid US equities.

## Context
Personal project built while learning ML applied to finance. First end-to-end competition submission: data exploration, 
feature selection, model design, local scoring and deployment via the CrunchDAO CLI.

## Competition
Each "moon" represents a time period. For each moon, ~3000 stocks are observed with 1150 
anonymized features. The target is ternary: -1 (strong negative move), 
0 (neutral), +1 (strong positive move). The metric is mean Pearson correlation between 
predictions and targets, averaged across moons.

## Approach
**Data**: 1150 anonymized features, normalized in [0, 1], no missing values, no categorical 
variables — no imputation, encoding or scaling needed.

**Feature selection**: Spearman correlation between each feature and the target, keeping 
the top 500 out of 1150. Spearman is preferred over Pearson here because it captures monotonic relationships without assuming linearity.

**Model**: Two-stage pipeline:
1. **LogisticRegression** (`C=0.5`) — classifies each stock as mover (±1) vs non-mover (0)
2. **Ridge regression** (`alpha=100`) — trained only on movers, predicts direction (+1 or -1)

Final prediction: `proba_extreme × direction`, clipped to [-1, 1]

## Results


## Structure
\```
notebook.ipynb   # exploration, model, local scoring
main.py          # entry point for CrunchDAO submission
requirements.txt # dependencies
\```
