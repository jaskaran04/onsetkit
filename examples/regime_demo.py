"""Synthetic volatility-regime labels; no fitted forecasting model."""
import numpy as np
from onsetkit import label_onsets, purged_forward_split, stratified_sample

rng = np.random.default_rng(42)
n = 10000
active = np.zeros(n, dtype=int)
for start in range(200, n-30, 400):
    active[start:start+20] = 1
returns = rng.normal(0, np.where(active, .035, .008))
labels = label_onsets(active, horizon=5)
train, test = purged_forward_split(labels.label_end, 8000, n, gap=20)
y = labels.labels[train]
sample = stratified_sample(y, 200, allocation='equal', seed=42)
selected = train[sample.indices]
print(f'Eligible training rows: {len(train)}; test rows: {len(test)}')
print(f'Original onset prevalence: {y.mean():.4%}')
print(f'Sampled onset prevalence: {labels.labels[selected].mean():.4%}')
print(f'Weighted onset prevalence: {sample.estimate_mean(labels.labels[selected]):.4%}')
print(f'Stratum counts: {sample.counts}; allocation: {sample.allocation}')
assert np.isclose(sample.estimate_mean(labels.labels[selected]), y.mean())
assert max(labels.label_end[train]) < min(test)-20
