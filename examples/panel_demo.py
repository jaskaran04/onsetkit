"""Synthetic multi-asset labels, held-out event scores and weighted statistics."""
import numpy as np
from onsetkit import label_panel_onsets, panel_forward_split, stratified_sample, event_metrics

rng = np.random.default_rng(42)
n, assets = 500, 3
time = np.tile(np.arange(n), assets)
entity = np.repeat(['A','B','C'], n)
active = np.zeros(n * assets, dtype=int)
for i in range(assets):
    for t in range(40 + 3*i, n-10, 70):
        active[i*n+t:i*n+t+7] = 1
# Random order intentionally exercises entity/time sorting.
order = rng.permutation(len(time))
time, entity, active = time[order], entity[order], active[order]
r = label_panel_onsets(active, entity, time, horizon=5)
train, test = panel_forward_split(r.label_end, time, 350, 500, train_cutoff=340)
s = stratified_sample(r.labels[train], budget=100, allocation='equal', minimum=2, seed=42)
print(f'Training rows: {len(train)}; test rows: {len(test)}; sample: {len(s.indices)}')
print(f'Population prevalence: {r.labels[train].mean():.4%}')
print(f'Weighted prevalence: {s.estimate_mean(r.labels[train[s.indices]]):.4%}')
# A fixed synthetic attribute, not a learned model's in-sample error.
values = rng.normal(size=len(time))
print(f'Weighted attribute mean: {s.estimate_mean(values[train[s.indices]]):.4f}')
print(f'Estimated design standard error: {np.sqrt(s.estimate_variance(values[train[s.indices]])):.4f}')
# Random alarm baseline; no claim of predictive ability.
alarms = rng.random(len(time)) < 0.15
for asset in np.unique(entity):
    rows = np.flatnonzero(entity == asset)
    rows = rows[np.argsort(time[rows])]
    m = event_metrics(active[rows], alarms[rows], 5, evaluation_mask=np.isin(rows, test))
    print(f'{asset}: events {m.detected_events}/{m.eligible_events}; alarms {m.alarms}; false alarms {m.false_alarms}')
