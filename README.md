# OnsetKit

[![PyPI](https://img.shields.io/pypi/v/onsetkit)](https://pypi.org/project/onsetkit/)
[![CI](https://github.com/jaskaran04/onsetkit/actions/workflows/ci.yml/badge.svg)](https://github.com/jaskaran04/onsetkit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/onsetkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jaskaran04/onsetkit/blob/main/LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)](https://pypi.org/project/onsetkit/)
[![Downloads](https://img.shields.io/pypi/dm/onsetkit)](https://pypi.org/project/onsetkit/)

**Construct, sample and validate rare-event forecasting datasets.**

OnsetKit is a NumPy-only toolkit for forward-looking event labels, finite-population stratified sampling and chronological validation. Applications include market volatility-regime transitions, liquidity shocks, equipment failures and operational incidents. It is experimental alpha software; it does not itself detect regimes or fit a forecasting model.

## Installation

OnsetKit is available on the official Python Package Index (PyPI).

```bash
pip install onsetkit
```

### Requirements

- Python 3.10 or later
- NumPy >= 1.24

### Project links

- [PyPI](https://pypi.org/project/onsetkit/)
- [GitHub](https://github.com/jaskaran04/onsetkit)
- [Documentation](https://github.com/jaskaran04/onsetkit#readme)
- [Issue tracker](https://github.com/jaskaran04/onsetkit/issues)

### Development installation

```bash
git clone https://github.com/jaskaran04/onsetkit.git
cd onsetkit
python -m pip install -e ".[dev]"
python -m pytest -v
```

Run the demonstrations from the repository root:

```bash
python examples/regime_demo.py
python examples/panel_demo.py
```

## Quick start

```python
import numpy as np
from onsetkit import label_onsets, purged_forward_split, stratified_sample

active = np.zeros(1000, dtype=int)
active[100:120] = 1
active[400:420] = 1
active[800:820] = 1
result = label_onsets(active, horizon=5)
train, test = purged_forward_split(result.label_end, 700, 1000, gap=5)
y_train = result.labels[train]
sample = stratified_sample(y_train, budget=20, allocation="equal", seed=42)
selected_rows = train[sample.indices]
print(sample.estimate_mean(result.labels[selected_rows]))
# X must contain causal features aligned with the original rows.
# model.fit(X[selected_rows], result.labels[selected_rows], sample_weight=sample.weights)
```

### Label semantics

`label_onsets(active, horizon)` accepts an ordered binary state sequence. A 0-to-1 transition at row j is an onset. For inactive row t, the target is 1 if an onset occurs in t+1 through t+h inclusive. Active rows and rows without a complete future horizon are excluded (label -1). The initial active state is not treated as an observed onset. `label_end` is the inclusive end row of each complete horizon, or -1 for excluded rows. Labels are retrospective training targets, not deployable features. The caller supplies the regime definition; ex-post regime labels may require additional dating/availability buffers beyond the nominal horizon.

### Sampling

`stratified_sample` selects uniformly without replacement within each stratum, with at least one observation per stratum by default. Indices are sorted and relative to the array supplied to the function, not the original dataset. Returned inclusion probabilities and inverse-probability weights are aligned to selected rows. The budget must be feasible. Allocation choices:

| Method | Objective |
| --- | --- |
| proportional | Equal scale across strata |
| equal | Minimize sum of reciprocal stratum sample counts |
| neyman | Minimize finite-population variance using supplied stratum scales |

For population stratum sizes N_h, scales S_h and sample counts n_h, the variable part of the variance of a stratified mean is proportional to sum(N_h^2 S_h^2 / n_h). `optimal_allocation` minimizes this separable objective under integer sample budgets and bounds by assigning each remaining unit to the largest marginal reduction. This is exact in real arithmetic; extreme floating-point scale ratios can introduce ties or underflow. The algorithm costs O((budget - sum(lower))*log(H)) for H strata. Zero-scale strata receive no extra units while positive marginal gains remain; ties are resolved by stratum index. Scales must be fixed before sampling and supplied in sorted unique stratum order; use an independent or training-only pilot. No claim is made that these scales are optimal for an arbitrary model's predictive loss.

Within a stratum, inclusion probability is n_h/N_h. `estimate_mean` implements the Horvitz–Thompson mean for fixed population values observed at selected rows. It is design-unbiased conditional on a fixed allocation and population; temporal independence is not required for that finite-population statement. This does not guarantee unbiased learned parameters, calibrated probabilities or out-of-sample performance. Inverse-probability weighting corrects the sampling design, not distribution shifts or leakage.

### Temporal validation

`purged_forward_split(label_end, test_start, test_stop, gap=0)` uses half-open test row bounds. Training label ends must be strictly earlier than test_start-gap. Test labels must resolve strictly before test_stop. Empty splits are rejected. Apply sampling only after splitting; fit feature transforms and pilot scales on permitted training data. The splitter cannot infer future-dependent features, revision delays, or cross-entity information leakage. A gap does not establish statistical independence.

## Demonstration

`examples/regime_demo.py` creates synthetic volatility regimes and compares original, sampled and weighted event prevalence. It is not a benchmark or a model forecast.

## Multi-entity panels (v0.2.0)

```python
from onsetkit import label_panel_onsets, panel_forward_split

# Arrays of equal length in any input row order; time can be numeric or np.datetime64.
r = label_panel_onsets(active, entities, time, horizon=5)
train, test = panel_forward_split(r.label_end, time, test_start=350,
                                test_stop=500, train_cutoff=340)
```

Labels are computed separately in each entity's timestamp order and returned in the original row order. Duplicate timestamps within one entity are rejected; simultaneous timestamps across entities are allowed. The horizon counts observations within an entity, not elapsed time. `label_end` contains original row indices, so use `panel_forward_split`, not the single-series row-index splitter, for panel labels. The panel splitter applies one global time cutoff across all entities. It does not check external feature or regime-label availability, and assumes label-end indices come from the corresponding panel labels. Explicitly sort selected rows by time if the downstream model requires it.

## Design-based variance

```python
sample = stratified_sample(strata, budget=100, allocation="equal", minimum=2)
mean = sample.estimate_mean(values[sample.indices])
variance = sample.estimate_variance(values[sample.indices])
```

For fixed finite-population attributes under independent stratified simple random sampling without replacement, the estimator is sum((N_h/N)^2 * (1 - n_h/N_h) * s_h^2/n_h). Census strata contribute zero. Noncensus strata with fewer than two sampled rows raise an error. This estimates the design variance of the population mean, not uncertainty from fitting a forecasting model, future market uncertainty or time-series process variance. The sample allocation must be fixed independently of the sampling draw.

## Event-level evaluation

```python
from onsetkit import event_metrics
m = event_metrics(active, alarms, horizon=5, evaluation_mask=held_out_mask)
print(m.event_recall, m.alarm_precision, m.mean_first_warning_lead)
```

This single-sequence API scores binary alarms on eligible inactive rows with complete horizons, optionally restricted to a boolean evaluation mask. Each positive alarm is attributed to the next onset, not to every onset within the horizon. An event is evaluable only if at least one scored row could predict it as the next onset. Repeated alarms count once toward event recall and separately toward alarm precision. False alarm rate is false alarms divided by eligible negative rows. Mean first-warning lead is computed over detected events in observation rows. Undefined rates are NaN. Use a mask restricted to labels that resolve within the held-out block; for panels call separately on time-sorted entity sequences. These are operational metrics, not pointwise classifier metrics or a statistical significance test.

Run `python examples/panel_demo.py` for a shuffled multi-asset synthetic example, weighted sampling and random-alarm baseline.

## Current limits

- Panel timestamps are used for ordering and splitting, but horizons count observations rather than elapsed time.
- Regime/event detection and forecasting models are supplied externally.
- No uncertainty intervals, streaming execution or probability calibration yet; event scoring is per sequence.
- No real-market or industrial performance benchmarks yet.
- NumPy arrays are mutable even inside frozen result dataclasses; treat them as read-only.

### Known numerical limitations

- Mixed integer and floating-point timestamp comparisons may lose precision for large integer timestamps. Use consistent timestamp dtypes, including split cutoffs.
- Extreme numerical scales may cause underflow in allocation calculations or intermediate overflow in weighted mean and variance estimation, even when the final result is representable.
- Complex-valued inputs are not supported; use appropriately scaled, real-valued inputs. Some complex inputs may currently be cast to real values instead of being rejected.
- Extremely large forecasting horizons may raise an overflow exception.

These are known limitations of the current implementation, not corrected defects in this documentation release.

## Roadmap

Explicit information availability timestamps, elapsed-time horizons, aggregate panel event metrics, calibration utilities, uncertainty intervals, and reproducible real-world financial and industrial benchmarks.

## Contributing

Bug reports, reproducible examples and regression tests are welcome. Please open an [issue](https://github.com/jaskaran04/onsetkit/issues) or submit a pull request. Changes to existing functionality should include regression tests. For release instructions, see [RELEASING.md](https://github.com/jaskaran04/onsetkit/blob/main/RELEASING.md).

## Author and maintainer

**Jaskaran Singh**

- [GitHub](https://github.com/jaskaran04)
- [PyPI project](https://pypi.org/project/onsetkit/)

## Licence

OnsetKit is distributed under the MIT License. See [LICENSE](https://github.com/jaskaran04/onsetkit/blob/main/LICENSE) for details.
