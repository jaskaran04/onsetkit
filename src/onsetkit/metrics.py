"""Event-centric evaluation for a single ordered binary regime sequence."""
from dataclasses import dataclass
import numpy as np
from .core import _vector, label_onsets


@dataclass(frozen=True)
class EventMetrics:
    eligible_rows: int
    eligible_events: int
    detected_events: int
    alarms: int
    true_alarms: int
    false_alarms: int
    event_recall: float
    alarm_precision: float
    false_alarm_rate: float
    mean_first_warning_lead: float


def event_metrics(active, alarms, horizon, *, evaluation_mask=None):
    """Score binary alarms, attributing each to the next onset within horizon.

    Only eligible inactive rows with complete horizons and a true evaluation
    mask are scored. Events count only if at least one scored row could have
    predicted them as the next onset. Multiple alarms for one onset count once
    toward event recall, but separately toward alarm precision. Undefined rates
    are NaN. Lead is measured in observation rows, not clock time.
    """
    r = label_onsets(active, horizon)
    a = _vector(alarms, 'alarms')
    if len(a) != len(r.labels) or a.dtype.kind not in 'biuf' or not np.all(np.isin(a, [0, 1])):
        raise ValueError('alarms must be aligned binary values')
    mask = r.eligible.copy()
    if evaluation_mask is not None:
        m = _vector(evaluation_mask, 'evaluation_mask')
        if len(m) != len(a) or m.dtype.kind != 'b':
            raise ValueError('evaluation_mask must be aligned boolean values')
        mask &= m
    rows = np.flatnonzero(mask)
    positive = rows[r.labels[rows] == 1]
    onsets = np.flatnonzero(r.onset)
    targets = onsets[np.searchsorted(onsets, positive, side='right')]
    alerted = a[positive].astype(bool)
    detected, first = np.unique(targets[alerted], return_index=True)
    leads = detected - positive[alerted][first]
    events = len(np.unique(targets))
    n_alarm = int(np.count_nonzero(a[rows]))
    tp = int(np.count_nonzero(alerted))
    fp = n_alarm - tp
    negatives = len(rows) - len(positive)
    def ratio(x, y):
        return x/y if y else float('nan')
    return EventMetrics(len(rows), events, len(detected), n_alarm, tp, fp,
                        ratio(len(detected), events), ratio(tp, n_alarm),
                        ratio(fp, negatives), float(np.mean(leads)) if len(leads) else float('nan'))
