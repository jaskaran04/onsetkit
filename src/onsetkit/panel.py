"""Entity-aware labels and chronological splits for interleaved panel rows."""
import numpy as np
from .core import OnsetLabels, _vector, _integer, label_onsets


def _time_vector(time):
    t = _vector(time, 'time')
    if t.dtype.kind not in 'iufM' or (t.dtype.kind == 'M' and np.any(np.isnat(t))) or (t.dtype.kind == 'f' and not np.all(np.isfinite(t))):
        raise ValueError('time must contain finite numeric or datetime64 values')
    return t


def label_panel_onsets(active, entities, time, horizon):
    """Label each entity in timestamp order; return arrays in input row order.

    Horizon counts observations within an entity, not elapsed clock time.
    label_end contains original input row indices, not positions after sorting.
    Duplicate timestamps within an entity are rejected.
    """
    a = _vector(active, 'active')
    e = _vector(entities, 'entities')
    t = _time_vector(time)
    h = _integer(horizon, 'horizon')
    if len(a) != len(e) or len(a) != len(t):
        raise ValueError('active, entities and time must have matching lengths')
    if e.dtype.kind not in 'biufUS' or (e.dtype.kind == 'f' and not np.all(np.isfinite(e))):
        raise ValueError('entities must be finite numeric or string labels')
    _, groups = np.unique(e, return_inverse=True)
    order = np.lexsort((t, groups))
    boundaries = np.r_[0, np.flatnonzero(np.diff(groups[order])) + 1, len(a)]
    labels = np.full(len(a), -1, dtype=np.int8)
    eligible = np.zeros(len(a), dtype=bool)
    onset = eligible.copy()
    end = np.full(len(a), -1, dtype=np.int64)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        idx = order[lo:hi]
        if np.any(t[idx][1:] == t[idx][:-1]):
            raise ValueError('duplicate timestamps within an entity')
        r = label_onsets(a[idx], h)
        labels[idx], eligible[idx], onset[idx] = r.labels, r.eligible, r.onset
        end[idx[r.eligible]] = idx[r.label_end[r.eligible]]
    return OnsetLabels(labels, eligible, onset, end)


def panel_forward_split(label_end, time, test_start, test_stop, *, train_cutoff=None):
    """Global timestamp split with inclusive label ends and exclusive cutoffs.

    train_cutoff defaults to test_start; set earlier for an additional buffer.
    Pass numeric bounds for numeric timestamps and datetime64 bounds for dates.
    Returns original row indices in input order; no entity crosses the cutoff
    through a label horizon. Does not check external feature availability.
    """
    t = _time_vector(time)
    end = _vector(label_end, 'label_end')
    if len(t) != len(end) or end.dtype.kind != 'i' or np.any(end < -1) or np.any(end >= len(t)):
        raise ValueError('label_end must be aligned signed indices or -1')
    def bound(x):
        b = np.asarray(x)
        if b.ndim != 0 or (t.dtype.kind == 'M' and (b.dtype.kind != 'M' or np.isnat(b))) or (t.dtype.kind != 'M' and (b.dtype.kind not in 'iuf' or not np.isfinite(b))):
            raise ValueError('invalid timestamp bound')
        return b
    start, stop = bound(test_start), bound(test_stop)
    cutoff = start if train_cutoff is None else bound(train_cutoff)
    if not cutoff <= start < stop:
        raise ValueError('require train_cutoff <= test_start < test_stop')
    rows = np.flatnonzero(end >= 0)
    ends = t[end[rows]]
    if np.any(ends < t[rows]):
        raise ValueError('label ends cannot precede their rows in time')
    train = rows[(t[rows] < cutoff) & (ends < cutoff)]
    test = rows[(t[rows] >= start) & (t[rows] < stop) & (ends < stop)]
    if not len(train) or not len(test):
        raise ValueError('split has no eligible training or test rows')
    return train, test
