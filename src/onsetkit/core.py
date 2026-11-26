"""Finite-population sampling and forward-onset labels on ordered rows."""
from dataclasses import dataclass
import heapq
from numbers import Integral
import numpy as np


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _vector(x, name):
    a = np.asarray(x)
    if a.ndim != 1 or not a.size:
        raise ValueError(f'{name} must be a nonempty one-dimensional array')
    return a


@dataclass(frozen=True)
class OnsetLabels:
    labels: np.ndarray  # -1 means excluded / censored
    eligible: np.ndarray
    onset: np.ndarray
    label_end: np.ndarray  # inclusive row; -1 for excluded


def label_onsets(active, horizon):
    """Predict a 0->1 transition in rows t+1,...,t+h, excluding active rows.

    The first row is never an onset (prior state unknown). All incomplete
    future windows are excluded, even if a positive is already observable.
    """
    a = _vector(active, 'active')
    h = _integer(horizon, 'horizon')
    if a.dtype.kind not in 'biuf' or not np.all(np.isin(a, [0, 1])):
        raise ValueError('active must contain only binary states')
    a = a.astype(bool)
    n = len(a)
    onset = np.zeros(n, dtype=bool)
    onset[1:] = a[1:] & ~a[:-1]
    idx = np.arange(n)
    eligible = ~a & (idx < n - h)
    labels = np.full(n, -1, dtype=np.int8)
    end = np.full(n, -1, dtype=np.int64)
    t = idx[eligible]
    cumulative = np.r_[0, np.cumsum(onset, dtype=np.int64)]
    labels[t] = cumulative[t+h+1] > cumulative[t+1]
    end[t] = t+h
    return OnsetLabels(labels, eligible, onset, end)


def optimal_allocation(counts, scales, budget, *, minimum=1):
    """Exact integer minimizer of sum N_h**2 * S_h**2 / n_h.

    Bounds: min(minimum,N_h) <= n_h <= N_h. Uses diminishing marginal
    gains; O((budget-sum(lower))*log(H)). Zero gains break ties by
    stratum index. Scales are supplied or pilot estimates, not fitted models.
    """
    N = _vector(counts, 'counts')
    if N.dtype.kind not in 'iu' or np.any(N <= 0) or np.any(N > np.iinfo(np.int64).max):
        raise ValueError('counts must be positive integers within int64')
    N = N.astype(np.int64)
    S = _vector(scales, 'scales').astype(float)
    if len(N) != len(S) or np.any(~np.isfinite(S)) or np.any(S < 0):
        raise ValueError('scales must be finite, nonnegative and match counts')
    k = _integer(budget, 'budget')
    m = _integer(minimum, 'minimum')
    lower = np.minimum(N, m) if m <= int(N.max()) else N.copy()
    if not sum(map(int, lower)) <= k <= sum(map(int, N)):
        raise ValueError('budget infeasible under stratum bounds')
    if k == sum(map(int, N)):
        return N.copy()
    # Normalize both factors before multiplying to avoid overflow.
    if np.max(S) == 0:
        A = np.zeros(len(N))
    else:
        A = ((N / N.max()) * (S / S.max())) ** 2
    out = lower.copy()
    def gain(i):
        q = int(out[i])
        return float(A[i]) / q / (q+1)
    heap = [(-gain(i), i) for i in range(len(N)) if out[i] < N[i]]
    heapq.heapify(heap)
    for _ in range(k - sum(map(int, out))):
        _, i = heapq.heappop(heap)
        out[i] += 1
        if out[i] < N[i]:
            heapq.heappush(heap, (-gain(i), i))
    return out


@dataclass(frozen=True)
class StratifiedSample:
    indices: np.ndarray
    inclusion_probabilities: np.ndarray
    weights: np.ndarray
    strata: np.ndarray
    counts: np.ndarray
    allocation: np.ndarray
    population_size: int
    sampled_strata: np.ndarray | None = None  # integer stratum codes

    def estimate_mean(self, values):
        """Horvitz-Thompson population mean of fixed values at sampled rows."""
        v = _vector(values, 'values').astype(float)
        if len(v) != len(self.indices) or not np.all(np.isfinite(v)):
            raise ValueError('values must be finite and aligned to sampled indices')
        return float(np.dot(v, self.weights) / self.population_size)


    def estimate_variance(self, values):
        """Design-unbiased estimated variance of the HT mean under stratified SRS.

        Noncensus strata require at least two sampled rows. Values must be fixed
        population attributes, not predictions fitted using this same sample.
        """
        v = _vector(values, 'values').astype(float)
        if len(v) != len(self.indices) or not np.all(np.isfinite(v)):
            raise ValueError('values must be finite and aligned to sampled indices')
        if self.sampled_strata is None:
            raise ValueError('sampled stratum codes are unavailable')
        if np.any((self.allocation < self.counts) & (self.allocation < 2)):
            raise ValueError('noncensus strata require at least two observations')
        total = 0.0
        for i, (N, n) in enumerate(zip(self.counts, self.allocation)):
            if n < N:
                s2 = np.var(v[self.sampled_strata == i], ddof=1)
                total += (int(N) / self.population_size)**2 * (1 - n/N) * s2/n
        return float(total)


def stratified_sample(strata, budget, *, allocation='proportional', scales=None,
                      minimum=1, seed=0):
    """SRS without replacement independently within each observed stratum.

    allocation: proportional (equal scales), equal (scales 1/N), or neyman
    (user-supplied scales aligned to sorted unique stratum labels).
    """
    s = _vector(strata, 'strata')
    if s.dtype.kind not in 'biufUS' or (s.dtype.kind == 'f' and not np.all(np.isfinite(s))):
        raise ValueError('strata must be finite numeric or string labels')
    names, inverse, N = np.unique(s, return_inverse=True, return_counts=True)
    if allocation == 'proportional' and scales is None:
        S = np.ones(len(N))
    elif allocation == 'equal' and scales is None:
        S = 1 / N
    elif allocation == 'neyman' and scales is not None:
        S = scales
    else:
        raise ValueError('choose proportional/equal without scales, or neyman with scales')
    n = optimal_allocation(N, S, budget, minimum=minimum)
    rng = np.random.default_rng(_integer(seed, 'seed', 0))
    # Group once, rather than scanning the population for each stratum.
    order = np.argsort(inverse, kind='stable')
    offsets = np.r_[0, np.cumsum(N)]
    selected = [rng.choice(order[offsets[i]:offsets[i+1]], size=int(k), replace=False)
                for i, k in enumerate(n)]
    idx = np.sort(np.concatenate(selected))
    p = (n / N)[inverse[idx]]
    return StratifiedSample(idx, p, 1/p, names, N, n, len(s), inverse[idx])


def purged_forward_split(label_end, test_start, test_stop, *, gap=0):
    """Return eligible train/test row indices for one forward block.

    Train labels must end strictly before test_start-gap; test rows lie
    in [test_start,test_stop), with label ends strictly before test_stop.
    Excluded rows use -1. gap is additional pre-test buffer in rows.
    """
    end = _vector(label_end, 'label_end')
    if end.dtype.kind != 'i':
        raise ValueError('label_end must be signed integer row indices')
    n = len(end)
    start = _integer(test_start, 'test_start')
    stop = _integer(test_stop, 'test_stop')
    g = _integer(gap, 'gap', 0)
    t = np.arange(n)
    valid = end >= 0
    if np.any(end < -1) or np.any(valid & ((end < t) | (end >= n))):
        raise ValueError('label_end must be -1 or a valid inclusive end row >= start row')
    if not start < stop <= n or g >= start:
        raise ValueError('invalid test block or gap')
    train = t[valid & (t < start) & (end < start-g)]
    test = t[valid & (t >= start) & (t < stop) & (end < stop)]
    if not len(train) or not len(test):
        raise ValueError('split has no eligible training or test rows')
    return train, test
