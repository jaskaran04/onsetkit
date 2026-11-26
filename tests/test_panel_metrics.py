import itertools
import numpy as np
import pytest
from onsetkit import (label_panel_onsets, panel_forward_split, label_onsets,
                      event_metrics, stratified_sample)


def test_panel_shuffled():
    a = np.array([0,0,1,0,0,0,0,1,1,0,0,0])
    e = np.repeat(['A','B'], 6)
    t = np.tile(np.arange(6), 2)
    perm = np.random.default_rng(3).permutation(len(a))
    r = label_panel_onsets(a[perm], e[perm], t[perm], 2)
    for entity in ['A','B']:
        idx = np.flatnonzero(e[perm] == entity)
        idx = idx[np.argsort(t[perm][idx])]
        expected = label_onsets(a[perm][idx], 2)
        np.testing.assert_array_equal(r.labels[idx], expected.labels)
        np.testing.assert_array_equal(r.onset[idx], expected.onset)
        np.testing.assert_array_equal(r.label_end[idx[expected.eligible]], idx[expected.label_end[expected.eligible]])


def test_panel_global_split():
    t = np.tile(np.arange(10), 2)
    r = label_panel_onsets(np.zeros(20), np.repeat(['A','B'],10), t, 2)
    train, test = panel_forward_split(r.label_end, t, 5, 10, train_cutoff=4)
    assert set(t[train]) == {0,1}
    assert set(t[test]) == {5,6,7}
    assert np.all(t[r.label_end[train]] < 4)


def test_datetime():
    t = np.datetime64('2020-01-01') + np.arange(10)
    r = label_panel_onsets(np.zeros(10), ['a']*10, t, 1)
    train,test = panel_forward_split(r.label_end,t,t[5],t[-1]+np.timedelta64(1,'D'))
    assert list(train)==[0,1,2,3]
    assert list(test)==[5,6,7,8]


@pytest.mark.parametrize('entities,time', [(['a','a'],[0,0]), (['a'],[0,1]), (['a','b'],[0,np.nan])])
def test_invalid_panel(entities,time):
    with pytest.raises(ValueError): label_panel_onsets([0,1],entities,time,1)


def test_no_cross_entity():
    r = label_panel_onsets([0,0,1,1], ['a','a','b','b'], [0,1,0,1],1)
    assert list(r.labels)==[0,-1,-1,-1]
    assert not r.onset.any()


def test_event_metrics():
    a = [0,0,0,1,0,0,0,1,0,0,0]
    alarms = [1,1,1,0,1,0,0,0,0,0,0]
    m = event_metrics(a,alarms,2)
    assert (m.eligible_events,m.detected_events,m.alarms,m.true_alarms,m.false_alarms)==(2,1,4,2,2)
    assert m.event_recall == .5 and m.alarm_precision == .5
    assert m.mean_first_warning_lead == 2
    assert m.false_alarm_rate == 2/3


def test_event_mask():
    m = event_metrics([0,0,1,0,0],[0,1,0,0,0],1,evaluation_mask=[False,True,False,False,False])
    assert m.event_recall == 1 and m.alarm_precision == 1
    assert np.isnan(m.false_alarm_rate)


def test_empty_events():
    m = event_metrics([0,0,0],[0,0,0],1)
    assert m.eligible_events == 0 and np.isnan(m.event_recall) and np.isnan(m.alarm_precision)


def test_next_event_attribution():
    m = event_metrics([0,1,0,1,0,0,0],[1,0,0,0,0,0,0],3)
    assert m.eligible_events == 2 and m.detected_events == 1


def test_variance_census():
    s = stratified_sample([0,1],2)
    assert s.estimate_variance([1,100]) == 0


def test_variance_singleton():
    s = stratified_sample([0,0,1,1],2)
    with pytest.raises(ValueError): s.estimate_variance([1,2])


def test_variance_exhaustive():
    # Enumerate all possible two-of-four samples; expectation of estimated
    # variance equals actual design variance of the HT mean.
    from dataclasses import replace
    v=np.array([1.,2.,4.,8.]); s=stratified_sample([0]*4,2)
    estimates=[]; means=[]
    for pair in itertools.combinations(range(4),2):
        x=replace(s,indices=np.array(pair)); estimates.append(x.estimate_variance(v[list(pair)])); means.append(x.estimate_mean(v[list(pair)]))
    assert np.mean(estimates)==pytest.approx(np.var(means))
