import itertools
import numpy as np
import pytest
from onsetkit import label_onsets, optimal_allocation, stratified_sample, purged_forward_split


def test_onsets():
    r = label_onsets([0,0,1,1,0,0,1,0,0], 2)
    np.testing.assert_array_equal(r.labels, [1,1,-1,-1,1,1,-1,-1,-1])
    np.testing.assert_array_equal(np.flatnonzero(r.onset), [2,6])
    np.testing.assert_array_equal(r.label_end, [2,3,-1,-1,6,7,-1,-1,-1])


def test_future_not_current():
    r = label_onsets([1,1,0,0,0,1,0], 1)
    np.testing.assert_array_equal(r.labels, [-1,-1,0,0,1,-1,-1])
    assert not r.onset[0]

@pytest.mark.parametrize('a,h', [([],1),([0,2],1),([0,np.nan],1),([0,1],0),([0,1],1.5),([[0,1]],1)])
def test_bad_labels(a,h):
    with pytest.raises(ValueError): label_onsets(a,h)


def test_long_horizon():
    assert not label_onsets([0,1], 10).eligible.any()


def test_random_labels_match_naive():
    rng=np.random.default_rng(42)
    for _ in range(100):
        a=rng.integers(0,2,30); h=int(rng.integers(1,35))
        y=label_onsets(a,h).labels
        expected=[-1 if a[t] or t+h>=len(a) else int(any(a[j] and not a[j-1] for j in range(t+1,t+h+1))) for t in range(len(a))]
        np.testing.assert_array_equal(y,expected)


def test_allocation_exact_exhaustive():
    rng=np.random.default_rng(10)
    for _ in range(100):
        N=rng.integers(1,7,3); S=rng.integers(0,5,3); k=int(rng.integers(3,N.sum()+1))
        n=optimal_allocation(N,S,k)
        obj=lambda x: np.sum((N*S)**2/np.asarray(x))
        best=min(obj(x) for x in itertools.product(*(range(1,int(z)+1) for z in N)) if sum(x)==k)
        assert obj(n)==pytest.approx(best)
        assert n.sum()==k


def test_allocation_minimum_and_caps():
    np.testing.assert_array_equal(optimal_allocation([1,4,9],[1,3,0],7,minimum=2),[1,4,2])

@pytest.mark.parametrize('N,S,k', [([1,2],[1,1],1),([1,2],[1,1],4),([0,2],[1,1],2),([1.2,2],[1,1],2),([1,2],[1,-1],2),([1,2],[1],2),([1,2],[1,np.inf],2)])
def test_bad_allocation(N,S,k):
    with pytest.raises(ValueError): optimal_allocation(N,S,k)


def test_sample_weights():
    s=np.repeat([0,1],[90,10]); r=stratified_sample(s,20,allocation='equal',seed=42)
    np.testing.assert_array_equal(r.allocation,[10,10])
    assert len(np.unique(r.indices))==20
    assert np.all(np.diff(r.indices)>0)
    assert r.weights.sum()==pytest.approx(100)
    assert r.estimate_mean(s[r.indices])==pytest.approx(.1)
    np.testing.assert_array_equal(r.indices,stratified_sample(s,20,allocation='equal',seed=42).indices)


def test_neyman():
    r=stratified_sample(np.repeat(['a','b'],[20,20]),10,allocation='neyman',scales=[1,3])
    np.testing.assert_array_equal(r.allocation,[3,7])


def test_census():
    r=stratified_sample([0,1,1],3)
    np.testing.assert_array_equal(r.indices,[0,1,2]); np.testing.assert_array_equal(r.weights,[1,1,1])

@pytest.mark.parametrize('kwargs',[{'allocation':'bad'}, {'allocation':'neyman'}, {'scales':[1,2]}, {'seed':-1}])
def test_bad_sample(kwargs):
    with pytest.raises(ValueError): stratified_sample([0,0,1,1],2,**kwargs)


def test_purge():
    r=label_onsets(np.zeros(20),3)
    train,test=purged_forward_split(r.label_end,10,20,gap=2)
    np.testing.assert_array_equal(train,np.arange(5)); np.testing.assert_array_equal(test,np.arange(10,17))
    assert max(r.label_end[train])<8


def test_excluded_rows():
    r=label_onsets([0,0,1,0,0,0,0,1,0,0,0,0],1)
    train,test=purged_forward_split(r.label_end,6,12)
    assert 2 not in train and 7 not in test and 5 not in train

@pytest.mark.parametrize('end,start,stop,gap',[([0,1,2],0,3,0),([0,1,2],2,4,0),([0,1,2],1,3,1),([0,0,2],1,3,0),([0,1,9],1,3,0),([-1,-1,2],1,3,0)])
def test_bad_split(end,start,stop,gap):
    with pytest.raises(ValueError): purged_forward_split(end,start,stop,gap=gap)
