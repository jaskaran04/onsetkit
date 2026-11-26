"""Tools for forward-onset forecasting datasets and finite-population sampling."""
from .core import (OnsetLabels, StratifiedSample, label_onsets, optimal_allocation,
                   stratified_sample, purged_forward_split)
from .panel import label_panel_onsets, panel_forward_split
from .metrics import EventMetrics, event_metrics
__version__ = '0.2.0'
__all__ = ['OnsetLabels', 'StratifiedSample', 'label_onsets', 'optimal_allocation',
           'stratified_sample', 'purged_forward_split', 'label_panel_onsets',
           'panel_forward_split', 'EventMetrics', 'event_metrics']
