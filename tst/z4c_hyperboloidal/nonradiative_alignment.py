#!/usr/bin/env python3
"""Common physical-cut comparisons without plotting dependencies."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'vis/python'))
from scri_news import interpolate_time

def aligned(archives,limit=None):
    labels=set(map(tuple,archives[0]['ell_m']))
    for a in archives[1:]:labels &= set(map(tuple,a['ell_m']))
    labels=sorted(pair for pair in labels if limit is None or pair[0]<=limit)
    begin=max(a['bondi_time'][0] for a in archives);end=min(a['bondi_time'][-1] for a in archives)
    assert end>begin
    # Include every original cut in the common interval, so sharp sampled
    # maxima are not lost by downsampling the comparison grid.
    times=np.unique(np.concatenate([np.linspace(begin,end,401)]+[a['bondi_time'][(a['bondi_time']>=begin)&(a['bondi_time']<=end)] for a in archives]));values=[]
    for a in archives:
        indices={tuple(pair):i for i,pair in enumerate(a['ell_m'])}
        values.append(np.column_stack([interpolate_time(a['bondi_time'],a['news_modes'][:,indices[pair]],times) for pair in labels]))
    return times,values,labels
