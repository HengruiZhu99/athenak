#!/usr/bin/env python3
"""End-to-end radiation on angle-dependent native-time cuts of an analytic wave."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'vis/python'))
from scri_news import analyze
spec=importlib.util.spec_from_file_location('manufactured',Path(__file__).with_name('scri_geometry.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
mu,w=np.polynomial.legendre.leggauss(12);phi=np.arange(24)*2*np.pi/24
mu=np.repeat(mu,24);w=np.repeat(w,24)*2*np.pi/24;phi=np.tile(phi,12)
x=np.column_stack((mu,np.sqrt(1-mu*mu)*np.cos(phi),np.sqrt(1-mu*mu)*np.sin(phi)))
times=np.linspace(0,.4,41)
with tempfile.TemporaryDirectory(prefix='scri-news-analytic-') as directory:
 root=Path(directory);cache=root/'cache';cache.mkdir()
 records=[dict(cycle=q,time=float(t),source='analytic',ranks=1) for q,t in enumerate(times)]
 (cache/'records.json').write_text(json.dumps(records))
 np.savez(cache/'grid.npz',x=x,weights=w,ids=np.arange(len(x)))
 series=np.lib.format.open_memmap(cache/'metric.npy',mode='w+',dtype='f8',shape=(len(times),len(x),4,4,13))
 for q,t in enumerate(times):
  g,dg,ddg,_,_=m.radiative_metric(t,x,time_gauge=.04)
  series[q]=np.concatenate((g[...,None],np.moveaxis(dg[:,1:],1,-1),
      np.moveaxis(ddg[:,1:,1:],(1,2),(-2,-1)).reshape(g.shape+(9,))),axis=-1)
 series.flush()
 result=analyze(cache,root/'result',lmax=8)
 data=np.load(root/'result/news.npz');labels=[tuple(row) for row in data['ell_m']]
 expected=.02*1.7*np.cos(1.7*data['bondi_time'])/np.sqrt(15/(32*np.pi))
 np.testing.assert_allclose(data['news_modes'][:,labels.index((2,0))],expected,atol=3e-9)
 other=data['news_modes'].copy();other[:,labels.index((2,0))]=0
 assert np.max(abs(other))<3e-9, np.max(abs(other))
 assert result['diagnostics_max']['round_cut_max']<2e-9
 assert result['diagnostics_max']['conformal_hessian_tf_max']<2e-9
 print('PASS end-to-end common Bondi-time news; angle-dependent clock, spin modes, normalization',float(np.max(abs(data['news_modes'][:,labels.index((2,0))]-expected))))
