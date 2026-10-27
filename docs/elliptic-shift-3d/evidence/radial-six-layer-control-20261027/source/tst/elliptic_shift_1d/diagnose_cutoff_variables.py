#!/usr/bin/env python3
"""Analytic CMC data isolate cutoff differentiation/variable conditioning."""
import argparse,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from evolve import Evolution
from puncture_chart import PunctureChart
from trumpet_data import CMCTrumpet
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);records=[]
for n in [505,1009,2017]:
 opts=SimpleNamespace(n=n,inner_radius=1/64,origin_grid=False,puncture=True,puncture_chart=True,origin_variables=False,backend='spherical-g1',width=7,quadrature_width=7,library=a.library,kappa1=4.,kappa2=-.99,theta='appendix-c',slicing=.25,lapse_restoring=1.,lapse_restoring_form='multiplicative',endpoint='raw-lhopital',scri_sat=4.,regularity_sat=8.,origin_ko=0.,origin_ko_radius=.4,advection='upwind',eta=0.,gauge_corner_sat=0.,damping_form='constant',lapse_background_source='collapsed-inner',lapse_restoring_profile='physical-log',initial_data='trumpet',mass=1/3)
 af=PunctureChart(opts);r=af.r;q,qd,qdd,meta=CMCTrumpet(1/3).data(r);O=(1-r*r)/2;A=(1+r*r)/2
 chi=np.array([z['chi_total'] for z in meta]);alpha=np.r_[[z['physical_lapse'] for z in meta[:-1]],0.]*O;alpha[-1]=1
 state=q.copy();state[:,0]=np.log(chi/af.refs['chi'][:,0]);state[:,6]=np.log(alpha/af.refs['alpha'][:,0])
 for field in [2,4]:state[:,field]=q[:,field]/af.refs['w'][:,0]
 state[:,3]=(q[:,3]-3*af.refs['gap'][:,0])/af.refs['w'][:,0]
 _,_,diag,_=af.parts(state)
 for name,values in [('AF-reference-coefficients',diag)]:
  records.append(dict(n=n,chart=name,h=float(r[1]-r[0]),H_left_max=float(max(abs(values[:10,0]))),M_left_max=float(max(abs(values[:10,1]))),H_rms=float(np.sqrt(np.sum(r*r*values[:,0]**2)/np.sum(r*r))),M_rms=float(np.sqrt(np.sum(r*r*values[:,1]**2)/np.sum(r*r)))))
 newopts=SimpleNamespace(**vars(opts));newopts.puncture_chart=False;newopts.metric_variable='log-trumpet';newopts.lapse_variable='log-reference';ev=Evolution(newopts)
 state=ev.from_physical(q,total_chi=chi);state[:,6]=np.log(alpha/A);_,_,diag,_=ev.parts(state)
 records.append(dict(n=n,chart='physical-curvature-and-trumpet-metric',h=float(r[1]-r[0]),H_left_max=float(max(abs(diag[:10,0]))),M_left_max=float(max(abs(diag[:10,1]))),H_rms=float(np.sqrt(np.sum(r*r*diag[:,0]**2)/np.sum(r*r))),M_rms=float(np.sqrt(np.sum(r*r*diag[:,1]**2)/np.sum(r*r)))))
review=dict(records=records,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Same analytic constraint-satisfying CMC physical fields, no evolution or constraint projection. Isolates representation and boundary differentiation; does not admit a wormhole or new discretization.')
(a.output/'review.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps(review))
