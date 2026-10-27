#!/usr/bin/env python3
"""Analyze all surface samples, retaining every face/edge/corner point."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();d=a.directory;table=d/'surface-poles.csv';table=table if table.exists() else table.with_suffix('.csv.gz');x=np.genfromtxt(table,delimiter=',',names=True);fields=['chi','Khat','Theta','alpha','Axx','Axy','Axz','Ayy','Ayz','Azz','LambdaX','LambdaY','LambdaZ'];modes=['full','chi_no_derivative','chi_radial_derivative','metric_no_derivative','trace_no_derivative','oracle_initial_first_jet'];out=dict(scope='No solve or evolution; full216node surface; no diagnostic ablation applied to PDE state',results=[])
for state in [0,1]:
 for nodes in [6,7]:
  b=x[(x['state']==state)&(x['nodes']==nodes)];v=np.column_stack([b['full_'+f] for f in fields]);i,j=np.unravel_index(np.argmax(abs(v)),v.shape);row=b[i];o=dict(state=state,nodes=nodes,maximum=float(abs(v[i,j])),field=fields[j],row=int(row['row']),face=int(row['face']),j=int(row['j']),k=int(row['k']),xyz=[float(row[f]) for f in ['x','y','z']],chi_dn_max=float(np.max(abs(b['chi_dn']))),chi_tangent_max=float(np.max(abs(b['chi_dt_norm']))),metric_d_norm_max=float(np.max(abs(b['metric_d_norm']))),modes={})
  for mode in modes:o['modes'][mode]={f:float(np.max(abs(b[mode+'_'+f]))) for f in fields}
  out['results'].append(o)
b=x[(x['state']==1)&(x['nodes']==7)];i=np.argmax(abs(b['full_LambdaY']));row=b[i];terms=['metric_normal','metric_tangent','evolved_lambda','trace_normal','trace_tangent','curvature'];out['worst_lambdaY_terms']={t:float(row['lambda_'+t+'_1']) for t in terms};out['decomposition_sum_error_max']=max(float(np.max(abs(x['lambda_sum_error_'+str(i)]))) for i in range(3))
n=np.array([b[k] for k in ['x','y','z']]).T;g=np.empty((len(b),3,3));ti=[0,0,0,1,1,2];tj=[0,1,2,1,2,2]
for t,(i,j) in enumerate(zip(ti,tj)):g[:,i,j]=g[:,j,i]=b['g_'+str(t)]
dev=g-np.eye(3);nn=n[:,:,None]*n[:,None,:];columns=np.stack([np.tile(np.eye(3),(len(b),1,1)),nn],axis=-1);coef=np.linalg.lstsq(columns.reshape(-1,2),dev.reshape(-1),rcond=None)[0];fit=coef[0]*np.eye(3)+coef[1]*nn;err=dev-fit;i=np.argmax(abs(b['full_LambdaY']));out['angular_metric_diagnostic']=dict(spherical_fit_a=float(coef[0]),spherical_fit_b=float(coef[1]),metric_deviation_max=float(np.max(abs(dev))),fit_residual_max=float(np.max(abs(err))),fit_residual_rms=float(np.sqrt(np.mean(err**2))),linearized_fitted_spherical_metric_tangent_pole=(4*coef[1]*n[i]).tolist(),measured_metric_tangent_pole=[float(b['lambda_metric_tangent_'+str(j)][i]) for j in range(3)],caveat='Fit never applied to state/acceptance; angular derivatives of Cartesian tensor components are nonzero even for spherical fields; fit alone does not identify which evolved discretization generated the error.')
(d/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(cases=len(out['results']),max_pole=out['results'][-1]['maximum'],sum_error=out['decomposition_sum_error_max'])))
