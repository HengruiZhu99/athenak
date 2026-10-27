#!/usr/bin/env python3
"""Audit completed spatial evidence against the declared gates, without admission.

This does not certify independent controls, characteristic time accuracy, or the
extension run. Those require their own evidence even if all spatial gates pass.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from telegraph_analysis_io import trajectory,physical_lapse_gradient

p=argparse.ArgumentParser()
p.add_argument('--review',type=Path,required=True)
p.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parents[2]/'docs/elliptic-shift-1d/telegraph-protocol-v1.json')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();review=json.loads(a.review.read_text());gate=json.loads(a.protocol.read_text())['acceptance']
assert len(review['comparisons'])==1,'Assess exactly one three-grid continuum configuration'
comparison=review['comparisons'][0];records=sorted(review['records'],key=lambda x:x['n']);assert len(records)==3
assert [r['n'] for r in records]==comparison['grids']
for record in records:
 path=Path(record['path'])/'manifest.json'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==record['manifest_sha256'],'Stale review manifest'
 manifest=json.loads(path.read_text());assert manifest['status']=='finished' and manifest['all_RK_stages']==record['stages']
assert comparison['rows'][0]['t']==0 and comparison['rows'][-1]['t']==records[0]['time'],'Incomplete common-time trajectory'
fine=records[-1];checks={}
def check(name,value,limit,relation='<='):
 passed=value is not None and np.isfinite(value) and (value<=limit if relation=='<=' else value>=limit)
 checks[name]=dict(value=value,relation=relation,limit=limit,passed=bool(passed))
check('main_duration',min(r['time'] for r in records),gate['main_duration'],'>=')
extension_complete=min(r['time'] for r in records)>=gate['extension_duration']
assert len(set(r['time'] for r in records))==1,'Mismatched trajectory durations'
check('bulk_field_order',comparison['bulk_peak_order'],gate['bulk_field_and_accepted_constraint_order_min'],'>=')
check('endpoint_field_order',comparison['endpoint_peak_order'],gate['endpoint_field_order_min'],'>=')
for name,entry in comparison['constraint_peak_orders'].items():
 check(name+'_finest_order',entry['orders'][-1],gate['bulk_field_and_accepted_constraint_order_min'],'>=')
s=fine['stages']
check('H_rms',s['max_accepted_H_rms'],gate['finest_accepted_H_rms_max'])
check('H_bulk',s['max_accepted_H_bulk'],gate['finest_accepted_H_bulk_max'])
for name,key in [('scri_clock','max_scri_clock'),('scri_nullness','max_scri_null'),('scri_pole','max_scri_pole')]:check(name,s[key],gate[name])
for field in ['alpha','chi']:checks['positive_'+field]=dict(value=s['min_'+field],passed=bool(np.isfinite(s['min_'+field]) and s['min_'+field]>0))
middle,finest=[trajectory(Path(x['path']))[0] for x in records[-2:]]
ratio=(records[-1]['n']-1)//(records[-2]['n']-1);times=sorted(set(middle)&set(finest));gradient=[];ray_errors=[]
invariants=[]
for t in times:
 A=middle[t];B=finest[t];r=A['r'];assert np.array_equal(r,B['r'][::ratio])
 mask=r<.3;ga=physical_lapse_gradient(A)[mask];gb=physical_lapse_gradient(B)[::ratio][mask]
 gradient.append(dict(t=t,middle_peak=float(max(abs(ga))),fine_peak_on_common_nodes=float(max(abs(gb))),max_difference=float(max(abs(ga-gb)))))
 ray_errors.append(dict(t=t,position=float(max(abs(A['rays'][:,0]-B['rays'][:,0]))),logJ=float(max(abs(A['rays'][:,1]-B['rays'][:,1])))))
 # All physical points with a finite mass diagnostic, plus AF coefficient mass.
 mass=B['diagnostic'][:,10];finite=np.isfinite(mass)
 invariants.append(dict(t=t,mass_absolute_error_max=float(max(abs(mass[finite]-1/3))),AF_mass_absolute_error=float(abs(mass[0]-1/3))))
assert all(x['fine_peak_on_common_nodes']>0 for x in gradient),'Relative front metric needs nonzero reference gradients'
# Normalize at each time: a large late peak must not hide an earlier mismatch.
for row in gradient:
 row['relative_difference']=row['max_difference']/row['fine_peak_on_common_nodes']
 row['relative_peak_difference']=abs(row['middle_peak']-row['fine_peak_on_common_nodes'])/row['fine_peak_on_common_nodes']
relative=max(x['relative_difference'] for x in gradient)
check('matched_gradient_discrepancy',relative,.1)
peak_difference=max(x['relative_peak_difference'] for x in gradient)
check('gradient_peak_discrepancy',peak_difference,.1)
result=dict(review=str(a.review.resolve()),review_sha256=hashlib.sha256(a.review.read_bytes()).hexdigest(),protocol=str(a.protocol.resolve()),protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest(),checks=checks,gradient_history=gradient,ray_spatial_errors=ray_errors,finest_invariants=invariants,
 spatial_status='passed_these_spatial_checks' if all(x['passed'] for x in checks.values()) else 'failed_spatial_checks',goal_status='incomplete',
 missing_evidence=['Independent timestep refinement and characteristic-time accuracy','Independent elliptic quadrature refinement','Independent dissipation variation','Adjacent fixed continuum relaxation parameters with convergence']+([] if extension_complete else ['t=10 extension and its spatial convergence']),extension_duration_completed=extension_complete,
 scope='Completed trajectories, verified ancestry and exact common times. Gradient comparisons use analytic AF factors and seven-point chart derivatives at matching physical r nodes. Invariant errors do not override constraint failures. Snapshots alone do not certify all intermediate characteristic or gradient extrema.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(spatial_status=result['spatial_status'],checks=checks,goal_status='incomplete'),indent=2))
