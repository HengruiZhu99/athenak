"""Verify and localize terminal C24 failure; never admit or restart its state."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import struct
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RAW = ROOT/'output/boosted-refined-core-resolution/aurora'
REL = 'campaigns/n24-boosted-refined-core-resolution-v1-c24-evolution-segment01'
P = RAW/REL
OUT = HERE/'n24-failure'
OUT.mkdir(exist_ok=True)
def read(p): return json.loads(p.read_text())
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()
def save(name, value): (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

sys.path.insert(0, str(RAW/'launcher-boosted-refined-core-resolution-v1'))
import native_transition_campaign as helper
plan, binding, worker, sub, collected = [read(P/n) for n in
    ['plan.json','binding.json','worker-status.json','submission.json','collected.json']]
r = read(P/'output/puncture.report.json')
science = helper.science(RAW)
assert sha(P/'plan.json') == binding['plan_sha256'] == worker['plan_sha256'] == sub['plan_sha256']
assert worker['job_id'] == sub['job_id'] == collected['job_id']
assert binding['source_manifest_sha256'] == worker['source_manifest_sha256'] == science['source_manifest_sha256']
assert binding['binaries'] == worker['binaries'] == {helper.binary('C24'):science['binaries'][helper.binary('C24')]}
for name, digest in binding['inputs'].items(): assert sha(RAW/name) == digest
for name, digest in binding['launcher'].items():
    assert sha(RAW/'launcher-boosted-refined-core-resolution-v1'/name) == digest
case = worker['cases'][0]
assert collected['scheduler']['job_state']=='F' and collected['scheduler']['Exit_status']==1
assert case['returncode']==2 and not case['timed_out'] and case['storage_error'] is None
assert worker['status']==r['status']=='failed' and not r['checkpoint_admitted']
assert r['n']==24 and r['cells']==1621728 and r['mpi_size']==48 and r['dt']==1/6000
checks = [helper.control.evaluate_check(r,c) for c in plan['commands'][0]['checks']
          if c['path'] not in [['status'],['checkpoint_admitted']]]
assert all(c['passed'] for c in checks)
for name, meta in case['artifacts'].items():
    f=RAW/name
    assert f.stat().st_size==meta['bytes'] and sha(f)==meta['sha256'],name
cp=P/'output/puncture.checkpoint'
with cp.open('rb') as f: header=struct.unpack('<16s4i3Q2d',f.read(72))
assert header[:7]==(b'HYP3D_MPI_BC2\0\0\0',24,31,2,0,1621728,6*24*24*31)
assert header[7]==r['steps'] and header[8]==r['t'] and header[9]==r['dt']
with cp.open('rb') as f:
    f.seek(168);boost=struct.unpack('<5d',f.read(40))
assert boost[0]==.1 and list(boost[1:4])==r['puncture_position'] and boost[4]==.16
payload=np.memmap(cp,dtype='<f8',mode='r',offset=208)
assert np.isfinite(payload).all()
save('local-verification.json',dict(all_artifacts_preserved=True,artifacts=case['artifacts'],
    plan_sha256=binding['plan_sha256'],source_manifest_sha256=binding['source_manifest_sha256'],
    binaries=binding['binaries'],checkpoint_admitted=False,resume_permitted=False,
    scheduler_exit=1,native_exit=2,report_scientific_checks_passed=True,
    finite_checkpoint_payload=True,tracker_report_header_agree=True))
for name in ['plan.json','binding.json','submission.json','worker-status.json','collected.json']:
    shutil.copy2(P/name,OUT/name)
for name in ['puncture.report.json','puncture.history.csv','puncture.log',
             'puncture.fastflow.horizon_summary_0.txt','puncture.fastflow.horizon_shape_0.txt']:
    shutil.copy2(P/'output'/name,OUT/name)

u=np.fromfile(P/'output/puncture.failed-stage.bin',dtype='<f8').reshape(-1,31)
n=r['cells'];assert len(u)==n+6*24*24 and np.isfinite(u).all()
g=np.zeros((len(u),3,3))
for field,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):
    g[:,i,j]=g[:,j,i]=u[:,field+1]+(i==j)
eig=np.linalg.eigvalsh(g)
bad=np.flatnonzero(eig[:n,0]<=0)
assert len(bad)>0 and all(bad<64*24**3), 'Unexpected non-core failure; inspect separately'
tracker=np.array(r['puncture_position']);locations=[]
for cell in bad:
    block=int(cell)//24**3;q=int(cell)%24**3
    # Nominal block/cell x-fast enumeration precedes every overlap cell.
    xyz=-.18+.09*np.array([block%4,(block//4)%4,block//16])
    xyz+=.09/24*(np.array([q%24,(q//24)%24,q//576])+.5)
    radius=float(np.linalg.norm(xyz))
    locations.append(dict(cell=int(cell),xyz=xyz.tolist(),r=radius,
        distance_to_last_accepted_tracker=float(np.linalg.norm(xyz-tracker)),
        metric_eigenvalues=eig[cell].tolist(),metric_determinant=float(np.linalg.det(g[cell])),
        chi=float(u[cell,0]),conformal_lapse=float(u[cell,18]+(1+radius**2)/2),
        beta=(u[cell,22:25]-xyz).tolist(),D=u[cell,25:28].tolist()))
with (P/'output/puncture.history.csv').open() as f: rows=list(csv.DictReader(f))
from post_horizon import classify
post=[classify(x,.01) for x in rows]
summary=dict(last_accepted_t_over_M=r['t_over_M'],steps=r['steps'],native_seconds=r['seconds'],
    last_diagnostic_t_over_M=float(rows[-1]['t_over_M']),error=r['error'],
    failed_stage_abscissa='Not separately recorded; report time is last accepted step.',
    capture_finite=True,nonpositive_volume_metric_cells=len(bad),locations=locations,
    surface_min_metric_eigenvalue=float(eig[n:,0].min()),puncture_position=tracker.tolist(),
    support_margin=.16-float(abs(tracker).max()),cube_margin=.18-float(abs(tracker).max()),
    null_surface_max=r['max_null_surface'],null_ghost_max=r['max_null_ghost'],
    pole_max=r['max_pole'],clock_max=r['max_clock'],max_cfl=r['max_cfl'],
    chi_floors_stage=r['chi_floor_stage_interventions'],chi_floors_final=r['chi_floor_acceptance_interventions'],
    lapse_floor_nodes=r['lapse_floor_node_interventions'],
    surface_floors_stage=r['surface_chi_floor_stage_interventions'],surface_floors_final=r['surface_chi_floor_final_interventions'],
    history_rows=len(rows),online_accepted_horizon_rows=sum(x['horizon_found']=='1' for x in rows),
    post_expansion_pass_rows=sum(x['post_expansion_pass'] for x in post),
    last_expansion_rms=float(rows[-1]['horizon_trial_expansion_rms']),
    qualification='Immediate finite indefinite metric localized in Cartesian core, not a proven root cause. No admitted endpoint or failed-state restart. N16 survived10M; N20/N24 failures exclude a claim that uniform refinement monotonically cures this instability. Space/time and rank count differ. Post1e-2 expansion pass is not full native finder acceptance; missing masses remain missing.')
save('summary.json',summary)
print(json.dumps(summary,indent=2))
