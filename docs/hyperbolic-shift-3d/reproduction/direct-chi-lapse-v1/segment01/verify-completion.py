from pathlib import Path
import json,hashlib,struct,math
import numpy as np
base=Path('output/hyperbolic-core-shell/direct-chi-lapse-v1/segment01');worker=json.loads((base/'worker-status.json').read_text());binding=json.loads((base/'binding.json').read_text());submission=json.loads((base/'submission.json').read_text());collected=json.loads((base/'collected.json').read_text())
assert worker['status']=='passed' and worker['source_manifest_sha256']==submission['source_manifest_sha256']=='56706997821cb434e142cbac3a217ac7360d4b7fdfc2efe9858e97df9f191b29'
assert worker['binaries']=={'build-sycl-mpi-direct-chi-lapse-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_direct_chi_lapse':'b8814162c43d2319f9d6db2370b54f270f592a5f7101adf36bf94a83b101d5b8'}
assert worker['plan_sha256']==submission['plan_sha256']==hashlib.sha256((base/'plan.json').read_bytes()).hexdigest()
verified=[]
for path,record in worker['cases'][0]['artifacts'].items():
 p=base/'output'/Path(path).name
 assert p.stat().st_size==record['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
 verified.append(dict(path=path,**record))
r=json.loads((base/'output/puncture.report.json').read_text());p=base/'output/puncture.checkpoint';data=p.read_bytes();h=struct.unpack('<16s4i3Q2d',data[:72]);magic,n,nf,kind,admitted,cells,surface,steps,t,dt=h
assert magic.rstrip(b'\0')==b'HYP3D_MPI_AC1' and (n,nf,kind,admitted)==(16,31,2,1)
assert r['status']=='completed' and r['checkpoint_admitted'] and r['diagnostics_current'] and t==r['t']==r['target_time']==20/3 and r['t_over_M']==20 and steps==r['steps']==13335 and dt==r['dt']==.0005
initial=(base.parent/'puncture-ac1-at3m.checkpoint').read_bytes();assert data[72:168]==initial[72:168]
assert struct.unpack('<i',data[144:148])[0]==3 and struct.unpack('<2d',data[152:168])==(1e-8,.03)
assert cells==159200 and surface==47616 and len(data)==168+16*(31*cells+surface)
values=np.frombuffer(data,dtype='<f8',offset=168);assert np.isfinite(values).all()
u=values[:31*cells].reshape(cells,31);surf=values[62*cells:62*cells+surface].reshape(-1,31);assert (u[:,0]>0).all() and (surf[:,0]>0).all() and r['min_lapse']>0
# Independently check the final conformal metric is positive definite.
g=np.zeros((cells,3,3));g[:,0,0]=1+u[:,1];g[:,1,1]=1+u[:,4];g[:,2,2]=1+u[:,6];g[:,0,1]=g[:,1,0]=u[:,2];g[:,0,2]=g[:,2,0]=u[:,3];g[:,1,2]=g[:,2,1]=u[:,5];eigmin=float(np.linalg.eigvalsh(g)[:,0].min());assert eigmin>0
seed=(base/'output/puncture.checkpoint.horizon-seed.txt').read_text().split();assert seed[0]=='HYP3D_FASTFLOW_SEED_V1' and float(seed[1])==t and int(seed[2])==steps and float(seed[3])==r['diagnostic_time'] and float(seed[4])==r['horizon_radius']
summary={}
for region in ['volume','surface']:
 for f in [0,18]:
  a=[e for e in r['candidate_sign_events'] if (e[0]>=0)==(region=='volume') and e[1]==f]
  summary[f'{region}_{"chi" if f==0 else "lapse"}']=dict(cell_field_pairs=len(a),negative_stage=sum(e[5] for e in a),negative_final=sum(e[6] for e in a),zero_stage=sum(e[7] for e in a),zero_final=sum(e[8] for e in a),minimum=min((e[9] for e in a),default=None),first_time_over_M=min((3*e[10] for e in a),default=None),max_radius=max((math.sqrt(sum(x*x for x in e[2:5])) for e in a),default=None))
keys=['min_chi','min_lapse','H_L2','M_L2','H_exterior_L2','M_exterior_L2','Theta_exterior_L2','Zr_exterior_L2','horizon_found','horizon_complete_surface','horizon_radius','horizon_areal_radius','horizon_mass','horizon_irreducible_mass','horizon_expansion_rms','max_null_surface','max_null_ghost','lapse_floor_node_interventions','lapse_floor_max_radius','chi_floor_stage_interventions','chi_floor_acceptance_interventions','chi_floor_max_radius','surface_chi_floor_stage_interventions','surface_chi_floor_final_interventions','scri_interpolated_chi_negative_count','scri_interpolated_chi_zero_count','scri_interpolated_chi_minimum','rhs_seconds','exchange_seconds','closure_seconds','diagnostic_seconds','bulk_halo_seconds']
result=dict(status='verified_completed_20M',t_over_M=20,checkpoint=dict(schema='HYP3D_MPI_AC1',admitted=True,header_bytes=168,config_matches_initial=True,all_payload_finite=True,positive_chi_volume_and_surface=True,minimum_metric_eigenvalue=eigmin,steps=steps,segment_steps=r['segment_steps']),artifacts_verified=verified,candidate_summary=summary,final_diagnostics={k:r[k] for k in keys},native_seconds=r['seconds'],worker_case_seconds=worker['cases'][0]['elapsed_seconds'],evolved_zone_cycles_per_second=cells*r['segment_steps']/r['seconds'],horizon_areal_radius_fractional_error_vs_2M=r['horizon_areal_radius']/(2/3)-1)
(base/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='artifacts_verified'},indent=2))
