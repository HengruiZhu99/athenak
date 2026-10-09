#!/usr/bin/env python3
"""HELD analytic projected-constraint point readback; existing API, no compile."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,shutil,subprocess,sys,time
O=Path(__file__).resolve().parent;R=O.parents[2]
P=R/'boundary/total-j-finite-rb-control-20261009'
BASE=O.parent/'readback_projected_defect.py'
FD=R/'continuum/finite-rb-constraint-rate-oracle/run_constraint_rates_v2.py'
OP=P/'J0-N8-rb.98-segmentedQ64-a12x24-refinement001/operator.npz'
EXE=P/'radial-bridge-release'
OLD=O.parent/'attempt-1791558113840144000'
HALF=O.parent/'refined-h0-half/attempt-1791558482849376000'
MAP=OLD/'point-map-readback/raw22-point-maps.npz'
SCHEMA=OLD/'point-map-readback/schema-and-ordering.json'
PINS={str(BASE):'abd42a99db4d53fa52a546ce2f316f4d5b83686f3ed6911fa2585b895212ebcd',
 str(FD):'cbd7b341492f5f920ddb3935be1c78e83bcc0a2c7ab6f33e26d23cb28d484f31',
 str(OP):'2ed0da45a995669f7e3e2fedba231eeda0dcb06f43577125d4a194974c4f4742',
 str(EXE):'2293e9be6f75042f926f22232039c3c3bdd28826eb9e80061905c272b7adce15',
 str(P/'constraint_rate_api.hpp'):'d60ea8266abebf5263eb78b81997b60af472dc7b1fe102bd6f2c21baca1a6017',
 str(P/'radial_bridge.cpp'):'65cc3df6ff7655a28beb61aab445055f7d84123c07101e6e4e5cfd6ab4251438',
 str(MAP):'8f419ece7c5e6b0b6318adefd4fca621d200bfb3019410d14ead849ce086231c',
 str(SCHEMA):'53e58a192492bdf23a069e51b1b36415f246309156c3f9934c0394c5c6b0e779',
 str(OLD/'receipt.json'):'02d1610281ba84f6451ca4363076e302cb1f8bed7822c67df1d5c35c3c0c17a5'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def plan():
    return {'status':'HELD pending source review and exact authorization','J':0,'N':8,'rb':.98,
      'centers':21,'witnesses':14,'query_envelopes':[0,1,2,3,4,5,6],
      'basis_envelopes':[0,1,2],'held_controls':[3,4,5,6],'query_rows':1176,
      'query_schema':'J m channel phase envelope x y z -> actual RHS22 plus physical initial constraints8',
      'analytic_outputs':['initial','bulk','SAT','total'],
      'general_nongauge_continuum_comparator_unresolved':True,
      'both_original_FD_gates_remain_failed':True,'no_compile_spectrum_propagation':True,
      'pins':PINS,'detail_file':str(O/'PLAN.md')}

def execute(args):
    import numpy as np
    from scipy.special import roots_jacobi,eval_jacobi
    import warnings
    np.seterr(all='raise');warnings.filterwarnings('error',category=RuntimeWarning)
    for name,value in PINS.items():
        if sha(name)!=value:raise RuntimeError('source/data pin changed: '+name)
    auth=json.loads(args.authorization.read_text())
    if auth.get('analytic_projected_constraint_points_admitted') is not True:raise RuntimeError('held scientific stage')
    for key,value in [('driver_sha256',sha(__file__)),('executable_sha256',sha(EXE)),('operator_sha256',sha(OP))]:
        if auth.get(key)!=value:raise RuntimeError('authorization mismatch '+key)
    base=load_module('pinned_projection_driver',BASE);fd=load_module('pinned_rate_driver',FD)
    attempt=O/('attempt-'+str(time.time_ns()));attempt.mkdir();(attempt/'calls').mkdir()
    paths=[Path(p) for p in PINS]+[Path(__file__),O/'PLAN.md',args.authorization,
      HALF/'receipt.json',OLD/'cases.jsonl',HALF/'cases.jsonl',
      P/'actual_bridge.cpp',P/'all_m_data.hpp',P/'baseline_dual_spatial.hpp',P/'spatial_dual.hpp',
      P/'configuration_rows.hpp',P/'inputs/basis/basis-data.json',P/'build-release-latest.json',
      R.parent/'docs/hyperboloidal-continuum-constraint-rate-audit.md',
      R/'continuum/finite-rb-constraint-rate-oracle/immutable-finite-rb-C0-constraint-rates-20261009/index.json']
    before={str(p.resolve()):sha(p) for p in paths};write(attempt/'source-before.json',before)
    shutil.copyfile(__file__,attempt/Path(__file__).name);shutil.copyfile(O/'PLAN.md',attempt/'PLAN.md')
    api=fd.API(EXE,attempt);start=time.monotonic();failure=None;cases=[];controls={}
    launch=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    try:
        for previous in (OLD,HALF):
            prior=json.loads((previous/'receipt.json').read_text())
            assert prior['sources_unchanged'] and prior['passed_projection_defect_readback_gate'] is False
        with np.load(OP,allow_pickle=False) as saved:
            arrays={k:saved[k].copy() for k in ('nodal_from_modal','Jbulk','Jsat','E','Kweak','SATload')}
        for key,value in arrays.items():
            assert value.shape==(64,64) and value.dtype.kind=='f' and np.isfinite(value).all()
        T,Jbulk,Jsat=arrays['nodal_from_modal'],arrays['Jbulk'],arrays['Jsat']
        matrix_checks={'bulk_Riesz':base.error(np.einsum('ij,jk->ik',arrays['E'],Jbulk,optimize=False),arrays['Kweak']),
          'SAT_Riesz':base.error(np.einsum('ij,jk->ik',arrays['E'],Jsat,optimize=False),arrays['SATload'])}
        assert all(v['scaled_l2']<=2e-9 for v in matrix_checks.values())
        nodes=(roots_jacobi(8,0,.5)[0]+1)*base.B/2;node_modes=base.modal(nodes,np,eval_jacobi)
        Tcheck=np.zeros_like(T)
        for c in range(8):Tcheck[c*8:(c+1)*8,c*8:(c+1)*8]=node_modes[:,c,0,:]
        matrix_checks['analytic_T']=base.error(Tcheck,T);assert matrix_checks['analytic_T']['scaled_l2']<=5e-11
        write(attempt/'matrix-schema-checks.json',matrix_checks)
        centers=[]
        for r in [.15,.30,.50,.70,.90,.96,.975]:
            for direction,n in enumerate(fd.DIRS):centers.append({'r':r,'direction':direction,'x':tuple(r*v for v in n)})
        points=np.asarray([v['x'] for v in centers]);rho=np.asarray([math.fsum(v*v for v in x) for x in points])
        queries=[[0,0,c,0,env,*x] for x in points for c in range(8) for env in range(7)]
        data=np.asarray(api.run('--manufactured-rate-batch',queries,30)).reshape(21,8,7,30)
        assert np.isfinite(data).all()
        maps=np.empty((21,8,3,30))
        maps[:,:,0,:]=data[:,:,0,:]
        maps[:,:,1,:]=data[:,:,1,:]-rho[:,None,None]*maps[:,:,0,:]
        maps[:,:,2,:]=.5*(data[:,:,2,:]-rho[:,None,None]**2*maps[:,:,0,:]-2*rho[:,None,None]*maps[:,:,1,:])
        assert np.isfinite(maps).all()
        held_actual=[];held_expected=[]
        for p,z in enumerate(rho):
            for c in range(8):
                for env in (3,4,5,6):
                    held_actual.append(data[p,c,env]);held_expected.append(np.einsum('df,d->f',maps[p,c],base.envelope(float(z),env),optimize=False))
        held_actual=np.asarray(held_actual);held_expected=np.asarray(held_expected)
        held_global=base.error(held_actual,held_expected)
        held_rows=[base.error(u,v)['scaled_l2'] for u,v in zip(held_actual,held_expected)]
        held_q_global=base.error(held_actual[:,22:],held_expected[:,22:])
        held_q_rows=[base.error(u[22:],v[22:])['scaled_l2'] for u,v in zip(held_actual,held_expected)]
        assert held_global['scaled_l2']<=5e-11 and max(held_rows)<=5e-11
        assert held_q_global['scaled_l2']<=5e-11 and max(held_q_rows)<=5e-11
        with np.load(MAP,allow_pickle=False) as cached:
            cache_points=cached['points'];cache_rhs=cached['rhs']
            assert cache_points.shape==(4809,3) and cache_rhs.shape==(4809,8,3,22)
            assert np.isfinite(cache_points).all() and np.isfinite(cache_rhs).all()
            lookup={tuple(x):i for i,x in enumerate(cache_points)}
            expected_rhs=np.asarray([cache_rhs[lookup[tuple(x)]] for x in points])
        rhs_global=base.error(maps[:,:,:,:22],expected_rhs)
        rhs_rows=[base.error(u,v)['scaled_l2'] for u,v in zip(maps[:,:,:,:22].reshape(-1,22),expected_rhs.reshape(-1,22))]
        assert rhs_global['scaled_l2']<=5e-11 and max(rhs_rows)<=5e-11
        controls={'held_envelope_rows':len(held_rows),'held_all30_global':held_global,
          'held_all30_max_row_scaled':max(held_rows),'held_constraints8_global':held_q_global,
          'held_constraints8_max_row_scaled':max(held_q_rows),'source_rhs22_global':rhs_global,
          'source_rhs22_max_row_scaled':max(rhs_rows),'query_rows':len(queries),
          'all_eight_fields_are_physical_Cartesian':True,'Omega_rescaling':False}
        write(attempt/'held-map-controls.json',controls)
        np.savez_compressed(attempt/'analytic-center-maps.npz',points=points,rhs=maps[:,:,:,:22],constraints=maps[:,:,:,22:])
        comparisons=[]
        for previous in (OLD,HALF):
            rows=[json.loads(s) for s in (previous/'cases.jsonl').read_text().splitlines()]
            comparisons.append((str(previous),{(r['witness']['name'],r['r'],r['direction']):r for r in rows}))
        modes=base.modal(rho,np,eval_jacobi);coefficients=[]
        with (attempt/'cases.jsonl').open('w') as out:
            for witness in base.witnesses():
                nodal=np.zeros(64)
                for c,weight in witness['channels']:
                    nodal[c*8:(c+1)*8]=[weight*base.envelope(float(z),witness['envelope'])[0] for z in nodes]
                X=np.linalg.solve(T,nodal);assert np.isfinite(X).all()
                interpolation=base.error(np.einsum('ij,j->i',T,X,optimize=False),nodal)
                assert interpolation['scaled_l2']<=2e-9
                bulk=np.einsum('ij,j->i',Jbulk,X,optimize=False);sat=np.einsum('ij,j->i',Jsat,X,optimize=False)
                coeff={'initial':X,'bulk':bulk,'sat':sat,'total':bulk+sat};vectors={}
                for label,value in coeff.items():
                    assert np.isfinite(value).all()
                    wjet=np.einsum('pcdk,ck->pcd',modes,value.reshape(8,8),optimize=False)
                    vectors[label]=np.einsum('pcdq,pcd->pq',maps[:,:,:,22:],wjet,optimize=False)
                    assert np.isfinite(vectors[label]).all()
                coefficients.append({'witness':witness,'X':X.tolist(),'Ybulk':bulk.tolist(),'Ysat':sat.tolist(),
                    'interpolation_nodal':interpolation})
                for p,center in enumerate(centers):
                    q={label:value[p].tolist() for label,value in vectors.items()}
                    checks={'total_linearity':fd.errors(q['total'],[u+v for u,v in zip(q['bulk'],q['sat'])])['scaled_l2']<=5e-11}
                    if witness['gauge']:checks['initial_gauge_zero']=fd.norm(q['initial'])<=5e-11
                    old_comparisons=[]
                    for name,oldrows in comparisons:
                        row=oldrows.get((witness['name'],center['r'],center['direction']))
                        if row is None:continue
                        old_comparisons.append({'attempt':name,'old_case_passed':row['passed'],
                          'old_failed_checks':[k for k,v in row['checks'].items() if not v],
                          'comparisons':{label:{'final':fd.errors(row['sequences'][label][-1],q[label]),
                            'richardson':fd.errors(row['sequence_info'][label]['richardson_last'],q[label])}
                            for label in ('bulk','sat','total','initial')}})
                    case={'witness':witness,**center,'physical8':q,'checks':checks,
                      'passed_analytic_point_controls':all(checks.values()),'old_FD_readback':old_comparisons,
                      'continuum_actual_source_numerically_evaluated':False,
                      'nongauge_continuum_comparator_unresolved':not witness['gauge'],
                      'gauge_continuum_zero_is_derived_Einstein_sector_tangency':witness['gauge'],
                      'gauge_projection_defect_derived':{k:q[k] for k in ('bulk','sat','total')} if witness['gauge'] else None,
                      'not_a_physical_energy_or_CPBC_measurement':True}
                    out.write(json.dumps(case,allow_nan=False)+'\n');out.flush();cases.append(case)
                    if not case['passed_analytic_point_controls']:raise RuntimeError('analytic projected point-control failure')
                print(witness['name']+': saved '+str(len(cases))+' analytic points',flush=True)
        write(attempt/'coefficients.json',coefficients)
    except Exception as e:failure={'type':type(e).__name__,'message':str(e)}
    write(attempt/'calls.json',api.calls);after={str(p.resolve()):sha(p) for p in paths}
    result={'passed_analytic_projected_constraint_point_gate':failure is None and before==after,
      'full_fourteen_witness_projection_defect_gate_passed':False,
      'reason_full_gate_unresolved':'General nongauge C_ref[L_actual Phi(X)] is not evaluated here; both ordinary-FD runs remain failed.',
      'launch_HEAD':launch,'command':[sys.executable,*sys.argv],'source_before':before,'source_after':after,
      'sources_unchanged':before==after,'error':failure,'cases':len(cases),'calls':len(api.calls),
      'seconds':time.monotonic()-start,'controls':controls,'scope':plan()}
    write(attempt/'receipt.json',result);print(json.dumps(result,indent=2),flush=True)
    if not result['passed_analytic_projected_constraint_point_gate']:raise SystemExit(1)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--authorization',type=Path);args=parser.parse_args()
    if not args.execute:print(json.dumps(plan(),indent=2));return
    if not args.authorization:raise RuntimeError('exact root authorization required')
    execute(args)
if __name__=='__main__':main()
