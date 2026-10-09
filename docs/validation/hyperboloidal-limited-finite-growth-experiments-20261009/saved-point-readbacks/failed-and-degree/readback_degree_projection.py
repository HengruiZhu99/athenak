#!/usr/bin/env python3
"""HELD J0 N12/N16 analytic projected constraint point readback; no API queries."""
from pathlib import Path
import argparse,json,math,subprocess,sys,time,warnings
import point_constraints as pc
HERE=Path(__file__).resolve().parent;R=pc.R
DEGREE=R/'boundary/total-j-finite-rb-degree-control-20261009'
OPS={12:('1420df9235c165ee0ae9de04096c3e350aee248a622f14a4d254f1e01d0db0c8',
          '1ebc6c87ca67cc1abb998eaebfde7c6f60825ef44a95196960d485b685e6564d'),
     16:('466c6386ed4ece0698bd0652c8bf064102fbca11bc606717755ad49459fb0cb6',
          'f8247ff66846c3b36afe11fa47d4f09b14c627c1bb7dc8317f80fa40fc14b599')}
OLD=R/'continuum/finite-rb-projection-defect/readback_projected_defect.py'
OLD_SHA='abd42a99db4d53fa52a546ce2f316f4d5b83686f3ed6911fa2585b895212ebcd'

def witnesses():
    for name,c,env in [('poly-alpha',0,0),('poly-P',2,1),('poly-Theta',3,2),('poly-beta',4,3)]:
        yield {'name':name,'channels':[(c,1.)],'envelope':env,'gauge':c in (0,4)}
    yield {'name':'poly-all-mixed','channels':[(c,(-1.)**c/(c+1)) for c in range(8)],'envelope':6,'gauge':False}
    for c,name in [(0,'gauge-alpha'),(4,'gauge-beta')]:
        yield {'name':name,'channels':[(c,1.)],'envelope':4,'gauge':True}
    shell=[1,2,3,5,6,7]
    for c in shell:yield {'name':'shell-channel-'+str(c),'channels':[(c,1.)],'envelope':5,'gauge':False}
    yield {'name':'shell-mixed','channels':[(c,(-1.)**c/(c+1)) for c in shell],'envelope':5,'gauge':False}

def envelope(rho,env):
    if env==0:return 1.
    if env==1:return rho
    if env==2:return rho*rho
    if env==3:return rho**3
    if env==4:return math.exp(-8*rho)
    if env==5:return math.exp(-((rho-.49)/.16)**2)
    if env==6:return 1+rho/3-rho*rho/5+rho**3/7
    raise ValueError('undeclared envelope')

def execute(args):
    import numpy as np
    from scipy.special import roots_jacobi,eval_jacobi
    warnings.filterwarnings('error',category=RuntimeWarning);np.seterr(all='raise',under='ignore')
    N=args.N;folder=DEGREE/f'J0-N{N}-rb.98-segmentedQ64-a12x24-primary001';op=folder/'operator.npz';report=folder/'report.json'
    pins={**pc.PINS,str(op):OPS[N][0],str(report):OPS[N][1],str(OLD):OLD_SHA};pc.verify_pins(pins)
    auth=json.loads(args.authorization.read_text())
    if auth.get('degree_projected_point_readback_admitted') is not True or auth.get('N')!=N:
        raise RuntimeError('HELD degree source lacks admission')
    for key,value in [('driver_sha256',pc.sha(__file__)),('helper_sha256',pc.sha(HERE/'point_constraints.py')),
                      ('plan_sha256',pc.sha(HERE/'DEGREE-PLAN.md')),('operator_sha256',OPS[N][0])]:
        if auth.get(key)!=value:raise RuntimeError('authorization pin mismatch '+key)
    paths=list(map(Path,pins))+[Path(__file__),HERE/'point_constraints.py',HERE/'DEGREE-PLAN.md',args.authorization]
    before={str(p.resolve()):pc.sha(p) for p in paths};args.output.mkdir(parents=True,exist_ok=False)
    receipt={'command':[sys.executable,*sys.argv],'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'source_before':before,'J':0,'N':N,'rb':.98,'error':None,'passed_degree_projected_point_readback':False,
       'both_ordinary_FD_attempts_remain_failed':True,'general_nongauge_continuum_comparator_unresolved':True,
       'original_full_projection_defect_gate_passed':False,'physical8_order':pc.FIELDS,
       'scope':'Saved J0 polynomial projected fields at21 Cartesian centers; bulk/SAT/total split. No new API, generator spectrum, propagation or CPBC test.'}
    pc.write(args.output/'launch.json',receipt);begin=time.monotonic()
    try:
        checked=json.loads(report.read_text());assert (checked['J'],checked['N'],checked['rb'])==(0,N,.98)
        assert checked['passed_single_quadrature_algebra'] is True and checked['operator_sha256']==OPS[N][0]
        with np.load(op,allow_pickle=False) as saved:data={k:saved[k].copy() for k in saved.files}
        checks,rho=pc.operator_checks(data,N,.98,np,roots_jacobi,eval_jacobi)
        receipt['checks']=checks
        points,qmap,_=pc.load_maps(np);modes=pc.modal_jets(points,N,.98,np,eval_jacobi);witness=list(witnesses())
        assert len(witness)==14
        nodal=np.zeros((8*N,14))
        for v,item in enumerate(witness):
            for c,weight in item['channels']:nodal[c*N:(c+1)*N,v]=weight*np.array([envelope(float(x),item['envelope']) for x in rho])
        X=np.linalg.solve(data['nodal_from_modal'],nodal);assert np.isfinite(X).all()
        checks['nodal_seed_solve']=pc.error(np.einsum('ij,jv->iv',data['nodal_from_modal'],X,optimize=False),nodal)
        assert checks['nodal_seed_solve']['scaled_l2']<=2e-9
        Ybulk=np.einsum('ij,jv->iv',data['Jbulk'],X,optimize=False)
        Ysat=np.einsum('ij,jv->iv',data['Jsat'],X,optimize=False);Ytotal=Ybulk+Ysat
        columns={'initial':X,'bulk':Ybulk,'SAT':Ysat,'total':Ytotal};arrays={'points':points,'rho_nodes':rho,'X':X,'Ybulk':Ybulk,'Ysat':Ysat,'Ytotal':Ytotal}
        stats={};constraints={}
        for name,value in columns.items():
            constraints[name]=pc.contract_real(qmap,modes,value,np)
            checks['scalar_'+name]=pc.error(constraints[name],pc.scalar_contract_real(qmap,modes,value,np))
            assert checks['scalar_'+name]['scaled_l2']<=5e-11
            arrays[name+'_physical8']=constraints[name];stats[name]=pc.sample_stats(constraints[name])
        checks['bulk_SAT_total_linearity']=pc.error(constraints['bulk']+constraints['SAT'],constraints['total'])
        assert checks['bulk_SAT_total_linearity']['scaled_l2']<=5e-11
        gauge_indices=[v for v,item in enumerate(witness) if item['gauge']]
        assert len(gauge_indices)==4 and np.count_nonzero(constraints['initial'][:,:,gauge_indices])==0
        held=np.linspace(0.,.98**2,2*N+3);heldm=pc.radial_jets(held,N,.98,np,eval_jacobi);interpolation=[]
        for v,item in enumerate(witness):
            errors=[]
            for c,weight in item['channels']:
                fit=np.einsum('pk,k->p',heldm[:,c,0,:],X[c*N:(c+1)*N,v],optimize=False)
                original=np.array([weight*envelope(float(x),item['envelope']) for x in held]);errors.append({'channel':c,**pc.error(fit,original)})
            interpolation.append({'name':item['name'],'scalar_envelope_errors':errors})
        rows=[]
        for v,item in enumerate(witness):
            for p,x in enumerate(points):rows.append({'witness':item['name'],'point':p,'x':x.tolist(),'gauge':item['gauge'],
                'initial_physical8':constraints['initial'][p,:,v].tolist(),
                'bulk_physical8':constraints['bulk'][p,:,v].tolist(),'SAT_physical8':constraints['SAT'][p,:,v].tolist(),
                'total_physical8':constraints['total'][p,:,v].tolist(),
                'continuum_comparator':'derived gauge Einstein-sector zero at Omega>0' if item['gauge'] else 'unresolved; no continuum defect assigned'})
        np.savez_compressed(args.output/'projected-physical8.npz',**arrays)
        (args.output/'cases.jsonl').write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in rows))
        pc.write(args.output/'summary.json',{'checks':checks,'stats':stats,'witnesses':witness,'interpolation':interpolation,
            'interpolation_scope':'Scalar envelopes at2N+3 held rho; initial data are the N-specific polynomial interpolants, not the original nonpolynomial envelopes.',
            'no_small_projection_or_SAT_defect_demand':True,'general_nongauge_continuum_comparator_unresolved':True})
        receipt.update({'case_count':len(rows),'initial_gauge_zero_count':len(points)*len(gauge_indices),'checks':checks,
          'outputs':[{'path':'projected-physical8.npz','sha256':pc.sha(args.output/'projected-physical8.npz'),'role':'large_payload'},
                     {'path':'cases.jsonl','sha256':pc.sha(args.output/'cases.jsonl'),'role':'source_or_receipt'},
                     {'path':'summary.json','sha256':pc.sha(args.output/'summary.json'),'role':'source_or_receipt'}]})
    except Exception as exc:receipt['error']={'type':type(exc).__name__,'message':str(exc)}
    receipt['source_after']={str(p.resolve()):pc.sha(p) for p in paths}
    receipt['sources_unchanged']=receipt['source_before']==receipt['source_after'];receipt['seconds']=time.monotonic()-begin
    receipt['passed_degree_projected_point_readback']=receipt['error'] is None and receipt['sources_unchanged']
    pc.write(args.output/'receipt.json',receipt);print(json.dumps(receipt,indent=2,allow_nan=False))
    return 0 if receipt['passed_degree_projected_point_readback'] else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--N',type=int,choices=(12,16));parser.add_argument('--authorization',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if not args.execute:print('HELD source-only degree adapter; separate N12/N16 authorization required.');return 0
    if args.N is None or args.authorization is None or args.output is None:parser.error('--N, --authorization and --output required')
    return execute(args)
if __name__=='__main__':sys.exit(main())
