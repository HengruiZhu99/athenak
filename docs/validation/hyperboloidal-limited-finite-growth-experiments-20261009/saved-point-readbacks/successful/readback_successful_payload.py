#!/usr/bin/env python3
"""HELD saved successful finite-matrix payload physical8 samples; no evolution."""
from pathlib import Path
import argparse,json,subprocess,sys,time,warnings
HERE=Path(__file__).resolve().parent
HELPER=HERE.parent/'finite-rb-saved-point-readback-20261009/point_constraints.py'
sys.path.insert(0,str(HELPER.parent));import point_constraints as pc
HELPER_SHA='99f1679c07ec7079e2386bbd54194772329a77e2a0d60e3eb9b81f10b0bee687'
TIMES=(0.,.25,.5,1.,2.,4.,6.)

def execute(args):
    import numpy as np
    from scipy.special import roots_jacobi,eval_jacobi
    warnings.filterwarnings('error',category=RuntimeWarning);np.seterr(all='raise',under='ignore')
    auth=json.loads(args.authorization.read_text())
    if auth.get('successful_payload_point_readback_admitted') is not True or auth.get('N')!=args.N:
        raise RuntimeError('HELD source lacks exact degree admission')
    for key,value in [('driver_sha256',pc.sha(__file__)),('helper_sha256',HELPER_SHA),('plan_sha256',pc.sha(HERE/'PLAN.md'))]:
        if auth.get(key)!=value:raise RuntimeError('authorization pin mismatch '+key)
    if pc.sha(HELPER)!=HELPER_SHA:raise RuntimeError('shared helper source changed')
    source_receipt=Path(auth['growth_receipt_path']);payload=Path(auth['payload_path']);op=Path(auth['operator_path'])
    pins={**pc.PINS,str(HELPER):HELPER_SHA,str(source_receipt):auth['growth_receipt_sha256'],
          str(payload):auth['payload_sha256'],str(op):auth['operator_sha256']};pc.verify_pins(pins)
    paths=list(map(Path,pins))+[Path(__file__),HERE/'PLAN.md',args.authorization]
    before={str(p.resolve()):pc.sha(p) for p in paths};args.output.mkdir(parents=True,exist_ok=False)
    receipt={'command':[sys.executable,*sys.argv],'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'source_before':before,'J':0,'N':args.N,'rb':.98,'error':None,'passed_saved_successful_payload_point_readback':False,
       'original_SciPy_expm_attempt_remains_failed':True,'both_ordinary_FD_attempts_remain_failed':True,
       'general_nongauge_continuum_comparator_unresolved':True,'physical8_order':pc.FIELDS,
       'scope':'21 sampled physical Cartesian constraint vectors from saved finite-matrix modes and states; no integrated energy, continuum classification, or new growth execution.'}
    pc.write(args.output/'launch.json',receipt);begin=time.monotonic()
    try:
        original=json.loads(source_receipt.read_text());N=args.N;dim=8*N
        assert (original['J'],original['N'],original['rb'])==(0,N,.98)
        assert original['passed_finite_ODE_numerical_checks'] is True and original['error'] is None
        assert original['input_pins_unchanged'] is True
        assert original['original_SciPy_expm_attempt_remains_failed'] is True
        assert original['both_original_FD_attempts_remain_failed'] is True
        assert original['general_nongauge_continuum_comparator_unresolved'] is True
        assert original['payload_sha256']==auth['payload_sha256']
        assert original['inputs_before'][str(op.resolve())]==auth['operator_sha256']
        assert tuple(item['time'] for item in original['propagation'])==TIMES
        receipt['growth_receipt_passed_finite_ODE_numerical_checks']=True
        receipt['growth_launch_HEAD']=original['launch_HEAD']
        with np.load(op,allow_pickle=False) as data:operator={k:data[k].copy() for k in data.files}
        checks,rho=pc.operator_checks(operator,N,.98,np,roots_jacobi,eval_jacobi);receipt['checks']=checks
        keys=('physical_seed_modal','seed_common_rho','selected_indices','selected_modal_modes',
              'selected_actual_Jv','eigenvalues','seed_states_modal','propagation_times')
        with np.load(payload,allow_pickle=False) as data:saved={k:data[k].copy() for k in keys}
        assert all(np.isfinite(a).all() for a in saved.values())
        assert saved['physical_seed_modal'].shape==(dim,8) and saved['selected_modal_modes'].shape==(dim,4)
        assert saved['selected_actual_Jv'].shape==(dim,4) and saved['selected_indices'].shape==(4,)
        assert saved['selected_indices'].dtype.kind in 'iu' and np.all((0<=saved['selected_indices'])&(saved['selected_indices']<dim))
        assert saved['eigenvalues'].shape==(dim,) and saved['seed_common_rho'].shape==(N,)
        assert saved['seed_states_modal'].shape==(7,dim,8)
        assert np.array_equal(saved['propagation_times'],np.array(TIMES))
        checks['saved_nodes']=pc.error(saved['seed_common_rho'],rho)
        checks['saved_t0_seed_state']=pc.error(saved['seed_states_modal'][0],saved['physical_seed_modal'])
        J=operator['Jbulk']+operator['Jsat'];m=saved['selected_modal_modes'];Jm=saved['selected_actual_Jv']
        checks['saved_actual_Jv']=pc.complex_error(pc.action_split(J,m,np),Jm)
        assert all(v['scaled_l2']<=2e-9 for v in checks.values())
        points,qmap,_=pc.load_maps(np);modes=pc.modal_jets(points,N,.98,np,eval_jacobi)
        columns={'seed':saved['physical_seed_modal'],'seed_J':pc.action_split(J,saved['physical_seed_modal'],np),
                 'selected_mode':m,'selected_actual_Jv':Jm}
        arrays={'points':points,'propagation_times':saved['propagation_times']};stats={};rows=[];values={}
        def sample(name,coefficients,time_value=None):
            out=pc.contract_split(qmap,modes,coefficients,np)
            independent=pc.scalar_contract_real(qmap,modes,coefficients.real,np)+1j*pc.scalar_contract_real(qmap,modes,coefficients.imag,np)
            checks['scalar_'+name]=pc.complex_error(out,independent)
            assert checks['scalar_'+name]['scaled_l2']<=5e-11
            stats[name]=pc.sample_stats(out)
            for column in range(coefficients.shape[1]):
                for p,x in enumerate(points):rows.append({'column_group':name,'column':column,'time':time_value,'point':p,'x':x.tolist(),
                    'physical8_real':out[p,:,column].real.tolist(),'physical8_imag':out[p,:,column].imag.tolist()})
            return out
        for name,coefficients in columns.items():
            values[name]=sample(name,coefficients)
            arrays[name+'_constraints_real']=values[name].real;arrays[name+'_constraints_imag']=values[name].imag
        states=[];rates=[]
        for t,state in zip(TIMES,saved['seed_states_modal']):
            states.append(sample('state_t'+str(t),state,t));rates.append(sample('Jstate_t'+str(t),pc.action_split(J,state,np),t))
        arrays['state_constraints_real']=np.array(states).real;arrays['state_constraints_imag']=np.array(states).imag
        arrays['Jstate_constraints_real']=np.array(rates).real;arrays['Jstate_constraints_imag']=np.array(rates).imag
        assert np.count_nonzero(values['seed'][:,:,[0,4]])==0
        selected_lambda=saved['eigenvalues'][saved['selected_indices']]
        checks['saved_lambda_constraint_residual_diagnostic_only']=pc.complex_error(values['selected_actual_Jv'],values['selected_mode']*selected_lambda[None,None,:])
        np.savez_compressed(args.output/'sample-physical8.npz',**arrays)
        (args.output/'cases.jsonl').write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in rows))
        pc.write(args.output/'summary.json',{'stats':stats,'selected_indices':saved['selected_indices'].tolist(),
          'selected_saved_eigenvalues':[[float(v.real),float(v.imag)] for v in selected_lambda],
          'checks':checks,'seed_metadata_from_growth_receipt':original['seeds'],
          'normalization':'Saved physical seed amplitudes and energy-normalized modes retained without renormalization.',
          'scope':receipt['scope'],'no_new_growth_execution':True})
        receipt.update({'case_count':len(rows),'saved_times':TIMES,'initial_gauge_zero_count':42,
            'outputs':[{'path':'sample-physical8.npz','sha256':pc.sha(args.output/'sample-physical8.npz'),'role':'large_payload'},
                       {'path':'cases.jsonl','sha256':pc.sha(args.output/'cases.jsonl'),'role':'source_or_receipt'},
                       {'path':'summary.json','sha256':pc.sha(args.output/'summary.json'),'role':'source_or_receipt'}]})
    except Exception as exc:receipt['error']={'type':type(exc).__name__,'message':str(exc)}
    receipt['source_after']={str(p.resolve()):pc.sha(p) for p in paths}
    receipt['sources_unchanged']=receipt['source_before']==receipt['source_after'];receipt['seconds']=time.monotonic()-begin
    receipt['passed_saved_successful_payload_point_readback']=receipt['error'] is None and receipt['sources_unchanged']
    pc.write(args.output/'receipt.json',receipt);print(json.dumps(receipt,indent=2,allow_nan=False))
    return 0 if receipt['passed_saved_successful_payload_point_readback'] else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--N',type=int,choices=(8,12,16));parser.add_argument('--authorization',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if not args.execute:print('HELD saved successful-payload adapter; exact per-degree authorization required.');return 0
    if args.N is None or args.authorization is None or args.output is None:parser.error('--N, --authorization and --output required')
    return execute(args)
if __name__=='__main__':sys.exit(main())
