#!/usr/bin/env python3
"""HELD saved failed-payload physical8 point diagnostics; no growth execution."""
from pathlib import Path
import argparse,json,subprocess,sys,time,warnings
import point_constraints as pc

HERE=Path(__file__).resolve().parent;R=pc.R
G=R/'continuum/finite-rb-limited-matrix-growth-20261009'
PAYLOAD=G/'J0-N8-rb98-growth001/failed-partial-payload.npz'
FAILED=PAYLOAD.parent/'receipt.json'
ADMISSION=G/'J0-N8-rb98-admission001.json'
OP=R/'boundary/total-j-finite-rb-control-20261009/J0-segmented-Q64-sector-readback001/operator.npz'
PINS={**pc.PINS,str(PAYLOAD):'1f9036edbba94142ec89145e069bb929cd4e5ae2f203e12d9b293c8dc957567b',
 str(FAILED):'2774bc40f893d8d5fc1f8667a039546624dcfd99f932253c0c3ae2a2eaae1537',
 str(ADMISSION):'fa1f07b8a8595679f54785ce7f7457af1b52df4d562b3fa966eb72ca542ff13e',
 str(OP):'8b4b9a5b33151d86359aa9e35f0ae44dc436ef4ab2d6c17796e77983f9422a27'}

def execute(args):
    import numpy as np
    from scipy.special import roots_jacobi,eval_jacobi
    warnings.filterwarnings('error',category=RuntimeWarning);np.seterr(all='raise',under='ignore')
    pc.verify_pins(PINS)
    auth=json.loads(args.authorization.read_text())
    if auth.get('failed_payload_point_readback_admitted') is not True:raise RuntimeError('HELD source lacks admission')
    for key,value in [('driver_sha256',pc.sha(__file__)),('helper_sha256',pc.sha(HERE/'point_constraints.py')),
                      ('plan_sha256',pc.sha(HERE/'GROWTH-PLAN.md')),('payload_sha256',PINS[str(PAYLOAD)]),
                      ('operator_sha256',PINS[str(OP)])]:
        if auth.get(key)!=value:raise RuntimeError('authorization pin mismatch '+key)
    paths=list(map(Path,PINS))+[Path(__file__),HERE/'point_constraints.py',HERE/'GROWTH-PLAN.md',args.authorization]
    before={str(p.resolve()):pc.sha(p) for p in paths};args.output.mkdir(parents=True,exist_ok=False)
    receipt={'command':[sys.executable,*sys.argv],'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'source_before':before,'error':None,'passed_saved_payload_point_readback':False,
       'growth_propagation_passed':False,'both_ordinary_FD_attempts_remain_failed':True,
       'general_nongauge_continuum_comparator_unresolved':True,'physical8_order':pc.FIELDS,
       'scope':'21 physical Cartesian constraint samples of saved finite-matrix columns; no integrated energy, continuum mode or growth execution.'}
    pc.write(args.output/'launch.json',receipt);begin=time.monotonic()
    try:
        failed=json.loads(FAILED.read_text());adm=json.loads(ADMISSION.read_text())
        assert (failed['J'],failed['N'],failed['rb'])==(0,8,.98)
        assert failed['passed_finite_ODE_numerical_checks'] is False and failed['error'] is not None
        assert failed['active_stage']=={'stage':'matrix_exponential','time':.25}
        assert (adm['J'],adm['N'],adm['rb'])==(0,8,.98)
        receipt['preserved_growth_failure']=failed['error'];receipt['growth_launch_HEAD']=failed['launch_HEAD']
        with np.load(OP,allow_pickle=False) as data:op={k:data[k].copy() for k in data.files}
        checks,rho=pc.operator_checks(op,8,.98,np,roots_jacobi,eval_jacobi)
        keys=('physical_seed_modal','seed_common_rho','selected_indices','selected_modal_modes',
              'selected_actual_Jv','eigenvalues','seed_states_modal','propagation_times')
        with np.load(PAYLOAD,allow_pickle=False) as data:saved={k:data[k].copy() for k in keys}
        assert all(np.isfinite(a).all() for a in saved.values())
        assert saved['physical_seed_modal'].shape==(64,8) and saved['selected_modal_modes'].shape==(64,4)
        assert saved['selected_actual_Jv'].shape==(64,4) and saved['selected_indices'].shape==(4,)
        assert saved['selected_indices'].dtype.kind in 'iu' and np.all((0<=saved['selected_indices'])&(saved['selected_indices']<64))
        assert saved['eigenvalues'].shape==(64,) and saved['seed_common_rho'].shape==(8,)
        assert saved['propagation_times'].shape==(1,) and saved['propagation_times'][0]==0
        assert saved['seed_states_modal'].shape==(1,64,8)
        checks['saved_nodes']=pc.error(saved['seed_common_rho'],rho)
        checks['saved_t0_seed_state']=pc.error(saved['seed_states_modal'][0],saved['physical_seed_modal'])
        J=op['Jbulk']+op['Jsat'];m=saved['selected_modal_modes'];Jm=saved['selected_actual_Jv']
        checks['saved_actual_Jv']=pc.complex_error(pc.action_split(J,m,np),Jm)
        assert all(v['scaled_l2']<=2e-9 for v in checks.values())
        receipt['checks']=checks
        points,qmap,_=pc.load_maps(np);modes=pc.modal_jets(points,8,.98,np,eval_jacobi)
        columns={'seed':saved['physical_seed_modal'],'seed_J':pc.action_split(J,saved['physical_seed_modal'],np),
                 'selected_mode':m,'selected_actual_Jv':Jm,
                 'saved_t0_state':saved['seed_states_modal'][0],
                 'saved_t0_Jstate':pc.action_split(J,saved['seed_states_modal'][0],np)}
        arrays={'points':points};stats={};rows=[]
        for name,value in columns.items():
            constraints=pc.contract_split(qmap,modes,value,np)
            independent=pc.scalar_contract_real(qmap,modes,value.real,np)+1j*pc.scalar_contract_real(qmap,modes,value.imag,np)
            checks['scalar_'+name]=pc.complex_error(constraints,independent)
            assert checks['scalar_'+name]['scaled_l2']<=5e-11
            arrays[name+'_constraints_real']=constraints.real;arrays[name+'_constraints_imag']=constraints.imag
            stats[name]=pc.sample_stats(constraints)
            for column in range(value.shape[1]):
                for p,x in enumerate(points):rows.append({'column_group':name,'column':column,'point':p,'x':x.tolist(),
                    'physical8_real':constraints[p,:,column].real.tolist(),'physical8_imag':constraints[p,:,column].imag.tolist()})
        selected_lambda=saved['eigenvalues'][saved['selected_indices']]
        qc=arrays['selected_mode_constraints_real']+1j*arrays['selected_mode_constraints_imag']
        qJ=arrays['selected_actual_Jv_constraints_real']+1j*arrays['selected_actual_Jv_constraints_imag']
        assert np.count_nonzero(arrays['seed_constraints_real'][:,:,[0,4]])==0
        assert np.count_nonzero(arrays['seed_constraints_imag'][:,:,[0,4]])==0
        checks['saved_lambda_constraint_residual_diagnostic_only']=pc.complex_error(qJ,qc*selected_lambda[None,None,:])
        # This residual is recorded, not used to replace actual Jv or identify a subsidiary mode.
        np.savez_compressed(args.output/'sample-physical8.npz',**arrays)
        (args.output/'cases.jsonl').write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in rows))
        pc.write(args.output/'summary.json',{'stats':stats,'selected_indices':saved['selected_indices'].tolist(),
          'selected_saved_eigenvalues':[[float(v.real),float(v.imag)] for v in selected_lambda],
          'checks':checks,'seed_metadata_from_failed_receipt':failed['seeds'],
          'normalization':'Saved physical seed amplitudes and saved energy-normalized modal modes retained without rescaling.',
          'scope':receipt['scope']})
        receipt.update({'checks':checks,'case_count':len(rows),'saved_time':[0.], 'initial_gauge_zero_count':42,
            'payload_classification':'Failed propagation payload; initial seeds, four selected modes/actualJv and saved t0 only.',
            'outputs':[{'path':'sample-physical8.npz','sha256':pc.sha(args.output/'sample-physical8.npz'),'role':'large_payload'},
                       {'path':'cases.jsonl','sha256':pc.sha(args.output/'cases.jsonl'),'role':'source_or_receipt'},
                       {'path':'summary.json','sha256':pc.sha(args.output/'summary.json'),'role':'source_or_receipt'}]})
    except Exception as exc:receipt['error']={'type':type(exc).__name__,'message':str(exc)}
    receipt['source_after']={str(p.resolve()):pc.sha(p) for p in paths}
    receipt['sources_unchanged']=receipt['source_before']==receipt['source_after'];receipt['seconds']=time.monotonic()-begin
    receipt['passed_saved_payload_point_readback']=receipt['error'] is None and receipt['sources_unchanged']
    pc.write(args.output/'receipt.json',receipt);print(json.dumps(receipt,indent=2,allow_nan=False))
    return 0 if receipt['passed_saved_payload_point_readback'] else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--authorization',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if not args.execute:print('HELD source-only saved-data adapter; exact authorization required.');return 0
    if args.authorization is None or args.output is None:parser.error('--authorization and --output required')
    return execute(args)
if __name__=='__main__':sys.exit(main())
